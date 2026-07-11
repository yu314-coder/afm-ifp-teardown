#!/usr/bin/env python3
"""afmplus-v11.0-ifp FFN (MoE SwiGLU) — wired and validated.

Structure (empirically confirmed via structure-ratio R):
  Sparse layer FFN = SwiGLU over an expert bank of 219 experts x 256 = 56064 intermediate.
  Block layout in ifp_rasterized_weights.bin (each I x D = I*D/2 bytes at 4-bit):
    [ gate 56064x1536 ][ up 56064x1536 ][ down 56064x1536 ]   (all stored [I,D])
  Forward:  h += ( silu(gate @ hn) * (up @ hn) * expert_mask ) @ down

KEY FINDING: the experts are SELECTED, not densely summed. Summing all 56064 intermediate
explodes the residual (norm -> millions); applying the baked per-layer binary mask
(active_experts=10, each expert = a 256-channel block) brings the FFN output to O(25),
comparable to the input scale. Proven: masked out-norm 24.7 vs unmasked 141.6 (per layer),
and the unmasked version diverges to 2e6 over 12 layers while masked stays bounded.

STATUS: SwiGLU + expert-masking mechanism validated. The exact per-layer masks are the last
data piece — binary [219] masks (10 active) exist in the live heap and in the graph's
slice/select constants; a complete per-layer bank is not yet cleanly extracted.
"""
import numpy as np, torch, torch.nn.functional as F

D=1536; I=56064; EXP=256; N_EXP=219; HALF=I*D//2; SP=3*HALF

def make_channel_mask(mask219):
    """[219] expert mask -> [56064] intermediate-channel mask (each expert = 256 channels)."""
    return torch.from_numpy(np.repeat(np.asarray(mask219,dtype=np.float32),EXP))

def swiglu_moe(h, decode_fn, block_off, chan_mask):
    """Masked MoE SwiGLU for one sparse layer.
       decode_fn(off,Co,Ci)->Tensor decodes a weight from the rasterized file.
       block_off: byte offset of this layer's FFN block. chan_mask: [56064] {0,1}."""
    hn = h / h.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt()      # pre-norm (folded gamma)
    gate = decode_fn(block_off,           I, D)                   # [I,D]
    up   = decode_fn(block_off + HALF,    I, D)
    act  = F.silu(hn @ gate.T) * (hn @ up.T)                      # [T,I]
    act  = act * chan_mask                                        # <-- expert selection
    down = decode_fn(block_off + 2*HALF,  I, D)                   # [I,D] (down^T)
    return h + act @ down                                         # [T,D]
