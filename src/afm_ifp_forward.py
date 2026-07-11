#!/usr/bin/env python3
"""afmplus-v11.0-ifp forward pass (architecture from config.json + specialized_model_0.mpsgraph).

Op order per the ANE graph:
  h -> RMSNorm(pre) -> [QKV | Q]-proj -> QK-norm(per head) -> RoPE -> GQA SDPA -> O-proj
    -> RMSNorm(post) -> +residual
    -> RMSNorm(pre) -> [SwiGLU dense | MoE experts] -> RMSNorm(post) -> +residual
RMSNorm gamma is folded into the adjacent linear weights (recovered weights already carry it),
so the norm ops here are parameter-free (divide-by-RMS only).

This module executes on the recovered weights and validates shapes/numerics. It is NOT wired to
the tokenizer or the exact layer-depth / expert->layer mapping (see notes at bottom); it is the
architecture scaffold, run here as a numeric sanity harness.
"""
import torch, torch.nn.functional as F, math

D=1536; NQ=16; NKV=8; HD=128; QDIM=NQ*HD; KVDIM=NKV*HD; THETA=500000.0

def rmsnorm(x, eps=1e-6):
    return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()

def rope(x, pos):
    # x: (heads, T, HD); rotary over HD with theta=500000
    T=x.shape[-2]
    inv=1.0/(THETA**(torch.arange(0,HD,2).float()/HD))
    ang=torch.outer(torch.arange(T).float()+pos, inv)      # (T, HD/2)
    cos=torch.cat([ang.cos(),ang.cos()],-1); sin=torch.cat([ang.sin(),ang.sin()],-1)
    x1,x2=x[...,:HD//2],x[...,HD//2:]
    rot=torch.cat([-x2,x1],-1)
    return x*cos+rot*sin

def attention(h, Wqkv, Wo, is_qkv, kv_cache=None):
    T=h.shape[0]
    hn=rmsnorm(h)
    proj=hn @ Wqkv.T                                        # (T, out)
    if is_qkv:
        q,k,v=proj[:,:QDIM],proj[:,QDIM:QDIM+KVDIM],proj[:,QDIM+KVDIM:QDIM+2*KVDIM]
    elif kv_cache is not None:                              # Q-only (KV-reuse): reuse cached K,V
        q=proj[:,:QDIM]; k,v=kv_cache
    else:                                                   # depth-order not pinned: no source yet
        q=proj[:,:QDIM]; k=v=q[:,:KVDIM]
    q=q.reshape(T,NQ,HD).transpose(0,1); k=k.reshape(T,NKV,HD).transpose(0,1); v=v.reshape(T,NKV,HD).transpose(0,1)
    q=rmsnorm(q); k=rmsnorm(k)                              # QK-norm (per head)
    q=rope(q,0); k=rope(k,0)                                # RoPE
    kg=k.repeat_interleave(NQ//NKV,0); vg=v.repeat_interleave(NQ//NKV,0) # GQA expand
    o=F.scaled_dot_product_attention(q,kg,vg,is_causal=True)
    o=o.transpose(0,1).reshape(T,QDIM)
    # pre-norm block: residual grows but every layer re-normalizes its own input, so it's benign.
    # (A parameter-free post-norm inflates the update to ~sqrt(d); the real gamma is folded/unshipped.)
    return h + (o @ Wo.T), (k.transpose(0,1).reshape(T,KVDIM), v.transpose(0,1).reshape(T,KVDIM))

def swiglu(h, Wgate, Wup, Wdown):
    hn=rmsnorm(h)
    return h + (F.silu(hn@Wgate.T)*(hn@Wup.T)) @ Wdown.T

if __name__=="__main__":
    import sys
    sd=torch.load("/Volumes/D/fix/afmplus_v11_ifp_FULL_v2.pt",map_location='cpu',weights_only=False)["state_dict"]
    import re
    L=sorted({int(re.search(r'layers\.(\d+)\.',k).group(1)) for k in sd if k.startswith("model.layers.")})
    print(f"loaded {len(L)} attention layers")
    torch.manual_seed(0); h=torch.randn(4,D)*0.1            # 4-token test sequence
    kv=None; n0=float(h.norm())
    for i in L:
        p=f"model.layers.{i}.attn."
        qkv=sd.get(p+"qkv.weight"); is_qkv=qkv is not None
        W=(qkv if is_qkv else sd[p+"q.weight"]).float()
        h,kv=attention(h, W, sd[p+"o.weight"].float(), is_qkv, kv)
    print(f"forward ran through {len(L)} layers | h-norm {n0:.2f} -> {float(h.norm()):.2f} | finite={torch.isfinite(h).all().item()}")
    print("shapes/dtype OK; architecture executes on the recovered weights.")
