"""Probe the FFN-neuron axis with the LoRA `down` A-matrix (ANE bytes known verbatim).

Setting: base `down` fails on BOTH axes (#23). Its 3200 axis is compared against gate/up's
OUTPUT axis, but that reference is itself only +0.222 (gate~up on the FFN axis) versus +0.700
on the residual axis -- so before blaming `down` it is worth asking whether the FFN-axis
reference is trustworthy at all.

The LoRA down adapter gives an independent read. `lora_32_constant_data.bin` is DMA'd
verbatim into __MKERN_0 (#19), so its bytes ARE the ANE stream. Per layer, down_A occupies
elements 483328..585728 (102400 fp16 = 3200 x 32), a boundary confirmed by log-RMS jumps.
It is unpalettized fp16 and OutTrans=0, and its 3200 axis indexes the SAME FFN neurons as
gate/up's 3200 output axis.

CoeffSize for the 3200->32 task is 0x3200 = 12800 B per bank x 16 banks = 204800 B = 102400
fp16, i.e. each bank holds 32/16 = 2 outputs x 3200 inputs = 6400 fp16. Two intra-bank
readings are possible; both are tested.

If down_A's 3200 axis agrees with gate/up's, the FFN reference is sound and base `down` is
genuinely misordered. If it does NOT, the FFN-axis reference is itself suspect and #23's
verdict on down's input axis has to be re-opened.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import pico_weights as pw

P = ('/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/'
     '031c7be6f8fddbff0a6650fee75e345b1ee9613c.asset/.AssetData/model.odixpackage/MPSGraph/'
     'mpsExecutable.mpsgraphpackage/lora_32_constant_data.bin')
v = np.frombuffer(np.fromfile(P, dtype=np.uint8).tobytes(), dtype=np.float16).astype(np.float32)
PL = 618496
DOWN_A0, DOWN_A1 = 483328, 585728          # 102400 fp16 = 3200 x 32
NBANK, OPB, NIN = 16, 2, 3200              # 16 banks, 2 outputs/bank, 3200 inputs

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e


def c(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decode_downA(blk, mode):
    """blk: 102400 fp16 for [3200 in, 32 out]. Returns [3200, 32]."""
    W = np.zeros((NIN, NBANK * OPB), np.float32)
    per = blk.size // NBANK
    for b in range(NBANK):
        w = blk[b * per:(b + 1) * per]
        slot = np.arange(per)
        if mode == 'ofast':
            o = slot % OPB; i = slot // OPB
        else:
            o = slot // NIN; i = slot % NIN
        W[i, b * OPB + o] = w
    return W


NL = 6
print('%-5s %-13s %-13s %-13s %-13s' % ('L', 'A_ofast~gate', 'A_ifast~gate', 'A_ofast~up', 'ref gate~up'))
acc = {k: [] for k in ('of_g', 'if_g', 'of_u', 'ref')}
for L in range(NL):
    ent = by_layer[L]
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)   # [1024,3200]
    U = np.asarray(pw.decode_tensor(ent['up']), dtype=np.float32)
    g_ffn = np.linalg.norm(G, axis=0)
    u_ffn = np.linalg.norm(U, axis=0)
    blk = v[L * PL + DOWN_A0: L * PL + DOWN_A1]
    a_of = np.linalg.norm(decode_downA(blk, 'ofast'), axis=1)
    a_if = np.linalg.norm(decode_downA(blk, 'ifast'), axis=1)
    r = [c(a_of, g_ffn), c(a_if, g_ffn), c(a_of, u_ffn), c(g_ffn, u_ffn)]
    for k, val in zip(('of_g', 'if_g', 'of_u', 'ref'), r):
        acc[k].append(val)
    print('%-5d %+.4f       %+.4f       %+.4f       %+.4f' % (L, *r))

print()
print('MEAN  downA(ofast)~gate %+.4f | downA(ifast)~gate %+.4f | downA(ofast)~up %+.4f | ref gate~up %+.4f'
      % (np.mean(acc['of_g']), np.mean(acc['if_g']), np.mean(acc['of_u']), np.mean(acc['ref'])))
print()
best = max(abs(np.mean(acc['of_g'])), abs(np.mean(acc['if_g'])))
if best > 0.5 * abs(np.mean(acc['ref'])):
    print('=> LoRA down_A agrees with the gate/up FFN axis: that reference is SOUND,')
    print('   so base down\'s 3200 axis really is misordered.')
else:
    print('=> LoRA down_A does NOT agree with the gate/up FFN axis either.')
    print('   The FFN-axis reference is itself unreliable -- #23\'s verdict on down\'s')
    print('   INPUT axis cannot be sustained on that evidence, and the +0.222 gate~up')
    print('   figure is better read as "weak shared structure" than as a ground truth.')
