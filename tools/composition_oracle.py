"""Build a POSITIVE-CONTROLLED oracle: does ||gate @ O||_F detect a scrambled residual axis?

Every previous oracle was assumed to work. This one is validated first, on a model known to
be correct, by deliberately scrambling it and checking the statistic notices.

Rationale from the actual computation rather than from weight statistics:
    residual_j = sum_i attn_i * O[j,i]        (O writes residual position j)
    ffn_k      = sum_j residual_j * gate[k,j] (gate reads residual position j)
    => composed map  gate @ O,  contracting over the residual index j
If O's residual index is permuted by pi, the composition is wrong. Trained networks align
successive transformations, so the CORRECT pairing should give a larger ||gate @ O||_F than a
random re-pairing. That is a claim about real models, and it is tested, not assumed.

Note this is exactly the QAP objective in disguise:
    ||gate @ O_pi||_F^2 = sum_{j,k} G_O[pi(j),pi(k)] * G_gate[j,k]
with RAW Gram matrices. Earlier attempts used COSINE (row-normalised) matrices, which discard
the magnitude information this objective depends on -- a likely reason they found nothing.

Control on Qwen3-4B: compare identity against random permutations, and report a z-score.
Only if the control passes is the same statistic applied to pico.
"""
import struct
import numpy as np

QWEN = '/Volumes/D/github/image/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
exec(open('/Volumes/D/fix/wf_scratch/validate_oracle.py').read().split("f, tens, ds = read_gguf_index")[0])

f, tens, ds = read_gguf_index(QWEN)
byname = {t[0]: t for t in tens}
rng = np.random.RandomState(0)


def fro_score(gate, O, perm=None):
    """||gate @ O_perm||_F with O's residual axis (rows) permuted."""
    Op = O if perm is None else O[perm]
    return float(np.linalg.norm(gate @ Op))


print('=== POSITIVE CONTROL on Qwen3-4B (known correct) ===')
print('%-5s %-13s %-13s %-13s %s' % ('L', 'identity', 'rand mean', 'rand std', 'z-score'))
zs = []
for L in range(0, 8):
    O = load(f, ds, byname['blk.%d.attn_output.weight' % L])      # [out=res, in=attn]
    G = load(f, ds, byname['blk.%d.ffn_gate.weight' % L])         # [out=ff, in=res]
    if O is None or G is None:
        continue
    O = O.astype(np.float32); G = G.astype(np.float32)
    s0 = fro_score(G, O)
    rs = [fro_score(G, O, rng.permutation(O.shape[0])) for _ in range(12)]
    m, sd = float(np.mean(rs)), float(np.std(rs))
    z = (s0 - m) / (sd + 1e-9)
    zs.append(z)
    print('%-5d %-13.1f %-13.1f %-13.3f %+.2f' % (L, s0, m, sd, z))

mz = float(np.mean(zs))
print()
print('MEAN z-score of the correct ordering vs random: %+.2f' % mz)
print()
if mz > 3.0:
    print('=> CONTROL PASSES: ||gate @ O||_F detects the correct residual pairing.')
    print('   The statistic is valid and can be used to search pico.')
else:
    print('=> CONTROL FAILS: the correct ordering is NOT distinguishable from random by this')
    print('   statistic, even in a known-good model. It cannot be used to search pico, and')
    print('   any search scored with it would be meaningless. Do not proceed with it.')
