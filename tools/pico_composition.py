"""Apply the CONTROL-VALIDATED composition oracle to pico.

composition_oracle.py established on Qwen3-4B that ||gate @ O||_F separates the correct
residual pairing from a random one at z = +300. So the statistic genuinely detects a
scrambled residual axis. Now use it on pico, where nothing else has been trustworthy.

pico's decoder returns W[in, out], so with O_dec = [attn_in, res_out] and
gate_dec = [res_in, ff_out] the composed map contracts over the residual index as

        score = || O_dec @ gate_dec ||_F

and a permutation of the residual pairing is applied as O_dec[:, pi].

Two orientations are compared, which tests the section-22 transpose functionally rather than
statistically:
   A  O_dec = [attn, res]  (current decoder assumption)  -> O_dec @ gate_dec
   B  O_dec = [res, attn]  (section 22: O stored transposed) -> O_dec.T @ gate_dec

Each is scored against random permutations of the contracted index, exactly as in the control.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/fix/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/fix/pico_shapes')
import pico_weights as pw

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

rng = np.random.RandomState(0)
NL = 6
NR = 12


def z_of(A, B, nrand=NR):
    """A: [x, res], B: [res, y]. score = ||A @ B||_F, permuting A's residual axis."""
    s0 = float(np.linalg.norm(A @ B))
    rs = [float(np.linalg.norm(A[:, rng.permutation(A.shape[1])] @ B)) for _ in range(nrand)]
    m, sd = float(np.mean(rs)), float(np.std(rs))
    return s0, m, sd, (s0 - m) / (sd + 1e-9)


print('%-5s | %-28s | %-28s' % ('L', 'A: O_dec=[attn,res] (current)', 'B: O_dec=[res,attn] (transposed)'))
print('%-5s | %-9s %-9s %-7s | %-9s %-9s %-7s' % ('', 'ident', 'rand', 'z', 'ident', 'rand', 'z'))
zA, zB = [], []
for L in range(NL):
    ent = by_layer[L]
    O = np.asarray(pw.decode_tensor(ent['O']), dtype=np.float32)      # [1024,1024]
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)   # [1024 res, 3200 ff]
    a0, am, asd, az = z_of(O, G)                                      # A: contract O axis1
    b0, bm, bsd, bz = z_of(O.T, G)                                    # B: contract O axis0
    zA.append(az); zB.append(bz)
    print('%-5d | %-9.1f %-9.1f %+-7.1f | %-9.1f %-9.1f %+-7.1f' % (L, a0, am, az, b0, bm, bz))

print()
print('MEAN z  A (current orientation)  %+.2f' % np.mean(zA))
print('MEAN z  B (transposed)           %+.2f' % np.mean(zB))
print()
print('Qwen3-4B reference for a CORRECT model: z = +300')
print()
best = 'B (transposed)' if np.mean(zB) > np.mean(zA) else 'A (current)'
if max(np.mean(zA), np.mean(zB)) > 20:
    print('=> %s shows real composition alignment: pico\'s residual pairing is largely CORRECT' % best)
    print('   in that orientation, and the remaining defect is elsewhere.')
else:
    print('=> BOTH orientations sit at the random level. pico\'s O/gate residual pairing is')
    print('   genuinely scrambled -- now established with a control-validated statistic')
    print('   rather than an assumed one. A permutation search on this objective is justified.')
