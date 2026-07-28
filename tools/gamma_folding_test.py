"""Is RMSNorm gamma actually FOLDED into the reading convs in pico?

make_gguf.py writes every RMSNorm weight as ones, justified by the comment
"gamma folded at ANE compile -> written as ones". That assumption has never been
tested. If it is wrong, the GGUF is missing 49 learned gamma vectors and the
O/down ordering is NOT the only blocker.

Test logic (purely static, no forward pass needed):
  RMSNorm folding means  y = (x/rms(x)) @ (diag(gamma) . W)  -- i.e. gamma multiplies
  the INPUT rows of every conv that reads that norm's output.
    * gate and up BOTH read the FFN norm  -> both carry the SAME diag(gamma_ffn)
    * Q, K, V ALL read the attention norm -> all carry the SAME diag(gamma_attn)
  So if folding happened, the per-input-row norm profiles of gate and up must share a
  common multiplicative factor and correlate strongly WITHIN a layer.

Confound: a trained residual stream may have intrinsically uneven per-dimension scale,
which would also correlate readers. Controlled for by comparing
    within-layer cross-role   corr(gate_L, up_L)          [same gamma]
against
    cross-layer same-role      corr(gate_L, gate_L')      [different gamma, same intrinsic structure]
Folding predicts within-layer >> cross-layer. Pure intrinsic structure predicts them similar.
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

NL = 8  # enough layers for the statistics; decoding all 24 is slow and unnecessary


def rownorms(W):
    """per-INPUT-position norm profile (W is [in, out])."""
    return np.linalg.norm(W, axis=1)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print('decoding %d layers ...' % NL, flush=True)
prof = {}
for L in range(NL):
    ent = by_layer[L]
    for role in ['Q', 'K', 'V', 'gate', 'up']:
        W = pw.decode_tensor(ent[role])
        prof[(L, role)] = rownorms(np.asarray(W, dtype=np.float32))
    print('  layer %d done' % L, flush=True)

print()
print('=== WITHIN-LAYER cross-role (same gamma if folded) ===')
w_gu, w_qk, w_qv, w_kv = [], [], [], []
for L in range(NL):
    c_gu = corr(prof[(L, 'gate')], prof[(L, 'up')])
    c_qk = corr(prof[(L, 'Q')], prof[(L, 'K')])
    c_qv = corr(prof[(L, 'Q')], prof[(L, 'V')])
    c_kv = corr(prof[(L, 'K')], prof[(L, 'V')])
    w_gu.append(c_gu); w_qk.append(c_qk); w_qv.append(c_qv); w_kv.append(c_kv)
    print('  L%-2d  gate~up %+.4f   Q~K %+.4f   Q~V %+.4f   K~V %+.4f' % (L, c_gu, c_qk, c_qv, c_kv))
print('  MEAN gate~up %+.4f | Q~K %+.4f | Q~V %+.4f | K~V %+.4f'
      % (np.mean(w_gu), np.mean(w_qk), np.mean(w_qv), np.mean(w_kv)))

print()
print('=== CROSS-LAYER same-role (different gamma; intrinsic-structure control) ===')
x_gate, x_up, x_q = [], [], []
for L in range(NL):
    for L2 in range(L + 1, NL):
        x_gate.append(corr(prof[(L, 'gate')], prof[(L2, 'gate')]))
        x_up.append(corr(prof[(L, 'up')], prof[(L2, 'up')]))
        x_q.append(corr(prof[(L, 'Q')], prof[(L2, 'Q')]))
print('  MEAN gate~gate %+.4f | up~up %+.4f | Q~Q %+.4f  (n=%d pairs)'
      % (np.mean(x_gate), np.mean(x_up), np.mean(x_q), len(x_gate)))

print()
print('=== VERDICT ===')
within = np.mean(w_gu)
across = np.mean(x_gate + x_up)
print('  within-layer gate~up : %+.4f' % within)
print('  cross-layer  same-role: %+.4f' % across)
if within > 0.5 and within > across + 0.25:
    print('  => gamma IS folded into the conv rows. Writing ones in the GGUF is CORRECT.')
elif abs(within - across) < 0.15:
    print('  => no per-layer shared factor beyond intrinsic structure.')
    print('     gamma is likely NOT folded -> the GGUF is MISSING learned norms.')
else:
    print('  => ambiguous; inspect the numbers above.')
