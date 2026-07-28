"""Do the O / down OUTPUT channels land in the same basis as the (known-correct) inputs?

O and down WRITE the residual stream; Q/K/V and gate/up READ it. All five readers are
decoded in the round-trip-validated OutTrans=0 order, so their INPUT axis is a trustworthy
sample of the residual basis. If O/down's OUTPUT axis is also correctly ordered, the two
should describe the same per-position structure and their norm profiles should correlate
at the same strength that two known-correct readers do.

Calibration (both known-correct, both input side, measured earlier):
    corr(rownorm(gate_L), rownorm(up_L)) ~ +0.65
    cross-layer same-role control        ~ +0.63

Test:
    corr(colnorm(O_L),    rownorm(gate_L))     # O writes residual, gate reads it (same layer)
    corr(colnorm(down_L), rownorm(Q_{L+1}))    # down writes residual, next layer's Q reads it

If O/down outputs are correctly ordered these should be comparable to the calibration.
If they are permuted, they collapse toward 0 while the calibration stays high -- which
would localise the defect to the writer side and contradict the LoRA-based inference.
A shuffled control is included so "0" has a meaning.
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

NL = 8
rng = np.random.RandomState(0)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


W = {}
print('decoding %d layers ...' % (NL + 1), flush=True)
for L in range(NL + 1):
    ent = by_layer[L]
    for role in ['Q', 'O', 'gate', 'up', 'down']:
        if role == 'down':
            W[(L, role)] = np.asarray(pw.decode_down(ent[role]), dtype=np.float32)   # [3200,1024]
        else:
            W[(L, role)] = np.asarray(pw.decode_tensor(ent[role]), dtype=np.float32)  # [in,out]
    print('  layer %d' % L, flush=True)

print()
print('%-6s %-26s %-26s %-26s %s' % ('layer', 'calib gate~up (in,in)', 'O_out ~ gate_in', 'down_out ~ Q_in(L+1)', 'shuffled ctrl'))
cal, t1, t2, ctl = [], [], [], []
for L in range(NL):
    g_in = np.linalg.norm(W[(L, 'gate')], axis=1)    # per input pos
    u_in = np.linalg.norm(W[(L, 'up')], axis=1)
    o_out = np.linalg.norm(W[(L, 'O')], axis=0)      # per OUTPUT pos (residual)
    d_out = np.linalg.norm(W[(L, 'down')], axis=0)   # per OUTPUT pos (residual)
    q_next = np.linalg.norm(W[(L + 1, 'Q')], axis=1)

    c0 = corr(g_in, u_in)
    c1 = corr(o_out, g_in)
    c2 = corr(d_out, q_next)
    c3 = corr(o_out[rng.permutation(o_out.size)], g_in)
    cal.append(c0); t1.append(c1); t2.append(c2); ctl.append(c3)
    print('%-6d %+.4f%18s %+.4f%18s %+.4f%18s %+.4f' % (L, c0, '', c1, '', c2, '', c3))

print()
print('MEAN  calib gate~up      %+.4f   (both known-correct, input side)' % np.mean(cal))
print('MEAN  O_out ~ gate_in    %+.4f' % np.mean(t1))
print('MEAN  down_out ~ Q_in    %+.4f' % np.mean(t2))
print('MEAN  shuffled control   %+.4f' % np.mean(ctl))
print()
if np.mean(t1) > 0.5 * np.mean(cal) and np.mean(t2) > 0.5 * np.mean(cal):
    print('=> writer outputs share the reader basis: O/down ordering looks CORRECT.')
elif abs(np.mean(t1)) < 0.1 and abs(np.mean(t2)) < 0.1:
    print('=> writer outputs are UNCORRELATED with the residual basis while readers agree')
    print('   strongly: the defect is localised to the O/down OUTPUT ordering.')
else:
    print('=> partial/ambiguous; read the per-layer numbers above.')
