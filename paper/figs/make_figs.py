#!/usr/bin/env python3
"""Data figures for the AFM reconstruction report. All numbers are measurements reported in the text."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'font.family': 'serif'})
BLUE, ORANGE, GREY, GREEN, RED = '#2f6db3', '#e08a1e', '#9a9a9a', '#2e8b57', '#c0392b'
L = np.arange(12)

# 1. the off-by-one diagnosis: negative FFN coupling vs negative post-FFN gain, and sign agreement after the fix
neg_coup = [0.465, 0.281, 0.301, 0.291, 0.296, 0.295, 0.279, 0.290, 0.292, 0.268, 0.293, 0.312]
neg_gain = [0.290, 0.272, 0.300, 0.287, 0.291, 0.292, 0.282, 0.289, 0.294, 0.268, 0.290, 0.286]
agree_fix = [0.579, 0.984, 0.992, 0.990, 0.982, 0.990, 0.991, 0.995, 0.997, 0.999, 0.996, 0.945]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].plot(L, neg_coup, 'o-', color=BLUE, label='channels with negative up$\\to$down coupling')
ax[0].plot(L, neg_gain, 's--', color=ORANGE, label='entries of post-FFN gain $g_2$ that are negative')
ax[0].set_xlabel('dense layer'); ax[0].set_ylabel('fraction'); ax[0].set_ylim(0.2, 0.5); ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title('Same count, different positions', fontsize=9)
ax[1].bar(L, agree_fix, color=GREEN, label='gains read 64-B aligned')
ax[1].axhline(0.59, color=RED, ls='--', lw=1); ax[1].text(11.4, 0.61, 'one-element shift\n(chance 0.59)', color=RED, ha='right', fontsize=7)
ax[1].set_ylim(0.5, 1.02); ax[1].set_xlabel('dense layer'); ax[1].set_ylabel('sign agreement')
ax[1].set_title('sign(coupling) vs sign($g_2$)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_offbyone.pdf'); plt.close(fig)

# 2. reconstructed K/V magnitudes vs Apple's int8 calibration ranges (127 / scale), dense layers
kmax = [9.250, 9.058, 7.834, 6.948, 7.832, 8.525, 7.648, 6.662, 9.830, 6.565, 9.664, 8.551]
k999 = [6.026, 6.555, 5.110, 4.027, 4.511, 5.177, 4.591, 4.914, 5.186, 4.039, 5.116, 4.535]
krng = [6.668, 6.668, 8.056, 8.056, 6.668, 8.056, 9.967, 6.668, 8.056, 6.668, 8.056, 8.056]
vmax = [4.4299, 0.2415, 0.1417, 0.3845, 0.3022, 0.2045, 0.2289, 0.3155, 0.1841, 0.2807, 0.1666, 0.0957]
v999 = [2.5712, 0.1379, 0.0866, 0.1727, 0.1665, 0.1185, 0.1303, 0.1244, 0.1166, 0.1205, 0.1122, 0.0559]
vrng = [5.4919, 0.3118, 0.3333, 0.4022, 0.3432, 0.3472, 0.3513, 0.3451, 0.3115, 0.3494, 0.2609, 0.1726]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for a, mx, p9, rg, nm in ((ax[0], kmax, k999, krng, 'K'), (ax[1], vmax, v999, vrng, 'V')):
    a.plot(L, rg, '_', ms=14, mew=2.5, color='k', label="Apple int8 range (127/scale)")
    a.plot(L, mx, 'o', color=BLUE, label='reconstruction max')
    a.plot(L, p9, '^', color=ORANGE, label='reconstruction 99.9th pct')
    a.set_xlabel('layer'); a.set_ylabel(f'|{nm}|'); a.set_title(f'{nm} cache magnitudes', fontsize=9)
ax[1].set_yscale('log'); ax[0].legend(frameon=False, fontsize=7, loc='lower left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_kv.pdf'); plt.close(fig)

# 3. calibrated OV copy probe: best head self-rank per layer (lower = sharper copying)
afm = [134435, 104835, 73226, 128566, 112854, 46171, 34612, 81352, 112276, 13862, 15739, 72738, 92130, 49936, 23974, 1546,
       1300, 15958, 21138, 15454, 4999, 53065, 33514, 2981, 14738, 11392, 6492, 9048, 45756, 35468, 71062, 29794, 11372,
       21336, 43608, 55145, 55643, 82795, 83130, 99598, 77838, 57268, 51044, 81586, 84104, 71388, 68315, 81776, 41558,
       89020, 86424, 90785, 55298, 105888, 96178, 101364]
clm = {2: 8559, 5: 65858, 10: 0, 15: 2989, 20: 0, 25: 0, 30: 1}
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.semilogy(range(56), np.maximum(afm, 1), 'o-', ms=3, color=BLUE, label='AFM v11.2 (decoded)')
ax.semilogy(list(clm), [max(v, 1) for v in clm.values()], 's', ms=6, color=GREEN, label='CodeLM v3 (healthy Apple tamm model)')
ax.axhline(131072, color=GREY, ls=':', lw=1); ax.text(55, 150000, 'chance', color=GREY, ha='right', fontsize=7)
ax.set_xlabel('layer'); ax.set_ylabel('best head self-rank\n(of 262k / 153k)'); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_copy.pdf'); plt.close(fig)

# 4. fidelity vs Apple's greedy answers (40 prompts, 1127 answer tokens, teacher-forced top-1)
rows = [("v1.2: + Neural Engine SiLU table (default)", 96.72, BLUE),
        ("v1.1: no embedding multiplier, input bias on generated tokens only", 96.10, ORANGE),
        ('v1.1, per-token routing', 93.79, ORANGE),
        ('v1.1 with the input bias on every token (router sees it)', 92.64, ORANGE),
        ("v1.0: embedding scaled x100 (bug), Apple's schedule", 91.75, GREY),
        ('v1.0, per-token routing', 90.51, GREY),
        ('earlier one-turn template, per-token routing', 87.5, GREY)]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
for yy, r in zip(y, rows): ax.text(r[1] + 0.3, yy, f'{r[1]:.1f}%', va='center', fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.set_xlim(80, 100)
ax.set_xlabel('teacher-forced top-1 agreement with Apple (%)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_fidelity.pdf'); plt.close(fig)

# 4b. expert re-picks during generation: experts changed per re-pick, Apple's runtime log vs our router (47 re-picks)
apple = [29, 29, 18, 15, 15, 25, 21, 9, 20, 9, 17, 36, 39, 25, 25, 17, 44, 29, 27, 10, 17, 19, 42, 22, 23, 25, 20, 44, 20, 27,
         21, 20, 18, 18, 13, 21, 16, 16, 48, 67, 19, 24, 43, 43, 48, 49, 73]
ours  = [25, 24, 21, 12, 14, 18, 17, 13, 17, 9, 18, 38, 42, 20, 26, 16, 44, 32, 25, 31, 34, 21, 36, 28, 32, 22, 18, 34, 20, 28,
         23, 21, 29, 24, 22, 23, 24, 21, 38, 47, 28, 29, 31, 34, 30, 38, 60]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot([0, 75], [0, 75], ':', color=GREY, lw=1)
ax.scatter(apple, ours, s=12, color=BLUE)
ax.set_xlabel("experts replaced (Apple's runtime log)"); ax.set_ylabel('experts replaced (our router)')
ax.set_xlim(0, 75); ax.set_ylim(0, 75); ax.text(4, 66, f'r = {np.corrcoef(apple, ours)[0, 1]:.2f}', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_repick.pdf'); plt.close(fig)

# 5. where the remaining disagreements sit
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ks = ['2nd', 'top-3', 'top-5']; vs = [76, 90, 98]
ax.bar(ks, vs, color=BLUE)
for i, v in enumerate(vs): ax.text(i, v + 1, f'{v}%', ha='center', fontsize=8)
ax.set_ylim(0, 108); ax.set_ylabel("Apple's token within our ...\n(% of mismatches)")
fig.tight_layout(); fig.savefig(f'{OUT}/fig_ranks.pdf'); plt.close(fig)
print('figures written to', OUT)
