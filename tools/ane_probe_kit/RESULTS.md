# Probe results

Fill this in and return it together with `results/` and the full console output.

## Host

| field | value |
|---|---|
| macOS version | |
| macOS build   | |
| Mac model / chip | |
| ANECompiler version / build | (from `results/env.txt`) |
| Xcode version | |

## Outcome

- [ ] `RESULT: NO weight-bearing OutTrans=1 on this host` — build ruled out
- [ ] `RESULT: *** WEIGHT-BEARING OutTrans=1 FOUND ***` — **breakthrough**

If FOUND, which graph(s) produced it, and did `02_posread_L.py` print `PERFECT BIJECTION`?

```
(paste the tail of the console output here)
```

## Files to return

- [ ] full console output of `./run_probe.sh`
- [ ] `results/env.txt`
- [ ] `results/outtrans_probe.json`
- [ ] `results/posread_L_result.npz`  (only if the positional read ran)

## Notes / anything unexpected

