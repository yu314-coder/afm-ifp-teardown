# ANE `OutTrans=1` probe kit

**One question, answered in ~5 minutes on any Apple Silicon Mac:**

> Does *this* machine's ANE compiler emit a **weight-bearing** `OutTrans=1` convolution?

If any machine answers **YES**, a long-standing reverse-engineering blocker is solved. If it answers
NO, that macOS build is ruled out and costs you five minutes.

Nothing here contains or requires Apple model weights. The probes compile **synthetic random
weights** only, so this kit is self-contained and safe to copy anywhere.

---

## Why this matters (short version)

Reconstructing Apple's on-device 300M draft model (`afmplus-v11.0-pico`) from its shipped asset is
complete except for **one weight tensor**, the FFN down-projection:

* 6 of 7 weight roles per layer decode correctly and are independently verified.
* The down-projection's *values* are provably correct (its singular-value spectrum is clearly that of
  a trained matrix, nothing like random). Only the **channel ordering** is wrong.
* Every tile that decodes correctly is `OutTrans=0`. Both tiles that needed intervention are
  `OutTrans=1`, and one of those (`O`) turned out to need a **transpose**, not a permutation.
* No synthetic graph has ever produced `OutTrans=1` on a conv that **carries coefficients** — the
  flag always lands on a weightless shuffle task instead. So the effect of that flag on the
  coefficient payload has never been observed, and ~15 controlled ordering hypotheses have all failed.

The ANE compile step is performed by **`/System/Library/PrivateFrameworks/ANECompiler.framework`,
which ships with macOS — not with Xcode.** Verified directly: two different Xcode versions (26.5 and
27.0 beta) on the same host produce byte-identical results, including identical failures. Simulator
runtimes cannot help either — the Simulator has no Neural Engine, ships no ANE frameworks at all, and
Apple's own weight files inside it are literally suffixed `_nonane`.

**So the only remaining variable is the macOS build.** Hence this kit.

---

## Will macOS 26.5 work?

**Unknown — that is exactly the experiment.** Being honest rather than encouraging:

* 26.5 ships a *different* `ANECompiler.framework` than 27.0, so it is a genuine, independent shot.
* But there is no positive evidence it behaves differently on this specific scheduler decision. Treat
  it as roughly a coin flip, not a known fix.
* It is cheap: ~5 minutes, and the result is unambiguous either way.

**Any macOS build is worth trying**, not just 26.5 — older, newer, or beta. Run it on every Mac you
have access to; each one either solves it or eliminates a build.

**Requirements:** Apple Silicon (M-series). Intel Macs have no Neural Engine and cannot run this.

---

## Setup

```bash
unzip ane_probe_kit.zip && cd ane_probe_kit
python3 -m venv venv && ./venv/bin/pip install -q coremltools numpy
```

`coremltools` must import cleanly. If you hit an ABI error, the Python version and the installed
`coremltools` wheel disagree — create the venv with a Python that has a matching wheel (3.11 works
well).

Xcode (any version) must be installed, because `xcrun coremlcompiler` is used for the front-end step.

---

## Run

```bash
./run_probe.sh
```

That prints the environment, runs the decisive test, and — **only if it passes** — automatically runs
the follow-up that extracts the layout.

To run the pieces manually:

```bash
./scripts/00_check_env.sh                 # OS build, Xcode, ANECompiler version
./venv/bin/python scripts/01_outtrans_probe.py     # THE test
./venv/bin/python scripts/02_posread_L.py          # only meaningful if 01 says YES
```

---

## How to read the result

`01_outtrans_probe.py` ends with one of two verdicts.

### `RESULT: NO weight-bearing OutTrans=1 on this host`

This macOS build behaves like the reference host. Nothing further to do — **please still send back
the output**, because it eliminates a build and records its ANECompiler version.

### `RESULT: *** WEIGHT-BEARING OutTrans=1 FOUND ***`

This is the breakthrough. The script prints which graph produced it and the task's
`CoeffSize` / bank count. `run_probe.sh` then runs `02_posread_L.py`, which performs a **positional
read**: it compiles convolutions whose 4-bit weight values encode their own position digits, reads
the emitted coefficient stream back, and inverts it to recover the exact
`slot -> (output, input)` map.

A successful run prints `PERFECT BIJECTION` over all 819,200 positions and the recovered formula.
**Send back the whole console output plus `posread_L_result.npz`.** That map is the missing piece.

---

## What to send back

Whatever the outcome:

1. Full console output of `./run_probe.sh`
2. `results/env.txt`
3. `results/posread_L_result.npz` — only if the positional read ran

Fill in `RESULTS.md` and return it with the above.

---

## Contents

```
run_probe.sh                  one-command runner
scripts/00_check_env.sh       environment + ANECompiler version report
scripts/01_outtrans_probe.py  the decisive test (synthetic weights only)
scripts/02_posread_L.py       positional read, runs only if 01 passes
bin/mil_to_hwx                CoreML -> ANE hwx compiler driver   (BSD-3, Koan-Sin Tan)
bin/hwx_parsing               hwx inspector                        (BSD-3, Koan-Sin Tan)
bin/LICENSE-coreml_to_ane_hwx upstream license
RESULTS.md                    template to fill in and return
```

`bin/` binaries are arm64 and link only system frameworks, so they run on any Apple Silicon Mac.
Upstream project: <https://github.com/freedomtan/coreml_to_ane_hwx>. If a binary is blocked by
Gatekeeper, clear the quarantine attribute:

```bash
xattr -dr com.apple.quarantine bin/
```
