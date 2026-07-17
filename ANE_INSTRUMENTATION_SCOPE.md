# Live-Instrumentation Route — Scope of Work

**Goal.** Recover the two pieces proven absent from all static and dynamic *memory-dump* analysis:
(1) the **token embedding** `[262144, 1536]`, and (2) the **deployed expert weight bytes** (the
resident set metadata.bin addresses but does not, in dump form, decode). Both are recoverable *in
principle* by instrumenting the running model — but this is a distinct, larger project than the
static teardown, with hard privilege prerequisites. This document scopes it honestly.

Everything here builds on the established findings (see `paper/afm_teardown.tex`,
`afm-embedding-not-validated`): the model runs in `TGOnDeviceInferenceProviderService` (as
`_modelmanagerd`, ~1.7 GB resident during inference); the embedding is a **CPU-side** `odix`
`load_embeddings` / `AFM_fused_interleaved_embedding_gather_dequant_reshape` op that dequantizes
per-token and feeds `in_embeddings` to the ANE; the FFN/attention weights are ANE-resident, patched
into `__MKERN` from the raster; intermediate activations are ANE-internal (host-memory search:
0.17 correlation).

---

## 0. The key reframe: "ANE instrumentation" is mostly CPU instrumentation

The token embedding is **not** computed on the ANE — the mpsgraph's only `Gather` is the rotary
positional one; the token lookup is a CPU op in the odix runtime. So the highest-value target
(the embedding) is reachable by **CPU** dynamic instrumentation (breakpoints/hooks in the odix
interpreter), which is far more tractable than ANE-internal capture. True ANE instrumentation is
only needed for goal (2), the resident FFN weights. **Do the CPU embedding route first.**

---

## 1. Prerequisites (go/no-go checked 2026-07-15 — lower than first feared)

Inspection of `TGOnDeviceInferenceProviderService.appex` on this machine (macOS 27.0, build
26A5378j; SIP already **disabled**):
- code-signing **flags=0x0 (none)** — *not* hardened runtime; **no `get-task-allow`** entitlement,
  **TeamIdentifier not set**. It **is** a platform binary (`Platform identifier=26`).
- The earlier attach failure was a pure **user mismatch** ("as user euler … running as
  `_modelmanagerd`"), i.e. a permission error, **not** an AMFI/code-signing error.

**Revised prerequisite (in likely-sufficient order):**
1. **Developer mode enabled** — currently *disabled*. `sudo DevToolsSecurity -enable` (one-time).
   macOS blocks `task_for_pid` on other processes without it.
2. **Root** — the target runs as `_modelmanagerd`; attach as `euler` fails on the user check.
   `sudo lldb` covers this. (Passwordless sudo is not configured; the operator runs privileged
   steps.)
3. **AMFI boot-arg — probably NOT required.** Because the binary is neither hardened-runtime nor
   `get-task-allow`-restricted, root + developer mode is expected to suffice. The platform-binary
   status *might* still gate `task_for_pid`; the definitive test below decides. Only if it fails
   with a Mach `(os/kern) failure`/codesign error do you fall back to booting once from Recovery
   with `nvram boot-args="amfi_get_out_of_my_way=1"` (revert after).
4. **The calibrated oracle.** Recovered bytes are validated with the semantic probe
   (`/tmp/control.py`, `/tmp/gguf.py`): genuine embedding **+0.53**; id-matched + shift controls
   reject artifacts. Never accept a recovery without it.

**Definitive go/no-go (operator runs, needs sudo, during inference):**
```
sudo DevToolsSecurity -enable                      # once
echo hi | /Volumes/D/fix/afm >/dev/null &          # warm the model
PID=$(pgrep -f TGOnDeviceInferenceProviderService | head -1)
sudo lldb -o "process attach --pid $PID" -o detach -o quit
#  GO : "Process <pid> stopped"       -> root+devmode suffices; proceed to Approach A
#  NO : "(os/kern) failure"/codesign  -> AMFI boot-arg needed (step 3), then retry
```
Confirmed relevant entitlements (context for later phases): `com.apple.private.ane.
mappingMutableWeightsBuffer.allow` (the `__MKERN` weight patching), `H11ANEInDirectPathClient`
+ `IOSurfaceRootUserClient` (ANE/IOSurface access), asset access to `GenerativeModels`/`Overrides`.

---

## 2. Approach A — CPU dequant interception (the embedding). *Recommended first.*

The gather+dequant runs on CPU and writes `in_embeddings` (an `[seq, 1536]` fp16 tensor) before the
ANE call. Intercept it.

**A1. Locate the op.** Symbols/strings to anchor on in the odix runtime / `coreai` interpreter:
`load_embeddings`, `AFM_fused_interleaved_embedding_gather_dequant_reshape`, `gather_embeddings_*`,
`emb_embedding`, `in_embeddings`. Find the framework hosting the odix interpreter
(`GenerativeExperiencesRuntime` / `ModelManager` / `coreai`) and the function that produces the
embedding output. `dtrace -n 'pid$target:::entry' ` on a warm process, or lldb `image lookup -r
-n embedding`, to find candidate frames.

**A2. Break-and-dump.** With lldb attached (post-AMFI), set a breakpoint at the dequant output
write (or the `memcpy` into `in_embeddings`), run a prompt, and dump the `[seq, 1536]` fp16 tensor.
These are **ground-truth embedding rows for the exact input tokens** — validate immediately
(content words must cluster; `▁Paris`~`▁London` if both present).

**A3. Harvest the full table.** Two options:
   - *Row harvest:* feed prompts that tile the vocabulary (batches of distinct token ids), dump the
     activation each time, accumulate rows → full `[262144, 1536]`. ~hundreds of prompt runs;
     scriptable once A2 works. Deterministic and complete.
   - *Source + codec (cheaper if it works):* the breakpoint's registers/args expose the **source
     table pointer** and the gather/dequant parameters. Dump the source once (201 MB) and read the
     exact interleave/stride from the op — this is the `AFM_fused_interleaved` codec that resisted
     static cracking. One dump + the parameters = the whole table.

**Tooling:** lldb (breakpoints, `memory read`, `register read`), optionally DTrace for function
discovery, optionally Frida for scripted hooks. **Effort:** ~1–2 weeks once AMFI is cleared, most
of it in A1 (finding the exact frame). **Kill criterion:** if the dequant output can't be pinned to
a stable address/frame across runs, fall back to A3 row-harvest which only needs the output buffer.

---

## 3. Approach B — mid-inference activation harvest (fallback for the embedding)

If A1 can't isolate the op, capture the activation by timing instead of by symbol. The static
dumps missed it because the input activation is **freed after the forward**; a dump *during* the
forward keeps it.

- Attach lldb, set a breakpoint on the ANE submit (`ANEProgramProcessRequest`-class calls, or the
  Espresso/MPSGraph forward entry), so execution halts *after* `in_embeddings` is populated but
  *before* the buffer is reused. Dump the `[seq, 1536]` fp16 region (find it by the repeated-token
  signature — identical rows for repeated ids — now valid because the buffer is live).
- Then proceed as A3 row-harvest.

**Effort:** days once AMFI is cleared. **Risk:** buffer may be an IOSurface handed to the ANE
(mapped but in a wired region); still host-readable with root. This is the most robust path to the
embedding if the symbol route (A) stalls.

---

## 4. Approach C — ANE-resident weight capture (deployed FFN/attention bytes)

Only needed if the *weight bytes* (not just shapes) are wanted; the deployed FFN **shape** is
already recovered (`find:deployedffn`). The weights are patched into `__MKERN` from the raster via
a content-hash `dsid`.

- **C1 (host-side, easiest):** the `__MKERN`/`__DATA.__bss` regions are *process* memory once
  patched. A full core taken *mid-inference* (per B's timing) should contain the patched
  `__MKERN_0` (202 MB). Map metadata.bin's tile addresses into that region (they are offsets into
  this resident buffer — they did not match the *idle* full core because the buffer wasn't yet
  patched / was re-encoded). Validate tiles with the decay test, then the down-proj 48×256 codec.
- **C2 (driver-level, hard):** trace the ANE driver (`aned`, `AppleNeuralEngine.framework`) via
  IOKit/`ioreg` and DTrace on the `IOSurface`/`H11ANEIn` calls to capture the weight DMA. This is
  genuine kernel-adjacent RE — weeks, and largely unnecessary given C1 + the recovered shapes.

**Effort:** C1 ~days (rides on B's mid-inference capture); C2 weeks. **Do C1, skip C2** unless C1's
buffer is also re-encoded.

---

## 5. Recommended sequence

1. **Prove the AMFI prerequisite** (§1.2) — a 10-minute go/no-go. If it fails, stop; the route is
   blocked and the current boundary is final.
2. **Approach A2** — break at the CPU dequant output, dump one prompt's `[seq,1536]`, validate
   against the oracle. This alone confirms the whole route works.
3. **Approach A3 source-dump** — read the source pointer + codec params; one dump may yield the
   full table.
4. If A stalls → **Approach B** (timing-based) for the embedding, then **C1** for the FFN bytes on
   the same mid-inference core.
5. Assemble the standalone forward with the recovered embedding + deployed FFN and generate text.

## 6. Success criteria
- Embedding: recovered `[262144,1536]` scores **≥ +0.30** on the semantic probe (target +0.5),
  shift control collapses, and `▁Paris`→city neighbours are real.
- FFN bytes: recovered tiles pass the decay test (8–11) and reproduce the validated dense-weight
  structure.
- End-to-end: the from-weights forward emits Apple's greedy token for held-out prompts.

## 7. Effort, risk, and honesty
- **Total:** 2–4 weeks of focused work, *gated entirely on the AMFI prerequisite*. If AMFI can't be
  cleared on this build, effort is zero-yield.
- **Risk register:** (a) AMFI boot-arg ineffective → route blocked; (b) dequant frame not stable →
  fall to B; (c) buffers are IOSurfaces re-encoded for the ANE → C2 needed (weeks); (d) the source
  table uses the uncracked `AFM_fused_interleaved` codec even in memory → A3 row-harvest still
  works (it captures the *dequantized output*, bypassing the codec).
- This is a **legitimate own-hardware interoperability RE** effort (SIP already user-disabled). It
  instruments the operator's own running model; it publishes only original research (never Apple's
  weights/tokenizer), consistent with the repo's standing constraints.
- **Not in scope:** distributing recovered weights; bypassing signing/DRM; anything on hardware the
  operator doesn't own.
