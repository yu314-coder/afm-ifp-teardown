# H16 ANE-ISA Decode — Scope of Work (evidence-based)

The **one** remaining gate to a complete per-tensor pico weight set (see `PICO_WEIGHT_SCOPE.md`): the
byte-exact tile→tensor map lives only in the compiled ANE program (`binary_0.hwx` `__TEXT`, 32.4 MB,
H16 ISA v17). Everything else — quant format (affine int4 `(q−7.5)·S`), segment layout, 64 KB tiling,
contiguous decode (R≈5 clean) — is already solved. This document scopes the ISA decode itself.

Static analysis of the operator's own asset only; no Apple weights are or will be committed.

---

## 0. What is already known about the ISA (do not re-derive)

From the 3B ANE-program analysis (`afm_odix/hwx_expert_dma.json`) and the pico `__TEXT` strings, the
mechanism is understood at the level below — the decode is *engineering it out*, not discovering it:

- **Container**: Mach-O-like, magic `0xbeefface`, ANE cputype `0x80`, H16/ISA v17, `LC_SEGMENT_64`
  load commands. Weight blobs = `__KERN_0/1/2`; LoRA = runtime-patched `__MKERN_*`; program = `__TEXT`.
- **Weight addressing is `rt_op`-relocated, not static.** Pico's `__TEXT` contains
  `rt_op_map_nm_kernel_` (maps **n**on-**m**utable = static `__KERN` weight buffers),
  `rt_op_map_mutable_kernel__ane` (LoRA), `rt_op_alloc/patch_m_kernel_`, and
  `_hi_replaceable_dsid_usage_dsid_relocation`. A weight buffer's runtime base is produced by an
  affine `rt_op` (`src_offset = arg5·stride + arg6·8 + planar_base`, arg5/arg6 = driver-supplied
  `dsid` relocation params `0x1abc`/`0x1abb`). So each `__KERN` tile range is bound to a **dsid**, and
  the program maps `dsid → compute-op operand`.
- **Program is region-structured.** The 3B splits into named `regions` (compute functions) each with
  `args` (`__arg0/1/…`), `mutable_kernels`, and `n_dynamic_offset_ops`; the parser that produced this
  already exists in `afm_odix/` (see `build_const_map*.py`, `build_ffn_assignment.py`,
  `build_shape_table.py`) and can be re-pointed at pico.
- **Command stream markers** (pico `__TEXT` u32 census): `0x00002080` (×47848), `0x00fff820`,
  and power-of-two operands (`0x40/0x80/0x200/0x400/0x800` = tile/stride sizes) — candidate
  opcode/field structure to lock down first.

## 1. The decode tasks (in order)

1. **Lock the command/instruction framing.** Determine the `__TEXT` command record structure (the
   `0x2080`-family markers, record stride, opcode field, operand-count). Deliverable: a walker that
   iterates ANE ops with (opcode, operands[]).
2. **Extract the static-weight bindings.** Decode the `rt_op_map_nm_kernel_` ops → the set of
   `(dsid → __KERN offset range)` for the 2280 weight tiles. This is the tile→**buffer** map.
3. **Reconstruct the compute graph.** For each conv/matmul op, recover its `(weight dsid, input act,
   output act, Cout, Cin, tile shape)`. The `dsid` links op→weight buffer (task 2); the act operands
   (`__DATA` 0x30000000 range) chain ops into the dataflow.
4. **Assign ops → transformer tensors.** Order the ops by dataflow (layer 0 → 23), and label each by
   role from shape + region name (`attention_qkv_transform`, `hidden_transform_linear_0/1`,
   `output_transform`) → the 168 `(name, __KERN offset, [Cout,Cin], scale block)` tuples.
5. **Decode + validate.** Read each tensor as contiguous affine int4 `(q−7.5)·S`, pair its fp16 scale,
   structure-test (target R≈5), and assemble the 24-layer forward; check residual stability.

## 2. Reusable prior art

- **In-repo**: the 3B hwx parser (`afm_odix/build_const_map*.py`, `build_ffn_assignment.py`,
  `hwx_expert_dma.json`) already walks regions, `rt_op`s, and dsid relocation for the *sparse* model;
  pico is *simpler* (dense, static `__KERN`, no IFP gather), so it is a reduction of solved work.
- **External**: `ANECompiler.framework` / `AppleNeuralEngine.framework` symbols, Espresso, and public
  ANE-RE (e.g. tinygrad's ANE backend, `coremltools` MIL→ANE) document the H-series op set and the
  `.hwx`/`ANEC` layout at a level sufficient to cross-check the framing.

## 3. Effort, risk, deliverable

- **Effort**: the mechanism is known and pico is the simplest variant (dense, static weights, one
  spec-decode program), so this is **bounded engineering, not open discovery** — realistically
  **1–2 focused weeks**, most of it in tasks 1 & 3 (framing + graph reconstruction).
- **Risk**: (a) the `0x2080` framing may be one of several op families → iterate against the known
  1740 tile-aligned refs as anchors; (b) the dsid→offset relocation may need a driver-side constant
  that is supplied at load, not baked (the 3B's arg5/arg6 were driver-supplied) — if so, the *relative*
  layout is still recoverable and only the absolute base needs one dynamic read.
- **Deliverable**: the 168 per-tensor `(offset, shape, scale)` bindings ⇒ a complete, structure-validated
  pico weight `state_dict` (matching the status the 3B linear weights reached), plus the
  dynamically-harvestable embedding. **Ceiling unchanged**: per-layer activations are ANE-internal, so
  a from-weights standalone can be structure/residual-validated but not verified against Apple's exact
  greedy token — the same information limit as the 3B, not an ISA problem.

## 4. Recommendation
Do tasks 1–2 first as a **go/no-go**: if the `rt_op_map_nm_kernel_` ops decode to clean
`(dsid → __KERN offset)` bindings that line up with the 1740 tile-aligned refs, the rest is mechanical
graph-walking; if the framing resists, fall back to the coreml2hwx route (compile a probe of each pico
shape through Apple's toolchain and match byte permutations, as the 3B did for its de-swizzle).

### Go/no-go RESULT (2026-07-17): the per-tensor map is in task 3, not task 2.
The 108 `rt_op_map_nm_kernel_*` ops name **`KernelSymbolStart0/1/2`** (= the `__KERN_0/1/2` segments),
each tied to a LoRA/seq **specialization variant** (`lora_32_extend_2048_8`, `…_ANE_region_0_0`, …).
So this layer maps whole `__KERN` **segments** to regions — information **already recovered** from the
`LC_SEGMENT_64` table. The **per-tensor `__KERN` offsets are therefore not in the `rt_op_map` layer**;
they are in the per-op operands of the compute graph (**task 3**), which requires the full command
framing (task 1) + operand decode. NET: the cheap shortcut is exhausted; a complete per-tensor pico
weight set genuinely needs the task-1/3 ISA framing+graph decode (the ~1–2 week engineering effort),
or the coreml2hwx fallback per shape. This is the true floor of the pico weight work.
