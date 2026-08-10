"""Export afmplus-v11.0-nano (dense 3.33B) to a loadable GGUF using the `qwen3` architecture.

READ THE `general.description` STRING BELOW BEFORE TRUSTING THIS FILE FOR ANYTHING.
It is a STRUCTURALLY VALID but MATHEMATICALLY UNFAITHFUL export. It will load and run
in llama.cpp; it will NOT produce coherent text.

Inputs (all already produced by earlier stages, all on /Volumes/D):
  nano_weights.npz      350 decoded linear tensors, float32, keys "<layer>_<role>"
  nano_embedding.npy    [262144, 2048] float32 token embedding (decoded + validated)
  nano_tokenizer.json   262144 pieces + real SentencePiece scores + token types

Streaming writer: peak RAM is one tensor (~55 MB) plus a 64 MB embedding chunk.
LOCAL ONLY -- contains Apple's weights + tokenizer. Never commit, never publish.
"""
import numpy as np, json, struct, os, sys, zipfile, time

OUTDIR = '/Volumes/D/github/afm-ifp-teardown/local/nano_gguf'
NPZ    = os.path.join(OUTDIR, 'nano_weights.npz')
EMB    = os.path.join(OUTDIR, 'nano_embedding.npy')
TOK    = os.path.join(OUTDIR, 'nano_tokenizer.json')
OUT    = os.path.join(OUTDIR, 'afmplus-v11.0-nano-F16.gguf')

ARCH   = 'qwen3'
NL     = 56          # 35 segment_0 + 21 segment_1
N_SEG0 = 35
D      = 2048        # embedding_length
FF     = 6656        # feed_forward_length
HD     = 128         # head_dim  -- MEASURED from the shipped RoPE table (see description)
NQ     = D // HD     # 16 query heads
NKV    = 256 // HD   # 2 kv heads   (K/V width is 256)
V      = 262144
CTX    = 4096
THETA  = 500000.0
EPS    = 1e-5        # NOT recovered from the asset; family default
ALIGN  = 32

# Every segment_1 layer (35..55) gets K/V duplicated from this segment_0 layer.
KV_DONOR = N_SEG0 - 1   # 34 = nearest preceding segment_0 layer == the YOCO best-inference source

DESCRIPTION = """\
afmplus-v11.0-nano (Apple Foundation Models, dense 3.33B) reverse-engineered from the
shipped on-device asset (program.odix + MPSGraph/binary_0.hwx) and re-emitted as GGUF.

*** THIS IS NOT A FAITHFUL EXPORT. IT LOADS AND RUNS BUT DOES NOT PRODUCE COHERENT TEXT. ***

Four deliberate, documented deviations from the real model:

1. FABRICATED K/V FOR 21 LAYERS (the big one).
   The real nano is a KVReuseTransformerLayerSequence: only its first 35 layers
   ("segment_0") have K and V projections. The last 21 layers ("segment_1") have
   Q and O only and re-use keys/values produced elsewhere (cross-layer KV sharing;
   Apple's own KV cache state tensor has exactly 35 slots for 56 layers).
   No llama.cpp architecture supports cross-layer KV sharing, so this export gives
   every segment_1 layer its own K/V tensor by DUPLICATING layer 34's K and V
   (the nearest preceding segment_0 layer, which is also the best-guess YOCO source
   -- the true source layer was never proven). Layers 35..55 therefore contain
   weights that do not exist in Apple's model. This changes the computation.

2. RMSNorm GAINS ARE ALL ONES, AND THAT IS KNOWN-WRONG.
   nano's exported program declares 280 real norm PARAMETERS (223 hidden sandwich
   gammas + 1 output_norm + 56 query_norms). Their VALUES were never located in any
   shipped file, and a folding probe shows they are not folded into the linears.
   Every *_norm.weight here is a vector of ones. (The 35 key-norms genuinely are
   parameter-free -- the string "key_norm" appears zero times in the asset -- so the
   attn_k_norm ones are correct.)

3. SANDWICH NORMS ARE DROPPED.
   The real per-layer recipe is h = x + post_n(attn(pre_n(x))); h = h + post_n(ffn(pre_n(h)))
   -- four hidden RMSNorms per layer (316 total normalizations counted in the MPSGraph).
   qwen3 has only the two pre-norms, so the two post-norms per layer are simply absent.

4. INTRA-TENSOR CHANNEL / HEAD ORDER IS UNSOLVED.
   The block-to-role layout and the 2-bit palettized codec are proven, and every
   tensor decodes at the correct shape with a trained spectrum, but the permutation
   of the FFN hidden axis and of the attention head axis inside each tensor was never
   recovered. Rows/columns are therefore in ANE storage order, not model order.
   (Additionally: gate=linear_0 / up=linear_1 rests on SwiGLU convention alone, and
   segment_0's V/O decode is the one role that still fails its spectral check.)

What IS trustworthy here: the tokenizer (decoded from Apple's own tokenizer asset,
real SentencePiece scores and token types, +5 offset shift proven three ways), the
token embedding (blind-sweep layout, semantically validated: nearest neighbours of
"Paris" are Paris/Paris/paris/France/London), rope_theta = 500000 (MEASURED from the
shipped cos/sin table, R^2 = 1.00000000), head_dim = 128 (the shipped rotate-half cos
table is 128 wide with 64 distinct frequencies -- so 16 Q heads / 2 KV heads, NOT the
32x64 that earlier notes assumed), the 56/2048/6656/4096 shape, tied embeddings, and
the 2-bit palette [-1.5,-0.5,+0.5,+1.5] weight codec.

norm_eps was NOT recoverable (no eps literal exists anywhere in the shipped files --
it is baked into the ANE kernel); 1e-5 is a placeholder.
"""

# ---------------------------------------------------------------- GGUF primitives
U8, I8, U16, I16, U32, I32, F32T, BOOL, STR, ARR, U64, I64, F64 = range(13)
GGML_F32, GGML_F16 = 0, 1

def gs(s):
    e = s.encode('utf-8')
    return struct.pack('<Q', len(e)) + e

KVS = []
def kv(key, vtype, payload): KVS.append(gs(key) + struct.pack('<I', vtype) + payload)
def add_str(k, s):  kv(k, STR,  gs(s))
def add_u32(k, v):  kv(k, U32,  struct.pack('<I', v))
def add_i32(k, v):  kv(k, I32,  struct.pack('<i', v))
def add_f32(k, v):  kv(k, F32T, struct.pack('<f', v))
def add_bool(k, v): kv(k, BOOL, struct.pack('<B', 1 if v else 0))
def add_arr_str(k, lst):
    kv(k, ARR, struct.pack('<IQ', STR, len(lst)) + b''.join(gs(x) for x in lst))
def add_arr_i32(k, a):
    a = np.asarray(a, np.int32); kv(k, ARR, struct.pack('<IQ', I32, a.size) + a.tobytes())
def add_arr_f32(k, a):
    a = np.asarray(a, np.float32); kv(k, ARR, struct.pack('<IQ', F32T, a.size) + a.tobytes())

# ---------------------------------------------------------------- metadata
add_str('general.architecture', ARCH)
add_u32('general.quantization_version', 2)
add_u32('general.alignment', ALIGN)
add_str('general.type', 'model')
add_str('general.name', 'afmplus-v11.0-nano')
add_str('general.basename', 'afmplus-v11.0-nano')
add_str('general.size_label', '3.3B')
add_str('general.description', DESCRIPTION)
add_u32('general.file_type', 1)                       # MOSTLY_F16

add_u32(ARCH + '.block_count', NL)
add_u32(ARCH + '.context_length', CTX)
add_u32(ARCH + '.embedding_length', D)
add_u32(ARCH + '.feed_forward_length', FF)
add_u32(ARCH + '.attention.head_count', NQ)
add_u32(ARCH + '.attention.head_count_kv', NKV)
add_u32(ARCH + '.attention.key_length', HD)
add_u32(ARCH + '.attention.value_length', HD)
add_f32(ARCH + '.attention.layer_norm_rms_epsilon', EPS)
add_f32(ARCH + '.rope.freq_base', THETA)
add_u32(ARCH + '.rope.dimension_count', HD)
add_u32(ARCH + '.vocab_size', V)

# provenance / warning keys (ignored by llama.cpp, readable with gguf-dump)
add_bool('afm.faithful', False)
add_str('afm.warning', 'NOT a faithful export: 21 layers have fabricated (duplicated) K/V, '
                       'all RMSNorm gammas are ones (known-wrong), sandwich post-norms dropped, '
                       'intra-tensor channel/head order unsolved. Will not produce coherent text.')
add_str('afm.kv_sharing.real', 'segment_0 = layers 0..34 have K/V; segment_1 = layers 35..55 have none '
                               '(KVReuseTransformerLayerSequence, 35-slot KV cache for 56 layers)')
add_str('afm.kv_sharing.export_hack', 'blk.35..55 attn_k/attn_v are byte-copies of blk.%d' % KV_DONOR)
add_str('afm.norm.gamma', 'NOT recovered; all *_norm.weight written as ones. 280 real norm parameters '
                          'exist in the asset (223 sandwich + 1 output + 56 query); only the 35 key-norms '
                          'are genuinely parameter-free.')
add_str('afm.norm.eps', 'NOT recovered from the asset (baked into the ANE kernel); 1e-5 is a placeholder')
add_str('afm.rope.theta', 'MEASURED 499999.94 (R^2=1.00000000) from the shipped cos/sin table; rotate-half, ctx 4096')
add_str('afm.head_dim', 'MEASURED 128 from the 128-wide rotate-half cos table (64 distinct freqs) '
                        '=> 16 Q heads, 2 KV heads. Earlier notes assuming 32x64 are wrong.')
add_str('afm.ordering', 'intra-tensor channel/head permutation UNSOLVED; rows are in ANE storage order')
add_str('afm.source', 'model.odixpackage: program.odix + MPSGraph/binary_0.hwx (2-bit palettized, '
                      'palette [-1.5,-0.5,+0.5,+1.5])')

# ---------------------------------------------------------------- tokenizer
tk = json.load(open(TOK))
assert tk['vocab_size'] == V and len(tk['tokens']) == V
tokens = tk['tokens']
# llama.cpp requires unique pieces; the asset has none-duplicated pieces but assert it.
assert len(set(tokens)) == V, 'duplicate token pieces'
add_str('tokenizer.ggml.model', tk['model'])          # 'llama' (SentencePiece, score-driven merges)
add_arr_str('tokenizer.ggml.tokens', tokens)
add_arr_f32('tokenizer.ggml.scores', tk['scores'])
# token_type_gguf promotes the named specials (ids 4..237) UNUSED->USER_DEFINED so that
# <start_of_turn>/<end_of_turn> are tokenizable; the verbatim Apple types are in the json too.
add_arr_i32('tokenizer.ggml.token_type', tk['token_type_gguf'])
add_u32('tokenizer.ggml.bos_token_id', tk['bos_token_id'])   # 2  <bos>
add_u32('tokenizer.ggml.eos_token_id', tk['eos_token_id'])   # 106 <end_of_turn>
add_u32('tokenizer.ggml.eot_token_id', tk['eot_token_id'])   # 106
add_u32('tokenizer.ggml.padding_token_id', tk['pad_token_id'])
add_u32('tokenizer.ggml.unknown_token_id', tk['unk_token_id'])
add_bool('tokenizer.ggml.add_bos_token', bool(tk['add_bos']))
add_bool('tokenizer.ggml.add_eos_token', bool(tk['add_eos']))
add_str('tokenizer.chat_template',
        "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n"
        "{{ m['content'] | trim }}<end_of_turn>\n{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")

# ---------------------------------------------------------------- tensor plan
# src spec: ('npz', key) | ('emb',) | ('ones', n)
TENS = []   # (name, ne_dims_low_to_high, ggml_type, src)
TENS.append(('token_embd.weight', [D, V], GGML_F16, ('emb',)))
for L in range(NL):
    kvL = L if L < N_SEG0 else KV_DONOR
    TENS += [
        ('blk.%d.attn_norm.weight'   % L, [D],      GGML_F32, ('ones', D)),
        ('blk.%d.attn_q.weight'      % L, [D, D],   GGML_F16, ('npz', '%d_Q' % L)),
        ('blk.%d.attn_k.weight'      % L, [D, 256], GGML_F16, ('npz', '%d_K' % kvL)),
        ('blk.%d.attn_v.weight'      % L, [D, 256], GGML_F16, ('npz', '%d_V' % kvL)),
        ('blk.%d.attn_output.weight' % L, [D, D],   GGML_F16, ('npz', '%d_O' % L)),
        ('blk.%d.attn_q_norm.weight' % L, [HD],     GGML_F32, ('ones', HD)),
        ('blk.%d.attn_k_norm.weight' % L, [HD],     GGML_F32, ('ones', HD)),
        ('blk.%d.ffn_norm.weight'    % L, [D],      GGML_F32, ('ones', D)),
        ('blk.%d.ffn_gate.weight'    % L, [D, FF],  GGML_F16, ('npz', '%d_gate' % L)),
        ('blk.%d.ffn_up.weight'      % L, [D, FF],  GGML_F16, ('npz', '%d_up'   % L)),
        ('blk.%d.ffn_down.weight'    % L, [FF, D],  GGML_F16, ('npz', '%d_down' % L)),
    ]
TENS.append(('output_norm.weight', [D], GGML_F32, ('ones', D)))
# tied embeddings: no output.weight -- llama.cpp reuses token_embd

ESZ = {GGML_F32: 4, GGML_F16: 2}
def nbytes(dims, gt): return int(np.prod(dims)) * ESZ[gt]
def pad(n): return (-n) % ALIGN

infos, off = [], 0
for name, dims, gt, src in TENS:
    infos.append(gs(name) + struct.pack('<I', len(dims)) +
                 b''.join(struct.pack('<Q', d) for d in dims) +
                 struct.pack('<IQ', gt, off))
    off += nbytes(dims, gt); off += pad(off)

hdr = b'GGUF' + struct.pack('<IQQ', 3, len(TENS), len(KVS)) + b''.join(KVS) + b''.join(infos)
hdr += b'\0' * pad(len(hdr))

# ---------------------------------------------------------------- write
z = zipfile.ZipFile(NPZ)
def npz_get(key):
    with z.open(key + '.npy') as f:
        return np.lib.format.read_array(f, allow_pickle=False)

emb = np.load(EMB, mmap_mode='r')
assert emb.shape == (V, D)

print('writing %s' % OUT, flush=True)
print('  %d tensors, %d metadata keys, header %d B, data %d B' % (len(TENS), len(KVS), len(hdr), off), flush=True)
t0 = time.time()
written = 0
with open(OUT, 'wb') as f:
    f.write(hdr)
    for i, (name, dims, gt, src) in enumerate(TENS):
        want = nbytes(dims, gt)
        if src[0] == 'ones':
            f.write(np.ones(src[1], np.float32).tobytes())
        elif src[0] == 'emb':
            for r0 in range(0, V, 8192):                      # 8192*2048*4 = 64 MB chunks
                f.write(np.ascontiguousarray(emb[r0:r0 + 8192]).astype(np.float16).tobytes())
        else:
            a = npz_get(src[1])
            # npz stores (cout, cin) row-major == exactly what GGUF wants for ne=[cin, cout]
            assert a.shape == (dims[1], dims[0]), (name, a.shape, dims)
            f.write(np.ascontiguousarray(a).astype(np.float16).tobytes())
            del a
        written += want
        f.write(b'\0' * pad(want))
        if i % 60 == 0:
            print('   [%3d/%d] %-32s %6.1f GB  %.0fs' %
                  (i, len(TENS), name, written / 1e9, time.time() - t0), flush=True)

sz = os.path.getsize(OUT)
print('done: %s' % OUT, flush=True)
print('  %d bytes (%.3f GiB), %d tensors, %d metadata keys, %.0fs' %
      (sz, sz / 2**30, len(TENS), len(KVS), time.time() - t0), flush=True)
assert sz == len(hdr) + off, (sz, len(hdr) + off)
nparam = sum(int(np.prod(d)) for _, d, _, _ in TENS)
print('  %d parameters in file' % nparam, flush=True)
