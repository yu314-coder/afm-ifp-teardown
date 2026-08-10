"""Export pico (afmplus-v11.0-pico, 300M) to GGUF using the qwen3 architecture.

WHY qwen3 AND NOT llama:
  PICO_NORMS.md section 3 proves pico applies **per-head QK-norm** to Q and K
  (RMSNorm over head_dim=64, 48 of the 96 normalizations), with gamma = 1
  (parameter-free -- proven positively, see the PICO_NORMS addendum).

  The previous export used the `llama` architecture, which has NO QK-norm. So the
  exported model simply never applied that normalization: Q and K went into the
  attention scores unnormalized. That is a genuine functional defect independent of
  any weight-ordering question, and it alone is enough to destroy coherence.

  llama.cpp's `qwen3` arch is RMSNorm + per-head QK-norm + GQA + SwiGLU + RoPE --
  pico's exact recipe -- and provides blk.N.attn_q_norm / blk.N.attn_k_norm.
  Since pico's QK-norm gamma is 1, those tensors are written as ones: the *gain* is
  unity but the *normalization* now actually happens.

Everything else matches make_gguf.py: validated codebook/z-order decode, bit-exact
embedding, 24 layers, D=1024, 16Q/4KV heads (head_dim 64), SwiGLU 3200, RoPE 500000,
vocab 262144, tied embeddings, hidden RMSNorm gains = 1.

LOCAL ONLY. Contains Apple's weights + tokenizer -> never committed or published.
"""
import numpy as np, json, struct, sys, os
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import importlib.util
spec = importlib.util.spec_from_file_location('pe', '/Volumes/D/github/afm-ifp-teardown/src/pico_embedding.py')
PE = importlib.util.module_from_spec(spec); spec.loader.exec_module(PE)
import picolib
d = picolib._d
M = json.load(open('/Volumes/D/github/afm-ifp-teardown/pico_weight_map.json'))

GEOM = {'N': (0x2080, 8192, 16), 's': (0x1080, 4096, 8), 'L': (0x6480, 25600, 16)}
NL, D, NQ, NKV, HD, FF, V = 24, 1024, 16, 4, 64, 3200, 262144
ARCH = 'qwen3'
OUT = '/Volumes/D/github/afm-ifp-teardown/local/pico_gguf/afmplus-v11.0-pico-qwen3-F16.gguf'
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def decode(L, role, cin, cout):
    e = [x for x in M if x.get('layer') == L and x.get('role') == role][0]
    W = np.zeros((cin, cout), np.float32); ob = 0
    for off, cls in zip(e['block_offsets'], e['block_classes']):
        stride, pay, nout = GEOM[cls]; base = int(off, 16); nsc = nout
        for b in range(16):
            p = base + b * stride
            cb = np.frombuffer(bytes(d[p:p + 32]), dtype=np.float16).astype(np.float32)
            sc = np.frombuffer(bytes(d[p + 64:p + 64 + nsc * 2]), dtype=np.float16).astype(np.float32)
            # CORRECTED 2026-08-10: the payload begins after [palette 32][zeros 32]
            # [nout fp16 scales], i.e. at 64 + 2*nout -- +96 for the 16-output classes
            # and +80 for the 8-output 's' class. The old fixed +128 skipped 32 bytes of
            # real payload and read 32 bytes of the zero tail, which injected a constant
            # block: |mean|/sd 0.0114 vs 0.0002 corrected. See TOKENIZER_CORRECTION.md
            # and paper/latecycle.tex.
            hdr = 64 + 2 * nout
            r = np.asarray(d[p + hdr:p + hdr + pay])
            nb = np.empty(pay * 2, np.uint8); nb[0::2] = r & 0xF; nb[1::2] = r >> 4
            slot = np.arange(pay * 2); o = slot % nout; i = slot // nout
            W[i, ob + b * nout + o] = cb[nb] * sc[o]
        ob += 16 * GEOM[cls][2]
    assert ob == cout, (role, ob, cout)
    return W


# ---------- GGUF primitives ----------
U8, I8, U16, I16, U32, I32, F32T, BOOL, STR, ARR, U64, I64, F64 = range(13)
def gs(s):
    e = s.encode('utf-8'); return struct.pack('<Q', len(e)) + e
def kv(key, vtype, payload):
    return gs(key) + struct.pack('<I', vtype) + payload

KVS = []
def add_str(k, s):  KVS.append(kv(k, STR, gs(s)))
def add_u32(k, v):  KVS.append(kv(k, U32, struct.pack('<I', v)))
def add_f32(k, v):  KVS.append(kv(k, F32T, struct.pack('<f', v)))
def add_arr_str(k, lst):
    p = struct.pack('<IQ', STR, len(lst)) + b''.join(gs(x) for x in lst)
    KVS.append(kv(k, ARR, p))
def add_arr_i32(k, arr):
    a = np.asarray(arr, np.int32)
    KVS.append(kv(k, ARR, struct.pack('<IQ', I32, a.size) + a.tobytes()))
def add_arr_f32(k, arr):
    a = np.asarray(arr, np.float32)
    KVS.append(kv(k, ARR, struct.pack('<IQ', F32T, a.size) + a.tobytes()))

add_str('general.architecture', ARCH)
add_str('general.name', 'afmplus-v11.0-pico')
add_u32(ARCH + '.block_count', NL)
add_u32(ARCH + '.context_length', 4096)
add_u32(ARCH + '.embedding_length', D)
add_u32(ARCH + '.feed_forward_length', FF)
add_u32(ARCH + '.attention.head_count', NQ)
add_u32(ARCH + '.attention.head_count_kv', NKV)
add_u32(ARCH + '.attention.key_length', HD)
add_u32(ARCH + '.attention.value_length', HD)
add_f32(ARCH + '.attention.layer_norm_rms_epsilon', 1e-6)
add_f32(ARCH + '.rope.freq_base', 500000.0)
add_u32(ARCH + '.rope.dimension_count', HD)
add_u32('general.file_type', 1)

# CORRECTED 2026-08-10. This previously read local/afm_odix/tok_vocab.json, which is
# OFF BY FOUR (it used string index +1 where the asset means +5) and carried no real
# scores and no BYTE class. See TOKENIZER_CORRECTION.md. Read the asset directly:
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afm_tokenizer_asset import AFMTokenizerAsset
TOKENIZER_ASSET = os.environ.get('AFM_TOKENIZER_ASSET')
if not TOKENIZER_ASSET:
    raise SystemExit('set AFM_TOKENIZER_ASSET to your device\'s AssetData/tokenizer path')
_tk = AFMTokenizerAsset(TOKENIZER_ASSET)
_tk.check()
assert _tk.vocab_size == V, 'asset vocab %d != model vocab %d' % (_tk.vocab_size, V)
vocab = _tk.tokens()
ttype = [int(x) for x in _tk.gguf_token_type()]
scores = [float(x) for x in _tk.scores]
add_str('tokenizer.ggml.model', 'llama')
add_arr_str('tokenizer.ggml.tokens', vocab)
add_arr_i32('tokenizer.ggml.token_type', ttype)
add_arr_f32('tokenizer.ggml.scores', _tk.scores)  # real scores from the asset
add_u32('tokenizer.ggml.bos_token_id', 2)      # <bos>; was 1, off by the +4 vocab shift
add_u32('tokenizer.ggml.unknown_token_id', 3)  # <unk>
add_u32('tokenizer.ggml.padding_token_id', 0)  # <pad>
add_u32('tokenizer.ggml.eos_token_id', 106)   # <end_of_turn>; was 110 = 106+4

# ---------- tensors ----------
print('decoding tensors ...', flush=True)
TENS = []
E = np.stack([PE.embed_row(t) for t in range(V)]).astype(np.float16)
TENS.append(('token_embd.weight', E, [D, V]))
print('  embedding %s' % (E.shape,), flush=True)
for L in range(NL):
    for gg, role, cin, cout in [('attn_q', 'Q', D, D), ('attn_k', 'K', D, NKV * HD),
                                ('attn_v', 'V', D, NKV * HD), ('attn_output', 'O', D, D),
                                ('ffn_gate', 'gate', D, FF), ('ffn_up', 'up', D, FF),
                                ('ffn_down', 'down', FF, D)]:
        W = decode(L, role, cin, cout)
        TENS.append(('blk.%d.%s.weight' % (L, gg), np.ascontiguousarray(W.T).astype(np.float16), [cin, cout]))
    TENS.append(('blk.%d.attn_norm.weight' % L, np.ones(D, np.float32), [D]))
    TENS.append(('blk.%d.ffn_norm.weight' % L, np.ones(D, np.float32), [D]))
    # per-head QK-norm: proven present, gamma = 1 (PICO_NORMS.md sec.2-3)
    TENS.append(('blk.%d.attn_q_norm.weight' % L, np.ones(HD, np.float32), [HD]))
    TENS.append(('blk.%d.attn_k_norm.weight' % L, np.ones(HD, np.float32), [HD]))
    if L % 6 == 0: print('  layer %d' % L, flush=True)
TENS.append(('output_norm.weight', np.ones(D, np.float32), [D]))

ALIGN = 32
def pad(n): return (-n) % ALIGN
infos, off = [], 0
for name, arr, dims in TENS:
    gt = 1 if arr.dtype == np.float16 else 0
    infos.append(gs(name) + struct.pack('<I', len(dims)) +
                 b''.join(struct.pack('<Q', x) for x in dims) +
                 struct.pack('<IQ', gt, off))
    off += arr.nbytes; off += pad(off)

hdr = b'GGUF' + struct.pack('<IQQ', 3, len(TENS), len(KVS)) + b''.join(KVS) + b''.join(infos)
hdr += b'\0' * pad(len(hdr))
print('\nwriting %s' % OUT, flush=True)
with open(OUT, 'wb') as f:
    f.write(hdr)
    for name, arr, dims in TENS:
        b = arr.tobytes()
        f.write(b); f.write(b'\0' * pad(len(b)))
print('done: %.2f GB, %d tensors, %d metadata keys' % (os.path.getsize(OUT) / 1e9, len(TENS), len(KVS)), flush=True)
