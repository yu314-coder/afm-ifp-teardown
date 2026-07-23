"""Export pico (afmplus-v11.0-pico, 300M) to GGUF.

Uses the validated decode path:
  * embedding  : bit-exact (src/pico_embedding.py), semantically validated
  * weights    : codebook[nibble] * scale[output], z-order o = 16*bank + slot%16, i = slot//16
                 (round-trip validated to 0.981 = the 4-bit palettization floor)
  * norms      : gamma folded at ANE compile -> written as ones
  * arch       : 24 layers, D=1024, 16Q/4KV heads (head_dim 64), SwiGLU 3200,
                 RMSNorm, RoPE theta 500000, vocab 262144, tied embeddings

LOCAL ONLY. Contains Apple's weights + tokenizer -> never committed or published.
"""
import numpy as np, json, struct, sys, os
sys.path.insert(0, '/Volumes/D/fix/pico_shapes')
import importlib.util
spec = importlib.util.spec_from_file_location('pe', '/Volumes/D/fix/afm-ifp-teardown/src/pico_embedding.py')
PE = importlib.util.module_from_spec(spec); spec.loader.exec_module(PE)
import picolib
d = picolib._d
M = json.load(open('/Volumes/D/fix/afm-ifp-teardown/pico_weight_map.json'))

GEOM = {'N': (0x2080, 8192, 16), 's': (0x1080, 4096, 8), 'L': (0x6480, 25600, 16)}
NL, D, NQ, NKV, HD, FF, V = 24, 1024, 16, 4, 64, 3200, 262144
OUT = '/Volumes/D/fix/pico_gguf/afmplus-v11.0-pico-F16.gguf'
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
            r = np.asarray(d[p + 128:p + 128 + pay])
            nb = np.empty(pay * 2, np.uint8); nb[0::2] = r & 0xF; nb[1::2] = r >> 4
            slot = np.arange(pay * 2); o = slot % nout; i = slot // nout
            W[i, ob + b * nout + o] = cb[nb] * sc[o]
        ob += 16 * GEOM[cls][2]
    assert ob == cout, (role, ob, cout)
    return W

# ---------- GGUF primitives ----------
U8, I8, U16, I16, U32, I32, F32T, BOOL, STR, ARR, U64, I64, F64 = range(13)
def w_str(b, s):
    e = s.encode('utf-8'); b += struct.pack('<Q', len(e)) + e
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

add_str('general.architecture', 'llama')
add_str('general.name', 'afmplus-v11.0-pico')
add_u32('llama.block_count', NL)
add_u32('llama.context_length', 4096)
add_u32('llama.embedding_length', D)
add_u32('llama.feed_forward_length', FF)
add_u32('llama.attention.head_count', NQ)
add_u32('llama.attention.head_count_kv', NKV)
add_f32('llama.attention.layer_norm_rms_epsilon', 1e-6)
add_f32('llama.rope.freq_base', 500000.0)
add_u32('llama.rope.dimension_count', HD)
add_u32('general.file_type', 1)

vocab = json.load(open('/Volumes/D/fix/afm_odix/tok_vocab.json'))[:V]
# llama.cpp requires unique token strings; ids 4-7 mirror <pad>/<eos>/<bos>/<unk>
seen = {}
for i, t in enumerate(vocab):
    if t in seen:
        vocab[i] = '<dup%d_%s' % (i, t[1:])
    seen[vocab[i]] = i
ttype = [3 if (t.startswith('<') and t.endswith('>')) else 1 for t in vocab]
add_str('tokenizer.ggml.model', 'llama')
add_arr_str('tokenizer.ggml.tokens', vocab)
add_arr_i32('tokenizer.ggml.token_type', ttype)
add_arr_f32('tokenizer.ggml.scores', np.zeros(V, np.float32))
add_u32('tokenizer.ggml.bos_token_id', 1)
add_u32('tokenizer.ggml.eos_token_id', 110)   # <end_of_turn>, recovered from in_embeddings

# ---------- tensors ----------
print('decoding tensors ...', flush=True)
TENS = []   # (name, data[out,in] C-order, dims[in,out])
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
