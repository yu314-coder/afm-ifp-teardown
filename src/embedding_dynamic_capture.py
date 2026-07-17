"""embedding_dynamic_capture.py -- recover an AFM token embedding by dynamic host capture.

This is the METHOD that broke the embedding wall (paper Findings find:embrecover / find:3bane).
It recovers the *dequantized* token-embedding rows the running model feeds forward, bypassing the
(uncracked) storage codec entirely. It works for the small on-device model (afmplus-v11.0-pico,
D=1024), whose gather/dequant runs on the host CPU; it does NOT work for the 3B (D=1536), whose
gather/dequant is ANE-delegated so the dequantized vector never reaches host DRAM.

NOTHING here contains or emits Apple's weights -- only the capture/decoding procedure and the
open-source semantic oracle. Recovered embedding bytes are the user's own runtime data and are
never committed.

--------------------------------------------------------------------------------------------------
PREREQUISITES (operator's own hardware; read-only static analysis, one privileged dynamic step)
--------------------------------------------------------------------------------------------------
  - SIP disabled (user choice) and Developer Mode on:  sudo DevToolsSecurity -enable
  - root for the attach (the daemon runs as _modelmanagerd):  sudo lldb ...
  - `afm`: any CLI/REPL over Apple's FoundationModels that drives the on-device model.

--------------------------------------------------------------------------------------------------
CAPTURE (why it is shaped this way)
--------------------------------------------------------------------------------------------------
The embedding lookup is position-independent (RoPE is applied later, in attention), so a prompt of
ONE token repeated N times makes `in_embeddings` an [N, D] buffer of N BYTE-IDENTICAL fp16 rows --
a period-D signature that is trivial to find and needs no knowledge of the storage layout.

    # one token, repeated ~1500x, keeps the prefill (and its buffer) live long enough to grab:
    python3 -c "import sys;sys.stdout.write('dog'+(' dog'*1500))" > /tmp/w.txt
    ( afm < /tmp/w.txt >/dev/null 2>&1 ) &
    # wait until the daemon is resident (RSS grows), then attach BY PID and core.
    # NB: attach-by-pid supports save-core; `--waitfor` does NOT (qMemoryRegionInfo unsupported)
    #     and also cannot resolve the E5RT symbols in the freshly-spawned process.
    PID=$(pgrep -f TGOnDeviceInferenceProviderService | head -1)
    sudo lldb -o "process attach --pid $PID" \
              -o "process save-core --style=modified-memory /tmp/tok.core" \
              -o detach -o quit

Repeat once per token to harvest. A full [V,D] table is NOT practical this way (V=262144); the
common-token subset (a few captures) is. The on-disk storage codec stays uncracked: rowmajor,
[8,1,1]-interleave, and the ANE [512,D] conv-tiling all score at chance (0.57-0.64 zero-pattern /
sign agreement) even fitted against multiple exact ground-truth rows -- which is exactly why the
dynamic route is needed.
"""
import numpy as np

ROW_FP16 = None  # set to D at call time


def find_identical_row_buffer(core_path, D=1024, min_run=8, min_unique_bytes=32):
    """Locate the repeated-token embedding buffer: the longest run of identical, NON-constant
    D-wide fp16 rows. Returns (row_fp32, n_rows, byte_offset) or (None, 0, -1).

    The non-constant filter (min_unique_bytes) rejects zero/padding fills, which trivially repeat
    at every period. Returns the exact dequantized embedding vector for the captured token.
    """
    d = np.memmap(core_path, dtype=np.uint8, mode="r")
    N = len(d)
    P = D * 2  # bytes per fp16 row
    a = np.asarray(d[: N - P]); b = np.asarray(d[P:])
    eq = (a == b)
    nb = len(eq) // P
    dens = eq[: nb * P].reshape(nb, P).mean(axis=1)   # fraction-equal per row-block
    hi = dens > 0.99
    best = (0, 0)
    i = 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j < len(hi) and hi[j]:
                j += 1
            if j - i >= min_run:
                off = i * P
                if len(np.unique(np.frombuffer(bytes(d[off:off + P]), dtype=np.uint8))) > min_unique_bytes:
                    if j - i > best[0]:
                        best = (j - i, off)
            i = j
        else:
            i += 1
    if best[0] == 0:
        return None, 0, -1
    row = np.frombuffer(bytes(d[best[1]:best[1] + P]), dtype=np.float16).astype(np.float32)
    return row, best[0], best[1]


def cos(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def oracle(emb):
    """Orthographic-pair semantic probe on a dict {token: vector}. A genuine embedding shows
    strong same-lemma similarity (dog~dogs, king~kings) and near-zero cross-category similarity.

    Validated result on recovered pico embeddings:
        dog~dogs = +0.648,  king~kings = +0.664,  cross-category +0.01..0.20  (real embedding space)
    """
    related = [("_dog", "_dogs"), ("_king", "_kings"), ("_time", "_times"), ("_year", "_years")]
    control = [("_dog", "_king"), ("_dogs", "_kings"), ("_dog", "_London"), ("_king", "_Paris")]
    r = [cos(emb[a], emb[b]) for a, b in related if a in emb and b in emb]
    c = [cos(emb[a], emb[b]) for a, b in control if a in emb and b in emb]
    for a, b in related:
        if a in emb and b in emb:
            print("  related %-8s~%-8s = %+.3f" % (a, b, cos(emb[a], emb[b])))
    for a, b in control:
        if a in emb and b in emb:
            print("  control %-8s~%-8s = %+.3f" % (a, b, cos(emb[a], emb[b])))
    if r and c:
        print("  MEAN related=%+.3f  control=%+.3f  GAP=%+.3f" % (np.mean(r), np.mean(c), np.mean(r) - np.mean(c)))
    return (np.mean(r) - np.mean(c)) if (r and c) else None


if __name__ == "__main__":
    import sys
    # usage: python embedding_dynamic_capture.py tok_dog.core [D]
    core = sys.argv[1]
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    row, n, off = find_identical_row_buffer(core, D=D)
    if row is None:
        print("no identical-row buffer at D=%d -- (3B is ANE-delegated: expect none at D=1536)" % D)
    else:
        print("recovered token embedding: D=%d, %d identical rows @0x%x, nnz=%d, |v|<=%.3f"
              % (D, n, off, (row != 0).sum(), np.abs(row).max()))
