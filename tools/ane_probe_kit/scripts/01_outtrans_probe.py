"""THE decisive test: does this host's ANE compiler emit a WEIGHT-BEARING OutTrans=1 conv?

Background. In Apple's shipped on-device model, the two weight tensors that resist decoding are
exactly the two marked `OutTrans=1`, and 180 of that model's 363 OutTrans=1 tasks carry
coefficient banks. On the reference host, no synthetic graph ever puts OutTrans=1 on a conv that
carries coefficients -- the flag always lands on a *weightless* shuffle task instead, so the
flag's effect on the coefficient payload cannot be observed. The ANE compile is done by
ANECompiler.framework, which ships with macOS, so a different macOS build is the only remaining
variable.

This script compiles several graphs and reports, per ANE task, whether OutTrans=1 coincides with
coefficient banks. Everything uses synthetic random weights -- no Apple assets are read.

Graphs, in increasing order of how hard they push the scheduler:
  bare_conv          minimal baseline
  Lclass_bias        the shipped model's exact down-proj geometry and header mode (0x6480)
  transpose_conv_add head-merge transpose -> conv -> residual add  (first thing that ever
                     produced OutTrans=1 at all, though weightlessly)
  real_MHA           full multi-head attention: QKV convs, head reshape, QK^T, softmax, V matmul,
                     head-merge transpose -> output conv -> residual add
  real_MHA_x2        two stacked MHA blocks sharing one residual stream
"""
import os, sys, glob, re, subprocess, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, 'bin')
WORK = os.path.join(ROOT, 'work')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.chdir(WORK)

try:
    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    from coremltools.converters.mil import Builder as mb
except Exception as e:
    print('FATAL: could not import coremltools/numpy: %s' % e)
    print('Create the venv first:  python3 -m venv venv && ./venv/bin/pip install coremltools numpy')
    sys.exit(2)

MIL2HWX = os.path.join(BIN, 'mil_to_hwx')
HP = os.path.join(BIN, 'hwx_parsing')

D, H, HD, S = 1024, 16, 64, 64          # residual width, heads, head_dim, sequence
CIN, COUT = 3200, 256                   # the shipped down-proj task geometry
rng = np.random.RandomState(0)
W_D = (rng.randn(D, D, 1, 1) * 0.02).astype(np.float32)
W_L = (rng.randn(COUT, CIN, 1, 1) * 0.05).astype(np.float32)
BIAS_L = np.full(COUT, 0.5, np.float32)     # nonzero bias -> emits the shipped 0x6480 header


# ------------------------------------------------------------------ graphs
def g_bare():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CIN, 1, S))])
    def prog(x):
        return mb.conv(x=x, weight=W_L, strides=[1, 1], pad_type='valid', name='conv')
    return prog


def g_lclass_bias():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CIN, 1, S))])
    def prog(x):
        return mb.conv(x=x, weight=W_L, bias=BIAS_L, strides=[1, 1], pad_type='valid', name='conv')
    return prog


def g_transpose_conv_add():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, H, HD, S)), mb.TensorSpec(shape=(1, D, 1, S))])
    def prog(x, res):
        t = mb.transpose(x=x, perm=[0, 1, 3, 2], name='tr')
        r = mb.reshape(x=t, shape=[1, D, 1, S], name='merge')
        c = mb.conv(x=r, weight=W_D, strides=[1, 1], pad_type='valid', name='O')
        return mb.add(x=c, y=res, name='add')
    return prog


def _mha_block(x, i):
    Wq = (np.random.RandomState(i + 1).randn(D, D, 1, 1) * 0.02).astype(np.float32)
    q = mb.conv(x=x, weight=Wq, strides=[1, 1], pad_type='valid', name='Q_%d' % i)
    k = mb.conv(x=x, weight=Wq, strides=[1, 1], pad_type='valid', name='K_%d' % i)
    v = mb.conv(x=x, weight=Wq, strides=[1, 1], pad_type='valid', name='V_%d' % i)
    qh = mb.reshape(x=q, shape=[1, H, HD, S], name='qh_%d' % i)
    kh = mb.reshape(x=k, shape=[1, H, HD, S], name='kh_%d' % i)
    vh = mb.reshape(x=v, shape=[1, H, HD, S], name='vh_%d' % i)
    qt = mb.transpose(x=qh, perm=[0, 1, 3, 2], name='qt_%d' % i)
    sc = mb.matmul(x=qt, y=kh, name='scores_%d' % i)
    p = mb.softmax(x=sc, axis=-1, name='sm_%d' % i)
    vt = mb.transpose(x=vh, perm=[0, 1, 3, 2], name='vt_%d' % i)
    a = mb.matmul(x=p, y=vt, name='attn_%d' % i)
    am = mb.transpose(x=a, perm=[0, 1, 3, 2], name='merge_tr_%d' % i)
    ar = mb.reshape(x=am, shape=[1, D, 1, S], name='merge_rs_%d' % i)
    o = mb.conv(x=ar, weight=W_D, strides=[1, 1], pad_type='valid', name='O_%d' % i)
    return mb.add(x=x, y=o, name='res_%d' % i)


def g_real_mha():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D, 1, S))])
    def prog(x):
        return _mha_block(x, 0)
    return prog


def g_real_mha_x2():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D, 1, S))])
    def prog(x):
        for i in range(2):
            x = _mha_block(x, i)
        return x
    return prog


GRAPHS = [('bare_conv', g_bare), ('Lclass_bias', g_lclass_bias),
          ('transpose_conv_add', g_transpose_conv_add),
          ('real_MHA', g_real_mha), ('real_MHA_x2', g_real_mha_x2)]


# ------------------------------------------------------------------ compile + audit
def _compile_mlpackage(pkg, tag):
    """Produce a .mlmodelc. Prefer xcrun (full Xcode); fall back to coremltools, which
    uses the system CoreML framework and needs NO Xcode -- only CommandLineTools."""
    mlc = 'mlc_' + tag
    subprocess.run(['rm', '-rf', mlc])
    r = subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg, mlc],
                       capture_output=True, text=True)
    got = glob.glob('%s/*.mlmodelc' % mlc)
    if got:
        return got[0], 'xcrun'
    # ---- fallback: no Xcode needed ----
    try:
        import shutil
        # NOTE: get_compiled_model_path() returns a TEMPORARY directory whose lifetime is
        # tied to the MLModel object. The reference must be held until the copy completes,
        # otherwise the model is garbage-collected and the directory vanishes mid-copy.
        ml = ct.models.MLModel(pkg)
        cp = ml.get_compiled_model_path()
        dst = os.path.abspath(tag + '.mlmodelc')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(cp, dst)
        del ml
        return dst, 'coremltools'
    except Exception as e:
        return None, 'both failed (xcrun: %s | coremltools: %s)' % (
            (r.stderr or r.stdout).strip().splitlines()[-1][:70] if (r.stderr or r.stdout) else '?',
            str(e)[:70])


def compile_graph(progfn, tag):
    m = ct.convert(progfn(), convert_to='mlprogram',
                   minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT16)
    m = cto.palettize_weights(m, cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        nbits=4, mode='kmeans', granularity='per_tensor', weight_threshold=256)))
    pkg = tag + '.mlpackage'
    subprocess.run(['rm', '-rf', pkg]); m.save(pkg)
    src, how = _compile_mlpackage(pkg, tag)
    if src is None:
        return None, 'compile: %s' % how
    hx = 'hx_' + tag
    subprocess.run(['rm', '-rf', hx])
    r2 = subprocess.run([MIL2HWX, '-a', 'h16g', '-i', src + '/', '-o', hx, tag],
                        capture_output=True, text=True, env=dict(os.environ, ANE_ARCH_ANY='1'))
    h = glob.glob('%s*/model.hwx' % hx)
    if not h:
        return None, 'mil_to_hwx failed: %s' % ((r2.stderr or r2.stdout)[-160:].replace('\n', ' '))
    COMPILED_VIA.add(how)
    return h[0], None


def audit(hwx):
    txt = subprocess.run([HP, '-r', hwx], capture_output=True, text=True).stdout
    tasks = []
    for p in re.split(r'(?=InDim\s*:)', txt)[1:]:
        ot = re.search(r'OutTrans=(\d+)', p)
        if not ot:
            continue
        nb = len(re.findall(r'CoeffBase\[', p))
        cs = re.search(r'CoeffSize\[0\]:\s*(\S+)', p)
        ic = re.search(r'InDim\s*:\s*W=\d+ H=\d+ C=(\d+)', p)
        oc = re.search(r'OutDim\s*:\s*W=\d+ H=\d+ C=(\d+)', p)
        tasks.append({'OutTrans': ot.group(1), 'banks': nb,
                      'CoeffSize0': cs.group(1) if cs else None,
                      'Cin': int(ic.group(1)) if ic else None,
                      'Cout': int(oc.group(1)) if oc else None})
    return tasks


print('=' * 74)
print('ANE OutTrans probe -- looking for a WEIGHT-BEARING OutTrans=1 conv')
print('=' * 74)
print('%-22s %-7s %-11s %-14s' % ('graph', 'tasks', 'OutTrans=1', 'weight-bearing'))
found = []
report = {}
COMPILED_VIA = set()
n_ok = 0
for name, fn in GRAPHS:
    try:
        hwx, err = compile_graph(fn, name)
    except Exception as e:
        print('%-22s EXC %s' % (name, str(e)[:44])); continue
    if err:
        print('%-22s FAILED  %s' % (name, err[:44])); report[name] = {'error': err}; continue
    n_ok += 1
    tasks = audit(hwx)
    ot1 = [t for t in tasks if t['OutTrans'] == '1']
    wb = [t for t in ot1 if t['banks'] > 0]
    report[name] = {'n_tasks': len(tasks), 'n_outtrans1': len(ot1), 'weight_bearing': wb}
    print('%-22s %-7d %-11d %-14d %s' % (name, len(tasks), len(ot1), len(wb),
                                         '  <== !!!' if wb else ''))
    if wb:
        found.append((name, wb))

with open(os.path.join(RESULTS, 'outtrans_probe.json'), 'w') as f:
    json.dump(report, f, indent=2)

print()
print('compiled via: %s' % (', '.join(sorted(COMPILED_VIA)) or 'nothing'))
print()
if n_ok == 0:
    print('!' * 74)
    print('ENVIRONMENT FAILURE -- nothing was compiled, so NOTHING WAS TESTED.')
    print('!' * 74)
    print('  This is NOT a negative result and does NOT rule out this macOS build.')
    print('  Every graph failed before reaching the ANE compiler. Common cause: no full')
    print('  Xcode AND a coremltools too old for the fallback. Fix and re-run:')
    print('     ./venv/bin/pip install -U coremltools')
    print('  Then:  ./run_probe.sh')
    sys.exit(2)
if found:
    print('*' * 74)
    print('RESULT: *** WEIGHT-BEARING OutTrans=1 FOUND ***')
    print('*' * 74)
    for name, wb in found:
        print('  graph %s -> %d task(s):' % (name, len(wb)))
        for t in wb[:4]:
            print('     Cin=%s Cout=%s banks=%d CoeffSize0=%s'
                  % (t['Cin'], t['Cout'], t['banks'], t['CoeffSize0']))
    print()
    print('  This host CAN emit the configuration. Run the positional read next:')
    print('     ./venv/bin/python scripts/02_posread_L.py')
    sys.exit(0)
else:
    print('RESULT: NO weight-bearing OutTrans=1 on this host')
    print('  (%d of %d graphs compiled successfully, so this IS a real test)' % (n_ok, len(GRAPHS)))
    print('  This macOS build is ruled out. Please still send back results/ and the console output.')
    sys.exit(1)
