#!/usr/bin/env python3
"""Teacher-forced next-token agreement between the GGUF and Apple's live model.

For each prompt, Apple's own greedy answer (from collect_oracle.sh) is fed to the GGUF one token after another; at
every position we check whether the GGUF's most likely next token is the token Apple produced. The prompt is rendered
by the GGUF's chat template (Apple's afm_cider_v2) with the same instructions tools/afm.swift passes.

usage: teacher_forced.py <model.gguf> <llama.cpp build/bin> <oracle-dir> [--per-token] [--stepwise]
  default      Apple's expert schedule (experts picked at the last prompt token, re-picked at position % 32 == 0)
  --per-token  experts picked for every token from its own router logits (AFM_ROUTE_TOKEN=1)
  --stepwise   run the answer one token per decode call (the generation code path; slower)
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import numpy as np

INSTRUCTIONS = 'You are a helpful, concise assistant.'   # what tools/afm.swift passes by default

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; flags = {a for a in sys.argv[1:] if a.startswith('--')}
    if len(args) != 3: sys.exit(__doc__)
    gguf, bindir, odir = args
    here = os.path.dirname(os.path.abspath(__file__))
    prompts = open(os.environ.get('AFM_PROMPTS', os.path.join(here, 'prompts.txt')), encoding='utf-8').read().strip().split('\n')
    exe = lambda n: os.path.join(bindir, n + ('.exe' if os.name == 'nt' else ''))

    port = 18300 + os.getpid() % 100
    srv = subprocess.Popen([exe('llama-server'), '-m', gguf, '--port', str(port), '--jinja', '-c', '512', '-ngl', '0'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def post(path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req))
    try:
        for _ in range(300):
            try:
                if b'ok' in urllib.request.urlopen(f'http://127.0.0.1:{port}/health').read(): break
            except Exception: time.sleep(1)
        eot = post('/tokenize', {'content': '<end_of_turn>', 'parse_special': True})['tokens']
        assert len(eot) == 1, eot
        seqs = []
        for i, p in enumerate(prompts):
            text = post('/apply-template', {'messages': [{'role': 'system', 'content': INSTRUCTIONS},
                                                         {'role': 'user', 'content': p}]})['prompt']
            pre = post('/tokenize', {'content': text, 'add_special': True, 'parse_special': True})['tokens']
            ans = open(os.path.join(odir, f'out_{i}.txt'), encoding='utf-8').read().strip()
            tgt = post('/tokenize', {'content': ans, 'parse_special': False})['tokens'] + eot
            seqs.append((pre, tgt))
    finally:
        srv.terminate()

    td = tempfile.mkdtemp(dir=odir)
    env = dict(os.environ)
    if '--per-token' in flags: env['AFM_ROUTE_TOKEN'] = '1'
    hit = tot = first = 0
    for i, (pre, tgt) in enumerate(seqs):
        ids = np.array(pre + tgt[:-1], np.int32); ids.tofile(f'{td}/{i}.bin')
        e = dict(env)
        if '--per-token' in flags: e['AFM_INBIAS_FROM'] = str(len(pre))     # answer tokens are generated tokens
        else: e['AFM_STEPWISE' if '--stepwise' in flags else 'AFM_ROUTE_SCHED'] = str(len(pre))
        subprocess.run([exe('llama-afm-logits'), gguf, f'{td}/{i}.bin', f'{td}/{i}.f32'], check=True, env=e,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lg = np.fromfile(f'{td}/{i}.f32', np.float32).reshape(len(ids), -1)[len(pre) - 1:]
        ok = lg.argmax(-1) == np.array(tgt)
        hit += int(ok.sum()); tot += len(tgt); first += int(ok[0])
        print(f'[{i:3d}] {int(ok.sum()):3d}/{len(tgt):3d} {ok.mean():.3f}  {prompts[i][:60]}', flush=True)
    shutil.rmtree(td, ignore_errors=True)
    print(f'teacher-forced top-1 agreement {hit / tot:.4f} ({hit}/{tot}), first answer token {first}/{len(seqs)}')

if __name__ == '__main__':
    main()
