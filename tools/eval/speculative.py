#!/usr/bin/env python3
"""Speculative decoding with Apple's draft model: is it lossless, and how often are draft tokens accepted?

Runs every prompt twice through llama-server with greedy sampling: once plainly, once with the draft model
(Apple's settings: 7 draft tokens, no confidence cutoff). Reports identical answers, speed and acceptance.

usage: speculative.py <model.gguf> <draft.gguf> <llama.cpp build/bin> [extra llama-server args...]
  e.g. speculative.py afm-v11.2-ifp.gguf afm-v11.2-draft.gguf ~/llama.cpp/build/bin -ngl 0 -devd none   (CPU only)
Environment: NPROMPTS=<n> limits the number of prompts; AFM_PROMPTS=<file> picks another prompt file.
"""
import json, os, subprocess, sys, time, urllib.request

INSTRUCTIONS = 'You are a helpful, concise assistant.'   # what tools/afm.swift passes by default

def run(exe, model, prompts, extra, tag):
    port = 18600 + (os.getpid() + len(tag)) % 100
    srv = subprocess.Popen([exe, '-m', model, '--port', str(port), '-c', '4096', '-np', '1', '--jinja'] + extra,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def post(path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req))
    out = []
    try:
        for _ in range(300):
            try:
                if b'ok' in urllib.request.urlopen(f'http://127.0.0.1:{port}/health').read(): break
            except Exception: time.sleep(1)
        for p in prompts:
            r = post('/v1/chat/completions', {'messages': [{'role': 'system', 'content': INSTRUCTIONS},
                                                           {'role': 'user', 'content': p}], 'temperature': 0, 'max_tokens': 300})
            out.append((r['choices'][0]['message']['content'], r.get('timings', {})))
            print(f'  [{tag}] {len(out):3d}/{len(prompts)}', end='\r', flush=True)
    finally:
        srv.terminate(); srv.wait()
    print()
    return out

def main():
    if len(sys.argv) < 4: sys.exit(__doc__)
    model, draft, bindir = sys.argv[1:4]; extra = sys.argv[4:]
    here = os.path.dirname(os.path.abspath(__file__))
    prompts = open(os.environ.get('AFM_PROMPTS', os.path.join(here, 'prompts.txt')), encoding='utf-8').read().strip().split('\n')
    prompts = prompts[:int(os.environ.get('NPROMPTS', len(prompts)))]
    exe = os.path.join(bindir, 'llama-server' + ('.exe' if os.name == 'nt' else ''))
    plain = run(exe, model, prompts, extra, 'plain')
    spec = run(exe, model, prompts, extra + ['-md', draft, '--draft-max', '7', '--draft-min', '0', '--draft-p-min', '0'], 'speculative')
    rate = lambda res: sum(t.get('predicted_n', 0) for _, t in res) / max(1e-9, sum(t.get('predicted_ms', 0) for _, t in res) / 1000)
    drafted = sum(t.get('draft_n') or 0 for _, t in spec); accepted = sum(t.get('draft_n_accepted') or 0 for _, t in spec)
    same = sum(a == b for (a, _), (b, _) in zip(plain, spec))
    print(f'identical answers {same}/{len(prompts)}; plain {rate(plain):.2f} tok/s, speculative {rate(spec):.2f} tok/s; '
          f'draft tokens accepted {accepted}/{drafted} ({accepted / max(1, drafted):.3f})')

if __name__ == '__main__':
    main()
