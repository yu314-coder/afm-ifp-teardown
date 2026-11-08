#!/usr/bin/env python3
"""Free-running comparison: the GGUF answers each prompt greedily on its own and the text is compared with Apple's.

usage: [AFM_ROUTE_TOKEN=1 ...] free_running.py <model.gguf> <llama.cpp build/bin> <oracle-dir>
Environment variables are passed through to llama-server (e.g. AFM_ROUTE_TOKEN=1 for per-token expert routing).
Reports answers identical to Apple's and the mean fraction of Apple's tokens reproduced before the first difference.
"""
import json, os, subprocess, sys, time, urllib.request

INSTRUCTIONS = 'You are a helpful, concise assistant.'

def main():
    if len(sys.argv) != 4: sys.exit(__doc__)
    gguf, bindir, odir = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    prompts = open(os.environ.get('AFM_PROMPTS', os.path.join(here, 'prompts.txt')), encoding='utf-8').read().strip().split('\n')
    exe = os.path.join(bindir, 'llama-server' + ('.exe' if os.name == 'nt' else ''))
    port = 18400 + os.getpid() % 100
    srv = subprocess.Popen([exe, '-m', gguf, '--port', str(port), '-c', '4096', '-np', '1', '--jinja'],
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
        exact = 0; frac = []
        for i, p in enumerate(prompts):
            r = post('/v1/chat/completions', {'messages': [{'role': 'system', 'content': INSTRUCTIONS},
                                                           {'role': 'user', 'content': p}], 'temperature': 0, 'max_tokens': 300})
            ours = r['choices'][0]['message']['content'].strip()
            apple = open(os.path.join(odir, f'out_{i}.txt'), encoding='utf-8').read().strip()
            ta = post('/tokenize', {'content': apple})['tokens']; to = post('/tokenize', {'content': ours})['tokens']
            k = 0
            while k < min(len(ta), len(to)) and ta[k] == to[k]: k += 1
            exact += ours == apple; frac.append(k / max(1, len(ta)))
            print(f'[{i:2d}] {"=" if ours == apple else " "} {frac[-1]:.2f}  {p[:60]}', flush=True)
        print(f'identical answers {exact}/{len(prompts)}, mean matched prefix {sum(frac) / len(frac):.3f}')
    finally:
        srv.terminate()

if __name__ == '__main__':
    main()
