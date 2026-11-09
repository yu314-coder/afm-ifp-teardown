#!/usr/bin/env python3
"""Run one of Apple's 300M task adapters (e.g. the safety guard) on text, the way Apple renders it.

The adapter GGUF (afm2gguf.py --pico-adapter NAME) carries Apple's instruction for the task in its metadata; this
script renders it as the system turn of the afm_cider_v2 template (system turn, empty developer turn, user turn)
and decodes greedily with llama.cpp.

usage: python3 afm300m_classify.py -m afm-300m-base.gguf -l afm-300m-safety.lora.gguf "text to classify" [...]
needs: gguf (pip install gguf) and llama-completion from a llama.cpp build with the afm patch (on PATH or --bin)
"""
import argparse, json, os, shutil, subprocess, sys

def instruction(lora, key=None):
    from gguf import GGUFReader
    f = GGUFReader(lora).fields.get('adapter.afm.systemInstructions')
    if f is None: sys.exit(f'error: {lora} has no adapter.afm.systemInstructions (is it a 300M task adapter?)')
    si = json.loads(bytes(f.parts[f.data[0]]).decode())
    return si.get(key or 'default') or next(iter(si.values()))

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-m', '--model', required=True); ap.add_argument('-l', '--lora', required=True)
    ap.add_argument('--key', help='which of the adapter\'s instructions to use (default: "default")')
    ap.add_argument('--bin', default=shutil.which('llama-completion') or 'llama-completion')
    ap.add_argument('-n', type=int, default=32); ap.add_argument('text', nargs='+')
    a = ap.parse_args(); ins = instruction(a.lora, a.key)
    for t in a.text:
        p = (f'<start_of_turn>system\n{ins}<end_of_turn>\n<start_of_turn>developer\n<end_of_turn>\n'
             f'<start_of_turn>user\n{t}<end_of_turn>\n<start_of_turn>model\n')
        out = subprocess.run([a.bin, '-m', a.model, '--lora', a.lora, '-p', p, '-n', str(a.n), '--temp', '0', '-no-cnv'],
                             capture_output=True, text=True).stdout
        ans = out.split('<start_of_turn>model\n')[-1].replace('[end of text]', '').strip()
        print(f'{t!r}\n  -> {ans}')

if __name__ == '__main__':
    main()
