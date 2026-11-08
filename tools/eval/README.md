# Fidelity evaluation

Measures how closely the GGUF reproduces Apple's live model. Needs a Mac with Apple Intelligence (for the reference
answers) and a llama.cpp build with the patch (for `llama-server` and `llama-afm-logits`).

```bash
tools/eval/collect_oracle.sh oracle/                                   # Apple's greedy answers, 40 prompts (~2 min)
python3 tools/eval/teacher_forced.py afm-v11.2-ifp.gguf llama.cpp/build/bin oracle/
python3 tools/eval/free_running.py   afm-v11.2-ifp.gguf llama.cpp/build/bin oracle/
```

* **Teacher-forced agreement**: Apple's answer is fed to the GGUF token by token; at each position, is the GGUF's most
  likely next token the one Apple produced? (Expected with v1.1: 0.9610, 1083/1127, first answer token 37/40 on
  macOS 27.0 26A5378n, asset 8d6a1a11.) `--per-token` uses per-token expert routing instead of Apple's schedule.
* **Free-running**: the GGUF answers on its own; reports answers identical to Apple's character for character
  (expected with v1.1: 17/40).

Apple's answers change when Apple updates the model asset, so collect the reference on the same OS build as the GGUF.
