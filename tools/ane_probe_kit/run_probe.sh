#!/bin/bash
# One-command runner. Prints the environment, runs the decisive test, and runs the
# positional read ONLY if that test passes.
cd "$(dirname "$0")" || exit 1
PY=./venv/bin/python
[ -x "$PY" ] || PY=python3
chmod +x bin/mil_to_hwx bin/hwx_parsing scripts/*.sh 2>/dev/null
xattr -dr com.apple.quarantine bin/ 2>/dev/null

./scripts/00_check_env.sh
echo
"$PY" scripts/01_outtrans_probe.py
rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "########## weight-bearing OutTrans=1 found -- running the positional read ##########"
  echo
  "$PY" scripts/02_posread_L.py
  echo
  echo "DONE. Send back: this console output, results/env.txt, results/outtrans_probe.json,"
  echo "      and results/posread_L_result.npz"
else
  echo "DONE. This macOS build is ruled out (still useful -- it eliminates a build)."
  echo "      Send back: this console output, results/env.txt, results/outtrans_probe.json"
fi
