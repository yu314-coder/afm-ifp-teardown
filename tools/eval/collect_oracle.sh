#!/bin/sh
# Collect Apple's greedy answers (the fidelity reference) through the public FoundationModels API.
# usage: tools/eval/collect_oracle.sh <out-dir>     (macOS with Apple Intelligence; writes <out-dir>/out_<i>.txt)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); OUT=${1:?usage: collect_oracle.sh <out-dir>}
mkdir -p "$OUT"
[ -x "$OUT/afm" ] || swiftc -O -o "$OUT/afm" "$HERE/../afm.swift"
i=0
while IFS= read -r p; do
  "$OUT/afm" -t 0 "$p" > "$OUT/out_$i.txt"
  echo "[$i] $p"
  i=$((i+1))
done < "$HERE/prompts.txt"
