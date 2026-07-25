#!/bin/bash
# Assemble the Drive-ready disclosure folder. Original work only -- no Apple content.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/Volumes/D/fix/apple_disclosure"
rm -rf "$OUT"; mkdir -p "$OUT/source"

cp "$SRC/paper/afm_teardown.pdf" "$OUT/afm_teardown.pdf"
cp "$SRC"/*.md "$OUT/source/" 2>/dev/null || true
cp "$SRC"/*.json "$OUT/source/" 2>/dev/null || true
cp -R "$SRC/src" "$OUT/source/src"
cp -R "$SRC/tools" "$OUT/source/tools" 2>/dev/null || true
rm -f "$OUT/source/LETTER_TO_APPLE.md"

# hard guard: refuse to ship anything Apple-proprietary
BAD=$(find "$OUT" -type f \( -name "*.gguf" -o -name "*.pt" -o -name "*.npz" -o -name "*.npy" \
      -o -name "*.dmg" -o -name "*.hwx" -o -name "*.odix" -o -name "*.asset" -o -name "*.bin" \
      -o -name "tok_vocab.json" \) | head)
if [ -n "$BAD" ]; then
  echo "REFUSING: Apple-proprietary files staged:"; echo "$BAD"; rm -rf "$OUT"; exit 1
fi
echo "Drive folder ready: $OUT"
du -sh "$OUT"
find "$OUT" -type f | wc -l | xargs echo "files:"
