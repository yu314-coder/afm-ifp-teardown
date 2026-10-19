#!/bin/zsh
# find_asset.sh -- locate the Foundation Models asset already present on YOUR Apple device.
# Reads nothing but the filesystem; copies nothing. Research / interoperability use only.
set -u
BASE="/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto"
if [ ! -d "$BASE" ]; then
  echo "No Foundation Models asset found at:"
  echo "  $BASE"
  echo "Apple Intelligence must be enabled and the model downloaded on this Mac."
  exit 1
fi
echo "== ifp (sparse-expert) rasterized weights (feed decode_experts.py) =="
find "$BASE" -name 'ifp_rasterized_weights.bin' 2>/dev/null | while read f; do
  printf "  %s  (%s)\n" "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
done
echo "== backbone ANE program (binary_0.hwx) and odix graph =="
find "$BASE" -name 'binary_0.hwx' 2>/dev/null | head
find "$BASE" -maxdepth 4 -name '*.odix' 2>/dev/null | head
echo
echo "The 3B ifp model is the large package (its .odixpackage carries an ifp/ subdir and a"
echo "~5GB ifp_rasterized_weights.bin). Pass that .bin to decode_experts.py."
