#!/bin/bash
# Environment report. The macOS build is the variable that matters -- the ANE compiler
# ships with the OS, not with Xcode.
OUT="$(cd "$(dirname "$0")/.." && pwd)/results"; mkdir -p "$OUT"
{
echo "==== host ===="
sw_vers
echo "arch: $(uname -m)   (must be arm64 -- Intel Macs have no Neural Engine)"
echo
echo "==== ANECompiler (ships with macOS; THIS is the variable) ===="
F=/System/Library/PrivateFrameworks/ANECompiler.framework
if [ -d "$F" ]; then
  echo "present: $F"
  /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$F/Versions/A/Resources/Info.plist" 2>/dev/null | sed 's/^/  version: /'
  /usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$F/Versions/A/Resources/Info.plist" 2>/dev/null | sed 's/^/  build:   /'
  ls -l "$F/Versions/A/ANECompiler" 2>/dev/null | awk '{print "  size:    "$5}'
else
  echo "MISSING -- this host cannot run the probe"
fi
echo
echo "==== Xcode (front end only; does NOT change the ANE result) ===="
xcode-select -p 2>/dev/null
xcodebuild -version 2>/dev/null | head -2
xcrun -f coremlcompiler 2>/dev/null
echo
echo "==== python / coremltools ===="
V="$(cd "$(dirname "$0")/.." && pwd)/venv/bin/python"
[ -x "$V" ] || V=python3
"$V" -c 'import sys,coremltools,numpy;print("python",sys.version.split()[0]);print("coremltools",coremltools.__version__);print("numpy",numpy.__version__)' 2>&1 | grep -v Warning
} | tee "$OUT/env.txt"
