#!/usr/bin/env bash
# Build the frozen aarch64 handoff. The original training build is historical.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${1:-aarch64}" != "aarch64" ]; then
    echo "This handoff supports the aarch64 package; other architectures require a separate port." >&2
    exit 1
fi
python3 "$ROOT/tools/build_release.py"
