#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if rg -n \
    --glob '!validate_no_python_runtime.sh' \
    'Command::new\("python(3)?"\)|StableVideoDiffusionPipeline|from diffusers|import torch' \
    src scripts; then
    echo "Python inference dependency detected" >&2
    exit 1
fi

if find scripts -maxdepth 1 -type f -name '*.py' -print -quit | grep -q .; then
    echo "Python runtime script detected under scripts/" >&2
    exit 1
fi

for binary in target/release/media-sd-worker native/runtime/lib/libmedia_sd_bridge.so; do
    if [[ ! -f "$binary" ]]; then
        echo "Missing native runtime artifact: $binary" >&2
        exit 1
    fi
    if ldd "$binary" | grep -qi 'libpython'; then
        echo "Python shared library linked by $binary" >&2
        exit 1
    fi
done

echo "Zero-Python runtime validation passed"
