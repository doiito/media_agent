#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${SVD_MODEL_DIR:-$ROOT_DIR/models/diffusers/stable-video-diffusion-img2vid-xt}"
OUTPUT="$MODEL_DIR/svd_xt.safetensors"
PART_DIR="$MODEL_DIR/.svd_xt.parts"
MANIFEST="$PART_DIR/seed-size"
URL="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors"
EXPECTED_SIZE=9559625980
EXPECTED_SHA256="b2652c23d64a1da5f14d55011b9b6dce55f2e72e395719f1cd1f8a079b00a451"
CONNECTIONS="${SVD_DOWNLOAD_CONNECTIONS:-4}"

if ! [[ "$CONNECTIONS" =~ ^[1-9][0-9]*$ ]] || (( CONNECTIONS > 16 )); then
    echo "SVD_DOWNLOAD_CONNECTIONS must be between 1 and 16" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR" "$PART_DIR"

file_size() {
    if [[ -f "$1" ]]; then
        stat -c '%s' "$1"
    else
        printf '0\n'
    fi
}

verify_complete_model() {
    [[ -f "$OUTPUT" ]] || return 1
    [[ "$(file_size "$OUTPUT")" == "$EXPECTED_SIZE" ]] || return 1
    [[ "$(sha256sum "$OUTPUT" | awk '{print $1}')" == "$EXPECTED_SHA256" ]]
}

if verify_complete_model; then
    echo "SVD model is already complete and verified: $OUTPUT"
    exit 0
fi

SEED_FILE=""
while IFS= read -r candidate; do
    if [[ -f "$candidate" ]] && (( $(file_size "$candidate") > 0 )); then
        SEED_FILE="$candidate"
        break
    fi
done < <(find "$MODEL_DIR/.cache/huggingface/download" -maxdepth 1 -type f \
    -name "*.${EXPECTED_SHA256}.incomplete" 2>/dev/null | sort)

SEED_SIZE=0
if [[ -n "$SEED_FILE" ]]; then
    SEED_SIZE="$(file_size "$SEED_FILE")"
fi
if (( SEED_SIZE > EXPECTED_SIZE )); then
    echo "Existing partial file is larger than the official model: $SEED_FILE" >&2
    exit 1
fi

if [[ -f "$MANIFEST" ]]; then
    RECORDED_SEED_SIZE="$(<"$MANIFEST")"
    if [[ "$RECORDED_SEED_SIZE" != "$SEED_SIZE" ]]; then
        echo "Partial seed size changed; stop other downloaders before resuming" >&2
        exit 1
    fi
else
    printf '%s\n' "$SEED_SIZE" > "$MANIFEST"
fi

echo "Downloading official SVD XT checkpoint"
echo "  URL: $URL"
echo "  Existing prefix: $SEED_SIZE / $EXPECTED_SIZE bytes"
echo "  Parallel ranges: $CONNECTIONS"

download_range() {
    local index="$1"
    local start="$2"
    local end="$3"
    local part="$PART_DIR/part-$(printf '%02d' "$index")"
    local expected=$((end - start + 1))
    local have

    have="$(file_size "$part")"
    if (( have > expected )); then
        echo "Range part is larger than expected: $part" >&2
        return 1
    fi

    while (( have < expected )); do
        local from=$((start + have))
        echo "  part $index: bytes $from-$end"
        if ! curl --http1.1 --fail --location --silent --show-error \
            --range "$from-$end" "$URL" >> "$part"; then
            echo "  part $index: transfer interrupted, retrying in 3 seconds" >&2
            sleep 3
        fi
        have="$(file_size "$part")"
        if (( have > expected )); then
            echo "Server ignored Range for part $index" >&2
            return 1
        fi
    done
    echo "  part $index: complete ($have bytes)"
}

REMAINING=$((EXPECTED_SIZE - SEED_SIZE))
if (( REMAINING > 0 )); then
    CHUNK_SIZE=$(((REMAINING + CONNECTIONS - 1) / CONNECTIONS))
    pids=()
    for ((index = 1; index <= CONNECTIONS; index++)); do
        start=$((SEED_SIZE + (index - 1) * CHUNK_SIZE))
        (( start < EXPECTED_SIZE )) || break
        end=$((start + CHUNK_SIZE - 1))
        (( end < EXPECTED_SIZE )) || end=$((EXPECTED_SIZE - 1))
        download_range "$index" "$start" "$end" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

ASSEMBLED="$PART_DIR/svd_xt.safetensors.assembled"
: > "$ASSEMBLED"
if [[ -n "$SEED_FILE" ]]; then
    cat "$SEED_FILE" >> "$ASSEMBLED"
fi
for ((index = 1; index <= CONNECTIONS; index++)); do
    part="$PART_DIR/part-$(printf '%02d' "$index")"
    [[ -f "$part" ]] || continue
    cat "$part" >> "$ASSEMBLED"
done

ACTUAL_SIZE="$(file_size "$ASSEMBLED")"
if [[ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
    echo "Assembled size mismatch: expected $EXPECTED_SIZE, got $ACTUAL_SIZE" >&2
    exit 1
fi

ACTUAL_SHA256="$(sha256sum "$ASSEMBLED" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "SHA-256 mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
    exit 1
fi

mv "$ASSEMBLED" "$OUTPUT"
echo "SVD model verified: $OUTPUT"
