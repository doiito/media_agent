#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${WAN22_MODEL_DIR:-$ROOT_DIR/models/wan2.2-ti2v-5b}"
CONNECTIONS="${WAN22_DOWNLOAD_CONNECTIONS:-4}"

if ! [[ "$CONNECTIONS" =~ ^[1-9][0-9]*$ ]] || (( CONNECTIONS > 16 )); then
    echo "WAN22_DOWNLOAD_CONNECTIONS must be between 1 and 16" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for model download" >&2
    exit 1
fi

# Always use direct repository URLs. HF_ENDPOINT and Python tooling are not used.
unset HF_ENDPOINT
mkdir -p "$MODEL_DIR"

file_size() {
    if [[ -f "$1" ]]; then
        stat -c '%s' "$1"
    else
        printf '0\n'
    fi
}

download_asset() {
    local label="$1"
    local url="$2"
    local relative_path="$3"
    local expected_size="$4"
    local expected_sha256="$5"
    local output="$MODEL_DIR/$relative_path"
    local part_dir="$MODEL_DIR/.download/${relative_path//\//_}.parts"
    local seed="$part_dir/seed"
    local manifest="$part_dir/manifest"

    mkdir -p "$(dirname "$output")" "$part_dir"

    if [[ -f "$output" ]] \
        && [[ "$(file_size "$output")" == "$expected_size" ]] \
        && [[ "$(sha256sum "$output" | awk '{print $1}')" == "$expected_sha256" ]]; then
        echo "$label is already complete and verified: $output"
        return
    fi

    if [[ ! -f "$seed" ]]; then
        local hf_partial=""
        hf_partial="$(find "$MODEL_DIR/.cache/huggingface/download" -type f \
            -name "*.${expected_sha256}.incomplete" -size +0c 2>/dev/null \
            | sort | head -n 1 || true)"
        if [[ -n "$hf_partial" ]]; then
            cp --reflink=auto "$hf_partial" "$seed"
            echo "Reused existing partial prefix: $(file_size "$seed") bytes"
        fi
    fi

    local seed_size
    seed_size="$(file_size "$seed")"
    if (( seed_size > expected_size )); then
        echo "$label partial prefix is larger than the expected file" >&2
        exit 1
    fi

    local manifest_value="${expected_size}:${expected_sha256}:${seed_size}:${CONNECTIONS}"
    if [[ -f "$manifest" ]] && [[ "$(<"$manifest")" != "$manifest_value" ]]; then
        find "$part_dir" -maxdepth 1 -type f -name 'part-*' -delete
    fi
    printf '%s\n' "$manifest_value" > "$manifest"

    echo "Downloading $label"
    echo "  URL: $url"
    echo "  Existing prefix: $seed_size / $expected_size bytes"
    echo "  Parallel ranges: $CONNECTIONS"

    download_range() {
        local index="$1"
        local start="$2"
        local end="$3"
        local part="$part_dir/part-$(printf '%02d' "$index")"
        local expected=$((end - start + 1))
        local have
        have="$(file_size "$part")"

        if (( have > expected )); then
            echo "$label range part is larger than expected: $part" >&2
            return 1
        fi

        while (( have < expected )); do
            local from=$((start + have))
            echo "  $label part $index: bytes $from-$end"
            if ! curl --http1.1 --fail --location --silent --show-error \
                --connect-timeout 30 --speed-time 60 --speed-limit 1024 \
                --range "$from-$end" "$url" >> "$part"; then
                echo "  $label part $index: transfer interrupted, retrying in 3 seconds" >&2
                sleep 3
            fi
            have="$(file_size "$part")"
            if (( have > expected )); then
                echo "$label server ignored Range for part $index" >&2
                return 1
            fi
        done
    }

    local remaining=$((expected_size - seed_size))
    if (( remaining > 0 )); then
        local chunk_size=$(((remaining + CONNECTIONS - 1) / CONNECTIONS))
        local pids=()
        local index start end
        for ((index = 1; index <= CONNECTIONS; index++)); do
            start=$((seed_size + (index - 1) * chunk_size))
            (( start < expected_size )) || break
            end=$((start + chunk_size - 1))
            (( end < expected_size )) || end=$((expected_size - 1))
            download_range "$index" "$start" "$end" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    fi

    local assembled="$part_dir/assembled"
    : > "$assembled"
    if [[ -f "$seed" ]]; then
        cat "$seed" >> "$assembled"
    fi
    local index part
    for ((index = 1; index <= CONNECTIONS; index++)); do
        part="$part_dir/part-$(printf '%02d' "$index")"
        [[ -f "$part" ]] || continue
        cat "$part" >> "$assembled"
    done

    local actual_size actual_sha256
    actual_size="$(file_size "$assembled")"
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "$label size mismatch: expected $expected_size, got $actual_size" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$assembled" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "$label SHA-256 mismatch: expected $expected_sha256, got $actual_sha256" >&2
        exit 1
    fi

    mv "$assembled" "$output"
    find "$part_dir" -maxdepth 1 -type f -delete
    rmdir "$part_dir"
    echo "$label verified: $output"
}

download_asset \
    "Wan2.2 TI2V 5B Q4_K_M" \
    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf" \
    "Wan2.2-TI2V-5B-Q4_K_M.gguf" \
    3433116000 \
    "95b19697b7f98e65b0a543640e9ca7b4dfec32e2a6e3731e8e10708be52655e2"

download_asset \
    "UMT5 XXL Q5_K_M" \
    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf" \
    "umt5-xxl-encoder-Q5_K_M.gguf" \
    4145878880 \
    "eaea358bb438c5d211721a4feecc162000e3636e9cb96f51e216f1f44ebd12ce"

download_asset \
    "Wan2.2 VAE" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors" \
    "split_files/vae/wan2.2_vae.safetensors" \
    1409400960 \
    "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"

echo "Wan2.2 TI2V native assets are ready under: $MODEL_DIR"
