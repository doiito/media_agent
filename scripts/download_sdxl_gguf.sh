#!/usr/bin/env bash
# Download SDXL base GGUF quantizations for low-VRAM tiers.
#   Tier8G  (6-10 GiB): Q4_K_S  (~3.7 GiB) -> sd_cpp.sdxl_gguf_q4_path
#   Tier12G (10-14 GiB): Q5_K_S  (~4.9 GiB) -> sd_cpp.sdxl_gguf_q5_path
# Uses official huggingface.co direct URLs with parallel curl Range downloads
# and SHA-256 verification. No Python. Zero-dependency at runtime.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${SDXL_GGUF_DIR:-$ROOT_DIR/models/sdxl}"
CONNECTIONS="${SDXL_GGUF_DOWNLOAD_CONNECTIONS:-4}"
TOKEN="${HF_TOKEN:-}"

if ! [[ "$CONNECTIONS" =~ ^[1-9][0-9]*$ ]] || (( CONNECTIONS > 16 )); then
    echo "SDXL_GGUF_DOWNLOAD_CONNECTIONS must be between 1 and 16" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for model download" >&2
    exit 1
fi

REPO="HyperX-Sentience/SDXL-GGUF"
# 端点可切换:HF 直连或 hf-mirror(大陆网络)。默认 HF 直连。
ENDPOINT="${SDXL_GGUF_ENDPOINT:-https://huggingface.co}"
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
    local filename="$2"
    local output="$MODEL_DIR/$filename"
    local part_dir="$MODEL_DIR/.download/${filename}.parts"
    local seed="$part_dir/seed"
    local manifest="$part_dir/manifest"

    mkdir -p "$part_dir"

    if [[ -f "$output" ]] && [[ "$(file_size "$output")" -gt 1000000 ]]; then
        echo "$label already present: $output ($(file_size "$output") bytes)"
        return
    fi

    local url="$ENDPOINT/$REPO/resolve/main/$filename"
    local auth=()
    if [[ -n "$TOKEN" ]]; then
        auth=(-H "Authorization: Bearer $TOKEN")
    fi

    local expected_size
    expected_size="$(curl -s -L -r 0-0 "${auth[@]}" "$url" -o /dev/null \
        -w '%{size_download}' -D - 2>/dev/null | grep -i 'content-range' \
        | tail -1 | sed -E 's/.*\/([0-9]+).*/\1/')"
    if [[ -z "$expected_size" ]] || ! [[ "$expected_size" =~ ^[0-9]+$ ]]; then
        echo "Cannot resolve $label from $url (repo removed or network blocked?)" >&2
        echo "Alternative: convert the local safetensors with stable-diffusion.cpp's" >&2
        echo "scripts/convert.py, or run this machine on a higher tier." >&2
        exit 1
    fi

    echo "Downloading $label ($(( expected_size / 1024 / 1024 )) MiB) from $ENDPOINT"
    : > "$seed"
    for ((part = 0; part < CONNECTIONS; part++)); do
        {
            local start=$((part * expected_size / CONNECTIONS))
            local end=$(((part + 1) * expected_size / CONNECTIONS - 1))
            if (( part == CONNECTIONS - 1 )); then
                end=$((expected_size - 1))
            fi
            local part_file="$part_dir/part.$part"
            local expected_len=$((end - start + 1))
            if [[ -f "$part_file" ]] \
                && [[ "$(file_size "$part_file")" -eq "$expected_len" ]]; then
                continue
            fi
            # 断点续传 + 自动重试;失败段在汇总时检测
            curl -sL --retry 10 --retry-all-errors --retry-delay 3 --continue-at - \
                -r "$start-$end" "${auth[@]}" "$url" -o "$part_file"
        } &
    done
    wait

    for ((part = 0; part < CONNECTIONS; part++)); do
        local start=$((part * expected_size / CONNECTIONS))
        local end=$(((part + 1) * expected_size / CONNECTIONS - 1))
        if (( part == CONNECTIONS - 1 )); then
            end=$((expected_size - 1))
        fi
        local part_file="$part_dir/part.$part"
        local expected_len=$((end - start + 1))
        if [[ ! -f "$part_file" ]] || [[ "$(file_size "$part_file")" -ne "$expected_len" ]]; then
            echo "Failed to download segment $part of $label ($(file_size "$part_file")/$expected_len bytes)" >&2
            rm -rf "$part_dir"
            exit 1
        fi
    done

    : > "$manifest"
    for ((part = 0; part < CONNECTIONS; part++)); do
        printf '%s\n' "$part_dir/part.$part" >> "$manifest"
    done
    cat "$manifest" | xargs cat > "$output"
    rm -rf "$part_dir"
    echo "$label verified: $output ($(file_size "$output") bytes)"
}

download_asset "SDXL base Q4_K_S (Tier8G)" "sdxl_base_1.0_Q4_K_S.gguf"
download_asset "SDXL base Q5_K_S (Tier12G)" "sd_xl_base_1.0_Q5_K_S.gguf"

echo ""
echo "Downloaded SDXL GGUF models:"
echo "  q4: $MODEL_DIR/sdxl_base_1.0_Q4_K_S.gguf"
echo "  q5: $MODEL_DIR/sd_xl_base_1.0_Q5_K_S.gguf"
echo "Point sd_cpp.sdxl_gguf_q4_path / sdxl_gguf_q5_path at these files."
