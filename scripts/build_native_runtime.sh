#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_CPP_DIR="${SD_CPP_SOURCE_DIR:-/dev-data/ai-test/stable-diffusion.cpp}"
BUILD_DIR="${SD_CPP_NATIVE_BUILD_DIR:-$ROOT_DIR/target/native/stable-diffusion-cuda}"
RUNTIME_DIR="${NATIVE_RUNTIME_DIR:-$ROOT_DIR/native/runtime}"
SVD_PATCHES=(
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-native.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-unet-kind.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-frame-timesteps.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-conv3d-layout.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-linear-projection.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-vae-scale.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-vae-roundtrip.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-vae-roundtrip-frames.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-edm-scheduler.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-diagnostics.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-cross-diagnostic.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-subblock-diagnostics.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-chunking.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-position-diagnostic.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-conditioning-latent.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-temporal-zero-position-diagnostic.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-frame-guidance.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-noise-augmentation.patch"
    "$ROOT_DIR/patches/stable-diffusion-cpp-svd-noise-augmentation-scale.patch"
)
SVD_PATCH_SENTINELS=(
    "prepared SVD image conditioning"
    "version == VERSION_SVD) {"
    "static_cast<size_t>(init_latent.shape()[3])"
    "explicit_channel_axis"
    "auto proj_in          = std::dynamic_pointer_cast<UnaryBlock>"
    "version == VERSION_SVD) {"
    "SD_API bool vae_roundtrip"
    "const int output_frames = std::max(1, frames);"
    "std::make_shared<EDMVDenoiser>(0.002f, 700.0f)"
    "MEDIA_SD_SVD_DISABLE_TEMPORAL_TRANSFORMER"
    "MEDIA_SD_SVD_DISABLE_TEMPORAL_CROSS_ATTENTION"
    "MEDIA_SD_SVD_DISABLE_TEMPORAL_SELF_ATTENTION"
    "MEDIA_SD_SVD_TEMPORAL_BATCH_CHUNK"
    "MEDIA_SD_SVD_DISABLE_TEMPORAL_POSITION_EMBEDDING"
    "encode_svd_conditioning_latent"
    "MEDIA_SD_SVD_ZERO_TEMPORAL_POSITION"
    "frame_guidance_min"
    "noise_aug_strength"
    "doubling this input-space noise"
)
SVD_PATCH_SENTINEL_FILES=(
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/model.h"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/core/ggml_extend.hpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/model/vae/auto_encoder_kl.hpp"
    "$SD_CPP_DIR/include/stable-diffusion.h"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/model/diffusion/unet.hpp"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
    "$SD_CPP_DIR/src/stable-diffusion.cpp"
)
JOBS="${NATIVE_BUILD_JOBS:-$(nproc)}"
BUILD_TYPE="${NATIVE_BUILD_TYPE:-Release}"
export CCACHE_DIR="${MEDIA_AGENT_CCACHE_DIR:-$ROOT_DIR/target/ccache}"
mkdir -p "$CCACHE_DIR"

if [[ ! -f "$SD_CPP_DIR/include/stable-diffusion.h" ]]; then
    echo "stable-diffusion.cpp source not found: $SD_CPP_DIR" >&2
    exit 1
fi

for index in "${!SVD_PATCHES[@]}"; do
    svd_patch="${SVD_PATCHES[$index]}"
    if git -C "$SD_CPP_DIR" apply --recount --check "$svd_patch" >/dev/null 2>&1; then
        git -C "$SD_CPP_DIR" apply --recount "$svd_patch"
    elif git -C "$SD_CPP_DIR" apply --recount --reverse --check "$svd_patch" >/dev/null 2>&1; then
        echo "Native SVD patch already applied: $(basename "$svd_patch")"
    elif grep -Fq "${SVD_PATCH_SENTINELS[$index]}" "${SVD_PATCH_SENTINEL_FILES[$index]}"; then
        # A later patch may modify this patch's hunk, making reverse-check fail.
        # The per-patch sentinel keeps incremental rebuilds idempotent.
        echo "Native SVD patch already applied (sentinel): $(basename "$svd_patch")"
    else
        echo "Native SVD patch does not apply cleanly: $svd_patch" >&2
        exit 1
    fi
done

cmake --fresh -S "$SD_CPP_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$RUNTIME_DIR" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}" \
    -DSD_CUDA=ON \
    -DSD_BUILD_SHARED_LIBS=ON \
    -DSD_BUILD_SHARED_GGML_LIB=OFF \
    -DSD_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

c++ -std=c++17 -O3 -fPIC -shared \
    "$ROOT_DIR/native/sd_bridge.cpp" \
    -I"$ROOT_DIR/native" \
    -I"$SD_CPP_DIR/include" \
    -L"$RUNTIME_DIR/lib" \
    -Wl,-rpath,'$ORIGIN' \
    -lstable-diffusion \
    -ldl \
    -o "$RUNTIME_DIR/lib/libmedia_sd_bridge.so"

cargo build --manifest-path "$ROOT_DIR/Cargo.toml" --release --bin media-sd-worker

echo "Native runtime built:"
echo "  worker: $ROOT_DIR/target/release/media-sd-worker"
echo "  bridge: $RUNTIME_DIR/lib/libmedia_sd_bridge.so"
echo "  stable-diffusion.cpp: $RUNTIME_DIR/lib/libstable-diffusion.so"
