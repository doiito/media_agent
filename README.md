# ComfyUI Rust Agent

**Enhanced Intelligent Image/Video Generation Workflow System**

A production-grade ComfyUI workflow system implemented in Rust, integrated with stable-diffusion.cpp inference engine, supporting PDCA mode and JSON-LD DAG workflows.

> The default runtime is now Gliding Horse + a Rust native worker +
> stable-diffusion.cpp. Inference requires no Python, PyTorch, or Diffusers.
> Agent LLM routing supports local llama.cpp and the DeepSeek OpenAI-compatible
> API. See [the native Agent plan](docs/native-agent-plan.md).

## Features

### Core Capabilities

- **Image Generation**: Text-to-Image (T2I), Image-to-Image (I2I), ControlNet guidance
- **Video Generation**: Wan2.2 prompt-directed text/image-to-video, SVD fallback, delivery scaling
- **Model Management**: 13 model types, auto-discovery indexing, dual-layer LRU cache
- **Multi-Backend Support**: stable-diffusion.cpp, llama.cpp, ONNX Runtime, local processor
- **Real-time Preview**: WebSocket push, sampling progress tracking, intermediate result caching

### Extended Node System (33+ Nodes)

| Category | Nodes |
|----------|-------|
| Model Loading | CheckpointLoader, UNETLoader, CLIPLoader, VAELoader, LoraLoader, ControlNetLoader, StyleModelLoader, UpscaleModelLoader |
| Samplers | KSampler, KSamplerAdvanced, SchedulerAdvanced, SamplerCustom, LatentNoiseInjection |
| Image Processing | ImageScale, ImageUpscale, ImageBlend, ImageCrop, ImageRotate, ImageFlip, ImageColorAdjust, ImageFilter, PreviewImage |
| Video Processing | VideoCombine, SVDImageToVideo, FrameInterpolation, AnimateDiffSampler, FrameSequenceGenerator, LatentInterpolation |

### Workflow Templates (29)

- Basic/Advanced text-to-image pipelines
- Image-to-image style transfer
- ControlNet edge/pose/depth control
- LoRA style/character fine-tuning
- Wan2.2 text/image-to-video and SVD fallback
- AnimateDiff animation
- Multi-stage composite workflows

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent API Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ HTTP Server │  │ WebSocket   │  │ Event Bus       │  │
│  │ (Axum)      │  │ Real-time   │  │ Publish/Subscribe│  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Workflow Engine                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ PDCA Mode   │  │ JSON-LD DAG │  │ Node Executor   │  │
│  │ (Default)   │  │ (Complex)   │  │ (Parallel)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Backend Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ SdCpp       │  │ LlamaCpp    │  │ LocalProcessor  │  │
│  │ (GPU)       │  │ (LLM)       │  │ (VAE)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐│
│  │              BackendPool (Load Balance/Failover)    ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                   Model Manager                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Scanner     │  │ Cache       │  │ Index           │  │
│  │ (Discovery) │  │ (VRAM+RAM)  │  │ (DashMap)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Model Support

### Model Types (13)

- Checkpoint (Main model)
- UNET (Diffusion model)
- VAE (Variational Autoencoder)
- CLIP (Text encoder)
- Lora (Fine-tuning model)
- ControlNet (Control network)
- Embeddings (Text embeddings)
- StyleModel (Style model)
- UpscaleModel (Upscaling model)
- GLIGEN (Position control)
- Hypernetwork (Hypernetwork)
- IPAdapter (Image adapter)
- T2IAdapter (Text-to-image adapter)

### Model Architectures (13)

- SD1.5 / SD2.1 / SDXL / SDXLRefiner
- SD3 / SD3.5
- Flux / FluxFill / FluxControl
- SVD / SVDXT (Video)
- CogVideo / CogVideoX

## Project Structure

```
media_agent/
├── deps/
│   └── gliding_horse/      # gliding_horse Agent OS dependency (Git Submodule)
├── src/
│   ├── lib.rs              # Library entry
│   ├── types.rs            # Core type definitions
│   ├── config/             # Configuration management
│   ├── model_manager/      # Enhanced model management
│   │   ├── model_info.rs   # Model type/architecture definitions
│   │   ├── scanner.rs      # Auto-discovery indexing
│   │   ├── cache.rs        # Dual-layer LRU cache
│   │   └── manager.rs      # Manager core
│   ├── backend/            # Inference backends
│   │   ├── sd_cpp.rs       # stable-diffusion.cpp
│   │   ├── llama_cpp.rs    # llama.cpp
│   │   ├── multi_backend.rs # Multi-backend pool
│   │   └── router.rs       # Backend router
│   ├── node/               # Node system
│   │   ├── core_nodes.rs   # Core nodes
│   │   ├── extended_nodes.rs # Extended nodes
│   │   ├── advanced_sampler.rs # Advanced samplers
│   │   ├── image_processing.rs # Image processing
│   │   └── video_nodes.rs  # Video nodes
│   ├── preview/            # Real-time preview
│   ├── workflow/           # Workflow engine
│   ├── execution/          # Execution engine
│   ├── agent/              # Agent module
│   ├── api/                # HTTP/WebSocket API
│   ├── monitor/            # System monitoring
│   └── storage/            # Storage management
├── workflows/              # JSON-LD workflow templates (29)
├── skills/                 # Agent skill definitions
├── config/                 # Configuration files
├── docs/                   # Documentation
└── tests/                  # Tests
```

## Installation & Build

### Prerequisites

- **Rust** (1.75+ with C++20 compatible compiler - Clang recommended for gliding_horse native deps)
- **protoc** (Protobuf compiler) — required by gliding_horse Agent OS for gRPC proto compilation
  - Linux: `sudo apt install protobuf-compiler`
  - macOS: `brew install protobuf`
  - Windows: download from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases)
- **stable-diffusion.cpp**: the C ABI and Rust worker are built from `/dev-data/ai-test/stable-diffusion.cpp`
- **llama.cpp or DeepSeek API**: the default local source is `/dev-data/ai-test/llama.cpp-b9810`
- **ffmpeg/ffprobe**: video encoding, delivery scaling, and artifact validation

### Clone

This project depends on [gliding_horse Agent OS](https://github.com/doiito/gliding_horse.git), managed as a Git Submodule. Make sure to pull submodules when cloning:

```bash
# Option 1: Clone with submodules automatically
git clone --recurse-submodules https://github.com/doiito/media_agent.git

# Option 2: Clone then initialize submodules
git clone https://github.com/doiito/media_agent.git
cd media_agent
git submodule update --init --recursive
```

### Build

```bash
# Set compiler environment (Clang for gliding_horse native deps)
export CC=clang
export CXX=clang++
export CCACHE_DISABLE=1

# Build CUDA stable-diffusion.cpp, the C bridge, and Rust worker
scripts/build_native_runtime.sh

# Build the server and evaluator
cargo build --release --bin comfyui-server --bin media-agent-eval

# Run tests (215 tests)
cargo test --lib
```

### Initialize & Run

```bash
# Start GPU llama-server and the media server
scripts/start_all.sh
```

Local llama.cpp uses `--n-gpu-layers auto --fit on`. It sleeps after one idle
second to release VRAM for stable-diffusion.cpp, then wakes for the next Agent phase.

### Video Generation Notes

The default prompt-conditioned video engine is Wan2.2 TI2V 5B. Install its
Q4_K_M diffusion model, Q5_K_M UMT5 encoder, and VAE with:

```bash
scripts/download_wan22_ti2v.sh
```

The script uses official `huggingface.co` URLs with parallel `curl` Range
downloads and SHA-256 verification; it does not require Python. Wan handles both
text-to-video and prompt-directed image-to-video, so actions such as dancing
remain part of the model conditioning rather than being discarded.

Wan generation uses an aspect-matched native canvas with approximately the
same area as 832x480, then the Rust worker scales and pads to each request's
delivery dimensions. GGUF tensor types are preserved and `max_vram=-1` enables
stable-diffusion.cpp graph segmentation on 16 GB GPUs.

SVD remains available as a faster image-conditioned fallback. Download it with
`scripts/download_svd_official.sh`; it cannot guarantee exact text-directed
actions. Text-to-video with an explicitly selected SVD model uses the native
composition path:

```text
text_to_video = native text_to_image -> native SVD image_to_video
```

Recommended video parameters (SVD):
- **per-frame CFG**: defaults to a linear `min_cfg=1.0` to final-frame `cfg=3.0` ramp; both endpoints remain request-selectable
- **conditioning variation**: `noise_aug_strength=0.02`, passed dynamically to both the conditioning image and added time IDs
- **fps**: 5
- **frames**: 25 (5 seconds video)
- **generation size**: 1024×576; each Web request controls the scaled/padded delivery size
- **motion_bucket_id**: dynamic per request (native range 0-1023, Web slider 0-255, default 127)
- **memory**: Wan video parameters use the upstream-recommended `*=cpu` residency policy while compute remains on CUDA; videos longer than two frames also use overlapping native VAE tiles so 25-frame generation fits a 16 GB GPU
- **timeouts**: image inference defaults to 300 seconds; video has an independent 1800-second limit

The Web UI provides Fast, Balanced, and High quality profiles while allowing
users to override the prompt, model, dimensions, steps, CFG, seed, frames, FPS,
and motion controls per request. The UI never hard-codes generated content. CA
rejects near-blank images and effectively static videos while reporting signal
range, decoded frame count, duration, and temporal-change metrics.

At the Rust tool boundary, each natural-language request is dynamically compiled
through the configured llama.cpp or DeepSeek provider into a validated English
model prompt. Invalid drafts never reach the native worker. Successful
`/agent/chat` responses expose the actual pipeline, prompt, negative prompt,
quality, seed, effective parameters, and output path in `generation_audit`; this adds no Python dependency and does not
hard-code generation content.

Image-to-image requests preserve the source aspect ratio unless the user
explicitly overrides both dimensions. Identity/composition preservation uses a
conservative default transform strength, while the Web UI slider remains an
explicit per-request override. High-quality text-to-video first creates a
native semantic keyframe and then runs Wan I2V; Fast and Balanced retain direct
Wan T2V for lower latency.

## Test Coverage

| Module | Tests |
|--------|-------|
| Model Management | 56 |
| Multi-Backend | 17 |
| Real-time Preview | 18 |
| Node System | 60+ |
| Workflow Engine | 15+ |
| Monitor System | 10+ |
| Backend Router | 5+ |
| Conditioning System | 5+ |
| **Total** | **215** |

## API Reference

### HTTP API

```
POST /api/prompt         # Submit workflow
GET  /api/queue          # Query queue
POST /api/interrupt      # Interrupt execution
GET  /api/models         # Model list
GET  /api/model/{id}     # Model details
POST /api/upload/image   # Upload image
GET  /api/view/{id}      # View result
```

### WebSocket

```
ws://host/ws?client_id=<id>

Event Types:
- ExecutionStart    # Execution started
- Progress          # Sampling progress
- Preview           # Preview frame push
- Executing         # Node execution status
- ExecutionSuccess  # Execution succeeded
- ExecutionError    # Execution failed
```

## Configuration Example

```yaml
# config/agent.yaml
paths:
  models_dir: "./models"
  output_dir: "./output"
  temp_dir: "./temp"

backend:
  sd_cpp:
    executable: "sd-cli"
    backend: "cuda"
    precision: "f16"
    flash_attention: true
  llama_cpp:
    executable: "llama-cli"
    n_ctx: 512
    n_gpu_layers: -1

preview:
  enabled: true
  step_interval: 5
  max_width: 512
  jpeg_quality: 85

cache:
  vram_capacity: 4GB
  ram_capacity: 8GB
```

## Roadmap

- [x] Dynamic Web UI for image and video generation
- [x] Native Wan2.2 and SVD video inference
- [ ] Distributed inference
- [x] Native CLI model download scripts

## License

MIT License

## Acknowledgments

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - Inference engine
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM backend
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Workflow design reference
