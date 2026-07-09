# ComfyUI Rust Agent

**Enhanced Intelligent Image/Video Generation Workflow System**

A production-grade ComfyUI workflow system implemented in Rust, integrated with stable-diffusion.cpp inference engine, supporting PDCA mode and JSON-LD DAG workflows.

## Features

### Core Capabilities

- **Image Generation**: Text-to-Image (T2I), Image-to-Image (I2I), ControlNet guidance
- **Video Generation**: SVD image-to-video, frame interpolation, AnimateDiff animation
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
- SVD video generation
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
- **sd-cli** (stable-diffusion.cpp CLI) — for image/video generation
- **llama.cpp** (optional) — for LLM text generation

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

# Build release
cargo build --release

# Run tests (215 tests)
cargo test --lib
```

### Initialize & Run

```bash
# Initialize system (download models, create directories)
cargo run --release --bin init

# Start server
cargo run --release --bin comfyui-server

# Or with config
cargo run --release --bin comfyui-server -- --config config/agent.yaml
```

### Video Generation Notes

SVD (Stable Video Diffusion) supports **image-to-video** only. For **text-to-video**, the system automatically uses a combination path:

```
text_to_video = text_to_image → image_to_video
```

1. First generates a first-frame image via text_to_image
2. Then animates it via image_to_video (SVD)

Recommended video parameters (SVD):
- **cfg**: 2.5 (not user-specified 7)
- **fps**: 5
- **frames**: 25 (5 seconds video)
- **motion_bucket_id**: 127

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

- [ ] WebUI frontend interface
- [ ] More video generation model support
- [ ] Distributed inference
- [ ] Automatic model download

## License

MIT License

## Acknowledgments

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - Inference engine
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM backend
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Workflow design reference