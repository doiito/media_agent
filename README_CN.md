# ComfyUI Rust Agent

**增强智能图片/视频生成工作流系统**

基于 Rust 语言实现的生产级别 ComfyUI 工作流系统，集成 stable-diffusion.cpp 推理引擎，支持 PDCA 模式和 JSON-LD DAG 工作流。

> 当前默认架构为 Gliding Horse Agent + Rust 原生 Worker + stable-diffusion.cpp。
> 推理运行时不依赖 Python、PyTorch 或 Diffusers；本地 llama.cpp 与 DeepSeek
> OpenAI 兼容 API 均可作为 Agent LLM。实施和部署说明见
> [Rust 原生媒体 Agent 方案](docs/native-agent-plan.md)。

## 功能特性

### 核心功能

- **图片生成**: 文生图(T2I)、图生图(I2I)、ControlNet 控制
- **视频生成**: Wan2.2 提示词控制的文/图生视频、SVD 回退、交付尺寸缩放
- **模型管理**: 13种模型类型支持、自动发现索引、双层LRU缓存
- **多后端支持**: stable-diffusion.cpp、llama.cpp、ONNX Runtime、本地处理器
- **实时预览**: WebSocket 推送、采样进度追踪、中间结果缓存

### 扩展节点系统 (33+节点)

| 类别 | 节点 |
|------|------|
| 模型加载 | CheckpointLoader, UNETLoader, CLIPLoader, VAELoader, LoraLoader, ControlNetLoader, StyleModelLoader, UpscaleModelLoader |
| 采样器 | KSampler, KSamplerAdvanced, SchedulerAdvanced, SamplerCustom, LatentNoiseInjection |
| 图片处理 | ImageScale, ImageUpscale, ImageBlend, ImageCrop, ImageRotate, ImageFlip, ImageColorAdjust, ImageFilter, PreviewImage |
| 视频处理 | VideoCombine, SVDImageToVideo, FrameInterpolation, AnimateDiffSampler, FrameSequenceGenerator, LatentInterpolation |

### 工作流模板 (29个)

- 文生图基础/高级流程
- 图生图风格迁移
- ControlNet 边缘/姿态/深度控制
- LoRA 风格/角色微调
- Wan2.2 文/图生视频及 SVD 回退
- AnimateDiff 动画制作
- 多阶段组合流程

## 技术架构

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
│  │ (默认)      │  │ (复杂任务)  │  │ (并行执行)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Backend Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ SdCpp       │  │ LlamaCpp    │  │ LocalProcessor  │  │
│  │ (GPU推理)   │  │ (LLM编码)   │  │ (VAE编解码)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐│
│  │              BackendPool (负载均衡/故障转移)         ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                   Model Manager                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Scanner     │  │ Cache       │  │ Index           │  │
│  │ (自动发现)  │  │ (VRAM+RAM)  │  │ (DashMap)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 模型支持

### 模型类型 (13种)

- Checkpoint (主模型)
- UNET (扩散模型)
- VAE (变分自编码器)
- CLIP (文本编码器)
- Lora (微调模型)
- ControlNet (控制网络)
- Embeddings (文本嵌入)
- StyleModel (风格模型)
- UpscaleModel (超分模型)
- GLIGEN (位置控制)
- Hypernetwork (超网络)
- IPAdapter (图像适配器)
- T2IAdapter (文图适配器)

### 模型架构 (13种)

- SD1.5 / SD2.1 / SDXL / SDXLRefiner
- SD3 / SD3.5
- Flux / FluxFill / FluxControl
- SVD / SVDXT (视频)
- CogVideo / CogVideoX

## 项目结构

```
media_agent/
├── deps/
│   └── gliding_horse/      # gliding_horse Agent OS 依赖 (Git Submodule)
├── src/
│   ├── lib.rs              # 库入口
│   ├── types.rs            # 核心类型定义
│   ├── config/             # 配置管理
│   ├── model_manager/      # 模型管理增强
│   │   ├── model_info.rs   # 模型类型/架构定义
│   │   ├── scanner.rs      # 自动发现索引
│   │   ├── cache.rs        # 双层LRU缓存
│   │   └── manager.rs      # 管理器核心
│   ├── backend/            # 推理后端
│   │   ├── sd_cpp.rs       # stable-diffusion.cpp
│   │   ├── llama_cpp.rs    # llama.cpp
│   │   ├── multi_backend.rs # 多后端池
│   │   └── router.rs       # 后端路由
│   ├── node/               # 节点系统
│   │   ├── core_nodes.rs   # 核心节点
│   │   ├── extended_nodes.rs # 扩展节点
│   │   ├── advanced_sampler.rs # 高级采样器
│   │   ├── image_processing.rs # 图片处理
│   │   └── video_nodes.rs  # 视频节点
│   ├── preview/            # 实时预览
│   ├── workflow/           # 工作流引擎
│   ├── execution/          # 执行引擎
│   ├── agent/              # Agent 模块
│   ├── api/                # HTTP/WebSocket API
│   ├── monitor/            # 系统监控
│   └── storage/            # 存储管理
├── workflows/              # JSON-LD 工作流模板 (29个)
├── skills/                 # Agent 技能定义
├── config/                 # 配置文件
├── docs/                   # 文档
└── tests/                  # 测试
```

## 安装与构建

### 前置依赖

- **Rust** (1.75+，需要 C++20 兼容编译器 - 推荐 Clang 用于 gliding_horse 原生依赖)
- **protoc** (Protobuf 编译器) — gliding_horse Agent OS 依赖，用于编译 gRPC proto 文件
  - Linux: `sudo apt install protobuf-compiler`
  - macOS: `brew install protobuf`
  - Windows: 从 [protobuf releases](https://github.com/protocolbuffers/protobuf/releases) 下载
- **stable-diffusion.cpp**：默认从 `/dev-data/ai-test/stable-diffusion.cpp` 构建 C ABI 和 Rust Worker
- **llama.cpp 或 DeepSeek API**：本地默认路径 `/dev-data/ai-test/llama.cpp-b9810`
- **ffmpeg/ffprobe**：视频编码、缩放和产物验证

### 克隆

本项目依赖 [gliding_horse Agent OS](https://github.com/doiito/gliding_horse.git)，作为 Git Submodule 管理。克隆时需要拉取子模块：

```bash
# 方式一：克隆时自动拉取子模块
git clone --recurse-submodules https://github.com/doiito/media_agent.git

# 方式二：已克隆后补拉子模块
git clone https://github.com/doiito/media_agent.git
cd media_agent
git submodule update --init --recursive
```

### 编译

```bash
# 设置编译器环境（Clang 用于 gliding_horse 原生依赖）
export CC=clang
export CXX=clang++
export CCACHE_DISABLE=1

# 构建 CUDA stable-diffusion.cpp、C bridge 和 Rust Worker
scripts/build_native_runtime.sh

# 编译服务
cargo build --release --bin comfyui-server --bin media-agent-eval

# 运行测试 (215 个测试)
cargo test --lib
```

### 初始化与运行

```bash
# 同时启动 GPU llama-server 和媒体服务
scripts/start_all.sh
```

本地 llama.cpp 默认使用 `--n-gpu-layers auto --fit on`。空闲一秒后模型休眠并
释放显存，stable-diffusion.cpp 采样结束后会自动唤醒执行下一个 Agent 阶段。

### 视频生成说明

默认文本可控视频引擎为 Wan2.2 TI2V 5B。以下脚本下载 Q4_K_M 扩散模型、
Q5_K_M UMT5 编码器和 VAE：

```bash
scripts/download_wan22_ti2v.sh
```

脚本通过官方 `huggingface.co` 直链执行并行 `curl` Range 断点下载和 SHA-256
校验，不设置镜像端点，也不依赖 Python。
Wan 同时支持文生视频和文本可控图生视频，“跳舞”等动作会进入模型条件，而不是被丢弃。

Wan 按交付纵横比自动选择与 832×480 面积接近的原生画布，再由 Rust Worker 等比
缩放/补边到每次请求的交付尺寸。GGUF 量化类型保持不变，`max_vram=-1` 在 16 GB
GPU 上启用 stable-diffusion.cpp 图分段。

SVD 保留为更快、更稳定的图像条件回退，可用 `scripts/download_svd_official.sh`
下载。它不保证严格执行文本动作；显式选择 SVD 做文生视频时采用全原生组合路径：

```text
文生视频 = 原生文生图 -> 原生 SVD 图生视频
```

推荐视频参数 (SVD)：
- **逐帧 CFG**: 默认从 `min_cfg=1.0` 线性提升到末帧 `cfg=3.0`，两端均可按请求覆盖
- **条件变化**: `noise_aug_strength=0.02`，按请求动态传入并同时用于条件图与 added time IDs
- **fps**: 5
- **frames**: 25（5秒视频）
- **生成尺寸**: 1024×576；交付尺寸由每次 Web UI 请求决定并等比缩放/补边
- **motion_bucket_id**: 每请求动态传递（原生范围 0-1023，Web 滑杆 0-255，默认 127）
- **显存**: Wan 视频参数默认使用 stable-diffusion.cpp 官方推荐的 `*=cpu` 驻留策略，计算仍在 CUDA；超过 2 帧时自动启用重叠的原生 VAE tile，使 25 帧视频可在 16 GB GPU 内完成
- **超时**: 图片默认 300 秒，视频独立为 1800 秒，避免高质量时序推理被误杀

Web UI 提供 Fast、Balanced、High 三个质量档位，也允许用户逐项覆盖提示词、模型、
尺寸、steps、CFG、seed、帧数、FPS 和运动强度。档位只提供动态默认值，不会固定
生成内容或任务参数。CA 会自动拒绝近空白图片和实质静止的视频，并返回亮度动态范围、
实际帧数、时长与帧间变化指标。

中文或其他自然语言请求会在 Rust 工具边界经当前 llama.cpp/DeepSeek Provider 动态编译为
模型可用的英文提示词；不合格提示词不会进入 native worker。`/agent/chat` 的
`generation_audit` 会返回实际推理管线、提示词、负面提示词、quality、seed、有效参数和输出路径，便于 Web UI
和评测器审计。此过程不依赖 Python，也不会写死用户生成内容。

图生图默认保持源图宽高比；请求包含保留人物身份或构图时，未显式设置的转换强度会
采用保守值，Web UI 的强度滑块仍可逐请求覆盖。`high` 文生视频先原生生成语义关键帧，
再通过 Wan I2V 生成视频；`fast`/`balanced` 保留直接 T2V 以降低延迟。

## 测试覆盖

| 模块 | 测试数量 |
|------|----------|
| 模型管理 | 56 |
| 多后端支持 | 17 |
| 实时预览 | 18 |
| 节点系统 | 60+ |
| 工作流引擎 | 15+ |
| 监控系统 | 10+ |
| 后端路由 | 5+ |
| Conditioning 系统 | 5+ |
| **总计** | **215** |

## API 接口

### HTTP API

```
POST /api/prompt         # 提交工作流
GET  /api/queue          # 查询队列
POST /api/interrupt      # 中断执行
GET  /api/models         # 模型列表
GET  /api/model/{id}     # 模型详情
POST /api/upload/image   # 上传图片
GET  /api/view/{id}      # 查看结果
```

### WebSocket

```
ws://host/ws?client_id=<id>

事件类型:
- ExecutionStart    # 执行开始
- Progress          # 采样进度
- Preview           # 预览图推送
- Executing         # 节点执行状态
- ExecutionSuccess  # 执行成功
- ExecutionError    # 执行失败
```

## 配置示例

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

## 后续开发计划

- [x] 动态图片/视频生成 Web UI
- [x] 原生 Wan2.2 与 SVD 视频推理
- [ ] 分布式推理
- [x] 原生 CLI 模型下载脚本

## 许可证

MIT License

## 致谢

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - 推理引擎
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM 后端
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 工作流设计参考
