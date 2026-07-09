# ComfyUI Rust Agent

**增强智能图片/视频生成工作流系统**

基于 Rust 语言实现的生产级别 ComfyUI 工作流系统，集成 stable-diffusion.cpp 推理引擎，支持 PDCA 模式和 JSON-LD DAG 工作流。

## 功能特性

### 核心功能

- **图片生成**: 文生图(T2I)、图生图(I2I)、ControlNet 控制
- **视频生成**: SVD 图转视频、帧插值、AnimateDiff 动画
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
- SVD 视频生成
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
- **sd-cli** (stable-diffusion.cpp CLI) — 用于图片/视频生成
- **llama.cpp** (可选) — 用于 LLM 文本生成

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

# 编译 release 版本
cargo build --release

# 运行测试 (215 个测试)
cargo test --lib
```

### 初始化与运行

```bash
# 初始化系统（下载模型、创建目录）
cargo run --release --bin init

# 启动服务
cargo run --release --bin comfyui-server

# 或指定配置
cargo run --release --bin comfyui-server -- --config config/agent.yaml
```

### 视频生成说明

SVD (Stable Video Diffusion) 仅支持 **图生视频**。对于 **文生视频**，系统自动采用组合路径：

```
文生视频 = 文生图 → 图生视频
```

1. 首先通过 text_to_image 生成首帧图像
2. 然后通过 image_to_video (SVD) 动画化

推荐视频参数 (SVD)：
- **cfg**: 2.5（非用户指定的 7）
- **fps**: 5
- **frames**: 25（5秒视频）
- **motion_bucket_id**: 127

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

- [ ] WebUI 前端界面
- [ ] 更多视频生成模型支持
- [ ] 分布式推理
- [ ] 模型自动下载

## 许可证

MIT License

## 致谢

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - 推理引擎
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM 后端
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 工作流设计参考