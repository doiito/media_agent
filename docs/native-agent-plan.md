# Rust 原生媒体 Agent 方案

## 目标

- Web UI 默认使用 Gliding Horse 的 PA、DA、CA、AA PDCA 控制链。
- stable-diffusion.cpp 通过独立 Rust Worker 和固定 C ABI 执行图片/视频推理。
- Agent LLM 可在本地 llama.cpp 与 DeepSeek OpenAI 兼容 API 之间切换。
- 运行和推理不依赖 Python、PyTorch、Diffusers、pip 或 venv。
- 只有通过真实文件解码或 ffprobe 验证的媒体产物才能标记成功。

## 运行架构

```text
Web UI /agent/chat
  -> Gliding Horse SupervisorAgent (explicit four-node DAG)
     -> PA: GenerationSpec
     -> DA: generate_media
        -> Rust BackendRouter
           -> media-sd-worker
              -> libmedia_sd_bridge.so
                 -> stable-diffusion.cpp 3590aa8 / CUDA
     -> CA: inspect_artifact
     -> AA: accept / retry / fail

Gliding Horse UnifiedGateway
  -> llama.cpp llama-server (local OpenAI API)
  or DeepSeek (remote OpenAI-compatible API)
```

stable-diffusion.cpp 与 llama.cpp 保持进程隔离，避免不同 GGML 版本的符号和 ABI
冲突。native 崩溃只终止 Worker，不终止 Web Server。

## 固定依赖

```text
stable-diffusion.cpp: /dev-data/ai-test/stable-diffusion.cpp
commit:               3590aa8d626e671a1b1dc84506ea2932a243a480

llama.cpp:             /dev-data/ai-test/llama.cpp-b9810
llama-server:          /dev-data/ai-test/llama.cpp-b9810/build/bin/llama-server
```

构建 stable-diffusion.cpp C API、bridge 和 Rust Worker：

```bash
scripts/build_native_runtime.sh
```

脚本将 CUDA 架构固定为 8.9，产物位于：

```text
native/runtime/lib/libstable-diffusion.so
native/runtime/lib/libmedia_sd_bridge.so
target/release/media-sd-worker
```

## LLM Provider

默认本地 llama.cpp：

```bash
export AGENT_LLM_PROVIDER=llama_cpp
export LLAMA_CPP_MODEL_PATH=/path/to/instruct-model.gguf
scripts/start_all.sh
```

The checked-in configuration points at the existing local Qwythos/Qwen3.5 9B
Q4 GGUF. llama.cpp defaults to `LLAMA_GPU_LAYERS=auto` and sleeps after one
idle second. Before native media inference, the Agent waits for llama.cpp to
unload its model; the next PA/CA/AA request wakes it automatically. This keeps
GPU acceleration without running the two independent GGML runtimes in VRAM at
the same time. `LLAMA_GPU_LAYERS`, `LLAMA_FIT_TARGET_MIB`, and
`LLAMA_SLEEP_IDLE_SECONDS` remain configurable.

The local server uses one 32K context slot by default. Legacy ComfyUI node
tools are disabled for the normal Agent path (`compatibility_tools_enabled:
false`), keeping PA/DA/CA prompts focused on the four native media tools.
`llama-server` is started with `--n-gpu-layers auto --fit on`, so a CUDA host
uses the GPU automatically. It sleeps after one idle second before media
sampling and wakes for the next Agent phase.

DeepSeek：

```bash
export AGENT_LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...
export AGENT_LLM_MODEL=deepseek-chat
scripts/start_all.sh
```

两种 Provider 都经过 Gliding Horse `UnifiedGateway` 的 OpenAI 兼容协议。地址可通过
`AGENT_LLM_BASE_URL` 覆盖，配置既可带 `/v1` 也可不带，服务会自动规范化。

## Wan2.2 文本可控视频模型

默认视频路由使用 stable-diffusion.cpp 原生支持的 Wan2.2 TI2V 5B，覆盖文生视频和
文本可控图生视频。安装命令：

```bash
scripts/download_wan22_ti2v.sh
```

脚本从 Hugging Face 官方直链下载 Q4_K_M GGUF 扩散模型、Q5_K_M UMT5 编码器和
Wan2.2 VAE，以并行 `curl` Range 请求断点续传并校验 SHA-256；不设置
`HF_ENDPOINT`，不依赖 Python。Rust Worker 通过 C ABI 分别传入
`diffusion_model_path`、`t5xxl_path` 和 `vae_path`；GGUF 不强制转换为 f16。

推理画布根据每次请求的纵横比动态计算，面积约等于 832x480，并对齐到 16 像素；
交付尺寸仍由 Web 请求决定。默认 `flow_shift=3.0`、`max_vram=-1`，用户提示词、
negative prompt、steps、CFG、帧数和 FPS 均逐请求传入。

## SVD 快速回退模型

完整下载 `stabilityai/stable-video-diffusion-img2vid-xt` 后，原生路径配置为：

```text
models/diffusers/stable-video-diffusion-img2vid-xt/svd_xt.safetensors
models/diffusers/stable-video-diffusion-img2vid-xt/image_encoder/model.fp16.safetensors
```

Diffusers 的 `model_index.json` 不参与推理。系统只使用 stable-diffusion.cpp 能直接
加载的 Safetensors 权重。预检会拒绝空文件、损坏文件和改名为 `.safetensors` 的
PyTorch ZIP。

不使用镜像端点、也不依赖 Python 下载工具的安装方式：

```bash
scripts/download_svd_official.sh
```

脚本只访问官方 `huggingface.co`，支持并行 Range 断点续传，并在落盘前校验官方
`9,559,625,980` 字节大小及 SHA-256。

SVD 在模型原生的 `1024x576` 空间采样，Rust Worker 再通过 ffmpeg Lanczos
缩放和等比补边到每次 Web UI 请求的交付尺寸。例如 `1366x780` 是动态输出参数，
不是写死的模型尺寸。

当前 stable-diffusion.cpp `3590aa8` 保留了 SVD 枚举和部分模块，但缺少可执行的
完整接线。`scripts/build_native_runtime.sh` 会幂等应用 `patches/` 下的有序兼容补丁集，
恢复 SVD conditioner、UNet 类型、逐帧 timestep、VideoResBlock Conv3D 布局、
二维 `proj_in/proj_out` 线性投影、官方 EDM/Karras 调度、确定性且不缩放的条件 VAE
posterior mode，以及首帧到末帧的线性 CFG。补丁只扩展 SVD 分支，不替换 Wan/LTX 路径。
条件图噪声增强同样按官方顺序在 VAE 编码前执行，并写入 added time IDs；默认值为
`0.02`，但 Web UI 和 Agent 请求可以动态覆盖。

25 帧时一次性 VAE 解码在上游会尝试申请约 `93.6 GiB` CUDA buffer。C bridge 对
超过 2 帧的请求自动启用 32x32 latent 空间 tile 和 50% 重叠融合；时序 UNet 仍对
完整帧序列采样，不会静默减少帧数。实测 RTX 4060 Ti 16 GB 可生成 1024x576、
25 帧、5 fps、5 秒 MP4。`motion_bucket_id` 从 Web/Agent 经 Rust Worker 和 C ABI
逐请求传入，原生范围为 0-1023，默认值 127 只是建议值而非固定值。SVD 引导范围也
逐请求传入：默认 `min_cfg=1.0`、末帧 `cfg=3.0`，中间帧按官方管线线性插值。

显式选择 SVD 时，文生视频在 Rust backend 内组合两次独立 native Worker：
先由配置的图片模型生成 SVD 原生尺寸首帧，再将首帧字节交给 SVD。两个阶段都使用
stable-diffusion.cpp，不调用 Python，也不会让两个模型同时常驻 GPU。

SVD XT 是图像条件运动模型，不是强文本动作控制模型。提示词会随请求完整保留，
但“跳舞”等具体动作不保证被 SVD 精确执行，因此默认路由使用 Wan2.2；SVD 仅作为
快速稳定动画或显式模型选择。图片任务默认超时为 300 秒，视频任务独立使用
1800 秒上限。Wan 默认设置 `video_params_backend="*=cpu"`，仅将参数驻留放到系统内存，
实际算子仍由 CUDA 执行；这是 stable-diffusion.cpp 推荐的低显存配置，不是 CPU 推理回退。

## 动态参数与质量档位

Web UI 的提示词、输入图片、任务类型、模型、宽高、steps、CFG、seed、帧数、FPS 和 motion
均随请求传递。固定的是 PA→DA→CA→AA 治理流程，不是生成内容或参数。

`generate_media` 不再盲目信任 DA 的提示词草稿。Rust 工具层保存原始用户请求，并通过
当前 Gliding Horse `UnifiedGateway` 做一次低温、无工具调用的英文模型提示词编译；它
会保留主体、动作、数量、场景、时间、天气、光照、风格和构图。编译结果必须通过长度、
ASCII 英文和 CJK 检查；编译失败时只允许回退到已通过同一检查的 DA 英文草稿，否则
任务直接失败。该步骤动态使用本地 llama.cpp 或 DeepSeek，不引入 Python，也不固化
任何用户内容。

- `fast`: 未显式指定 steps 时，图片 14 steps、视频 18 steps。
- `balanced`: 默认图片 24 steps、视频 25 steps。
- `high`: 未显式指定 steps 时，图片 36 steps、视频 30 steps。
- 用户显式设置永远优先于质量档位默认值。
- 图片尺寸要求 8 像素对齐；视频交付尺寸要求偶数；参数越界在加载模型前失败。

图生图在用户没有显式设置宽高时自动保持输入图纵横比，按质量档位选择对齐到 8
像素的推理尺寸；只显式设置一个边时，另一个边由源图比例推导。未显式设置转换强度
时，普通编辑默认 `0.45`，检测到“保留身份/主体/构图”等动态语义时默认 `0.30`。
Web UI 的 `I2I Transform Strength` 可逐请求覆盖该策略，不会写死生成内容。

文生视频的 `fast`/`balanced` 档直接调用 Wan T2V；`high` 档先用图片模型在不超过
768 长边的同纵横比画布生成语义关键帧，再把内存中的关键帧交给 Wan I2V。两阶段
均由 Rust Worker + stable-diffusion.cpp 执行，能以额外推理时间换取更可靠的场景、
主体和光照锚定，不创建 Python/Diffusers 回退。

默认图片模型当前为 SD 1.5，适合 512 级别生成。高质量不是由 Agent 文本宣称：
CA 必须调用 `inspect_artifact` 验证文件、容器、尺寸、亮度动态范围和帧间变化；近空白
图片与实质静止的视频会被拒绝。CA 还会比较原请求和 `effective_prompt`，成功响应通过
`generation_audit` 返回实际管线、有效提示词、负面提示词、quality、seed、有效参数和输出路径。语义、美学
和具体人物动作若要对像素内容自动打分，还需要另配
可在 C++/Rust 路径运行的视觉编码器；纯文本 llama.cpp 不能诚实地看到生成图片。

## 接口与验证

```text
POST /agent/chat       Gliding Horse 默认生成入口
POST /generate         无 Agent 的对照/诊断入口
GET  /agent/status     Agent、Provider 和原生运行时状态
GET  /runtime/preflight 原生二进制与模型容器预检
```

运行 Gliding Horse 能力验收：

```bash
cargo run --bin media-agent-eval
```

报告写入 `output/gliding_horse_evaluation.json`，检查：

- 响应必须来自 `execution_mode=gliding_horse`。
- 成功响应必须有 `artifact_verified=true`。
- 文件必须真实存在、非空、可解码，并通过非空白/视频动态信号检查。
- `pdca_trace` 必须严格为 Plan、Do、Check、Act，且 Check 阶段必须返回真实 artifact 证据。
- 质量用例可配置 `required_prompt_terms`、`required_pipeline` 和期望宽高，评测最终审计是否保留属性并实际走指定管线。
- 统计假成功、通过率、平均 Agent turns 和工具调用数。

## 验收条件

- `cargo check --all-targets` 通过。
- 不安装 Python 也能启动 Server 和两个原生推理后端。
- 文生图、图生图、图生视频、文生视频均通过真实模型能力检查。
- 模型缺失或容器错误在生成前失败，且不重复 PDCA。
- Worker 崩溃或超时不会导致 Web Server 崩溃。
- CA 未验证真实媒体文件时，Server 不发布 `ExecutionSuccess`。
- Web `/view` 只允许读取配置的 output/input/temp 根目录，拒绝绝对路径与 `..` 穿越。
- 上传只接受 10 MiB 内的真实 PNG/JPEG/WebP，限制解码尺寸/内存，使用服务端 UUID
  文件名，并拒绝不安全子目录或符号链接逃逸。
