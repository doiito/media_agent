# 视频生成质量提升笔记 (Video Generation Quality Notes)

> 聚焦图/文本生成视频 (I2V / T2V) 中的人脸与肢体结构变形问题,
> 记录根因分析、已应用的改动以及后续可选提升路径。
> Focus: facial/body structural distortion in image/text-to-video, root-cause analysis, applied changes and further improvement paths.

## 1. 背景 (Background)

用户通过 web UI 提交请求 `利用该图片生成一个胖子跳舞的短视频,5秒钟`,使用 **图生视频 (image_to_video)** 链路生成了 5 秒视频。生成结果存在 **人脸及其他身体部位的结构变形**(identity / anatomy drift),整体质量不理想。

The web test `image_to_video` chain produced a 5-second clip with marked facial and body structural distortion.

## 2. 生成链路与默认参数 (Pipeline & Defaults)

| 环节 | 值 |
|---|---|
| 意图 (intent) | `image_to_video` (I2V) |
| 推理框架 | stable-diffusion.cpp (native worker) |
| 视频模型 | Wan2.2 TI2V 5B `Q4_K_M` 量化 (约 3.4GB) |
| 分辨率 | 832×480 |
| 帧数 / fps | 25 帧 / 5fps = 5.00 秒 |
| cfg | 6.0 |
| min_cfg | 1.0 |
| noise_aug_strength | 0.02 |
| motion_bucket_id | 127 |
| GPU 档位 | Tier16G (16GB) |

默认参数入口: [`src/agent/tools.rs`](../src/agent/tools.rs) 的 `image_to_video` 分支。

## 3. 根因分析 (Root Causes)

Wan2.2 在"人物 + 大幅运动 + 人脸保持"场景表现弱的已知因素, 按影响排序:

| # | 根因 | 影响 |
|---|---|---|
| 1 | **Q4_K_M 量化模型** 精度损失 | 面部/手部细节漂移, 结构错误是最典型症状 |
| 2 | **cfg=6.0 偏低** | 文本引导弱 → 模型自由发挥 → 结构发散 |
| 3 | **25 帧 @5s 长序列无控制** | 高幅运动 + 长序列误差累积 → identity drift |
| 4 | **832×480 分辨率低** | 人脸细节不足以在低分辨率下维持 |
| 5 | Wan I2V 本身人脸保持能力有限 | 非人脸专用模型, 相比 SVD 吃显存但人脸不强 |

## 4. 已应用的改进 (Applied Change)

- **文件**: [`config/config.json`](../config/config.json)
- **改动**: `sd_cpp.video_flow_shift`: `3.0` → `5.0`
- **理由**: Wan 对人物/运动类视频的 `flow_shift` 官方建议值约为 **5.0**。较低的 3.0 采样更激进, 结构容易发散; 提升到 5.0 让采样轨迹更平滑, 有助于在跳舞这类大幅运动场景中维持身份与结构稳定。
- **生效方式**: `flow_shift` 在 server 启动时快照进 backend(`uses_standalone_video_model == true` 时从 `config.video_flow_shift` 读取), 无热重载 → **修改后需重启 server**。

```bash
# 重启 server (从仓库根目录, 依赖 DEEPSEEK_API_KEY 环境变量)
pkill -f target/release/comfyui-server
DEEPSEEK_API_KEY=xxx ./target/release/comfyui-server
```

验证: `curl http://127.0.0.1:8188/agent/status` 返回 `"ready": true` 等。

### 相关代码位置

- `sd_cpp` 配置字段: [`src/backend/sd_cpp.rs:191`](../src/backend/sd_cpp.rs)
- `flow_shift` 进入生成请求:  `src/backend/sd_cpp.rs:1299`
- 默认值定义: `src/backend/sd_cpp.rs:310`

## 5. 进一步提升路线 (Recommended Next Steps)

按成本从低到高:

1. **调整请求侧参数** (无需改代码, web/API)
   - `cfg` 6.0 → **8.0–9.0**(Wan 建议范围 6–9, 人物/动作用高值)
   - `steps` 24 → **30**
   - 帧数 25 → **16**(缩短序列, 减少漂移); fps 3–4
   - 提示词弱化动作幅度 ("slight, gentle dance moves")
2. **质量档切 High** 走 `native_t2i_keyframe_to_i2v` 关键帧流程, 先重建 1024 关键帧再 I2V
3. **换更高精度模型** (结构性提升最大)
   - 下载 `Wan2.2-TI2V-5B-Q8_0.gguf` (约 6.8GB) 并改 `video_model_path`
   - 需 Tier16G / 12G 显存
4. **调 flow_shift** 已做 (5.0), 可按需再微调
5. **提高原生分辨率** `semantic_video_native_width/height` 至 960×544 / 1024×576 (需显存)
6. **换 SVD** — 若更关注"人不变形"而非"严格按指令跳舞", SVD 人脸保持通常优于量化 Wan (代价: 文本指令跟随弱)

## 6. 推荐组合 (Recommended Combo)

1. `video_flow_shift` → 5.0(本次已改)
2. 请求侧 `cfg` 8.5, `steps` 30, `frames` 16
3. 若仍不足 → 换 `Q8_0` 模型

验证方法与现有测试一致: PDCA (Plan → Do → Check → Act) 流程, 视频 artifact 通过 ffprobe 校验时长/分辨率/编码, 并检查 `has_motion`、`temporal_diff` 等运动指标。

---

*Generated on 2026-08-07. 内容基于 `src/agent/tools.rs`、`src/backend/sd_cpp.rs` 实测代码路径。*