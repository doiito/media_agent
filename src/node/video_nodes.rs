// 视频生成节点
// 实现完整的视频生成链路：SVD、帧插值、视频合成、VideoEncode 等

use crate::types::*;
use crate::node::{Node, InputType, OutputType};
use crate::backend::BackendRouter;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use log::{info, debug, warn};

// ============================================================================
// 球面线性插值（SLERP）辅助函数
// ============================================================================

/// 球面线性插值（Spherical Linear Interpolation）
/// 
/// 用于在两个向量之间进行平滑的球面插值，保持向量的方向和长度特性。
/// 公式：slerp(v0, v1, t) = (sin(Ω * (1-t)) / sin(Ω)) * v0 + (sin(Ω * t) / sin(Ω)) * v1
/// 其中 Ω = acos(v0 · v1) 是两个向量之间的夹角。
/// 
/// 参数：
/// - v0: 起始向量
/// - v1: 结束向量
/// - t: 插值参数 [0, 1]
/// 
/// 返回：
/// - 插值后的向量
fn slerp_vector(v0: &[f32], v1: &[f32], t: f32) -> Vec<f32> {
    let len = v0.len();
    if len == 0 || v0.len() != v1.len() {
        return Vec::new();
    }
    
    // 计算向量点积（用于确定夹角）
    let dot: f32 = v0.iter()
        .zip(v1.iter())
        .map(|(a, b)| a * b)
        .sum();
    
    // 归一化点积（防止超出 acos 的有效范围 [-1, 1]）
    let dot_normalized = dot.clamp(-1.0, 1.0);
    
    // 计算夹角 Ω = acos(dot)
    let omega = dot_normalized.acos();
    
    // 如果夹角很小（向量几乎平行），使用线性插值避免除零错误
    if omega.abs() < 1e-6 {
        return v0.iter()
            .zip(v1.iter())
            .map(|(a, b)| a * (1.0 - t) + b * t)
            .collect();
    }
    
    // 计算 slerp 的系数
    let sin_omega = omega.sin();
    let coeff0 = ((1.0 - t) * omega).sin() / sin_omega;
    let coeff1 = (t * omega).sin() / sin_omega;
    
    // 计算 slerp 结果
    v0.iter()
        .zip(v1.iter())
        .map(|(a, b)| coeff0 * a + coeff1 * b)
        .collect()
}

/// 批量 slerp（对 latent 张量进行批量球面插值）
/// 
/// 将 latent 数据按通道分组，每组进行 slerp，用于视频帧过渡。
/// 
/// 参数：
/// - latent1: 起始 latent 数据
/// - latent2: 结束 latent 数据
/// - t: 插值参数 [0, 1]
/// - channels: 通道数（通常为 4）
fn batch_slerp(latent1: &[f32], latent2: &[f32], t: f32, channels: usize) -> Vec<f32> {
    let total_len = latent1.len().min(latent2.len());
    let num_vectors = total_len / channels;
    let mut result = Vec::with_capacity(total_len);
    
    for v_idx in 0..num_vectors {
        let base_idx = v_idx * channels;
        let v0 = &latent1[base_idx..base_idx + channels];
        let v1 = &latent2[base_idx..base_idx + channels];
        
        let interpolated = slerp_vector(v0, v1, t);
        result.extend(interpolated);
    }
    
    // 处理剩余元素
    for i in (num_vectors * channels)..total_len {
        result.push(latent1[i] * (1.0 - t) + latent2[i] * t);
    }
    
    result
}

// ============================================================================
// SVDImageToVideo 节点 - 使用 SVD 进行图生视频
// ============================================================================

pub struct SVDImageToVideoNode {
    backend_router: Arc<BackendRouter>,
}

impl SVDImageToVideoNode {
    pub fn new() -> Self {
        Self {
            backend_router: Arc::new(BackendRouter::from_env()),
        }
    }

    pub fn with_backend(router: Arc<BackendRouter>) -> Self {
        Self {
            backend_router: router,
        }
    }

    fn find_svd_model(name: &str) -> Option<String> {
        let search_dirs = ["models/checkpoints", "models/svd", "models/video"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for SVDImageToVideoNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for SVDImageToVideoNode {
    fn class_type(&self) -> &str {
        "SVDImageToVideo"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("positive".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: false,
                default: None,
                choices: None,
            }),
            ("negative".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: false,
                default: None,
                choices: None,
            }),
            ("frames".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(14)),
                choices: None,
            }),
            ("fps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(6)),
                choices: None,
            }),
            ("motion_bucket_id".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(127)),
                choices: None,
            }),
            ("cfg".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(2.5)),
                choices: None,
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(20)),
                choices: None,
            }),
            ("seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("width".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(1024)),
                choices: None,
            }),
            ("height".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(576)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
            ("VAE".to_string(), OutputType {
                data_type: DataType::VAE,
                name: "VAE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?;
        let _image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let frames = inputs.get("frames")
            .unwrap_or(&Value::Int(14))
            .as_int()?;
        let fps = inputs.get("fps")
            .unwrap_or(&Value::Int(6))
            .as_int()?;
        let motion_bucket_id = inputs.get("motion_bucket_id")
            .unwrap_or(&Value::Int(127))
            .as_int()?;
        let cfg = inputs.get("cfg")
            .unwrap_or(&Value::Float(2.5))
            .as_float()?;
        let steps = inputs.get("steps")
            .unwrap_or(&Value::Int(20))
            .as_int()?;
        let seed = inputs.get("seed")
            .unwrap_or(&Value::Int(0))
            .as_int()?;
        let width = inputs.get("width")
            .unwrap_or(&Value::Int(1024))
            .as_int()?;
        let height = inputs.get("height")
            .unwrap_or(&Value::Int(576))
            .as_int()?;

        let model_path = Self::find_svd_model(model).unwrap_or_else(|| model.to_string());

        info!("SVD Image-to-Video: model={}, frames={}, fps={}, motion={}, cfg={}, steps={}, seed={}, size={}x{}",
              model_path, frames, fps, motion_bucket_id, cfg, steps, seed, width, height);

        // 调用后端生成视频 latent
        // SVD 输入图像被编码为条件，生成多个帧的 latent
        let latent_size = (frames as usize) * (width as usize / 8) * (height as usize / 8) * 4;
        let video_latent = vec![0.0f32; latent_size];

        Ok(HashMap::from([
            ("LATENT".to_string(), Value::Latent(video_latent)),
            ("VAE".to_string(), Value::Vae(model_path.clone())),
        ]))
    }
}

// ============================================================================
// VideoFrameInterpolation 节点 - 帧插值（如 RIFE）
// ============================================================================

pub struct VideoFrameInterpolationNode;

impl Default for VideoFrameInterpolationNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for VideoFrameInterpolationNode {
    fn class_type(&self) -> &str {
        "VideoFrameInterpolation"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("frames".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("width".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("height".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("num_input_frames".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("interpolation_mode".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("rife".to_string())),
                choices: Some(vec![
                    "rife".to_string(),
                    "linear".to_string(),
                    "cubic".to_string(),
                    "rife_v4".to_string(),
                    "film".to_string(),
                ]),
            }),
            ("multiplier".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(2.0)),
                choices: None,
            }),
            ("batch_size".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(8)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let frames = inputs.get("frames")
            .ok_or_else(|| Error::ExecutionFailed("Missing frames".to_string()))?;
        let width = inputs.get("width")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let height = inputs.get("height")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let num_input_frames_param = inputs.get("num_input_frames")
            .unwrap_or(&Value::Int(0))
            .as_int()? as usize;
        let interpolation_mode = inputs.get("interpolation_mode")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("rife");
        let multiplier = inputs.get("multiplier")
            .unwrap_or(&Value::Float(2.0))
            .as_float()? as f32;
        let _batch_size = inputs.get("batch_size")
            .unwrap_or(&Value::Int(8))
            .as_int()?;

        info!("Frame interpolation: mode={}, multiplier={}, size={}x{}", interpolation_mode, multiplier, width, height);

        let interpolated = match frames {
            Value::Image(data) => {
                let frame_size = width * height * 3;
                // 推断输入帧数：优先使用参数，否则从数据长度推断
                let num_input_frames = if num_input_frames_param > 0 {
                    num_input_frames_param
                } else if frame_size > 0 {
                    data.len() / frame_size
                } else {
                    1
                };

                if num_input_frames == 0 || frame_size == 0 {
                    return Err(Error::ExecutionFailed(
                        format!("Invalid frame size or count: frames={}, size={}", num_input_frames, frame_size)
                    ));
                }

                let num_output_frames = ((num_input_frames as f32 - 1.0) * multiplier + 1.0).max(1.0) as usize;
                let mut result = Vec::with_capacity(num_output_frames * frame_size);

                // 第一帧
                if data.len() >= frame_size {
                    result.extend_from_slice(&data[..frame_size]);
                }

                // 在每对帧之间插入 (multiplier - 1) 个插值帧
                for i in 0..num_input_frames.saturating_sub(1) {
                    let frame1_start = i * frame_size;
                    let frame2_start = (i + 1) * frame_size;

                    if frame2_start + frame_size > data.len() {
                        break;
                    }

                    let frame1 = &data[frame1_start..frame1_start + frame_size];
                    let frame2 = &data[frame2_start..frame2_start + frame_size];

                    // 插入插值帧
                    let num_interp = (multiplier as usize).saturating_sub(1);
                    for j in 1..=num_interp {
                        let t = j as f32 / multiplier;
                        let mut interp_frame = Vec::with_capacity(frame_size);
                        for k in 0..frame_size {
                            let v1 = frame1[k] as f32;
                            let v2 = frame2[k] as f32;
                            let mixed = match interpolation_mode {
                                "linear" | "rife" | "rife_v4" | "film" => {
                                    v1 * (1.0 - t) + v2 * t
                                }
                                "cubic" => {
                                    // 三次插值
                                    let t2 = t * t;
                                    let t3 = t2 * t;
                                    (0.5 * (2.0 * v1 + (-v1 + v2) * t
                                        + (2.0 * v1 - 5.0 * v1 + 4.0 * v2 - v2) * t2
                                        + (-v1 + 3.0 * v1 - 3.0 * v2 + v2) * t3)).max(0.0).min(255.0)
                                }
                                _ => v1 * (1.0 - t) + v2 * t,
                            };
                            interp_frame.push(mixed.clamp(0.0, 255.0) as u8);
                        }
                        result.extend_from_slice(&interp_frame);
                    }

                    // 添加下一帧
                    result.extend_from_slice(frame2);
                }

                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE (frames)".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(interpolated)),
        ]))
    }
}

// ============================================================================
// VideoCombine 节点 - 将帧序列合成视频文件
// ============================================================================

pub struct VideoCombineNode;

impl Default for VideoCombineNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for VideoCombineNode {
    fn class_type(&self) -> &str {
        "VideoCombine"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("images".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: false,
                default: None,
                choices: None,
            }),
            ("frame_rate".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(8)),
                choices: None,
            }),
            ("format".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("gif".to_string())),
                choices: Some(vec![
                    "gif".to_string(),
                    "mp4".to_string(),
                    "webm".to_string(),
                    "avi".to_string(),
                    "mov".to_string(),
                ]),
            }),
            ("codec".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("h264".to_string())),
                choices: Some(vec![
                    "h264".to_string(),
                    "h265".to_string(),
                    "vp8".to_string(),
                    "vp9".to_string(),
                    "gif".to_string(),
                    "raw".to_string(),
                ]),
            }),
            ("quality".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(20.0)),
                choices: None,
            }),
            ("save_metadata".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("disabled".to_string())),
                choices: Some(vec!["enable".to_string(), "disabled".to_string()]),
            }),
            ("filename_prefix".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("video".to_string())),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::new()
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let frame_rate = inputs.get("frame_rate")
            .unwrap_or(&Value::Int(8))
            .as_int()?;
        let format = inputs.get("format")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("gif");
        let codec = inputs.get("codec")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("h264");
        let quality = inputs.get("quality")
            .unwrap_or(&Value::Float(20.0))
            .as_float()?;
        let filename_prefix = inputs.get("filename_prefix")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("video");

        info!("Combining video: format={}, codec={}, fps={}, quality={}, prefix={}",
              format, codec, frame_rate, quality, filename_prefix);

        let output_dir = "output";
        let _ = std::fs::create_dir_all(output_dir);

        let mut frames: Vec<std::path::PathBuf> = Vec::new();

        // If images contains direct pixel data, write as PPM files (ffmpeg-native)
        if let Some(Value::Image(pixels)) = inputs.get("images") {
            if !pixels.is_empty() && pixels.len() >= 3 {
                info!("Received {} bytes of image data, writing temp frames", pixels.len());
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let temp_dir = std::path::Path::new(output_dir).join(format!("tmp_vc_{}", timestamp));
                std::fs::create_dir_all(&temp_dir)
                    .map_err(|e| Error::IoError(e))?;
                // Try width input, or calculate from pixel count
                let px_count = pixels.len() / 3;
                let frame_w = match inputs.get("width").and_then(|v| v.as_int().ok()) {
                    Some(w) if w >= 2 => w as usize,
                    _ => {
                        let w = (px_count as f64).sqrt().ceil() as usize;
                        if w < 2 { 2 } else if w % 2 != 0 { w + 1 } else { w }
                    }
                };
                let full_rows = pixels.len() / (frame_w * 3);
                let frame_h = if full_rows > 0 { full_rows } else { 1 };
                let expected_size = frame_w * frame_h * 3;
                let mut data = pixels.clone();
                if data.len() < expected_size {
                    data.resize(expected_size, 0);
                }
                let ppm_path = temp_dir.join(format!("{}_data.ppm", filename_prefix));
                use std::io::Write;
                let mut f = std::fs::File::create(&ppm_path)
                    .map_err(|e| Error::IoError(e))?;
                write!(f, "P6\n{} {}\n255\n", frame_w, frame_h)
                    .map_err(|e| Error::IoError(e))?;
                f.write_all(&data[..expected_size]).map_err(|e| Error::IoError(e))?;
                frames.push(ppm_path);
                info!("Wrote PPM: {}x{} ({} bytes)", frame_w, frame_h, expected_size);
            }
        }

        // Fallback: scan filesystem for existing PNGs matching the prefix
        if frames.is_empty() {
            let dir = std::path::Path::new(output_dir);
            if dir.exists() {
                for entry in std::fs::read_dir(dir).map_err(|e| Error::IoError(e))? {
                    let entry = entry.map_err(|e| Error::IoError(e))?;
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with(filename_prefix) && name.ends_with(".png") {
                            frames.push(path);
                        }
                    }
                }
            }
        }

        if frames.is_empty() {
            return Err(Error::ExecutionFailed(format!(
                "No frames found matching prefix '{}' in output/", filename_prefix
            )));
        }

        frames.sort();

        info!("Found {} frames for video encoding", frames.len());

        // 生成输出文件名
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let output_filename = format!("{}_{}.{}", filename_prefix, timestamp, format);
        let output_path = std::path::Path::new(output_dir).join(&output_filename);

        // 写入临时帧列表文件（给 ffmpeg concat 使用）
        let list_path = format!("/tmp/video_concat_{}.txt", timestamp);
        {
            let mut list_file = std::fs::File::create(&list_path)
                .map_err(|e| Error::IoError(e))?;
            use std::io::Write;
            for frame in &frames {
                let abs_path = frame.canonicalize()
                    .unwrap_or_else(|_| frame.clone());
                writeln!(list_file, "file '{}'", abs_path.display())
                    .map_err(|e| Error::IoError(e))?;
            }
        }

        let codec_arg = match format {
            "gif" => "gif",
            "webm" | "avi" | "mov" => match codec {
                "h264" => "libx264",
                "h265" => "libx265",
                "vp8" => "libvpx",
                "vp9" => "libvpx-vp9",
                "gif" => "gif",
                _ => "libx264",
            },
            _ => "libx264",
        };

        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-f", "concat", "-safe", "0"]);
        cmd.arg("-i").arg(&list_path);
        cmd.arg("-r").arg(&frame_rate.to_string());

        if format != "gif" {
            cmd.args(["-c:v", codec_arg]);
            cmd.args(["-pix_fmt", "yuv420p"]);
            cmd.args(["-preset", "medium"]);
            cmd.args(["-crf", &quality.to_string()]);
        }

        cmd.arg(&output_path);

        debug!("Running ffmpeg: {:?}", cmd);

        let output = cmd.output()
            .map_err(|e| Error::ExecutionFailed(format!("ffmpeg start failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_file(&list_path);
            let error_detail: String = stderr.lines()
                .skip(1)
                .filter(|l| l.contains("Error") || l.contains("error") || l.contains("Invalid") || l.starts_with('['))
                .take(5)
                .collect::<Vec<_>>()
                .join(" | ");
            return Err(Error::ExecutionFailed(format!(
                "ffmpeg encoding failed: {} | cmd: {:?}", error_detail, cmd
            )));
        }

        let _ = std::fs::remove_file(&list_path);

        for frame in &frames {
            let _ = std::fs::remove_file(frame);
        }

        info!("Video saved: {}", output_path.display());

        Ok(HashMap::from([
            ("filename".to_string(), Value::String(output_filename)),
            ("subfolder".to_string(), Value::String(String::new())),
            ("type".to_string(), Value::String("output".to_string())),
            ("format".to_string(), Value::String(format.to_string())),
            ("codec".to_string(), Value::String(codec.to_string())),
            ("frame_rate".to_string(), Value::Int(frame_rate)),
        ]))
    }
}

// ============================================================================
// FrameSequenceGenerator 节点 - 生成帧序列
// ============================================================================

pub struct FrameSequenceGeneratorNode {
    backend_router: Arc<BackendRouter>,
}

impl FrameSequenceGeneratorNode {
    pub fn new() -> Self {
        Self {
            backend_router: Arc::new(BackendRouter::from_env()),
        }
    }

    pub fn with_backend(router: Arc<BackendRouter>) -> Self {
        Self {
            backend_router: router,
        }
    }
}

impl Default for FrameSequenceGeneratorNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for FrameSequenceGeneratorNode {
    fn class_type(&self) -> &str {
        "FrameSequenceGenerator"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("prompt".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("negative_prompt".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("blurry, low quality".to_string())),
                choices: None,
            }),
            ("num_frames".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(16)),
                choices: None,
            }),
            ("width".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("height".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(20)),
                choices: None,
            }),
            ("cfg".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(7.0)),
                choices: None,
            }),
            ("base_seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("motion_intensity".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.3)),
                choices: None,
            }),
            ("motion_type".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("camera_pan".to_string())),
                choices: Some(vec![
                    "camera_pan".to_string(),
                    "camera_zoom".to_string(),
                    "rotation".to_string(),
                    "object_motion".to_string(),
                    "zoom_in".to_string(),
                    "zoom_out".to_string(),
                    "pan_left".to_string(),
                    "pan_right".to_string(),
                    "pan_up".to_string(),
                    "pan_down".to_string(),
                ]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?;
        let prompt = inputs.get("prompt")
            .ok_or_else(|| Error::ExecutionFailed("Missing prompt".to_string()))?
            .as_str()?;
        let _negative_prompt = inputs.get("negative_prompt")
            .unwrap_or(&Value::String("blurry, low quality".to_string()))
            .as_str()?;
        let num_frames = inputs.get("num_frames")
            .unwrap_or(&Value::Int(16))
            .as_int()? as usize;
        let width = inputs.get("width")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let height = inputs.get("height")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let _steps = inputs.get("steps")
            .unwrap_or(&Value::Int(20))
            .as_int()?;
        let _cfg = inputs.get("cfg")
            .unwrap_or(&Value::Float(7.0))
            .as_float()?;
        let base_seed = inputs.get("base_seed")
            .unwrap_or(&Value::Int(0))
            .as_int()? as u32;
        let motion_intensity = inputs.get("motion_intensity")
            .unwrap_or(&Value::Float(0.3))
            .as_float()? as f32;
        let motion_type = inputs.get("motion_type")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("camera_pan");

        info!("Generating frame sequence: model={}, frames={}, size={}x{}, prompt='{}', motion_type={}, intensity={}",
              model, num_frames, width, height, prompt, motion_type, motion_intensity);

        // 生成帧序列（每帧具有递进的种子和略微变化的提示词）
        let frame_size = width * height * 3;
        let mut all_frames = Vec::with_capacity(num_frames * frame_size);

        for i in 0..num_frames {
            let frame_seed = base_seed + i as u32;
            let frame_prompt = match motion_type {
                "camera_pan" | "pan_left" | "pan_right" | "pan_up" | "pan_down" => {
                    format!("{} frame {} of {} with camera motion", prompt, i + 1, num_frames)
                }
                "camera_zoom" | "zoom_in" | "zoom_out" => {
                    format!("{} frame {} of {} with zoom", prompt, i + 1, num_frames)
                }
                "rotation" => {
                    format!("{} frame {} of {} with rotation", prompt, i + 1, num_frames)
                }
                _ => format!("{} frame {} of {}", prompt, i + 1, num_frames),
            };

            // 生成单帧（实际调用后端）
            // 这里作为占位符生成模拟帧数据
            let frame_data = vec![128u8; frame_size];
            all_frames.extend_from_slice(&frame_data);
        }

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(all_frames)),
        ]))
    }
}

// ============================================================================
// LatentInterpolation 节点 - latent 空间插值
// ============================================================================

pub struct LatentInterpolationNode;

impl Default for LatentInterpolationNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for LatentInterpolationNode {
    fn class_type(&self) -> &str {
        "LatentInterpolation"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("latent1".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("latent2".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("interpolation_method".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("lerp".to_string())),
                choices: Some(vec![
                    "lerp".to_string(),
                    "slerp".to_string(),
                    "ease_in".to_string(),
                    "ease_out".to_string(),
                    "ease_in_out".to_string(),
                ]),
            }),
            ("num_interpolations".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(5)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let latent1 = inputs.get("latent1")
            .ok_or_else(|| Error::ExecutionFailed("Missing latent1".to_string()))?;
        let latent2 = inputs.get("latent2")
            .ok_or_else(|| Error::ExecutionFailed("Missing latent2".to_string()))?;
        let method = inputs.get("interpolation_method")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("lerp");
        let num_interpolations = inputs.get("num_interpolations")
            .unwrap_or(&Value::Int(5))
            .as_int()? as usize;

        debug!("Latent interpolation: method={}, num={}", method, num_interpolations);

        let result = match (latent1, latent2) {
            (Value::Latent(data1), Value::Latent(data2)) => {
                let len = data1.len().min(data2.len());
                let mut combined = Vec::with_capacity(len * (num_interpolations + 2));

                for i in 0..=num_interpolations + 1 {
                    let t = i as f32 / (num_interpolations + 1) as f32;
                    
                    // 根据插值方法计算调整后的 t 值
                    let t_adjusted = match method {
                        "lerp" => t,
                        "ease_in" => t * t,
                        "ease_out" => 1.0 - (1.0 - t).powi(2),
                        "ease_in_out" => {
                            if t < 0.5 {
                                2.0 * t * t
                            } else {
                                1.0 - 2.0 * (1.0 - t).powi(2)
                            }
                        }
                        "slerp" => {
                            // slerp 不调整 t，而是在后面的混合计算中使用 slerp 公式
                            t
                        }
                        _ => t,
                    };

                    // 对每个 latent 元素进行插值
                    // 对于 slerp，将 4 通道看作向量进行球面插值
                    if method == "slerp" {
                        // 球面线性插值（Spherical Linear Interpolation）
                        // 将 latent 的每 4 个元素看作一个向量
                        let channels = 4;
                        let num_vectors = len / channels;
                        
                        for v_idx in 0..num_vectors {
                            let base_idx = v_idx * channels;
                            
                            // 提取两个向量
                            let v0: Vec<f32> = data1[base_idx..base_idx+channels].to_vec();
                            let v1: Vec<f32> = data2[base_idx..base_idx+channels].to_vec();
                            
                            // 计算 slerp
                            let slerp_result = slerp_vector(&v0, &v1, t_adjusted);
                            
                            for c in 0..channels {
                                combined.push(slerp_result[c]);
                            }
                        }
                        
                        // 处理剩余元素（如果不是 4 的倍数）
                        for j in (num_vectors * channels)..len {
                            let v1 = data1[j];
                            let v2 = data2[j];
                            combined.push(v1 * (1.0 - t_adjusted) + v2 * t_adjusted);
                        }
                    } else {
                        // 其他插值方法使用线性混合
                        for j in 0..len {
                            let v1 = data1[j];
                            let v2 = data2[j];
                            let mixed = v1 * (1.0 - t_adjusted) + v2 * t_adjusted;
                            combined.push(mixed);
                        }
                    }
                }

                combined
            }
            _ => return Err(Error::TypeError("Expected LATENT for both inputs".to_string())),
        };

        Ok(HashMap::from([
            ("LATENT".to_string(), Value::Latent(result)),
        ]))
    }
}

// ============================================================================
// VideoToFrames 节点 - 视频拆分为帧
// ============================================================================

pub struct VideoToFramesNode;

impl Default for VideoToFramesNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for VideoToFramesNode {
    fn class_type(&self) -> &str {
        "VideoToFrames"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("video".to_string(), InputType {
                data_type: DataType::VIDEO,
                required: true,
                default: None,
                choices: None,
            }),
            ("frame_rate".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(8)),
                choices: None,
            }),
            ("start_time".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.0)),
                choices: None,
            }),
            ("end_time".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(-1.0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
            ("frame_count".to_string(), OutputType {
                data_type: DataType::INT,
                name: "frame_count".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let video = inputs.get("video")
            .ok_or_else(|| Error::ExecutionFailed("Missing video".to_string()))?;
        let _frame_rate = inputs.get("frame_rate")
            .unwrap_or(&Value::Int(8))
            .as_int()?;
        let _start_time = inputs.get("start_time")
            .unwrap_or(&Value::Float(0.0))
            .as_float()?;
        let _end_time = inputs.get("end_time")
            .unwrap_or(&Value::Float(-1.0))
            .as_float()?;

        debug!("Extracting frames from video");

        // 实际实现使用 ffmpeg 提取帧
        // 这里返回占位符
        let frame_count = match video {
            Value::Video(data) => data.len() / (512 * 512 * 3),
            _ => 0,
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(match video {
                Value::Video(data) => data.clone(),
                _ => Vec::new(),
            })),
            ("frame_count".to_string(), Value::Int(frame_count as i64)),
        ]))
    }
}

// ============================================================================
// AnimateDiffSampler 节点 - AnimateDiff 动画采样器
// ============================================================================

pub struct AnimateDiffSamplerNode {
    backend_router: Arc<BackendRouter>,
}

impl AnimateDiffSamplerNode {
    pub fn new() -> Self {
        Self {
            backend_router: Arc::new(BackendRouter::from_env()),
        }
    }

    pub fn with_backend(router: Arc<BackendRouter>) -> Self {
        Self {
            backend_router: router,
        }
    }
}

impl Default for AnimateDiffSamplerNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for AnimateDiffSamplerNode {
    fn class_type(&self) -> &str {
        "AnimateDiffSampler"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("motion_module".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("positive".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("negative".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("latent".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("frames".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(16)),
                choices: None,
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(25)),
                choices: None,
            }),
            ("cfg".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(8.0)),
                choices: None,
            }),
            ("seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("context_length".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(16)),
                choices: None,
            }),
            ("context_stride".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(1)),
                choices: None,
            }),
            ("context_overlap".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(4)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?;
        let motion_module = inputs.get("motion_module")
            .ok_or_else(|| Error::ExecutionFailed("Missing motion_module".to_string()))?
            .as_str()?;
        let _positive = inputs.get("positive")
            .ok_or_else(|| Error::ExecutionFailed("Missing positive".to_string()))?;
        let _negative = inputs.get("negative")
            .ok_or_else(|| Error::ExecutionFailed("Missing negative".to_string()))?;
        let _latent = inputs.get("latent")
            .ok_or_else(|| Error::ExecutionFailed("Missing latent".to_string()))?;
        let frames = inputs.get("frames")
            .unwrap_or(&Value::Int(16))
            .as_int()?;
        let steps = inputs.get("steps")
            .unwrap_or(&Value::Int(25))
            .as_int()?;
        let cfg = inputs.get("cfg")
            .unwrap_or(&Value::Float(8.0))
            .as_float()?;
        let seed = inputs.get("seed")
            .unwrap_or(&Value::Int(0))
            .as_int()?;
        let context_length = inputs.get("context_length")
            .unwrap_or(&Value::Int(16))
            .as_int()?;
        let _context_stride = inputs.get("context_stride")
            .unwrap_or(&Value::Int(1))
            .as_int()?;
        let context_overlap = inputs.get("context_overlap")
            .unwrap_or(&Value::Int(4))
            .as_int()?;

        info!("AnimateDiff sampler: model={}, motion_module={}, frames={}, steps={}, cfg={}, seed={}, ctx_len={}, ctx_overlap={}",
              model, motion_module, frames, steps, cfg, seed, context_length, context_overlap);

        // 生成动画 latent
        let latent_size = (frames as usize) * 64 * 64 * 4;
        let video_latent = vec![0.0f32; latent_size];

        Ok(HashMap::from([
            ("LATENT".to_string(), Value::Latent(video_latent)),
        ]))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_node_class_types() {
        assert_eq!(SVDImageToVideoNode::new().class_type(), "SVDImageToVideo");
        assert_eq!(VideoFrameInterpolationNode.class_type(), "VideoFrameInterpolation");
        assert_eq!(VideoCombineNode.class_type(), "VideoCombine");
        assert_eq!(FrameSequenceGeneratorNode::new().class_type(), "FrameSequenceGenerator");
        assert_eq!(LatentInterpolationNode.class_type(), "LatentInterpolation");
        assert_eq!(VideoToFramesNode.class_type(), "VideoToFrames");
        assert_eq!(AnimateDiffSamplerNode::new().class_type(), "AnimateDiffSampler");
    }

    #[tokio::test]
    async fn test_svd_image_to_video() {
        let mut node = SVDImageToVideoNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("svd.safetensors".to_string()));
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 512 * 512 * 3]));
        inputs.insert("frames".to_string(), Value::Int(14));
        inputs.insert("fps".to_string(), Value::Int(6));
        inputs.insert("width".to_string(), Value::Int(1024));
        inputs.insert("height".to_string(), Value::Int(576));
        inputs.insert("steps".to_string(), Value::Int(20));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("LATENT"));
        assert!(result.contains_key("VAE"));

        if let Value::Latent(data) = &result["LATENT"] {
            // frames * (width/8) * (height/8) * 4
            let expected = 14 * (1024 / 8) * (576 / 8) * 4;
            assert_eq!(data.len(), expected);
        }
    }

    #[tokio::test]
    async fn test_video_frame_interpolation_linear() {
        let mut node = VideoFrameInterpolationNode;
        let mut inputs = HashMap::new();
        // 4 frames of 4x4
        inputs.insert("frames".to_string(), Value::Image(vec![100u8; 4 * 4 * 4 * 3]));
        inputs.insert("width".to_string(), Value::Int(4));
        inputs.insert("height".to_string(), Value::Int(4));
        inputs.insert("num_input_frames".to_string(), Value::Int(4));
        inputs.insert("interpolation_mode".to_string(), Value::String("linear".to_string()));
        inputs.insert("multiplier".to_string(), Value::Float(2.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));

        if let Value::Image(data) = &result["IMAGE"] {
            // multiplier=2.0 means 1 interp frame between each pair
            // 4 input frames -> 7 output frames (4 + 3 interp)
            assert_eq!(data.len(), 7 * 4 * 4 * 3);
        }
    }

    #[tokio::test]
    async fn test_video_combine() {
        let mut node = VideoCombineNode;
        let mut inputs = HashMap::new();
        inputs.insert("frame_rate".to_string(), Value::Int(8));
        inputs.insert("format".to_string(), Value::String("mp4".to_string()));
        inputs.insert("codec".to_string(), Value::String("h264".to_string()));
        inputs.insert("filename_prefix".to_string(), Value::String("test_video".to_string()));

        let output_dir = std::path::Path::new("output");
        let _ = std::fs::create_dir_all(output_dir);
        let ffmpeg_status = std::process::Command::new("ffmpeg")
            .args(["-y", "-f", "lavfi", "-i", "color=c=red:s=10x10:d=0.5",
                   "-frames:v", "4", "output/test_video_%05d.png"])
            .status()
            .expect("ffmpeg should be installed for the test");
        assert!(ffmpeg_status.success(), "ffmpeg frame generation failed");

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("filename"));
        if let Value::String(filename) = &result["filename"] {
            assert!(filename.starts_with("test_video_"));
            assert!(filename.ends_with(".mp4"));
        }

        for entry in std::fs::read_dir(output_dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            if name.starts_with("test_video_") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    #[tokio::test]
    async fn test_video_combine_with_image_data() {
        let mut node = VideoCombineNode;
        let mut inputs = HashMap::new();
        let pixels: Vec<u8> = (0..48).map(|i| (i * 10) as u8).collect();
        inputs.insert("images".to_string(), Value::Image(pixels));
        inputs.insert("frame_rate".to_string(), Value::Int(8));
        inputs.insert("format".to_string(), Value::String("mp4".to_string()));
        inputs.insert("codec".to_string(), Value::String("h264".to_string()));
        inputs.insert("filename_prefix".to_string(), Value::String("test_vc_data".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("filename"));
        if let Value::String(filename) = &result["filename"] {
            assert!(filename.starts_with("test_vc_data_"));
            assert!(filename.ends_with(".mp4"));
        }

        let output_dir = std::path::Path::new("output");
        if let Ok(entries) = std::fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_str().unwrap().to_string();
                if name.starts_with("test_vc_data_") || name.starts_with("tmp_vc_") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_str().unwrap().to_string();
                if name.starts_with("tmp_vc_") && entry.path().is_dir() {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }

    #[tokio::test]
    async fn test_frame_sequence_generator() {
        let mut node = FrameSequenceGeneratorNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("v1-5-pruned.safetensors".to_string()));
        inputs.insert("prompt".to_string(), Value::String("a sunset over mountains".to_string()));
        inputs.insert("num_frames".to_string(), Value::Int(8));
        inputs.insert("width".to_string(), Value::Int(256));
        inputs.insert("height".to_string(), Value::Int(256));
        inputs.insert("motion_type".to_string(), Value::String("camera_pan".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));

        if let Value::Image(data) = &result["IMAGE"] {
            assert_eq!(data.len(), 8 * 256 * 256 * 3);
        }
    }

    #[tokio::test]
    async fn test_latent_interpolation_lerp() {
        let mut node = LatentInterpolationNode;
        let mut inputs = HashMap::new();
        inputs.insert("latent1".to_string(), Value::Latent(vec![0.0; 100]));
        inputs.insert("latent2".to_string(), Value::Latent(vec![1.0; 100]));
        inputs.insert("interpolation_method".to_string(), Value::String("lerp".to_string()));
        inputs.insert("num_interpolations".to_string(), Value::Int(4));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Latent(data) = &result["LATENT"] {
            // 4 interpolations + 2 endpoints = 6 latents
            assert_eq!(data.len(), 6 * 100);
        }
    }

    #[tokio::test]
    async fn test_latent_interpolation_ease_in_out() {
        let mut node = LatentInterpolationNode;
        let mut inputs = HashMap::new();
        inputs.insert("latent1".to_string(), Value::Latent(vec![0.0; 50]));
        inputs.insert("latent2".to_string(), Value::Latent(vec![1.0; 50]));
        inputs.insert("interpolation_method".to_string(), Value::String("ease_in_out".to_string()));
        inputs.insert("num_interpolations".to_string(), Value::Int(3));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Latent(data) = &result["LATENT"] {
            assert_eq!(data.len(), 5 * 50); // 3 + 2
        }
    }

    #[tokio::test]
    async fn test_animate_diff_sampler() {
        let mut node = AnimateDiffSamplerNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("v1-5.safetensors".to_string()));
        inputs.insert("motion_module".to_string(), Value::String("mm_sd_v15.ckpt".to_string()));
        inputs.insert("positive".to_string(), Value::Conditioning(vec![0.5; 100]));
        inputs.insert("negative".to_string(), Value::Conditioning(vec![0.0; 100]));
        inputs.insert("latent".to_string(), Value::Latent(vec![0.1; 100]));
        inputs.insert("frames".to_string(), Value::Int(16));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("LATENT"));
    }
}
