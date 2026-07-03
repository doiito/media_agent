// 模型信息定义
// 包含模型元数据、类型枚举、状态等核心数据结构

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use chrono::{DateTime, Utc};

/// 模型类型枚举
/// 对应 ComfyUI 的 models/ 目录下的子目录
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// 主检查点模型（SD 1.5 / SDXL / SD3 / Flux 等）
    Checkpoint,
    /// UNET 模型（diffusion 模型，用于 SDXL/SD3/Flux）
    UNET,
    /// VAE 模型
    VAE,
    /// CLIP 文本编码器
    CLIP,
    /// 双 CLIP 模型（SDXL/SD3 需要）
    DualCLIP,
    /// CLIP Vision 模型（用于 ControlNet/IPAdapter）
    CLIPVision,
    /// LoRA 模型
    Lora,
    /// ControlNet 模型
    ControlNet,
    /// 放大模型（ESRGAN/RealESRGAN等）
    Upscaler,
    /// 风格模型
    StyleModel,
    /// Embeddings（文本反转）
    Embedding,
    /// 视频模型（SVD/AnimateDiff）
    Video,
    /// 音频模型
    Audio,
    /// 其他类型
    Other(String),
}

impl ModelType {
    /// 从目录名获取模型类型
    pub fn from_dir_name(dir: &str) -> Self {
        match dir.to_lowercase().as_str() {
            "checkpoints" | "checkpoint" => ModelType::Checkpoint,
            "diffusion" | "unet" | "unets" => ModelType::UNET,
            "vae" => ModelType::VAE,
            "clip" => ModelType::CLIP,
            "dual_clip" | "dualclip" => ModelType::DualCLIP,
            "clip_vision" | "clipvision" => ModelType::CLIPVision,
            "lora" | "loras" => ModelType::Lora,
            "controlnet" | "controlnets" => ModelType::ControlNet,
            "upscale" | "upscalers" | "upscale_models" => ModelType::Upscaler,
            "style" | "style_models" => ModelType::StyleModel,
            "embeddings" | "embedding" => ModelType::Embedding,
            "video" | "video_models" => ModelType::Video,
            "audio" | "audio_models" => ModelType::Audio,
            other => ModelType::Other(other.to_string()),
        }
    }

    /// 转换为目录名
    pub fn to_dir_name(&self) -> &str {
        match self {
            ModelType::Checkpoint => "checkpoints",
            ModelType::UNET => "diffusion",
            ModelType::VAE => "vae",
            ModelType::CLIP => "clip",
            ModelType::DualCLIP => "clip",
            ModelType::CLIPVision => "clip_vision",
            ModelType::Lora => "lora",
            ModelType::ControlNet => "controlnet",
            ModelType::Upscaler => "upscale",
            ModelType::StyleModel => "style",
            ModelType::Embedding => "embeddings",
            ModelType::Video => "video",
            ModelType::Audio => "audio",
            ModelType::Other(s) => s.as_str(),
        }
    }

    /// 获取类型显示名
    pub fn display_name(&self) -> String {
        match self {
            ModelType::Checkpoint => "Checkpoint".to_string(),
            ModelType::UNET => "UNET".to_string(),
            ModelType::VAE => "VAE".to_string(),
            ModelType::CLIP => "CLIP".to_string(),
            ModelType::DualCLIP => "DualCLIP".to_string(),
            ModelType::CLIPVision => "CLIPVision".to_string(),
            ModelType::Lora => "LoRA".to_string(),
            ModelType::ControlNet => "ControlNet".to_string(),
            ModelType::Upscaler => "Upscaler".to_string(),
            ModelType::StyleModel => "StyleModel".to_string(),
            ModelType::Embedding => "Embedding".to_string(),
            ModelType::Video => "Video".to_string(),
            ModelType::Audio => "Audio".to_string(),
            ModelType::Other(s) => s.clone(),
        }
    }

    /// 获取所有预定义类型
    pub fn all_predefined() -> Vec<ModelType> {
        vec![
            ModelType::Checkpoint,
            ModelType::UNET,
            ModelType::VAE,
            ModelType::CLIP,
            ModelType::DualCLIP,
            ModelType::CLIPVision,
            ModelType::Lora,
            ModelType::ControlNet,
            ModelType::Upscaler,
            ModelType::StyleModel,
            ModelType::Embedding,
            ModelType::Video,
            ModelType::Audio,
        ]
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// 模型架构（用于区分 SD1.5/SDXL/SD3/Flux 等）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelArchitecture {
    SD15,
    SD20,
    SDXL,
    SDXLRefiner,
    SD3,
    Flux,
    SVD,
    AnimateDiff,
    ControlNet,
    UpscalerESRGAN,
    UpscalerRealESRGAN,
    CLIPVITL,
    CLIPVITH,
    Other(String),
}

impl ModelArchitecture {
    /// 根据模型文件名推断架构
    pub fn infer_from_filename(filename: &str) -> Option<Self> {
        let lower = filename.to_lowercase();
        // 检查 SDXL 系列（sdxl 或 sd_xl 都算）
        let is_sdxl = lower.contains("sdxl") || lower.contains("sd_xl") || lower.contains("sd-xl");
        if is_sdxl {
            if lower.contains("refiner") {
                Some(ModelArchitecture::SDXLRefiner)
            } else {
                Some(ModelArchitecture::SDXL)
            }
        } else if lower.contains("sd3") || lower.contains("stable-diffusion-3") {
            Some(ModelArchitecture::SD3)
        } else if lower.contains("flux") {
            Some(ModelArchitecture::Flux)
        } else if lower.contains("svd") {
            Some(ModelArchitecture::SVD)
        } else if lower.contains("animatediff") || lower.contains("animate_diff") {
            Some(ModelArchitecture::AnimateDiff)
        } else if lower.contains("control") {
            Some(ModelArchitecture::ControlNet)
        } else if lower.contains("esrgan") {
            if lower.contains("real") {
                Some(ModelArchitecture::UpscalerRealESRGAN)
            } else {
                Some(ModelArchitecture::UpscalerESRGAN)
            }
        } else if lower.contains("clip_vit_l") || lower.contains("clip-vit-large") {
            Some(ModelArchitecture::CLIPVITL)
        } else if lower.contains("clip_vit_h") || lower.contains("clip-vit-huge") {
            Some(ModelArchitecture::CLIPVITH)
        } else if lower.contains("v1-5") || lower.contains("sd15") || lower.contains("sd-1.5") {
            Some(ModelArchitecture::SD15)
        } else if lower.contains("v2-") || lower.contains("sd20") || lower.contains("sd-2.0") {
            Some(ModelArchitecture::SD20)
        } else {
            None
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            ModelArchitecture::SD15 => "SD 1.5".to_string(),
            ModelArchitecture::SD20 => "SD 2.0".to_string(),
            ModelArchitecture::SDXL => "SDXL".to_string(),
            ModelArchitecture::SDXLRefiner => "SDXL Refiner".to_string(),
            ModelArchitecture::SD3 => "SD 3".to_string(),
            ModelArchitecture::Flux => "Flux".to_string(),
            ModelArchitecture::SVD => "SVD".to_string(),
            ModelArchitecture::AnimateDiff => "AnimateDiff".to_string(),
            ModelArchitecture::ControlNet => "ControlNet".to_string(),
            ModelArchitecture::UpscalerESRGAN => "ESRGAN".to_string(),
            ModelArchitecture::UpscalerRealESRGAN => "RealESRGAN".to_string(),
            ModelArchitecture::CLIPVITL => "CLIP ViT-L".to_string(),
            ModelArchitecture::CLIPVITH => "CLIP ViT-H".to_string(),
            ModelArchitecture::Other(s) => s.clone(),
        }
    }
}

/// 模型文件格式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFormat {
    Safetensors,
    Ckpt,
    Pt,
    Bin,
    Onnx,
    Engine, // TensorRT engine
    Bbpr,  // bitsandbytes 量化
    Gguf,
    Unknown,
}

impl ModelFormat {
    /// 从文件扩展名推断格式
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().trim_start_matches('.') {
            "safetensors" => ModelFormat::Safetensors,
            "ckpt" => ModelFormat::Ckpt,
            "pt" | "pth" => ModelFormat::Pt,
            "bin" => ModelFormat::Bin,
            "onnx" => ModelFormat::Onnx,
            "engine" | "trt" | "plan" => ModelFormat::Engine,
            "bbpr" => ModelFormat::Bbpr,
            "gguf" | "ggml" => ModelFormat::Gguf,
            _ => ModelFormat::Unknown,
        }
    }

    /// 获取支持的扩展名列表
    pub fn supported_extensions() -> &'static [&'static str] {
        &["safetensors", "ckpt", "pt", "pth", "bin", "onnx", "gguf", "ggml"]
    }

    pub fn display_name(&self) -> &str {
        match self {
            ModelFormat::Safetensors => "safetensors",
            ModelFormat::Ckpt => "ckpt",
            ModelFormat::Pt => "pt",
            ModelFormat::Bin => "bin",
            ModelFormat::Onnx => "onnx",
            ModelFormat::Engine => "engine",
            ModelFormat::Bbpr => "bbpr",
            ModelFormat::Gguf => "gguf",
            ModelFormat::Unknown => "unknown",
        }
    }
}

/// 模型加载状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LoadState {
    /// 未加载
    Unloaded,
    /// 加载中
    Loading,
    /// 已加载到内存
    Loaded,
    /// 已加载到显存（VRAM）
    LoadedVRAM,
    /// 加载失败
    Failed(String),
    /// 正在卸载
    Unloading,
}

/// 模型元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型唯一标识（基于路径的哈希）
    pub id: String,
    /// 模型名称（不含路径和扩展名）
    pub name: String,
    /// 模型显示名（带子目录）
    pub display_name: String,
    /// 文件绝对路径
    pub path: PathBuf,
    /// 相对于 models_dir 的子目录
    pub subdir: String,
    /// 模型类型
    pub model_type: ModelType,
    /// 文件格式
    pub format: ModelFormat,
    /// 文件大小（字节）
    pub size_bytes: u64,
    /// 文件大小（人类可读）
    pub size_human: String,
    /// 文件修改时间
    pub modified: DateTime<Utc>,
    /// 文件 BLAKE3 哈希（前16字节的十六进制，用于唯一标识）
    pub hash: Option<String>,
    /// 推断的架构
    pub architecture: Option<ModelArchitecture>,
    /// 加载状态
    pub load_state: LoadState,
    /// 上次加载时间
    pub last_loaded: Option<DateTime<Utc>>,
    /// 加载次数
    pub load_count: u64,
    /// 自定义标签
    pub tags: Vec<String>,
    /// 额外元数据（来自 safetensors header 等）
    pub metadata: std::collections::HashMap<String, String>,
}

impl ModelInfo {
    /// 从文件路径创建模型信息（不计算哈希，扫描时使用）
    pub fn from_path(
        path: PathBuf,
        models_dir: &std::path::Path,
    ) -> Result<Self, ModelManagerError> {
        let metadata = std::fs::metadata(&path)
            .map_err(|e| ModelManagerError::IoError(format!("{}: {}", path.display(), e)))?;

        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| ModelManagerError::InvalidPath("No filename".to_string()))?;

        // 推断模型类型：从相对路径的第一级目录
        let rel_path = path.strip_prefix(models_dir)
            .map_err(|e| ModelManagerError::InvalidPath(format!("Not under models_dir: {}", e)))?;
        let type_dir = rel_path.iter().next()
            .and_then(|s| s.to_str())
            .unwrap_or("other");
        let model_type = ModelType::from_dir_name(type_dir);

        // 子目录（去除首层类型目录和文件名）
        let subdir: String = rel_path.parent()
            .and_then(|p| p.to_str())
            .map(|s| s.trim_start_matches(type_dir).trim_start_matches('/').to_string())
            .unwrap_or_default();

        // 名称（不含扩展名）
        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename)
            .to_string();

        // 显示名（带子目录）
        let display_name = if subdir.is_empty() {
            stem.clone()
        } else {
            format!("{}/{}", subdir, stem)
        };

        // 扩展名
        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let format = ModelFormat::from_extension(ext);

        // 修改时间
        let modified = metadata.modified()
            .map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .map(|d| {
                        DateTime::<Utc>::from_timestamp(d.as_secs() as i64, 0)
                            .unwrap_or_else(|| Utc::now())
                    })
                    .unwrap_or_else(|_| Utc::now())
            })
            .unwrap_or_else(|_| Utc::now());

        // 推断架构
        let architecture = ModelArchitecture::infer_from_filename(filename);

        // 唯一 ID（基于相对路径）
        let id = blake3::hash(rel_path.to_string_lossy().as_bytes())
            .to_hex()
            .to_string();

        let size_bytes = metadata.len();
        let size_human = format_size(size_bytes);

        Ok(Self {
            id,
            name: stem,
            display_name,
            path,
            subdir,
            model_type,
            format,
            size_bytes,
            size_human,
            modified,
            hash: None,
            architecture,
            load_state: LoadState::Unloaded,
            last_loaded: None,
            load_count: 0,
            tags: Vec::new(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// 是否为 safetensors 格式
    pub fn is_safetensors(&self) -> bool {
        self.format == ModelFormat::Safetensors
    }

    /// 获取文件扩展名
    pub fn extension(&self) -> &str {
        self.path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
    }

    /// 匹配搜索关键词（名称、显示名、架构、标签）
    pub fn matches(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        if q.is_empty() {
            return true;
        }
        self.name.to_lowercase().contains(&q)
            || self.display_name.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.to_lowercase().contains(&q))
            || self.architecture.as_ref().map(|a| a.display_name().to_lowercase().contains(&q)).unwrap_or(false)
            || self.model_type.display_name().to_lowercase().contains(&q)
    }
}

/// 格式化文件大小为人类可读
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// 模型管理错误类型
#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelExists(String),

    #[error("Hash error: {0}")]
    HashError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Load error: {0}")]
    LoadError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<std::io::Error> for ModelManagerError {
    fn from(e: std::io::Error) -> Self {
        ModelManagerError::IoError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_from_dir() {
        assert_eq!(ModelType::from_dir_name("checkpoints"), ModelType::Checkpoint);
        assert_eq!(ModelType::from_dir_name("lora"), ModelType::Lora);
        assert_eq!(ModelType::from_dir_name("loras"), ModelType::Lora);
        assert_eq!(ModelType::from_dir_name("vae"), ModelType::VAE);
        assert_eq!(ModelType::from_dir_name("clip_vision"), ModelType::CLIPVision);
        assert_eq!(ModelType::from_dir_name("custom"), ModelType::Other("custom".to_string()));
    }

    #[test]
    fn test_model_type_to_dir_name() {
        assert_eq!(ModelType::Checkpoint.to_dir_name(), "checkpoints");
        assert_eq!(ModelType::Lora.to_dir_name(), "lora");
        assert_eq!(ModelType::VAE.to_dir_name(), "vae");
    }

    #[test]
    fn test_model_format_from_extension() {
        assert_eq!(ModelFormat::from_extension("safetensors"), ModelFormat::Safetensors);
        assert_eq!(ModelFormat::from_extension("ckpt"), ModelFormat::Ckpt);
        assert_eq!(ModelFormat::from_extension("pt"), ModelFormat::Pt);
        assert_eq!(ModelFormat::from_extension("pth"), ModelFormat::Pt);
        assert_eq!(ModelFormat::from_extension("bin"), ModelFormat::Bin);
        assert_eq!(ModelFormat::from_extension("onnx"), ModelFormat::Onnx);
        assert_eq!(ModelFormat::from_extension("gguf"), ModelFormat::Gguf);
        assert_eq!(ModelFormat::from_extension("xyz"), ModelFormat::Unknown);
    }

    #[test]
    fn test_architecture_inference() {
        assert_eq!(
            ModelArchitecture::infer_from_filename("sdxl_base_1.0.safetensors"),
            Some(ModelArchitecture::SDXL)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("sd_xl_refiner_1.0.safetensors"),
            Some(ModelArchitecture::SDXLRefiner)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("svd.safetensors"),
            Some(ModelArchitecture::SVD)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("svd_xt.safetensors"),
            Some(ModelArchitecture::SVD)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("v1-5-pruned-emaonly.safetensors"),
            Some(ModelArchitecture::SD15)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("control_v11p_sd15_canny.safetensors"),
            Some(ModelArchitecture::ControlNet)
        );
        assert_eq!(
            ModelArchitecture::infer_from_filename("4x_NMKD-Siax_200k.pth"),
            None
        );
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1024 * 1024 * 1024 * 2), "2.00 GB");
    }

    #[test]
    fn test_model_info_from_path() {
        // 创建临时目录和文件
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_model_info_{}", unique));
        let models_dir = temp_dir.join("models");
        let checkpoints_dir = models_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints_dir).unwrap();

        let model_path = checkpoints_dir.join("test_model.safetensors");
        std::fs::write(&model_path, b"fake model data for testing").unwrap();

        let info = ModelInfo::from_path(model_path.clone(), &models_dir).unwrap();

        assert_eq!(info.name, "test_model");
        assert_eq!(info.display_name, "test_model");
        assert_eq!(info.model_type, ModelType::Checkpoint);
        assert_eq!(info.format, ModelFormat::Safetensors);
        assert!(info.size_bytes > 0);
        assert!(info.is_safetensors());
        assert!(!info.id.is_empty());

        // 测试带子目录的模型
        let sub_dir = checkpoints_dir.join("sdxl");
        std::fs::create_dir_all(&sub_dir).unwrap();
        let sub_model = sub_dir.join("sdxl_base.safetensors");
        std::fs::write(&sub_model, b"fake sdxl model").unwrap();

        let sub_info = ModelInfo::from_path(sub_model, &models_dir).unwrap();
        assert_eq!(sub_info.subdir, "sdxl");
        assert_eq!(sub_info.display_name, "sdxl/sdxl_base");
        assert_eq!(sub_info.architecture, Some(ModelArchitecture::SDXL));
    }

    #[test]
    fn test_model_info_matches() {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_model_match_{}", unique));
        let models_dir = temp_dir.join("models");
        let lora_dir = models_dir.join("lora");
        std::fs::create_dir_all(&lora_dir).unwrap();

        let model_path = lora_dir.join("epic_realism.safetensors");
        std::fs::write(&model_path, b"fake").unwrap();

        let info = ModelInfo::from_path(model_path, &models_dir).unwrap();

        assert!(info.matches("epic"));
        assert!(info.matches("realism"));
        assert!(info.matches("lora"));
        assert!(info.matches("LoRA"));
        assert!(!info.matches("controlnet"));
        assert!(info.matches("")); // 空字符串匹配所有
    }

    #[test]
    fn test_load_state() {
        let unloaded = LoadState::Unloaded;
        let loaded = LoadState::Loaded;
        let failed = LoadState::Failed("OOM".to_string());

        assert_eq!(unloaded, LoadState::Unloaded);
        assert_eq!(loaded, LoadState::Loaded);
        assert_ne!(failed, LoadState::Unloaded);
    }
}
