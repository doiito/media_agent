use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdWorkerOperation {
    TextToImage,
    ImageToImage,
    TextToVideo,
    ImageToVideo,
    VaeRoundtrip,
}

impl SdWorkerOperation {
    pub fn is_video(self) -> bool {
        matches!(self, Self::TextToVideo | Self::ImageToVideo)
    }

    pub fn is_vae_roundtrip(self) -> bool {
        matches!(self, Self::VaeRoundtrip)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdWorkerRequest {
    pub request_id: String,
    pub operation: SdWorkerOperation,
    pub bridge_library_path: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub diffusion_model_path: String,
    #[serde(default)]
    pub high_noise_diffusion_model_path: String,
    #[serde(default)]
    pub clip_vision_path: String,
    #[serde(default)]
    pub t5xxl_path: String,
    #[serde(default)]
    pub vae_path: String,
    pub backend: String,
    #[serde(default)]
    pub params_backend: String,
    #[serde(default)]
    pub max_vram: String,
    pub weight_type: String,
    pub rng_type: String,
    pub threads: i32,
    pub flash_attention: bool,
    #[serde(default)]
    pub stream_layers: bool,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub sampler: String,
    pub scheduler: String,
    pub width: u32,
    pub height: u32,
    /// Requested delivery dimensions. Zero preserves the generated dimensions.
    #[serde(default)]
    pub output_width: u32,
    #[serde(default)]
    pub output_height: u32,
    pub steps: i32,
    pub cfg: f32,
    #[serde(default)]
    pub flow_shift: f32,
    #[serde(default = "default_min_cfg")]
    pub min_cfg: f32,
    #[serde(default = "default_noise_aug_strength")]
    pub noise_aug_strength: f32,
    pub strength: f32,
    pub seed: i64,
    #[serde(default)]
    pub frames: i32,
    #[serde(default)]
    pub fps: i32,
    #[serde(default = "default_motion_bucket_id")]
    pub motion_bucket_id: i32,
    pub input_path: Option<String>,
    pub output_path: String,
    /// 附加 LoRA(原生透传到 stable-diffusion.cpp)。
    #[serde(default)]
    pub loras: Vec<WorkerLora>,
    /// 原生 hires-fix(latent upscale)。None 不启用。
    #[serde(default)]
    pub hires: Option<WorkerHires>,
}

/// LoRA 规格(与 sd_lora_t 对应)。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLora {
    pub path: String,
    pub multiplier: f32,
}

/// hires-fix 规格(与 sd_hires_params_t 对应;upscaler 固定 LATENT_BICUBIC_ANTIALIASED)。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorkerHires {
    pub scale: f32,
    pub steps: i32,
    pub denoising_strength: f32,
}

fn default_motion_bucket_id() -> i32 {
    127
}

fn default_min_cfg() -> f32 {
    1.0
}

fn default_noise_aug_strength() -> f32 {
    0.02
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdWorkerResponse {
    pub request_id: String,
    pub status: String,
    pub output_path: Option<String>,
    pub model_version: Option<String>,
    pub model_commit: Option<String>,
    pub error: Option<String>,
}

impl SdWorkerResponse {
    pub fn success(
        request_id: String,
        output_path: String,
        model_version: String,
        model_commit: String,
    ) -> Self {
        Self {
            request_id,
            status: "success".to_string(),
            output_path: Some(output_path),
            model_version: Some(model_version),
            model_commit: Some(model_commit),
            error: None,
        }
    }

    pub fn failure(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            status: "failed".to_string(),
            output_path: None,
            model_version: None,
            model_commit: None,
            error: Some(error.into()),
        }
    }
}
