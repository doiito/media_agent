// stable-diffusion.cpp 集成实现
// 包含进程管理、错误处理、重试机制、并发控制

use crate::types::*;
use crate::backend::{T2IParams, I2IParams, T2VParams, I2VParams};
use crate::backend::sd_worker_protocol::{SdWorkerOperation, SdWorkerRequest, SdWorkerResponse};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tokio::io::AsyncWriteExt;
use log::{debug, info, warn};

// ============================================================================
// 错误类型定义
// ============================================================================

/// stable-diffusion.cpp 特定错误类型
#[derive(Debug, thiserror::Error)]
pub enum SdError {
    #[error("Failed to start process: {0}")]
    ProcessStartFailed(String),

    #[error("Process crashed: {0}")]
    ProcessCrashed(String),

    #[error("Process not running")]
    ProcessNotRunning,

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Operation timed out after {0:?}")]
    TimeoutError(Duration),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Circuit breaker is open after {0} consecutive failures")]
    CircuitBreakerOpen(usize),
}

impl From<SdError> for Error {
    fn from(e: SdError) -> Self {
        Error::BackendError(e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BackendAttempt {
    backend: String,
    rng_mode: String,
}

// ============================================================================
// 配置管理
// ============================================================================

/// stable-diffusion.cpp 后端配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdCppConfig {
    /// 固定的 stable-diffusion.cpp 源码目录，用于版本与构建校验。
    #[serde(default = "default_source_path")]
    pub source_path: String,

    #[serde(default = "default_executable_path")]
    pub executable_path: String,

    /// `native_worker` 使用 Rust Worker + C API；`cli` 仅用于诊断兼容。
    #[serde(default = "default_execution_mode")]
    pub execution_mode: String,

    #[serde(default = "default_worker_path")]
    pub worker_path: String,

    #[serde(default = "default_bridge_library_path")]
    pub bridge_library_path: String,

    #[serde(default)]
    pub model_path: String,

    /// SD1.5 快速档模型(fast 质量档与 Tier4G 使用)。
    #[serde(default)]
    pub fast_model_path: String,

    /// SDXL Q4_K_S GGUF(Tier8G 档,需 scripts/download_sdxl_gguf.sh)。
    #[serde(default)]
    pub sdxl_gguf_q4_path: String,

    /// SDXL Q5_K_M GGUF(Tier12G 档)。
    #[serde(default)]
    pub sdxl_gguf_q5_path: String,

    /// 图片模型独立 VAE(SDXL 改进 VAE)。空则用模型内置 VAE。
    #[serde(default)]
    pub image_vae_path: String,

    /// 图片 LoRA(如 Tier4G 档 SD1.5 写实增强 epic_realism)。空则不启用。
    #[serde(default)]
    pub image_lora_path: String,

    /// 图片 LoRA 权重。
    #[serde(default = "default_image_lora_scale")]
    pub image_lora_scale: f32,

    /// 图片推理显存预算。`-1` 自动分段,小显存档位防 OOM。
    #[serde(default = "default_image_max_vram")]
    pub image_max_vram: String,

    /// 显存档位覆盖(tier4g/tier8g/tier12g/tier16g)。None 时自动探测。
    #[serde(default)]
    pub gpu_tier: Option<String>,

    /// 默认原生视频模型，与图片 checkpoint 分开管理。
    #[serde(default)]
    pub video_model_path: String,

    /// Optional SVD checkpoint retained as a fast, image-conditioned fallback.
    #[serde(default)]
    pub svd_model_path: String,

    /// Optional second diffusion stage for Wan2.2 A14B models.
    #[serde(default)]
    pub video_high_noise_model_path: String,

    /// Text encoder used by prompt-conditioned native video models such as Wan.
    #[serde(default)]
    pub video_t5xxl_path: String,

    /// Standalone VAE used by prompt-conditioned native video models.
    #[serde(default)]
    pub video_vae_path: String,

    /// 视频模型使用的 CLIP Vision 权重。
    #[serde(default)]
    pub clip_vision_path: String,

    /// SVD model-space dimensions. Delivery dimensions remain request-specific.
    #[serde(default = "default_svd_native_width")]
    pub svd_native_width: usize,

    #[serde(default = "default_svd_native_height")]
    pub svd_native_height: usize,

    /// Native Wan/LTX inference dimensions. Delivery dimensions are request-specific.
    #[serde(default = "default_semantic_video_native_width")]
    pub semantic_video_native_width: usize,

    #[serde(default = "default_semantic_video_native_height")]
    pub semantic_video_native_height: usize,

    /// 计算后端 (cuda/vulkan/cpu/metal)
    #[serde(default = "default_backend")]
    pub backend: String,

    /// 精度设置 (f32/f16/q4_0/q5_0/q8_0)
    #[serde(default = "default_precision")]
    pub precision: String,

    #[serde(default = "default_flash_attention")]
    pub flash_attention: bool,

    #[serde(default)]
    pub offload_to_cpu: bool,

    /// Parameter residency assignment accepted by stable-diffusion.cpp.
    #[serde(default)]
    pub video_params_backend: String,

    /// CUDA graph VRAM budget. `-1` keeps one GiB free and segments large graphs.
    #[serde(default = "default_video_max_vram")]
    pub video_max_vram: String,

    #[serde(default)]
    pub video_stream_layers: bool,

    #[serde(default = "default_video_flow_shift")]
    pub video_flow_shift: f32,

    /// RNG 模式 (cuda/cpu - ComfyUI兼容使用cpu)
    #[serde(default = "default_rng_mode")]
    pub rng_mode: String,

    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Video inference has substantially larger temporal graphs than image inference.
    #[serde(default = "default_video_timeout_secs")]
    pub video_timeout_secs: u64,

    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,

    #[serde(default = "default_max_queue_size")]
    pub max_queue_size: usize,

    #[serde(default = "default_health_check_interval")]
    pub health_check_interval: u64,

    #[serde(default)]
    pub idle_timeout_secs: u64,

    #[serde(default = "default_circuit_breaker_threshold")]
    pub circuit_breaker_threshold: usize,

    #[serde(default = "default_circuit_breaker_reset_time")]
    pub circuit_breaker_reset_secs: u64,

    #[serde(default)]
    pub extra_args: Vec<String>,

    #[serde(default)]
    pub env_vars: std::collections::HashMap<String, String>,
}

fn default_executable_path() -> String {
    std::env::var("SD_CPP_EXECUTABLE").unwrap_or_else(|_| {
        let local = "/dev-data/ai-test/stable-diffusion.cpp/build-cuda/bin/sd-cli";
        if std::path::Path::new(local).is_file() {
            local.to_string()
        } else {
            "sd-cli".to_string()
        }
    })
}

fn default_source_path() -> String {
    std::env::var("SD_CPP_SOURCE_DIR")
        .unwrap_or_else(|_| "/dev-data/ai-test/stable-diffusion.cpp".to_string())
}

fn default_execution_mode() -> String {
    std::env::var("SD_CPP_EXECUTION_MODE")
        .unwrap_or_else(|_| "native_worker".to_string())
}

fn default_worker_path() -> String {
    std::env::var("SD_CPP_WORKER")
        .unwrap_or_else(|_| "target/release/media-sd-worker".to_string())
}

fn default_bridge_library_path() -> String {
    std::env::var("SD_CPP_BRIDGE_LIBRARY")
        .unwrap_or_else(|_| "native/runtime/lib/libmedia_sd_bridge.so".to_string())
}

fn default_svd_native_width() -> usize {
    1024
}

fn default_svd_native_height() -> usize {
    576
}

fn default_semantic_video_native_width() -> usize {
    832
}

fn default_semantic_video_native_height() -> usize {
    480
}

fn default_backend() -> String {
    std::env::var("SD_CPP_BACKEND").unwrap_or_else(|_| "cuda".to_string())
}

fn default_precision() -> String {
    std::env::var("SD_CPP_PRECISION").unwrap_or_else(|_| "f16".to_string())
}

fn default_flash_attention() -> bool {
    std::env::var("SD_CPP_FLASH_ATTENTION")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true)
}

fn default_rng_mode() -> String {
    "cpu".to_string()
}

fn default_timeout_secs() -> u64 {
    300
}

fn default_video_timeout_secs() -> u64 {
    1800
}

fn default_video_max_vram() -> String {
    "-1".to_string()
}

fn default_video_flow_shift() -> f32 {
    3.0
}

fn default_image_lora_scale() -> f32 {
    0.7
}

fn default_image_max_vram() -> String {
    "-1".to_string()
}

fn default_max_retries() -> usize {
    3
}

fn default_max_concurrent_tasks() -> usize {
    2
}

fn default_max_queue_size() -> usize {
    100
}

fn default_health_check_interval() -> u64 {
    30
}

fn default_circuit_breaker_threshold() -> usize {
    5
}

fn default_circuit_breaker_reset_time() -> u64 {
    60
}

fn normalize_rng_mode(rng_mode: &str, backend: &str) -> String {
    match rng_mode.trim() {
        "" | "auto" => {
            if backend.starts_with("cuda") {
                "cuda".to_string()
            } else {
                "cpu".to_string()
            }
        }
        value => value.to_string(),
    }
}

fn build_backend_attempts(backend: &str, rng_mode: &str) -> Vec<BackendAttempt> {
    let preferred_backend = match backend.trim() {
        "" | "auto" => "cuda",
        other => other,
    };

    let mut attempts = vec![BackendAttempt {
        backend: preferred_backend.to_string(),
        rng_mode: normalize_rng_mode(rng_mode, preferred_backend),
    }];

    if preferred_backend != "cpu" {
        attempts.push(BackendAttempt {
            backend: "cpu".to_string(),
            rng_mode: "cpu".to_string(),
        });
    }

    attempts
}

fn should_retry_with_cpu(stderr: &str) -> bool {
    let detail = stderr.to_ascii_lowercase();
    detail.contains("cuda driver version is insufficient")
        || detail.contains("failed to initialize cuda")
        || detail.contains("backend config failed: backend 'cuda' was not found")
        || detail.contains("backend 'cuda' was not found")
}

fn summarize_process_output(stderr: &[u8], stdout: &[u8]) -> String {
    let stderr_text = String::from_utf8_lossy(stderr);
    let stdout_text = String::from_utf8_lossy(stdout);

    let mut lines: Vec<String> = stderr_text
        .lines()
        .chain(stdout_text.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(20)
        .map(|line| line.to_string())
        .collect();

    if lines.is_empty() {
        lines.push("unknown error".to_string());
    }

    lines.join(" | ")
}

impl Default for SdCppConfig {
    fn default() -> Self {
        Self {
            source_path: default_source_path(),
            executable_path: default_executable_path(),
            execution_mode: default_execution_mode(),
            worker_path: default_worker_path(),
            bridge_library_path: default_bridge_library_path(),
            model_path: String::new(),
            fast_model_path: String::new(),
            sdxl_gguf_q4_path: String::new(),
            sdxl_gguf_q5_path: String::new(),
            image_vae_path: String::new(),
            image_lora_path: String::new(),
            image_lora_scale: default_image_lora_scale(),
            image_max_vram: default_image_max_vram(),
            gpu_tier: None,
            video_model_path: String::new(),
            svd_model_path: String::new(),
            video_high_noise_model_path: String::new(),
            video_t5xxl_path: String::new(),
            video_vae_path: String::new(),
            clip_vision_path: String::new(),
            svd_native_width: default_svd_native_width(),
            svd_native_height: default_svd_native_height(),
            semantic_video_native_width: default_semantic_video_native_width(),
            semantic_video_native_height: default_semantic_video_native_height(),
            backend: default_backend(),
            precision: default_precision(),
            flash_attention: default_flash_attention(),
            offload_to_cpu: false,
            video_params_backend: String::new(),
            video_max_vram: default_video_max_vram(),
            video_stream_layers: false,
            video_flow_shift: default_video_flow_shift(),
            rng_mode: default_rng_mode(),
            timeout_secs: default_timeout_secs(),
            video_timeout_secs: default_video_timeout_secs(),
            max_retries: default_max_retries(),
            max_concurrent_tasks: default_max_concurrent_tasks(),
            max_queue_size: default_max_queue_size(),
            health_check_interval: default_health_check_interval(),
            idle_timeout_secs: 0,
            circuit_breaker_threshold: default_circuit_breaker_threshold(),
            circuit_breaker_reset_secs: default_circuit_breaker_reset_time(),
            extra_args: Vec::new(),
            env_vars: std::collections::HashMap::new(),
        }
    }
}

impl SdCppConfig {
    pub fn from_env() -> Self {
        // 优先级：环境变量 > config.json > 硬编码默认
        // 节点层（KSampler 等）用 from_env()，不读 AppConfig，
        // 所以在这里 fallback 到 config.json
        let mut config = Self::load_from_config_file().unwrap_or_default();

        if let Ok(val) = std::env::var("SD_CPP_EXECUTABLE") {
            config.executable_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_SOURCE_DIR") {
            config.source_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_EXECUTION_MODE") {
            config.execution_mode = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_WORKER") {
            config.worker_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_BRIDGE_LIBRARY") {
            config.bridge_library_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_MODEL_PATH") {
            config.model_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_FAST_MODEL_PATH") {
            config.fast_model_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_SDXL_GGUF_Q4_PATH") {
            config.sdxl_gguf_q4_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_SDXL_GGUF_Q5_PATH") {
            config.sdxl_gguf_q5_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_IMAGE_VAE_PATH") {
            config.image_vae_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_IMAGE_LORA_PATH") {
            config.image_lora_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_IMAGE_LORA_SCALE") {
            if let Ok(value) = val.parse() {
                config.image_lora_scale = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_IMAGE_MAX_VRAM") {
            config.image_max_vram = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_GPU_TIER") {
            config.gpu_tier = Some(val);
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_MODEL_PATH") {
            config.video_model_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_SVD_MODEL_PATH") {
            config.svd_model_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_HIGH_NOISE_MODEL_PATH") {
            config.video_high_noise_model_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_T5XXL_PATH") {
            config.video_t5xxl_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_VAE_PATH") {
            config.video_vae_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_CLIP_VISION_PATH") {
            config.clip_vision_path = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_SVD_NATIVE_WIDTH") {
            if let Ok(value) = val.parse() {
                config.svd_native_width = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_SVD_NATIVE_HEIGHT") {
            if let Ok(value) = val.parse() {
                config.svd_native_height = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_SEMANTIC_VIDEO_NATIVE_WIDTH") {
            if let Ok(value) = val.parse() {
                config.semantic_video_native_width = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_SEMANTIC_VIDEO_NATIVE_HEIGHT") {
            if let Ok(value) = val.parse() {
                config.semantic_video_native_height = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_BACKEND") {
            config.backend = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_PRECISION") {
            config.precision = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_FLASH_ATTENTION") {
            config.flash_attention = val == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("SD_CPP_RNG_MODE") {
            config.rng_mode = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_OFFLOAD_CPU") {
            config.offload_to_cpu = val == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_PARAMS_BACKEND") {
            config.video_params_backend = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_MAX_VRAM") {
            config.video_max_vram = val;
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_STREAM_LAYERS") {
            config.video_stream_layers = val == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_FLOW_SHIFT") {
            if let Ok(value) = val.parse() {
                config.video_flow_shift = value;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse() {
                config.timeout_secs = secs;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_VIDEO_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse() {
                config.video_timeout_secs = secs;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_MAX_RETRIES") {
            if let Ok(retries) = val.parse() {
                config.max_retries = retries;
            }
        }
        if let Ok(val) = std::env::var("SD_CPP_MAX_CONCURRENT") {
            if let Ok(concurrent) = val.parse() {
                config.max_concurrent_tasks = concurrent;
            }
        }
        config
    }

    fn load_from_config_file() -> Option<Self> {
        #[derive(Deserialize)]
        struct AppConfigShell {
            #[serde(default)]
            sd_cpp: Option<SdCppConfig>,
        }

        for path in &["config/config.json", "config.json"] {
            let content = std::fs::read_to_string(path).ok()?;
            if let Ok(cfg) = serde_json::from_str::<AppConfigShell>(&content) {
                if cfg.sd_cpp.is_some() {
                    return cfg.sd_cpp;
                }
            }
        }
        None
    }

    /// 从配置文件加载
    pub fn from_file(path: &str) -> Result<Self, SdError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// 构建命令行参数
    pub fn build_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        if !self.model_path.is_empty() {
            args.push("--model".to_string());
            args.push(self.model_path.clone());
        }

        args.push("--backend".to_string());
        args.push(self.backend.clone());

        args.push("--precision".to_string());
        args.push(self.precision.clone());

        if self.flash_attention {
            args.push("--diffusion-fa".to_string());
        }

        if self.offload_to_cpu {
            args.push("--offload-to-cpu".to_string());
        }

        args.push("--rng".to_string());
        args.push(self.rng_mode.clone());

        // 附加额外参数
        args.extend(self.extra_args.iter().cloned());

        args
    }
}

// ============================================================================
// 进程状态管理
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessStatus {
    NotStarted,
    Running,
    Idle,
    Busy,
    Stopped,
    Error(String),
}

impl std::fmt::Display for ProcessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessStatus::NotStarted => write!(f, "NotStarted"),
            ProcessStatus::Running => write!(f, "Running"),
            ProcessStatus::Idle => write!(f, "Idle"),
            ProcessStatus::Busy => write!(f, "Busy"),
            ProcessStatus::Stopped => write!(f, "Stopped"),
            ProcessStatus::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

/// 断路器
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_count: usize,
    threshold: usize,
    last_failure_time: Option<Instant>,
    reset_duration: Duration,
    is_open: bool,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, reset_secs: u64) -> Self {
        Self {
            failure_count: 0,
            threshold,
            last_failure_time: None,
            reset_duration: Duration::from_secs(reset_secs),
            is_open: false,
        }
    }

    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.is_open = false;
        self.last_failure_time = None;
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.threshold {
            self.is_open = true;
            warn!(
                "Circuit breaker opened after {} consecutive failures",
                self.failure_count
            );
        }
    }

    pub fn is_allowed(&mut self) -> bool {
        if !self.is_open {
            return true;
        }

        if let Some(last_failure) = self.last_failure_time {
            if last_failure.elapsed() >= self.reset_duration {
                info!("Circuit breaker reset after timeout");
                self.is_open = false;
                self.failure_count = 0;
                return true;
            }
        }

        false
    }

    pub fn get_failure_count(&self) -> usize {
        self.failure_count
    }

    pub fn is_open(&self) -> bool {
        self.is_open
    }
}

// ============================================================================
// 请求和响应类型
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct SdRequest {
    pub mode: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg: f32,
    pub sampler: String,
    pub seed: usize,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controlnet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoise: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SdResponse {
    pub status: String,
    #[serde(default)]
    pub output_path: String,
    #[serde(default)]
    pub seed: usize,
    #[serde(default)]
    pub time: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl SdResponse {
    pub fn is_success(&self) -> bool {
        self.status == "success"
    }
}

// ============================================================================
// 进程管理器
// ============================================================================

/// stable-diffusion.cpp 进程管理器
pub struct SdCppProcessManager {
    config: SdCppConfig,
    process: Option<Child>,
    status: ProcessStatus,
    circuit_breaker: CircuitBreaker,
    last_activity: Option<Instant>,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
}

impl SdCppProcessManager {
    pub fn new(config: SdCppConfig) -> Self {
        let circuit_breaker = CircuitBreaker::new(
            config.circuit_breaker_threshold,
            config.circuit_breaker_reset_secs,
        );
        Self {
            config,
            process: None,
            status: ProcessStatus::NotStarted,
            circuit_breaker,
            last_activity: None,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
        }
    }

    /// 启动进程
    pub fn start(&mut self) -> Result<(), SdError> {
        if self.process.is_some() {
            warn!("Process already running, restarting...");
            self.stop()?;
        }

        info!("Starting stable-diffusion.cpp process: {}", self.config.executable_path);

        let mut cmd = Command::new(&self.config.executable_path);
        cmd.args(self.config.build_args());

        // 设置环境变量
        for (key, value) in &self.config.env_vars {
            cmd.env(key, value);
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn().map_err(|e| {
            SdError::ProcessStartFailed(format!(
                "Failed to spawn '{}': {}",
                self.config.executable_path, e
            ))
        })?;

        self.process = Some(child);
        self.status = ProcessStatus::Idle;
        self.last_activity = Some(Instant::now());

        info!("stable-diffusion.cpp process started successfully");
        Ok(())
    }

    /// 停止进程
    pub fn stop(&mut self) -> Result<(), SdError> {
        if let Some(mut process) = self.process.take() {
            info!("Stopping stable-diffusion.cpp process");

            // 尝试优雅关闭
            if let Some(stdin) = process.stdin.as_mut() {
                let _ = stdin.write_all(b"{\"mode\":\"quit\"}\n");
                let _ = stdin.flush();
            }

            // 等待一段时间
            std::thread::sleep(Duration::from_millis(500));

            // 强制终止
            match process.kill() {
                Ok(()) => {
                    let _ = process.wait();
                    self.status = ProcessStatus::Stopped;
                    info!("Process stopped");
                }
                Err(e) => {
                    self.status = ProcessStatus::Error(e.to_string());
                    return Err(SdError::ProcessCrashed(e.to_string()));
                }
            }
        }
        Ok(())
    }

    /// 检查进程是否运行
    pub fn is_running(&mut self) -> bool {
        if let Some(process) = self.process.as_mut() {
            match process.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => {
                    self.status = ProcessStatus::Stopped;
                    false
                }
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// 获取进程状态
    pub fn status(&self) -> ProcessStatus {
        self.status.clone()
    }

    /// 发送请求并获取响应（带重试机制）
    pub fn execute_request(&mut self, request: &SdRequest) -> Result<SdResponse, SdError> {
        // 检查断路器
        if !self.circuit_breaker.is_allowed() {
            return Err(SdError::CircuitBreakerOpen(
                self.circuit_breaker.get_failure_count(),
            ));
        }

        // 确保进程运行
        if !self.is_running() {
            info!("Process not running, starting...");
            self.start()?;
        }

        self.total_requests += 1;
        self.status = ProcessStatus::Busy;

        let result = self.execute_with_retry(request);

        match &result {
            Ok(response) => {
                if response.is_success() {
                    self.successful_requests += 1;
                    self.circuit_breaker.record_success();
                    self.status = ProcessStatus::Idle;
                } else {
                    self.failed_requests += 1;
                    self.circuit_breaker.record_failure();
                    self.status = ProcessStatus::Error(
                        response.error.clone().unwrap_or_else(|| "Unknown error".to_string())
                    );
                }
            }
            Err(e) => {
                self.failed_requests += 1;
                self.circuit_breaker.record_failure();
                self.status = ProcessStatus::Error(e.to_string());
            }
        }

        self.last_activity = Some(Instant::now());
        result
    }

    /// 带重试的执行
    fn execute_with_retry(&mut self, request: &SdRequest) -> Result<SdResponse, SdError> {
        let mut last_error: Option<SdError> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * (2_u64.pow(attempt as u32 - 1)));
                info!("Retry attempt {} after {:?}", attempt, delay);
                std::thread::sleep(delay);

                // 重启进程
                if !self.is_running() {
                    if let Err(e) = self.start() {
                        last_error = Some(e);
                        continue;
                    }
                }
            }

            match self.send_request_once(request) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!("Attempt {} failed: {}", attempt + 1, e);
                    let needs_restart = matches!(&e, SdError::ProcessCrashed(_) | SdError::ProcessNotRunning);
                    last_error = Some(e);

                    // 检查是否需要重启进程
                    if needs_restart {
                        let _ = self.stop();
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| SdError::ExecutionFailed("Unknown error".to_string())))
    }

    /// 单次请求发送
    fn send_request_once(&mut self, request: &SdRequest) -> Result<SdResponse, SdError> {
        let process = self.process.as_mut().ok_or(SdError::ProcessNotRunning)?;

        let json = serde_json::to_string(request)?;

        // 获取stdin
        let stdin = process.stdin.as_mut().ok_or_else(|| {
            SdError::CommunicationError("Failed to get stdin".to_string())
        })?;

        debug!("Sending request: {}", json);

        // 发送请求
        stdin.write_all(json.as_bytes()).map_err(|e| {
            SdError::CommunicationError(format!("Failed to write to stdin: {}", e))
        })?;
        stdin.write_all(b"\n").map_err(|e| {
            SdError::CommunicationError(format!("Failed to write newline: {}", e))
        })?;
        stdin.flush().map_err(|e| {
            SdError::CommunicationError(format!("Failed to flush stdin: {}", e))
        })?;

        // 读取响应
        let stdout = process.stdout.as_mut().ok_or_else(|| {
            SdError::CommunicationError("Failed to get stdout".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();

        // 读取响应行（带超时检测）
        let start_time = Instant::now();
        loop {
            response_line.clear();
            match reader.read_line(&mut response_line) {
                Ok(0) => {
                    return Err(SdError::ProcessCrashed(
                        "Process closed stdout (likely crashed)".to_string(),
                    ));
                }
                Ok(_) => {
                    let trimmed = response_line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // 尝试解析为JSON响应
                    if trimmed.starts_with('{') {
                        match serde_json::from_str::<SdResponse>(trimmed) {
                            Ok(response) => {
                                debug!("Received response: {:?}", response);
                                return Ok(response);
                            }
                            Err(e) => {
                                debug!("Failed to parse line as response: {}, line: {}", e, trimmed);
                                continue;
                            }
                        }
                    } else {
                        debug!("Process output: {}", trimmed);
                    }
                }
                Err(e) => {
                    return Err(SdError::CommunicationError(format!(
                        "Failed to read from stdout: {}",
                        e
                    )));
                }
            }

            // 超时检测
            if start_time.elapsed() > Duration::from_secs(self.config.timeout_secs) {
                return Err(SdError::TimeoutError(Duration::from_secs(
                    self.config.timeout_secs,
                )));
            }
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> SdProcessStats {
        SdProcessStats {
            status: self.status.to_string(),
            total_requests: self.total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            success_rate: if self.total_requests > 0 {
                self.successful_requests as f64 / self.total_requests as f64
            } else {
                0.0
            },
            circuit_breaker_open: self.circuit_breaker.is_open(),
            last_activity_ago_secs: self.last_activity.map(|t| t.elapsed().as_secs()),
        }
    }

    /// 健康检查
    pub fn health_check(&mut self) -> Result<bool, SdError> {
        if !self.is_running() {
            return Ok(false);
        }

        // 发送ping请求
        let ping_request = SdRequest {
            mode: "ping".to_string(),
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 64,
            height: 64,
            steps: 1,
            cfg: 1.0,
            sampler: "euler".to_string(),
            seed: 0,
            model_path: String::new(),
            input_image: None,
            controlnet: None,
            denoise: None,
            request_id: Some("health_check".to_string()),
        };

        match self.send_request_once(&ping_request) {
            Ok(response) => Ok(response.is_success()),
            Err(_) => Ok(false),
        }
    }
}

impl Drop for SdCppProcessManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// 进程统计信息
#[derive(Debug, Clone, Serialize)]
pub struct SdProcessStats {
    pub status: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub circuit_breaker_open: bool,
    pub last_activity_ago_secs: Option<u64>,
}

// ============================================================================
// 高级后端接口
// ============================================================================

/// stable-diffusion.cpp 后端实现
pub struct StableDiffusionCppBackend {
    process_manager: Arc<Mutex<SdCppProcessManager>>,
    semaphore: Arc<Semaphore>,
    config: SdCppConfig,
}

struct NativeTemporaryFiles {
    input: Option<String>,
    output: String,
}

impl Drop for NativeTemporaryFiles {
    fn drop(&mut self) {
        if let Some(path) = &self.input {
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_file(&self.output);
    }
}

impl StableDiffusionCppBackend {
    pub fn new(config: SdCppConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));
        let process_manager = Arc::new(Mutex::new(SdCppProcessManager::new(config.clone())));

        Self {
            process_manager,
            semaphore,
            config,
        }
    }

    fn uses_native_worker(&self) -> bool {
        self.config.execution_mode.eq_ignore_ascii_case("native_worker")
    }

    fn native_request(
        &self,
        operation: SdWorkerOperation,
        model_path: String,
        prompt: String,
        negative_prompt: String,
        width: usize,
        height: usize,
        steps: usize,
        cfg: f32,
        sampler: &str,
        strength: f32,
        seed: i64,
        frames: usize,
        fps: usize,
        input_path: Option<String>,
        output_path: String,
    ) -> SdWorkerRequest {
        let is_video = operation.is_video();
        let is_svd = is_video && is_svd_model_path(&model_path);
        let uses_standalone_video_model = is_video && !is_svd;
        let (sampler, mut scheduler) = normalize_native_sampler(sampler);
        if is_svd && scheduler == "discrete" {
            // SVD-XT was trained with continuous EDM timesteps and Karras
            // sigmas. Keep the simple Web UI "Euler" choice on that valid
            // model-specific schedule instead of ordinary SD beta sigmas.
            scheduler = "karras".to_string();
        } else if uses_standalone_video_model {
            // Wan/LTX choose a scheduler from model metadata when none is set.
            scheduler.clear();
        }
        let params_backend = if uses_standalone_video_model
            && (self.config.offload_to_cpu
                || (self.config.video_stream_layers
                    && self.config.video_params_backend.trim().is_empty()))
        {
            "*=cpu".to_string()
        } else if uses_standalone_video_model {
            self.config.video_params_backend.clone()
        } else {
            String::new()
        };
        let model_is_gguf = is_gguf_path(&model_path);
        SdWorkerRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            operation,
            bridge_library_path: self.config.bridge_library_path.clone(),
            model_path: if uses_standalone_video_model {
                String::new()
            } else {
                model_path.clone()
            },
            diffusion_model_path: if uses_standalone_video_model {
                model_path
            } else {
                String::new()
            },
            high_noise_diffusion_model_path: if uses_standalone_video_model {
                self.config.video_high_noise_model_path.clone()
            } else {
                String::new()
            },
            clip_vision_path: if is_svd {
                self.config.clip_vision_path.clone()
            } else {
                String::new()
            },
            t5xxl_path: if uses_standalone_video_model {
                self.config.video_t5xxl_path.clone()
            } else {
                String::new()
            },
            vae_path: if uses_standalone_video_model {
                self.config.video_vae_path.clone()
            } else if model_is_gguf || !self.config.image_vae_path.is_empty() {
                self.config.image_vae_path.clone()
            } else {
                String::new()
            },
            backend: self.config.backend.clone(),
            params_backend,
            max_vram: if uses_standalone_video_model {
                self.config.video_max_vram.clone()
            } else {
                self.config.image_max_vram.clone()
            },
            weight_type: if uses_standalone_video_model || model_is_gguf {
                // GGUF 保留文件内张量类型,避免把量化模型展开成 f16 内存。
                String::new()
            } else {
                self.config.precision.clone()
            },
            rng_type: normalize_rng_mode(&self.config.rng_mode, &self.config.backend),
            threads: std::thread::available_parallelism()
                .map(|value| value.get() as i32)
                .unwrap_or(4),
            flash_attention: self.config.flash_attention,
            stream_layers: uses_standalone_video_model && self.config.video_stream_layers,
            prompt,
            negative_prompt,
            sampler,
            scheduler,
            width: width as u32,
            height: height as u32,
            output_width: width as u32,
            output_height: height as u32,
            steps: steps as i32,
            cfg,
            flow_shift: if uses_standalone_video_model {
                self.config.video_flow_shift
            } else {
                0.0
            },
            min_cfg: 1.0,
            noise_aug_strength: 0.02,
            strength,
            seed,
            frames: frames as i32,
            fps: fps as i32,
            motion_bucket_id: 127,
            input_path,
            output_path,
            loras: Vec::new(),
            hires: None,
        }
    }

    async fn run_native_worker(&self, request: SdWorkerRequest) -> Result<Vec<u8>, SdError> {
        if !std::path::Path::new(&self.config.worker_path).is_file() {
            return Err(SdError::ConfigurationError(format!(
                "Rust stable-diffusion worker not found: {}. Run scripts/build_native_runtime.sh",
                self.config.worker_path
            )));
        }
        if !std::path::Path::new(&self.config.bridge_library_path).is_file() {
            return Err(SdError::ConfigurationError(format!(
                "stable-diffusion.cpp bridge not found: {}. Run scripts/build_native_runtime.sh",
                self.config.bridge_library_path
            )));
        }

        let timeout_secs = if request.operation.is_video() {
            self.config.video_timeout_secs
        } else {
            self.config.timeout_secs
        };
        let serialized = serde_json::to_vec(&request)?;
        let output_path = request.output_path.clone();
        let _temporary_files = NativeTemporaryFiles {
            input: request.input_path.clone(),
            output: output_path.clone(),
        };
        let mut command = tokio::process::Command::new(&self.config.worker_path);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in &self.config.env_vars {
            command.env(key, value);
        }

        let mut child = command.spawn().map_err(|error| {
            SdError::ProcessStartFailed(format!(
                "Failed to start Rust stable-diffusion worker '{}': {}",
                self.config.worker_path, error
            ))
        })?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            SdError::CommunicationError("native worker stdin is unavailable".to_string())
        })?;
        stdin.write_all(&serialized).await.map_err(|error| {
            SdError::CommunicationError(format!("cannot send native worker request: {}", error))
        })?;
        stdin.shutdown().await.map_err(|error| {
            SdError::CommunicationError(format!("cannot close native worker request: {}", error))
        })?;
        drop(stdin);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| SdError::TimeoutError(Duration::from_secs(timeout_secs)))?
        .map_err(|error| {
            SdError::CommunicationError(format!("cannot collect native worker output: {}", error))
        })?;

        let response = serde_json::from_slice::<SdWorkerResponse>(&output.stdout).map_err(|error| {
            SdError::ExecutionFailed(format!(
                "native worker returned invalid JSON (exit={}): {} | {}",
                output.status.code().unwrap_or(-1),
                error,
                summarize_process_output(&output.stderr, &output.stdout)
            ))
        })?;
        if !output.status.success() || response.status != "success" {
            return Err(SdError::ExecutionFailed(
                response.error.unwrap_or_else(|| {
                    summarize_process_output(&output.stderr, &output.stdout)
                }),
            ));
        }

        let bytes = std::fs::read(&output_path).map_err(|error| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("native worker output '{}' is unavailable: {}", output_path, error),
            ))
        })?;
        if bytes.is_empty() {
            return Err(SdError::ExecutionFailed(
                "native worker produced an empty media file".to_string(),
            ));
        }
        Ok(bytes)
    }

    async fn run_sd_cli_text_to_image(&self, params: &T2IParams) -> Result<Vec<u8>, SdError> {
        if self.config.executable_path.is_empty() {
            return Err(SdError::ConfigurationError(
                "executable_path is not configured".to_string()
            ));
        }
        if self.config.model_path.is_empty() {
            return Err(SdError::ConfigurationError(
                "model_path is not configured".to_string()
            ));
        }
        let output_path = format!(
            "/tmp/sd_output_{}.png",
            uuid::Uuid::new_v4()
        );

        let executable = &self.config.executable_path;
        let model = &self.config.model_path;

        let mut last_error = String::new();

        for attempt in build_backend_attempts(&self.config.backend, &self.config.rng_mode) {
            let mut cmd = std::process::Command::new(executable);
            cmd.arg("--model").arg(model)
                .arg("--prompt").arg(&params.prompt)
                .arg("--output").arg(&output_path)
                .arg("--backend").arg(&attempt.backend)
                .arg("--rng").arg(&attempt.rng_mode)
                .arg("--steps").arg(params.steps.to_string())
                .arg("--cfg-scale").arg(params.cfg.to_string())
                .arg("--sampling-method").arg(&params.sampler)
                .arg("--width").arg(params.width.to_string())
                .arg("--height").arg(params.height.to_string())
                .arg("--seed").arg(params.seed.to_string());

            if !params.negative_prompt.is_empty() {
                cmd.arg("--negative-prompt").arg(&params.negative_prompt);
            }

            info!(
                "Running sd-cli: {} --model {} --backend {} --steps {} --sampling-method {} --width {}x{}",
                executable, model, attempt.backend, params.steps, params.sampler, params.width, params.height
            );

            let output = cmd.output().map_err(|e| {
                SdError::ProcessStartFailed(format!("Failed to run sd-cli: {}", e))
            })?;

            if output.status.success() {
                let image_data = std::fs::read(&output_path).map_err(|e| {
                    SdError::IoError(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Failed to read sd-cli output '{}': {}", output_path, e),
                    ))
                })?;

                let _ = std::fs::remove_file(&output_path);
                return Ok(image_data);
            }

            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            last_error = format!(
                "sd-cli failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                summarize_process_output(&output.stderr, &output.stdout)
            );

            if attempt.backend == "cpu" || !should_retry_with_cpu(&stderr) {
                break;
            }

            warn!(
                "sd-cli backend '{}' unavailable, retrying with CPU fallback",
                attempt.backend
            );
        }

        Err(SdError::ExecutionFailed(last_error))
    }

    /// 文生图
    pub async fn text_to_image(&self, params: T2IParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        if self.uses_native_worker() {
            let model_path = resolve_model_path(&params.model_path, &self.config.model_path)?;
            validate_native_model_file(&model_path, "diffusion model")?;
            let output_path = format!("/tmp/media_sd_{}.png", uuid::Uuid::new_v4());
            let loras = params.loras.clone();
            let hires = params.hires;
            let mut request = self.native_request(
                SdWorkerOperation::TextToImage,
                model_path,
                params.prompt,
                params.negative_prompt,
                params.width,
                params.height,
                params.steps,
                params.cfg,
                &params.sampler,
                1.0,
                params.seed as i64,
                0,
                0,
                None,
                output_path,
            );
            request.loras = loras
                .into_iter()
                .map(|lora| crate::backend::sd_worker_protocol::WorkerLora {
                    path: lora.path,
                    multiplier: lora.multiplier,
                })
                .collect();
            if let Some(hires) = hires {
                request.hires = Some(crate::backend::sd_worker_protocol::WorkerHires {
                    scale: hires.scale,
                    steps: hires.steps as i32,
                    denoising_strength: hires.denoising_strength,
                });
            }
            return self.run_native_worker(request).await;
        }

        self.run_sd_cli_text_to_image(&params).await
    }

    async fn run_sd_cli_image_to_image(&self, params: &I2IParams) -> Result<Vec<u8>, SdError> {
        if self.config.executable_path.is_empty() {
            return Err(SdError::ConfigurationError(
                "executable_path is not configured".to_string()
            ));
        }
        if self.config.model_path.is_empty() {
            return Err(SdError::ConfigurationError(
                "model_path is not configured".to_string()
            ));
        }
        let output_path = format!(
            "/tmp/sd_output_{}.png",
            uuid::Uuid::new_v4()
        );

        // Normalize uploaded JPEG/WebP/PNG bytes to a real PNG before passing a
        // path to native decoders. Merely renaming JPEG bytes to .png makes the
        // decoder select the wrong codec.
        let input_path = write_temporary_input_png("sd_input", &params.input_image)?;

        let executable = &self.config.executable_path;
        let model = &self.config.model_path;

        let mut last_error = String::new();

        for attempt in build_backend_attempts(&self.config.backend, &self.config.rng_mode) {
            let mut cmd = std::process::Command::new(executable);
            cmd.arg("--model").arg(model)
                .arg("--prompt").arg(&params.prompt)
                .arg("--image").arg(&input_path)
                .arg("--output").arg(&output_path)
                .arg("--backend").arg(&attempt.backend)
                .arg("--rng").arg(&attempt.rng_mode)
                .arg("--steps").arg(params.steps.to_string())
                .arg("--cfg-scale").arg(params.cfg.to_string())
                .arg("--sampling-method").arg(&params.sampler)
                .arg("--strength").arg(params.denoise.to_string())
                .arg("--width").arg(params.width.to_string())
                .arg("--height").arg(params.height.to_string())
                .arg("--seed").arg(params.seed.to_string());

            if !params.negative_prompt.is_empty() {
                cmd.arg("--negative-prompt").arg(&params.negative_prompt);
            }

            info!(
                "Running sd-cli img2img: {} --backend {} --steps {} --sampling-method {} --denoise {} --width {}x{}",
                executable, attempt.backend, params.steps, params.sampler, params.denoise, params.width, params.height
            );

            let output = cmd.output().map_err(|e| {
                SdError::ProcessStartFailed(format!("Failed to run sd-cli: {}", e))
            })?;

            if output.status.success() {
                let image_data = std::fs::read(&output_path).map_err(|e| {
                    SdError::IoError(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Failed to read sd-cli output '{}': {}", output_path, e),
                    ))
                })?;

                let _ = std::fs::remove_file(&input_path);
                let _ = std::fs::remove_file(&output_path);
                return Ok(image_data);
            }

            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            last_error = format!(
                "sd-cli img2img failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                summarize_process_output(&output.stderr, &output.stdout)
            );

            if attempt.backend == "cpu" || !should_retry_with_cpu(&stderr) {
                break;
            }

            warn!(
                "sd-cli img2img backend '{}' unavailable, retrying with CPU fallback",
                attempt.backend
            );
        }

        let _ = std::fs::remove_file(&input_path);
        Err(SdError::ExecutionFailed(last_error))
    }

    /// 图生图
    pub async fn image_to_image(&self, params: I2IParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        if self.uses_native_worker() {
            let model_path = resolve_model_path(&params.model_path, &self.config.model_path)?;
            validate_native_model_file(&model_path, "diffusion model")?;
            let input_path =
                write_temporary_input_png("media_sd_input", &params.input_image)?;
            let output_path = format!("/tmp/media_sd_{}.png", uuid::Uuid::new_v4());
            let loras = params.loras.clone();
            let mut request = self.native_request(
                SdWorkerOperation::ImageToImage,
                model_path,
                params.prompt,
                params.negative_prompt,
                params.width,
                params.height,
                params.steps,
                params.cfg,
                &params.sampler,
                params.denoise,
                params.seed as i64,
                0,
                0,
                Some(input_path),
                output_path,
            );
            request.loras = loras
                .into_iter()
                .map(|lora| crate::backend::sd_worker_protocol::WorkerLora {
                    path: lora.path,
                    multiplier: lora.multiplier,
                })
                .collect();
            return self.run_native_worker(request).await;
        }

        self.run_sd_cli_image_to_image(&params).await
    }

    /// 文生视频
    pub async fn text_to_video(&self, params: T2VParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        if self.uses_native_worker() {
            let model_path = resolve_model_path(&params.model_path, &self.config.video_model_path)?;
            validate_native_model_file(&model_path, "video diffusion model")?;
            validate_native_video_assets(&self.config, &model_path)?;

            if is_svd_model_path(&model_path) {
                let image_model = resolve_model_path("", &self.config.model_path)?;
                validate_native_model_file(&image_model, "text-to-video first-frame model")?;
                let first_frame_output = format!(
                    "/tmp/media_sd_t2v_first_{}.png",
                    uuid::Uuid::new_v4()
                );
                let first_frame_request = self.native_request(
                    SdWorkerOperation::TextToImage,
                    image_model,
                    params.prompt.clone(),
                    params.negative_prompt.clone(),
                    self.config.svd_native_width,
                    self.config.svd_native_height,
                    params.steps,
                    7.0,
                    "dpm++2m_karras",
                    1.0,
                    params.seed as i64,
                    0,
                    0,
                    None,
                    first_frame_output,
                );
                info!(
                    "Native text-to-video composition: generating {}x{} first frame",
                    self.config.svd_native_width,
                    self.config.svd_native_height
                );
                let first_frame = self.run_native_worker(first_frame_request).await?;
                let input_path =
                    write_temporary_input_png("media_sd_t2v_input", &first_frame)?;

                let output_path = format!("/tmp/media_sd_{}.mp4", uuid::Uuid::new_v4());
                let mut video_request = self.native_request(
                    SdWorkerOperation::ImageToVideo,
                    model_path,
                    params.prompt,
                    params.negative_prompt,
                    self.config.svd_native_width,
                    self.config.svd_native_height,
                    params.steps,
                    params.cfg,
                    "euler",
                    1.0,
                    params.seed as i64,
                    params.frames,
                    params.fps,
                    Some(input_path),
                    output_path,
                );
                video_request.output_width = params.width as u32;
                video_request.output_height = params.height as u32;
                video_request.motion_bucket_id = params.motion_bucket_id;
                video_request.min_cfg = params.min_cfg;
                video_request.noise_aug_strength = params.noise_aug_strength;
                return self.run_native_worker(video_request).await;
            }

            let output_path = format!("/tmp/media_sd_{}.mp4", uuid::Uuid::new_v4());
            let (generation_width, generation_height) = semantic_video_dimensions(
                &self.config,
                params.width,
                params.height,
            );
            let mut request = self.native_request(
                SdWorkerOperation::TextToVideo,
                model_path,
                params.prompt,
                params.negative_prompt,
                generation_width,
                generation_height,
                params.steps,
                params.cfg,
                "euler",
                1.0,
                params.seed as i64,
                params.frames,
                params.fps,
                None,
                output_path,
            );
            request.output_width = params.width as u32;
            request.output_height = params.height as u32;
            return self.run_native_worker(request).await;
        }

        let request = SdRequest {
            mode: "text_to_video".to_string(),
            prompt: params.prompt,
            negative_prompt: params.negative_prompt,
            width: params.width,
            height: params.height,
            steps: params.steps,
            cfg: params.cfg,
            sampler: "euler".to_string(),
            seed: params.seed,
            model_path: params.model_path,
            input_image: None,
            controlnet: None,
            denoise: None,
            request_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        let mut pm = self.process_manager.lock().await;
        let response = pm.execute_request(&request)?;

        if !response.is_success() {
            return Err(SdError::ExecutionFailed(
                response.error.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        let video_data = std::fs::read(&response.output_path)?;
        Ok(video_data)
    }

    /// 图生视频 - 直接子进程调用（sd-cli 不支持 stdin/stdout 协议）
    async fn run_sd_cli_image_to_video(&self, params: &I2VParams) -> Result<Vec<u8>, SdError> {
        if self.config.executable_path.is_empty() {
            return Err(SdError::ConfigurationError(
                "executable_path is not configured".to_string()
            ));
        }

        // 使用 params.model_path（SVD 模型路径），而非 self.config.model_path
        let model_path = if !params.model_path.is_empty() {
            if std::path::Path::new(&params.model_path).exists() {
                params.model_path.clone()
            } else {
                // 尝试在 models/checkpoints 下查找
                let candidate = format!("models/checkpoints/{}", params.model_path);
                if std::path::Path::new(&candidate).exists() {
                    candidate
                } else {
                    params.model_path.clone()
                }
            }
        } else {
            return Err(SdError::ConfigurationError(
                "params.model_path is empty - SVD model path required".to_string()
            ));
        };

        validate_native_model_file(&model_path, "SVD checkpoint")?;

        // 输出路径（视频文件）
        let output_path = format!("/tmp/svd_output_{}.mp4", uuid::Uuid::new_v4());

        let input_path = write_temporary_input_png("svd_input", &params.input_image)?;

        let executable = &self.config.executable_path;

        let clip_vision_path = if !self.config.clip_vision_path.trim().is_empty() {
            Some(self.config.clip_vision_path.clone())
        } else {
            [
                "models/clip_vision/clip_vit_h_14.safetensors",
                "models/clip_vision/clip-vit-h-14.safetensors",
                "models/clip_vision/clip_vit_h14.safetensors",
            ]
            .iter()
            .find(|path| std::path::Path::new(path).is_file())
            .map(|path| (*path).to_string())
        }
        .ok_or_else(|| {
            SdError::ConfigurationError(
                "SVD requires a CLIP ViT-H/14 vision model; configure sd_cpp.clip_vision_path"
                    .to_string(),
            )
        })?;
        validate_native_model_file(&clip_vision_path, "CLIP Vision model")?;
        info!("Using clip_vision: {}", clip_vision_path);

        let mut last_error = String::new();

        for attempt in build_backend_attempts(&self.config.backend, &self.config.rng_mode) {
            let mut cmd = std::process::Command::new(executable);
            cmd.arg("-M").arg("vid_gen")
                .arg("-m").arg(&model_path)
                .arg("--image").arg(&input_path)
                .arg("-o").arg(&output_path)
                .arg("--backend").arg(&attempt.backend)
                .arg("--rng").arg(&attempt.rng_mode)
                .arg("--steps").arg(params.steps.to_string())
                .arg("--cfg-scale").arg(params.cfg.to_string())
                .arg("--seed").arg(params.seed.to_string())
                .arg("--video-frames").arg(params.frames.to_string())
                .arg("--fps").arg(params.fps.to_string());

            cmd.arg("--clip_vision").arg(&clip_vision_path);

            if !params.negative_prompt.is_empty() {
                cmd.arg("--negative-prompt").arg(&params.negative_prompt);
            }

            info!(
                "Running sd-cli SVD: {} -M vid_gen -m {} --backend {} --image {} --video-frames {} --fps {}",
                executable, model_path, attempt.backend, input_path, params.frames, params.fps
            );

            let output = cmd.output().map_err(|e| {
                SdError::ProcessStartFailed(format!("Failed to run sd-cli SVD: {}", e))
            })?;

            if output.status.success() {
                let _ = std::fs::remove_file(&input_path);
                return self.read_video_output_or_frames(&output_path, params);
            }

            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            last_error = format!(
                "sd-cli SVD failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                summarize_process_output(&output.stderr, &output.stdout)
            );

            if attempt.backend == "cpu" || !should_retry_with_cpu(&stderr) {
                break;
            }

            warn!(
                "sd-cli SVD backend '{}' unavailable, retrying with CPU fallback",
                attempt.backend
            );
        }

        let _ = std::fs::remove_file(&input_path);
        Err(SdError::ExecutionFailed(last_error))
    }

    fn read_video_output_or_frames(&self, output_path: &str, params: &I2VParams) -> Result<Vec<u8>, SdError> {
        let output_path_obj = std::path::Path::new(output_path);
        if !output_path_obj.exists() {
            // sd-cli 可能输出 PNG 帧序列而非 MP4（未编译 ffmpeg 支持时）
            let output_dir = std::path::Path::new(output_path).parent().unwrap_or(std::path::Path::new("/tmp"));
            let stem = std::path::Path::new(output_path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let mut frames: Vec<String> = Vec::new();
            for entry in std::fs::read_dir(output_dir).into_iter().flatten() {
                if let Ok(entry) = entry {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(&stem) && (name.ends_with(".png") || name.ends_with(".jpg")) {
                        frames.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }
            if !frames.is_empty() {
                frames.sort();
                info!("SVD generated {} PNG frame files, combining into MP4 via ffmpeg", frames.len());

                let glob_pattern = format!("{}/{}*.png", output_dir.to_string_lossy(), stem);

                let ffmpeg_result = std::process::Command::new("ffmpeg")
                    .arg("-y")
                    .arg("-framerate").arg(params.fps.to_string())
                    .arg("-pattern_type").arg("glob")
                    .arg("-i").arg(&glob_pattern)
                    .arg("-c:v").arg("libx264")
                    .arg("-pix_fmt").arg("yuv420p")
                    .arg("-r").arg(params.fps.to_string())
                    .arg(output_path)
                    .output();

                let ffmpeg_ok = match &ffmpeg_result {
                    Ok(out) if out.status.success() => {
                        info!("ffmpeg combined {} frames into MP4: {}", frames.len(), output_path);
                        true
                    }
                    Ok(out) => {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        warn!("ffmpeg failed (exit={:?}): {}", out.status.code(), stderr.lines().next().unwrap_or(""));
                        false
                    }
                    Err(e) => {
                        warn!("ffmpeg not found: {}", e);
                        false
                    }
                };

                // 清理帧文件
                for f in &frames {
                    let _ = std::fs::remove_file(f);
                }

                if ffmpeg_ok && std::path::Path::new(output_path).exists() {
                    let video_data = std::fs::read(output_path).map_err(|e| {
                        SdError::IoError(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("Failed to read ffmpeg output '{}': {}", output_path, e),
                        ))
                    })?;
                    let _ = std::fs::remove_file(output_path);
                    return Ok(video_data);
                }

                return Err(SdError::ExecutionFailed(format!(
                    "sd-cli output {} PNG frames but ffmpeg failed to combine them. Install ffmpeg to enable video output.",
                    frames.len()
                )));
            }

            return Err(SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("SVD output file not found: {}", output_path),
            )));
        }

        let video_data = std::fs::read(output_path).map_err(|e| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to read SVD output '{}': {}", output_path, e),
            ))
        })?;

        // 清理输出临时文件
        let _ = std::fs::remove_file(output_path);

        Ok(video_data)
    }

    /// 图生视频（SVD）
    pub async fn image_to_video(&self, params: I2VParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        if self.uses_native_worker() {
            let model_path = resolve_model_path(&params.model_path, &self.config.video_model_path)?;
            validate_native_model_file(&model_path, "video diffusion model")?;
            validate_native_video_assets(&self.config, &model_path)?;
            let input_path =
                write_temporary_input_png("media_sd_input", &params.input_image)?;
            let output_path = format!("/tmp/media_sd_{}.mp4", uuid::Uuid::new_v4());
            let (generation_width, generation_height) = if is_svd_model_path(&model_path) {
                (self.config.svd_native_width, self.config.svd_native_height)
            } else {
                semantic_video_dimensions(&self.config, params.width, params.height)
            };
            let mut request = self.native_request(
                SdWorkerOperation::ImageToVideo,
                model_path,
                params.prompt,
                params.negative_prompt,
                generation_width,
                generation_height,
                params.steps,
                params.cfg,
                "euler",
                1.0,
                params.seed as i64,
                params.frames,
                params.fps,
                Some(input_path),
                output_path,
            );
            request.output_width = params.width as u32;
            request.output_height = params.height as u32;
            request.motion_bucket_id = params.motion_bucket_id;
            request.min_cfg = params.min_cfg;
            request.noise_aug_strength = params.noise_aug_strength;
            return self.run_native_worker(request).await;
        }

        self.run_sd_cli_image_to_video(&params).await
    }

    /// 启动后端
    pub async fn start(&self) -> Result<(), SdError> {
        let mut pm = self.process_manager.lock().await;
        pm.start()
    }

    /// 停止后端
    pub async fn stop(&self) -> Result<(), SdError> {
        let mut pm = self.process_manager.lock().await;
        pm.stop()
    }

    pub async fn health_check(&self) -> Result<bool, SdError> {
        if self.uses_native_worker() {
            return Ok(
                std::path::Path::new(&self.config.worker_path).is_file()
                    && std::path::Path::new(&self.config.bridge_library_path).is_file()
            );
        }
        if self.config.executable_path.is_empty() {
            return Ok(false);
        }
        let exe = self.config.executable_path.clone();
        let result = tokio::task::spawn_blocking(move || {
            std::process::Command::new(&exe)
                .arg("--help")
                .output()
        })
        .await
        .map_err(|e| SdError::ProcessStartFailed(format!("Health check join: {}", e)))?
        .map_err(|e| SdError::ProcessStartFailed(format!("Health check failed: {}", e)))?;
        Ok(result.status.success())
    }

    /// 获取统计信息
    pub async fn get_stats(&self) -> SdProcessStats {
        let pm = self.process_manager.lock().await;
        pm.get_stats()
    }

    /// 释放显存
    pub async fn free_memory(&self) -> Result<(), SdError> {
        let mut pm = self.process_manager.lock().await;
        pm.stop()?;
        Ok(())
    }
}

fn validate_native_model_file(path: &str, role: &str) -> Result<(), SdError> {
    let info = crate::native_runtime::inspect_model_file(path)
        .map_err(|error| SdError::ConfigurationError(format!("{}: {}", role, error)))?;
    match info.container {
        crate::native_runtime::ModelContainer::Safetensors
        | crate::native_runtime::ModelContainer::Gguf => Ok(()),
        crate::native_runtime::ModelContainer::TorchZip => Err(SdError::ConfigurationError(
            format!(
                "{} '{}' is a PyTorch ZIP archive renamed as a model file; native inference requires Safetensors or GGUF",
                role, path
            ),
        )),
        crate::native_runtime::ModelContainer::Unknown => Err(SdError::ConfigurationError(
            format!("{} '{}' has an unsupported or corrupt container", role, path),
        )),
    }
}

fn normalize_native_sampler(value: &str) -> (String, String) {
    let normalized = value.trim().to_ascii_lowercase();
    let scheduler = if normalized.contains("karras") {
        "karras"
    } else {
        "discrete"
    };
    let sampler = normalized
        .trim_end_matches("_karras")
        .trim_end_matches(" karras")
        .replace("dpmpp", "dpm++");
    (sampler, scheduler.to_string())
}

fn is_svd_model_path(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.contains("svd") || path.contains("stable-video-diffusion")
}

fn is_gguf_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
}

fn validate_native_video_assets(config: &SdCppConfig, model_path: &str) -> Result<(), SdError> {
    if is_svd_model_path(model_path) {
        if config.clip_vision_path.trim().is_empty() {
            return Err(SdError::ConfigurationError(
                "SVD image-to-video requires sd_cpp.clip_vision_path".to_string(),
            ));
        }
        return validate_native_model_file(&config.clip_vision_path, "CLIP Vision model");
    }

    if config.video_t5xxl_path.trim().is_empty() {
        return Err(SdError::ConfigurationError(
            "prompt-conditioned video requires sd_cpp.video_t5xxl_path".to_string(),
        ));
    }
    if config.video_vae_path.trim().is_empty() {
        return Err(SdError::ConfigurationError(
            "prompt-conditioned video requires sd_cpp.video_vae_path".to_string(),
        ));
    }
    validate_native_model_file(&config.video_t5xxl_path, "video T5 text encoder")?;
    validate_native_model_file(&config.video_vae_path, "video VAE")?;
    if !config.video_high_noise_model_path.trim().is_empty() {
        validate_native_model_file(
            &config.video_high_noise_model_path,
            "high-noise video diffusion model",
        )?;
    }
    Ok(())
}

fn write_temporary_input_png(prefix: &str, bytes: &[u8]) -> Result<String, SdError> {
    let decoded = image::load_from_memory(bytes).map_err(|error| {
        SdError::ExecutionFailed(format!("cannot decode input image bytes: {}", error))
    })?;
    let path = std::env::temp_dir().join(format!("{}_{}.png", prefix, uuid::Uuid::new_v4()));
    decoded
        .to_rgb8()
        .save_with_format(&path, image::ImageFormat::Png)
        .map_err(|error| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("cannot encode temporary PNG '{}': {}", path.display(), error),
            ))
        })?;
    Ok(path.to_string_lossy().into_owned())
}

fn semantic_video_dimensions(
    config: &SdCppConfig,
    delivery_width: usize,
    delivery_height: usize,
) -> (usize, usize) {
    let base_area = config
        .semantic_video_native_width
        .saturating_mul(config.semantic_video_native_height)
        .max(256 * 256) as f64;
    let aspect = delivery_width.max(1) as f64 / delivery_height.max(1) as f64;
    let width = (base_area * aspect).sqrt();
    let height = base_area / width;
    (align_video_dimension(width), align_video_dimension(height))
}

fn align_video_dimension(value: f64) -> usize {
    (((value / 16.0).round() as usize).max(16) * 16).clamp(256, 1280)
}

fn resolve_model_path(requested: &str, configured: &str) -> Result<String, SdError> {
    let value = if requested.trim().is_empty() {
        configured
    } else {
        requested
    };
    if value.trim().is_empty() {
        return Err(SdError::ConfigurationError(
            "model path is not configured".to_string(),
        ));
    }
    if std::path::Path::new(value).is_file() {
        return Ok(value.to_string());
    }
    let candidate = std::path::Path::new("models/checkpoints").join(value);
    if candidate.is_file() {
        return Ok(candidate.to_string_lossy().into_owned());
    }
    Err(SdError::ConfigurationError(format!(
        "model file not found: {}",
        value
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SdCppConfig::default();
        assert_eq!(config.backend, "cuda");
        assert_eq!(config.precision, "f16");
        assert!(config.flash_attention);
        assert_eq!(config.rng_mode, "cpu");
        assert_eq!(config.timeout_secs, 300);
        assert_eq!(config.video_timeout_secs, 1800);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_build_args() {
        let config = SdCppConfig {
            model_path: "/models/sd15.safetensors".to_string(),
            backend: "vulkan".to_string(),
            precision: "q4_0".to_string(),
            flash_attention: false,
            offload_to_cpu: true,
            ..Default::default()
        };

        let args = config.build_args();
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"/models/sd15.safetensors".to_string()));
        assert!(args.contains(&"--backend".to_string()));
        assert!(args.contains(&"vulkan".to_string()));
        assert!(args.contains(&"--offload-to-cpu".to_string()));
        assert!(!args.contains(&"--diffusion-fa".to_string()));
    }

    #[test]
    fn test_circuit_breaker() {
        let mut cb = CircuitBreaker::new(3, 60);

        // 初始状态允许请求
        assert!(cb.is_allowed());

        // 记录2次失败，仍未打开
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_allowed());

        // 第3次失败后打开
        cb.record_failure();
        assert!(!cb.is_allowed());
        assert!(cb.is_open());

        // 记录成功后重置
        cb.record_success();
        assert!(cb.is_allowed());
        assert!(!cb.is_open());
        assert_eq!(cb.get_failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_reset_after_timeout() {
        // 使用0秒重置时间意味着立即重置
        let mut cb = CircuitBreaker::new(1, 0);
        cb.record_failure();
        // 由于 reset_duration = 0，断路器立即重置
        // 所以 is_allowed 应该返回 true（因为elapsed >= 0）
        assert!(cb.is_allowed());
    }

    #[test]
    fn test_normalize_native_sampler() {
        assert_eq!(
            normalize_native_sampler("dpmpp2m_karras"),
            ("dpm++2m".to_string(), "karras".to_string())
        );
        assert_eq!(
            normalize_native_sampler("euler"),
            ("euler".to_string(), "discrete".to_string())
        );
    }

    #[test]
    fn test_svd_model_detection() {
        assert!(is_svd_model_path("models/checkpoints/svd_xt.safetensors"));
        assert!(is_svd_model_path(
            "models/diffusers/stable-video-diffusion-img2vid-xt/model.safetensors"
        ));
        assert!(!is_svd_model_path("models/checkpoints/sdxl_base.safetensors"));
    }

    #[test]
    fn semantic_video_dimensions_preserve_delivery_orientation() {
        let config = SdCppConfig::default();
        assert_eq!(semantic_video_dimensions(&config, 1366, 780), (832, 480));
        assert_eq!(semantic_video_dimensions(&config, 400, 500), (560, 704));
    }

    #[test]
    fn image_gguf_request_preserves_quantization_and_uses_image_assets() {
        let config = SdCppConfig {
            image_vae_path: "/models/sdxl_vae.safetensors".to_string(),
            image_max_vram: "-1".to_string(),
            ..Default::default()
        };
        let backend = StableDiffusionCppBackend::new(config);
        let request = backend.native_request(
            SdWorkerOperation::TextToImage,
            "/models/sdxl_base_1.0_Q4_K_S.gguf".to_string(),
            "a cat".to_string(),
            String::new(),
            768,
            768,
            26,
            6.0,
            "dpm++2m_karras",
            1.0,
            42,
            0,
            0,
            None,
            "/tmp/out.png".to_string(),
        );
        assert_eq!(request.vae_path, "/models/sdxl_vae.safetensors");
        assert_eq!(request.max_vram, "-1");
        assert!(request.weight_type.is_empty(), "GGUF must preserve tensor types");
    }

    #[test]
    fn image_request_passes_loras_and_hires() {
        let config = SdCppConfig::default();
        let backend = StableDiffusionCppBackend::new(config);
        let mut request = backend.native_request(
            SdWorkerOperation::TextToImage,
            "/models/sd15.safetensors".to_string(),
            "a cat".to_string(),
            String::new(),
            512,
            512,
            20,
            7.0,
            "dpm++2m_karras",
            1.0,
            42,
            0,
            0,
            None,
            "/tmp/out.png".to_string(),
        );
        request.loras = vec![crate::backend::sd_worker_protocol::WorkerLora {
            path: "/models/epic_realism.safetensors".to_string(),
            multiplier: 0.7,
        }];
        request.hires = Some(crate::backend::sd_worker_protocol::WorkerHires {
            scale: 1.5,
            steps: 16,
            denoising_strength: 0.34,
        });
        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("epic_realism"));
        assert!(serialized.contains("\"scale\":1.5"));
        let parsed: crate::backend::sd_worker_protocol::SdWorkerRequest =
            serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed.loras.len(), 1);
        assert_eq!(parsed.loras[0].multiplier, 0.7);
        let hires = parsed.hires.unwrap();
        assert_eq!(hires.scale, 1.5);
        assert_eq!(hires.steps, 16);
    }

    #[test]
    fn wan_worker_request_uses_standalone_native_assets() {
        let config = SdCppConfig {
            video_t5xxl_path: "/models/umt5.gguf".to_string(),
            video_vae_path: "/models/wan.vae.safetensors".to_string(),
            video_flow_shift: 3.0,
            ..Default::default()
        };
        let backend = StableDiffusionCppBackend::new(config);
        let request = backend.native_request(
            SdWorkerOperation::TextToVideo,
            "/models/wan.gguf".to_string(),
            "a dancer".to_string(),
            String::new(),
            832,
            480,
            20,
            6.0,
            "euler",
            1.0,
            42,
            25,
            5,
            None,
            "/tmp/wan.mp4".to_string(),
        );
        assert!(request.model_path.is_empty());
        assert_eq!(request.diffusion_model_path, "/models/wan.gguf");
        assert_eq!(request.t5xxl_path, "/models/umt5.gguf");
        assert_eq!(request.vae_path, "/models/wan.vae.safetensors");
        assert!(request.weight_type.is_empty());
        assert!(request.scheduler.is_empty());
        assert_eq!(request.flow_shift, 3.0);
    }

    #[test]
    fn temporary_input_normalizes_jpeg_bytes_to_png() {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            3,
            image::Rgb([12, 34, 56]),
        ));
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();

        let path = write_temporary_input_png("media-agent-test", jpeg.get_ref()).unwrap();
        let encoded = std::fs::read(&path).unwrap();
        assert_eq!(image::guess_format(&encoded).unwrap(), image::ImageFormat::Png);
        assert_eq!(image::image_dimensions(&path).unwrap(), (4, 3));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn temporary_input_rejects_invalid_image_bytes() {
        let error = write_temporary_input_png("media-agent-test", b"not an image").unwrap_err();
        assert!(error.to_string().contains("cannot decode input image bytes"));
    }

    #[test]
    fn test_build_backend_attempts_adds_cpu_fallback() {
        let attempts = build_backend_attempts("cuda", "auto");
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].backend, "cuda");
        assert_eq!(attempts[0].rng_mode, "cuda");
        assert_eq!(attempts[1].backend, "cpu");
        assert_eq!(attempts[1].rng_mode, "cpu");
    }

    #[test]
    fn test_build_backend_attempts_keeps_cpu_only() {
        let attempts = build_backend_attempts("cpu", "cpu");
        assert_eq!(attempts, vec![BackendAttempt {
            backend: "cpu".to_string(),
            rng_mode: "cpu".to_string(),
        }]);
    }

    #[test]
    fn test_should_retry_with_cpu_detects_cuda_runtime_errors() {
        assert!(should_retry_with_cpu(
            "ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version"
        ));
        assert!(should_retry_with_cpu(
            "backend config failed: backend 'cuda' was not found"
        ));
        assert!(!should_retry_with_cpu("failed to parse torch zip pickle metadata"));
    }

    #[test]
    fn test_summarize_process_output_prefers_multiline_detail() {
        let summary = summarize_process_output(
            b"line 1\nline 2\n",
            b"stdout line\n",
        );
        assert!(summary.contains("line 1"));
        assert!(summary.contains("line 2"));
        assert!(summary.contains("stdout line"));
    }

    #[test]
    fn test_process_status_display() {
        assert_eq!(ProcessStatus::NotStarted.to_string(), "NotStarted");
        assert_eq!(ProcessStatus::Running.to_string(), "Running");
        assert_eq!(
            ProcessStatus::Error("test error".to_string()).to_string(),
            "Error: test error"
        );
    }

    #[test]
    fn test_sd_response_is_success() {
        let success = SdResponse {
            status: "success".to_string(),
            output_path: "/tmp/out.png".to_string(),
            seed: 42,
            time: 1.5,
            error: None,
            request_id: None,
        };
        assert!(success.is_success());

        let failure = SdResponse {
            status: "error".to_string(),
            output_path: String::new(),
            seed: 0,
            time: 0.0,
            error: Some("something went wrong".to_string()),
            request_id: None,
        };
        assert!(!failure.is_success());
    }

    #[test]
    fn test_config_from_env() {
        // 设置环境变量
        std::env::set_var("SD_CPP_MODEL_PATH", "/test/model.safetensors");
        std::env::set_var("SD_CPP_TIMEOUT_SECS", "600");
        std::env::set_var("SD_CPP_MAX_RETRIES", "5");

        let config = SdCppConfig::from_env();
        assert_eq!(config.model_path, "/test/model.safetensors");
        assert_eq!(config.timeout_secs, 600);
        assert_eq!(config.max_retries, 5);

        // 清理
        std::env::remove_var("SD_CPP_MODEL_PATH");
        std::env::remove_var("SD_CPP_TIMEOUT_SECS");
        std::env::remove_var("SD_CPP_MAX_RETRIES");
    }

    #[test]
    fn test_request_serialization() {
        let request = SdRequest {
            mode: "text_to_image".to_string(),
            prompt: "a cat".to_string(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg: 7.0,
            sampler: "euler".to_string(),
            seed: 42,
            model_path: "/models/sd15.safetensors".to_string(),
            input_image: None,
            controlnet: None,
            denoise: None,
            request_id: Some("test-123".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"mode\":\"text_to_image\""));
        assert!(json.contains("\"prompt\":\"a cat\""));
        assert!(!json.contains("negative_prompt")); // 空字符串应被跳过

        // 反序列化验证
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["mode"], "text_to_image");
        assert_eq!(parsed["width"], 512);
    }
}
