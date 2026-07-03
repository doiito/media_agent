// 后端集成层模块

pub mod sd_cpp;
pub mod llama_cpp;
pub mod router;
pub mod multi_backend;

pub use router::BackendRouter;
pub use sd_cpp::{
    SdCppConfig, SdError, SdRequest, SdResponse, SdProcessStats,
    ProcessStatus, CircuitBreaker, StableDiffusionCppBackend, SdCppProcessManager,
};
pub use llama_cpp::{LlamaCppBackend, LlamaCppConfig, LlamaError, LlamaRequest, LlamaResponse, LlamaProcessStats};
pub use multi_backend::{
    InferenceBackend, LocalProcessor, BackendPool, BackendFactory,
    BackendOperation, BackendStats, BackendInfo,
    FailoverStrategy,
};

use crate::types::*;

/// 后端类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendType {
    StableDiffusionCpp,
    LlamaCpp,
    LocalProcessor,
    /// ONNX Runtime（轻量级，可选）
    OnnxRuntime,
}

/// 文生图参数
#[derive(Debug, Clone)]
pub struct T2IParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg: f32,
    pub sampler: String,
    pub seed: usize,
    pub model_path: String,
}

impl Default for T2IParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg: 7.0,
            sampler: "euler".to_string(),
            seed: 0,
            model_path: String::new(),
        }
    }
}

/// 图生图参数
#[derive(Debug, Clone)]
pub struct I2IParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub input_image: Vec<u8>,
    pub denoise: f32,
    pub steps: usize,
    pub cfg: f32,
    pub seed: usize,
    pub model_path: String,
}

impl Default for I2IParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            input_image: Vec::new(),
            denoise: 0.75,
            steps: 20,
            cfg: 7.0,
            seed: 0,
            model_path: String::new(),
        }
    }
}

/// 文生视频参数
#[derive(Debug, Clone)]
pub struct T2VParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub steps: usize,
    pub cfg: f32,
    pub seed: usize,
    pub model_path: String,
}

impl Default for T2VParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            frames: 16,
            fps: 8,
            steps: 20,
            cfg: 7.0,
            seed: 0,
            model_path: String::new(),
        }
    }
}
