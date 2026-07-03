// llama.cpp 集成实现
// 用于LLM文本编码器（T5/Qwen/Llama等）

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use log::{debug, error, info, warn};

/// llama.cpp 错误类型
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
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
}

impl From<LlamaError> for Error {
    fn from(e: LlamaError) -> Self {
        Error::BackendError(e.to_string())
    }
}

/// llama.cpp 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    #[serde(default = "default_executable_path")]
    pub executable_path: String,

    #[serde(default)]
    pub model_path: String,

    /// 计算后端
    #[serde(default = "default_backend")]
    pub backend: String,

    /// 上下文长度
    #[serde(default = "default_n_ctx")]
    pub n_ctx: usize,

    /// 线程数
    #[serde(default = "default_n_threads")]
    pub n_threads: usize,

    /// GPU层数 (-1为全部)
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: i32,

    /// 批处理大小
    #[serde(default = "default_n_batch")]
    pub n_batch: usize,

    /// 是否使用Flash Attention
    #[serde(default = "default_flash_attn")]
    pub flash_attn: bool,

    /// 超时时间（秒）
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// 并发任务限制
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_executable_path() -> String {
    std::env::var("LLAMA_CPP_EXECUTABLE")
        .unwrap_or_else(|_| "llama-cli".to_string())
}

fn default_backend() -> String {
    "cuda".to_string()
}

fn default_n_ctx() -> usize {
    512
}

fn default_n_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn default_n_gpu_layers() -> i32 {
    -1
}

fn default_n_batch() -> usize {
    512
}

fn default_flash_attn() -> bool {
    true
}

fn default_timeout_secs() -> u64 {
    60
}

fn default_max_retries() -> usize {
    2
}

fn default_max_concurrent() -> usize {
    2
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            executable_path: default_executable_path(),
            model_path: String::new(),
            backend: default_backend(),
            n_ctx: default_n_ctx(),
            n_threads: default_n_threads(),
            n_gpu_layers: default_n_gpu_layers(),
            n_batch: default_n_batch(),
            flash_attn: default_flash_attn(),
            timeout_secs: default_timeout_secs(),
            max_retries: default_max_retries(),
            max_concurrent: default_max_concurrent(),
            extra_args: Vec::new(),
        }
    }
}

impl LlamaCppConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(val) = std::env::var("LLAMA_CPP_MODEL_PATH") {
            config.model_path = val;
        }
        if let Ok(val) = std::env::var("LLAMA_CPP_N_CTX") {
            if let Ok(ctx) = val.parse() {
                config.n_ctx = ctx;
            }
        }
        if let Ok(val) = std::env::var("LLAMA_CPP_N_THREADS") {
            if let Ok(threads) = val.parse() {
                config.n_threads = threads;
            }
        }
        config
    }

    pub fn build_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        if !self.model_path.is_empty() {
            args.push("--model".to_string());
            args.push(self.model_path.clone());
        }

        args.push("--ctx-size".to_string());
        args.push(self.n_ctx.to_string());

        args.push("--threads".to_string());
        args.push(self.n_threads.to_string());

        args.push("--n-gpu-layers".to_string());
        args.push(self.n_gpu_layers.to_string());

        args.push("--batch-size".to_string());
        args.push(self.n_batch.to_string());

        if self.flash_attn {
            args.push("--flash-attn".to_string());
        }

        args.extend(self.extra_args.iter().cloned());

        args
    }
}

/// llama.cpp 请求
#[derive(Debug, Clone, Serialize)]
pub struct LlamaRequest {
    pub mode: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// llama.cpp 响应
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaResponse {
    pub status: String,
    #[serde(default)]
    pub embeddings: Vec<f32>,
    #[serde(default)]
    pub tokens: usize,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub time: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl LlamaResponse {
    pub fn is_success(&self) -> bool {
        self.status == "success"
    }
}

/// llama.cpp 进程管理器
pub struct LlamaCppProcessManager {
    config: LlamaCppConfig,
    process: Option<Child>,
    last_activity: Option<Instant>,
    total_requests: u64,
    successful_requests: u64,
}

impl LlamaCppProcessManager {
    pub fn new(config: LlamaCppConfig) -> Self {
        Self {
            config,
            process: None,
            last_activity: None,
            total_requests: 0,
            successful_requests: 0,
        }
    }

    pub fn start(&mut self) -> Result<(), LlamaError> {
        if self.process.is_some() {
            self.stop()?;
        }

        info!("Starting llama.cpp process: {}", self.config.executable_path);

        let mut cmd = Command::new(&self.config.executable_path);
        cmd.args(self.config.build_args());

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn().map_err(|e| {
            LlamaError::ProcessStartFailed(format!(
                "Failed to spawn '{}': {}",
                self.config.executable_path, e
            ))
        })?;

        self.process = Some(child);
        self.last_activity = Some(Instant::now());

        info!("llama.cpp process started successfully");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), LlamaError> {
        if let Some(mut process) = self.process.take() {
            info!("Stopping llama.cpp process");

            if let Some(stdin) = process.stdin.as_mut() {
                let _ = stdin.write_all(b"{\"mode\":\"quit\"}\n");
                let _ = stdin.flush();
            }

            std::thread::sleep(Duration::from_millis(200));

            let _ = process.kill();
            let _ = process.wait();
        }
        Ok(())
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(process) = self.process.as_mut() {
            match process.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => false,
                Err(_) => false,
            }
        } else {
            false
        }
    }

    pub fn execute_request(&mut self, request: &LlamaRequest) -> Result<LlamaResponse, LlamaError> {
        if !self.is_running() {
            self.start()?;
        }

        self.total_requests += 1;

        let result = self.execute_with_retry(request);

        if result.is_ok() {
            self.successful_requests += 1;
        }

        self.last_activity = Some(Instant::now());
        result
    }

    fn execute_with_retry(&mut self, request: &LlamaRequest) -> Result<LlamaResponse, LlamaError> {
        let mut last_error: Option<LlamaError> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * (2_u64.pow(attempt as u32 - 1)));
                info!("Retry attempt {} after {:?}", attempt, delay);
                std::thread::sleep(delay);

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
                    let needs_restart = matches!(&e, LlamaError::ProcessCrashed(_) | LlamaError::ProcessNotRunning);
                    last_error = Some(e);

                    if needs_restart {
                        let _ = self.stop();
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            LlamaError::ExecutionFailed("Unknown error".to_string())
        }))
    }

    fn send_request_once(&mut self, request: &LlamaRequest) -> Result<LlamaResponse, LlamaError> {
        let process = self.process.as_mut().ok_or(LlamaError::ProcessNotRunning)?;

        let json = serde_json::to_string(request)?;

        let stdin = process.stdin.as_mut().ok_or_else(|| {
            LlamaError::CommunicationError("Failed to get stdin".to_string())
        })?;

        debug!("Sending llama.cpp request: {}", json);

        stdin.write_all(json.as_bytes()).map_err(|e| {
            LlamaError::CommunicationError(format!("Failed to write to stdin: {}", e))
        })?;
        stdin.write_all(b"\n").map_err(|e| {
            LlamaError::CommunicationError(format!("Failed to write newline: {}", e))
        })?;
        stdin.flush().map_err(|e| {
            LlamaError::CommunicationError(format!("Failed to flush stdin: {}", e))
        })?;

        let stdout = process.stdout.as_mut().ok_or_else(|| {
            LlamaError::CommunicationError("Failed to get stdout".to_string())
        })?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();

        let start_time = Instant::now();
        loop {
            response_line.clear();
            match reader.read_line(&mut response_line) {
                Ok(0) => {
                    return Err(LlamaError::ProcessCrashed(
                        "Process closed stdout (likely crashed)".to_string(),
                    ));
                }
                Ok(_) => {
                    let trimmed = response_line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    if trimmed.starts_with('{') {
                        match serde_json::from_str::<LlamaResponse>(trimmed) {
                            Ok(response) => {
                                debug!("Received llama.cpp response: {:?}", response);
                                return Ok(response);
                            }
                            Err(e) => {
                                debug!("Failed to parse line: {}, line: {}", e, trimmed);
                                continue;
                            }
                        }
                    } else {
                        debug!("llama.cpp output: {}", trimmed);
                    }
                }
                Err(e) => {
                    return Err(LlamaError::CommunicationError(format!(
                        "Failed to read from stdout: {}",
                        e
                    )));
                }
            }

            if start_time.elapsed() > Duration::from_secs(self.config.timeout_secs) {
                return Err(LlamaError::TimeoutError(Duration::from_secs(
                    self.config.timeout_secs,
                )));
            }
        }
    }

    pub fn get_stats(&self) -> LlamaProcessStats {
        LlamaProcessStats {
            total_requests: self.total_requests,
            successful_requests: self.successful_requests,
            success_rate: if self.total_requests > 0 {
                self.successful_requests as f64 / self.total_requests as f64
            } else {
                0.0
            },
            last_activity_ago_secs: self.last_activity.map(|t| t.elapsed().as_secs()),
        }
    }
}

impl Drop for LlamaCppProcessManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LlamaProcessStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub success_rate: f64,
    pub last_activity_ago_secs: Option<u64>,
}

/// llama.cpp 后端
pub struct LlamaCppBackend {
    process_manager: Arc<Mutex<LlamaCppProcessManager>>,
    semaphore: Arc<Semaphore>,
}

impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        let process_manager = Arc::new(Mutex::new(LlamaCppProcessManager::new(config)));

        Self {
            process_manager,
            semaphore,
        }
    }

    /// 文本编码（获取embeddings）
    pub async fn encode_text(&self, text: &str) -> Result<Vec<f32>, LlamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            LlamaError::ExecutionFailed(format!("Failed to acquire semaphore: {}", e))
        })?;

        let request = LlamaRequest {
            mode: "embed".to_string(),
            text: text.to_string(),
            max_tokens: None,
            temperature: None,
            request_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        let mut pm = self.process_manager.lock().await;
        let response = pm.execute_request(&request)?;

        if !response.is_success() {
            return Err(LlamaError::ExecutionFailed(
                response.error.unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        Ok(response.embeddings)
    }

    /// 文本生成
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, LlamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            LlamaError::ExecutionFailed(format!("Failed to acquire semaphore: {}", e))
        })?;

        let request = LlamaRequest {
            mode: "generate".to_string(),
            text: prompt.to_string(),
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            request_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        let mut pm = self.process_manager.lock().await;
        let response = pm.execute_request(&request)?;

        if !response.is_success() {
            return Err(LlamaError::ExecutionFailed(
                response.error.unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        Ok(response.text)
    }

    pub async fn start(&self) -> Result<(), LlamaError> {
        let mut pm = self.process_manager.lock().await;
        pm.start()
    }

    pub async fn stop(&self) -> Result<(), LlamaError> {
        let mut pm = self.process_manager.lock().await;
        pm.stop()
    }

    pub async fn get_stats(&self) -> LlamaProcessStats {
        let pm = self.process_manager.lock().await;
        pm.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.backend, "cuda");
        assert_eq!(config.n_ctx, 512);
        assert!(config.n_threads >= 1);
        assert_eq!(config.n_gpu_layers, -1);
        assert!(config.flash_attn);
    }

    #[test]
    fn test_config_build_args() {
        let config = LlamaCppConfig {
            model_path: "/models/llama.gguf".to_string(),
            n_ctx: 1024,
            n_threads: 8,
            n_gpu_layers: 0,
            flash_attn: false,
            ..Default::default()
        };

        let args = config.build_args();
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"/models/llama.gguf".to_string()));
        assert!(args.contains(&"--ctx-size".to_string()));
        assert!(args.contains(&"1024".to_string()));
        assert!(args.contains(&"--threads".to_string()));
        assert!(args.contains(&"8".to_string()));
        assert!(!args.contains(&"--flash-attn".to_string()));
    }

    #[test]
    fn test_response_is_success() {
        let success = LlamaResponse {
            status: "success".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
            tokens: 5,
            text: "Hello".to_string(),
            time: 0.5,
            error: None,
            request_id: None,
        };
        assert!(success.is_success());

        let failure = LlamaResponse {
            status: "error".to_string(),
            embeddings: vec![],
            tokens: 0,
            text: String::new(),
            time: 0.0,
            error: Some("test error".to_string()),
            request_id: None,
        };
        assert!(!failure.is_success());
    }

    #[test]
    fn test_request_serialization() {
        let request = LlamaRequest {
            mode: "embed".to_string(),
            text: "hello world".to_string(),
            max_tokens: None,
            temperature: None,
            request_id: Some("test-123".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"mode\":\"embed\""));
        assert!(json.contains("\"text\":\"hello world\""));
    }
}
