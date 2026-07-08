// stable-diffusion.cpp 集成实现
// 包含进程管理、错误处理、重试机制、并发控制

use crate::types::*;
use crate::backend::{T2IParams, I2IParams, T2VParams, I2VParams};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use log::{debug, error, info, warn};

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

// ============================================================================
// 配置管理
// ============================================================================

/// stable-diffusion.cpp 后端配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdCppConfig {
    #[serde(default = "default_executable_path")]
    pub executable_path: String,

    #[serde(default)]
    pub model_path: String,

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

    /// RNG 模式 (cuda/cpu - ComfyUI兼容使用cpu)
    #[serde(default = "default_rng_mode")]
    pub rng_mode: String,

    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

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
    std::env::var("SD_CPP_EXECUTABLE").unwrap_or_else(|_| "sd-cli".to_string())
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

impl Default for SdCppConfig {
    fn default() -> Self {
        Self {
            executable_path: default_executable_path(),
            model_path: String::new(),
            backend: default_backend(),
            precision: default_precision(),
            flash_attention: default_flash_attention(),
            offload_to_cpu: false,
            rng_mode: default_rng_mode(),
            timeout_secs: default_timeout_secs(),
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
        if let Ok(val) = std::env::var("SD_CPP_MODEL_PATH") {
            config.model_path = val;
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
        if let Ok(val) = std::env::var("SD_CPP_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse() {
                config.timeout_secs = secs;
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

        let mut cmd = std::process::Command::new(executable);
        cmd.arg("--model").arg(model)
            .arg("--prompt").arg(&params.prompt)
            .arg("--output").arg(&output_path)
            .arg("--backend").arg(&self.config.backend)
            .arg("--rng").arg(&self.config.rng_mode)
            .arg("--steps").arg(params.steps.to_string())
            .arg("--cfg-scale").arg(params.cfg.to_string())
            .arg("--width").arg(params.width.to_string())
            .arg("--height").arg(params.height.to_string())
            .arg("--seed").arg(params.seed.to_string());

        if !params.negative_prompt.is_empty() {
            cmd.arg("--negative-prompt").arg(&params.negative_prompt);
        }

        // sd-cli doesn't support standalone --flash-attention flag;
        // flash attention is compiled in via GGML_CUDA_FA cmake option.
        // Add any model-specific args here if needed.

        info!("Running sd-cli: {} --model {} --backend {} --steps {} --width {}x{}",
            executable, model, self.config.backend, params.steps, params.width, params.height);

        let output = cmd.output().map_err(|e| {
            SdError::ProcessStartFailed(format!("Failed to run sd-cli: {}", e))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SdError::ExecutionFailed(format!(
                "sd-cli failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                stderr.lines().next().unwrap_or("unknown error")
            )));
        }

        let image_data = std::fs::read(&output_path).map_err(|e| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to read sd-cli output '{}': {}", output_path, e),
            ))
        })?;

        // 清理临时文件
        let _ = std::fs::remove_file(&output_path);

        Ok(image_data)
    }

    /// 文生图
    pub async fn text_to_image(&self, params: T2IParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        // 使用直接子进程调用（sd-cli 不支持 stdin/stdout 协议）
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

        // 将输入图像写入临时文件（sd-cli 的 --image 接受路径）
        let input_path = format!("/tmp/sd_input_{}.png", uuid::Uuid::new_v4());
        std::fs::write(&input_path, &params.input_image).map_err(|e| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write input image: {}", e),
            ))
        })?;

        let executable = &self.config.executable_path;
        let model = &self.config.model_path;

        let mut cmd = std::process::Command::new(executable);
        cmd.arg("--model").arg(model)
            .arg("--prompt").arg(&params.prompt)
            .arg("--image").arg(&input_path)
            .arg("--output").arg(&output_path)
            .arg("--backend").arg(&self.config.backend)
            .arg("--rng").arg(&self.config.rng_mode)
            .arg("--steps").arg(params.steps.to_string())
            .arg("--cfg-scale").arg(params.cfg.to_string())
            .arg("--width").arg("512")
            .arg("--height").arg("512")
            .arg("--seed").arg(params.seed.to_string());

        if !params.negative_prompt.is_empty() {
            cmd.arg("--negative-prompt").arg(&params.negative_prompt);
        }

        info!("Running sd-cli img2img: {} --backend {} --steps {} --denoise {}",
            executable, self.config.backend, params.steps, params.denoise);

        let output = cmd.output().map_err(|e| {
            SdError::ProcessStartFailed(format!("Failed to run sd-cli: {}", e))
        })?;

        // 清理输入临时文件
        let _ = std::fs::remove_file(&input_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SdError::ExecutionFailed(format!(
                "sd-cli img2img failed (exit={}): {}",
                output.status.code().unwrap_or(-1),
                stderr.lines().next().unwrap_or("unknown error")
            )));
        }

        let image_data = std::fs::read(&output_path).map_err(|e| {
            SdError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to read sd-cli output '{}': {}", output_path, e),
            ))
        })?;

        let _ = std::fs::remove_file(&output_path);
        Ok(image_data)
    }

    /// 图生图
    pub async fn image_to_image(&self, params: I2IParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        // 使用直接子进程调用（sd-cli 不支持 stdin/stdout 协议）
        self.run_sd_cli_image_to_image(&params).await
    }

    /// 文生视频
    pub async fn text_to_video(&self, params: T2VParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

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

    /// 图生视频（SVD）
    pub async fn image_to_video(&self, params: I2VParams) -> Result<Vec<u8>, SdError> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            SdError::ResourceLimitExceeded(format!("Failed to acquire semaphore: {}", e))
        })?;

        // 将输入图像写入临时文件
        let input_path = std::env::temp_dir().join(format!("svd_input_{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&input_path, &params.input_image).map_err(SdError::from)?;

        // SVD 模式：通过进程协议发送 image_to_video 请求
        let request = SdRequest {
            mode: "image_to_video".to_string(),
            prompt: params.prompt,
            negative_prompt: params.negative_prompt,
            width: params.width,
            height: params.height,
            steps: params.steps,
            cfg: params.cfg,
            sampler: "euler".to_string(),
            seed: params.seed,
            model_path: params.model_path,
            input_image: Some(input_path.to_string_lossy().into_owned()),
            controlnet: None,
            denoise: None,
            request_id: Some(uuid::Uuid::new_v4().to_string()),
        };

        let mut pm = self.process_manager.lock().await;
        let response = pm.execute_request(&request)?;

        // 清理临时文件
        let _ = std::fs::remove_file(&input_path);

        if !response.is_success() {
            return Err(SdError::ExecutionFailed(
                response.error.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        let video_data = std::fs::read(&response.output_path)?;
        Ok(video_data)
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

/// 简单的base64编码（避免引入额外依赖）
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }

    result
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
    fn test_base64_encode() {
        // 测试空数据
        assert_eq!(base64_encode(b""), "");

        // 测试 "Hello"
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");

        // 测试 "Hel"
        assert_eq!(base64_encode(b"Hel"), "SGVs");

        // 测试 "He"
        assert_eq!(base64_encode(b"He"), "SGU=");
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
