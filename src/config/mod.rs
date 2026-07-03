// 统一配置管理模块
// 支持从配置文件、环境变量、命令行参数加载配置

use crate::backend::{LlamaCppConfig, SdCppConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 应用顶层配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// 服务器配置
    #[serde(default)]
    pub server: ServerConfig,

    /// stable-diffusion.cpp 后端配置
    #[serde(default)]
    pub sd_cpp: SdCppConfig,

    /// llama.cpp 后端配置
    #[serde(default)]
    pub llama_cpp: LlamaCppConfig,

    /// 日志配置
    #[serde(default)]
    pub log: LogConfig,

    /// 监控配置
    #[serde(default)]
    pub monitor: MonitorConfig,

    /// 路径配置
    #[serde(default)]
    pub paths: PathsConfig,
}

/// 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// 监听地址
    #[serde(default = "default_host")]
    pub host: String,

    /// 监听端口
    #[serde(default = "default_port")]
    pub port: u16,

    /// 输出目录
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// 最大工作流数
    #[serde(default = "default_max_workflows")]
    pub max_workflows: usize,

    /// 请求超时（秒）
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// 是否启用 CORS
    #[serde(default = "default_true")]
    pub enable_cors: bool,

    /// 最大请求体大小（MB）
    #[serde(default = "default_max_body_size_mb")]
    pub max_body_size_mb: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            output_dir: default_output_dir(),
            max_workflows: default_max_workflows(),
            request_timeout_secs: default_request_timeout(),
            enable_cors: true,
            max_body_size_mb: default_max_body_size_mb(),
        }
    }
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// 日志级别 (trace/debug/info/warn/error)
    #[serde(default = "default_log_level")]
    pub level: String,

    /// 日志文件路径（可选，不设置则只输出到 stderr）
    #[serde(default)]
    pub file: Option<String>,

    /// 是否在日志中包含时间戳
    #[serde(default = "default_true")]
    pub timestamp: bool,

    /// 是否在日志中包含模块路径
    #[serde(default)]
    pub module_path: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            timestamp: true,
            module_path: false,
        }
    }
}

/// 监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// 是否启用监控
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// 采集间隔（秒）
    #[serde(default = "default_collect_interval")]
    pub collect_interval_secs: u64,

    /// 历史数据保留数量
    #[serde(default = "default_history_size")]
    pub history_size: usize,

    /// CPU 使用率告警阈值（百分比）
    #[serde(default = "default_cpu_alert_threshold")]
    pub cpu_alert_threshold: f32,

    /// 内存使用率告警阈值（百分比）
    #[serde(default = "default_mem_alert_threshold")]
    pub mem_alert_threshold: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collect_interval_secs: default_collect_interval(),
            history_size: default_history_size(),
            cpu_alert_threshold: default_cpu_alert_threshold(),
            mem_alert_threshold: default_mem_alert_threshold(),
        }
    }
}

/// 路径配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    /// 模型根目录
    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    /// 输入目录（上传图像等）
    #[serde(default = "default_input_dir")]
    pub input_dir: String,

    /// 临时目录
    #[serde(default = "default_temp_dir")]
    pub temp_dir: String,

    /// 提示词模板目录
    #[serde(default = "default_prompts_dir")]
    pub prompts_dir: String,

    /// Schema 目录
    #[serde(default = "default_schemas_dir")]
    pub schemas_dir: String,

    /// 工作流模板目录
    #[serde(default = "default_workflows_dir")]
    pub workflows_dir: String,

    /// 技能定义目录
    #[serde(default = "default_skills_dir")]
    pub skills_dir: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            input_dir: default_input_dir(),
            temp_dir: default_temp_dir(),
            prompts_dir: default_prompts_dir(),
            schemas_dir: default_schemas_dir(),
            workflows_dir: default_workflows_dir(),
            skills_dir: default_skills_dir(),
        }
    }
}

// ============================================================================
// 默认值函数
// ============================================================================

fn default_host() -> String {
    std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string())
}

fn default_port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8188)
}

fn default_output_dir() -> String {
    std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "output".to_string())
}

fn default_max_workflows() -> usize {
    100
}

fn default_request_timeout() -> u64 {
    300
}

fn default_max_body_size_mb() -> usize {
    50
}

fn default_log_level() -> String {
    std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string())
}

fn default_collect_interval() -> u64 {
    10
}

fn default_history_size() -> usize {
    360
}

fn default_cpu_alert_threshold() -> f32 {
    90.0
}

fn default_mem_alert_threshold() -> f32 {
    85.0
}

fn default_models_dir() -> String {
    "models".to_string()
}

fn default_input_dir() -> String {
    "input".to_string()
}

fn default_temp_dir() -> String {
    std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string())
}

fn default_prompts_dir() -> String {
    std::env::var("PROMPTS_DIR").unwrap_or_else(|_| "prompts".to_string())
}

fn default_schemas_dir() -> String {
    std::env::var("SCHEMAS_DIR").unwrap_or_else(|_| "schemas".to_string())
}

fn default_workflows_dir() -> String {
    std::env::var("WORKFLOWS_DIR").unwrap_or_else(|_| "workflows".to_string())
}

fn default_skills_dir() -> String {
    std::env::var("SKILLS_DIR").unwrap_or_else(|_| "skills".to_string())
}

fn default_true() -> bool {
    true
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            sd_cpp: SdCppConfig::from_env(),
            llama_cpp: LlamaCppConfig::from_env(),
            log: LogConfig::default(),
            monitor: MonitorConfig::default(),
            paths: PathsConfig::default(),
        }
    }
}

impl AppConfig {
    /// 从环境变量加载配置
    pub fn from_env() -> Self {
        Self::default()
    }

    /// 从 JSON 配置文件加载
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::FileReadError(format!("{}: {}", path, e)))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ConfigError::ParseError(format!("{}: {}", path, e)))?;
        Ok(config)
    }

    /// 尝试从默认路径加载配置，失败则返回默认配置
    pub fn load_or_default() -> Self {
        let candidates = [
            "config.json",
            "config/config.json",
            "/etc/comfyui-rs/config.json",
        ];

        for path in &candidates {
            if let Ok(config) = Self::from_file(path) {
                log::info!("Loaded config from {}", path);
                return config;
            }
        }

        log::info!("Using default configuration (no config file found)");
        Self::default()
    }

    /// 合并环境变量覆盖（环境变量优先级最高）
    pub fn merge_env_overrides(&mut self) {
        // 服务器配置覆盖
        if let Ok(host) = std::env::var("HOST") {
            self.server.host = host;
        }
        if let Ok(port) = std::env::var("PORT") {
            if let Ok(p) = port.parse() {
                self.server.port = p;
            }
        }
        if let Ok(dir) = std::env::var("OUTPUT_DIR") {
            self.server.output_dir = dir;
        }

        // 日志配置覆盖
        if let Ok(level) = std::env::var("LOG_LEVEL") {
            self.log.level = level;
        }

        // 路径配置覆盖
        if let Ok(dir) = std::env::var("MODELS_DIR") {
            self.paths.models_dir = dir;
        }
        if let Ok(dir) = std::env::var("INPUT_DIR") {
            self.paths.input_dir = dir;
        }
    }

    /// 确保所有目录存在
    pub fn ensure_directories(&self) -> Result<(), ConfigError> {
        let dirs = [
            &self.server.output_dir,
            &self.paths.models_dir,
            &self.paths.input_dir,
            &self.paths.temp_dir,
            &self.paths.prompts_dir,
            &self.paths.schemas_dir,
            &self.paths.workflows_dir,
            &self.paths.skills_dir,
        ];

        for dir in &dirs {
            if !dir.is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| {
                    ConfigError::DirectoryCreateError(format!("{}: {}", dir, e))
                })?;
            }
        }

        // 创建模型子目录
        let model_subdirs = [
            "checkpoints",
            "diffusion",
            "vae",
            "lora",
            "clip",
            "clip_vision",
            "controlnet",
            "embeddings",
        ];
        for sub in &model_subdirs {
            let path = PathBuf::from(&self.paths.models_dir).join(sub);
            std::fs::create_dir_all(&path).map_err(|e| {
                ConfigError::DirectoryCreateError(format!("{:?}: {}", path, e))
            })?;
        }

        // 创建提示词子目录（PA/DA/CA/AA/SA 角色）
        let prompt_subdirs = ["pa", "da", "ca", "aa", "sa"];
        for sub in &prompt_subdirs {
            let path = PathBuf::from(&self.paths.prompts_dir).join(sub);
            std::fs::create_dir_all(&path).map_err(|e| {
                ConfigError::DirectoryCreateError(format!("{:?}: {}", path, e))
            })?;
        }

        Ok(())
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.server.port == 0 {
            return Err(ConfigError::ValidationError(
                "Server port cannot be 0".to_string(),
            ));
        }

        if self.server.max_workflows == 0 {
            return Err(ConfigError::ValidationError(
                "max_workflows must be greater than 0".to_string(),
            ));
        }

        if self.monitor.collect_interval_secs == 0 {
            return Err(ConfigError::ValidationError(
                "monitor.collect_interval_secs must be greater than 0".to_string(),
            ));
        }

        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.log.level.to_lowercase().as_str()) {
            return Err(ConfigError::ValidationError(format!(
                "Invalid log level: {} (must be one of {:?})",
                self.log.level, valid_log_levels
            )));
        }

        Ok(())
    }
}

/// 配置错误
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileReadError(String),

    #[error("Failed to parse config file: {0}")]
    ParseError(String),

    #[error("Failed to create directory: {0}")]
    DirectoryCreateError(String),

    #[error("Config validation error: {0}")]
    ValidationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert!(config.server.port > 0);
        assert!(!config.server.host.is_empty());
        assert!(!config.server.output_dir.is_empty());
        assert_eq!(config.log.level, "info");
        assert!(config.monitor.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        assert!(config.validate().is_ok());

        config.server.port = 0;
        assert!(config.validate().is_err());

        let mut config = AppConfig::default();
        config.log.level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_from_json() {
        let json = r#"{
            "server": {
                "host": "127.0.0.1",
                "port": 9999,
                "output_dir": "/tmp/test_output"
            },
            "log": {
                "level": "debug"
            }
        }"#;

        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.output_dir, "/tmp/test_output");
        assert_eq!(config.log.level, "debug");
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.server.port, parsed.server.port);
        assert_eq!(config.log.level, parsed.log.level);
    }

    #[test]
    fn test_ensure_directories() {
        let mut config = AppConfig::default();
        config.server.output_dir = "/tmp/comfyui_test_output".to_string();
        config.paths.models_dir = "/tmp/comfyui_test_models".to_string();
        config.paths.input_dir = "/tmp/comfyui_test_input".to_string();
        config.paths.temp_dir = "/tmp/comfyui_test_temp".to_string();

        assert!(config.ensure_directories().is_ok());

        // 验证目录确实被创建
        assert!(std::path::Path::new(&config.server.output_dir).exists());
        assert!(std::path::Path::new(&config.paths.models_dir).exists());

        // 清理
        let _ = std::fs::remove_dir_all("/tmp/comfyui_test_output");
        let _ = std::fs::remove_dir_all("/tmp/comfyui_test_models");
        let _ = std::fs::remove_dir_all("/tmp/comfyui_test_input");
        let _ = std::fs::remove_dir_all("/tmp/comfyui_test_temp");
    }
}
