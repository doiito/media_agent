// 核心类型定义

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// 工作流ID
pub type WorkflowId = String;
/// 节点ID
pub type NodeId = String;
/// Prompt任务ID
pub type PromptId = String;

/// 工作流定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// 节点集合
    pub nodes: HashMap<NodeId, WorkflowNode>,
    /// 连接集合
    #[serde(default)]
    pub links: Vec<Link>,
}

/// 工作流节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    /// 节点类型
    pub class_type: String,
    /// 输入参数
    pub inputs: HashMap<String, InputValue>,
    /// 位置信息（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pos: Option<(f32, f32)>,
    /// 尺寸信息（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<(f32, f32)>,
    /// IS_CHANGED缓存值（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_changed: Option<Vec<Option<f64>>>,
}

/// 输入值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputValue {
    /// 直接值
    Direct(Value),
    /// 连接到其他节点的输出 [源节点id, 输出索引]
    Link([String; 2]),
}

/// 数据类型枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataType {
    MODEL,
    CLIP,
    VAE,
    CONDITIONING,
    LATENT,
    IMAGE,
    VIDEO,
    AUDIO,
    INT,
    FLOAT,
    STRING,
    CONTROL_NET,
}

/// 节点连接
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    /// 源节点ID
    pub from_node: NodeId,
    /// 源输出槽
    pub from_slot: usize,
    /// 目标节点ID
    pub to_node: NodeId,
    /// 目标输入槽
    pub to_slot: usize,
    /// 数据类型
    #[serde(rename = "type")]
    pub data_type: DataType,
}

/// 通用值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    /// 模型引用（内部使用）
    Model(String),
    /// CLIP引用（内部使用）
    Clip(String),
    /// VAE引用（内部使用）
    Vae(String),
    /// Latent张量（内部使用）
    Latent(Vec<f32>),
    /// 图像数据（内部使用）
    Image(Vec<u8>),
    /// 视频数据（内部使用）
    Video(Vec<u8>),
    /// 音频数据（内部使用）
    Audio(Vec<f32>),
    /// Conditioning数据（内部使用）
    Conditioning(Vec<f32>),
    /// ControlNet引用（内部使用）
    ControlNet(String),
}

impl Value {
    pub fn as_str(&self) -> Result<&str, Error> {
        match self {
            Value::String(s) => Ok(s),
            _ => Err(Error::TypeError("Expected String".to_string())),
        }
    }

    /// 返回内部引用字符串，适用于 Model/Clip/Vae 等内部引用类型
    pub fn as_ref_str(&self) -> Result<&str, Error> {
        match self {
            Value::String(s)
            | Value::Model(s)
            | Value::Clip(s)
            | Value::Vae(s) => Ok(s),
            _ => Err(Error::TypeError(format!(
                "Expected string-like reference, got {:?}",
                self
            ))),
        }
    }

    pub fn as_int(&self) -> Result<i64, Error> {
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(Error::TypeError("Expected Int".to_string())),
        }
    }

    pub fn as_float(&self) -> Result<f64, Error> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(Error::TypeError("Expected Float".to_string())),
        }
    }
}

/// 错误类型
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Invalid connection: {0}")]
    InvalidConnection(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Image error: {0}")]
    ImageError(String),

    #[error("Parse error: {0}")]
    ParseError(#[from] std::num::ParseIntError),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Workflow error: {0}")]
    WorkflowError(String),
}

impl Error {
    /// 获取 HTTP 状态码
    pub fn status_code(&self) -> u16 {
        match self {
            Error::TypeError(_) | Error::InvalidConnection(_) | Error::BadRequest(_) => 400,
            Error::Unauthorized(_) => 401,
            Error::Forbidden(_) => 403,
            Error::NotFound(_) | Error::NodeNotFound(_) => 404,
            Error::Conflict(_) => 409,
            Error::Timeout(_) => 408,
            Error::ValidationFailed(_) => 422,
            Error::ServiceUnavailable(_) => 503,
            _ => 500,
        }
    }

    /// 是否为可重试错误
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::Timeout(_) | Error::ServiceUnavailable(_) | Error::BackendError(_)
        )
    }

    /// 错误码（用于 API 响应）
    pub fn error_code(&self) -> &'static str {
        match self {
            Error::TypeError(_) => "TYPE_ERROR",
            Error::NodeNotFound(_) => "NODE_NOT_FOUND",
            Error::InvalidConnection(_) => "INVALID_CONNECTION",
            Error::ValidationFailed(_) => "VALIDATION_FAILED",
            Error::ExecutionFailed(_) => "EXECUTION_FAILED",
            Error::BackendError(_) => "BACKEND_ERROR",
            Error::IoError(_) => "IO_ERROR",
            Error::JsonError(_) => "JSON_ERROR",
            Error::ImageError(_) => "IMAGE_ERROR",
            Error::ParseError(_) => "PARSE_ERROR",
            Error::NotFound(_) => "NOT_FOUND",
            Error::Unauthorized(_) => "UNAUTHORIZED",
            Error::Forbidden(_) => "FORBIDDEN",
            Error::Conflict(_) => "CONFLICT",
            Error::Timeout(_) => "TIMEOUT",
            Error::BadRequest(_) => "BAD_REQUEST",
            Error::Internal(_) => "INTERNAL_ERROR",
            Error::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            Error::WorkflowError(_) => "WORKFLOW_ERROR",
        }
    }
}

/// 执行结果
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    Success(HashMap<NodeId, HashMap<String, Value>>),
    Failure(String),
    Pending,
}

/// Prompt任务
#[derive(Debug, Clone)]
pub struct PromptTask {
    pub workflow: Workflow,
    pub prompt_id: PromptId,
    pub client_id: String,
    pub priority: usize,
    pub timestamp: std::time::Instant,
}

impl PromptTask {
    pub fn new(workflow: Workflow, prompt_id: String, client_id: String) -> Self {
        Self {
            workflow,
            prompt_id,
            client_id,
            // 默认优先级为 10，enqueue_front 会用 0 作为最高优先级
            priority: 10,
            timestamp: std::time::Instant::now(),
        }
    }

    pub fn with_priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }
}

/// 优先级队列实现 (优先级越小越优先)
impl PartialEq for PromptTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PromptTask {}

impl PartialOrd for PromptTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PromptTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 优先级越小越优先
        other.priority.cmp(&self.priority)
    }
}

/// 节点指纹 (用于缓存)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NodeFingerprint {
    pub class_type: String,
    pub input_hash: blake3::Hash,
}

/// 系统状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub devices: Vec<DeviceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub device_type: String,
    pub vram_total: usize,
    pub vram_free: usize,
    pub compute_capability: Option<String>,
}

/// 执行历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub prompt_id: PromptId,
    pub workflow: Workflow,
    pub outputs: HashMap<NodeId, HashMap<String, Value>>,
    pub status: String,
    pub start_time: f64,
    pub end_time: Option<f64>,
}

/// 上传响应
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadResponse {
    pub name: String,
    pub subfolder: String,
    #[serde(rename = "type")]
    pub file_type: String,
}

/// 提示请求
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub prompt: Workflow,
    pub client_id: String,
    #[serde(default)]
    pub extra_data: ExtraData,
}

#[derive(Debug, Deserialize, Default)]
pub struct ExtraData {
    #[serde(default)]
    pub front: bool,
}

/// 提示响应
#[derive(Debug, Serialize)]
pub struct PromptResponse {
    pub prompt_id: String,
    pub number: usize,
    pub queue_remaining: usize,
}