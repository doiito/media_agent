// API 数据传输对象 (DTO)
// 仅定义 types.rs 中未包含的 DTO

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// 请求 DTO (types.rs 中未定义的)
// ============================================================================

/// 视图查询参数
#[derive(Debug, Deserialize)]
pub struct ViewQuery {
    pub filename: String,
    #[serde(default)]
    pub subfolder: String,
    #[serde(default = "default_view_type")]
    #[serde(rename = "type")]
    pub view_type: String,
    #[serde(default)]
    pub preview: bool,
}

/// 监控历史查询参数
#[derive(Debug, Deserialize)]
pub struct MonitorHistoryQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

fn default_view_type() -> String {
    "output".to_string()
}

/// 队列查询参数
#[derive(Debug, Deserialize)]
pub struct QueueQuery {
    #[serde(default)]
    pub client_id: Option<String>,
}

/// 历史查询参数
#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    #[serde(default)]
    pub prompt_id: Option<String>,
    #[serde(default)]
    pub max_items: Option<usize>,
}

/// 释放内存请求
#[derive(Debug, Deserialize)]
pub struct FreeMemoryRequest {
    #[serde(default)]
    pub unload_models: bool,
    #[serde(default)]
    pub free_memory: bool,
}

// ============================================================================
// 响应 DTO (types.rs 中未定义的)
// ============================================================================

/// 队列信息响应
#[derive(Debug, Serialize)]
pub struct QueueResponse {
    #[serde(rename = "QueueRunning")]
    pub queue_running: Vec<QueueItem>,
    #[serde(rename = "QueuePending")]
    pub queue_pending: Vec<QueueItem>,
}

#[derive(Debug, Serialize)]
pub struct QueueItem {
    pub number: usize,
    pub prompt_id: String,
    pub client_id: String,
    pub priority: usize,
}

/// 历史响应
#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub history: HashMap<String, crate::types::HistoryEntry>,
}

/// 系统状态响应
#[derive(Debug, Serialize)]
pub struct SystemStatsResponse {
    pub system: SystemInfo,
    pub devices: Vec<crate::types::DeviceInfo>,
}

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub ram_total: u64,
    pub ram_free: u64,
    pub python_version: Option<String>,
    pub pytorch_version: Option<String>,
    pub comfyui_version: String,
}

/// 对象信息响应
#[derive(Debug, Serialize)]
pub struct ObjectInfoResponse {
    pub nodes: HashMap<String, NodeInfo>,
}

#[derive(Debug, Serialize)]
pub struct NodeInfo {
    pub input: HashMap<String, NodeInputInfo>,
    pub output: Vec<String>,
    pub output_name: Vec<String>,
    pub output_is_list: Vec<bool>,
    pub name: String,
    pub display_name: String,
    pub category: String,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct NodeInputInfo {
    #[serde(rename = "type")]
    pub input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
    #[serde(rename = "is_list")]
    pub is_list: bool,
}

/// 模型列表响应
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<String>,
}

/// 健康检查响应
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: u64,
    pub backends: HashMap<String, BackendHealth>,
}

#[derive(Debug, Serialize)]
pub struct BackendHealth {
    pub healthy: bool,
    pub stats: Option<serde_json::Value>,
}

/// 错误响应
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// 统计信息响应
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub server_uptime: u64,
    pub total_requests: u64,
    pub total_workflows: u64,
    pub successful_workflows: u64,
    pub failed_workflows: u64,
    pub active_connections: usize,
    pub queue_size: usize,
}

/// 中断响应
#[derive(Debug, Serialize)]
pub struct InterruptResponse {
    pub success: bool,
    pub message: String,
}

/// 释放内存响应
#[derive(Debug, Serialize)]
pub struct FreeMemoryResponse {
    pub success: bool,
    pub message: String,
}
