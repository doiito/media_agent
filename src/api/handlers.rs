// HTTP 请求处理器
// ComfyUI 兼容的 REST API

use crate::api::dto::*;
use crate::types::*;
use crate::execution::ExecutionEngine;
use crate::backend::BackendRouter;
use crate::node::registry::NodeRegistry;
use crate::monitor::Monitor;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use log::{info, warn, error};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use std::time::Instant;

/// API 共享状态
#[derive(Clone)]
pub struct ApiState {
    pub engine: Arc<Mutex<ExecutionEngine>>,
    pub backend_router: Arc<BackendRouter>,
    pub node_registry: Arc<Mutex<NodeRegistry>>,
    pub start_time: Instant,
    pub stats: Arc<RwLock<ServerStats>>,
    pub history: Arc<RwLock<std::collections::HashMap<String, HistoryEntry>>>,
    pub output_dir: String,
    pub monitor: Arc<Monitor>,
    /// Agent Engine（gliding_horse 集成）
    pub agent: Arc<tokio::sync::Mutex<crate::agent::AgentEngine>>,
}

/// 服务器统计信息
#[derive(Debug, Default, Clone)]
pub struct ServerStats {
    pub total_requests: u64,
    pub total_workflows: u64,
    pub successful_workflows: u64,
    pub failed_workflows: u64,
}

impl ApiState {
    pub async fn increment_requests(&self) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
    }

    pub async fn increment_workflows(&self) {
        let mut stats = self.stats.write().await;
        stats.total_workflows += 1;
    }

    pub async fn record_workflow_success(&self) {
        let mut stats = self.stats.write().await;
        stats.successful_workflows += 1;
    }

    pub async fn record_workflow_failure(&self) {
        let mut stats = self.stats.write().await;
        stats.failed_workflows += 1;
    }
}

// ============================================================================
// 健康检查和系统状态
// ============================================================================

/// GET /health - 健康检查
pub async fn health_check(State(state): State<ApiState>) -> impl IntoResponse {
    let mut backends = std::collections::HashMap::new();

    // 检查后端健康状态
    let sd_healthy = state.backend_router.health_check().await;
    backends.insert(
        "stable-diffusion.cpp".to_string(),
        BackendHealth {
            healthy: sd_healthy,
            stats: None,
        },
    );

    let response = HealthResponse {
        status: if sd_healthy { "healthy".to_string() } else { "degraded".to_string() },
        version: crate::VERSION.to_string(),
        uptime: state.start_time.elapsed().as_secs(),
        backends,
    };

    (StatusCode::OK, Json(response))
}

/// GET /system_stats - 系统状态
pub async fn system_stats(State(state): State<ApiState>) -> impl IntoResponse {
    state.increment_requests().await;

    let system_info = SystemInfo {
        os: std::env::consts::OS.to_string(),
        ram_total: get_total_memory(),
        ram_free: get_free_memory(),
        python_version: None,
        pytorch_version: None,
        comfyui_version: crate::VERSION.to_string(),
    };

    let devices = state.backend_router.get_system_stats().await.devices;

    let response = SystemStatsResponse {
        system: system_info,
        devices,
    };

    (StatusCode::OK, Json(response))
}

/// GET /stats - 服务器统计
pub async fn get_stats(State(state): State<ApiState>) -> impl IntoResponse {
    let stats = state.stats.read().await;
    // 队列大小：当前没有独立队列系统，返回 0
    // 后续可通过 ExecutionEngine.queue_size() 获取
    let queue_size = 0;

    let response = StatsResponse {
        server_uptime: state.start_time.elapsed().as_secs(),
        total_requests: stats.total_requests,
        total_workflows: stats.total_workflows,
        successful_workflows: stats.successful_workflows,
        failed_workflows: stats.failed_workflows,
        active_connections: 0, // 由 WebSocket 模块更新
        queue_size,
    };

    (StatusCode::OK, Json(response))
}

// ============================================================================
// 工作流 API
// ============================================================================

/// POST /prompt - 提交工作流
pub async fn submit_prompt(
    State(state): State<ApiState>,
    Json(request): Json<PromptRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    state.increment_requests().await;

    info!("Submitting workflow from client: {}", request.client_id);

    // 验证工作流
    if request.prompt.nodes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("Workflow has no nodes", "INVALID_WORKFLOW")),
        ));
    }

    // 提交到执行引擎
    let prompt_id = {
        let mut engine = state.engine.lock().await;
        match engine.submit(request.prompt, request.client_id.clone()).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to submit workflow: {}", e);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("Failed to submit workflow", "SUBMIT_ERROR")
                        .with_details(e.to_string())),
                ));
            }
        }
    };

    state.increment_workflows().await;

    // 异步执行工作流
    let engine_clone = state.engine.clone();
    let stats_state = state.clone();
    let prompt_id_clone = prompt_id.clone();
    let history = state.history.clone();

    tokio::spawn(async move {
        loop {
            let result = {
                let mut engine = engine_clone.lock().await;
                match engine.execute_next().await {
                    Ok(Some(result)) => result,
                    Ok(None) => break,
                    Err(e) => {
                        error!("Workflow execution error: {}", e);
                        stats_state.record_workflow_failure().await;
                        break;
                    }
                }
            };

            match result {
                ExecutionResult::Success(outputs) => {
                    info!("Workflow {} completed successfully", prompt_id_clone);
                    stats_state.record_workflow_success().await;

                    // 保存到历史
                    let entry = HistoryEntry {
                        prompt_id: prompt_id_clone.clone(),
                        workflow: Workflow {
                            nodes: std::collections::HashMap::new(),
                            links: vec![],
                        },
                        outputs,
                        status: "success".to_string(),
                        start_time: 0.0,
                        end_time: Some(0.0),
                    };
                    history.write().await.insert(prompt_id_clone.clone(), entry);
                    break;
                }
                ExecutionResult::Failure(err) => {
                    error!("Workflow {} failed: {}", prompt_id_clone, err);
                    stats_state.record_workflow_failure().await;

                    let entry = HistoryEntry {
                        prompt_id: prompt_id_clone.clone(),
                        workflow: Workflow {
                            nodes: std::collections::HashMap::new(),
                            links: vec![],
                        },
                        outputs: std::collections::HashMap::new(),
                        status: format!("error: {}", err),
                        start_time: 0.0,
                        end_time: Some(0.0),
                    };
                    history.write().await.insert(prompt_id_clone.clone(), entry);
                    break;
                }
                ExecutionResult::Pending => continue,
            }
        }
    });

    let response = PromptResponse {
        prompt_id,
        number: 1,
        queue_remaining: 0,
    };

    Ok((StatusCode::OK, Json(response)))
}

/// POST /interrupt - 中断当前执行
pub async fn interrupt(State(state): State<ApiState>) -> impl IntoResponse {
    state.increment_requests().await;

    let mut engine = state.engine.lock().await;
    engine.interrupt();

    let response = InterruptResponse {
        success: true,
        message: "Interrupt signal sent".to_string(),
    };

    (StatusCode::OK, Json(response))
}

/// POST /free - 释放内存
pub async fn free_memory(
    State(state): State<ApiState>,
    Json(request): Json<FreeMemoryRequest>,
) -> impl IntoResponse {
    state.increment_requests().await;

    if request.unload_models || request.free_memory {
        state.backend_router.free_memory().await;
    }

    let response = FreeMemoryResponse {
        success: true,
        message: "Memory freed".to_string(),
    };

    (StatusCode::OK, Json(response))
}

// ============================================================================
// 队列和历史 API
// ============================================================================

/// GET /queue - 获取队列状态
pub async fn get_queue(State(state): State<ApiState>) -> impl IntoResponse {
    state.increment_requests().await;

    let response = QueueResponse {
        queue_running: vec![],
        queue_pending: vec![],
    };

    (StatusCode::OK, Json(response))
}

/// GET /history - 获取执行历史
pub async fn get_history(
    State(state): State<ApiState>,
    Query(query): Query<HistoryQuery>,
) -> impl IntoResponse {
    state.increment_requests().await;

    let history = state.history.read().await;

    let filtered: std::collections::HashMap<String, HistoryEntry> = if let Some(prompt_id) = &query.prompt_id {
        history.iter().filter(|(id, _)| *id == prompt_id).map(|(k, v)| (k.clone(), v.clone())).collect()
    } else if let Some(max) = query.max_items {
        history.iter().take(max).map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        history.clone()
    };

    let response = HistoryResponse { history: filtered };

    (StatusCode::OK, Json(response))
}

/// DELETE /history - 清除历史
pub async fn clear_history(State(state): State<ApiState>) -> impl IntoResponse {
    state.increment_requests().await;

    let mut history = state.history.write().await;
    history.clear();

    (StatusCode::OK, Json(serde_json::json!({"deleted": true})))
}

// ============================================================================
// 节点信息 API
// ============================================================================

/// GET /object_info - 获取所有节点信息
pub async fn get_object_info(State(state): State<ApiState>) -> impl IntoResponse {
    state.increment_requests().await;

    let registry = state.node_registry.lock().await;
    let node_inputs = registry.get_all_node_info();

    let mut nodes = std::collections::HashMap::new();

    for (class_type, inputs) in node_inputs {
        let node_info = NodeInfo {
            input: inputs.into_iter().map(|(name, input_type)| {
                (name, NodeInputInfo {
                    input_type: format!("{:?}", input_type),
                    default: None,
                    options: None,
                    is_list: false,
                })
            }).collect(),
            output: vec![],
            output_name: vec![],
            output_is_list: vec![],
            name: class_type.clone(),
            display_name: class_type.clone(),
            category: "default".to_string(),
            description: String::new(),
        };
        nodes.insert(class_type, node_info);
    }

    let response = ObjectInfoResponse { nodes };

    (StatusCode::OK, Json(response))
}

/// GET /object_info/{node_class} - 获取特定节点信息
pub async fn get_node_info(
    State(state): State<ApiState>,
    Path(node_class): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    state.increment_requests().await;

    let registry = state.node_registry.lock().await;
    let all_inputs = registry.get_all_node_info();

    let inputs = all_inputs.into_iter()
        .find(|(name, _)| name == &node_class)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse::new(
                    format!("Node class '{}' not found", node_class),
                    "NODE_NOT_FOUND",
                )),
            )
        })?;

    let node_info = NodeInfo {
        input: inputs.1.into_iter().map(|(name, input_type)| {
            (name, NodeInputInfo {
                input_type: format!("{:?}", input_type),
                default: None,
                options: None,
                is_list: false,
            })
        }).collect(),
        output: vec![],
        output_name: vec![],
        output_is_list: vec![],
        name: node_class.clone(),
        display_name: node_class.clone(),
        category: "default".to_string(),
        description: String::new(),
    };

    Ok((StatusCode::OK, Json(node_info)))
}

// ============================================================================
// 模型管理 API
// ============================================================================

/// GET /models/{folder} - 获取模型列表
pub async fn get_models(
    State(state): State<ApiState>,
    Path(folder): Path<String>,
) -> impl IntoResponse {
    state.increment_requests().await;

    let models = list_models_in_folder(&folder, &state);

    let response = ModelsResponse { models };

    (StatusCode::OK, Json(response))
}

/// 获取文件夹中的模型列表
fn list_models_in_folder(folder: &str, _state: &ApiState) -> Vec<String> {
    let base_dir = match folder {
        "checkpoints" => "models/checkpoints",
        "vae" => "models/vae",
        "clip" => "models/clip",
        "controlnet" => "models/controlnet",
        "embeddings" => "models/embeddings",
        "loras" => "models/loras",
        _ => "models",
    };

    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if matches!(ext, "safetensors" | "ckpt" | "pt" | "bin" | "gguf") {
                        models.push(name.to_string());
                    }
                }
            }
        }
    }

    models.sort();
    models
}

// ============================================================================
// 图像 API
// ============================================================================

/// GET /view - 查看图像
pub async fn view_image(
    State(state): State<ApiState>,
    Query(query): Query<ViewQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    state.increment_requests().await;

    let base_dir = match query.view_type.as_str() {
        "input" => "input",
        "temp" => "temp",
        _ => "output",
    };

    let mut path = std::path::PathBuf::from(base_dir);
    if !query.subfolder.is_empty() {
        path.push(&query.subfolder);
    }
    path.push(&query.filename);

    if !path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("Image '{}' not found", query.filename),
                "FILE_NOT_FOUND",
            )),
        ));
    }

    let data = std::fs::read(&path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("Failed to read image", "READ_ERROR")
                .with_details(e.to_string())),
        )
    })?;

    let mime_type = path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| match ext.to_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "gif" => "image/gif",
            _ => "application/octet-stream",
        })
        .unwrap_or("application/octet-stream");

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, mime_type)
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(data))
        .unwrap())
}

/// POST /upload/image - 上传图像
pub async fn upload_image(
    State(state): State<ApiState>,
    mut multipart: axum::extract::Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    state.increment_requests().await;

    let mut image_data: Option<Vec<u8>> = None;
    let mut filename = "uploaded.png".to_string();
    let mut subfolder = String::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("Multipart error", "MULTIPART_ERROR")
                .with_details(e.to_string())),
        )
    })? {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "image" => {
                filename = field.file_name().unwrap_or("uploaded.png").to_string();
                let data = field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse::new("Failed to read image data", "READ_ERROR")
                            .with_details(e.to_string())),
                    )
                })?;
                image_data = Some(data.to_vec());
            }
            "subfolder" => {
                let data = field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse::new("Failed to read subfolder", "READ_ERROR")
                            .with_details(e.to_string())),
                    )
                })?;
                subfolder = String::from_utf8_lossy(&data).to_string();
            }
            _ => {}
        }
    }

    let image_data = image_data.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("No image provided", "NO_IMAGE")),
        )
    })?;

    // 创建上传目录
    let mut upload_dir = std::path::PathBuf::from("input");
    if !subfolder.is_empty() {
        upload_dir.push(&subfolder);
    }
    std::fs::create_dir_all(&upload_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("Failed to create directory", "DIR_ERROR")
                .with_details(e.to_string())),
        )
    })?;

    let file_path = upload_dir.join(&filename);
    std::fs::write(&file_path, &image_data).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("Failed to save image", "SAVE_ERROR")
                .with_details(e.to_string())),
        )
    })?;

    let response = UploadResponse {
        name: filename,
        subfolder,
        file_type: "input".to_string(),
    };

    Ok((StatusCode::OK, Json(response)))
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 获取系统总内存（字节）
/// Linux: 读取 /proc/meminfo
fn get_total_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse().unwrap_or(0);
                        return kb * 1024;
                    }
                }
            }
        }
    }
    // 默认值
    16 * 1024 * 1024 * 1024
}

/// 获取系统可用内存（字节）
/// Linux: 读取 /proc/meminfo
fn get_free_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse().unwrap_or(0);
                        return kb * 1024;
                    }
                }
            }
            // 如果 MemAvailable 不存在，使用 MemFree + Buffers + Cached
            let mut mem_free = 0u64;
            let mut buffers = 0u64;
            let mut cached = 0u64;
            for line in content.lines() {
                if line.starts_with("MemFree:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        mem_free = parts[1].parse().unwrap_or(0) * 1024;
                    }
                } else if line.starts_with("Buffers:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        buffers = parts[1].parse().unwrap_or(0) * 1024;
                    }
                } else if line.starts_with("Cached:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        cached = parts[1].parse().unwrap_or(0) * 1024;
                    }
                }
            }
            return mem_free + buffers + cached;
        }
    }
    // 默认值
    8 * 1024 * 1024 * 1024
}

// ============================================================================
// 监控 API
// ============================================================================

/// GET /monitor/latest - 获取最新监控指标
pub async fn monitor_latest(State(state): State<ApiState>) -> impl IntoResponse {
    match state.monitor.latest().await {
        Some(snapshot) => Json(serde_json::json!(snapshot)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "No metrics available yet",
                "NO_METRICS",
            )),
        )
            .into_response(),
    }
}

/// GET /monitor/history - 获取历史监控数据
pub async fn monitor_history(
    State(state): State<ApiState>,
    Query(query): Query<MonitorHistoryQuery>,
) -> impl IntoResponse {
    let limit = query.limit;
    let history = state.monitor.history(limit).await;
    Json(serde_json::json!({ "history": history, "count": history.len() })).into_response()
}

/// GET /monitor/alerts - 获取告警列表
pub async fn monitor_alerts(State(state): State<ApiState>) -> impl IntoResponse {
    let alerts = state.monitor.alerts().await;
    Json(serde_json::json!({ "alerts": alerts, "count": alerts.len() })).into_response()
}

/// DELETE /monitor/alerts - 清除告警
pub async fn monitor_clear_alerts(State(state): State<ApiState>) -> impl IntoResponse {
    state.monitor.clear_alerts().await;
    Json(serde_json::json!({ "status": "cleared" }))
}

/// GET /monitor/averages - 获取平均指标
pub async fn monitor_averages(State(state): State<ApiState>) -> impl IntoResponse {
    match state.monitor.averages().await {
        Some(avg) => Json(avg).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "No metrics available yet",
                "NO_METRICS",
            )),
        )
            .into_response(),
    }
}

/// POST /monitor/collect - 触发一次主动采集
pub async fn monitor_collect(State(state): State<ApiState>) -> impl IntoResponse {
    let snapshot = state.monitor.collect().await;
    Json(snapshot)
}

/// GET /monitor/processes - 获取相关进程信息
pub async fn monitor_processes(State(state): State<ApiState>) -> impl IntoResponse {
    let processes = state.monitor.related_processes().await;
    Json(serde_json::json!({ "processes": processes, "count": processes.len() }))
        .into_response()
}
