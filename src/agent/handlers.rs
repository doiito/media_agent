// Agent HTTP API handlers
// /agent/chat, /agent/status, /agent/workflows 端点

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::api::handlers::ApiState;

/// Agent 聊天请求
#[derive(Debug, Deserialize)]
pub struct AgentChatRequest {
    pub message: String,
    pub workflow: Option<String>,
    pub max_iterations: Option<u32>,
    pub client_id: Option<String>,
    /// 上传的图片路径（用于 img2img 工作流）
    pub image_path: Option<String>,
    /// 生成参数（steps, cfg, width, height 等）
    pub params: Option<serde_json::Value>,
}

/// Agent 聊天响应
#[derive(Debug, Serialize)]
pub struct AgentChatResponse {
    pub task_id: String,
    pub status: String,
    pub summary: String,
    pub output: Option<serde_json::Value>,
    pub turn_count: u32,
    pub tool_calls: u32,
    pub errors: Vec<String>,
}

/// Agent 状态响应
#[derive(Debug, Serialize)]
pub struct AgentStatusResponse {
    pub ready: bool,
    pub context_ready: bool,
    pub supervisor_ready: bool,
    pub workflows: Vec<String>,
    pub skills: Vec<String>,
    pub tools: Vec<String>,
}

/// 工作流列表响应
#[derive(Debug, Serialize)]
pub struct WorkflowListResponse {
    pub workflows: Vec<WorkflowInfo>,
}

/// 工作流信息
#[derive(Debug, Serialize)]
pub struct WorkflowInfo {
    pub name: String,
    pub path: String,
    pub description: Option<String>,
}

/// POST /agent/chat - 处理自然语言请求
pub async fn agent_chat(
    State(state): State<ApiState>,
    Json(req): Json<AgentChatRequest>,
) -> Response {
    let agent = state.agent.clone();

    {
        let agent_guard = agent.lock().await;
        if !agent_guard.status().supervisor_ready {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Agent not initialized",
                    "message": "Call /agent/init first or check configuration"
                })),
            ).into_response();
        }
    }

    // 构建增强消息：将图片路径和参数以结构化方式注入，让 PA/DA 能正确解析
    let enhanced_message = build_enhanced_message(&req);

    let result = {
        let mut agent_guard = agent.lock().await;
        agent_guard.process_task(&enhanced_message, req.workflow.as_deref()).await
    };

    match result {
        Ok((task_id, task_result)) => {
            let response = AgentChatResponse {
                task_id,
                status: task_result.status,
                summary: task_result.summary,
                output: task_result.output,
                turn_count: task_result.turn_count,
                tool_calls: task_result.tool_call_count,
                errors: task_result.errors,
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_default())).into_response()
        }
        Err(e) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": e,
                    "task_id": "",
                    "status": "failed"
                })),
            ).into_response()
        }
    }
}

/// GET /agent/status - 获取 Agent 状态
pub async fn agent_status(
    State(state): State<ApiState>,
) -> impl IntoResponse {
    let agent = state.agent.clone();
    let agent_guard = agent.lock().await;
    let status = agent_guard.status();

    let workflows = crate::agent::workflow::load_workflow_templates("workflows")
        .unwrap_or_default();

    let skills = vec![
        "text_to_image".to_string(),
        "image_to_image".to_string(),
        "generate_video".to_string(),
    ];

    let tools = vec![
        "submit_workflow".to_string(),
        "build_t2i_workflow".to_string(),
        "build_i2i_workflow".to_string(),
        "build_i2v_workflow".to_string(),
        "backend_sample".to_string(),
        "list_nodes".to_string(),
        "interrupt".to_string(),
        "free_memory".to_string(),
        "health_check".to_string(),
        "list_available_nodes".to_string(),
        "create_node".to_string(),
        "connect_nodes".to_string(),
        "find_compatible_sources".to_string(),
        "validate_workflow".to_string(),
        "suggest_workflow".to_string(),
        "get_node_schema".to_string(),
        "discover_comfyui_skills".to_string(),
        "recommend_parameters".to_string(),
        "analyze_failure".to_string(),
        "record_execution".to_string(),
        "find_similar_workflows".to_string(),
        "get_skill_stats".to_string(),
    ];

    let response = AgentStatusResponse {
        ready: status.context_ready && status.supervisor_ready,
        context_ready: status.context_ready,
        supervisor_ready: status.supervisor_ready,
        workflows,
        skills,
        tools,
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_default()))
}

/// GET /agent/workflows - 列出可用工作流
pub async fn agent_workflows() -> impl IntoResponse {
    let workflows = crate::agent::workflow::load_workflow_templates("workflows")
        .unwrap_or_default();

    let workflow_infos: Vec<WorkflowInfo> = workflows.iter()
        .map(|path| {
            let name = std::path::Path::new(path)
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            WorkflowInfo {
                name: name.to_string(),
                path: path.clone(),
                description: None,
            }
        })
        .collect();

    (StatusCode::OK, Json(WorkflowListResponse { workflows: workflow_infos }))
}

/// POST /agent/init - 初始化 Agent
pub async fn agent_init(
    State(state): State<ApiState>,
) -> Response {
    let agent = state.agent.clone();

    let result = {
        let mut agent_guard = agent.lock().await;
        agent_guard.build_supervisor()
    };

    match result {
        Ok(()) => {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "initialized",
                "message": "Agent SupervisorAgent built successfully"
            }))).into_response()
        }
        Err(e) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": e,
                    "status": "failed"
                })),
            ).into_response()
        }
    }
}

/// 构建增强消息：将图片路径和用户参数以结构化方式注入消息
///
/// 格式说明：
/// - 如果有图片路径，添加 `<input_image>` 标签让 PA/DA 明确知道图片位置
/// - 如果有自定义参数，添加 `<user_params>` 标签传递结构化参数
/// - 原始用户消息保持在最后，确保 LLM 理解核心需求
fn build_enhanced_message(req: &AgentChatRequest) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 注入图片路径（结构化标签，PA/DA 可解析）
    if let Some(ref img_path) = req.image_path {
        // 验证图片文件是否存在
        let full_path = if img_path.starts_with("input/") || img_path.starts_with("input\\") {
            img_path.clone()
        } else {
            format!("input/{}", img_path)
        };

        let path_check = std::path::Path::new(&full_path);
        if path_check.exists() {
            parts.push(format!(
                "<input_image>\npath: {}\nfilename: {}\n</input_image>",
                full_path,
                path_check.file_name().and_then(|n| n.to_str()).unwrap_or("unknown")
            ));
        } else {
            // 也检查直接路径
            if std::path::Path::new(img_path).exists() {
                parts.push(format!(
                    "<input_image>\npath: {}\nfilename: {}\n</input_image>",
                    img_path,
                    std::path::Path::new(img_path).file_name().and_then(|n| n.to_str()).unwrap_or("unknown")
                ));
            } else {
                log::warn!("Image not found: {} or {}", img_path, full_path);
                parts.push(format!(
                    "<input_image>\npath: {}\nwarning: file not found, please verify path\n</input_image>",
                    img_path
                ));
            }
        }
    }

    // 注入用户自定义参数
    if let Some(ref params) = req.params {
        if !params.is_null() {
            parts.push(format!("<user_params>\n{}\n</user_params>",
                serde_json::to_string_pretty(params).unwrap_or_default()));
        }
    }

    // 原始用户消息
    parts.push(format!("<user_request>\n{}\n</user_request>", req.message));

    parts.join("\n\n")
}
