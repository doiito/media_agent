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

    // 如果有上传图片，注入到消息中
    let enhanced_message = if let Some(ref img_path) = req.image_path {
        format!("[img: {}] {}", img_path, req.message)
    } else {
        req.message.clone()
    };

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
        "backend_sample".to_string(),
        "list_nodes".to_string(),
        "interrupt".to_string(),
        "free_memory".to_string(),
        "health_check".to_string(),
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
