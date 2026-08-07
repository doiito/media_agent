// Agent HTTP API handlers
// /agent/chat, /agent/status, /agent/workflows 端点

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use crate::api::handlers::ApiState;
use crate::agent::context::{GenerationAudit, GenerationRequestContext};
use crate::backend::T2VParams;
use crate::types::{ExecutionResult, HistoryEntry, Value, Workflow};
use crate::workflow::builder::WorkflowBuilder;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalGenerationRoute {
    intent: &'static str,
}

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
    pub execution_mode: String,
    pub artifact_verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_audit: Option<GenerationAudit>,
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
    pub llm_provider: String,
    pub llm_base_url: String,
    pub llm_reachable: bool,
    pub native_runtime_ready: bool,
    pub native_image_ready: bool,
    pub native_video_ready: bool,
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
    let app_config = {
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
        agent_guard.context().app_config.clone()
    };
    if let Err(error) = validate_request_preflight(&req, &app_config) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "Native request preflight failed",
                "message": error,
                "status": "failed"
            })),
        )
            .into_response();
    }
    if !probe_agent_llm(&app_config).await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Agent LLM provider is unavailable",
                "provider": format!("{:?}", app_config.agent.llm.provider),
                "base_url": app_config.agent.llm.gateway_base_url()
            })),
        )
            .into_response();
    }
    // 构建增强消息：将图片路径和参数以结构化方式注入，让 PA/DA 能正确解析
    let enhanced_message = build_enhanced_message(&req);
    let generation_request = GenerationRequestContext {
        user_request: req.message.clone(),
        intent: classify_local_generation_request(&req).intent.to_string(),
        image_path: req.image_path.clone(),
        params: req.params.clone().unwrap_or_else(|| serde_json::json!({})),
    };

    let (result, generation_audit) = {
        let mut agent_guard = agent.lock().await;
        {
            let mut active_request = agent_guard.context().generation_request.write().await;
            *active_request = Some(generation_request);
        }
        {
            let mut active_audit = agent_guard.context().generation_audit.write().await;
            *active_audit = None;
        }
        let result = agent_guard.process_task(&enhanced_message, req.workflow.as_deref()).await;
        let generation_audit = agent_guard.context().generation_audit.read().await.clone();
        {
            let mut active_request = agent_guard.context().generation_request.write().await;
            *active_request = None;
        }
        (result, generation_audit)
    };

    match result {
        Ok((task_id, task_result)) => {
            let mut media_paths = Vec::new();
            for artifact in &task_result.artifacts {
                collect_json_media_paths(artifact, &mut media_paths);
            }
            media_paths.sort();
            media_paths.dedup();
            if !media_paths.is_empty() {
                let mut node_output = std::collections::HashMap::new();
                node_output.insert(
                    "files".to_string(),
                    Value::Array(media_paths.into_iter().map(Value::String).collect()),
                );
                let mut outputs = std::collections::HashMap::new();
                outputs.insert("agent_result".to_string(), node_output);
                record_history(
                    &state,
                    &task_id,
                    Workflow { nodes: std::collections::HashMap::new(), links: vec![] },
                    outputs,
                    "success",
                )
                .await;
            }
            let output = if task_result.artifacts.is_empty() {
                task_result.output.clone().or_else(|| task_result.jsonld_output.clone())
            } else {
                Some(serde_json::json!({
                    "result": task_result.output,
                    "artifacts": task_result.artifacts
                }))
            };
            let response = AgentChatResponse {
                task_id,
                status: task_result.status,
                summary: task_result.summary,
                output,
                turn_count: task_result.turn_count,
                tool_calls: task_result.tool_call_count,
                errors: task_result.errors,
                execution_mode: "gliding_horse".to_string(),
                artifact_verified: true,
                generation_audit,
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

/// POST /generate - 基础图片/视频生成直连接口
///
/// 该接口不依赖 Agent/PDCA/LLM，供 Web UI 直接调用。
pub async fn agent_generate(
    State(state): State<ApiState>,
    Json(req): Json<AgentChatRequest>,
) -> Response {
    let app_config = {
        let agent = state.agent.lock().await;
        agent.context().app_config.clone()
    };
    if let Err(error) = validate_request_preflight(&req, &app_config) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "Native request preflight failed",
                "message": error,
                "status": "failed"
            })),
        )
            .into_response();
    }
    match try_local_generation(&state, &req).await {
        Some(response) => response.into_response(),
        None => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Unsupported request",
                "message": "Could not classify generation intent"
            })),
        )
            .into_response(),
    }
}

async fn try_local_generation(
    state: &ApiState,
    req: &AgentChatRequest,
) -> Option<(StatusCode, Json<serde_json::Value>)> {
    if req.workflow.is_some() {
        return None;
    }

    let route = classify_local_generation_request(req);
    let result = match route.intent {
        "text_to_image" => execute_local_text_to_image(state, req).await,
        "image_to_image" => execute_local_image_to_image(state, req).await,
        "image_to_video" => execute_local_image_to_video(state, req).await,
        "text_to_video" => execute_local_text_to_video(state, req).await,
        _ => return None,
    };

    Some(match result {
        Ok(response) => (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_default())),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": error,
                "task_id": "",
                "status": "failed"
            })),
        ),
    })
}

/// GET /agent/status - 获取 Agent 状态
pub async fn agent_status(
    State(state): State<ApiState>,
) -> impl IntoResponse {
    let agent = state.agent.clone();
    let agent_guard = agent.lock().await;
    let status = agent_guard.status();
    let app_config = &agent_guard.context().app_config;
    let runtime = crate::native_runtime::NativeRuntimeReport::inspect(app_config);
    let check_ready = |id: &str| {
        runtime.checks.iter().any(|check| {
            check.id == id
                && check.status == crate::native_runtime::RuntimeCheckStatus::Ready
        })
    };
    let native_core_ready = if app_config.sd_cpp.execution_mode == "native_worker" {
        check_ready("stable_diffusion_rust_worker")
            && check_ready("stable_diffusion_c_bridge")
    } else {
        check_ready("stable_diffusion_executable")
    };
    let native_image_ready = native_core_ready && check_ready("default_diffusion_model");
    let default_video_is_svd = app_config
        .sd_cpp
        .video_model_path
        .to_ascii_lowercase()
        .contains("svd");
    let native_video_ready = native_core_ready
        && check_ready("default_video_model")
        && if default_video_is_svd {
            check_ready("clip_vision_model")
        } else {
            check_ready("video_t5xxl_model") && check_ready("video_vae_model")
        };
    let llm_reachable = probe_agent_llm(app_config).await;

    let workflows = crate::agent::workflow::load_workflow_templates("workflows")
        .unwrap_or_default();

    let skills = vec![
        "text_to_image".to_string(),
        "image_to_image".to_string(),
        "generate_video".to_string(),
    ];

    let mut tools = vec![
        "inspect_native_runtime".to_string(),
        "validate_model".to_string(),
        "generate_media".to_string(),
        "inspect_artifact".to_string(),
        "discover_comfyui_skills".to_string(),
        "recommend_parameters".to_string(),
        "analyze_failure".to_string(),
        "record_execution".to_string(),
        "find_similar_workflows".to_string(),
        "get_skill_stats".to_string(),
    ];
    if app_config.agent.compatibility_tools_enabled {
        tools.extend([
            "submit_workflow",
            "build_t2i_workflow",
            "build_i2i_workflow",
            "build_i2v_workflow",
            "backend_sample",
            "list_nodes",
            "interrupt",
            "free_memory",
            "health_check",
            "list_available_nodes",
            "create_node",
            "connect_nodes",
            "find_compatible_sources",
            "validate_workflow",
            "suggest_workflow",
            "get_node_schema",
        ]
        .into_iter()
        .map(str::to_string));
    }

    let response = AgentStatusResponse {
        ready: status.context_ready && status.supervisor_ready && llm_reachable,
        context_ready: status.context_ready,
        supervisor_ready: status.supervisor_ready,
        workflows,
        skills,
        tools,
        llm_provider: format!("{:?}", app_config.agent.llm.provider),
        llm_base_url: app_config.agent.llm.gateway_base_url(),
        llm_reachable,
        native_runtime_ready: runtime.ready,
        native_image_ready,
        native_video_ready,
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_default()))
}

async fn probe_agent_llm(config: &crate::config::AppConfig) -> bool {
    match config.agent.llm.provider {
        crate::config::AgentLlmProvider::Deepseek => {
            !config.agent.llm.api_key.trim().is_empty()
        }
        crate::config::AgentLlmProvider::LlamaCpp => {
            let url = format!("{}/health", config.agent.llm.gateway_base_url());
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(2))
                .build()
            {
                Ok(client) => client,
                Err(_) => return false,
            };
            client
                .get(url)
                .send()
                .await
                .map(|response| response.status().is_success())
                .unwrap_or(false)
        }
    }
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
            parts.push(format!("<ui_generation_parameters>\n{}\nThe keys listed in _explicit_keys were changed by the user and are constraints. Other values are UI defaults; explicit dimensions, duration, or task type in user_request take precedence.\n</ui_generation_parameters>",
                serde_json::to_string_pretty(params).unwrap_or_default()));
        }
    }

    // 原始用户消息
    parts.push(format!("<user_request>\n{}\n</user_request>", req.message));

    parts.join("\n\n")
}

fn classify_local_generation_request(req: &AgentChatRequest) -> LocalGenerationRoute {
    let requested_intent = get_param_string(req, "intent", "auto");
    if matches!(
        requested_intent.as_str(),
        "text_to_image" | "image_to_image" | "text_to_video" | "image_to_video"
    ) {
        return LocalGenerationRoute {
            intent: match requested_intent.as_str() {
                "text_to_image" => "text_to_image",
                "image_to_image" => "image_to_image",
                "text_to_video" => "text_to_video",
                "image_to_video" => "image_to_video",
                _ => unreachable!(),
            },
        };
    }

    let msg = req.message.to_lowercase();
    let has_image = req.image_path.is_some();
    let has_still_output = [
        "一张", "图片", "图像", "照片", "海报", "封面", "插画", "image", "photo", "poster", "cover art",
    ]
    .iter()
    .any(|kw| msg.contains(kw));
    let has_strong_video_output = [
        "短视频", "视频片段", "生成视频", "生成一段视频", "制作视频", "做成视频", "转成视频", "图生视频",
        "文生视频", "动画", "动起来", "video clip", "generate a video", "create a video", "make a video",
        "image to video", "text to video", "animate", "animation",
    ]
    .iter()
    .any(|kw| msg.contains(kw));
    let mentions_video = msg.contains("视频") || msg.contains("video");
    let has_motion = ["跳舞", "舞动", "运动", "dance", "dancing", "moving"]
        .iter()
        .any(|kw| msg.contains(kw));
    let has_duration = msg.contains('秒') || msg.contains(" second");
    let looks_like_video = has_strong_video_output
        || (!has_still_output
            && (mentions_video || (has_image && has_motion) || (has_duration && has_motion)));

    let intent = if has_image && looks_like_video {
        "image_to_video"
    } else if has_image {
        "image_to_image"
    } else if looks_like_video {
        "text_to_video"
    } else {
        "text_to_image"
    };

    LocalGenerationRoute { intent }
}

fn validate_request_preflight(
    req: &AgentChatRequest,
    config: &crate::config::AppConfig,
) -> Result<(), String> {
    let route = classify_local_generation_request(req);
    let is_video = matches!(route.intent, "text_to_video" | "image_to_video");
    let explicit_model = get_param_string(req, "model", "");
    let configured_model = if is_video {
        &config.sd_cpp.video_model_path
    } else {
        &config.sd_cpp.model_path
    };
    let requested_model = if explicit_model.trim().is_empty() {
        configured_model.as_str()
    } else {
        explicit_model.trim()
    };
    let model_path = resolve_model_for_preflight(requested_model);
    validate_inference_model(&model_path, if is_video { "video" } else { "image" })?;

    if matches!(route.intent, "image_to_image" | "image_to_video") {
        let image_path = req
            .image_path
            .as_deref()
            .ok_or_else(|| format!("{} requires an uploaded input image", route.intent))?;
        let image_path = normalize_image_path(image_path);
        image::image_dimensions(&image_path).map_err(|error| {
            format!("Input image '{}' is missing or invalid: {}", image_path, error)
        })?;
    }

    let normalized_video_model = model_path.to_string_lossy().to_ascii_lowercase();
    let is_svd = normalized_video_model.contains("svd")
        || normalized_video_model.contains("stable-video-diffusion");
    if is_video && is_svd {
        let clip_path = resolve_model_for_preflight(&config.sd_cpp.clip_vision_path);
        validate_inference_model(&clip_path, "SVD CLIP Vision")?;
        if route.intent == "text_to_video" {
            let image_model = resolve_model_for_preflight(&config.sd_cpp.model_path);
            validate_inference_model(&image_model, "text-to-video first-frame")?;
        }
    } else if is_video {
        let t5xxl_path = resolve_model_for_preflight(&config.sd_cpp.video_t5xxl_path);
        validate_inference_model(&t5xxl_path, "video T5 text encoder")?;
        let vae_path = resolve_model_for_preflight(&config.sd_cpp.video_vae_path);
        validate_inference_model(&vae_path, "video VAE")?;
        if !config.sd_cpp.video_high_noise_model_path.trim().is_empty() {
            let high_noise =
                resolve_model_for_preflight(&config.sd_cpp.video_high_noise_model_path);
            validate_inference_model(&high_noise, "high-noise video")?;
        }
    }
    Ok(())
}

fn resolve_model_for_preflight(value: &str) -> std::path::PathBuf {
    let direct = std::path::PathBuf::from(value);
    if direct.is_file() || direct.is_absolute() {
        return direct;
    }
    let checkpoint = std::path::Path::new("models/checkpoints").join(value);
    if checkpoint.is_file() {
        return checkpoint;
    }
    std::path::Path::new("models/diffusers").join(value)
}

fn validate_inference_model(path: &std::path::Path, role: &str) -> Result<(), String> {
    let info = crate::native_runtime::inspect_model_file(path)
        .map_err(|error| format!("{} model preflight failed: {}", role, error))?;
    if !matches!(
        info.container,
        crate::native_runtime::ModelContainer::Safetensors
            | crate::native_runtime::ModelContainer::Gguf
    ) {
        return Err(format!(
            "{} model '{}' has unsupported or corrupt {:?} container",
            role,
            path.display(),
            info.container
        ));
    }
    Ok(())
}

fn get_param_u64(req: &AgentChatRequest, key: &str, default: u64) -> u64 {
    req.params
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

fn get_param_f64(req: &AgentChatRequest, key: &str, default: f64) -> f64 {
    req.params
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

fn get_param_i64(req: &AgentChatRequest, key: &str, default: i64) -> i64 {
    req.params
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_i64())
        .unwrap_or(default)
}

fn get_param_string(req: &AgentChatRequest, key: &str, default: &str) -> String {
    req.params
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

fn normalize_image_path(path: &str) -> String {
    if std::path::Path::new(path).exists() || path.starts_with("input/") || path.starts_with("input\\") {
        path.to_string()
    } else {
        format!("input/{}", path)
    }
}

fn ensure_non_negative_seed(seed: i64) -> i64 {
    if seed < 0 {
        rand::random::<u32>() as i64
    } else {
        seed
    }
}

fn extract_output_files(outputs: &std::collections::HashMap<String, std::collections::HashMap<String, Value>>) -> Vec<String> {
    let mut files = Vec::new();
    for node_outputs in outputs.values() {
        for value in node_outputs.values() {
            match value {
                Value::String(s) if is_media_path(s) => files.push(s.clone()),
                Value::Array(items) => {
                    for item in items {
                        if let Value::String(s) = item {
                            if is_media_path(s) {
                                files.push(s.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    files.sort();
    files.dedup();
    files
}

fn is_media_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm"].iter().any(|ext| lower.ends_with(ext))
}

fn collect_json_media_paths(value: &serde_json::Value, paths: &mut Vec<String>) {
    match value {
        serde_json::Value::String(path) if is_media_path(path) => paths.push(path.clone()),
        serde_json::Value::Array(values) => {
            for value in values {
                collect_json_media_paths(value, paths);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                collect_json_media_paths(value, paths);
            }
        }
        _ => {}
    }
}

async fn execute_local_workflow(
    state: &ApiState,
    workflow: Workflow,
    summary: String,
) -> Result<AgentChatResponse, String> {
    state.increment_requests().await;
    state.increment_workflows().await;

    let workflow_result = {
        let mut engine = state.engine.lock().await;
        let prompt_id = engine
            .submit(workflow.clone(), "web-local".to_string())
            .await
            .map_err(|e| format!("Failed to submit workflow: {}", e))?;

        state
            .event_bus
            .publish(crate::execution::Event::ExecutionStart {
                prompt_id: prompt_id.clone(),
            })
            .await;

        let result = engine
            .execute_next()
            .await
            .map_err(|e| format!("Failed to execute workflow: {}", e))?;
        (prompt_id, result)
    };
    let (prompt_id, result) = workflow_result;

    match result {
        Some(ExecutionResult::Success(outputs)) => {
            let files = extract_output_files(&outputs);
            if files.is_empty() {
                state.record_workflow_failure().await;
                state
                    .event_bus
                    .publish(crate::execution::Event::ExecutionError {
                        prompt_id: prompt_id.clone(),
                        error: "Workflow completed but produced no media files".to_string(),
                    })
                    .await;
                return Err("Workflow completed but produced no media files".to_string());
            }

            state.record_workflow_success().await;
            record_history(state, &prompt_id, workflow, outputs.clone(), "success").await;
            state
                .event_bus
                .publish(crate::execution::Event::ExecutionSuccess {
                    prompt_id: prompt_id.clone(),
                    outputs,
                })
                .await;
            Ok(AgentChatResponse {
                task_id: prompt_id,
                status: "completed".to_string(),
                summary,
                output: Some(serde_json::json!({
                    "files": files,
                    "intent": "workflow"
                })),
                turn_count: 0,
                tool_calls: 0,
                errors: vec![],
                execution_mode: "direct".to_string(),
                artifact_verified: true,
                generation_audit: None,
            })
        }
        Some(ExecutionResult::Failure(err)) => {
            state.record_workflow_failure().await;
            state
                .event_bus
                .publish(crate::execution::Event::ExecutionError {
                    prompt_id: prompt_id.clone(),
                    error: err.clone(),
                })
                .await;
            Err(err)
        }
        Some(ExecutionResult::Pending) => {
            state.record_workflow_failure().await;
            state
                .event_bus
                .publish(crate::execution::Event::ExecutionError {
                    prompt_id: prompt_id.clone(),
                    error: "Workflow execution is still pending".to_string(),
                })
                .await;
            Err("Workflow execution is still pending".to_string())
        }
        None => {
            state.record_workflow_failure().await;
            state
                .event_bus
                .publish(crate::execution::Event::ExecutionError {
                    prompt_id: prompt_id.clone(),
                    error: "Workflow queue returned no execution result".to_string(),
                })
                .await;
            Err("Workflow queue returned no execution result".to_string())
        }
    }
}

async fn execute_local_text_to_image(state: &ApiState, req: &AgentChatRequest) -> Result<AgentChatResponse, String> {
    let workflow = WorkflowBuilder::text_to_image(
        req.message.clone(),
        get_param_string(req, "negative_prompt", ""),
        get_param_u64(req, "width", 512) as usize,
        get_param_u64(req, "height", 512) as usize,
        get_param_u64(req, "steps", 20) as usize,
        get_param_f64(req, "cfg", 7.0) as f32,
        ensure_non_negative_seed(get_param_i64(req, "seed", -1)) as usize,
        get_param_string(req, "model", "v1-5-pruned-emaonly.safetensors"),
    )
    .map_err(|e| format!("Failed to build text-to-image workflow: {}", e))?;

    execute_local_workflow(state, workflow, "Text-to-image generation completed.".to_string()).await
}

async fn execute_local_image_to_image(state: &ApiState, req: &AgentChatRequest) -> Result<AgentChatResponse, String> {
    let image_path = req
        .image_path
        .as_deref()
        .ok_or_else(|| "Image-to-image requires an uploaded image".to_string())?;
    let workflow = WorkflowBuilder::image_to_image(
        req.message.clone(),
        get_param_string(req, "negative_prompt", ""),
        normalize_image_path(image_path),
        get_param_f64(req, "strength", 0.75) as f32,
        get_param_u64(req, "steps", 20) as usize,
        get_param_f64(req, "cfg", 7.0) as f32,
        ensure_non_negative_seed(get_param_i64(req, "seed", -1)) as usize,
        get_param_string(req, "model", "v1-5-pruned-emaonly.safetensors"),
    )
    .map_err(|e| format!("Failed to build image-to-image workflow: {}", e))?;

    execute_local_workflow(state, workflow, "Image-to-image generation completed.".to_string()).await
}

async fn execute_local_image_to_video(state: &ApiState, req: &AgentChatRequest) -> Result<AgentChatResponse, String> {
    let image_path = req
        .image_path
        .as_deref()
        .ok_or_else(|| "Image-to-video requires an uploaded image".to_string())?;
    let default_video_model = {
        let agent = state.agent.lock().await;
        agent.context().app_config.sd_cpp.video_model_path.clone()
    };
    let default_video_cfg = if default_video_model.to_ascii_lowercase().contains("svd") {
        3.0
    } else {
        6.0
    };
    let workflow = WorkflowBuilder::image_to_video(
        normalize_image_path(image_path),
        get_param_string(req, "model", &default_video_model),
        req.message.clone(),
        get_param_string(req, "negative_prompt", ""),
        get_param_u64(req, "width", 1024) as usize,
        get_param_u64(req, "height", 576) as usize,
        get_param_u64(req, "frames", 25) as usize,
        get_param_u64(req, "fps", 5) as usize,
        get_param_i64(req, "motion_bucket_id", 127) as i32,
        get_param_f64(req, "cfg", default_video_cfg) as f32,
        get_param_f64(req, "min_cfg", 1.0) as f32,
        get_param_f64(req, "noise_aug_strength", 0.02) as f32,
        get_param_u64(req, "steps", 25) as usize,
        ensure_non_negative_seed(get_param_i64(req, "seed", -1)),
    )
    .map_err(|e| format!("Failed to build image-to-video workflow: {}", e))?;

    execute_local_workflow(state, workflow, "Image-to-video generation completed.".to_string()).await
}

async fn execute_local_text_to_video(state: &ApiState, req: &AgentChatRequest) -> Result<AgentChatResponse, String> {
    state.increment_requests().await;
    state.increment_workflows().await;

    let prompt_id = uuid::Uuid::new_v4().to_string();
    state
        .event_bus
        .publish(crate::execution::Event::ExecutionStart {
            prompt_id: prompt_id.clone(),
        })
        .await;
    let video_data = state
        .backend_router
        .text_to_video(T2VParams {
            prompt: req.message.clone(),
            negative_prompt: get_param_string(req, "negative_prompt", ""),
            width: get_param_u64(req, "width", 512) as usize,
            height: get_param_u64(req, "height", 512) as usize,
            frames: get_param_u64(req, "frames", 25) as usize,
            fps: get_param_u64(req, "fps", 5) as usize,
            motion_bucket_id: get_param_i64(req, "motion_bucket_id", 127) as i32,
            steps: get_param_u64(req, "steps", 20) as usize,
            cfg: get_param_f64(req, "cfg", 6.0) as f32,
            min_cfg: get_param_f64(req, "min_cfg", 1.0) as f32,
            noise_aug_strength: get_param_f64(req, "noise_aug_strength", 0.02) as f32,
            seed: ensure_non_negative_seed(get_param_i64(req, "seed", -1)) as usize,
            model_path: get_param_string(req, "model", ""),
        })
        .await;
    let video_data = match video_data {
        Ok(data) => data,
        Err(e) => {
            let msg = format!("Text-to-video generation failed: {}", e);
            state.record_workflow_failure().await;
            state
                .event_bus
                .publish(crate::execution::Event::ExecutionError {
                    prompt_id: prompt_id.clone(),
                    error: msg.clone(),
                })
                .await;
            return Err(msg);
        }
    };

    std::fs::create_dir_all(&state.output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;
    let filename = format!("text_to_video_{}.mp4", prompt_id);
    let output_path = std::path::Path::new(&state.output_dir).join(&filename);
    std::fs::write(&output_path, &video_data)
        .map_err(|e| format!("Failed to save generated video: {}", e))?;

    let outputs = std::collections::HashMap::from([(
        "result".to_string(),
        std::collections::HashMap::from([
            ("video".to_string(), Value::String(filename.clone())),
            ("size_bytes".to_string(), Value::Int(video_data.len() as i64)),
        ]),
    )]);

    state.record_workflow_success().await;
    record_history(
        state,
        &prompt_id,
        Workflow {
            nodes: std::collections::HashMap::new(),
            links: vec![],
        },
        outputs.clone(),
        "success",
    )
    .await;
    state
        .event_bus
        .publish(crate::execution::Event::ExecutionSuccess {
            prompt_id: prompt_id.clone(),
            outputs: outputs.clone(),
        })
        .await;

    Ok(AgentChatResponse {
        task_id: prompt_id,
        status: "completed".to_string(),
        summary: "Text-to-video generation completed.".to_string(),
        output: Some(serde_json::json!({
            "files": [filename],
            "intent": "text_to_video"
        })),
        turn_count: 0,
        tool_calls: 0,
        errors: vec![],
        execution_mode: "direct".to_string(),
        artifact_verified: true,
        generation_audit: None,
    })
}

async fn record_history(
    state: &ApiState,
    prompt_id: &str,
    workflow: Workflow,
    outputs: std::collections::HashMap<String, std::collections::HashMap<String, Value>>,
    status: &str,
) {
    let entry = HistoryEntry {
        prompt_id: prompt_id.to_string(),
        workflow,
        outputs,
        status: status.to_string(),
        start_time: 0.0,
        end_time: Some(0.0),
    };
    state.history.write().await.insert(prompt_id.to_string(), entry);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_text_to_image_without_input_image() {
        let req = AgentChatRequest {
            message: "生成一张赛博朋克猫咪海报".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: None,
            params: None,
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "text_to_image");
    }

    #[test]
    fn classifies_image_to_video_when_image_and_motion_words_present() {
        let req = AgentChatRequest {
            message: "让这张图里的胖子跳舞，生成5秒短视频".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: Some("input/bk_0019.jpg".to_string()),
            params: None,
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "image_to_video");
    }

    #[test]
    fn classifies_text_to_video_without_input_image() {
        let req = AgentChatRequest {
            message: "生成一个5秒钟的猫咪跳舞视频".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: None,
            params: None,
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "text_to_video");
    }

    #[test]
    fn explicit_ui_intent_overrides_keyword_classification() {
        let req = AgentChatRequest {
            message: "生成一张包含视频播放器的海报".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: None,
            params: Some(serde_json::json!({
                "intent": "text_to_image",
                "_explicit_keys": ["intent"]
            })),
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "text_to_image");
    }

    #[test]
    fn auto_intent_does_not_treat_video_as_a_subject_as_video_output() {
        let req = AgentChatRequest {
            message: "生成一张包含视频播放器的海报".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: None,
            params: None,
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "text_to_image");
    }

    #[test]
    fn auto_intent_respects_still_output_even_when_subject_is_dancing() {
        let req = AgentChatRequest {
            message: "把图中跳舞的人制作成一张海报".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: Some("input/bk_0019.jpg".to_string()),
            params: None,
        };

        let route = classify_local_generation_request(&req);
        assert_eq!(route.intent, "image_to_image");
    }

    #[test]
    fn enhanced_message_marks_dynamic_ui_constraints() {
        let req = AgentChatRequest {
            message: "生成5秒视频".to_string(),
            workflow: None,
            max_iterations: None,
            client_id: None,
            image_path: None,
            params: Some(serde_json::json!({
                "frames": 30,
                "fps": 6,
                "_explicit_keys": ["frames", "fps"]
            })),
        };

        let message = build_enhanced_message(&req);
        assert!(message.contains("ui_generation_parameters"));
        assert!(message.contains("_explicit_keys"));
        assert!(message.contains("30"));
    }
}
