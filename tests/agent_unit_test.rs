// Agent 模块单元测试
// 测试 LlmConfig、LlmClient、ChatMessage、ToolDefinition 等组件

use comfyui_rust_agent::agent::llm::*;
use comfyui_rust_agent::types::*;
use std::collections::HashMap;

// ============================================================================
// LlmConfig 测试
// ============================================================================

#[test]
fn test_llm_config_default() {
    let config = LlmConfig::default();
    assert_eq!(config.base_url, "https://api.deepseek.com/v1");
    assert_eq!(config.default_model, "deepseek-chat");
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.max_retries, 3);
}

#[test]
fn test_llm_config_from_env() {
    std::env::remove_var("DEEPSEEK_API_URL");
    std::env::remove_var("DEEPSEEK_API_KEY");

    let config = LlmConfig::from_env();
    assert_eq!(config.base_url, "https://api.deepseek.com/v1");
    assert!(config.api_key.is_empty());
    assert!(!config.is_valid());

    std::env::set_var("DEEPSEEK_API_URL", "http://localhost:8081/v1");
    std::env::set_var("DEEPSEEK_API_KEY", "test-key");

    let config = LlmConfig::from_env();
    assert_eq!(config.base_url, "http://localhost:8081/v1");
    assert_eq!(config.api_key, "test-key");
    assert!(config.is_valid());

    std::env::remove_var("DEEPSEEK_API_URL");
    std::env::remove_var("DEEPSEEK_API_KEY");
}

// ============================================================================
// ChatMessage 测试
// ============================================================================

#[test]
fn test_chat_message_user() {
    let msg = ChatMessage::user("Hello");
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, Some("Hello".to_string()));
    assert!(msg.tool_calls.is_none());
}

#[test]
fn test_chat_message_system() {
    let msg = ChatMessage::system("You are helpful");
    assert_eq!(msg.role, "system");
    assert_eq!(msg.content, Some("You are helpful".to_string()));
}

#[test]
fn test_chat_message_assistant() {
    let msg = ChatMessage::assistant("Hi there");
    assert_eq!(msg.role, "assistant");
    assert_eq!(msg.content, Some("Hi there".to_string()));
}

#[test]
fn test_chat_message_tool_result() {
    let msg = ChatMessage::tool_result("call_123", "success");
    assert_eq!(msg.role, "tool");
    assert_eq!(msg.name, Some("call_123".to_string()));
}

// ============================================================================
// ToolDefinition 测试
// ============================================================================

#[test]
fn test_tool_definition_creation() {
    let tool = ToolDefinition::function(
        "build_t2i_workflow",
        "Build text-to-image workflow",
        serde_json::json!({"type": "object"}),
    );
    assert_eq!(tool.tool_type, "function");
    assert_eq!(tool.function.name, "build_t2i_workflow");
    assert_eq!(tool.function.description, "Build text-to-image workflow");
}

#[test]
fn test_tool_definition_serialization() {
    let tool = ToolDefinition::function(
        "test_tool",
        "A test tool",
        serde_json::json!({"type": "object", "properties": {}}),
    );

    let json = serde_json::to_string(&tool).unwrap();
    assert!(json.contains("\"type\":\"function\""));
    assert!(json.contains("test_tool"));
}

#[test]
fn test_get_comfyui_tools() {
    let tools = get_comfyui_tools();
    assert!(tools.len() >= 8);

    let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
    assert!(names.contains(&"build_t2i_workflow"));
    assert!(names.contains(&"build_i2i_workflow"));
    assert!(names.contains(&"submit_workflow"));
    assert!(names.contains(&"list_nodes"));
    assert!(names.contains(&"get_node_info"));
    assert!(names.contains(&"validate_workflow"));
    assert!(names.contains(&"get_status"));
    assert!(names.contains(&"estimate_quality"));
}

// ============================================================================
// FunctionCall 测试
// ============================================================================

#[test]
fn test_function_call_parse() {
    let func_call = FunctionCall {
        name: "build_t2i_workflow".to_string(),
        arguments: r#"{"prompt":"a cat","width":512}"#.to_string(),
    };

    #[derive(serde::Deserialize)]
    struct Args {
        prompt: String,
        width: i32,
    }

    let args: Args = func_call.parse_arguments().unwrap();
    assert_eq!(args.prompt, "a cat");
    assert_eq!(args.width, 512);
}

#[test]
fn test_function_call_parse_invalid() {
    let func_call = FunctionCall {
        name: "test".to_string(),
        arguments: "invalid json".to_string(),
    };

    #[derive(serde::Deserialize)]
    struct Args { x: i32 }

    let result: Result<Args, _> = func_call.parse_arguments();
    assert!(result.is_err());
}

// ============================================================================
// Error 类型测试
// ============================================================================

#[test]
fn test_error_status_codes() {
    assert_eq!(Error::ValidationFailed("test".to_string()).status_code(), 422);
    assert_eq!(Error::NotFound("test".to_string()).status_code(), 404);
    assert_eq!(Error::Timeout("test".to_string()).status_code(), 408);
    assert_eq!(Error::ServiceUnavailable("test".to_string()).status_code(), 503);
    assert_eq!(Error::BadRequest("test".to_string()).status_code(), 400);
    assert_eq!(Error::Unauthorized("test".to_string()).status_code(), 401);
    assert_eq!(Error::Forbidden("test".to_string()).status_code(), 403);
    assert_eq!(Error::Conflict("test".to_string()).status_code(), 409);
}

#[test]
fn test_error_retryable() {
    assert!(Error::Timeout("test".to_string()).is_retryable());
    assert!(Error::ServiceUnavailable("test".to_string()).is_retryable());
    assert!(Error::BackendError("test".to_string()).is_retryable());
    assert!(!Error::ValidationFailed("test".to_string()).is_retryable());
    assert!(!Error::NotFound("test".to_string()).is_retryable());
}

#[test]
fn test_error_codes() {
    assert_eq!(Error::ValidationFailed("test".to_string()).error_code(), "VALIDATION_FAILED");
    assert_eq!(Error::NotFound("test".to_string()).error_code(), "NOT_FOUND");
    assert_eq!(Error::Timeout("test".to_string()).error_code(), "TIMEOUT");
    assert_eq!(Error::BackendError("test".to_string()).error_code(), "BACKEND_ERROR");
}

// ============================================================================
// Value 类型测试
// ============================================================================

#[test]
fn test_value_as_str() {
    let v = Value::String("hello".to_string());
    assert_eq!(v.as_str().unwrap(), "hello");

    let v = Value::Int(42);
    assert!(v.as_str().is_err());
}

#[test]
fn test_value_as_ref_str() {
    let v = Value::Model("model_path".to_string());
    assert_eq!(v.as_ref_str().unwrap(), "model_path");

    let v = Value::Clip("clip_path".to_string());
    assert_eq!(v.as_ref_str().unwrap(), "clip_path");

    let v = Value::Vae("vae_path".to_string());
    assert_eq!(v.as_ref_str().unwrap(), "vae_path");

    let v = Value::String("test".to_string());
    assert_eq!(v.as_ref_str().unwrap(), "test");

    let v = Value::Int(42);
    assert!(v.as_ref_str().is_err());
}

#[test]
fn test_value_as_int() {
    let v = Value::Int(42);
    assert_eq!(v.as_int().unwrap(), 42);

    let v = Value::String("42".to_string());
    assert!(v.as_int().is_err());
}

#[test]
fn test_value_as_float() {
    let v = Value::Float(3.14);
    assert!((v.as_float().unwrap() - 3.14).abs() < 0.001);

    let v = Value::Int(42);
    assert!((v.as_float().unwrap() - 42.0).abs() < 0.001);

    let v = Value::String("3.14".to_string());
    assert!(v.as_float().is_err());
}

// ============================================================================
// WorkflowBuilder 测试
// ============================================================================

#[test]
fn test_workflow_builder_t2i() {
    let workflow = comfyui_rust_agent::workflow::WorkflowBuilder::text_to_image(
        "a cat".to_string(),
        "".to_string(),
        512, 512, 20, 7.0, 42,
        "model.safetensors".to_string(),
    );

    assert!(workflow.is_ok());
    let wf = workflow.unwrap();
    assert!(wf.nodes.len() >= 7);
    // Builder 把连接信息嵌入 node 的 InputValue::Link 中，而非顶层 links 数组
    let has_links = wf.nodes.values()
        .any(|n| n.inputs.values().any(|v| matches!(v, comfyui_rust_agent::types::InputValue::Link(_))));
    assert!(has_links, "workflow should contain link connections in node inputs");
}

// ============================================================================
// PromptTask 测试
// ============================================================================

#[test]
fn test_prompt_task_priority() {
    use comfyui_rust_agent::types::Workflow;

    let workflow = Workflow {
        nodes: HashMap::new(),
        links: vec![],
    };

    let task = PromptTask::new(workflow.clone(), "id1".to_string(), "client".to_string());
    assert_eq!(task.priority, 10);

    let task_high = PromptTask::new(workflow, "id2".to_string(), "client".to_string())
        .with_priority(0);
    assert_eq!(task_high.priority, 0);
}
