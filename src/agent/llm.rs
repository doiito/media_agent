// LLM 客户端模块
// 支持 DeepSeek API（通过环境变量配置）和本地 llama.cpp server
// OpenAI 兼容 API

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use log::{info, warn, error, debug};

/// LLM 配置
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// API base URL（从 DEEPSEEK_API_URL 环境变量获取）
    pub base_url: String,
    /// API key（从 DEEPSEEK_API_KEY 环境变量获取）
    pub api_key: String,
    /// 默认模型
    pub default_model: String,
    /// 备用模型
    pub fallback_model: Option<String>,
    /// 超时时间（秒）
    pub timeout_seconds: u64,
    /// 最大重试次数
    pub max_retries: u32,
    /// 重试间隔（秒）
    pub retry_delay_seconds: u64,
    /// 最大 tokens
    pub max_tokens: u32,
    /// 温度
    pub temperature: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.deepseek.com/v1".to_string(),
            api_key: "".to_string(),
            default_model: "deepseek-chat".to_string(),
            fallback_model: Some("deepseek-reasoner".to_string()),
            timeout_seconds: 120,
            max_retries: 3,
            retry_delay_seconds: 2,
            max_tokens: 4096,
            temperature: 0.7,
        }
    }
}

impl LlmConfig {
    /// 从环境变量加载配置
    pub fn from_env() -> Self {
        let base_url = std::env::var("DEEPSEEK_API_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
        
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .unwrap_or_else(|_| {
                warn!("DEEPSEEK_API_KEY not set, LLM calls may fail");
                "".to_string()
            });
        
        let default_model = std::env::var("DEEPSEEK_MODEL")
            .unwrap_or_else(|_| "deepseek-chat".to_string());
        
        info!("LLM config loaded: base_url={}, model={}", base_url, default_model);
        
        Self {
            base_url,
            api_key,
            default_model,
            fallback_model: Some("deepseek-reasoner".to_string()),
            timeout_seconds: 120,
            max_retries: 3,
            retry_delay_seconds: 2,
            max_tokens: 4096,
            temperature: 0.7,
        }
    }
    
    /// 检查配置是否有效（API key 已设置）
    pub fn is_valid(&self) -> bool {
        !self.api_key.is_empty() && !self.base_url.is_empty()
    }
}

/// LLM 客户端
pub struct LlmClient {
    config: LlmConfig,
    http_client: Client,
}

impl LlmClient {
    /// 创建 LLM 客户端
    pub fn new(config: LlmConfig) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            config,
            http_client,
        }
    }
    
    /// 从环境变量创建客户端
    pub fn from_env() -> Self {
        Self::new(LlmConfig::from_env())
    }
    
    /// 发送聊天请求
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Box::pin(self.chat_with_retry(request, 0)).await
    }
    
    /// 带重试的聊天请求
    async fn chat_with_retry(&self, request: ChatRequest, retry_count: u32) -> Result<ChatResponse, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        
        debug!("Sending chat request to {} (model: {}, retry: {})", 
               url, request.model, retry_count);
        
        let response = self.http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                let status = resp.status();
                
                if status.is_success() {
                    let body = resp.text().await
                        .map_err(|e| LlmError::NetworkError(e.to_string()))?;
                    
                    let chat_response: ChatResponse = serde_json::from_str(&body)
                        .map_err(|e| LlmError::ParseError(e.to_string()))?;
                    
                    info!("Chat response received: {} choices", chat_response.choices.len());
                    Ok(chat_response)
                } else {
                    let error_body = resp.text().await
                        .map_err(|e| LlmError::NetworkError(e.to_string()))?;
                    
                    error!("LLM API error: status={}, body={}", status, error_body);
                    
                    // 可重试的错误
                    if status.as_u16() == 429 || status.as_u16() >= 500 {
                        if retry_count < self.config.max_retries {
                            warn!("Retrying LLM request (attempt {})", retry_count + 1);
                            tokio::time::sleep(Duration::from_secs(self.config.retry_delay_seconds)).await;
                            return Box::pin(self.chat_with_retry(request, retry_count + 1)).await;
                        }
                    }
                    
                    Err(LlmError::ApiError(status.as_u16(), error_body))
                }
            }
            Err(e) => {
                error!("Network error: {}", e);
                
                // 网络错误可重试
                if retry_count < self.config.max_retries {
                    warn!("Retrying LLM request due to network error (attempt {})", retry_count + 1);
                    tokio::time::sleep(Duration::from_secs(self.config.retry_delay_seconds)).await;
                    return Box::pin(self.chat_with_retry(request, retry_count + 1)).await;
                }
                
                Err(LlmError::NetworkError(e.to_string()))
            }
        }
    }
    
    /// 简单聊天（单轮，无工具）
    pub async fn simple_chat(&self, message: &str) -> Result<String, LlmError> {
        let request = ChatRequest {
            model: self.config.default_model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(message.to_string()),
                tool_calls: None,
                name: None,
            }],
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            tools: None,
            tool_choice: None,
            stream: Some(false),
        };
        
        let response = self.chat(request).await?;
        
        response.choices.first()
            .map(|c| c.message.content.clone().unwrap_or_default())
            .ok_or_else(|| LlmError::ParseError("No choices in response".to_string()))
    }
    
    /// 带工具的聊天（用于 Agent ReAct 循环）
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest {
            model: self.config.default_model.clone(),
            messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            tools: Some(tools),
            tool_choice: Some("auto".to_string()),
            stream: Some(false),
        };
        
        self.chat(request).await
    }
    
    /// 使用备用模型聊天
    pub async fn chat_with_fallback(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        // 先尝试主模型
        let primary_result = self.chat(request.clone()).await;
        
        if primary_result.is_ok() {
            return primary_result;
        }
        
        // 如果失败且有备用模型，尝试备用
        if let Some(fallback_model) = &self.config.fallback_model {
            warn!("Primary model failed, trying fallback model: {}", fallback_model);
            
            let fallback_request = ChatRequest {
                model: fallback_model.clone(),
                ..request
            };
            
            self.chat(fallback_request).await
        } else {
            primary_result
        }
    }
    
    /// 获取配置
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }
}

/// LLM 错误
#[derive(Debug)]
pub enum LlmError {
    /// API 错误（状态码 + 响应体）
    ApiError(u16, String),
    /// 网络错误
    NetworkError(String),
    /// 解析错误
    ParseError(String),
    /// 配置错误
    ConfigError(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::ApiError(status, body) => write!(f, "API error ({}): {}", status, body),
            LlmError::NetworkError(e) => write!(f, "Network error: {}", e),
            LlmError::ParseError(e) => write!(f, "Parse error: {}", e),
            LlmError::ConfigError(e) => write!(f, "Config error: {}", e),
        }
    }
}

impl std::error::Error for LlmError {}

// ============================================================================
// OpenAI 兼容 API 数据结构
// ============================================================================

/// 聊天请求
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// 创建用户消息
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            name: None,
        }
    }
    
    /// 创建助手消息
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            name: None,
        }
    }
    
    /// 创建系统消息
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            name: None,
        }
    }
    
    /// 创建工具调用结果消息
    pub fn tool_result(tool_call_id: &str, result: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(result.to_string()),
            tool_calls: None,
            name: Some(tool_call_id.to_string()),
        }
    }
}

/// 聊天响应
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Usage,
}

/// 聊天选择
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// 使用统计
#[derive(Debug, Default, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    pub fn function(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
            },
        }
    }
}

/// 函数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// 工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

/// 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON 字符串
}

impl FunctionCall {
    /// 解析参数
    pub fn parse_arguments<T: for<'de> Deserialize<'de>>(&self) -> Result<T, LlmError> {
        serde_json::from_str(&self.arguments)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse tool arguments: {}", e)))
    }
}

// ============================================================================
// ComfyUI Agent 工具定义
// ============================================================================

/// 获取 ComfyUI Agent 工具定义列表
pub fn get_comfyui_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::function(
            "build_t2i_workflow",
            "构建文生图工作流（text-to-image）。参数：prompt（描述）、width/height（尺寸）、steps（步数）、cfg（引导强度）、seed（随机种子）",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图片描述，如 'a cyberpunk cat'"
                    },
                    "width": {
                        "type": "integer",
                        "description": "图片宽度，必须是 8 的倍数，默认 512",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "description": "图片高度，必须是 8 的倍数，默认 512",
                        "default": 512
                    },
                    "steps": {
                        "type": "integer",
                        "description": "采样步数，默认 20",
                        "default": 20
                    },
                    "cfg": {
                        "type": "number",
                        "description": "CFG 引导强度，默认 7.0",
                        "default": 7.0
                    },
                    "seed": {
                        "type": "integer",
                        "description": "随机种子，默认 0",
                        "default": 0
                    },
                    "model": {
                        "type": "string",
                        "description": "模型名称，默认 'v1-5-pruned-emaonly.safetensors'",
                        "default": "v1-5-pruned-emaonly.safetensors"
                    }
                },
                "required": ["prompt"]
            }),
        ),
        ToolDefinition::function(
            "build_i2i_workflow",
            "构建图生图工作流（image-to-image）。参数：prompt、input_image、strength、尺寸等",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "风格转换描述"
                    },
                    "input_image": {
                        "type": "string",
                        "description": "输入图片路径"
                    },
                    "strength": {
                        "type": "number",
                        "description": "转换强度 0-1，默认 0.7",
                        "default": 0.7
                    },
                    "width": {
                        "type": "integer",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "default": 512
                    },
                    "steps": {
                        "type": "integer",
                        "default": 15
                    },
                    "seed": {
                        "type": "integer",
                        "default": 0
                    }
                },
                "required": ["prompt", "input_image"]
            }),
        ),
        ToolDefinition::function(
            "submit_workflow",
            "提交工作流到执行引擎。参数：workflow（JSON 工作流定义）",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "ComfyUI 工作流 JSON"
                    }
                },
                "required": ["workflow"]
            }),
        ),
        ToolDefinition::function(
            "list_nodes",
            "列出所有可用的 ComfyUI 节点类型",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        ),
        ToolDefinition::function(
            "get_node_info",
            "获取特定节点的参数信息。参数：node_class（节点类型名）",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "node_class": {
                        "type": "string",
                        "description": "节点类型，如 'KSampler'"
                    }
                },
                "required": ["node_class"]
            }),
        ),
        ToolDefinition::function(
            "validate_workflow",
            "验证工作流是否有效。参数：workflow（JSON 工作流）",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": "待验证的工作流"
                    }
                },
                "required": ["workflow"]
            }),
        ),
        ToolDefinition::function(
            "get_status",
            "获取系统状态：后端健康、队列状态、内存使用等",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        ),
        ToolDefinition::function(
            "estimate_quality",
            "评估生成图片的质量。参数：image_path（图片路径）",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "待评估的图片路径"
                    },
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "评估维度：sharpness, contrast, color_balance, composition",
                        "default": ["sharpness", "contrast", "color_balance"]
                    }
                },
                "required": ["image_path"]
            }),
        ),
    ]
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_llm_config_from_env() {
        // 测试默认值
        std::env::remove_var("DEEPSEEK_API_URL");
        std::env::remove_var("DEEPSEEK_API_KEY");
        
        let config = LlmConfig::from_env();
        assert_eq!(config.base_url, "https://api.deepseek.com/v1");
        assert!(config.api_key.is_empty());
        
        // 测试环境变量
        std::env::set_var("DEEPSEEK_API_URL", "http://localhost:8081/v1");
        std::env::set_var("DEEPSEEK_API_KEY", "test-key");
        
        let config = LlmConfig::from_env();
        assert_eq!(config.base_url, "http://localhost:8081/v1");
        assert_eq!(config.api_key, "test-key");
        
        // 清理
        std::env::remove_var("DEEPSEEK_API_URL");
        std::env::remove_var("DEEPSEEK_API_KEY");
    }
    
    #[test]
    fn test_chat_message_creation() {
        let user_msg = ChatMessage::user("画一只猫");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, Some("画一只猫".to_string()));
        
        let system_msg = ChatMessage::system("你是一个图像生成助手");
        assert_eq!(system_msg.role, "system");
        
        let tool_result = ChatMessage::tool_result("call_123", "success");
        assert_eq!(tool_result.role, "tool");
        assert_eq!(tool_result.name, Some("call_123".to_string()));
    }
    
    #[test]
    fn test_tool_definition_serialization() {
        let tools = get_comfyui_tools();
        assert!(tools.len() >= 8);
        
        let t2i_tool = &tools[0];
        assert_eq!(t2i_tool.function.name, "build_t2i_workflow");
        
        // 验证可以序列化
        let json = serde_json::to_string(&t2i_tool).unwrap();
        assert!(json.contains("build_t2i_workflow"));
    }
    
    #[test]
    fn test_function_call_parse() {
        let func_call = FunctionCall {
            name: "build_t2i_workflow".to_string(),
            arguments: r#"{"prompt":"a cat","width":512,"height":512}"#.to_string(),
        };
        
        #[derive(Deserialize)]
        struct T2iArgs {
            prompt: String,
            width: i32,
            height: i32,
        }
        
        let args: T2iArgs = func_call.parse_arguments().unwrap();
        assert_eq!(args.prompt, "a cat");
        assert_eq!(args.width, 512);
        assert_eq!(args.height, 512);
    }
}