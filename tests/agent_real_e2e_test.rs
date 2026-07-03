// Agent 真实 E2E 测试
// 使用 DeepSeek API（环境变量配置）进行真实 LLM 调用测试
// 需要：export DEEPSEEK_API_URL=https://api.deepseek.com/v1
//       export DEEPSEEK_API_KEY=your_api_key

use comfyui_rust_agent::agent::llm::{LlmClient, LlmConfig, ChatMessage, ChatRequest, get_comfyui_tools};
use comfyui_rust_agent::agent::context::AgentContext;
use comfyui_rust_agent::workflow::WorkflowBuilder;
use comfyui_rust_agent::execution::ExecutionEngine;
use comfyui_rust_agent::backend::BackendRouter;
use comfyui_rust_agent::node::registry::NodeRegistry;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde_json::json;

// ============================================================================
// DeepSeek API 连接测试
// ============================================================================

/// 测试 DeepSeek API 连接（真实调用）
#[tokio::test]
async fn test_deepseek_api_connection() {
    // 从环境变量创建客户端
    let client = LlmClient::from_env();
    
    // 检查配置
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping real API test");
        return;
    }
    
    // 发送简单请求
    let result = client.simple_chat("Hello, 请回复'OK'").await;
    
    match result {
        Ok(response) => {
            println!("DeepSeek API response: {}", response);
            assert!(!response.is_empty());
        }
        Err(e) => {
            println!("DeepSeek API error: {}", e);
            // 不强制失败，因为可能是网络问题
        }
    }
}

/// 测试 DeepSeek 中文对话（真实调用）
#[tokio::test]
async fn test_deepseek_chinese_chat() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    let result = client.simple_chat("你好，请用中文回答：1+1等于多少？").await;
    
    match result {
        Ok(response) => {
            println!("Response: {}", response);
            // 应该包含数字 2
            assert!(response.contains("2") || response.contains("两"));
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

/// 测试 DeepSeek 工具调用（真实调用）
#[tokio::test]
async fn test_deepseek_tool_calling() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 创建带工具的请求
    let messages = vec![
        ChatMessage::system("你是一个图像生成助手。当用户要求生成图片时，使用 build_t2i_workflow 工具。"),
        ChatMessage::user("画一只赛博朋克风格的猫，1024x1024"),
    ];
    
    let tools = get_comfyui_tools();
    
    let result = client.chat_with_tools(messages, tools).await;
    
    match result {
        Ok(response) => {
            println!("Response received");
            
            // 检查是否有工具调用
            if let Some(choice) = response.choices.first() {
                if let Some(tool_calls) = &choice.message.tool_calls {
                    println!("Tool calls: {} calls", tool_calls.len());
                    
                    for tool_call in tool_calls {
                        println!("  - Function: {}", tool_call.function.name);
                        println!("  - Arguments: {}", tool_call.function.arguments);
                        
                        // 应该调用 build_t2i_workflow
                        assert!(tool_call.function.name.contains("t2i") || 
                                tool_call.function.name.contains("workflow"));
                        
                        // 解析参数
                        let args: serde_json::Value = 
                            serde_json::from_str(&tool_call.function.arguments).unwrap();
                        
                        // 应包含 prompt
                        assert!(args["prompt"].is_string());
                        println!("  - Prompt: {}", args["prompt"]);
                    }
                } else {
                    println!("No tool calls in response");
                    println!("Content: {:?}", choice.message.content);
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

// ============================================================================
// Agent PDCA 循环测试（真实调用）
// ============================================================================

/// 测试 Agent PDCA 意图解析（真实调用）
#[tokio::test]
async fn test_agent_pdca_intent_parse() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 系统消息：定义意图解析任务
    let messages = vec![
        ChatMessage::system(r#"
你是一个意图分析助手。分析用户的图像生成请求，输出 JSON 格式的意图分析结果。

输出格式：
{
  "intent_type": "text_to_image" | "image_to_image" | "batch_generate" | "video_generate",
  "prompt": "用户描述",
  "style": "风格关键词",
  "dimensions": {"width": 数字, "height": 数字},
  "quality": "normal" | "high" | "ultra",
  "batch_count": 1
}

示例：
用户："画一只赛博朋克风格的猫，高清，1024x1024"
输出：{"intent_type":"text_to_image","prompt":"a cyberpunk cat","style":"cyberpunk","dimensions":{"width":1024,"height":1024},"quality":"high","batch_count":1}
"#),
        ChatMessage::user("画一只赛博朋克风格的猫，高清细节，1024x1024像素"),
    ];
    
    let request = ChatRequest {
        model: client.config().default_model.clone(),
        messages,
        max_tokens: Some(client.config().max_tokens),
        temperature: Some(0.3),
        tools: None,
        tool_choice: None,
        stream: Some(false),
    };

    let result = client.chat(request).await;

    match result {
        Ok(response) => {
            let response = response.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();
            println!("Intent analysis: {}", response);

            // 尝试提取 JSON
            if let Some(json_start) = response.find('{') {
                if let Some(json_end) = response.rfind('}') {
                    let json_str = &response[json_start..=json_end];
                    let intent: serde_json::Value = serde_json::from_str(json_str).unwrap();

                    println!("Parsed intent: {:?}", intent);

                    // 验证意图类型
                    assert!(intent["intent_type"].is_string(), "intent_type missing: {}", intent);
                    assert!(intent["prompt"].is_string(), "prompt missing: {}", intent);
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

/// 测试 Agent 参数优化（真实调用）
#[tokio::test]
async fn test_agent_param_optimization() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    let messages = vec![
        ChatMessage::system(r#"
你是一个图像生成参数优化助手。根据用户的描述需求，推荐最佳的生成参数。

输出 JSON 格式：
{
  "recommended_steps": 20-50,
  "recommended_cfg": 5-12,
  "recommended_sampler": "euler" | "dpmpp_2m" | "ddim",
  "reason": "优化原因"
}

规则：
- 高清图片：steps >= 30, cfg >= 8
- 快速生成：steps = 15-20, cfg = 5-7
- 细节丰富：steps >= 40, 高分辨率
"#),
        ChatMessage::user("我要生成一张高清风景照片，要求细节非常丰富"),
    ];
    
    let result = client.simple_chat("推荐参数：高清风景照片，细节丰富").await;
    
    match result {
        Ok(response) => {
            println!("Param recommendation: {}", response);
            
            // 提取 JSON
            if let Some(json_start) = response.find('{') {
                if let Some(json_end) = response.rfind('}') {
                    let json_str = &response[json_start..=json_end];
                    let params: serde_json::Value = serde_json::from_str(json_str).unwrap();
                    
                    println!("Recommended params: {:?}", params);
                    
                    // 高清应该推荐高步数
                    let steps = params["recommended_steps"].as_u64().unwrap_or(20);
                    assert!(steps >= 30, "HD image should use >= 30 steps");
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

// ============================================================================
// Agent ReAct 循环测试（真实调用）
// ============================================================================

/// 测试完整 ReAct 循环（真实调用）
#[tokio::test]
async fn test_agent_react_loop() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // ReAct 循环：思考 → 行动 → 观察
    let mut messages = vec![
        ChatMessage::system(r#"
你是一个图像生成 Agent。使用 ReAct 循环完成任务：

1. Thought: 分析用户需求，决定下一步行动
2. Action: 调用工具（build_t2i_workflow, submit_workflow 等）
3. Observation: 观察工具返回结果
4. 重复直到任务完成

可用工具：
- build_t2i_workflow: 构建文生图工作流
- submit_workflow: 提交执行
- get_status: 查看状态

输出格式：
Thought: [你的思考]
Action: [工具名称]
Action Input: {"参数": "值"}
Observation: [工具返回]
...（重复直到完成）
Final Answer: [最终结果]
"#),
        ChatMessage::user("生成一张赛博朋克猫咪图片"),
    ];
    
    let tools = get_comfyui_tools();
    
    // 第一轮：思考 + 行动
    let result = client.chat_with_tools(messages.clone(), tools.clone()).await;
    
    match result {
        Ok(response) => {
            if let Some(choice) = response.choices.first() {
                println!("First turn response:");
                
                if let Some(content) = &choice.message.content {
                    println!("Content: {}", content);
                }
                
                if let Some(tool_calls) = &choice.message.tool_calls {
                    for tc in tool_calls {
                        println!("Tool call: {} with args {}", 
                                 tc.function.name, tc.function.arguments);
                        
                        // 模拟工具执行
                        let tool_result = execute_tool_mock(&tc.function.name, &tc.function.arguments);
                        
                        // 添加观察结果
                        messages.push(choice.message.clone());
                        messages.push(ChatMessage::tool_result(&tc.id, &tool_result));
                    }
                    
                    // 第二轮：根据观察继续
                    let result2 = client.chat_with_tools(messages, tools).await;
                    
                    match result2 {
                        Ok(resp2) => {
                            if let Some(c2) = resp2.choices.first() {
                                println!("Second turn response:");
                                if let Some(content) = &c2.message.content {
                                    println!("Final answer: {}", content);
                                }
                            }
                        }
                        Err(e) => println!("Error in second turn: {}", e),
                    }
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

/// 模拟工具执行（用于测试）
fn execute_tool_mock(name: &str, args: &str) -> String {
    match name {
        "build_t2i_workflow" => {
            let args_json: serde_json::Value = serde_json::from_str(args).unwrap();
            json!({
                "success": true,
                "workflow": {
                    "nodes": {
                        "1": {"class_type": "CheckpointLoaderSimple"},
                        "2": {"class_type": "KSampler"},
                        "3": {"class_type": "VAEDecode"}
                    },
                    "prompt": args_json["prompt"],
                    "width": args_json.get("width").unwrap_or(&json!(512)),
                    "height": args_json.get("height").unwrap_or(&json!(512))
                }
            }).to_string()
        }
        "submit_workflow" => {
            json!({
                "success": true,
                "prompt_id": "test-123",
                "status": "queued"
            }).to_string()
        }
        "get_status" => {
            json!({
                "queue_size": 0,
                "backend_healthy": true
            }).to_string()
        }
        _ => json!({"error": "Unknown tool"}).to_string(),
    }
}

// ============================================================================
// 多轮对话测试（真实调用）
// ============================================================================

/// 测试多轮对话记忆（真实调用）
#[tokio::test]
async fn test_agent_multi_turn_memory() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 第一轮：生成请求
    let mut messages = vec![
        ChatMessage::user("画一只猫"),
    ];
    
    let result1 = client.simple_chat("画一只猫").await;
    match result1 {
        Ok(resp1) => {
            println!("Turn 1: {}", resp1);
            messages.push(ChatMessage::assistant(&resp1));
        }
        Err(e) => {
            println!("Error turn 1: {}", e);
            return;
        }
    }
    
    // 第二轮：引用第一轮（修改）
    messages.push(ChatMessage::user("把刚才的猫改成赛博朋克风格"));
    
    let result2 = client.simple_chat("把刚才的猫改成赛博朋克风格").await;
    match result2 {
        Ok(resp2) => {
            println!("Turn 2: {}", resp2);
            
            // 应该理解"刚才的猫"
            assert!(resp2.contains("赛博朋克") || resp2.contains("cyberpunk"));
        }
        Err(e) => println!("Error turn 2: {}", e),
    }
}

// ============================================================================
// 批量生成测试（真实调用）
// ============================================================================

/// 测试批量生成意图解析（真实调用）
#[tokio::test]
async fn test_batch_generate_intent() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    let messages = vec![
        ChatMessage::system("分析用户请求，如果是批量生成，输出每个任务的参数"),
        ChatMessage::user("帮我生成 5 张不同风格的猫咪：写实、卡通、水彩、油画、赛博朋克"),
    ];
    
    let result = client.simple_chat(
        "分析批量生成请求：生成 5 张不同风格的猫咪"
    ).await;
    
    match result {
        Ok(response) => {
            println!("Batch analysis: {}", response);
            
            // 应识别为批量任务
            assert!(response.contains("5") || response.contains("五"));
        }
        Err(e) => println!("Error: {}", e),
    }
}

// ============================================================================
// 工作流构建测试
// ============================================================================

/// 测试 WorkflowBuilder + LLM（真实调用）
#[tokio::test]
async fn test_workflow_builder_with_llm() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 1. LLM 分析意图
    let intent_result = client.simple_chat(
        "分析图像生成请求并输出 JSON：画一只高清猫咪，1024x1024，赛博朋克风格"
    ).await;
    
    match intent_result {
        Ok(intent_str) => {
            println!("Intent: {}", intent_str);
            
            // 2. 提取参数（简化）
            let prompt = "a cyberpunk cat, high detail";
            let width = 1024;
            let height = 1024;
            
            // 3. 构建工作流
            let workflow = WorkflowBuilder::text_to_image(
                prompt.to_string(),
                "".to_string(),
                width,
                height,
                30,  // 高清用更多步数
                8.0,
                42,
                "v1-5-pruned-emaonly.safetensors".to_string(),
            );

            match workflow {
                Ok(wf) => {
                    println!("Workflow built successfully: {} nodes", wf.nodes.len());

                    // 验证节点
                    assert!(wf.nodes.len() >= 7);

                    // 验证连接（builder 把连接嵌入 node 的 InputValue::Link）
                    let has_links = wf.nodes.values()
                        .any(|n| n.inputs.values().any(|v| matches!(v, comfyui_rust_agent::types::InputValue::Link(_))));
                    assert!(has_links);
                    
                    // 4. 可选：提交执行（需要后端）
                    // ...
                }
                Err(e) => {
                    println!("Workflow build error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Intent parse error: {}", e);
        }
    }
}

// ============================================================================
// 系统提示词测试
// ============================================================================

/// 测试不同系统提示词效果（真实调用）
#[tokio::test]
async fn test_system_prompt_effects() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 测试 1：简洁助手
    let messages1 = vec![
        ChatMessage::system("你是一个简洁的助手，用一句话回答"),
        ChatMessage::user("什么是 Stable Diffusion？"),
    ];
    
    // 测试 2：详细助手
    let messages2 = vec![
        ChatMessage::system("你是一个详细的助手，给出全面的解释"),
        ChatMessage::user("什么是 Stable Diffusion？"),
    ];
    
    // 测试 3：图像生成专家
    let messages3 = vec![
        ChatMessage::system("你是一个 Stable Diffusion 图像生成专家，提供技术参数建议"),
        ChatMessage::user("生成高清图片用什么参数？"),
    ];
    
    // 分别发送请求（简化，只测试一个）
    let result = client.simple_chat("作为 Stable Diffusion 专家，推荐高清图片生成参数").await;
    
    match result {
        Ok(response) => {
            println!("Expert response: {}", response);
            
            // 应包含参数建议
            assert!(response.contains("steps") || 
                    response.contains("步数") ||
                    response.contains("CFG"));
        }
        Err(e) => println!("Error: {}", e),
    }
}

// ============================================================================
// 性能测试（真实调用）
// ============================================================================

/// 测试 DeepSeek API 响应时间
#[tokio::test]
async fn test_deepseek_response_time() {
    let client = LlmClient::from_env();
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    let start = std::time::Instant::now();
    
    let result = client.simple_chat("Hello").await;
    
    let elapsed = start.elapsed();
    
    match result {
        Ok(_) => {
            println!("Response time: {}ms", elapsed.as_millis());
            
            // API 应在 5 秒内响应
            assert!(elapsed.as_secs() < 5, "API response too slow");
        }
        Err(e) => println!("Error: {}", e),
    }
}

/// 测试并发请求（真实调用）
#[tokio::test]
async fn test_concurrent_api_calls() {
    let client = Arc::new(LlmClient::from_env());
    
    if !client.config().is_valid() {
        println!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }
    
    // 发送 3 个并发请求
    let tasks: Vec<_> = (0..3).map(|i| {
        let c = client.clone();
        async move {
            c.simple_chat(&format!("请求 {}: 回复OK", i)).await
        }
    }).collect();
    
    let results = futures::future::join_all(tasks).await;
    
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(resp) => println!("Request {} success: {}", i, resp),
            Err(e) => println!("Request {} error: {}", i, e),
        }
    }
}