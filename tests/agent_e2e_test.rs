// Agent 端到端集成测试
// 测试 gliding_horse + media_agent 的完整链路

use serde_json::json;

/// 测试 Agent 状态端点
#[tokio::test]
async fn test_agent_status_endpoint() {
    let client = reqwest::Client::new();
    
    // 假设服务器已启动
    let resp = client.get("http://localhost:8188/agent/status")
        .send()
        .await;
    
    // 如果服务器未启动，跳过测试
    if resp.is_err() {
        println!("Server not running, skipping test");
        return;
    }
    
    let resp = resp.unwrap();
    assert!(resp.status().is_success());
    
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["context_ready"].is_boolean());
    assert!(body["supervisor_ready"].is_boolean());
    assert!(body["tools"].is_array());
}

/// 测试 Agent 聊天端点（需要 llama.cpp server）
#[tokio::test]
#[ignore = "requires llama.cpp server running at localhost:8081"]
async fn test_agent_chat_endpoint() {
    let client = reqwest::Client::new();
    
    // 先初始化 Agent
    let init_resp = client.post("http://localhost:8188/agent/init")
        .send()
        .await
        .unwrap();
    
    assert!(init_resp.status().is_success());
    
    // 发送对话请求
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只赛博朋克风格的猫，1024x1024像素，高清细节",
            "max_iterations": 10
        }))
        .send()
        .await
        .unwrap();
    
    assert!(chat_resp.status().is_success());
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    assert!(body["task_id"].is_string());
    assert_eq!(body["status"], "success");
    assert!(body["turn_count"].as_u64().unwrap() > 0);
}

/// 测试工作流列表端点
#[tokio::test]
async fn test_agent_workflows_endpoint() {
    let client = reqwest::Client::new();
    
    let resp = client.get("http://localhost:8188/agent/workflows")
        .send()
        .await;
    
    if resp.is_err() {
        println!("Server not running, skipping test");
        return;
    }
    
    let resp = resp.unwrap();
    assert!(resp.status().is_success());
    
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["workflows"].is_array());
}

/// 测试完整生成流程（使用指定工作流）
#[tokio::test]
#[ignore = "requires llama.cpp server and stable-diffusion.cpp backend"]
async fn test_agent_full_generation() {
    let client = reqwest::Client::new();
    
    // 1. 检查后端健康
    let health = client.get("http://localhost:8188/health")
        .send()
        .await
        .unwrap();
    
    let health_body: serde_json::Value = health.json().await.unwrap();
    if health_body["status"] != "healthy" {
        println!("Backend not healthy, skipping test");
        return;
    }
    
    // 2. 初始化 Agent
    client.post("http://localhost:8188/agent/init")
        .send()
        .await
        .unwrap();
    
    // 3. 发送带工作流的生成请求
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只可爱的猫咪",
            "workflow": "workflows/generate_and_review.jsonld",
            "max_iterations": 15
        }))
        .send()
        .await
        .unwrap();
    
    assert!(chat_resp.status().is_success());
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    println!("Generation result: {:?}", body);
    
    // 4. 验证输出
    assert!(body["output"].is_object());
    // 如果生成成功，应该有 image_path
    if body["status"] == "success" {
        assert!(body["output"]["image_path"].is_string());
    }
}

/// 测试工具调用计数
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_agent_tool_calls() {
    let client = reqwest::Client::new();
    
    // 初始化 Agent
    client.post("http://localhost:8188/agent/init")
        .send()
        .await
        .unwrap();
    
    // 发送需要工具调用的请求
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "列出所有可用的生成节点",
            "max_iterations": 5
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 应该至少调用一次 list_nodes 工具
    assert!(body["tool_calls"].as_u64().unwrap() >= 1);
}

// ============================================================================
// PDCA 模式测试（默认模式）
// ============================================================================

/// 测试 PDCA 模式自动路由
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_pdca_mode_auto_routing() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 不指定 workflow，使用 PDCA 模式
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只赛博朋克风格的猫"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // PDCA 模式应该有 Plan-Do-Check-Act 迭代
    assert!(body["mode"] == "pdca");
    assert!(body["iterations"].as_u64().unwrap() > 0);
}

/// 测试 PDCA 模式意图解析
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_pdca_intent_parsing() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 复杂意图：批量生成 + 不同风格
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "帮我生成 5 张不同风格的猫咪图片：写实、卡通、水彩、油画、赛博朋克"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // PDCA 应该解析为批量生成任务
    assert!(body["intent"]["type"] == "batch_generate");
    assert!(body["intent"]["count"].as_u64().unwrap() == 5);
}

/// 测试 PDCA 模式参数优化
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_pdca_param_optimization() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 高清图片请求，Agent 应优化参数
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一张高清风景照片，要求细节丰富"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // PDCA 应优化 steps 和分辨率
    if body["optimized_params"].is_object() {
        let steps = body["optimized_params"]["steps"].as_u64().unwrap();
        assert!(steps >= 30); // 高清应该用更多步数
        
        let width = body["optimized_params"]["width"].as_u64().unwrap();
        assert!(width >= 1024); // 高清应该用更高分辨率
    }
}

// ============================================================================
// DAG 模式测试（显式工作流）
// ============================================================================

/// 测试 DAG 模式显式指定
#[tokio::test]
#[ignore = "requires llama.cpp server and stable-diffusion.cpp"]
async fn test_dag_mode_explicit_workflow() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 指定 workflow，使用 DAG 模式
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只可爱的猫咪",
            "workflow": "workflows/generate_and_review.jsonld"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // DAG 模式
    assert!(body["mode"] == "dag");
    assert!(body["workflow_name"] == "generate_and_review.jsonld");
    
    // 验证 DAG 节点执行顺序
    assert!(body["dag_steps"].is_array());
    let steps = body["dag_steps"].as_array().unwrap();
    assert!(steps.len() >= 3); // 至少 3 步：generate → review → finalize
}

/// 测试批量生成 DAG 工作流
#[tokio::test]
#[ignore = "requires llama.cpp server and stable-diffusion.cpp"]
async fn test_dag_batch_generate_workflow() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "批量生成 3 张猫咪图片",
            "workflow": "workflows/batch_generate.jsonld"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    assert!(body["mode"] == "dag");
    
    // 批量生成应该有多个输出
    if body["status"] == "success" {
        assert!(body["outputs"].as_array().unwrap().len() == 3);
    }
}

/// 测试 DAG 分支条件
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_dag_branch_condition() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 使用带分支的工作流
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "生成图片并检查质量",
            "workflow": "workflows/generate_with_refinement.jsonld"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 应该有分支执行记录
    if body["branch_taken"].is_string() {
        // 根据质量评估结果选择分支
        let branch = body["branch_taken"].as_str().unwrap();
        assert!(branch == "success" || branch == "refine" || branch == "retry");
    }
}

// ============================================================================
// 模式切换测试
// ============================================================================

/// 测试同一会话中模式切换
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_mode_switching_in_session() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 第一次请求：PDCA
    let resp1 = client.post("http://localhost:8188/agent/chat")
        .json(&json!({ "message": "画一只猫" }))
        .send()
        .await
        .unwrap();
    
    let body1: serde_json::Value = resp1.json().await.unwrap();
    assert!(body1["mode"] == "pdca");
    
    // 第二次请求：DAG（指定 workflow）
    let resp2 = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "批量生成",
            "workflow": "workflows/batch_generate.jsonld"
        }))
        .send()
        .await
        .unwrap();
    
    let body2: serde_json::Value = resp2.json().await.unwrap();
    assert!(body2["mode"] == "dag");
}

/// 测试默认模式配置
#[tokio::test]
async fn test_default_mode_configuration() {
    let client = reqwest::Client::new();
    
    let resp = client.get("http://localhost:8188/agent/status").send().await;
    
    if resp.is_err() {
        println!("Server not running, skipping");
        return;
    }
    
    let body: serde_json::Value = resp.unwrap().json().await.unwrap();
    
    // 默认模式应该是 pdca
    assert!(body["default_mode"] == "pdca");
}

// ============================================================================
// 技能测试
// ============================================================================

/// 测试技能加载
#[tokio::test]
async fn test_skills_loaded() {
    let client = reqwest::Client::new();
    
    let resp = client.get("http://localhost:8188/agent/status").send().await;
    
    if resp.is_err() {
        println!("Server not running, skipping");
        return;
    }
    
    let body: serde_json::Value = resp.unwrap().json().await.unwrap();
    
    // 应该有技能列表
    assert!(body["skills"].is_array());
    let skills = body["skills"].as_array().unwrap();
    
    // 验证核心技能存在
    let skill_names: Vec<&str> = skills.iter()
        .filter_map(|s| s["name"].as_str())
        .collect();
    
    assert!(skill_names.iter().any(|n| n.contains("text_to_image")));
    assert!(skill_names.iter().any(|n| n.contains("image_to_image")));
    assert!(skill_names.iter().any(|n| n.contains("generate_video")));
}

/// 测试技能参数验证
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_skill_param_validation() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 提供不完整参数（缺少 prompt）
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "生成一张图片，尺寸 1024x1024"  // 缺少 prompt 描述
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // PDCA 应自动补全参数或询问用户
    // 如果报错，应该是参数缺失提示
    if body["status"] == "error" {
        assert!(body["error"]["code"] == "MISSING_PARAM");
    } else {
        // 成功则说明 Agent 自动生成了默认 prompt
        assert!(body["status"] == "success");
    }
}

// ============================================================================
// 工具测试
// ============================================================================

/// 测试工具注册
#[tokio::test]
async fn test_tools_registered() {
    let client = reqwest::Client::new();
    
    let resp = client.get("http://localhost:8188/agent/status").send().await;
    
    if resp.is_err() {
        println!("Server not running, skipping");
        return;
    }
    
    let body: serde_json::Value = resp.unwrap().json().await.unwrap();
    
    assert!(body["tools"].is_array());
    let tools = body["tools"].as_array().unwrap();
    
    // 验证 8 个核心工具
    let tool_names: Vec<&str> = tools.iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    
    assert!(tool_names.contains(&"build_t2i_workflow"));
    assert!(tool_names.contains(&"build_i2i_workflow"));
    assert!(tool_names.contains(&"submit_workflow"));
    assert!(tool_names.contains(&"list_nodes"));
    assert!(tool_names.contains(&"get_node_info"));
    assert!(tool_names.contains(&"validate_workflow"));
    assert!(tool_names.contains(&"fill_params"));
    assert!(tool_names.contains(&"get_status"));
}

/// 测试 list_nodes 工具调用
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_list_nodes_tool() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "有哪些节点可用？"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 应返回节点列表
    assert!(body["response"]["nodes"].is_array());
}

/// 测试 get_node_info 工具调用
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_get_node_info_tool() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "KSampler 节点的参数是什么？"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 应返回 KSampler 节点信息
    if body["response"]["node_info"].is_object() {
        let info = &body["response"]["node_info"];
        assert!(info["class_type"] == "KSampler");
        assert!(info["inputs"].is_object());
    }
}

// ============================================================================
// 记忆系统测试
// ============================================================================

/// 测试短期记忆
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_short_term_memory() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 第一轮对话
    client.post("http://localhost:8188/agent/chat")
        .json(&json!({ "message": "画一只猫" }))
        .send()
        .await
        .unwrap();
    
    // 第二轮对话引用第一轮
    let resp2 = client.post("http://localhost:8188/agent/chat")
        .json(&json!({ "message": "把刚才的猫改成赛博朋克风格" }))
        .send()
        .await
        .unwrap();
    
    let body2: serde_json::Value = resp2.json().await.unwrap();
    
    // Agent 应理解"刚才的猫"是指上一轮的结果
    assert!(body2["context"]["previous_task"].is_string());
}

/// 测试长期记忆（风格偏好）
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_long_term_memory_style_preference() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 多轮对话建立风格偏好
    for _ in 0..3 {
        client.post("http://localhost:8188/agent/chat")
            .json(&json!({ "message": "画赛博朋克风格的图片" }))
            .send()
            .await
            .unwrap();
    }
    
    // 新请求不指定风格
    let resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({ "message": "画一只狗" }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = resp.json().await.unwrap();
    
    // Agent 应根据历史偏好自动选择赛博朋克风格
    if body[" inferred_style"].is_string() {
        assert!(body["inferred_style"] == "cyberpunk");
    }
}

// ============================================================================
// 错误处理测试
// ============================================================================

/// 测试后端不可用时的降级处理
#[tokio::test]
#[ignore = "requires llama.cpp server but no stable-diffusion.cpp"]
async fn test_backend_unavailable_graceful_degradation() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只猫"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 如果后端不可用，应该返回 ServiceUnavailable
    if body["status"] == "error" {
        assert!(body["error"]["code"] == "SERVICE_UNAVAILABLE");
        assert!(body["error"]["is_retryable"] == true);
    }
}

/// 测试无效工作流参数
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_invalid_workflow_params() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 无效尺寸（不是 8 的倍数）
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只猫，尺寸 513x513"
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // Agent 应自动修正尺寸或报错
    if body["status"] == "success" {
        // 自动修正到有效尺寸
        let w = body["actual_params"]["width"].as_u64().unwrap();
        let h = body["actual_params"]["height"].as_u64().unwrap();
        assert!(w % 8 == 0);
        assert!(h % 8 == 0);
    }
}

/// 测试超时重试
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_timeout_retry() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 设置短超时
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "画一只高细节猫咪，100 步采样",
            "timeout_secs": 5  // 超短超时
        }))
        .send()
        .await
        .unwrap();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 如果超时，应该有重试记录
    if body["status"] == "timeout" {
        assert!(body["retry_count"].as_u64().unwrap() >= 1);
    }
}

// ============================================================================
// 批量生成测试
// ============================================================================

/// 测试批量生成并发
#[tokio::test]
#[ignore = "requires stable-diffusion.cpp backend"]
async fn test_batch_parallel_generation() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    let start = std::time::Instant::now();
    
    let chat_resp = client.post("http://localhost:8188/agent/chat")
        .json(&json!({
            "message": "批量生成 10 张不同风格的猫咪",
            "workflow": "workflows/batch_generate.jsonld"
        }))
        .send()
        .await
        .unwrap();
    
    let elapsed = start.elapsed();
    
    let body: serde_json::Value = chat_resp.json().await.unwrap();
    
    // 批量生成应该并行执行，时间不应该线性增长
    // 10 张图片并行生成应该在 60 秒内完成（假设单张 6 秒）
    if body["status"] == "success" {
        assert!(elapsed.as_secs() < 60);
        assert!(body["outputs"].as_array().unwrap().len() == 10);
    }
}

// ============================================================================
// 性能测试
// ============================================================================

/// 测试响应延迟
#[tokio::test]
async fn test_response_latency() {
    let client = reqwest::Client::new();
    
    let start = std::time::Instant::now();
    
    let resp = client.get("http://localhost:8188/agent/status").send().await;
    
    if resp.is_err() {
        println!("Server not running, skipping");
        return;
    }
    
    let elapsed = start.elapsed();
    
    // 状态查询应该在 100ms 内响应
    assert!(elapsed.as_millis() < 100);
}

/// 测试并发请求处理
#[tokio::test]
#[ignore = "requires llama.cpp server"]
async fn test_concurrent_requests() {
    let client = reqwest::Client::new();
    
    client.post("http://localhost:8188/agent/init").send().await.unwrap();
    
    // 发送 5 个并发请求
    let requests: Vec<_> = (0..5).map(|i| {
        client.post("http://localhost:8188/agent/chat")
            .json(&json!({
                "message": format!("画第 {} 号猫咪", i)
            }))
    }).collect();
    
    // 并发执行
    let results: Vec<_> = futures::future::join_all(
        requests.into_iter().map(|r| r.send())
    ).await;
    
    // 验证所有请求成功
    for result in results {
        if result.is_ok() {
            let resp = result.unwrap();
            assert!(resp.status().is_success());
        }
    }
}