// ComfyUI 工具集 - 注册到 gliding_horse ToolExecutor
// 8 个核心工具：submit_workflow, build_t2i, build_i2i, backend_sample, list_nodes, interrupt, free_memory, health_check

use std::sync::Arc;
use serde_json::{json, Value};
use crate::agent::context::AgentContext;
use crate::types::Workflow;
use crate::workflow::builder::WorkflowBuilder;

/// 注册所有 ComfyUI 工具（基础 + 智能）
pub fn register_comfyui_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    ctx: Arc<AgentContext>,
) {
    // 先注册基础工具
    register_basic_tools(executor, ctx.clone());

    // 再注册智能工作流工具
    crate::agent::smart_tools::register_smart_workflow_tools(executor, ctx);
}

/// 注册智能引擎工具（SkillGraph + Discovery + Evolution + Causal）
pub fn register_intelligence_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    intel: Arc<crate::agent::advanced_intelligence::ComfyUiIntelligence>,
) {
    use crate::agent::advanced_intelligence::WorkflowExecutionRecord;

    // === discover_comfyui_skills ===
    // 根据用户请求发现匹配的 ComfyUI 技能
    let intel_for_discover = intel.clone();
    executor.register(
        "discover_comfyui_skills",
        "Discover matching ComfyUI skills based on user request and intent. Returns ranked skill recommendations.",
        json!({
            "type": "object",
            "properties": {
                "user_request": {"type": "string", "description": "User's natural language request"},
                "intent": {"type": "string", "description": "Parsed intent (text_to_image, image_to_image, video, upscale)"}
            },
            "required": ["user_request", "intent"]
        }),
        Arc::new(move |input: Value| {
            let intel = intel_for_discover.clone();
            Box::pin(async move {
                let user_request = input.get("user_request")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let intent = input.get("intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("text_to_image");

                let recommendations = intel.discover_skills(user_request, intent).await;

                let recs_json: Vec<Value> = recommendations.iter().map(|r| {
                    json!({
                        "skill_iri": r.skill_iri,
                        "skill_name": r.skill_name,
                        "score": r.score,
                        "reasons": r.reasons,
                        "required_dependencies": r.required_dependencies
                    })
                }).collect();

                Ok(json!({
                    "recommendations": recs_json,
                    "count": recommendations.len()
                }))
            })
        }),
        &["PA", "DA"],
    );

    // === recommend_parameters ===
    // 基于历史成功案例推荐参数
    let intel_for_params = intel.clone();
    executor.register(
        "recommend_parameters",
        "Recommend optimal parameters based on historical successful executions.",
        json!({
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "Generation intent"},
                "user_request": {"type": "string", "description": "User request for context"}
            },
            "required": ["intent"]
        }),
        Arc::new(move |input: Value| {
            let intel = intel_for_params.clone();
            Box::pin(async move {
                let intent = input.get("intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("text_to_image");
                let user_request = input.get("user_request")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let rec = intel.recommend_parameters(intent, user_request).await;

                Ok(json!({
                    "parameters": rec.parameters,
                    "reasoning": rec.reasoning,
                    "confidence": rec.confidence,
                    "similar_success_count": rec.similar_success_count
                }))
            })
        }),
        &["PA"],
    );

    // === analyze_failure ===
    // 分析执行失败的根因
    let intel_for_failure = intel.clone();
    executor.register(
        "analyze_failure",
        "Analyze the root cause of a workflow execution failure. Returns root cause and fix suggestions.",
        json!({
            "type": "object",
            "properties": {
                "failed_skill": {"type": "string", "description": "IRI of the failed skill"},
                "error_message": {"type": "string", "description": "Error message"}
            },
            "required": ["failed_skill", "error_message"]
        }),
        Arc::new(move |input: Value| {
            let intel = intel_for_failure.clone();
            Box::pin(async move {
                let failed_skill = input.get("failed_skill")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let error = input.get("error_message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let analysis = intel.analyze_failure(failed_skill, error);

                Ok(json!({
                    "failed_skill": analysis.failed_skill,
                    "root_cause_skill": analysis.root_cause_skill,
                    "root_cause_description": analysis.root_cause_description,
                    "confidence": analysis.confidence,
                    "fix_suggestions": analysis.fix_suggestions,
                    "propagation_path": analysis.propagation_path
                }))
            })
        }),
        &["CA", "DA"],
    );

    // === record_execution ===
    // 记录工作流执行（用于知识积累和自进化）
    let intel_for_record = intel.clone();
    executor.register(
        "record_execution",
        "Record a workflow execution for knowledge accumulation and skill evolution.",
        json!({
            "type": "object",
            "properties": {
                "execution_id": {"type": "string"},
                "user_request": {"type": "string"},
                "intent": {"type": "string"},
                "success": {"type": "boolean"},
                "duration_ms": {"type": "integer"},
                "node_count": {"type": "integer"},
                "parameters": {"type": "object"},
                "error": {"type": "string", "description": "Error message if failed"}
            },
            "required": ["execution_id", "intent", "success"]
        }),
        Arc::new(move |input: Value| {
            let intel = intel_for_record.clone();
            Box::pin(async move {
                let record = WorkflowExecutionRecord {
                    execution_id: input.get("execution_id")
                        .and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    user_request: input.get("user_request")
                        .and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    intent: input.get("intent")
                        .and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    workflow_json: input.get("workflow_json").cloned().unwrap_or(json!({})),
                    success: input.get("success").and_then(|v| v.as_bool()).unwrap_or(false),
                    duration_ms: input.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0),
                    node_count: input.get("node_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                    parameters: input.get("parameters").cloned().unwrap_or(json!({})),
                    timestamp: chrono::Utc::now(),
                    error: input.get("error").and_then(|v| v.as_str()).map(|s| s.to_string()),
                };

                intel.record_execution(record).await;

                Ok(json!({"recorded": true}))
            })
        }),
        &["DA", "CA"],
    );

    // === find_similar_workflows ===
    // 搜索相似工作流
    let intel_for_similar = intel.clone();
    executor.register(
        "find_similar_workflows",
        "Find similar historical workflows based on a query.",
        json!({
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "parameters": {"type": "object"},
                "top_k": {"type": "integer", "default": 5}
            },
            "required": ["intent"]
        }),
        Arc::new(move |input: Value| {
            let intel = intel_for_similar.clone();
            Box::pin(async move {
                let query = WorkflowExecutionRecord {
                    execution_id: "query".to_string(),
                    user_request: "".to_string(),
                    intent: input.get("intent").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    workflow_json: json!({}),
                    success: true,
                    duration_ms: 0,
                    node_count: 0,
                    parameters: input.get("parameters").cloned().unwrap_or(json!({})),
                    timestamp: chrono::Utc::now(),
                    error: None,
                };
                let top_k = input.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

                let similar = intel.find_similar_workflows(&query, top_k).await;

                let results: Vec<Value> = similar.iter().map(|(r, score)| {
                    json!({
                        "execution_id": r.execution_id,
                        "intent": r.intent,
                        "success": r.success,
                        "parameters": r.parameters,
                        "similarity_score": score
                    })
                }).collect();

                Ok(json!({"similar_workflows": results, "count": results.len()}))
            })
        }),
        &["PA", "DA"],
    );

    // === get_skill_stats ===
    // 获取技能统计
    let intel_for_stats = intel;
    executor.register(
        "get_skill_stats",
        "Get skill execution statistics.",
        json!({"type": "object", "properties": {}}),
        Arc::new(move |_input: Value| {
            let intel = intel_for_stats.clone();
            Box::pin(async move {
                let stats = intel.get_skill_stats().await;
                Ok(stats)
            })
        }),
        &["PA", "DA", "CA"],
    );
}

/// 注册基础工具（submit_workflow, build_t2i, build_i2i 等）
fn register_basic_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    ctx: Arc<AgentContext>,
) {
    // === 1. submit_workflow ===
    executor.register(
        "submit_workflow",
        "Submit a ComfyUI workflow for execution. Returns prompt_id and status.",
        json!({
            "type": "object",
            "properties": {
                "workflow": {"type": "object", "description": "Workflow JSON object with nodes and links"},
                "client_id": {"type": "string", "description": "Client identifier", "default": "agent"}
            },
            "required": ["workflow"]
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let workflow_json = input.get("workflow")
                        .cloned()
                        .ok_or_else(|| "Missing workflow parameter".to_string())?;

                    let workflow: Workflow = serde_json::from_value(workflow_json)
                        .map_err(|e| format!("Invalid workflow JSON: {}", e))?;

                    let client_id = input.get("client_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("agent");

                    let mut engine = ctx.engine.lock().await;
                    let prompt_id = engine.submit(workflow, client_id.to_string()).await
                        .map_err(|e| format!("Failed to submit workflow: {}", e))?;

                    Ok(json!({
                        "prompt_id": prompt_id,
                        "status": "submitted",
                        "message": "Workflow submitted successfully"
                    }))
                })
            }
        }),
        &["DA"],
    );

    // === 2. build_t2i_workflow ===
    executor.register(
        "build_t2i_workflow",
        "Build a text-to-image workflow from parameters. Returns workflow JSON ready for submission.",
        json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Positive prompt"},
                "negative_prompt": {"type": "string", "default": ""},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "steps": {"type": "integer", "default": 20},
                "cfg": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": -1},
                "model": {"type": "string", "default": "v1-5-pruned-emaonly.safetensors"},
                "sampler": {"type": "string", "default": "euler"},
                "scheduler": {"type": "string", "default": "normal"}
            },
            "required": ["prompt"]
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let _ctx = ctx.clone();
                Box::pin(async move {
                    let prompt = input.get("prompt")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing prompt parameter".to_string())?;

                    let negative_prompt = input.get("negative_prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let width = input.get("width")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1024) as usize;

                    let height = input.get("height")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1024) as usize;

                    let steps = input.get("steps")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(20) as usize;

                    let cfg = input.get("cfg")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(7.0) as f32;

                    let seed = input.get("seed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(-1);
                    let seed = if seed < 0 { rand::random::<usize>() } else { seed as usize };

                    let model = input.get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("v1-5-pruned-emaonly.safetensors");

                    let workflow = WorkflowBuilder::text_to_image(
                        prompt.to_string(),
                        negative_prompt.to_string(),
                        width, height, steps, cfg, seed, model.to_string(),
                    ).map_err(|e| format!("Failed to build workflow: {}", e))?;

                    let workflow_json = serde_json::to_value(&workflow)
                        .map_err(|e| format!("Failed to serialize workflow: {}", e))?;

                    Ok(json!({
                        "workflow": workflow_json,
                        "node_count": workflow.nodes.len(),
                        "params": { "width": width, "height": height, "steps": steps, "cfg": cfg, "seed": seed }
                    }))
                })
            }
        }),
        &["DA", "PA"],
    );

    // === 3. build_i2i_workflow ===
    executor.register(
        "build_i2i_workflow",
        "Build an image-to-image workflow from parameters. Requires input image path.",
        json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string", "default": ""},
                "strength": {"type": "number", "default": 0.75},
                "steps": {"type": "integer", "default": 20},
                "cfg": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": -1},
                "model": {"type": "string", "default": "v1-5-pruned-emaonly.safetensors"}
            },
            "required": ["image_path", "prompt"]
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let _ctx = ctx.clone();
                Box::pin(async move {
                    let image_path = input.get("image_path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing image_path parameter".to_string())?;

                    let prompt = input.get("prompt")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing prompt parameter".to_string())?;

                    let negative_prompt = input.get("negative_prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let strength = input.get("strength")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.75) as f32;

                    let steps = input.get("steps")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(20) as usize;

                    let cfg = input.get("cfg")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(7.0) as f32;

                    let seed = input.get("seed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(-1);
                    let seed = if seed < 0 { rand::random::<usize>() } else { seed as usize };

                    let model = input.get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("v1-5-pruned-emaonly.safetensors");

                    // 构建 image-to-image 工作流
                    let workflow = WorkflowBuilder::image_to_image(
                        prompt.to_string(),
                        negative_prompt.to_string(),
                        image_path.to_string(),
                        strength,
                        steps,
                        cfg,
                        seed,
                        model.to_string(),
                    ).map_err(|e| format!("Failed to build I2I workflow: {}", e))?;

                    let workflow_json = serde_json::to_value(&workflow)
                        .map_err(|e| format!("Failed to serialize workflow: {}", e))?;

                    Ok(json!({
                        "workflow": workflow_json,
                        "node_count": workflow.nodes.len(),
                        "params": {
                            "image_path": image_path,
                            "prompt": prompt,
                            "strength": strength,
                            "steps": steps,
                            "cfg": cfg,
                            "seed": seed
                        }
                    }))
                })
            }
        }),
        &["DA"],
    );

    // === 4. backend_sample ===
    executor.register(
        "backend_sample",
        "Execute sampling inference directly on backend. Low-level API for custom workflows.",
        json!({
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "seed": {"type": "integer", "default": 0},
                "steps": {"type": "integer", "default": 20},
                "cfg": {"type": "number", "default": 7.0},
                "sampler": {"type": "string", "default": "euler"},
                "scheduler": {"type": "string", "default": "normal"},
                "denoise": {"type": "number", "default": 1.0}
            },
            "required": ["model"]
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let model = input.get("model")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing model parameter".to_string())?;

                    let seed = input.get("seed").and_then(|v| v.as_i64()).unwrap_or(0);
                    let steps = input.get("steps").and_then(|v| v.as_i64()).unwrap_or(20);
                    let cfg = input.get("cfg").and_then(|v| v.as_f64()).unwrap_or(7.0) as f64;
                    let sampler = input.get("sampler").and_then(|v| v.as_str()).unwrap_or("euler");
                    let scheduler = input.get("scheduler").and_then(|v| v.as_str()).unwrap_or("normal");
                    let denoise = input.get("denoise").and_then(|v| v.as_f64()).unwrap_or(1.0);

                    let result = ctx.backend.sample(
                        model,
                        crate::types::Value::Conditioning(vec![]),
                        crate::types::Value::Conditioning(vec![]),
                        crate::types::Value::Latent(vec![]),
                        seed, steps, cfg, sampler, scheduler, denoise
                    ).await;

                    match result {
                        Ok(_output) => Ok(json!({
                            "status": "success",
                            "output_type": "latent",
                            "message": "Sampling completed"
                        })),
                        Err(e) => Err(format!("Backend sample failed: {}", e)),
                    }
                })
            }
        }),
        &["DA"],
    );

    // === 5. list_nodes ===
    executor.register(
        "list_nodes",
        "List all available ComfyUI node types and their input/output specifications.",
        json!({
            "type": "object",
            "properties": {
                "filter": {"type": "string", "default": ""}
            }
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let filter = input.get("filter")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let nodes = ctx.nodes.lock().await;
                    let all_info = nodes.get_all_node_info();

                    let filtered: Vec<_> = if filter.is_empty() {
                        all_info.into_iter().collect()
                    } else {
                        all_info.into_iter().filter(|(name, _)| name.contains(filter)).collect()
                    };

                    let node_list: Vec<Value> = filtered.iter()
                        .map(|(name, inputs)| json!({"class_type": name, "input_count": inputs.len()}))
                        .collect();

                    Ok(json!({"nodes": node_list, "total_count": filtered.len()}))
                })
            }
        }),
        &["DA", "PA", "CA"],
    );

    // === 6. interrupt ===
    executor.register(
        "interrupt",
        "Interrupt the currently running workflow execution.",
        json!({"type": "object", "properties": {}}),
        Arc::new({
            let ctx = ctx.clone();
            move |_input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let mut engine = ctx.engine.lock().await;
                    engine.interrupt();

                    Ok(json!({"status": "interrupted", "message": "Interrupt signal sent"}))
                })
            }
        }),
        &["DA"],
    );

    // === 7. free_memory ===
    executor.register(
        "free_memory",
        "Free GPU memory by unloading models and clearing caches.",
        json!({
            "type": "object",
            "properties": {
                "unload_models": {"type": "boolean", "default": true},
                "free_memory": {"type": "boolean", "default": true}
            }
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let unload_models = input.get("unload_models").and_then(|v| v.as_bool()).unwrap_or(true);
                    let free_mem = input.get("free_memory").and_then(|v| v.as_bool()).unwrap_or(true);

                    if unload_models || free_mem {
                        ctx.backend.free_memory().await;
                    }

                    Ok(json!({"status": "success", "message": "Memory freed", "unload_models": unload_models, "free_memory": free_mem}))
                })
            }
        }),
        &["DA"],
    );

    // === 8. health_check ===
    executor.register(
        "health_check",
        "Check backend health status. Returns backend availability and stats.",
        json!({"type": "object", "properties": {}}),
        Arc::new({
            let ctx = ctx.clone();
            move |_input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let healthy = ctx.backend.health_check().await;
                    let system_stats = ctx.backend.get_system_stats().await;

                    Ok(json!({
                        "healthy": healthy,
                        "backends": {"stable_diffusion_cpp": healthy},
                        "devices": system_stats.devices,
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    }))
                })
            }
        }),
        &["DA", "PA", "CA"],
    );

    log::info!("Registered 8 ComfyUI tools to ToolExecutor");
}
