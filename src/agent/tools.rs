// Native generation and compatibility tools registered with Gliding Horse.

use std::sync::Arc;
use serde::Deserialize;
use serde_json::{json, Value};
use crate::agent::context::AgentContext;
use crate::agent::context::GenerationRequestContext;
use crate::agent::quality::{adjust_for_retry, score_image_bytes};
use crate::backend::{HiresSpec, I2IParams, LoraSpec, T2IParams};
use crate::config::gpu_tier::{GpuTier, ImageModelChoice, ImageModelKind};
use crate::types::Workflow;
use crate::workflow::builder::WorkflowBuilder;

#[derive(Debug, Deserialize)]
struct NativeGenerationSpec {
    intent: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    image_path: Option<String>,
    model: Option<String>,
    quality: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    steps: Option<usize>,
    cfg: Option<f32>,
    min_cfg: Option<f32>,
    noise_aug_strength: Option<f32>,
    seed: Option<i64>,
    strength: Option<f32>,
    frames: Option<usize>,
    fps: Option<usize>,
    motion_bucket_id: Option<i32>,
}

/// 注册所有 ComfyUI 工具（基础 + 智能）
pub fn register_comfyui_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    ctx: Arc<AgentContext>,
) {
    let compatibility_tools_enabled = ctx.app_config.agent.compatibility_tools_enabled;
    register_basic_tools(executor, ctx.clone());

    if compatibility_tools_enabled {
        crate::agent::smart_tools::register_smart_workflow_tools(executor, ctx);
    }
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
        &["PA"],
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
        &["CA"],
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
                    quality_score: None,
                    gpu_tier: crate::config::gpu_tier::GpuTier::detect(None).name().to_string(),
                };

                intel.record_execution(record).await;

                Ok(json!({"recorded": true}))
            })
        }),
        &["CA"],
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
                    quality_score: None,
                    gpu_tier: String::new(),
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
        &["PA"],
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
        &["PA"],
    );
}

/// 注册基础工具（submit_workflow, build_t2i, build_i2i 等）
fn register_basic_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    ctx: Arc<AgentContext>,
) {
    executor.register(
        "inspect_native_runtime",
        "Inspect the zero-Python native runtime, configured stable-diffusion.cpp/llama.cpp paths, and model containers. Use this before planning a generation that depends on an uncertain model.",
        json!({"type": "object", "properties": {}}),
        Arc::new({
            let ctx = ctx.clone();
            move |_input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    serde_json::to_value(crate::native_runtime::NativeRuntimeReport::inspect(
                        &ctx.app_config,
                    ))
                    .map_err(|error| format!("Failed to serialize runtime report: {}", error))
                })
            }
        }),
        &["PA", "DA"],
    );

    executor.register(
        "validate_model",
        "Validate a local model's real container format. This rejects empty, corrupt, or PyTorch ZIP files renamed as Safetensors.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local model path"}
            },
            "required": ["path"]
        }),
        Arc::new(move |input: Value| {
            Box::pin(async move {
                let path = input.get("path")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| "Missing model path".to_string())?;
                let info = crate::native_runtime::inspect_model_file(path)?;
                serde_json::to_value(info)
                    .map_err(|error| format!("Failed to serialize model information: {}", error))
            })
        }),
        &["PA", "DA"],
    );

    executor.register(
        "generate_media",
        "Execute one high-level native media task through Rust and stable-diffusion.cpp. Prefer this over constructing low-level workflow nodes. Returns only after a real non-empty media artifact has been written.",
        json!({
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["text_to_image", "image_to_image", "text_to_video", "image_to_video"]},
                "prompt": {"type": "string", "description": "Model-ready English prompt, normally 25-80 words. Preserve every requested subject, action, count, setting, time, weather, lighting, style, and composition attribute; do not replace the request with a short summary."},
                "negative_prompt": {"type": "string", "default": "", "description": "Undesired content and quality defects. Include visual conditions that directly contradict the request when relevant."},
                "image_path": {"type": "string", "description": "REQUIRED for image_to_image and image_to_video. Copy the exact path from the <input_image> block."},
                "model": {"type": "string", "description": "Optional exact model selected by the user. Omit it to use the configured native model; never invent aliases or filenames."},
                "quality": {"type": "string", "enum": ["fast", "balanced", "high"], "default": "balanced"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "steps": {"type": "integer"},
                "cfg": {"type": "number"},
                "min_cfg": {"type": "number", "description": "SVD first-frame guidance. The final frame uses cfg and intermediate frames are linearly interpolated."},
                "noise_aug_strength": {"type": "number", "description": "SVD conditioning-image noise augmentation, normally 0.02."},
                "seed": {"type": "integer", "default": -1},
                "strength": {"type": "number"},
                "frames": {"type": "integer"},
                "fps": {"type": "integer"},
                "motion_bucket_id": {"type": "integer", "minimum": 0, "maximum": 1023, "description": "SVD motion intensity. The value is passed through to native stable-diffusion.cpp inference."}
            },
            "required": ["intent", "prompt"]
        }),
        Arc::new({
            let ctx = ctx.clone();
            move |input: Value| {
                let ctx = ctx.clone();
                Box::pin(async move {
                    let spec: NativeGenerationSpec = serde_json::from_value(input)
                        .map_err(|error| format!("Invalid generation specification: {}", error))?;
                    execute_native_generation(&ctx, spec).await
                })
            }
        }),
        &["DA"],
    );

    executor.register(
        "inspect_artifact",
        "Verify that a generated image or video exists, is non-empty, and can be decoded/probed. CA must use this before accepting a task.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Generated artifact path"}
            },
            "required": ["path"]
        }),
        Arc::new(move |input: Value| {
            Box::pin(async move {
                let path = input.get("path")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| "Missing artifact path".to_string())?;
                let artifact = inspect_media_artifact(path)?;
                Ok(json!({
                    "status": "success",
                    "artifact": artifact,
                    "artifacts": [artifact]
                }))
            })
        }),
        &["CA"],
    );

    if !ctx.app_config.agent.compatibility_tools_enabled {
        log::info!("Legacy ComfyUI workflow tools are disabled for the default Agent path");
        return;
    }

    // === 1. submit_workflow ===
    executor.register(
        "submit_workflow",
        "REQUIRED: Execute the workflow and generate outputs. This is the SECOND and FINAL step after build_*_workflow. Workflow JSON is automatically generated by build_*_workflow tools - pass the exact 'workflow' field from build results. DO NOT save workflow to file. DO NOT analyze workflow. JUST SUBMIT IT.",
        json!({
            "type": "object",
            "properties": {
                "workflow": {"type": "object", "description": "REQUIRED: Workflow JSON from build_*_workflow's 'workflow' field. DO NOT pass the entire build result - only the 'workflow' field value."},
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

                    // 立即执行工作流（submit 只入队，必须调用 execute_next 才会真正执行）
                    let exec_result = engine.execute_next().await
                        .map_err(|e| format!("Failed to execute workflow: {}", e))?;

                    match exec_result {
                        Some(crate::types::ExecutionResult::Success(outputs)) => {
                            // 提取输出文件路径（outputs 是 HashMap<NodeId, HashMap<String, Value>>）
                            let mut output_files: Vec<String> = Vec::new();
                            let mut has_video_file = false;
                            let mut has_image_file = false;
                            for node_outputs in outputs.values() {
                                for val in node_outputs.values() {
                                    match val {
                                        crate::types::Value::String(s) => {
                                            if s.ends_with(".mp4") || s.ends_with(".webm") || s.ends_with(".gif") {
                                                output_files.push(s.clone());
                                                has_video_file = true;
                                            } else if s.ends_with(".png") || s.ends_with(".jpg") || s.ends_with(".jpeg") {
                                                output_files.push(s.clone());
                                                has_image_file = true;
                                            }
                                        }
                                        crate::types::Value::Image(data) => {
                                            if !has_image_file {
                                                output_files.push(format!("image: {} bytes", data.len()));
                                            }
                                        }
                                        crate::types::Value::Video(data) => {
                                            if !has_video_file {
                                                output_files.push(format!("video: {} bytes", data.len()));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }

                            Ok(json!({
                                "prompt_id": prompt_id,
                                "status": "success",
                                "message": "Workflow executed successfully",
                                "outputs": output_files,
                                "output_count": output_files.len()
                            }))
                        }
                        Some(crate::types::ExecutionResult::Failure(err)) => {
                            Err(format!("Workflow {} execution failed: {}", prompt_id, err))
                        }
                        Some(crate::types::ExecutionResult::Pending) => {
                            Err(format!("Workflow {} did not complete: pending", prompt_id))
                        }
                        None => {
                            Err(format!("Workflow {} returned no execution result", prompt_id))
                        }
                    }
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
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "steps": {"type": "integer", "default": 20},
                "cfg": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": -1},
                "model": {"type": "string", "default": "v1-5-pruned-emaonly.safetensors"},
                "sampler": {"type": "string", "default": "dpm++2m"},
                "scheduler": {"type": "string", "default": "karras"}
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
                        .unwrap_or(512) as usize;

                    let height = input.get("height")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(512) as usize;

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
                        "description": "Ready-to-execute workflow JSON. Use this exact value for submit_workflow's 'workflow' parameter.",
                        "usage": "Pass the 'workflow' field to submit_workflow tool",
                        "node_count": workflow.nodes.len(),
                        "params": {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "width": width,
                            "height": height,
                            "steps": steps,
                            "cfg": cfg,
                            "seed": seed,
                            "model": model
                        },
                        "example": {
                            "tool": "submit_workflow",
                            "arguments": {
                                "workflow": workflow_json
                            }
                        }
                    }))
                })
            }
        }),
        &["DA", "PA"],
    );

    // === 3. build_i2i_workflow ===
    executor.register(
        "build_i2i_workflow",
        "RECOMMENDED for image-to-image tasks. Constructs a pre-configured I2I workflow with proper node connections. Use this for: style transfer, watermark removal, image editing. Returns a ready-to-execute workflow JSON. Parameters: image_path (required), prompt, strength (default 0.75), steps, cfg, seed, model.",
        json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to input image"},
                "prompt": {"type": "string", "description": "Generation prompt"},
                "negative_prompt": {"type": "string", "default": ""},
                "strength": {"type": "number", "default": 0.75, "description": "How much to transform the image (0.0-1.0)"},
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
                        "description": "Ready-to-execute workflow JSON. Use this exact value for submit_workflow's 'workflow' parameter.",
                        "usage": "Pass the 'workflow' field to submit_workflow tool",
                        "node_count": 4,
                        "params": {
                            "prompt": prompt,
                            "strength": strength,
                            "model": model,
                            "steps": steps,
                            "cfg": cfg,
                            "seed": seed
                        },
                        "example": {
                            "tool": "submit_workflow",
                            "arguments": {
                                "workflow": workflow_json
                            }
                        }
                    }))
                })
            }
        }),
        &["DA"],
    );

    // === 3.5. build_i2v_workflow ===
    // 构建原生图生视频工作流（Wan 文本可控或 SVD 快速动画）。
    executor.register(
        "build_i2v_workflow",
        "Build a native image-to-video workflow. The configured Wan model follows the prompt; SVD remains available as a fast image-only fallback.",
        json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Input image path (e.g., 'input/bk_0015.jpg')"},
                "prompt": {"type": "string", "description": "Requested subject motion and scene behavior"},
                "negative_prompt": {"type": "string", "default": "static, blurry, distorted", "description": "Undesired motion or quality traits"},
                "model": {"type": "string", "description": "Optional native video model; omit to use the configured Wan/SVD default"},
                "width": {"type": "integer", "default": 1024, "description": "Delivery width; native inference resolution is selected automatically"},
                "height": {"type": "integer", "default": 576, "description": "Delivery height; native inference resolution is selected automatically"},
                "frames": {"type": "integer", "default": 25, "description": "Number of video frames (25 = ~5s at 5fps)"},
                "fps": {"type": "integer", "default": 5, "description": "Output video FPS (5fps × 5s = 25 frames)"},
                "motion_bucket_id": {"type": "integer", "default": 127, "description": "Motion intensity (127 = standard)"},
                "cfg": {"type": "number", "default": 6.0, "description": "Text CFG for Wan or final-frame CFG for SVD"},
                "min_cfg": {"type": "number", "default": 1.0, "description": "First-frame CFG scale for SVD"},
                "noise_aug_strength": {"type": "number", "default": 0.02, "description": "Conditioning-image noise augmentation"},
                "steps": {"type": "integer", "default": 25, "description": "Sampling steps"},
                "seed": {"type": "integer", "default": -1}
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
                        .unwrap_or("static, blurry, distorted");

                    // 验证图片文件存在
                    let check_path = if std::path::Path::new(image_path).exists() {
                        image_path.to_string()
                    } else {
                        let alt = format!("input/{}", image_path);
                        if std::path::Path::new(&alt).exists() {
                            alt
                        } else {
                            return Err(format!(
                                "Image file not found: {} (also tried input/{})",
                                image_path, image_path
                            ));
                        }
                    };

                    let model = input.get("model")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| _ctx.app_config.sd_cpp.video_model_path.clone());

                    let frames = input.get("frames")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(25) as usize;

                    let width = input.get("width")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1024) as usize;

                    let height = input.get("height")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(576) as usize;

                    let fps = input.get("fps")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(5) as usize;

                    let motion_bucket_id = input.get("motion_bucket_id")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(127) as i32;

                    let cfg = input.get("cfg")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(6.0) as f32;

                    let min_cfg = input.get("min_cfg")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32;

                    let noise_aug_strength = input.get("noise_aug_strength")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.02) as f32;

                    let steps = input.get("steps")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(25) as usize;

                    let seed = input.get("seed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(-1);
                    let seed = if seed < 0 { rand::random::<i64>() } else { seed };

                    // 构建 image-to-video 工作流 JSON
                    // 3节点：LoadImage → SVDImageToVideo (保存视频到output目录)
                    // SVDImageToVideo已生成视频文件，直接输出VIDEO路径
                    let workflow_json = json!({
                        "nodes": {
                            "1": {
                                "class_type": "CheckpointLoaderSimple",
                                "inputs": {
                                    "ckpt_name": model
                                }
                            },
                            "2": {
                                "class_type": "LoadImage",
                                "inputs": {
                                    "image": check_path
                                }
                            },
                            "3": {
                                "class_type": "SVDImageToVideo",
                                "inputs": {
                                    "model": ["1", "MODEL"],
                                    "image": ["2", "IMAGE"],
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt,
                                    "width": width,
                                    "height": height,
                                    "frames": frames,
                                    "fps": fps,
                                    "motion_bucket_id": motion_bucket_id,
                                    "cfg": cfg,
                                    "min_cfg": min_cfg,
                                    "noise_aug_strength": noise_aug_strength,
                                    "steps": steps,
                                    "seed": seed
                                }
                            }
                        }
                    });

                    Ok(json!({
                        "workflow": workflow_json,
                        "description": "Ready-to-execute workflow JSON. Use this exact value for submit_workflow's 'workflow' parameter.",
                        "usage": "Pass the 'workflow' field to submit_workflow tool",
                        "node_count": 3,
                        "params": {
                            "image_path": check_path,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "model": model,
                            "width": width,
                            "height": height,
                            "frames": frames,
                            "fps": fps,
                            "motion_bucket_id": motion_bucket_id,
                            "cfg": cfg,
                            "min_cfg": min_cfg,
                            "noise_aug_strength": noise_aug_strength,
                            "steps": steps,
                            "seed": seed
                        },
                        "example": {
                            "tool": "submit_workflow",
                            "arguments": {
                                "workflow": workflow_json
                            }
                        }
                    }))
                })
            }
        }),
        &["DA", "PA"],
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
                        crate::types::Value::Conditioning(String::new()),
                        crate::types::Value::Conditioning(String::new()),
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

    log::info!("Registered native media and compatibility tools to ToolExecutor");
}

fn resolve_gpu_tier(config: &crate::config::AppConfig) -> GpuTier {
    GpuTier::detect(config.sd_cpp.gpu_tier.as_deref())
}

fn kind_of_model_path(path: &str) -> ImageModelKind {
    let lower = path.to_ascii_lowercase();
    if lower.contains("sdxl") || lower.contains("sd_xl") {
        ImageModelKind::Sdxl
    } else {
        ImageModelKind::Sd15
    }
}

fn select_image_model(config: &crate::config::AppConfig, tier: GpuTier, quality: &str) -> ImageModelChoice {
    let fast_fallback = || {
        let path = if !config.sd_cpp.fast_model_path.trim().is_empty() {
            config.sd_cpp.fast_model_path.clone()
        } else {
            config.sd_cpp.model_path.clone()
        };
        ImageModelChoice {
            path,
            kind: kind_of_model_path(&config.sd_cpp.fast_model_path),
        }
    };
    if quality == "fast" {
        return fast_fallback();
    }
    let available = |path: &str| {
        if path.trim().is_empty() || !std::path::Path::new(path).is_file() {
            return false;
        }
        // 校验容器有效性,损坏/截断的模型文件自动降级,避免选中后推理失败
        matches!(
            crate::native_runtime::inspect_model_file(path).map(|info| info.container),
            Ok(
                crate::native_runtime::ModelContainer::Safetensors
                    | crate::native_runtime::ModelContainer::Gguf
            )
        )
    };
    match tier {
        GpuTier::Tier16G => {
            if available(&config.sd_cpp.model_path) {
                ImageModelChoice {
                    path: config.sd_cpp.model_path.clone(),
                    kind: kind_of_model_path(&config.sd_cpp.model_path),
                }
            } else {
                fast_fallback()
            }
        }
        GpuTier::Tier12G => {
            if available(&config.sd_cpp.sdxl_gguf_q5_path) {
                ImageModelChoice {
                    path: config.sd_cpp.sdxl_gguf_q5_path.clone(),
                    kind: ImageModelKind::Sdxl,
                }
            } else if available(&config.sd_cpp.model_path) {
                ImageModelChoice {
                    path: config.sd_cpp.model_path.clone(),
                    kind: kind_of_model_path(&config.sd_cpp.model_path),
                }
            } else {
                fast_fallback()
            }
        }
        GpuTier::Tier8G => {
            if available(&config.sd_cpp.sdxl_gguf_q4_path) {
                ImageModelChoice {
                    path: config.sd_cpp.sdxl_gguf_q4_path.clone(),
                    kind: ImageModelKind::Sdxl,
                }
            } else {
                fast_fallback()
            }
        }
        GpuTier::Tier4G => fast_fallback(),
    }
}

fn sd15_loras(config: &crate::config::AppConfig) -> Vec<LoraSpec> {
    let path = config.sd_cpp.image_lora_path.trim();
    if path.is_empty() || !std::path::Path::new(path).is_file() {
        return Vec::new();
    }
    vec![LoraSpec {
        path: path.to_string(),
        multiplier: config.sd_cpp.image_lora_scale,
    }]
}

fn resize_to_delivery(
    bytes: Vec<u8>,
    delivery_width: u32,
    delivery_height: u32,
    gen_width: u32,
    gen_height: u32,
) -> Result<Vec<u8>, String> {
    if delivery_width == gen_width && delivery_height == gen_height {
        return Ok(bytes);
    }
    let image = image::load_from_memory(&bytes)
        .map_err(|error| format!("cannot decode generated image for delivery scaling: {}", error))?;
    let resized = image.resize_exact(
        delivery_width.max(8),
        delivery_height.max(8),
        image::imageops::FilterType::Lanczos3,
    );
    let mut output = Vec::new();
    resized
        .write_to(
            &mut std::io::Cursor::new(&mut output),
            image::ImageFormat::Png,
        )
        .map_err(|error| format!("cannot re-encode delivery image: {}", error))?;
    Ok(output)
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn interpolate_video_to_fps(path: &str, target_fps: u32) -> Result<bool, String> {
    if !ffmpeg_available() {
        return Ok(false);
    }
    let temporary = format!("{}.interpolated.mp4", path);
    let output = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-vf")
        .arg(format!(
            "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:vsbmc=1:scd=none",
            target_fps
        ))
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("slow")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-movflags")
        .arg("+faststart")
        .arg(&temporary)
        .output()
        .map_err(|error| format!("cannot start ffmpeg interpolation: {}", error))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg interpolation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    std::fs::rename(&temporary, path)
        .map_err(|error| format!("cannot replace video with interpolated version: {}", error))?;
    Ok(true)
}

async fn execute_native_generation(
    ctx: &AgentContext,
    mut spec: NativeGenerationSpec,
) -> Result<Value, String> {
    let request = ctx.generation_request.read().await.clone();
    apply_generation_request_constraints(&mut spec, request.as_ref())?;
    compile_model_prompt(ctx, &mut spec, request.as_ref()).await?;
    validate_native_generation_spec(&spec)?;
    apply_quality_prompt_profile(&mut spec);
    log::info!(
        "Native media prompt compiled (intent={}, prompt={:?}, negative_prompt={:?})",
        spec.intent,
        spec.prompt,
        spec.negative_prompt
    );
    let effective_prompt = spec.prompt.clone();
    let effective_negative_prompt = spec.negative_prompt.clone();
    wait_for_local_llama_sleep(&ctx.app_config).await?;
    let quality = spec.quality.as_deref().unwrap_or("balanced").to_string();
    let tier = resolve_gpu_tier(&ctx.app_config);
    let mut seed = match spec.seed.unwrap_or(-1) {
        value if value < 0 => rand::random::<u32>() as i64,
        value => value,
    };
    let is_video = matches!(spec.intent.as_str(), "text_to_video" | "image_to_video");
    let output_dir = std::path::Path::new(&ctx.app_config.server.output_dir);
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("Failed to create output directory: {}", error))?;
    let extension = if is_video { "mp4" } else { "png" };
    let output_path = output_dir
        .join(format!("agent_{}.{}", uuid::Uuid::new_v4(), extension))
        .to_string_lossy()
        .into_owned();

    let default_image_model = ctx.app_config.sd_cpp.model_path.clone();
    let mut pipeline = match spec.intent.as_str() {
        "text_to_image" => "native_t2i",
        "image_to_image" => "native_i2i",
        "text_to_video" => "native_t2v",
        "image_to_video" => "native_i2v",
        _ => "native",
    }
    .to_string();
    let mut effective_parameters = json!({});
    let bytes = match spec.intent.as_str() {
        "text_to_image" => {
            let image_model = select_image_model(&ctx.app_config, tier, &quality);
            let default_edge = if image_model.kind == ImageModelKind::Sdxl { 1024 } else { 512 };
            let delivery_width = spec.width.unwrap_or(default_edge) as u32;
            let delivery_height = spec.height.unwrap_or(default_edge) as u32;
            let (gen_width, gen_height) =
                tier.generation_canvas(image_model.kind, &quality, delivery_width, delivery_height);
            let steps = spec.steps.unwrap_or(tier.default_steps(&quality) as usize);
            let cfg = spec.cfg.unwrap_or(image_model.kind.default_cfg());
            let hires = tier
                .hires_policy(&quality)
                .map(|policy| HiresSpec {
                    scale: policy.scale,
                    steps: policy.steps,
                    denoising_strength: policy.denoising_strength,
                });
            if hires.is_some() {
                pipeline = "native_t2i_hires".to_string();
            }
            let loras = if image_model.kind == ImageModelKind::Sd15 {
                sd15_loras(&ctx.app_config)
            } else {
                Vec::new()
            };
            let mut effective_hires = hires;
            let mut chosen_steps = steps;
            let mut chosen_cfg = cfg;
            let mut selected: Option<(Vec<u8>, f32, i64)> = None;
            let mut last_error: Option<String> = None;
            let mut iterations = 0_u32;
            let attempts = if quality == "high" { 2 } else { 1 };
            for attempt in 0..attempts {
                iterations += 1;
                let attempt_seed = seed.wrapping_add(attempt as i64);
                match ctx
                    .backend
                    .text_to_image(crate::backend::T2IParams {
                        prompt: spec.prompt.clone(),
                        negative_prompt: spec.negative_prompt.clone(),
                        width: gen_width as usize,
                        height: gen_height as usize,
                        steps: chosen_steps,
                        cfg: chosen_cfg,
                        sampler: "dpm++2m_karras".to_string(),
                        seed: attempt_seed as usize,
                        model_path: image_model.path.clone(),
                        loras: loras.clone(),
                        hires: effective_hires,
                    })
                    .await
                {
                    Ok(raw) => {
                        let delivery = resize_to_delivery(
                            raw,
                            delivery_width,
                            delivery_height,
                            gen_width,
                            gen_height,
                        )?;
                        let quality_score = score_image_bytes(&delivery)
                            .map(|score| score.overall)
                            .unwrap_or(0.5);
                        let scored = score_image_bytes(&delivery).ok();
                        let keep = selected
                            .as_ref()
                            .map(|(_, best_score, _)| quality_score > *best_score)
                            .unwrap_or(true);
                        if keep {
                            selected = Some((delivery, quality_score, attempt_seed));
                        }
                        log::info!(
                            "Native T2I attempt {} scored {:.3} (tier={}, canvas={}x{}, seed={})",
                            attempt + 1,
                            quality_score,
                            tier.name(),
                            gen_width,
                            gen_height,
                            attempt_seed
                        );
                        if let Some(score) = scored {
                            if !score.acceptable()
                                && adjust_for_retry(
                                    &mut chosen_steps,
                                    &mut chosen_cfg,
                                    &score.failure_modes(),
                                )
                            {
                                continue;
                            }
                        }
                        break;
                    }
                    Err(error) => {
                        log::warn!("Native T2I attempt {} failed: {}", attempt + 1, error);
                        if effective_hires.is_some() && attempt == 0 {
                            log::warn!("T2I with hires-fix failed; retrying without hires-fix");
                            effective_hires = None;
                            continue;
                        }
                        last_error = Some(error.to_string());
                        break;
                    }
                }
            }
            let (bytes, quality_score, final_seed) = selected
                .ok_or_else(|| last_error.unwrap_or_else(|| "Native T2I produced no output".to_string()))?;
            effective_parameters = json!({
                "width": delivery_width,
                "height": delivery_height,
                "generation_width": gen_width,
                "generation_height": gen_height,
                "steps": chosen_steps,
                "cfg": chosen_cfg,
                "hires": effective_hires.map(|h| json!({
                    "scale": h.scale,
                    "steps": h.steps,
                    "denoising_strength": h.denoising_strength
                })).unwrap_or(Value::Null),
                "gpu_tier": tier.name(),
                "quality_score": quality_score,
                "iterations": iterations,
                "model": image_model.path
            });
            log::info!(
                "Native T2I delivered {}x{} (selected quality score {:.3}, {} attempts)",
                delivery_width,
                delivery_height,
                quality_score,
                iterations
            );
            seed = final_seed;
            Ok::<Vec<u8>, String>(bytes)
        }
        "image_to_image" => {
            let image_path = require_input_image(spec.image_path.as_deref())?;
            let input_image = std::fs::read(&image_path)
                .map_err(|error| format!("Failed to read input image '{}': {}", image_path, error))?;
            let image_model = select_image_model(&ctx.app_config, tier, &quality);
            let default_edge = if image_model.kind == ImageModelKind::Sdxl { 1024 } else { 512 };
            let delivery_width = spec.width.unwrap_or(default_edge) as u32;
            let delivery_height = spec.height.unwrap_or(default_edge) as u32;
            let (gen_width, gen_height) =
                tier.generation_canvas(image_model.kind, &quality, delivery_width, delivery_height);
            let steps = spec.steps.unwrap_or(tier.default_steps(&quality) as usize);
            let cfg = spec.cfg.unwrap_or(image_model.kind.default_cfg());
            let strength = spec.strength.unwrap_or(0.45);
            let loras = if image_model.kind == ImageModelKind::Sd15 {
                sd15_loras(&ctx.app_config)
            } else {
                Vec::new()
            };
            let raw = ctx
                .backend
                .image_to_image(crate::backend::I2IParams {
                    prompt: spec.prompt,
                    negative_prompt: spec.negative_prompt,
                    input_image,
                    denoise: strength,
                    steps,
                    cfg,
                    sampler: "dpm++2m_karras".to_string(),
                    width: gen_width as usize,
                    height: gen_height as usize,
                    seed: seed as usize,
                    model_path: image_model.path.clone(),
                    loras,
                })
                .await
                .map_err(|error| format!("Native image-to-image failed: {}", error))?;
            let delivery = resize_to_delivery(
                raw,
                delivery_width,
                delivery_height,
                gen_width,
                gen_height,
            )?;
            let i2i_score = score_image_bytes(&delivery)
                .map(|score| score.overall)
                .unwrap_or(0.5);
            effective_parameters = json!({
                "width": delivery_width,
                "height": delivery_height,
                "generation_width": gen_width,
                "generation_height": gen_height,
                "steps": steps,
                "cfg": cfg,
                "strength": strength,
                "gpu_tier": tier.name(),
                "model": image_model.path,
                "quality_score": i2i_score,
                "iterations": 1
            });
            Ok::<Vec<u8>, String>(delivery)
        }
        "text_to_video" => {
            let model = spec.model.unwrap_or_else(|| {
                let config = &ctx.app_config.sd_cpp;
                if tier.prefer_svd_video() && !config.svd_model_path.trim().is_empty() {
                    config.svd_model_path.clone()
                } else {
                    config.video_model_path.clone()
                }
            });
            if model.trim().is_empty() {
                return Err(
                    "text_to_video requires sd_cpp.video_model_path or an explicit native video model"
                        .to_string(),
                );
            }
            let model_name = model.to_ascii_lowercase();
            let is_svd = model_name.contains("svd")
                || model_name.contains("stable-video-diffusion");
            let width = spec.width.unwrap_or(if is_svd {
                ctx.app_config.sd_cpp.svd_native_width
            } else {
                ctx.app_config.sd_cpp.semantic_video_native_width
            });
            let height = spec.height.unwrap_or(if is_svd {
                ctx.app_config.sd_cpp.svd_native_height
            } else {
                ctx.app_config.sd_cpp.semantic_video_native_height
            });
            let video_params = crate::backend::I2VParams {
                prompt: spec.prompt.clone(),
                negative_prompt: spec.negative_prompt.clone(),
                input_image: Vec::new(),
                width,
                height,
                frames: spec
                    .frames
                    .unwrap_or(25)
                    .min(tier.video_max_frames() as usize),
                fps: spec.fps.unwrap_or(5),
                motion_bucket_id: spec.motion_bucket_id.unwrap_or(127),
                motion_scale: 1024.0,
                steps: spec.steps.unwrap_or(if quality == "high" { 30 } else { 25 }),
                cfg: spec.cfg.unwrap_or(if is_svd { 3.0 } else { 6.0 }),
                min_cfg: spec.min_cfg.unwrap_or(1.0),
                noise_aug_strength: spec.noise_aug_strength.unwrap_or(0.02),
                seed: seed as usize,
                model_path: model.clone(),
            };
            effective_parameters = json!({
                "width": width,
                "height": height,
                "frames": video_params.frames,
                "fps": video_params.fps,
                "steps": video_params.steps,
                "cfg": video_params.cfg,
                "min_cfg": video_params.min_cfg,
                "motion_bucket_id": video_params.motion_bucket_id,
                "noise_aug_strength": video_params.noise_aug_strength,
                "gpu_tier": tier.name()
            });

            if quality == "high" {
                pipeline = "native_t2i_keyframe_to_i2v".to_string();
                let keyframe_model = select_image_model(&ctx.app_config, tier, &quality);
                let keyframe_max_edge = if keyframe_model.kind == ImageModelKind::Sdxl {
                    1024
                } else {
                    768
                };
                let (keyframe_width, keyframe_height) =
                    fit_dimensions_within(width, height, keyframe_max_edge, 8);
                let keyframe_steps = tier.default_steps(&quality) as usize;
                let keyframe_cfg = keyframe_model.kind.default_cfg();
                let keyframe_loras = if keyframe_model.kind == ImageModelKind::Sd15 {
                    sd15_loras(&ctx.app_config)
                } else {
                    Vec::new()
                };
                effective_parameters["keyframe_width"] = json!(keyframe_width);
                effective_parameters["keyframe_height"] = json!(keyframe_height);
                effective_parameters["keyframe_steps"] = json!(keyframe_steps);
                effective_parameters["keyframe_model"] = json!(keyframe_model.path);
                let keyframe = ctx
                    .backend
                    .text_to_image(crate::backend::T2IParams {
                        prompt: spec.prompt,
                        negative_prompt: spec.negative_prompt,
                        width: keyframe_width,
                        height: keyframe_height,
                        steps: keyframe_steps,
                        cfg: keyframe_cfg,
                        sampler: "dpm++2m_karras".to_string(),
                        seed: seed as usize,
                        model_path: keyframe_model.path,
                        loras: keyframe_loras,
                        hires: None,
                    })
                    .await
                    .map_err(|error| format!("Native keyframe generation failed: {}", error))?;
                ctx.backend
                    .image_to_video(crate::backend::I2VParams {
                        input_image: keyframe,
                        ..video_params
                    })
                    .await
                    .map_err(|error| format!("Native keyframe video generation failed: {}", error))
            } else {
                ctx.backend
                    .text_to_video(crate::backend::T2VParams {
                        prompt: spec.prompt,
                        negative_prompt: spec.negative_prompt,
                        width,
                        height,
                        frames: video_params.frames,
                        fps: video_params.fps,
                        motion_bucket_id: video_params.motion_bucket_id,
                        steps: video_params.steps,
                        cfg: video_params.cfg,
                        min_cfg: video_params.min_cfg,
                        noise_aug_strength: video_params.noise_aug_strength,
                        seed: video_params.seed,
                        model_path: model,
                    })
                    .await
                    .map_err(|error| format!("Native text-to-video generation failed: {}", error))
            }
        }
        "image_to_video" => {
            let image_path = require_input_image(spec.image_path.as_deref())?;
            let input_image = std::fs::read(&image_path)
                .map_err(|error| format!("Failed to read input image '{}': {}", image_path, error))?;
            let model = spec.model.unwrap_or_else(|| {
                let config = &ctx.app_config.sd_cpp;
                if tier.prefer_svd_video() && !config.svd_model_path.trim().is_empty() {
                    config.svd_model_path.clone()
                } else {
                    config.video_model_path.clone()
                }
            });
            let model_name = model.to_ascii_lowercase();
            let is_svd = model_name.contains("svd")
                || model_name.contains("stable-video-diffusion");
            let width = spec.width.unwrap_or(1024);
            let height = spec.height.unwrap_or(576);
            let frames = spec
                .frames
                .unwrap_or(25)
                .min(tier.video_max_frames() as usize);
            let fps = spec.fps.unwrap_or(5);
            let motion_bucket_id = spec.motion_bucket_id.unwrap_or(127);
            let steps = spec.steps.unwrap_or(if quality == "high" { 30 } else { 25 });
            let cfg = spec.cfg.unwrap_or(if is_svd { 3.0 } else { 6.0 });
            let min_cfg = spec.min_cfg.unwrap_or(1.0);
            let noise_aug_strength = spec.noise_aug_strength.unwrap_or(0.02);
            effective_parameters = json!({
                "width": width,
                "height": height,
                "frames": frames,
                "fps": fps,
                "steps": steps,
                "cfg": cfg,
                "min_cfg": min_cfg,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_strength": noise_aug_strength
            });
            ctx.backend
                .image_to_video(crate::backend::I2VParams {
                    prompt: spec.prompt,
                    negative_prompt: spec.negative_prompt,
                    input_image,
                    width,
                    height,
                    frames,
                    fps,
                    motion_bucket_id,
                    motion_scale: 1024.0,
                    steps,
                    cfg,
                    min_cfg,
                    noise_aug_strength,
                    seed: seed as usize,
                    model_path: model,
                })
                .await
                .map_err(|error| format!("Native image-to-video generation failed: {}", error))
        }
        other => return Err(format!("Unsupported generation intent: {}", other)),
    }
    .map_err(|error| format!("Native generation failed: {}", error))?;

    if bytes.is_empty() {
        return Err("Native generation returned empty media data".to_string());
    }
    std::fs::write(&output_path, &bytes)
        .map_err(|error| format!("Failed to write generated media '{}': {}", output_path, error))?;

    if is_video && quality == "high" {
        let source_fps = spec.fps.unwrap_or(5);
        let target_fps = (source_fps * 2).clamp(source_fps + 1, 30);
        match interpolate_video_to_fps(&output_path, target_fps as u32) {
            Ok(true) => {
                log::info!(
                    "Native video interpolated {}fps -> {}fps: {}",
                    source_fps,
                    target_fps,
                    output_path
                );
            }
            Ok(false) => {
                log::info!(
                    "ffmpeg unavailable; skipping video interpolation (zero-dependency fallback): {}",
                    output_path
                );
            }
            Err(error) => {
                log::warn!("Video interpolation failed, keeping original: {}", error);
            }
        }
    }

    let artifact = inspect_media_artifact(&output_path)?;
    let quality_score = effective_parameters
        .get("quality_score")
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let iterations = effective_parameters
        .get("iterations")
        .and_then(Value::as_u64)
        .unwrap_or(1) as u32;
    {
        let mut audit = ctx.generation_audit.write().await;
        *audit = Some(crate::agent::context::GenerationAudit {
            intent: spec.intent.clone(),
            pipeline: pipeline.clone(),
            effective_prompt: effective_prompt.clone(),
            effective_negative_prompt: effective_negative_prompt.clone(),
            quality: quality.clone(),
            seed,
            parameters: effective_parameters.clone(),
            output_path: output_path.clone(),
            quality_score,
            iterations,
        });
    }
    Ok(json!({
        "status": "success",
        "intent": spec.intent,
        "pipeline": pipeline,
        "quality": quality,
        "seed": seed,
        "effective_parameters": effective_parameters,
        "effective_prompt": effective_prompt,
        "effective_negative_prompt": effective_negative_prompt,
        "output_path": output_path,
        "artifact": artifact,
        "artifacts": [artifact]
    }))
}

async fn compile_model_prompt(
    ctx: &AgentContext,
    spec: &mut NativeGenerationSpec,
    request: Option<&GenerationRequestContext>,
) -> Result<(), String> {
    let Some(request) = request else {
        return Ok(());
    };
    if request.user_request.trim().is_empty() {
        return Ok(());
    }

    let draft = spec.prompt.clone();
    let gateway = ctx
        .gateway
        .as_ref()
        .ok_or_else(|| "Prompt compiler requires the configured Agent LLM gateway".to_string())?;
    let task_guidance = match spec.intent.as_str() {
        "image_to_image" => {
            "Describe the requested visual transformation while explicitly preserving the uploaded subject's identity and requested composition. Include setting, lighting, style, and requested details."
        }
        "image_to_video" => {
            "Describe the requested motion, action, scene evolution, and temporal consistency while preserving the uploaded subject's identity and appearance. Include camera behavior only when the user requested it; otherwise keep the camera stable."
        }
        "text_to_video" => {
            "Describe the requested subjects and their explicit actions with strong motion verbs, how the scene evolves over time, and temporal consistency. Describe the shot and camera movement only when the user requested it; otherwise keep the camera stable."
        }
        _ => {
            "Describe the visible subject, setting, time, weather, lighting, style, composition, and requested details."
        }
    };
    let system = format!(
        "You are a media-generation prompt compiler. Rewrite the exact user request as one model-ready English prompt for intent '{}'. {} Return only one plain-text line of 25-80 words: no heading, quotes, Markdown, explanation, parameters, or negative prompt. Preserve every requested subject, action, count, setting, time, weather, lighting, visual style, and composition. Never add an unrelated subject or scene. End with a complete noun or action phrase; never end on a determiner, conjunction, or preposition.",
        spec.intent, task_guidance
    );
    let messages = vec![
        glidinghorse::gateway::unified_gateway::ChatMessage {
            role: "system".to_string(),
            content: system,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        glidinghorse::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: request.user_request.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    let model = gateway.default_model();
    let compiled = gateway
        .chat_with_params(&model, messages, Some(0.0), Some(1024), None, None)
        .await
        .map_err(|error| error.to_string())
        .and_then(|response| {
            response
                .choices
                .first()
                .and_then(|choice| choice.message.content.as_deref())
                .map(normalize_compiled_prompt)
                .filter(|prompt| is_model_ready_english_prompt(prompt))
                .ok_or_else(|| "Prompt compiler returned no valid model-ready English prompt".to_string())
        });

    match compiled {
        Ok(prompt) => {
            spec.prompt = prompt;
            Ok(())
        }
        Err(error) if is_model_ready_english_prompt(&draft) => {
            log::warn!("Prompt compiler failed; using validated DA prompt: {}", error);
            spec.prompt = draft;
            Ok(())
        }
        Err(error) => Err(format!(
            "Prompt compilation failed and the DA prompt is not safe for native inference: {}",
            error
        )),
    }
}

fn normalize_compiled_prompt(value: &str) -> String {
    let mut prompt = value
        .trim()
        .trim_matches('`')
        .trim()
        .trim_matches(|character| matches!(character, '\'' | '"'))
        .trim()
        .replace(['\r', '\n'], " ");
    for prefix in ["Prompt:", "English prompt:", "Stable Diffusion prompt:"] {
        if prompt
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        {
            prompt = prompt[prefix.len()..].trim().to_string();
            break;
        }
    }
    prompt.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_model_ready_english_prompt(prompt: &str) -> bool {
    let word_count = prompt.split_whitespace().count();
    let alphabetic_count = prompt
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .count();
    let contains_cjk = prompt.chars().any(|character| {
        matches!(character as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
    });
    let terminal_word = prompt
        .trim_end_matches(|character: char| !character.is_ascii_alphabetic())
        .split_whitespace()
        .last()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let has_incomplete_tail = matches!(
        terminal_word.as_str(),
        "a" | "an" | "the" | "and" | "or" | "but" | "of" | "to" | "with" | "for"
            | "from" | "in" | "on" | "at" | "by" | "through" | "toward" | "towards"
            | "over" | "under" | "its" | "their" | "his" | "her"
    );
    !contains_cjk
        && (8..=120).contains(&word_count)
        && alphabetic_count >= 24
        && prompt.len() <= 1_500
        && !has_incomplete_tail
}

fn apply_generation_request_constraints(
    spec: &mut NativeGenerationSpec,
    request: Option<&GenerationRequestContext>,
) -> Result<(), String> {
    let Some(request) = request else {
        return Ok(());
    };

    spec.intent = request.intent.clone();
    if matches!(spec.intent.as_str(), "image_to_image" | "image_to_video") {
        spec.image_path = request.image_path.clone();
    }

    let explicit_keys = match request.params.get("_explicit_keys") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .collect::<std::collections::HashSet<_>>(),
        Some(_) => {
            return Err("UI parameter '_explicit_keys' must be an array of strings".to_string());
        }
        None => request
            .params
            .as_object()
            .map(|params| params.keys().map(String::as_str).collect())
            .unwrap_or_default(),
    };

    // Model paths are a trust boundary. Agent-supplied aliases are discarded unless
    // the user explicitly selected a model and preflight already validated it.
    spec.model = if explicit_keys.contains("model") {
        explicit_string(&request.params, "model")?.filter(|value| !value.trim().is_empty())
    } else {
        None
    };

    if explicit_keys.contains("negative_prompt") {
        spec.negative_prompt = explicit_string(&request.params, "negative_prompt")?
            .unwrap_or_default();
    }
    if explicit_keys.contains("quality") {
        spec.quality = explicit_string(&request.params, "quality")?;
    }
    if explicit_keys.contains("width") {
        spec.width = explicit_usize(&request.params, "width")?;
    }
    if explicit_keys.contains("height") {
        spec.height = explicit_usize(&request.params, "height")?;
    }
    if explicit_keys.contains("steps") {
        spec.steps = explicit_usize(&request.params, "steps")?;
    }
    if explicit_keys.contains("cfg") {
        spec.cfg = explicit_f32(&request.params, "cfg")?;
    }
    if explicit_keys.contains("min_cfg") {
        spec.min_cfg = explicit_f32(&request.params, "min_cfg")?;
    }
    if explicit_keys.contains("noise_aug_strength") {
        spec.noise_aug_strength = explicit_f32(&request.params, "noise_aug_strength")?;
    }
    if explicit_keys.contains("seed") {
        spec.seed = explicit_i64(&request.params, "seed")?;
    }
    if explicit_keys.contains("strength") {
        spec.strength = explicit_f32(&request.params, "strength")?;
    }
    if explicit_keys.contains("frames") {
        spec.frames = explicit_usize(&request.params, "frames")?;
    }
    if explicit_keys.contains("fps") {
        spec.fps = explicit_usize(&request.params, "fps")?;
    }
    if explicit_keys.contains("motion_bucket_id") {
        spec.motion_bucket_id = explicit_i64(&request.params, "motion_bucket_id")?
            .map(|value| i32::try_from(value).map_err(|_| {
                "Explicit UI parameter 'motion_bucket_id' is outside the i32 range".to_string()
            }))
            .transpose()?;
    }

    if spec.intent == "image_to_image" {
        apply_adaptive_i2i_defaults(
            spec,
            request,
            explicit_keys.contains("width"),
            explicit_keys.contains("height"),
            explicit_keys.contains("strength"),
        )?;
    }

    Ok(())
}

fn apply_adaptive_i2i_defaults(
    spec: &mut NativeGenerationSpec,
    request: &GenerationRequestContext,
    width_explicit: bool,
    height_explicit: bool,
    strength_explicit: bool,
) -> Result<(), String> {
    if !strength_explicit {
        spec.strength = Some(if requests_source_preservation(&request.user_request) {
            0.30
        } else {
            0.45
        });
    }

    if width_explicit && height_explicit {
        return Ok(());
    }

    let image_path = require_input_image(spec.image_path.as_deref())?;
    let (source_width, source_height) = image::image_dimensions(&image_path)
        .map_err(|error| format!("Cannot inspect I2I source dimensions: {}", error))?;
    let (width, height) = if width_explicit {
        let width = spec.width.unwrap_or(512);
        let height = aligned_dimension(
            width as f64 * source_height as f64 / source_width as f64,
            8,
        );
        (width, height)
    } else if height_explicit {
        let height = spec.height.unwrap_or(512);
        let width = aligned_dimension(
            height as f64 * source_width as f64 / source_height as f64,
            8,
        );
        (width, height)
    } else {
        let short_edge = match spec.quality.as_deref().unwrap_or("balanced") {
            "fast" => 384,
            "high" => 576,
            _ => 512,
        };
        let scale = short_edge as f64 / source_width.min(source_height) as f64;
        let max_scaled = source_width.max(source_height) as f64 * scale;
        let bounded_scale = if max_scaled > 1024.0 {
            1024.0 / source_width.max(source_height) as f64
        } else {
            scale
        };
        (
            aligned_dimension(source_width as f64 * bounded_scale, 8),
            aligned_dimension(source_height as f64 * bounded_scale, 8),
        )
    };

    spec.width = Some(width);
    spec.height = Some(height);
    Ok(())
}

fn requests_source_preservation(request: &str) -> bool {
    let normalized = request.to_ascii_lowercase();
    [
        "preserve identity",
        "preserve the identity",
        "keep identity",
        "same person",
        "same subject",
        "preserve subject",
        "preserve composition",
        "keep composition",
        "保留人物",
        "保持人物",
        "保留身份",
        "保持身份",
        "人物不变",
        "主体不变",
        "保留构图",
        "保持构图",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase))
}

fn fit_dimensions_within(
    width: usize,
    height: usize,
    max_edge: usize,
    alignment: usize,
) -> (usize, usize) {
    let longest = width.max(height);
    let scale = if longest > max_edge {
        max_edge as f64 / longest as f64
    } else {
        1.0
    };
    (
        aligned_dimension(width as f64 * scale, alignment),
        aligned_dimension(height as f64 * scale, alignment),
    )
}

fn aligned_dimension(value: f64, alignment: usize) -> usize {
    let aligned = ((value / alignment as f64).round() as usize).max(1) * alignment;
    aligned.clamp(64, 4096)
}

fn apply_quality_prompt_profile(spec: &mut NativeGenerationSpec) {
    let quality = spec.quality.as_deref().unwrap_or("balanced");
    let is_video = matches!(spec.intent.as_str(), "text_to_video" | "image_to_video");
    let is_image_to_image = spec.intent == "image_to_image";
    let positive = match (is_video, is_image_to_image, quality) {
        (_, _, "fast") => "",
        (false, true, "high") => {
            "preserved subject identity, preserved composition, highly detailed, accurate lighting, refined textures, sharp focus"
        }
        (false, true, _) => {
            "preserved subject identity, preserved composition, high detail, coherent transformation"
        }
        (false, false, "high") => {
            "highly detailed, coherent composition, clear subject, accurate lighting, refined textures, sharp focus"
        }
        (false, false, _) => "high detail, coherent composition, clear subject",
        (true, _, "high") => {
            "natural expressive motion, consistent subject identity, stable anatomy, temporally coherent details, smooth camera movement"
        }
        (true, _, _) => "natural coherent motion, consistent subject identity, temporally stable details",
    };
    let negative = if is_video {
        "low quality, static, frozen pose, jitter, flicker, identity drift, distorted anatomy, extra limbs, camera shake"
    } else {
        "low quality, blurry, malformed, deformed, duplicate subjects, text, watermark"
    };

    append_prompt_clause(&mut spec.prompt, positive);
    append_prompt_clause(&mut spec.negative_prompt, negative);
}

fn append_prompt_clause(prompt: &mut String, clause: &str) {
    if clause.is_empty() || prompt.to_ascii_lowercase().contains(&clause.to_ascii_lowercase()) {
        return;
    }
    if !prompt.trim().is_empty() {
        prompt.push_str(", ");
    }
    prompt.push_str(clause);
}

fn explicit_string(params: &Value, key: &str) -> Result<Option<String>, String> {
    params
        .get(key)
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("Explicit UI parameter '{}' must be a string", key))
        })
        .transpose()
}

fn explicit_usize(params: &Value, key: &str) -> Result<Option<usize>, String> {
    params
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("Explicit UI parameter '{}' must be a non-negative integer", key))
        })
        .transpose()
}

fn explicit_i64(params: &Value, key: &str) -> Result<Option<i64>, String> {
    params
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| format!("Explicit UI parameter '{}' must be an integer", key))
        })
        .transpose()
}

fn explicit_f32(params: &Value, key: &str) -> Result<Option<f32>, String> {
    params
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("Explicit UI parameter '{}' must be a finite number", key))
        })
        .transpose()
}

fn validate_native_generation_spec(spec: &NativeGenerationSpec) -> Result<(), String> {
    if spec.prompt.trim().is_empty() {
        return Err("Generation prompt must not be empty".to_string());
    }
    if !matches!(
        spec.intent.as_str(),
        "text_to_image" | "image_to_image" | "text_to_video" | "image_to_video"
    ) {
        return Err(format!("Unsupported generation intent: {}", spec.intent));
    }
    if let Some(quality) = spec.quality.as_deref() {
        if !matches!(quality, "fast" | "balanced" | "high") {
            return Err(format!("Unsupported quality profile: {}", quality));
        }
    }

    let is_video = matches!(spec.intent.as_str(), "text_to_video" | "image_to_video");
    for (name, value) in [("width", spec.width), ("height", spec.height)] {
        if let Some(value) = value {
            if !(64..=4096).contains(&value) {
                return Err(format!("{} must be between 64 and 4096", name));
            }
            let alignment = if is_video { 2 } else { 8 };
            if value % alignment != 0 {
                return Err(format!("{} must be divisible by {} for this intent", name, alignment));
            }
        }
    }
    if let Some(steps) = spec.steps {
        if !(1..=100).contains(&steps) {
            return Err("steps must be between 1 and 100".to_string());
        }
    }
    if let Some(cfg) = spec.cfg {
        if !cfg.is_finite() || !(0.1..=30.0).contains(&cfg) {
            return Err("cfg must be a finite value between 0.1 and 30".to_string());
        }
    }
    if let Some(min_cfg) = spec.min_cfg {
        if !min_cfg.is_finite() || !(0.1..=30.0).contains(&min_cfg) {
            return Err("min_cfg must be a finite value between 0.1 and 30".to_string());
        }
        if is_video {
            if let Some(cfg) = spec.cfg {
            if min_cfg > cfg {
                return Err("min_cfg must not exceed final-frame cfg".to_string());
            }
            }
        }
    }
    if let Some(noise_aug_strength) = spec.noise_aug_strength {
        if !noise_aug_strength.is_finite() || !(0.0..=1.0).contains(&noise_aug_strength) {
            return Err("noise_aug_strength must be between 0 and 1".to_string());
        }
    }
    if let Some(strength) = spec.strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err("strength must be a finite value between 0 and 1".to_string());
        }
    }
    if let Some(frames) = spec.frames {
        if !(1..=161).contains(&frames) {
            return Err("frames must be between 1 and 161".to_string());
        }
    }
    if let Some(fps) = spec.fps {
        if !(1..=60).contains(&fps) {
            return Err("fps must be between 1 and 60".to_string());
        }
    }
    if let Some(motion_bucket_id) = spec.motion_bucket_id {
        if !(0..=1023).contains(&motion_bucket_id) {
            return Err("motion_bucket_id must be between 0 and 1023".to_string());
        }
    }
    Ok(())
}

async fn wait_for_local_llama_sleep(config: &crate::config::AppConfig) -> Result<(), String> {
    if config.agent.llm.provider != crate::config::AgentLlmProvider::LlamaCpp {
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|error| format!("Cannot create llama.cpp resource probe: {}", error))?;
    let props_url = format!("{}/props", config.agent.llm.gateway_base_url());
    for _ in 0..50 {
        let response = client
            .get(&props_url)
            .send()
            .await
            .map_err(|error| format!("Cannot query llama.cpp resource state: {}", error))?;
        let props: Value = response
            .json()
            .await
            .map_err(|error| format!("Invalid llama.cpp /props response: {}", error))?;
        if props.get("is_sleeping").and_then(Value::as_bool) == Some(true) {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    Err(
        "llama.cpp did not release model memory before media inference; start it with --sleep-idle-seconds 1"
            .to_string(),
    )
}

fn require_input_image(path: Option<&str>) -> Result<String, String> {
    let path = path.ok_or_else(|| "This generation intent requires image_path".to_string())?;
    let direct = std::path::Path::new(path);
    let resolved = if direct.is_file() {
        direct.to_path_buf()
    } else {
        std::path::Path::new("input").join(path)
    };
    if !resolved.is_file() {
        return Err(format!("Input image not found: {}", resolved.display()));
    }
    image::image_dimensions(&resolved)
        .map_err(|error| format!("Input image '{}' is not decodable: {}", resolved.display(), error))?;
    Ok(resolved.to_string_lossy().into_owned())
}

pub(crate) fn inspect_media_artifact(path: &str) -> Result<Value, String> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| format!("Artifact '{}' is unavailable: {}", path, error))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!("Artifact '{}' is not a non-empty file", path));
    }
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif") {
        let decoded = image::open(path)
            .map_err(|error| format!("Artifact image '{}' cannot be decoded: {}", path, error))?;
        let width = decoded.width();
        let height = decoded.height();
        let luma = decoded.to_luma8();
        let signal = analyze_luma_signal(luma.as_raw(), luma.len())?;
        validate_visual_signal(path, &signal, false)?;
        return Ok(json!({
            "valid": true,
            "kind": "image",
            "path": path,
            "size_bytes": metadata.len(),
            "width": width,
            "height": height,
            "quality": signal.to_json()
        }));
    }

    if matches!(extension.as_str(), "mp4" | "webm") {
        let output = std::process::Command::new("ffprobe")
            .arg("-v")
            .arg("error")
            .arg("-count_frames")
            .arg("-show_entries")
            .arg("stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:format=duration")
            .arg("-of")
            .arg("json")
            .arg(path)
            .output()
            .map_err(|error| format!("ffprobe is required to validate video artifacts: {}", error))?;
        if !output.status.success() {
            return Err(format!(
                "Artifact video '{}' failed ffprobe validation: {}",
                path,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let probe: Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("Invalid ffprobe response for '{}': {}", path, error))?;
        let decoded = std::process::Command::new("ffmpeg")
            .arg("-v")
            .arg("error")
            .arg("-i")
            .arg(path)
            .arg("-vf")
            .arg("scale=64:64:flags=area,format=gray")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("gray")
            .arg("-")
            .output()
            .map_err(|error| format!("ffmpeg is required to inspect video quality: {}", error))?;
        if !decoded.status.success() {
            return Err(format!(
                "Artifact video '{}' failed frame decoding: {}",
                path,
                String::from_utf8_lossy(&decoded.stderr).trim()
            ));
        }
        let signal = analyze_luma_signal(&decoded.stdout, 64 * 64)?;
        validate_visual_signal(path, &signal, true)?;
        return Ok(json!({
            "valid": true,
            "kind": "video",
            "path": path,
            "size_bytes": metadata.len(),
            "probe": probe,
            "quality": signal.to_json()
        }));
    }

    Err(format!("Unsupported artifact extension for '{}': {}", path, extension))
}

#[derive(Debug)]
struct VisualSignalStats {
    frame_count: usize,
    luma_stddev: f64,
    luma_p01: u8,
    luma_p99: u8,
    temporal_mean_absolute_difference: Option<f64>,
}

impl VisualSignalStats {
    fn dynamic_range(&self) -> u8 {
        self.luma_p99.saturating_sub(self.luma_p01)
    }

    fn to_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "luma_stddev": round_metric(self.luma_stddev),
            "luma_p01": self.luma_p01,
            "luma_p99": self.luma_p99,
            "dynamic_range": self.dynamic_range(),
            "temporal_mean_absolute_difference": self
                .temporal_mean_absolute_difference
                .map(round_metric),
            "non_blank": true,
            "has_motion": self
                .temporal_mean_absolute_difference
                .map(|value| value >= 0.25)
        })
    }
}

fn analyze_luma_signal(data: &[u8], frame_size: usize) -> Result<VisualSignalStats, String> {
    if frame_size == 0 || data.is_empty() || data.len() % frame_size != 0 {
        return Err("Decoded media returned an invalid luma frame buffer".to_string());
    }

    let mut histogram = [0_u64; 256];
    let mut sum = 0_f64;
    let mut sum_squares = 0_f64;
    for &value in data {
        histogram[value as usize] += 1;
        let value = f64::from(value);
        sum += value;
        sum_squares += value * value;
    }
    let sample_count = data.len() as f64;
    let mean = sum / sample_count;
    let variance = (sum_squares / sample_count - mean * mean).max(0.0);
    let frame_count = data.len() / frame_size;
    let temporal_mean_absolute_difference = if frame_count > 1 {
        let mut difference_sum = 0_u64;
        for frame in 1..frame_count {
            let previous = &data[(frame - 1) * frame_size..frame * frame_size];
            let current = &data[frame * frame_size..(frame + 1) * frame_size];
            difference_sum += previous
                .iter()
                .zip(current)
                .map(|(&left, &right)| u64::from(left.abs_diff(right)))
                .sum::<u64>();
        }
        Some(difference_sum as f64 / ((frame_count - 1) * frame_size) as f64)
    } else {
        None
    };

    Ok(VisualSignalStats {
        frame_count,
        luma_stddev: variance.sqrt(),
        luma_p01: histogram_percentile(&histogram, data.len(), 0.01),
        luma_p99: histogram_percentile(&histogram, data.len(), 0.99),
        temporal_mean_absolute_difference,
    })
}

fn histogram_percentile(histogram: &[u64; 256], sample_count: usize, quantile: f64) -> u8 {
    let target = ((sample_count.saturating_sub(1)) as f64 * quantile).round() as u64;
    let mut cumulative = 0_u64;
    for (value, count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative > target {
            return value as u8;
        }
    }
    u8::MAX
}

fn validate_visual_signal(
    path: &str,
    signal: &VisualSignalStats,
    require_motion: bool,
) -> Result<(), String> {
    if signal.luma_stddev < 2.0 && signal.dynamic_range() < 8 {
        return Err(format!(
            "Artifact '{}' is visually blank or near-uniform (luma stddev {:.3}, dynamic range {})",
            path,
            signal.luma_stddev,
            signal.dynamic_range()
        ));
    }
    if require_motion
        && signal.frame_count > 1
        && signal.temporal_mean_absolute_difference.unwrap_or_default() < 0.25
    {
        return Err(format!(
            "Artifact video '{}' is effectively static (temporal difference {:.3})",
            path,
            signal.temporal_mean_absolute_difference.unwrap_or_default()
        ));
    }
    Ok(())
}

fn round_metric(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod native_generation_spec_tests {
    use super::*;

    fn spec(intent: &str) -> NativeGenerationSpec {
        NativeGenerationSpec {
            intent: intent.to_string(),
            prompt: "dynamic test prompt".to_string(),
            negative_prompt: String::new(),
            image_path: None,
            model: None,
            quality: Some("balanced".to_string()),
            width: None,
            height: None,
            steps: None,
            cfg: None,
            min_cfg: None,
            noise_aug_strength: None,
            seed: None,
            strength: None,
            frames: None,
            fps: None,
            motion_bucket_id: None,
        }
    }

    #[test]
    fn accepts_dynamic_even_video_delivery_dimensions() {
        let mut value = spec("image_to_video");
        value.width = Some(1366);
        value.height = Some(780);
        assert!(validate_native_generation_spec(&value).is_ok());
    }

    #[test]
    fn rejects_unaligned_image_generation_dimensions() {
        let mut value = spec("text_to_image");
        value.width = Some(1366);
        let error = validate_native_generation_spec(&value).unwrap_err();
        assert!(error.contains("divisible by 8"));
    }

    #[test]
    fn accepts_dynamic_motion_bucket() {
        let mut value = spec("image_to_video");
        value.motion_bucket_id = Some(180);
        validate_native_generation_spec(&value).unwrap();
    }

    #[test]
    fn rejects_out_of_range_motion_bucket() {
        let mut value = spec("image_to_video");
        value.motion_bucket_id = Some(1024);
        let error = validate_native_generation_spec(&value).unwrap_err();
        assert!(error.contains("between 0 and 1023"));
    }

    #[test]
    fn discards_agent_model_when_user_did_not_select_one() {
        let mut value = spec("text_to_image");
        value.model = Some("hallucinated-model".to_string());
        let request = GenerationRequestContext {
            user_request: "dynamic user request".to_string(),
            intent: "text_to_image".to_string(),
            image_path: None,
            params: json!({
                "model": "",
                "_explicit_keys": ["width"]
            }),
        };

        apply_generation_request_constraints(&mut value, Some(&request)).unwrap();

        assert_eq!(value.model, None);
    }

    #[test]
    fn preserves_explicit_user_model_and_parameters() {
        let mut value = spec("text_to_image");
        value.model = Some("agent-choice".to_string());
        value.width = Some(512);
        let request = GenerationRequestContext {
            user_request: "dynamic user request".to_string(),
            intent: "image_to_video".to_string(),
            image_path: Some("input/user.jpg".to_string()),
            params: json!({
                "model": "models/video/user-model.gguf",
                "width": 1366,
                "height": 780,
                "frames": 25,
                "_explicit_keys": ["model", "width", "height", "frames"]
            }),
        };

        apply_generation_request_constraints(&mut value, Some(&request)).unwrap();

        assert_eq!(value.intent, "image_to_video");
        assert_eq!(value.image_path.as_deref(), Some("input/user.jpg"));
        assert_eq!(value.model.as_deref(), Some("models/video/user-model.gguf"));
        assert_eq!(value.width, Some(1366));
        assert_eq!(value.height, Some(780));
        assert_eq!(value.frames, Some(25));
    }

    #[test]
    fn rejects_invalid_explicit_ui_parameter_type() {
        let mut value = spec("text_to_image");
        let request = GenerationRequestContext {
            user_request: "dynamic user request".to_string(),
            intent: "text_to_image".to_string(),
            image_path: None,
            params: json!({
                "steps": "twenty",
                "_explicit_keys": ["steps"]
            }),
        };

        let error = apply_generation_request_constraints(&mut value, Some(&request)).unwrap_err();
        assert!(error.contains("steps"));
    }

    #[test]
    fn direct_api_parameters_are_explicit_without_marker_array() {
        let mut value = spec("text_to_image");
        value.width = Some(512);
        let request = GenerationRequestContext {
            user_request: "dynamic user request".to_string(),
            intent: "text_to_image".to_string(),
            image_path: None,
            params: json!({"width": 768, "steps": 30}),
        };

        apply_generation_request_constraints(&mut value, Some(&request)).unwrap();

        assert_eq!(value.width, Some(768));
        assert_eq!(value.steps, Some(30));
    }

    #[test]
    fn balanced_quality_preserves_dynamic_prompt_and_adds_quality_guards() {
        let mut value = spec("text_to_image");
        value.prompt = "a user-selected subject in a user-selected setting".to_string();

        apply_quality_prompt_profile(&mut value);

        assert!(value.prompt.starts_with("a user-selected subject"));
        assert!(value.prompt.contains("coherent composition"));
        assert!(value.negative_prompt.contains("watermark"));
    }

    #[test]
    fn image_to_image_quality_preserves_identity_and_composition() {
        let mut value = spec("image_to_image");
        value.prompt = "dynamic user transformation".to_string();

        apply_quality_prompt_profile(&mut value);

        assert!(value.prompt.starts_with("dynamic user transformation"));
        assert!(value.prompt.contains("preserved subject identity"));
        assert!(value.prompt.contains("preserved composition"));
    }

    #[test]
    fn detects_dynamic_source_preservation_requests() {
        assert!(requests_source_preservation(
            "保留人物身份和构图，转换成暖色电影海报"
        ));
        assert!(requests_source_preservation(
            "Keep the same person and preserve composition"
        ));
        assert!(!requests_source_preservation(
            "Replace the entire scene with an abstract landscape"
        ));
    }

    #[test]
    fn high_quality_video_keyframe_preserves_aspect_ratio() {
        assert_eq!(fit_dimensions_within(1024, 576, 768, 8), (768, 432));
        assert_eq!(fit_dimensions_within(512, 640, 768, 8), (512, 640));
    }

    #[test]
    fn fast_quality_does_not_add_positive_style_terms() {
        let mut value = spec("image_to_video");
        value.quality = Some("fast".to_string());
        value.prompt = "dynamic user motion".to_string();

        apply_quality_prompt_profile(&mut value);

        assert_eq!(value.prompt, "dynamic user motion");
        assert!(value.negative_prompt.contains("identity drift"));
    }

    #[test]
    fn normalizes_plain_compiler_output() {
        let prompt = normalize_compiled_prompt(
            "```\nPrompt: cinematic rainy city street at midnight with red neon reflections\n```",
        );

        assert_eq!(
            prompt,
            "cinematic rainy city street at midnight with red neon reflections"
        );
    }

    #[test]
    fn rejects_non_english_or_too_short_inference_prompts() {
        assert!(!is_model_ready_english_prompt("雨夜城市街道，霓虹灯倒影"));
        assert!(!is_model_ready_english_prompt("rainy street"));
        assert!(is_model_ready_english_prompt(
            "cinematic rainy city street at midnight with vivid red and cyan neon reflections across wet pavement"
        ));
        assert!(!is_model_ready_english_prompt(
            "A golden mechanical butterfly flies through a rainy neon street while the camera tracks the"
        ));
    }

    #[test]
    fn artifact_signal_rejects_blank_image() {
        let pixels = vec![255_u8; 64 * 64];
        let signal = analyze_luma_signal(&pixels, pixels.len()).unwrap();
        let error = validate_visual_signal("blank.png", &signal, false).unwrap_err();
        assert!(error.contains("visually blank"));
    }

    #[test]
    fn artifact_signal_accepts_non_uniform_image() {
        let pixels: Vec<u8> = (0..64 * 64).map(|index| (index % 256) as u8).collect();
        let signal = analyze_luma_signal(&pixels, pixels.len()).unwrap();
        validate_visual_signal("image.png", &signal, false).unwrap();
        assert!(signal.dynamic_range() > 200);
    }

    #[test]
    fn artifact_signal_rejects_static_video() {
        let frame: Vec<u8> = (0..64 * 64).map(|index| (index % 256) as u8).collect();
        let pixels = [frame.as_slice(), frame.as_slice()].concat();
        let signal = analyze_luma_signal(&pixels, frame.len()).unwrap();
        let error = validate_visual_signal("static.mp4", &signal, true).unwrap_err();
        assert!(error.contains("effectively static"));
    }

    #[test]
    fn artifact_signal_accepts_video_motion() {
        let first: Vec<u8> = (0..64 * 64).map(|index| (index % 224) as u8).collect();
        let second: Vec<u8> = first.iter().map(|value| value.saturating_add(24)).collect();
        let pixels = [first.as_slice(), second.as_slice()].concat();
        let signal = analyze_luma_signal(&pixels, first.len()).unwrap();
        validate_visual_signal("moving.mp4", &signal, true).unwrap();
        assert!(signal.temporal_mean_absolute_difference.unwrap() > 20.0);
    }
}
