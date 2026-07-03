// ComfyUI 工具集 - 注册到 gliding_horse ToolExecutor
// 8 个核心工具：submit_workflow, build_t2i, build_i2i, backend_sample, list_nodes, interrupt, free_memory, health_check

use std::sync::Arc;
use serde_json::{json, Value};
use crate::agent::context::AgentContext;
use crate::types::Workflow;
use crate::workflow::builder::WorkflowBuilder;

/// 注册 ComfyUI 工具到 ToolExecutor
pub fn register_comfyui_tools(
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

                    let strength = input.get("strength")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.75);

                    let steps = input.get("steps")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(20) as usize;

                    Ok(json!({
                        "workflow": {"nodes": {}, "links": []},
                        "message": "image_to_image workflow builder placeholder",
                        "params": { "image_path": image_path, "prompt": prompt, "strength": strength, "steps": steps }
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
