// 智能工作流工具集
// 提供动态构建工作流的工具，让 LLM 自主决定节点连接

use std::sync::Arc;
use serde_json::{json, Value};
use crate::agent::context::AgentContext;
use crate::node::node_registry::{NodeRegistry, DataKind};
use crate::types::{Workflow, WorkflowNode, InputValue};
use std::collections::HashMap as StdHashMap;

/// 注册智能工作流构建工具
pub fn register_smart_workflow_tools(
    executor: &mut glidinghorse::tools::tool_executor::ToolExecutor,
    ctx: Arc<AgentContext>,
) {
    let registry = Arc::new(NodeRegistry::new());
    
    // === 1. list_available_nodes ===
    // 列出所有可用节点类型及其输入输出
    executor.register(
        "list_available_nodes",
        "List all available node types with their inputs and outputs. Use this to understand what nodes you can use.",
        json!({
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category (optional)", "default": "all"}
            }
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let category = input.get("category")
                        .and_then(|v| v.as_str())
                        .unwrap_or("all");
                    
                    let nodes = if category == "all" {
                        registry.list_all()
                    } else {
                        registry.list_by_category(category)
                    };
                    
                    let node_list = nodes.iter().map(|spec| {
                        json!({
                            "class_type": spec.class_type,
                            "display_name": spec.display_name,
                            "category": spec.category,
                            "inputs": spec.inputs.iter().map(|p| {
                                json!({
                                    "name": p.name,
                                    "type": p.data_kind.name(),
                                    "required": p.required
                                })
                            }).collect::<Vec<_>>(),
                            "outputs": spec.outputs.iter().map(|p| {
                                json!({
                                    "name": p.name,
                                    "type": p.data_kind.name()
                                })
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>();
                    
                    Ok(json!({
                        "nodes": node_list,
                        "total_count": nodes.len()
                    }))
                })
            }
        }),
        &["DA"],
    );
    
    // === 2. create_node ===
    // 创建一个新节点实例
    executor.register(
        "create_node",
        "Create a new node instance with specified class_type and parameters. Returns node_id.",
        json!({
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Unique node identifier (e.g., '1', '2', 'n1')"},
                "class_type": {"type": "string", "description": "Node type to create (e.g., 'KSampler', 'CheckpointLoaderSimple')"},
                "params": {"type": "object", "description": "Node parameters (key-value pairs)"},
                "workflow_id": {"type": "string", "description": "Workflow ID to add node to", "default": "current"}
            },
            "required": ["node_id", "class_type"]
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let node_id = input.get("node_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing node_id".to_string())?;
                    
                    let class_type = input.get("class_type")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing class_type".to_string())?;
                    
                    // 验证节点类型存在
                    let spec = registry.get_spec(class_type)
                        .ok_or_else(|| format!("Unknown node type: {}", class_type))?;
                    
                    // 构建输入参数
                    let params = input.get("params")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    
                    let mut inputs = StdHashMap::new();
                    for (key, value) in params {
                        inputs.insert(key, InputValue::Direct(crate::types::Value::from_json(value)));
                    }
                    
                    let inputs_count = inputs.len();
                    
                    // 创建节点
                    let node = WorkflowNode {
                        class_type: class_type.to_string(),
                        inputs,
                        pos: None,
                        size: None,
                        is_changed: None,
                    };
                    
                    Ok(json!({
                        "node_id": node_id,
                        "class_type": class_type,
                        "display_name": spec.display_name,
                        "inputs_count": inputs_count,
                        "outputs_count": spec.outputs.len(),
                        "node_created": true
                    }))
                })
            }
        }),
        &["DA"],
    );
    
    // === 3. connect_nodes ===
    // 连接两个节点的端口
    executor.register(
        "connect_nodes",
        "Connect source node output to target node input. Validates type compatibility.",
        json!({
            "type": "object",
            "properties": {
                "source_node": {"type": "string", "description": "Source node ID"},
                "source_output": {"type": "string", "description": "Source output port name (e.g., 'MODEL', 'LATENT')"},
                "target_node": {"type": "string", "description": "Target node ID"},
                "target_input": {"type": "string", "description": "Target input port name"}
            },
            "required": ["source_node", "source_output", "target_node", "target_input"]
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let source_node = input.get("source_node")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing source_node".to_string())?;
                    
                    let source_output = input.get("source_output")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing source_output".to_string())?;
                    
                    let target_node = input.get("target_node")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing target_node".to_string())?;
                    
                    let target_input = input.get("target_input")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing target_input".to_string())?;
                    
                    // 验证连接
                    let validation = registry.validate_connection(
                        source_node, source_output,
                        target_node, target_input
                    );
                    
                    match validation {
                        Ok(true) => {
                            // 返回连接信息（实际的连接由 LLM 在最终工作流 JSON 中构建）
                            Ok(json!({
                                "connection": {
                                    "source": [source_node, source_output],
                                    "target": [target_node, target_input]
                                },
                                "link_format": [source_node, source_output],
                                "valid": true,
                                "message": "Connection is valid. Use this link format in workflow JSON."
                            }))
                        }
                        Err(e) => {
                            Ok(json!({
                                "valid": false,
                                "error": e,
                                "message": "Connection is invalid. Find alternative node or port."
                            }))
                        }
                        _ => Ok(json!({"valid": false, "error": "Unknown validation result"}))
                    }
                })
            }
        }),
        &["DA"],
    );
    
    // === 4. find_compatible_sources ===
    // 找到可以连接到指定输入的节点输出
    executor.register(
        "find_compatible_sources",
        "Find all node types that can provide input to a specified port type.",
        json!({
            "type": "object",
            "properties": {
                "input_type": {"type": "string", "description": "Target input type (MODEL, CLIP, VAE, CONDITIONING, LATENT, IMAGE, etc.)"}
            },
            "required": ["input_type"]
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let input_type_str = input.get("input_type")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing input_type".to_string())?;
                    
                    // 解析数据类型
                    let data_kind = match input_type_str {
                        "MODEL" => DataKind::MODEL,
                        "CLIP" => DataKind::CLIP,
                        "VAE" => DataKind::VAE,
                        "CONDITIONING" => DataKind::CONDITIONING,
                        "LATENT" => DataKind::LATENT,
                        "IMAGE" => DataKind::IMAGE,
                        "CONTROL_NET" => DataKind::CONTROL_NET,
                        "FRAMES" => DataKind::FRAMES,
                        "VIDEO" => DataKind::VIDEO,
                        "ANY" => DataKind::ANY,
                        _ => DataKind::ANY,
                    };
                    
                    let sources = registry.find_compatible_outputs(data_kind);
                    
                    let source_list = sources.iter().map(|(class_type, output)| {
                        let spec = registry.get_spec(class_type);
                        json!({
                            "node_type": class_type,
                            "output_port": output,
                            "display_name": spec.map(|s| s.display_name.clone()).unwrap_or_default()
                        })
                    }).collect::<Vec<_>>();
                    
                    Ok(json!({
                        "input_type": input_type_str,
                        "compatible_sources": source_list,
                        "count": sources.len()
                    }))
                })
            }
        }),
        &["DA"],
    );
    
    // === 5. validate_workflow ===
    // 验证整个工作流是否有效
    executor.register(
        "validate_workflow",
        "Validate a complete workflow JSON. Check all connections and required inputs.",
        json!({
            "type": "object",
            "properties": {
                "workflow": {"type": "object", "description": "Workflow JSON to validate"}
            },
            "required": ["workflow"]
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let workflow_json = input.get("workflow")
                        .ok_or_else(|| "Missing workflow".to_string())?;
                    
                    let nodes = workflow_json.get("nodes")
                        .and_then(|v| v.as_object())
                        .ok_or_else(|| "Workflow has no nodes object".to_string())?;
                    
                    let mut errors = Vec::new();
                    let mut warnings = Vec::new();
                    let mut node_count = 0;
                    let mut link_count = 0;
                    
                    for (node_id, node_data) in nodes {
                        node_count += 1;
                        let class_type = node_data.get("class_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        
                        // 检查节点类型是否已知
                        if registry.get_spec(class_type).is_none() {
                            warnings.push(format!("Node '{}' uses unknown class_type '{}'", node_id, class_type));
                            continue;
                        }
                        
                        let spec = registry.get_spec(class_type).unwrap();
                        let empty_map = serde_json::Map::new();
                        let inputs = node_data.get("inputs")
                            .and_then(|v| v.as_object())
                            .unwrap_or(&empty_map);
                        
                        // 检查必需输入是否都已提供
                        for input_port in &spec.inputs {
                            if input_port.required {
                                if !inputs.contains_key(&input_port.name) {
                                    errors.push(format!(
                                        "Node '{}' ({}) missing required input '{}'",
                                        node_id, class_type, input_port.name
                                    ));
                                }
                            }
                        }
                        
                        // 检查连接类型
                        for (input_name, input_value) in inputs {
                            if let Some(link) = input_value.as_array() {
                                if link.len() == 2 {
                                    link_count += 1;
                                    let source_node_id = link[0].as_str().unwrap_or("");
                                    let source_output = link[1].as_str().unwrap_or("");
                                    
                                    // 找到源节点类型
                                    if let Some(source_node) = nodes.get(source_node_id) {
                                        let source_class_type = source_node.get("class_type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        
                                        if let Err(e) = registry.validate_connection(
                                            source_class_type, source_output,
                                            class_type, input_name
                                        ) {
                                            errors.push(format!(
                                                "Invalid connection: {} -> {}: {}",
                                                source_node_id, node_id, e
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    Ok(json!({
                        "valid": errors.is_empty(),
                        "node_count": node_count,
                        "link_count": link_count,
                        "errors": errors,
                        "warnings": warnings,
                        "message": if errors.is_empty() {
                            "Workflow is valid and ready for execution".to_string()
                        } else {
                            format!("Workflow has {} errors that must be fixed", errors.len())
                        }
                    }))
                })
            }
        }),
        &["DA"],
    );
    
    // === 6. suggest_workflow ===
    // 根意图推荐工作流结构
    executor.register(
        "suggest_workflow",
        "Suggest a workflow structure based on user intent. Returns recommended nodes and connections.",
        json!({
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "User intent (text_to_image, image_to_image, video, upscale, inpaint)"},
                "model_type": {"type": "string", "description": "Model type (sd15, sdxl, sd3)", "default": "sd15"},
                "use_lora": {"type": "boolean", "description": "Include LoRA nodes", "default": false},
                "use_controlnet": {"type": "boolean", "description": "Include ControlNet nodes", "default": false}
            },
            "required": ["intent"]
        }),
        Arc::new({
            let registry = registry.clone();
            move |input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    let intent = input.get("intent")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "Missing intent".to_string())?;
                    
                    let model_type = input.get("model_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("sd15");
                    
                    let use_lora = input.get("use_lora")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    let use_controlnet = input.get("use_controlnet")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    // 根据意图生成推荐结构
                    let suggested = match intent {
                        "text_to_image" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple", "description": "Load main model"},
                                {"id": "2", "type": "CLIPTextEncode", "description": "Encode positive prompt"},
                                {"id": "3", "type": "CLIPTextEncode", "description": "Encode negative prompt"},
                                {"id": "4", "type": "EmptyLatentImage", "description": "Create empty latent"},
                                {"id": "5", "type": "KSampler", "description": "Sampling"},
                                {"id": "6", "type": "VAEDecode", "description": "Decode latent to image"},
                                {"id": "7", "type": "SaveImage", "description": "Save output"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["5", "model"]},
                                {"from": ["1", "CLIP"], "to": ["2", "clip"]},
                                {"from": ["1", "CLIP"], "to": ["3", "clip"]},
                                {"from": ["1", "VAE"], "to": ["6", "vae"]},
                                {"from": ["2", "CONDITIONING"], "to": ["5", "positive"]},
                                {"from": ["3", "CONDITIONING"], "to": ["5", "negative"]},
                                {"from": ["4", "LATENT"], "to": ["5", "latent_image"]},
                                {"from": ["5", "LATENT"], "to": ["6", "samples"]},
                                {"from": ["6", "IMAGE"], "to": ["7", "images"]}
                            ]
                        }),
                        "image_to_image" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple"},
                                {"id": "2", "type": "CLIPTextEncode", "description": "Positive prompt"},
                                {"id": "3", "type": "CLIPTextEncode", "description": "Negative prompt"},
                                {"id": "4", "type": "LoadImage", "description": "Load input image (set image path in params)"},
                                {"id": "5", "type": "VAEEncode", "description": "Encode input image to latent"},
                                {"id": "6", "type": "KSampler", "description": "Sampling with denoise < 1.0"},
                                {"id": "7", "type": "VAEDecode"},
                                {"id": "8", "type": "SaveImage"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["6", "model"]},
                                {"from": ["1", "CLIP"], "to": ["2", "clip"]},
                                {"from": ["1", "CLIP"], "to": ["3", "clip"]},
                                {"from": ["1", "VAE"], "to": ["5", "vae"]},
                                {"from": ["1", "VAE"], "to": ["7", "vae"]},
                                {"from": ["4", "IMAGE"], "to": ["5", "pixels"]},
                                {"from": ["2", "CONDITIONING"], "to": ["6", "positive"]},
                                {"from": ["3", "CONDITIONING"], "to": ["6", "negative"]},
                                {"from": ["5", "LATENT"], "to": ["6", "latent_image"]},
                                {"from": ["6", "LATENT"], "to": ["7", "samples"]},
                                {"from": ["7", "IMAGE"], "to": ["8", "images"]}
                            ],
                            "note": "Set denoise to 0.5-0.8 for style transfer. Set LoadImage image param to the uploaded image path."
                        }),
                        "image_to_video" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple", "description": "Load SVD or SD checkpoint"},
                                {"id": "2", "type": "LoadImage", "description": "Load input image (set image path in params, e.g. 'bk_0015.jpg')"},
                                {"id": "3", "type": "SVDImageToVideo", "description": "Image to video using SVD"},
                                {"id": "4", "type": "VideoCombine", "description": "Combine frames to MP4"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["3", "model"]},
                                {"from": ["1", "VAE"], "to": ["3", "vae"]},
                                {"from": ["2", "IMAGE"], "to": ["3", "image"]},
                                {"from": ["3", "FRAMES"], "to": ["4", "frames"]}
                            ],
                            "note": "For image_to_video: 1) LoadImage must use the uploaded image path. 2) SVDImageToVideo needs svd_xt checkpoint. 3) Set motion_bucket_id=127, motion_scale=1024, frames=25 for 5s video at 5fps. 4) VideoCombine fps should match output fps."
                        }),
                        "video" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple", "description": "Load SD checkpoint for AnimateDiff"},
                                {"id": "2", "type": "CLIPTextEncode", "description": "Positive prompt"},
                                {"id": "3", "type": "CLIPTextEncode", "description": "Negative prompt"},
                                {"id": "4", "type": "EmptyLatentImage", "description": "Create latent with batch_size=16 (16 frames)"},
                                {"id": "5", "type": "AnimateDiffSampler", "description": "AnimateDiff sampling for animation"},
                                {"id": "6", "type": "VAEDecode", "description": "Decode to frames"},
                                {"id": "7", "type": "VideoCombine", "description": "Combine frames to MP4"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["5", "model"]},
                                {"from": ["1", "CLIP"], "to": ["2", "clip"]},
                                {"from": ["1", "CLIP"], "to": ["3", "clip"]},
                                {"from": ["1", "VAE"], "to": ["6", "vae"]},
                                {"from": ["2", "CONDITIONING"], "to": ["5", "positive"]},
                                {"from": ["3", "CONDITIONING"], "to": ["5", "negative"]},
                                {"from": ["4", "LATENT"], "to": ["5", "latent_image"]},
                                {"from": ["5", "LATENT"], "to": ["6", "samples"]},
                                {"from": ["6", "IMAGE"], "to": ["7", "frames"]}
                            ],
                            "note": "For text_to_video: 1) Set EmptyLatentImage batch_size=16 for 16 frames. 2) AnimateDiffSampler generates animation. 3) Use motion-friendly prompts like 'dancing, moving, walking'."
                        }),
                        "upscale" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple"},
                                {"id": "2", "type": "CLIPTextEncode"},
                                {"id": "3", "type": "CLIPTextEncode"},
                                {"id": "4", "type": "EmptyLatentImage"},
                                {"id": "5", "type": "KSampler"},
                                {"id": "6", "type": "VAEDecode"},
                                {"id": "7", "type": "ImageUpscaleWithModel", "description": "AI upscale"},
                                {"id": "8", "type": "SaveImage"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["5", "model"]},
                                {"from": ["1", "MODEL"], "to": ["7", "upscale_model"]},
                                {"from": ["6", "IMAGE"], "to": ["7", "image"]},
                                {"from": ["7", "IMAGE"], "to": ["8", "images"]}
                            ]
                        }),
                        "inpaint" => json!({
                            "nodes": [
                                {"id": "1", "type": "CheckpointLoaderSimple"},
                                {"id": "2", "type": "CLIPTextEncode", "description": "Positive prompt for inpaint area"},
                                {"id": "3", "type": "CLIPTextEncode", "description": "Negative prompt"},
                                {"id": "4", "type": "LoadImage", "description": "Load input image"},
                                {"id": "5", "type": "VAEEncodeForInpaint", "description": "Encode image with mask"},
                                {"id": "6", "type": "KSampler", "description": "Sampling with denoise=1.0"},
                                {"id": "7", "type": "VAEDecode"},
                                {"id": "8", "type": "SaveImage"}
                            ],
                            "connections": [
                                {"from": ["1", "MODEL"], "to": ["6", "model"]},
                                {"from": ["1", "CLIP"], "to": ["2", "clip"]},
                                {"from": ["1", "CLIP"], "to": ["3", "clip"]},
                                {"from": ["1", "VAE"], "to": ["5", "vae"]},
                                {"from": ["1", "VAE"], "to": ["7", "vae"]},
                                {"from": ["4", "IMAGE"], "to": ["5", "pixels"]},
                                {"from": ["4", "MASK"], "to": ["5", "mask"]},
                                {"from": ["2", "CONDITIONING"], "to": ["6", "positive"]},
                                {"from": ["3", "CONDITIONING"], "to": ["6", "negative"]},
                                {"from": ["5", "LATENT"], "to": ["6", "latent_image"]},
                                {"from": ["6", "LATENT"], "to": ["7", "samples"]},
                                {"from": ["7", "IMAGE"], "to": ["8", "images"]}
                            ],
                            "note": "LoadImage outputs both IMAGE and MASK. MASK is the alpha channel used for inpaint region."
                        }),
                        _ => json!({
                            "nodes": [],
                            "connections": [],
                            "message": "Unknown intent. Use text_to_image, image_to_image, image_to_video, video, inpaint, or upscale."
                        })
                    };
                    
                    // 如果启用 LoRA，添加 LoRA 节点
                    let final_suggestion = if use_lora && intent != "upscale" {
                        let mut nodes = suggested["nodes"].as_array().cloned().unwrap_or_default();
                        let mut connections = suggested["connections"].as_array().cloned().unwrap_or_default();
                        
                        // 在模型加载后添加 LoRA 节点
                        nodes.push(json!({"id": "lora", "type": "LoraLoader", "description": "Apply LoRA style"}));
                        
                        // 修改连接：模型先经过 LoRA
                        connections.retain(|c| {
                            let from = c.get("from").and_then(|f| f.as_array());
                            from.map(|f| f[0] != "1" || f[1] != "MODEL").unwrap_or(true)
                        });
                        connections.push(json!({"from": ["1", "MODEL"], "to": ["lora", "model"]}));
                        connections.push(json!({"from": ["1", "CLIP"], "to": ["lora", "clip"]}));
                        connections.push(json!({"from": ["lora", "MODEL"], "to": ["5", "model"]}));
                        
                        json!({"nodes": nodes, "connections": connections})
                    } else {
                        suggested
                    };
                    
                    Ok(json!({
                        "intent": intent,
                        "model_type": model_type,
                        "suggestion": final_suggestion,
                        "use_lora": use_lora,
                        "use_controlnet": use_controlnet
                    }))
                })
            }
        }),
        &["DA"],
    );
    
    // === 7. get_node_schema ===
    // 获取完整的节点类型 Schema（供 LLM 参考）
    executor.register(
        "get_node_schema",
        "Get the complete node type schema for building workflows.",
        json!({
            "type": "object",
            "properties": {}
        }),
        Arc::new({
            let registry = registry.clone();
            move |_input: Value| {
                let registry = registry.clone();
                Box::pin(async move {
                    Ok(registry.to_json_schema())
                })
            }
        }),
        &["DA"],
    );
}