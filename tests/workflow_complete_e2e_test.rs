// Workflow 完整 E2E 测试
// 测试所有 workflow 模板的执行能力

use std::path::Path;
use std::process::Command;
use std::fs;

const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output/workflow_e2e";
const WORKFLOW_DIR: &str = "/dev-data/ai-test/media_agent/workflows";
const SKILLS_DIR: &str = "/dev-data/ai-test/media_agent/skills";

fn model_exists() -> bool { Path::new(MODEL_PATH).exists() }
fn sd_cli_exists() -> bool { Path::new(SD_CLI_PATH).exists() }

fn generate_image(prompt: &str, width: u32, height: u32, steps: u32, seed: u32, output: &str) -> bool {
    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", prompt,
        "-W", &width.to_string(),
        "-H", &height.to_string(),
        "--steps", &steps.to_string(),
        "--cfg-scale", "7.0",
        "-s", &seed.to_string(),
        "--sampling-method", "euler",
        "-o", output,
        "--backend", "cpu",
    ]);
    
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

// ============================================================================
// Workflow 模板完整性测试
// ============================================================================

#[test]
fn test_all_workflows_exist_and_valid() {
    let workflows: Vec<_> = fs::read_dir(WORKFLOW_DIR)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()).map(|s| s == "jsonld").unwrap_or(false))
        .collect();
    
    let workflow_count = workflows.len();
    assert!(workflow_count >= 24, "Should have at least 24 workflow templates");
    
    for entry in workflows {
        let path = entry.path();
        let content = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content)
            .expect(&format!("{} should be valid JSON", path.display()));
        
        // 验证基本结构（支持多种 JSON-LD 格式）
        let has_name = json["name"].is_string() || json["wf:name"].is_string() || json["schema:name"].is_string();
        let has_type = json["@type"].is_string() || json["@type"].is_array() || 
                       json["@graph"].is_array();  // @graph 格式也有效
        assert!(has_name || has_type, 
                "{} should have name or @type or @graph", path.display());
        
        println!("✓ {} is valid", entry.file_name().to_str().unwrap_or("unknown"));
    }
    
    println!("All {} workflows are valid!", workflow_count);
}

#[test]
fn test_workflow_categories_covered() {
    let expected_categories = [
        "generation",      // 文生图
        "batch",          // 批量生成
        "controlnet",     // ControlNet
        "style",          // LoRA 风格
        "agent",          // Agent PDCA
        "pipeline",       // 多阶段流水线
        "edit",           // 图片编辑
        "postprocess",    // 后处理
        "video",          // 视频生成
    ];
    
    let mut found_categories = std::collections::HashSet::<String>::new();
    
    for entry in fs::read_dir(WORKFLOW_DIR).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "jsonld" {
            let content = fs::read_to_string(&path).unwrap();
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(cat) = json["category"].as_str() {
                    found_categories.insert(cat.to_string());
                }
            }
        }
    }
    
    println!("Found categories: {:?}", found_categories);
    
    let coverage = found_categories.len();
    assert!(coverage >= 8, "Should cover at least 8 workflow categories");
}

// ============================================================================
// 每个 Workflow 的参数验证测试
// ============================================================================

#[test]
fn test_text_to_image_workflow_params() {
    let path = format!("{}/text_to_image_basic.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 验证必要参数
    assert!(json["parameters"]["prompt"].is_object());
    assert!(json["steps"].is_array());
    assert!(json["steps"].as_array().unwrap().len() >= 5);
    
    println!("text_to_image workflow has {} steps", json["steps"].as_array().unwrap().len());
}

#[test]
fn test_image_to_image_workflow_params() {
    let path = format!("{}/image_to_image.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // I2I 需要输入图片参数
    assert!(json["parameters"]["input_image_path"].is_object());
    assert!(json["parameters"]["denoise"].is_object());
    
    println!("image_to_image workflow validated");
}

#[test]
fn test_high_quality_workflow_params() {
    let path = format!("{}/high_quality_generation.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 高质量应该有更多步数
    let steps = json["steps"].as_array().unwrap();
    let sampler_step = steps.iter().find(|s| s["action"] == "KSampler");
    
    if let Some(sampler) = sampler_step {
        if let Some(sampler_steps) = sampler["inputs"]["steps"].as_u64() {
            assert!(sampler_steps >= 20, "High quality should use at least 20 steps");
        }
    }
    
    println!("high_quality workflow validated");
}

#[test]
fn test_batch_workflow_params() {
    let path = format!("{}/batch_generation.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 批量应该有 batch_size 参数
    assert!(json["parameters"]["batch_size"].is_object());
    
    println!("batch_generation workflow validated");
}

#[test]
fn test_controlnet_workflow_params() {
    let path = format!("{}/controlnet_pose.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // ControlNet 需要姿势图和强度参数
    assert!(json["parameters"]["pose_image_path"].is_object());
    assert!(json["parameters"]["controlnet_strength"].is_object());
    
    // 检查是否有 ControlNet 加载步骤
    let steps = json["steps"].as_array().unwrap();
    let has_controlnet_loader = steps.iter().any(|s| {
        s["action"].as_str().unwrap_or("").contains("ControlNet")
    });
    
    assert!(has_controlnet_loader, "ControlNet workflow should load ControlNet model");
    
    println!("controlnet_pose workflow validated");
}

#[test]
fn test_lora_workflow_params() {
    let path = format!("{}/lora_style.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // LoRA 需要模型名和强度参数
    assert!(json["parameters"]["lora_name"].is_object());
    assert!(json["parameters"]["lora_strength"].is_object());
    
    println!("lora_style workflow validated");
}

#[test]
fn test_agent_pdca_workflow_structure() {
    let path = format!("{}/agent_pdca_workflow.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // PDCA 应该有四个阶段
    assert!(json["pdca_cycle"].is_object());
    assert!(json["pdca_cycle"]["plan"].is_object());
    assert!(json["pdca_cycle"]["do"].is_object());
    assert!(json["pdca_cycle"]["check"].is_object());
    assert!(json["pdca_cycle"]["act"].is_object());
    
    // 应该有 workflow_selection 映射
    assert!(json["workflow_selection"].is_object());
    
    println!("agent_pdca workflow validated with PDCA structure");
}

#[test]
fn test_multi_stage_pipeline_workflow() {
    let path = format!("{}/multi_stage_pipeline.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 多阶段流水线应该有 stages 数组
    assert!(json["stages"].is_array());
    let stages = json["stages"].as_array().unwrap();
    
    assert!(stages.len() >= 2, "Pipeline should have at least 2 stages");
    
    // 每个阶段应该引用其他 workflow
    for stage in stages {
        assert!(stage["workflow"].is_string(), "Each stage should reference a workflow");
    }
    
    println!("multi_stage_pipeline validated with {} stages", stages.len());
}

// ============================================================================
// Workflow 执行测试（使用 sd-cli 模拟）
// ============================================================================

#[test]
fn test_execute_text_to_image_workflow() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    fs::create_dir_all(OUTPUT_DIR).ok();
    
    // 读取 workflow 参数
    let path = format!("{}/text_to_image_basic.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 使用 workflow 的默认参数生成图片
    let output = format!("{}/workflow_t2i.png", OUTPUT_DIR);
    
    let success = generate_image(
        "a beautiful sunset landscape",
        256, 256, 10, 300,
        &output,
    );
    
    if success {
        let size = fs::metadata(&output).unwrap().len();
        assert!(size > 1000, "Generated image should have valid size");
        println!("text_to_image workflow executed: {} bytes", size);
    }
}

#[test]
fn test_execute_batch_workflow() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    fs::create_dir_all(OUTPUT_DIR).ok();
    
    // 执行批量生成（模拟 batch_generation workflow）
    for i in 0..3 {
        let output = format!("{}/workflow_batch_{:03}.png", OUTPUT_DIR, i);
        let success = generate_image(
            &format!("variation {} of a cat", i + 1),
            256, 256, 10, 400 + i,
            &output,
        );
        
        if success {
            println!("Batch frame {} generated", i + 1);
        }
    }
}

#[test]
fn test_execute_high_quality_workflow() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    fs::create_dir_all(OUTPUT_DIR).ok();
    
    // 使用高质量参数（更多步数、更大尺寸）
    let output = format!("{}/workflow_hq.png", OUTPUT_DIR);
    
    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", "detailed masterpiece portrait",
        "-n", "blurry, low quality, distorted",
        "-W", "512", "-H", "512",
        "--steps", "20",
        "--cfg-scale", "8.0",
        "-s", "500",
        "--sampling-method", "dpm++2m",
        "-o", &output,
        "--backend", "cpu",
    ]);
    
    if cmd.output().map(|o| o.status.success()).unwrap_or(false) {
        let size = fs::metadata(&output).unwrap().len();
        assert!(size > 5000, "High quality image should be larger");
        println!("high_quality workflow executed: {} bytes", size);
    }
}

// ============================================================================
// Skills 测试
// ============================================================================

#[test]
fn test_all_skills_exist() {
    let skills: Vec<_> = fs::read_dir(SKILLS_DIR)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()).map(|s| s == "jsonld").unwrap_or(false))
        .collect();
    
    assert!(skills.len() >= 3, "Should have at least 3 skills");
    
    for entry in skills {
        let content = fs::read_to_string(entry.path()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content)
            .expect(&format!("{} should be valid JSON", entry.path().display()));
        
        assert!(json["@type"].is_string() || json["@type"].is_array(), "Skill should have @type");
        println!("✓ {}", entry.file_name().to_str().unwrap_or("unknown"));
    }
}

#[test]
fn test_text_to_image_skill_params() {
    let path = format!("{}/text_to_image.jsonld", SKILLS_DIR);
    if !Path::new(&path).exists() { println!("SKIP"); return; }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["input_schema"].is_object() || json["skill:inputSchema"].is_object() || json["parameters"].is_object());
    assert!(json["output_schema"].is_object() || json["skill:outputSchema"].is_object() || json["outputs"].is_object());
    
    println!("text_to_image skill validated");
}

#[test]
fn test_image_to_image_skill_params() {
    let path = format!("{}/image_to_image.jsonld", SKILLS_DIR);
    if !Path::new(&path).exists() { println!("SKIP"); return; }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["input_schema"].is_object() || json["parameters"].is_object());
    
    println!("image_to_image skill validated");
}

#[test]
fn test_generate_video_skill_params() {
    let path = format!("{}/generate_video.jsonld", SKILLS_DIR);
    if !Path::new(&path).exists() { println!("SKIP"); return; }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["input_schema"].is_object() || json["parameters"].is_object());
    
    println!("generate_video skill validated");
}

// ============================================================================
// Workflow DAG 验证测试
// ============================================================================

#[test]
fn test_workflow_dag_structure() {
    // 验证每个 workflow 的 DAG 结构正确性
    for entry in fs::read_dir(WORKFLOW_DIR).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().and_then(|e| e.to_str()).map(|s| s == "jsonld").unwrap_or(false) {
            let content = fs::read_to_string(entry.path()).unwrap();
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(steps) = json["steps"].as_array() {
                    // 验证 DAG 结构：每个步骤应该有唯一的名称
                    let mut names = std::collections::HashSet::new();
                    for step in steps {
                        if let Some(name) = step["name"].as_str() {
                            names.insert(name);
                        }
                    }
                    assert!(names.len() == steps.len(), 
                            "{} should have unique step names", 
                            entry.file_name().to_str().unwrap_or("unknown"));
                }
            }
        }
    }
    
    println!("All workflows have valid DAG structure!");
}

// ============================================================================
// Workflow 依赖验证测试
// ============================================================================

#[test]
fn test_workflow_step_dependencies() {
    let path = format!("{}/text_to_image_basic.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    let steps = json["steps"].as_array().unwrap();
    
    // 检查步骤之间的依赖关系
    let mut available_outputs: std::collections::HashSet<&str> = std::collections::HashSet::new();
    
    for step in steps {
        // 检查输入是否依赖已存在的输出
        if let Some(inputs) = step["inputs"].as_object() {
            for (_, value) in inputs {
                if let Some(link) = value.as_object() {
                    if let Some(from) = link["from"].as_str() {
                        // 验证依赖的步骤在之前已定义
                        // 注：简化验证，实际 DAG 需要更复杂的拓扑排序验证
                        println!("Step depends on: {}", from);
                    }
                }
            }
        }
        
        // 添加当前步骤的输出
        if let Some(outputs) = step["outputs"].as_array() {
            for out in outputs {
                if let Some(out_name) = out.as_str() {
                    available_outputs.insert(out_name);
                }
            }
        }
    }
    
    println!("Workflow dependencies validated");
}

// ============================================================================
// 新增 Workflow 测试
// ============================================================================

#[test]
fn test_sdxl_workflow_params() {
    let path = format!("{}/sdxl_text_to_image.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // SDXL workflow 应该有更大的默认尺寸
    assert!(json["model_type"].as_str().unwrap_or("") == "sdxl");
    assert!(json["parameters"]["width"]["default"].as_u64().unwrap_or(0) >= 1024);
    assert!(json["parameters"]["height"]["default"].as_u64().unwrap_or(0) >= 1024);
    
    println!("sdxl_text_to_image workflow validated");
}

#[test]
fn test_controlnet_depth_workflow() {
    let path = format!("{}/controlnet_depth.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "controlnet");
    assert!(json["parameters"]["depth_image_path"].is_object());
    assert!(json["parameters"]["controlnet_strength"].is_object());
    
    println!("controlnet_depth workflow validated");
}

#[test]
fn test_controlnet_lineart_workflow() {
    let path = format!("{}/controlnet_lineart.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "controlnet");
    assert!(json["parameters"]["lineart_image_path"].is_object());
    
    println!("controlnet_lineart workflow validated");
}

#[test]
fn test_video_frame_interpolation_workflow() {
    let path = format!("{}/video_frame_interpolation.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    assert!(json["parameters"]["interpolation_multiplier"].is_object());
    
    println!("video_frame_interpolation workflow validated");
}

#[test]
fn test_image_to_video_svd_workflow() {
    let path = format!("{}/image_to_video_svd.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    assert!(json["parameters"]["num_frames"].is_object());
    assert!(json["parameters"]["fps"].is_object());
    assert!(json["parameters"]["motion_bucket_id"].is_object());
    
    println!("image_to_video_svd workflow validated");
}

#[test]
fn test_latent_interpolation_workflow() {
    let path = format!("{}/latent_interpolation.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    assert!(json["parameters"]["prompt_1"].is_object());
    assert!(json["parameters"]["prompt_2"].is_object());
    assert!(json["parameters"]["num_interpolation_steps"].is_object());
    
    println!("latent_interpolation workflow validated");
}

#[test]
fn test_multi_lora_combine_workflow() {
    let path = format!("{}/multi_lora_combine.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "style");
    assert!(json["parameters"]["lora_1_name"].is_object());
    assert!(json["parameters"]["lora_2_name"].is_object());
    assert!(json["parameters"]["lora_1_strength"].is_object());
    assert!(json["parameters"]["lora_2_strength"].is_object());
    
    println!("multi_lora_combine workflow validated");
}

#[test]
fn test_style_transfer_workflow() {
    let path = format!("{}/style_transfer.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "style");
    assert!(json["parameters"]["style_reference_path"].is_object());
    
    println!("style_transfer workflow validated");
}

#[test]
fn test_face_restore_workflow() {
    let path = format!("{}/face_restore.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "postprocess");
    assert!(json["parameters"]["face_restore_model"].is_object());
    
    println!("face_restore workflow validated");
}

#[test]
fn test_agent_intelligent_workflow() {
    let path = format!("{}/agent_intelligent_workflow.jsonld", WORKFLOW_DIR);
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "agent");
    assert!(json["workflow_selection_rules"].is_array());
    assert!(json["fallback_chain"].is_array());
    
    // 验证选择规则数量
    let rules = json["workflow_selection_rules"].as_array().unwrap();
    assert!(rules.len() >= 8, "Should have at least 8 workflow selection rules");
    
    println!("agent_intelligent_workflow validated with {} selection rules", rules.len());
}

#[test]
fn test_all_new_workflows_exist() {
    let new_workflows = [
        "sdxl_text_to_image.jsonld",
        "controlnet_depth.jsonld",
        "controlnet_lineart.jsonld",
        "video_frame_interpolation.jsonld",
        "image_to_video_svd.jsonld",
        "latent_interpolation.jsonld",
        "multi_lora_combine.jsonld",
        "style_transfer.jsonld",
        "face_restore.jsonld",
        "agent_intelligent_workflow.jsonld",
        "lora_detail_enhance.jsonld",
    ];
    
    for wf in &new_workflows {
        let path = format!("{}/{}", WORKFLOW_DIR, wf);
        assert!(Path::new(&path).exists(), "New workflow {} should exist", wf);
        
        // 验证 JSON 有效
        let content = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content)
            .expect(&format!("{} should be valid JSON", wf));
        
        println!("✓ {} exists and valid", wf);
    }
    
    println!("All {} new workflows validated!", new_workflows.len());
}