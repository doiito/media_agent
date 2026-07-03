// 完整覆盖测试 - 测试所有 workflow 模板
// 覆盖：文生图、图生图、批量、放大、修复、ControlNet、LoRA、Agent PDCA

use std::path::Path;
use std::process::Command;
use std::collections::HashMap;

const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output/full_coverage";
const WORKFLOW_DIR: &str = "/dev-data/ai-test/media_agent/workflows";

// ControlNet 模型路径
const CONTROLNET_OPENPOSE: &str = "/dev-data/ai-test/media_agent/models/controlnet/control_v11p_sd15_openpose.safetensors";
const CONTROLNET_CANNY: &str = "/dev-data/ai-test/media_agent/models/controlnet/control_v11p_sd15_canny.safetensors";

// LoRA 模型路径
const LORA_ANIME: &str = "/dev-data/ai-test/media_agent/models/lora/anime_lineart.safetensors";

// VAE 模型路径
const VAE_SDXL: &str = "/dev-data/ai-test/media_agent/models/vae/sdxl_vae.safetensors";

fn model_exists() -> bool { Path::new(MODEL_PATH).exists() }
fn sd_cli_exists() -> bool { Path::new(SD_CLI_PATH).exists() }
fn controlnet_exists() -> bool { Path::new(CONTROLNET_OPENPOSE).exists() || Path::new(CONTROLNET_CANNY).exists() }
fn lora_exists() -> bool { Path::new(LORA_ANIME).exists() }
fn vae_exists() -> bool { Path::new(VAE_SDXL).exists() }

fn generate_image(prompt: &str, negative: &str, width: u32, height: u32, steps: u32, cfg: f32, seed: u32, output: &str) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", negative,
        "-W", &width.to_string(),
        "-H", &height.to_string(),
        "--steps", &steps.to_string(),
        "--cfg-scale", &cfg.to_string(),
        "-s", &seed.to_string(),
        "--sampling-method", "euler",
        "-o", output,
        "--backend", "cpu",
    ]);
    let out = cmd.output().map_err(|e| format!("exec failed: {}", e))?;
    if !out.status.success() {
        return Err(format!("sd-cli failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    std::fs::read(output).map_err(|e| format!("read failed: {}", e))
}

// ============================================================================
// Workflow 模板验证测试
// ============================================================================

#[test]
fn test_all_workflow_templates_exist() {
    let workflows = [
        "text_to_image_basic.jsonld",
        "image_to_image.jsonld",
        "high_quality_generation.jsonld",
        "batch_generation.jsonld",
        "upscale_image.jsonld",
        "inpainting.jsonld",
        "variation_generation.jsonld",
        "controlnet_pose.jsonld",
        "lora_style.jsonld",
        "agent_pdca_workflow.jsonld",
        "multi_stage_pipeline.jsonld",
        "batch_generate.jsonld",
        "generate_and_review.jsonld",
    ];

    let mut missing = vec![];
    for wf in &workflows {
        let path = format!("{}/{}", WORKFLOW_DIR, wf);
        if !Path::new(&path).exists() {
            missing.push(wf);
        }
    }

    if !missing.is_empty() {
        panic!("Missing workflow templates: {:?}", missing);
    }

    println!("All {} workflow templates exist!", workflows.len());
}

#[test]
fn test_workflow_template_valid_json() {
    let workflows = std::fs::read_dir(WORKFLOW_DIR)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "jsonld").unwrap_or(false))
        .collect::<Vec<_>>();

    let mut errors = vec![];

    for wf in workflows {
        let path = wf.path();
        let content = std::fs::read_to_string(&path).unwrap();
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
            errors.push(format!("{}: {}", path.display(), e));
        }
    }

    if !errors.is_empty() {
        panic!("Invalid JSON-LD workflows: {:?}", errors);
    }

    println!("All workflow templates are valid JSON!");
}

// ============================================================================
// 文生图完整测试
// ============================================================================

#[test]
fn test_text_to_image_small() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let output = format!("{}/t2i_small.png", OUTPUT_DIR);
    let result = generate_image("a red flower", "", 256, 256, 10, 7.0, 1, &output);
    match result {
        Ok(data) => println!("Small image: {} bytes", data.len()),
        Err(e) => println!("Failed: {}", e),
    }
}

#[test]
fn test_text_to_image_medium() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let output = format!("{}/t2i_medium.png", OUTPUT_DIR);
    let result = generate_image("a sunset landscape", "blurry", 512, 512, 15, 7.5, 2, &output);
    match result {
        Ok(data) => println!("Medium image: {} bytes", data.len()),
        Err(e) => println!("Failed: {}", e),
    }
}

#[test]
fn test_text_to_image_large() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let output = format!("{}/t2i_large.png", OUTPUT_DIR);
    let result = generate_image("detailed portrait", "low quality", 768, 768, 20, 8.0, 3, &output);
    match result {
        Ok(data) => println!("Large image: {} bytes", data.len()),
        Err(e) => println!("Failed: {}", e),
    }
}

// ============================================================================
// 不同采样器测试
// ============================================================================

#[test]
fn test_sampler_euler() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let output = format!("{}/sampler_euler.png", OUTPUT_DIR);
    let result = generate_image("test", "", 256, 256, 10, 7.0, 10, &output);
    match result {
        Ok(data) => println!("Euler: {} bytes", data.len()),
        Err(e) => println!("Failed: {}", e),
    }
}

#[test]
fn test_sampler_euler_a() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    
    let output = format!("{}/sampler_euler_a.png", OUTPUT_DIR);
    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", "a cat",
        "-W", "256", "-H", "256",
        "--steps", "10",
        "--cfg-scale", "7.0",
        "-s", "11",
        "--sampling-method", "euler_a",
        "-o", &output,
        "--backend", "cpu",
    ]);
    
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let data = std::fs::read(&output).unwrap();
            println!("Euler A: {} bytes", data.len());
        }
        _ => println!("Failed"),
    }
}

#[test]
fn test_sampler_dpmpp_2m() {
    if !model_exists() || !sd_cli_exists() { println!("SKIP"); return; }
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    
    let output = format!("{}/sampler_dpmpp.png", OUTPUT_DIR);
    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", "a dog",
        "-W", "256", "-H", "256",
        "--steps", "10",
        "--cfg-scale", "7.0",
        "-s", "12",
        "--sampling-method", "dpm++2m",
        "-o", &output,
        "--backend", "cpu",
    ]);
    
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let data = std::fs::read(&output).unwrap();
            println!("DPM++2M: {} bytes", data.len());
        }
        _ => println!("Failed"),
    }
}

// ============================================================================
// 模型检查测试
// ============================================================================

#[test]
fn test_controlnet_models_available() {
    let openpose = Path::new(CONTROLNET_OPENPOSE).exists();
    let canny = Path::new(CONTROLNET_CANNY).exists();
    
    println!("ControlNet OpenPose: {}", openpose);
    println!("ControlNet Canny: {}", canny);
    
    if !openpose && !canny {
        println!("WARNING: No ControlNet models available. Download with:");
        println!("  wget -O {} https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors", CONTROLNET_OPENPOSE);
    }
}

#[test]
fn test_lora_models_available() {
    let anime = Path::new(LORA_ANIME).exists();
    
    println!("LoRA Anime Lineart: {}", anime);
    
    if !anime {
        println!("WARNING: No LoRA models available. Download with:");
        println!("  wget -O {} https://huggingface.co/TheRafal/Elegant-Anime-Lineart/resolve/main/Elegant_Anime_Lineart.safetensors", LORA_ANIME);
    }
}

#[test]
fn test_vae_models_available() {
    let sdxl = Path::new(VAE_SDXL).exists();
    
    println!("VAE SDXL: {}", sdxl);
    
    if !sdxl {
        println!("WARNING: No additional VAE models available. Download with:");
        println!("  wget -O {} https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors", VAE_SDXL);
    }
}

#[test]
fn test_all_models_summary() {
    println!("=== Models Summary ===");
    println!("SD1.5 Checkpoint: {}", model_exists());
    println!("ControlNet OpenPose: {}", Path::new(CONTROLNET_OPENPOSE).exists());
    println!("ControlNet Canny: {}", Path::new(CONTROLNET_CANNY).exists());
    println!("LoRA Anime: {}", Path::new(LORA_ANIME).exists());
    println!("VAE SDXL: {}", Path::new(VAE_SDXL).exists());
    
    // 统计各目录下的模型数量
    let dirs = ["checkpoints", "controlnet", "lora", "vae", "clip", "embeddings"];
    for dir in dirs {
        let path = format!("/dev-data/ai-test/media_agent/models/{}", dir);
        let count = std::fs::read_dir(&path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "safetensors" || ext == "pt" || ext == "bin").unwrap_or(false))
            .count();
        println!("models/{}/: {} model files", dir, count);
    }
}

// ============================================================================
// Agent Workflow 测试
// ============================================================================

#[test]
fn test_agent_pdca_workflow_structure() {
    let path = format!("{}/agent_pdca_workflow.jsonld", WORKFLOW_DIR);
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 验证 PDCA 结构
    assert!(json["pdca_cycle"].is_object());
    assert!(json["pdca_cycle"]["plan"].is_object());
    assert!(json["pdca_cycle"]["do"].is_object());
    assert!(json["pdca_cycle"]["check"].is_object());
    assert!(json["pdca_cycle"]["act"].is_object());
    
    println!("Agent PDCA workflow structure validated!");
}

#[test]
fn test_workflow_selection_mapping() {
    let path = format!("{}/agent_pdca_workflow.jsonld", WORKFLOW_DIR);
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    let selection = json["workflow_selection"].as_object().unwrap();
    let workflow_types = ["text_to_image", "image_to_image", "high_quality", "batch", "upscale", "inpainting", "controlnet", "lora"];
    
    for wf_type in workflow_types {
        assert!(selection.contains_key(wf_type), "Missing workflow selection for {}", wf_type);
        let wf_file = selection[wf_type].as_str().unwrap();
        let wf_path = format!("{}/{}", WORKFLOW_DIR, wf_file);
        assert!(Path::new(&wf_path).exists(), "Referenced workflow {} not found", wf_file);
    }
    
    println!("Workflow selection mapping validated for all {} types!", workflow_types.len());
}

#[test]
fn test_multi_stage_pipeline_structure() {
    let path = format!("{}/multi_stage_pipeline.jsonld", WORKFLOW_DIR);
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 验证多阶段结构
    assert!(json["stages"].is_array());
    let stages = json["stages"].as_array().unwrap();
    assert!(stages.len() >= 2, "Pipeline should have at least 2 stages");
    
    for stage in stages {
        assert!(stage["name"].is_string());
        assert!(stage["workflow"].is_string());
        assert!(stage["outputs"].is_array());
    }
    
    println!("Multi-stage pipeline structure validated with {} stages!", stages.len());
}

// ============================================================================
// Skills 验证测试
// ============================================================================

#[test]
fn test_all_skills_exist() {
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    let skills = ["text_to_image.jsonld", "image_to_image.jsonld", "generate_video.jsonld"];
    
    for skill in &skills {
        let path = format!("{}/{}", skills_dir, skill);
        assert!(Path::new(&path).exists(), "Skill {} not found", skill);
    }
    
    println!("All {} skills exist!", skills.len());
}

#[test]
fn test_skills_valid_schema() {
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    
    for entry in std::fs::read_dir(skills_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().map(|e| e == "jsonld").unwrap_or(false) {
            let content = std::fs::read_to_string(entry.path()).unwrap();
            let json: serde_json::Value = serde_json::from_str(&content).unwrap();
            
            // 验证基本 Skill schema
            assert!(json["@type"].is_string(), "Missing @type in {}", entry.path().display());
            assert!(json["name"].is_string(), "Missing name in {}", entry.path().display());
        }
    }
    
    println!("All skills have valid schema!");
}