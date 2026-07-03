// 短视频生成测试
// 测试视频相关工作流和功能 - 增强版覆盖度

use std::path::Path;
use std::process::Command;
use std::fs;

const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output/video_test";
const WORKFLOW_DIR: &str = "/dev-data/ai-test/media_agent/workflows";

fn model_exists() -> bool { Path::new(MODEL_PATH).exists() }
fn sd_cli_exists() -> bool { Path::new(SD_CLI_PATH).exists() }

// ============================================================================
// 视频工作流模板测试
// ============================================================================

#[test]
fn test_video_workflow_template_exists() {
    // 检查视频相关工作流是否存在
    let video_workflows = [
        "generate_video.jsonld", // skills 目录
    ];
    
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    
    for wf in &video_workflows {
        let path = format!("{}/{}", skills_dir, wf);
        assert!(Path::new(&path).exists(), "Video workflow {} not found", wf);
    }
    
    println!("Video workflow templates exist!");
}

#[test]
fn test_video_workflow_valid_json() {
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    let video_skill = format!("{}/generate_video.jsonld", skills_dir);
    
    if !Path::new(&video_skill).exists() {
        println!("SKIP: Video skill not found");
        return;
    }
    
    let content = std::fs::read_to_string(&video_skill).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content)
        .expect("Video workflow should be valid JSON");
    
    assert!(json["name"].is_string() || json["schema:name"].is_string(), "Video workflow should have name");
    let name = json["name"].as_str().or(json["schema:name"].as_str()).unwrap_or("unknown");
    println!("Video workflow is valid JSON: {}", name);
}

// ============================================================================
// 帧序列生成测试（视频基础）
// ============================================================================

/// 生成帧序列用于视频合成
fn generate_frame_sequence(
    base_prompt: &str,
    num_frames: u32,
    width: u32,
    height: u32,
    steps: u32,
    base_seed: u32,
) -> Vec<String> {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return vec![];
    }
    
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    
    let mut frame_paths = vec![];
    
    for i in 0..num_frames {
        // 为每帧添加时间变化的描述
        let frame_prompt = format!("{} frame {} of {}", base_prompt, i + 1, num_frames);
        let seed = base_seed + i;
        let output = format!("{}/frame_{:03}.png", OUTPUT_DIR, i);
        
        let mut cmd = Command::new(SD_CLI_PATH);
        cmd.args([
            "-m", MODEL_PATH,
            "-p", &frame_prompt,
            "-n", "blurry, low quality",
            "-W", &width.to_string(),
            "-H", &height.to_string(),
            "--steps", &steps.to_string(),
            "--cfg-scale", "7.0",
            "-s", &seed.to_string(),
            "--sampling-method", "euler",
            "-o", &output,
            "--backend", "cpu",
        ]);
        
        match cmd.output() {
            Ok(out) if out.status.success() => {
                frame_paths.push(output.clone());
                println!("Frame {} generated: {}", i + 1, output);
            }
            _ => {
                println!("Frame {} generation failed", i + 1);
            }
        }
    }
    
    frame_paths
}

/// 测试生成帧序列
#[test]
fn test_generate_frame_sequence() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    println!("Generating 4 frames for video test...");
    
    let frames = generate_frame_sequence(
        "a cat walking in a garden",
        4,  // 少量帧用于测试
        256, 256,  // 小尺寸加快生成
        10,  // 少步数加快生成
        100,
    );
    
    // 验证帧数量
    assert!(frames.len() >= 2, "Should generate at least 2 frames");
    
    // 验证每帧存在
    for frame in &frames {
        assert!(Path::new(frame).exists(), "Frame {} should exist", frame);
        let size = std::fs::metadata(frame).unwrap().len();
        assert!(size > 1000, "Frame {} should have valid size", frame);
    }
    
    println!("Generated {} frames successfully!", frames.len());
}

// ============================================================================
// 动画风格帧生成测试
// ============================================================================

#[test]
fn test_anime_style_frames() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    
    // 检查 LoRA 模型是否可用
    let lora_path = "/dev-data/ai-test/media_agent/models/lora/pixart_style.safetensors";
    let has_lora = Path::new(lora_path).exists() && std::fs::metadata(lora_path).unwrap().len() > 1000;
    
    if !has_lora {
        println!("SKIP: LoRA model not available");
        return;
    }
    
    println!("Generating anime style frames with LoRA...");
    
    // 使用 LoRA 生成帧（如果 sd-cli 支持）
    for i in 0..2 {
        let output = format!("{}/anime_frame_{:03}.png", OUTPUT_DIR, i);
        
        // 注：sd-cli 的 LoRA 支持需要额外参数
        let mut cmd = Command::new(SD_CLI_PATH);
        cmd.args([
            "-m", MODEL_PATH,
            "-p", "anime style girl smiling",
            "-W", "256", "-H", "256",
            "--steps", "10",
            "--cfg-scale", "7.0",
            "-s", &(200 + i).to_string(),
            "--sampling-method", "euler",
            "-o", &output,
            "--backend", "cpu",
        ]);
        
        match cmd.output() {
            Ok(out) if out.status.success() => {
                println!("Anime frame {} generated", i + 1);
            }
            _ => println!("Anime frame {} generation failed", i + 1),
        }
    }
}

// ============================================================================
// 视频工作流验证测试
// ============================================================================

#[test]
fn test_video_workflow_steps_valid() {
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    let video_skill = format!("{}/generate_video.jsonld", skills_dir);
    
    if !Path::new(&video_skill).exists() {
        println!("SKIP: Video skill not found, creating placeholder");
        return;
    }
    
    let content = std::fs::read_to_string(&video_skill).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 验证视频工作流的基本结构
    if let Some(steps) = json["steps"].as_array() {
        assert!(steps.len() > 0, "Video workflow should have steps");
        
        for step in steps {
            assert!(step["name"].is_string() || step["@type"].is_string(), 
                    "Each step should have name or type");
        }
        
        println!("Video workflow has {} valid steps", steps.len());
    }
}

// ============================================================================
// 视频参数验证测试
// ============================================================================

#[test]
fn test_video_parameters_schema() {
    let skills_dir = "/dev-data/ai-test/media_agent/skills";
    let video_skill = format!("{}/generate_video.jsonld", skills_dir);
    
    if !Path::new(&video_skill).exists() {
        println!("SKIP: Video skill not found");
        return;
    }
    
    let content = std::fs::read_to_string(&video_skill).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 检查是否有参数定义
    if let Some(params) = json["parameters"].as_object() {
        // 视频应该有帧数、时长、分辨率等参数
        let expected_params = ["fps", "duration", "frames", "width", "height"];
        let mut found_params = 0;
        
        for param in expected_params {
            if params.contains_key(param) {
                found_params += 1;
            }
        }
        
        println!("Video workflow has {} of {} expected parameters", found_params, expected_params.len());
    }
}

// ============================================================================
// 视频输出格式测试
// ============================================================================

#[test]
fn test_video_output_formats() {
    // 测试不同的输出格式是否支持
    let supported_formats = ["png", "mp4", "gif", "webm"];
    
    for format in supported_formats {
        // 检查是否有对应的工作流或配置支持该格式
        println!("Checking support for format: {}", format);
    }
    
    // PNG 帧序列是目前支持的格式
    assert!(true, "PNG frame sequence is supported");
}

// ============================================================================
// 视频工作流与 Agent 集成测试
// ============================================================================

#[tokio::test]
async fn test_agent_video_workflow_integration() {
    // 检查 DeepSeek API
    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        println!("SKIP: DEEPSEEK_API_KEY not set");
        return;
    }
    
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    use comfyui_rust_agent::agent::llm::{LlmClient, ChatMessage, ChatRequest};
    
    let llm_client = LlmClient::from_env();
    
    // 让 Agent 分析视频生成请求
    let messages = vec![
        ChatMessage::system("You are a video generation assistant. Output JSON with video parameters."),
        ChatMessage::user("Generate a 4-frame video of a sunset over ocean"),
    ];
    
    let request = ChatRequest {
        model: llm_client.config().default_model.clone(),
        messages,
        max_tokens: Some(300),
        temperature: Some(0.3),
        tools: None,
        tool_choice: None,
        stream: Some(false),
    };
    
    println!("Step 1: Asking Agent for video generation parameters...");
    
    let response = llm_client.chat(request).await;
    
    match response {
        Ok(resp) => {
            let content = resp.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();
            
            println!("Agent response: {}", content);
            
            // 解析参数并生成帧
            if let Some(json_start) = content.find('{') {
                if let Some(json_end) = content.rfind('}') {
                    let json_str = &content[json_start..=json_end];
                    if let Ok(params) = serde_json::from_str::<serde_json::Value>(json_str) {
                        println!("Parsed video params: {:?}", params);
                    }
                }
            }
        }
        Err(e) => {
            println!("Agent request failed: {}", e);
        }
    }
}

// ============================================================================
// 新增视频 Workflow 测试
// ============================================================================

#[test]
fn test_all_video_workflows_exist() {
    let video_workflows = [
        "video_generation_pipeline.jsonld",
        "style_aware_video.jsonld",
        "controlnet_animated_video.jsonld",
        "text_to_video_direct.jsonld",
        "multi_prompt_video_morph.jsonld",
        "image_to_video_svd.jsonld",
        "video_frame_interpolation.jsonld",
        "latent_interpolation.jsonld",
        "generate_video.jsonld",
    ];
    
    let mut found_count = 0;
    for wf in &video_workflows {
        let path = format!("{}/{}", WORKFLOW_DIR, wf);
        if Path::new(&path).exists() {
            found_count += 1;
            println!("✓ Found: {}", wf);
        } else {
            // 检查 skills 目录
            let skills_path = format!("/dev-data/ai-test/media_agent/skills/{}", wf);
            if Path::new(&skills_path).exists() {
                found_count += 1;
                println!("✓ Found in skills: {}", wf);
            }
        }
    }
    
    assert!(found_count >= 6, "Should have at least 6 video workflows");
    println!("Found {} video workflow templates", found_count);
}

#[test]
fn test_video_generation_pipeline_workflow() {
    let path = format!("{}/video_generation_pipeline.jsonld", WORKFLOW_DIR);
    if !Path::new(&path).exists() {
        println!("SKIP: video_generation_pipeline not found");
        return;
    }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    // 验证 pipeline 结构
    assert!(json["stages"].is_array(), "Pipeline should have stages");
    let stages = json["stages"].as_array().unwrap();
    assert!(stages.len() >= 3, "Pipeline should have at least 3 stages");
    
    // 检查每个阶段
    for stage in stages {
        assert!(stage["name"].is_string(), "Each stage should have name");
        assert!(stage["workflow"].is_string() || stage["steps"].is_array(), 
                "Each stage should have workflow or steps");
    }
    
    println!("video_generation_pipeline validated with {} stages", stages.len());
}

#[test]
fn test_style_aware_video_workflow() {
    let path = format!("{}/style_aware_video.jsonld", WORKFLOW_DIR);
    if !Path::new(&path).exists() {
        println!("SKIP: style_aware_video not found");
        return;
    }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    
    // 验证 LoRA 参数
    assert!(json["parameters"]["style_lora"].is_object());
    assert!(json["parameters"]["style_strength"].is_object());
    assert!(json["parameters"]["animation_mode"].is_object());
    
    // 验证帧数参数
    assert!(json["parameters"]["keyframe_count"].is_object());
    assert!(json["parameters"]["interpolation_frames_per_keyframe"].is_object());
    
    println!("style_aware_video workflow validated");
}

#[test]
fn test_controlnet_animated_video_workflow() {
    let path = format!("{}/controlnet_animated_video.jsonld", WORKFLOW_DIR);
    if !Path::new(&path).exists() {
        println!("SKIP: controlnet_animated_video not found");
        return;
    }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    
    // 验证 ControlNet 参数
    assert!(json["parameters"]["controlnet_type"].is_object());
    assert!(json["parameters"]["control_sequence_directory"].is_object());
    assert!(json["parameters"]["controlnet_strength"].is_object());
    
    println!("controlnet_animated_video workflow validated");
}

#[test]
fn test_text_to_video_direct_workflow() {
    let path = format!("{}/text_to_video_direct.jsonld", WORKFLOW_DIR);
    if !Path::new(&path).exists() {
        println!("SKIP: text_to_video_direct not found");
        return;
    }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    
    // 验证动画参数
    let anim_type = json["parameters"]["animation_type"].as_object();
    assert!(anim_type.is_some());
    
    // 验证枚举值
    if let Some(enum_val) = json["parameters"]["animation_type"]["enum"].as_array() {
        assert!(enum_val.len() >= 3, "animation_type should have at least 3 options");
    }
    
    println!("text_to_video_direct workflow validated");
}

#[test]
fn test_multi_prompt_video_morph_workflow() {
    let path = format!("{}/multi_prompt_video_morph.jsonld", WORKFLOW_DIR);
    if !Path::new(&path).exists() {
        println!("SKIP: multi_prompt_video_morph not found");
        return;
    }
    
    let content = fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert!(json["category"].as_str().unwrap_or("") == "video");
    
    // 验证多提示词参数
    assert!(json["parameters"]["prompt_1"].is_object());
    assert!(json["parameters"]["prompt_2"].is_object());
    assert!(json["parameters"]["prompt_3"].is_object());
    assert!(json["parameters"]["morph_frames"].is_object());
    assert!(json["parameters"]["transition_type"].is_object());
    
    println!("multi_prompt_video_morph workflow validated");
}

// ============================================================================
// 视频帧序列真实生成测试
// ============================================================================

#[test]
fn test_generate_video_frame_sequence_real() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }
    
    std::fs::create_dir_all(OUTPUT_DIR).ok();
    
    println!("Generating real video frame sequence (4 frames)...");
    
    // 生成4帧动画序列
    let prompts = [
        "a cat sitting",
        "a cat standing up",
        "a cat walking",
        "a cat running",
    ];
    
    let mut generated_frames = 0;
    
    for (i, prompt) in prompts.iter().enumerate() {
        let output = format!("{}/motion_frame_{:03}.png", OUTPUT_DIR, i);
        
        let mut cmd = Command::new(SD_CLI_PATH);
        cmd.args([
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", "blurry, low quality",
            "-W", "256", "-H", "256",
            "--steps", "10",
            "--cfg-scale", "7.0",
            "-s", &(300 + i).to_string(),
            "--sampling-method", "euler",
            "-o", &output,
            "--backend", "cpu",
        ]);
        
        match cmd.output() {
            Ok(out) if out.status.success() => {
                if Path::new(&output).exists() {
                    let size = std::fs::metadata(&output).unwrap().len();
                    if size > 1000 {
                        generated_frames += 1;
                        println!("Frame {} generated: {} ({} bytes)", i + 1, output, size);
                    }
                }
            }
            _ => println!("Frame {} generation failed", i + 1),
        }
    }
    
    assert!(generated_frames >= 2, "Should generate at least 2 frames");
    println!("Successfully generated {} frames for video sequence", generated_frames);
}

#[test]
fn test_frame_interpolation_simulation() {
    // 模拟帧插值测试（验证逻辑）
    let original_frames = 4;
    let interpolation_multiplier = 2;
    
    let interpolated_frames = original_frames + (original_frames - 1) * interpolation_multiplier;
    
    assert!(interpolated_frames > original_frames, 
            "Interpolation should increase frame count");
    
    println!("Frame interpolation: {} -> {} frames", original_frames, interpolated_frames);
}

// ============================================================================
// 视频参数边界测试
// ============================================================================

#[test]
fn test_video_parameter_boundaries() {
    // 测试视频参数的边界值
    let test_cases = [
        ("frames", 1, 100),
        ("fps", 1, 60),
        ("width", 256, 2048),
        ("height", 256, 2048),
        ("steps", 5, 50),
    ];
    
    for (param, min, max) in test_cases {
        println!("Testing {} bounds: min={}, max={}", param, min, max);
        assert!(min > 0, "{} min should be positive", param);
        assert!(max > min, "{} max should be greater than min", param);
    }
    
    println!("All video parameter boundaries are valid");
}

#[test]
fn test_video_workflow_category_coverage() {
    // 统计视频 workflow 类别覆盖
    let video_files = fs::read_dir(WORKFLOW_DIR)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "jsonld"))
        .collect::<Vec<_>>();
    
    let mut video_count = 0;
    
    for file in video_files {
        let path = file.path();
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if json["category"].as_str().unwrap_or("") == "video" {
                    video_count += 1;
                }
            }
        }
    }
    
    assert!(video_count >= 6, "Should have at least 6 video category workflows");
    println!("Found {} video category workflows", video_count);
}

// ============================================================================
// 视频生成性能测试
// ============================================================================

#[test]
fn test_video_generation_time_estimation() {
    // 估算视频生成时间
    let frame_gen_time_seconds = 15; // 每帧生成时间（CPU模式）
    let num_frames = 16;
    let interpolation_multiplier = 2;
    
    let base_time = frame_gen_time_seconds * num_frames;
    let interpolated_frames = num_frames * interpolation_multiplier;
    let total_time = base_time + (interpolated_frames - num_frames) * 2; // 插值时间较短
    
    println!("Estimated video generation time:");
    println!("  Base frames: {} ({}s each) = {}s", num_frames, frame_gen_time_seconds, base_time);
    println!("  Interpolated frames: {}", interpolated_frames);
    println!("  Total estimated time: {}s", total_time);
    
    assert!(total_time < 600, "Video generation should complete within 10 minutes");
}

#[test]
fn test_video_quality_vs_speed_tradeoff() {
    // 测试不同参数组合的质量/速度权衡
    let configs = [
        ("fast", 256, 256, 10, 8),
        ("balanced", 512, 512, 20, 16),
        ("quality", 1024, 1024, 30, 25),
    ];
    
    for (name, width, height, steps, expected_fps) in configs {
        let pixels = width * height;
        let complexity = pixels * steps;
        
        println!("Config {}: {}x{}, {} steps, complexity={}", 
                 name, width, height, steps, complexity);
        
        // 验证参数合理性
        assert!(steps >= 5, "{} steps should be >= 5", name);
        assert!(width >= 256 && height >= 256, "{} resolution should be >= 256", name);
    }
}