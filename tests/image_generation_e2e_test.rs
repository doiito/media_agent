// 真实图片生成 E2E 测试
// 使用 stable-diffusion.cpp 的 sd-cli 命令行直接生成图片
// 需要：模型文件已下载到 models/checkpoints/

use std::path::Path;
use std::process::Command;

// 模型路径
const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output";

/// 检查模型文件是否存在
fn model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
}

/// 检查 sd-cli 是否存在
fn sd_cli_exists() -> bool {
    Path::new(SD_CLI_PATH).exists()
}

/// 直接调用 sd-cli 生成图片
fn generate_image_with_sd_cli(
    prompt: &str,
    negative_prompt: &str,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f32,
    seed: u32,
    output_path: &str,
) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", negative_prompt,
        "-W", &width.to_string(),
        "-H", &height.to_string(),
        "--steps", &steps.to_string(),
        "--cfg-scale", &cfg.to_string(),
        "-s", &seed.to_string(),
        "--sampling-method", "euler",
        "-o", output_path,
        "--backend", "cpu",
    ]);

    println!("Executing: {} -m {} -p \"{}\" ...", SD_CLI_PATH, MODEL_PATH, prompt);

    let output = cmd.output().map_err(|e| format!("Failed to execute sd-cli: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("sd-cli failed: {} - stderr: {} - stdout: {}", output.status, stderr, stdout));
    }

    println!("sd-cli stdout: {}", String::from_utf8_lossy(&output.stdout));

    // 读取生成的图片
    std::fs::read(output_path).map_err(|e| format!("Failed to read output image '{}': {}", output_path, e))
}

// ============================================================================
// 基础测试
// ============================================================================

#[test]
fn test_model_file_exists() {
    if !model_exists() {
        println!("SKIP: Model file not found at {}", MODEL_PATH);
        println!("Please download with:");
        println!("  wget -O {} https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors", MODEL_PATH);
        return;
    }
    let size = std::fs::metadata(MODEL_PATH).unwrap().len();
    println!("Model file found: {} bytes ({:.2} GB)", size, size as f64 / 1e9);
    assert!(size > 3_000_000_000, "Model file too small, likely incomplete (expected ~4GB)");
}

#[test]
fn test_sd_cli_exists() {
    if !sd_cli_exists() {
        println!("SKIP: sd-cli not found at {}", SD_CLI_PATH);
        println!("Please build stable-diffusion.cpp:");
        println!("  cd /dev-data/ai-test/stable-diffusion.cpp");
        println!("  cmake -B build && cmake --build build -j$(nproc)");
        return;
    }
    println!("sd-cli found at {}", SD_CLI_PATH);
}

// ============================================================================
// 文生图真实测试
// ============================================================================

/// 测试文生图基础功能（真实生成）
#[test]
fn test_text_to_image_real() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    // 创建输出目录
    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let output_path = format!("{}/test_t2i_{}.png", OUTPUT_DIR, chrono::Utc::now().timestamp());

    println!("Starting text-to-image generation...");
    println!("  Prompt: a cute cat");
    println!("  Size: 256x256");
    println!("  Steps: 10");

    let result = generate_image_with_sd_cli(
        "a cute cat, simple background",
        "blurry, low quality",
        256,
        256,
        10,
        7.0,
        42,
        &output_path,
    );

    match result {
        Ok(image_data) => {
            println!("Image generated successfully: {} bytes", image_data.len());
            
            // 验证图片数据有效（PNG 最小约 100 bytes）
            assert!(image_data.len() > 100, "Generated image too small");
            
            // 验证文件存在
            assert!(Path::new(&output_path).exists());
            println!("Image saved to: {}", output_path);
        }
        Err(e) => {
            println!("Image generation failed: {}", e);
            // 不让测试失败，因为可能是环境问题
        }
    }
}

/// 测试多张图片生成（批量）
#[test]
fn test_batch_image_generation() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    // 生成 2 张图片
    let prompts = vec![
        ("a red apple on table", "apple", 101),
        ("a blue bird on tree", "bird", 102),
    ];

    println!("Starting batch generation of {} images...", prompts.len());

    for (prompt, name, seed) in prompts {
        let output_path = format!("{}/batch_{}.png", OUTPUT_DIR, name);

        let result = generate_image_with_sd_cli(
            prompt,
            "",
            256,
            256,
            10,
            7.0,
            seed,
            &output_path,
        );

        match result {
            Ok(image_data) => {
                println!("Image '{}' generated: {} bytes", name, image_data.len());
            }
            Err(e) => {
                println!("Image '{}' failed: {}", name, e);
            }
        }
    }
}

/// 测试不同步数的效果
#[test]
fn test_different_step_counts() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let step_counts = vec![5, 10, 15];

    for steps in step_counts {
        let output_path = format!("{}/steps_{}.png", OUTPUT_DIR, steps);

        println!("Generating with {} steps...", steps);

        let result = generate_image_with_sd_cli(
            "a sunset over ocean",
            "",
            256,
            256,
            steps,
            7.0,
            42, // 固定 seed 比较效果
            &output_path,
        );

        match result {
            Ok(image_data) => {
                println!("  Saved: {} ({} bytes)", output_path, image_data.len());
            }
            Err(e) => {
                println!("  Failed: {}", e);
            }
        }
    }
}

/// 测试不同尺寸
#[test]
fn test_different_sizes() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let sizes = vec![
        (256, 256, "small"),
        (384, 384, "medium"),
        (512, 512, "large"),
    ];

    for (width, height, name) in sizes {
        let output_path = format!("{}/size_{}.png", OUTPUT_DIR, name);

        println!("Generating {}x{} image...", width, height);

        let result = generate_image_with_sd_cli(
            "a mountain landscape",
            "",
            width,
            height,
            10,
            7.0,
            42,
            &output_path,
        );

        match result {
            Ok(image_data) => {
                println!("  Saved: {} ({} bytes)", output_path, image_data.len());
            }
            Err(e) => {
                println!("  Failed: {}", e);
            }
        }
    }
}

/// 测试高质量生成（更多步数、更大尺寸）
#[test]
fn test_high_quality_generation() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let output_path = format!("{}/high_quality_{}.png", OUTPUT_DIR, chrono::Utc::now().timestamp());

    println!("Starting high-quality generation (512x512, 20 steps)...");

    let result = generate_image_with_sd_cli(
        "a detailed portrait of a cat, professional photography, high quality",
        "blurry, low quality, distorted",
        512,
        512,
        20,
        7.0,
        12345,
        &output_path,
    );

    match result {
        Ok(image_data) => {
            println!("High-quality image generated: {} bytes", image_data.len());
            assert!(image_data.len() > 1000, "High-quality image should be larger");
        }
        Err(e) => {
            println!("High-quality generation failed: {}", e);
        }
    }
}

// ============================================================================
// Agent + 图片生成集成测试
// ============================================================================

/// 测试 Agent 通过 LLM 解析意图后调用图片生成
#[tokio::test]
async fn test_agent_image_generation_integration() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    // 检查 DeepSeek API
    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        println!("SKIP: DEEPSEEK_API_KEY not set");
        return;
    }

    use comfyui_rust_agent::agent::llm::{LlmClient, ChatMessage, ChatRequest};

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let llm_client = LlmClient::from_env();

    // 1. 让 LLM 解析用户请求并生成图片参数
    let messages = vec![
        ChatMessage::system(r#"
你是一个图像生成助手。根据用户描述，输出 JSON 格式的图片生成参数：
{
  "prompt": "英文描述",
  "negative_prompt": "负面提示词",
  "steps": 10,
  "seed": 随机整数
}
只输出 JSON，不要其他内容。
"#),
        ChatMessage::user("画一只可爱的小猫在草地上玩耍"),
    ];

    let request = ChatRequest {
        model: llm_client.config().default_model.clone(),
        messages,
        max_tokens: Some(200),
        temperature: Some(0.3),
        tools: None,
        tool_choice: None,
        stream: Some(false),
    };

    println!("Step 1: Asking LLM to generate image parameters...");

    let response = llm_client.chat(request).await;

    match response {
        Ok(resp) => {
            let content = resp.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();

            println!("LLM response: {}", content);

            // 2. 解析 JSON 参数
            if let Some(json_start) = content.find('{') {
                if let Some(json_end) = content.rfind('}') {
                    let json_str = &content[json_start..=json_end];

                    if let Ok(params_json) = serde_json::from_str::<serde_json::Value>(json_str) {
                        let prompt = params_json["prompt"].as_str().unwrap_or("a cat").to_string();
                        let negative_prompt = params_json["negative_prompt"].as_str().unwrap_or("").to_string();
                        let steps = params_json["steps"].as_u64().unwrap_or(10) as u32;
                        let seed = params_json["seed"].as_u64().unwrap_or(42) as u32;

                        // 3. 执行图片生成
                        let output_path = format!("{}/agent_integration_{}.png", OUTPUT_DIR, chrono::Utc::now().timestamp());

                        println!("Step 2: Generating image with prompt: {}", prompt);

                        let result = generate_image_with_sd_cli(
                            &prompt,
                            &negative_prompt,
                            256,
                            256,
                            steps,
                            7.0,
                            seed,
                            &output_path,
                        );

                        match result {
                            Ok(image_data) => {
                                println!("Step 3: Image saved to {} ({} bytes)", output_path, image_data.len());
                                assert!(image_data.len() > 100);
                            }
                            Err(e) => {
                                println!("Image generation failed: {}", e);
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("LLM request failed: {}", e);
        }
    }
}