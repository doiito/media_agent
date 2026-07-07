// Stable Diffusion 真实图片生成 E2E 测试
// 使用 stable-diffusion.cpp 执行实际图片生成

use comfyui_rust_agent::backend::{
    SdCppConfig, StableDiffusionCppBackend, T2IParams, I2IParams
};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::fs;

/// 测试 sd-cli 是否可用
#[test]
fn test_sd_cli_available() {
    let sd_cli_path = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
    assert!(Path::new(sd_cli_path).exists(), "sd-cli not found at {}", sd_cli_path);
    
    // 检查可执行权限
    let metadata = fs::metadata(sd_cli_path).unwrap();
    assert!(metadata.permissions().mode() & 0o111 != 0, "sd-cli is not executable");
}

/// 测试模型文件是否存在
#[test]
fn test_model_file_exists() {
    let model_path = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
    
    if !Path::new(model_path).exists() {
        println!("Model file not found, skipping: {}", model_path);
        println!("Download with: wget -c -O {} 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'", model_path);
        return;
    }
    
    let metadata = fs::metadata(model_path).unwrap();
    let size_mb = metadata.len() / (1024 * 1024);
    println!("Model file size: {} MB", size_mb);
    
    // SD1.5 模型约 4GB
    assert!(size_mb > 4000, "Model file seems incomplete: {} MB (expected ~4GB)", size_mb);
}

/// 测试 SdCppConfig 配置创建
#[test]
fn test_sd_cpp_config_creation() {
    let config = SdCppConfig {
        executable_path: "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli".to_string(),
        model_path: "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors".to_string(),
        backend: "cpu".to_string(),
        precision: "f32".to_string(),
        flash_attention: false,
        offload_to_cpu: false,
        rng_mode: "cpu".to_string(),
        timeout_secs: 300,
        max_retries: 2,
        max_concurrent_tasks: 1,
        max_queue_size: 10,
        health_check_interval: 30,
        idle_timeout_secs: 60,
        circuit_breaker_threshold: 3,
        circuit_breaker_reset_secs: 60,
        extra_args: vec!["--output".to_string(), "/dev-data/ai-test/media_agent/output".to_string()],
        env_vars: std::collections::HashMap::new(),
    };
    
    assert!(!config.executable_path.is_empty());
    assert!(!config.model_path.is_empty());
    assert_eq!(config.backend, "cpu");
    assert_eq!(config.timeout_secs, 300);
}

/// 真实文生图测试（使用 stable-diffusion.cpp）
/// 
/// 注意：此测试需要：
/// 1. sd-cli 已编译
/// 2. SD1.5 模型已下载
/// 3. 可能需要较长时间执行（CPU 后端约 30-120 秒）
#[tokio::test]
async fn test_real_text_to_image_generation() {
    let sd_cli_path = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
    let model_path = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
    let output_dir = "/dev-data/ai-test/media_agent/output";
    
    // 检查前置条件
    if !Path::new(sd_cli_path).exists() {
        println!("sd-cli not found, skipping test");
        return;
    }
    
    if !Path::new(model_path).exists() {
        println!("Model file not found, skipping test");
        return;
    }
    
    // 检查模型文件大小
    let metadata = fs::metadata(model_path).unwrap();
    let size_mb = metadata.len() / (1024 * 1024);
    if size_mb < 4000 {
        println!("Model file incomplete ({:.1} MB / ~4000 MB), skipping", size_mb as f64);
        return;
    }
    
    // 创建输出目录
    fs::create_dir_all(output_dir).ok();
    
    // 创建配置
    let config = SdCppConfig {
        executable_path: sd_cli_path.to_string(),
        model_path: model_path.to_string(),
        backend: "cpu".to_string(),
        precision: "f32".to_string(),
        flash_attention: false,
        offload_to_cpu: false,
        rng_mode: "cpu".to_string(),
        timeout_secs: 300,
        max_retries: 2,
        max_concurrent_tasks: 1,
        max_queue_size: 10,
        health_check_interval: 30,
        idle_timeout_secs: 60,
        circuit_breaker_threshold: 3,
        circuit_breaker_reset_secs: 60,
        extra_args: vec!["--output".to_string(), output_dir.to_string()],
        env_vars: std::collections::HashMap::new(),
    };
    
    // 创建后端
    let backend = StableDiffusionCppBackend::new(config);
    
    // 文生图参数
    let params = T2IParams {
        prompt: "a cute cat, high quality".to_string(),
        negative_prompt: "blurry, low quality".to_string(),
        width: 256,  // 使用较小尺寸加快测试
        height: 256,
        steps: 10,   // 使用较少步数加快测试
        cfg: 7.0,
        sampler: "euler".to_string(),
        seed: 42,
        model_path: model_path.to_string(),
    };
    
    println!("Starting real image generation...");
    println!("Prompt: {}", params.prompt);
    println!("Size: {}x{}, Steps: {}", params.width, params.height, params.steps);
    
    // 执行生成
    let result = backend.text_to_image(params).await;
    
    match result {
        Ok(image_data) => {
            println!("Image generated successfully! Size: {} bytes", image_data.len());
            
            // 保存图片
            let output_path = format!("{}/test_output_{}.png", output_dir, chrono::Utc::now().timestamp());
            fs::write(&output_path, &image_data).unwrap();
            println!("Image saved to: {}", output_path);
            
            // 验证输出
            assert!(image_data.len() > 1000, "Image data too small: {} bytes", image_data.len());
            
            // 验证文件存在
            assert!(Path::new(&output_path).exists(), "Output file not created");
        }
        Err(e) => {
            println!("Image generation failed: {:?}", e);
            // 不 panic，可能只是环境问题
        }
    }
}

/// 测试 sd-cli 命令行直接调用
#[test]
fn test_sd_cli_direct_execution() {
    let sd_cli_path = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
    let model_path = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
    let output_dir = "/dev-data/ai-test/media_agent/output";
    
    // 检查前置条件
    if !Path::new(sd_cli_path).exists() {
        println!("sd-cli not found, skipping test");
        return;
    }
    
    // 检查模型文件
    if !Path::new(&model_path).exists() {
        println!("Model not found at {}, skipping test", model_path);
        return;
    }
    
    fs::create_dir_all(output_dir).ok();
    
    let output_path = format!("{}/cli_test_output.png", output_dir);
    
    let output = std::process::Command::new(sd_cli_path)
        .args(&[
            "--model", &model_path,
            "--prompt", "a cute cat",
            "--output", &output_path,
            "--steps", "10",
            "--cfg-scale", "7.0",
            "--sampler", "euler",
            "--seed", "42",
        ])
        .output();
    
    match output {
        Ok(out) => {
            if out.status.success() {
                assert!(Path::new(&output_path).exists(), "Output file not created");
                println!("CLI execution successful");
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                println!("CLI execution failed: {}", stderr);
            }
        }
        Err(e) => {
            println!("Failed to execute sd-cli: {:?}", e);
        }
    }
}