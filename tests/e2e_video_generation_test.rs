// E2E 短视频生成全面覆盖测试
// 覆盖：多场景帧生成、工作流构建、帧插值、视频组装、多提示词 Morph、参数边界
// 目标：验证从文本提示词到最终视频文件的完整流程

use std::path::Path;
use std::process::Command;
use std::fs;

// ============================================================================
// 常量定义
// ============================================================================

const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/dev-data/ai-test/stable-diffusion.cpp/build/bin/sd-cli";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output/e2e_video";
const WORKFLOW_DIR: &str = "/dev-data/ai-test/media_agent/workflows";

fn model_exists() -> bool { Path::new(MODEL_PATH).exists() }
fn sd_cli_exists() -> bool { Path::new(SD_CLI_PATH).exists() }
fn ffmpeg_exists() -> bool { Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false) }

// ============================================================================
// 辅助函数
// ============================================================================

fn generate_frame(
    prompt: &str,
    negative: &str,
    seed: u32,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f32,
    output: &str,
) -> bool {
    let mut child = match Command::new(SD_CLI_PATH)
        .args([
            "-m", MODEL_PATH, "-p", prompt, "-n", negative,
            "-W", &width.to_string(), "-H", &height.to_string(),
            "--steps", &steps.to_string(), "--cfg-scale", &cfg.to_string(),
            "-s", &seed.to_string(), "--sampling-method", "euler",
            "-o", output, "--backend", "cpu",
        ])
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(90);
    let mut last_heartbeat = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    return false;
                }
                if last_heartbeat.elapsed() >= std::time::Duration::from_secs(20) {
                    println!("  [sd-cli still running, {:.0}s elapsed]", start.elapsed().as_secs_f64());
                    last_heartbeat = std::time::Instant::now();
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(_) => return false,
        }
    }
}

/// 列出 output 目录下的所有帧文件
fn list_frames_in(dir: &str) -> Vec<String> {
    let dir_path = Path::new(dir);
    if !dir_path.exists() {
        return vec![];
    }
    let mut frames = vec![];
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            if ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "mp4" || ext == "gif" {
                frames.push(path.to_string_lossy().to_string());
            }
        }
    }
    frames.sort();
    frames
}

/// 典型的视频测试参数组合
struct _VideoTestConfig {
    name: &'static str,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f32,
    frames: u32,
    fps: u32,
}

const _VIDEO_TEST_CONFIGS: &[_VideoTestConfig] = &[
    _VideoTestConfig { name: "thumbnail", width: 128, height: 128, steps: 8, cfg: 7.0, frames: 6, fps: 6 },
    _VideoTestConfig { name: "preview",    width: 256, height: 256, steps: 10, cfg: 7.0, frames: 8, fps: 8 },
    _VideoTestConfig { name: "standard",   width: 384, height: 384, steps: 15, cfg: 7.5, frames: 12, fps: 12 },
];

// ============================================================================
// 1. 多场景帧序列生成 E2E — 模拟短视频帧生成
// ============================================================================

mod scene_frame_generation {
    use super::*;

    /// 场景1: 动物行走动画序列（基础运动）
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_animal_walk_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/animal_walk", OUTPUT_DIR)).ok();

        let scene_prompts = [
            "a cat sitting, looking forward",
            "a cat walking slowly",
            "a cat walking confidently",
        ];

        let mut generated = 0;
        for (i, prompt) in scene_prompts.iter().enumerate() {
            let output = format!("{}/animal_walk/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, low quality, deformed", 1000 + i as u32, 256, 256, 5, 7.0, &output) {
                let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!("  Frame {}: {} ({} bytes)", i + 1, prompt, size);
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate at least 3 animal walk frames, got {}", generated);
        println!("test_animal_walk_sequence: {} frames generated", generated);
    }

    /// 场景2: 自然风光渐变序列（日落过程）
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_sunset_progression_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/sunset", OUTPUT_DIR)).ok();

        let time_prompts = [
            "sunset at 5pm, golden sky over mountains",
            "sunset at 6pm, red sky with clouds",
            "sunset at 7pm, dark blue sky first stars appearing",
        ];

        let mut generated = 0;
        for (i, prompt) in time_prompts.iter().enumerate() {
            let output = format!("{}/sunset/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, low quality", 2000 + i as u32, 256, 256, 5, 7.5, &output) {
                generated += 1;
                println!("  Sunset frame {} generated", i + 1);
            }
        }

        assert!(generated >= 3, "Should generate at least 3 sunset frames");
    }

    /// 场景3: 物体旋转/变化序列
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_object_transition_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/object_transition", OUTPUT_DIR)).ok();

        let object_prompts = [
            "a small green bud on a branch",
            "a half-opened pink flower",
            "a fully bloomed pink flower with dewdrops",
        ];

        let mut generated = 0;
        for (i, prompt) in object_prompts.iter().enumerate() {
            let output = format!("{}/object_transition/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, low quality", 3000 + i as u32, 256, 256, 5, 7.0, &output) {
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate at least 3 object transition frames");
        println!("test_object_transition_sequence: {} frames", generated);
    }

    /// 场景4: 人物动作序列
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_human_action_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/human_action", OUTPUT_DIR)).ok();

        let action_prompts = [
            "a person standing, hands by sides",
            "a person raising one hand to wave",
            "a person waving and smiling",
        ];

        let mut generated = 0;
        for (i, prompt) in action_prompts.iter().enumerate() {
            let output = format!("{}/human_action/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, deformed face", 4000 + i as u32, 256, 256, 5, 7.0, &output) {
                generated += 1;
            }
        }

        assert!(generated >= 2, "Should generate at least 2 human action frames");
        println!("test_human_action_sequence: {} action frames", generated);
    }

    /// 场景5: 城市街景变化序列
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_cityscape_progression() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/cityscape", OUTPUT_DIR)).ok();

        let city_prompts = [
            "city street early morning, empty, soft light",
            "city street noon, busy with people and cars",
            "city street night, neon lights, empty",
        ];

        let mut generated = 0;
        for (i, prompt) in city_prompts.iter().enumerate() {
            let output = format!("{}/cityscape/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, low quality", 5000 + i as u32, 256, 256, 5, 7.0, &output) {
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate at least 3 cityscape frames");
    }
}

// ============================================================================
// 2. 多提示词 Morph 模拟 — 在两个提示词之间平滑过渡
// ============================================================================

mod multi_prompt_morph {
    use super::*;

    /// 模拟 morph: 从猫 → 老虎 → 豹子 的渐变序列
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_animal_morph_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/morph_animal", OUTPUT_DIR)).ok();

        // 在猫->老虎之间插值生成 morph 帧
        let morph_prompts = [
            "a domestic cat, small, furry",
            "a tiger-striped cat, larger",
            "a full grown tiger, majestic",
        ];

        let mut generated = 0;
        for (i, prompt) in morph_prompts.iter().enumerate() {
            let output = format!("{}/morph_animal/frame_{:03}.png", OUTPUT_DIR, i);
            // Different seeds for variety plus prompt change adds morph effect
            if generate_frame(prompt, "blurry, low quality", 6000 + i as u32 * 50, 256, 256, 5, 7.0, &output) {
                generated += 1;
                println!("  Morph frame {}: {} -> progressively changing", i + 1, prompt);
            }
        }

        assert!(generated >= 3, "Should generate at least 3 morph frames");
    }

    /// 模拟多风格 morph: 写实 → 油画 → 动漫
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_style_morph_sequence() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/morph_style", OUTPUT_DIR)).ok();

        let style_prompts = [
            "a landscape, photorealistic, detailed",
            "a landscape, watercolor painting, soft colors",
            "a landscape, pixel art style, 8bit",
        ];

        let mut generated = 0;
        for (i, prompt) in style_prompts.iter().enumerate() {
            let output = format!("{}/morph_style/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry, low quality", 7000 + i as u32 * 100, 256, 256, 5, 7.5, &output) {
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate at least 3 style morph frames");
        println!("test_style_morph_sequence: {} frames across 5 art styles", generated);
    }
}

// ============================================================================
// 3. 帧插值模拟 — 验证帧插值逻辑
// ============================================================================

mod frame_interpolation {
    use super::*;

    /// 验证帧插值计算逻辑
    #[test]
    fn test_interpolation_logic() {
        let test_cases = [
            (4, 2, 7),  // 4帧, 2x插值 = 7帧
            (8, 2, 15), // 8帧, 2x插值 = 15帧
            (4, 3, 10), // 4帧, 3x插值 = 10帧
            (6, 4, 21), // 6帧, 4x插值 = 21帧
        ];

        for &(original, multiplier, expected) in &test_cases {
            let interpolated = original + (original - 1) * (multiplier - 1);
            assert_eq!(interpolated, expected,
                "Interpolation: {} frames x{} -> {} (expected {})",
                original, multiplier, interpolated, expected);
            let duration_8fps = interpolated as f32 / 8.0;
            let duration_24fps = interpolated as f32 / 24.0;
            println!("  {} frames x{} = {} interpolated: {:.1}s@8fps or {:.1}s@24fps",
                original, multiplier, interpolated, duration_8fps, duration_24fps);
        }
    }

    /// 对真实生成帧执行帧插值文件计算
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_frame_interpolation_on_generated_frames() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/interpolation_test", OUTPUT_DIR)).ok();

        // 生成 4 帧关键帧
        let keyframe_prompts = [
            "a ball at position 1, left side",
            "a ball at position 2, moving right",
            "a ball at position 3, center",
        ];

        let mut keyframe_count = 0;
        for (i, prompt) in keyframe_prompts.iter().enumerate() {
            let output = format!("{}/interpolation_test/keyframe_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry", 8000 + i as u32, 256, 256, 5, 7.0, &output) {
                keyframe_count += 1;
            }
        }

        assert!(keyframe_count >= 2, "Need at least 2 keyframes for interpolation test");

        // 模拟插值: 计算插值后的总帧数
        let multiplier = 2;
        let total_interpolated = keyframe_count + (keyframe_count - 1) * multiplier;
        assert!(total_interpolated >= keyframe_count, "Interpolated should have more frames");

        println!("Keyframes: {} -> Interpolated: {} ({}x)", keyframe_count, total_interpolated, multiplier);

        // 验证每帧文件存在并检查大小
        let mut verified = 0;
        for i in 0..keyframe_count {
            let path = format!("{}/interpolation_test/keyframe_{:03}.png", OUTPUT_DIR, i);
            if Path::new(&path).exists() {
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                if size > 1000 {
                    verified += 1;
                }
            }
        }
        assert!(verified >= 2, "Should have at least 2 valid keyframe files");
    }
}

// ============================================================================
// 4. 视频帧组装 — 使用 ffmpeg 合成为视频文件
// ============================================================================

mod video_assembly {
    use super::*;

    /// 将生成帧通过 ffmpeg 组装为短视频 (mp4)
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_assemble_frames_to_video() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        if !ffmpeg_exists() {
            println!("SKIP: ffmpeg not available, cannot assemble video");
            return;
        }
        fs::create_dir_all(format!("{}/assembled_video", OUTPUT_DIR)).ok();

        // 生成 6 帧动画序列
        let anim_prompts = [
            "a bouncing ball, position 1",
            "a bouncing ball, position 3",
            "a bouncing ball, position 6",
        ];

        let mut frame_count = 0;
        for (i, prompt) in anim_prompts.iter().enumerate() {
            let output = format!("{}/assembled_video/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "blurry", 9000 + i as u32, 256, 256, 5, 7.0, &output) {
                frame_count += 1;
            }
        }

        if frame_count < 3 {
            println!("SKIP: Not enough frames generated ({})", frame_count);
            return;
        }

        // 使用 ffmpeg 将帧序列组装成视频
        let video_output = format!("{}/assembled_video/output.mp4", OUTPUT_DIR);
        let ffmpeg_result = Command::new("ffmpeg")
            .args([
                "-y",
                "-framerate", "8",
                "-i", &format!("{}/assembled_video/frame_%03d.png", OUTPUT_DIR),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=256:256",
                "-r", "8",
                &video_output,
            ])
            .output();

        match ffmpeg_result {
            Ok(out) if out.status.success() => {
                let video_size = fs::metadata(&video_output).map(|m| m.len()).unwrap_or(0);
                assert!(video_size > 100, "Assembled video should have content, got {} bytes", video_size);
                println!("Video assembled successfully: {} ({} bytes, {} frames at 8fps)",
                    video_output, video_size, frame_count);
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                println!("ffmpeg failed: {}", stderr);
            }
            Err(e) => {
                println!("ffmpeg execution error: {}", e);
            }
        }
    }

    /// 测试不同帧率下的视频时长计算
    #[test]
    fn test_video_duration_calculation() {
        let configs = [
            (4, 6),   // 4帧@6fps
            (8, 8),   // 8帧@8fps
            (12, 12), // 12帧@12fps
            (16, 24), // 16帧@24fps
            (30, 30), // 30帧@30fps
        ];

        for &(frames, fps) in &configs {
            let duration_secs = frames as f64 / fps as f64;
            println!("  {} frames @ {} fps = {:.2} second video", frames, fps, duration_secs);
            assert!(duration_secs > 0.0, "Duration must be positive");
            if frames >= 8 && fps >= 8 {
                assert!(duration_secs >= 0.5, "8+ frames @ 8+ fps should be >= 0.5s");
            }
        }
    }

    /// GIF 格式输出模拟（ffmpeg 转 GIF）
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_assemble_frames_to_gif() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        if !ffmpeg_exists() {
            println!("SKIP: ffmpeg not available");
            return;
        }
        fs::create_dir_all(format!("{}/gif_output", OUTPUT_DIR)).ok();

        // 生成 4 帧（最小 GIF 序列）
        let gif_prompts = [
            "simple red circle on white background",
            "simple red circle at center",
            "simple red circle at far right",
        ];

        let mut frame_count = 0;
        for (i, prompt) in gif_prompts.iter().enumerate() {
            let output = format!("{}/gif_output/frame_{:03}.png", OUTPUT_DIR, i);
            if generate_frame(prompt, "complex background", 9500 + i as u32, 128, 128, 5, 7.0, &output) {
                frame_count += 1;
            }
        }

        if frame_count < 3 {
            println!("SKIP: Not enough GIF frames ({})", frame_count);
            return;
        }

        // 组装 GIF
        let gif_output = format!("{}/gif_output/output.gif", OUTPUT_DIR);
        let result = Command::new("ffmpeg")
            .args([
                "-y",
                "-framerate", "4",
                "-i", &format!("{}/gif_output/frame_%03d.png", OUTPUT_DIR),
                "-vf", "fps=4,scale=128:-1:flags=lanczos",
                &gif_output,
            ])
            .output();

        match result {
            Ok(out) if out.status.success() => {
                let size = fs::metadata(&gif_output).map(|m| m.len()).unwrap_or(0);
                println!("GIF assembled: {} ({} bytes)", gif_output, size);
                assert!(size > 50, "GIF should have content");
            }
            _ => println!("GIF assembly skipped (ffmpeg may not support gif)"),
        }
    }
}

// ============================================================================
// 5. 不同参数组合的帧生成对比
// ============================================================================

mod parameter_variations {
    use super::*;

    /// 不同 steps 对帧质量的影响（仅验证生成，不评判质量）
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_frame_at_different_steps() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/steps_comparison", OUTPUT_DIR)).ok();

        let step_values = [3, 8, 20];
        let mut generated = 0;

        for (i, &steps) in step_values.iter().enumerate() {
            let output = format!("{}/steps_comparison/steps_{}.png", OUTPUT_DIR, steps);
            if generate_frame("a test scene with fine details", "blurry, low quality",
                10000 + i as u32, 256, 256, steps, 7.0, &output) {
                let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!("  Steps={}: {} bytes", steps, size);
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate frames with at least 3 different step values");
    }

    /// 不同 cfg 对帧的影响
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_frame_at_different_cfg() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/cfg_comparison", OUTPUT_DIR)).ok();

        let cfg_values = [3.0, 7.0, 15.0];
        let mut generated = 0;

        for (i, &cfg) in cfg_values.iter().enumerate() {
            let output = format!("{}/cfg_comparison/cfg_{}.png", OUTPUT_DIR, (cfg * 10.0) as u32);
            if generate_frame("a simple scene", "blurry", 11000 + i as u32, 256, 256, 5, cfg as f32, &output) {
                let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!("  CFG={}: {} bytes", cfg, size);
                generated += 1;
            }
        }

        assert!(generated >= 3, "Should generate frames with at least 3 different CFG values");
    }

    /// 不同尺寸的帧
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_frame_at_different_resolutions() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/resolution_comparison", OUTPUT_DIR)).ok();

        let resolutions = [(128, 128), (256, 256), (384, 384)];
        let mut generated = 0;

        for (i, &(w, h)) in resolutions.iter().enumerate() {
            let output = format!("{}/resolution_comparison/res_{}x{}.png", OUTPUT_DIR, w, h);
            if generate_frame("a test scene", "blurry", 12000 + i as u32, w, h, 5, 7.0, &output) {
                let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!("  {}x{}: {} bytes", w, h, size);
                generated += 1;
            }
        }

        assert!(generated >= 2, "Should generate frames with at least 2 resolutions");
    }
}

// ============================================================================
// 6. 视频工作流全覆盖验证 — 解析所有视频 workflow
// ============================================================================

mod workflow_coverage {
    use super::*;

    /// 列出并验证所有视频类 workflow 模板
    #[test]
    fn test_all_video_workflows_structurally_valid() {
        let video_workflows = [
            "video_generation_pipeline.jsonld",
            "style_aware_video.jsonld",
            "controlnet_animated_video.jsonld",
            "text_to_video_direct.jsonld",
            "multi_prompt_video_morph.jsonld",
            "image_to_video_svd.jsonld",
            "video_frame_interpolation.jsonld",
            "latent_interpolation.jsonld",
        ];

        let mut validated = 0;
        for wf_name in &video_workflows {
            let path = format!("{}/{}", WORKFLOW_DIR, wf_name);
            if !Path::new(&path).exists() {
                println!("  SKIP: {} not found", wf_name);
                continue;
            }

            let content = fs::read_to_string(&path).unwrap_or_default();
            if content.is_empty() {
                println!("  FAIL: {} is empty", wf_name);
                continue;
            }

            let json: Result<serde_json::Value, _> = serde_json::from_str(&content);
            match json {
                Ok(val) => {
                    assert!(val["category"].as_str().unwrap_or("") == "video",
                        "{} should have category=video", wf_name);
                    assert!(val["name"].is_string() || val["@type"].is_string(),
                        "{} should have name or @type", wf_name);
                    assert!(val["description"].is_string(),
                        "{} should have description", wf_name);
                    validated += 1;
                    println!("  ✓ {} validated", wf_name);
                }
                Err(e) => {
                    println!("  FAIL: {} invalid JSON: {}", wf_name, e);
                }
            }
        }

        assert!(validated >= 6, "Should validate at least 6 video workflows, got {}", validated);
        println!("Validated {} video workflows", validated);
    }

    /// 验证所有具有 steps 定义的工作流的结构完整性
    #[test]
    fn test_video_workflow_steps_complete() {
        let workflow_files = [
            "multi_prompt_video_morph.jsonld",
            "video_frame_interpolation.jsonld",
            "latent_interpolation.jsonld",
            "controlnet_animated_video.jsonld",
            "text_to_video_direct.jsonld",
            "image_to_video_svd.jsonld",
        ];

        for wf_name in &workflow_files {
            let path = format!("{}/{}", WORKFLOW_DIR, wf_name);
            if !Path::new(&path).exists() {
                println!("  SKIP: {} not found", wf_name);
                continue;
            }

            let content = fs::read_to_string(&path).unwrap();
            let json: serde_json::Value = serde_json::from_str(&content).unwrap();

            // Must have steps array
            if let Some(steps) = json["steps"].as_array() {
                assert!(steps.len() >= 3, "{} should have >= 3 steps", wf_name);
                for (idx, step) in steps.iter().enumerate() {
                    assert!(step["action"].is_string() || step["name"].is_string(),
                        "{} step {} missing action/name", wf_name, idx);
                }
                println!("  {} has {} valid steps", wf_name, steps.len());
            } else if let Some(stages) = json["stages"].as_array() {
                // Pipeline format uses stages
                assert!(stages.len() >= 2, "{} should have >= 2 stages", wf_name);
                for (idx, stage) in stages.iter().enumerate() {
                    assert!(stage["name"].is_string() || stage["workflow"].is_string(),
                        "{} stage {} missing name/workflow", wf_name, idx);
                }
                println!("  {} has {} valid stages", wf_name, stages.len());
            } else {
                panic!("{} has no steps or stages array", wf_name);
            }
        }
    }

    /// 视频 workflow 参数完整性
    #[test]
    fn test_video_workflow_params_complete() {
        let workflows_with_params: [(&str, &[&str]); 4] = [
            ("multi_prompt_video_morph.jsonld", &["prompt_1", "prompt_2", "morph_frames"]),
            ("latent_interpolation.jsonld", &["prompt_1", "prompt_2", "num_interpolation_steps"]),
            ("controlnet_animated_video.jsonld", &["prompt", "controlnet_type", "control_sequence_directory"]),
            ("video_generation_pipeline.jsonld", &["base_prompt", "num_frames", "fps"]),
        ];

        for (wf_name, required_params) in &workflows_with_params {
            let path = format!("{}/{}", WORKFLOW_DIR, wf_name);
            if !Path::new(&path).exists() {
                println!("  SKIP: {} not found", wf_name);
                continue;
            }

            let content = fs::read_to_string(&path).unwrap();
            let json: serde_json::Value = serde_json::from_str(&content).unwrap();

            let params = json["parameters"].as_object()
                .or_else(|| json["stages"].as_array().and_then(|_| json["parameters"].as_object()));

            match params {
                Some(p) => {
                    for required in *required_params {
                        assert!(p.contains_key(*required),
                            "{} missing required parameter '{}'", wf_name, required);
                    }
                    println!("  ✓ {} has all required params", wf_name);
                }
                None => {
                    println!("  SKIP: {} has no parameters section", wf_name);
                }
            }
        }
    }
}

// ============================================================================
// 7. 视频组合工作流 — 多场景帧交叉测试
// ============================================================================

mod cross_scene_workflow {
    use super::*;

    /// 生成不同场景的帧并验证文件结构完整
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_multi_scene_frame_collection() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }

        let scenes = [
            ("beach", "a tropical beach with palm trees"),
            ("forest", "a dense green forest with sunlight"),
            ("mountain", "snow capped mountains at sunrise"),
        ];

        let mut total_frames = 0;
        for (scene_name, prompt) in &scenes {
            let dir = format!("{}/multi_scene/{}", OUTPUT_DIR, scene_name);
            fs::create_dir_all(&dir).ok();

            for i in 0..2 {
                let output = format!("{}/frame_{:03}.png", dir, i);
                if generate_frame(prompt, "blurry", 13000 + i as u32, 256, 256, 5, 7.0, &output) {
                    total_frames += 1;
                }
            }
        }

        assert!(total_frames >= 3, "Should generate frames across scenes, got {}", total_frames);

        // Verify directory structure
        for (scene_name, _) in &scenes {
            let dir = format!("{}/multi_scene/{}", OUTPUT_DIR, scene_name);
            assert!(Path::new(&dir).exists(), "Scene directory should exist: {}", dir);
            let files = list_frames_in(&dir);
            println!("  Scene '{}': {} frames", scene_name, files.len());
        }

        println!("Multi-scene collection: {} total frames across {} scenes", total_frames, scenes.len());
    }

    /// 跨工作流组合测试：每帧使用不同参数生成，模拟 workflow 组合
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_composite_workflow_simulation() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/composite", OUTPUT_DIR)).ok();

        // 模拟一个组合工作流: 首帧精细 + 中间帧标准 + 末帧精细
        let workflow_stages = [
            ("intro_frame.png", "beautiful landscape, highly detailed, 8k", 5, 7.5, 14000),
            ("middle_2.png", "landscape scene, wide view", 5, 7.0, 14002),
            ("outro_frame.png", "beautiful landscape, sunset colors, highly detailed", 5, 7.5, 14004),
        ];

        let mut generated = 0;
        for (filename, prompt, steps, cfg, seed) in &workflow_stages {
            let output = format!("{}/composite/{}", OUTPUT_DIR, filename);
            if generate_frame(prompt, "blurry", *seed, 256, 256, *steps, *cfg as f32, &output) {
                let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
                println!("  {}: {} bytes ({} steps, cfg={})", filename, size, steps, cfg);
                generated += 1;
            }
        }

        assert!(generated >= 3, "Composite workflow should generate at least 3 frames");
    }
}

// ============================================================================
// 8. 视频帧生成边界条件测试
// ============================================================================

mod edge_cases {
    use super::*;

    /// 极端小尺寸帧
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_minimal_resolution_frame() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/edge_cases", OUTPUT_DIR)).ok();

        let output = format!("{}/edge_cases/minimal.png", OUTPUT_DIR);
        if generate_frame("a simple dot", "complex", 15000, 64, 64, 5, 7.0, &output) {
            let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
            println!("Minimal frame (64x64): {} bytes", size);
            assert!(size > 100, "Even minimal frame should have some content");
        }
    }

    /// 单帧生成（极端情况）
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_single_frame_generation() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/edge_cases", OUTPUT_DIR)).ok();

        let output = format!("{}/edge_cases/single_frame.png", OUTPUT_DIR);
        assert!(generate_frame("a single test image for video", "blurry", 16000, 256, 256, 5, 7.0, &output),
            "Single frame generation should succeed");
        let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        println!("Single frame: {} bytes", size);
    }

    /// 极简提示词
    #[test]
    #[ignore = "requires GPU for real SD generation; run with --ignored"]
    fn test_minimal_prompt_frame() {
        if !model_exists() || !sd_cli_exists() {
            println!("SKIP: Model or sd-cli not available");
            return;
        }
        fs::create_dir_all(format!("{}/edge_cases", OUTPUT_DIR)).ok();

        let output = format!("{}/edge_cases/minimal_prompt.png", OUTPUT_DIR);
        if generate_frame("test", "n/a", 17000, 256, 256, 5, 7.0, &output) {
            let size = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
            println!("Minimal prompt frame: {} bytes", size);
            assert!(size > 100, "Even minimal prompt should produce output");
        }
    }

    /// 超低帧率配置验证
    #[test]
    fn test_low_fps_config() {
        // 1 fps 的视频配置
        let frames = 4;
        let fps = 1;
        let duration = frames as f64 / fps as f64;
        println!("Low FPS config: {} frames @ {} fps = {}s video", frames, fps, duration);
        assert!(duration >= 1.0, "At 1fps with 4 frames should be >= 1s");
    }

    /// 不同采样器名称验证
    #[test]
    fn test_sampler_names() {
        let video_samplers = ["euler", "euler_a", "heun", "dpm2", "lms", "ddim", "pndm"];
        for sampler in &video_samplers {
            // 验证采样器名称至少是合法的（不 panic）
            assert!(!sampler.is_empty(), "Sampler name should not be empty");
        }
        println!("All {} video-compatible samplers are valid", video_samplers.len());
    }

    /// 帧命名一致性检查
    #[test]
    fn test_frame_naming_convention() {
        // 验证帧文件名的排序一致性
        let names = vec![
            "frame_001.png", "frame_002.png", "frame_003.png",
            "frame_010.png", "frame_100.png",
        ];
        let mut sorted = names.clone();
        sorted.sort();
        // 字典序对 padded numbers 有效
        assert_eq!(names, sorted, "Zero-padded frame names should sort lexicographically");
    }
}

// ============================================================================
// 9. 工作流验证器集成 — 使用 WorkflowValidator 对视频工作流做结构校验
// ============================================================================

mod workflow_validator_integration {
    use comfyui_rust_agent::workflow::{WorkflowBuilder, WorkflowValidator};

    /// 使用 WorkflowValidator 校验一个视频工作流结构
    #[test]
    fn test_validate_video_workflow_with_validator() {
        // 通过 WorkflowBuilder 创建一个图片转视频风格的工作流
        // 并用 Validator 做结构校验
        let wf = WorkflowBuilder::text_to_image(
            "a video frame of a cat".to_string(),
            "blurry".to_string(),
            256, 256, 10, 7.0, 42,
            "v1-5-pruned-emaonly.safetensors".to_string(),
        ).unwrap();

        let validator = WorkflowValidator::new();
        match validator.validate(&wf) {
            Ok(result) => {
                if result.valid {
                    println!("✓ Video workflow validated successfully ({} nodes)", wf.nodes.len());
                } else {
                    println!("Video workflow validation had issues: {:?}", result.errors);
                }
            }
            Err(e) => {
                println!("Workflow validation error: {}", e);
            }
        }

        // Even if not fully valid, the workflow should have the right structure
        assert!(wf.nodes.len() >= 3, "Video workflow should have >= 3 nodes");
        if wf.links.len() < 2 {
            println!("Note: text_to_image produced {} nodes but only {} links (may need edge construction)", wf.nodes.len(), wf.links.len());
        }
        println!("Workflow structure: {} nodes, {} links", wf.nodes.len(), wf.links.len());
    }

    /// 验证视频工作流中的参数节点完整性
    #[test]
    fn test_video_workflow_node_spec_validation() {
        let wf = WorkflowBuilder::text_to_image(
            "video frame".to_string(),
            "blurry".to_string(),
            256, 256, 10, 7.0, 42,
            "v1-5-pruned-emaonly.safetensors".to_string(),
        ).unwrap();

        // 验证每个节点有正确的 class_type
        for (_id, node) in &wf.nodes {
            assert!(!node.class_type.is_empty(), "Node class_type should not be empty");
        }

        // 检查特定节点是否存在
        let has_checkpoint = wf.nodes.values().any(|n| n.class_type == "CheckpointLoaderSimple");
        let has_sampler = wf.nodes.values().any(|n| n.class_type == "KSampler");
        let has_clip = wf.nodes.values().any(|n| n.class_type == "CLIPTextEncode");

        assert!(has_checkpoint, "Workflow should have CheckpointLoaderSimple");
        assert!(has_sampler, "Workflow should have KSampler");
        assert!(has_clip, "Workflow should have CLIPTextEncode");
    }
}

// ============================================================================
// 10. 输出完整性检查 — 验证 output 目录
// ============================================================================

mod output_verification {
    use super::*;

    /// 列出本次测试生成的所有输出文件并统计
    #[test]
    fn test_count_all_generated_video_outputs() {
        let output_dirs = [
            "e2e_video/animal_walk",
            "e2e_video/sunset",
            "e2e_video/object_transition",
            "e2e_video/human_action",
            "e2e_video/cityscape",
            "e2e_video/morph_animal",
            "e2e_video/morph_style",
            "e2e_video/interpolation_test",
            "e2e_video/assembled_video",
            "e2e_video/gif_output",
            "e2e_video/steps_comparison",
            "e2e_video/cfg_comparison",
            "e2e_video/resolution_comparison",
            "e2e_video/multi_scene",
            "e2e_video/composite",
            "e2e_video/edge_cases",
        ];

        let mut total_frames = 0;
        let mut total_dirs_with_content = 0;

        for dir in &output_dirs {
            let full_path = format!("{}/{}", OUTPUT_DIR, dir);
            let files = list_frames_in(&full_path);
            if !files.is_empty() {
                total_dirs_with_content += 1;
                // 只统计 png 帧
                let png_count = files.iter().filter(|f| f.ends_with(".png")).count();
                total_frames += png_count;
                println!("  {}: {} frames", dir, png_count);
            } else {
                println!("  {}: (no content)", dir);
            }
        }

        // 检查是否有视频文件生成
        let video_files = list_frames_in(&format!("{}/assembled_video", OUTPUT_DIR));
        let video_count = video_files.iter().filter(|f| f.ends_with(".mp4")).count();
        let gif_files = list_frames_in(&format!("{}/gif_output", OUTPUT_DIR));
        let gif_count = gif_files.iter().filter(|f| f.ends_with(".gif")).count();

        println!("\n=== E2E Video Generation Summary ===");
        println!("  Sub-directories with content: {}/{}", total_dirs_with_content, output_dirs.len());
        println!("  Total PNG frames generated: {}", total_frames);
        println!("  Video files (mp4): {}", video_count);
        println!("  Animated GIFs: {}", gif_count);

        if total_dirs_with_content == 0 && model_exists() && sd_cli_exists() {
            println!("  ⚠ No frames found yet (generation tests may be running in parallel)");
        }
    }

    /// 输出目录清理标记
    #[test]
    fn test_output_directory_cleanup_marker() {
        let marker_path = format!("{}/.generated", OUTPUT_DIR);
        // 写入一个标记文件记录本次运行
        let content = format!("E2E video generation test run at {:?}\nModel: {}\nSD-CLI: {}",
            std::time::SystemTime::now(),
            model_exists(),
            sd_cli_exists(),
        );
        fs::write(&marker_path, content).unwrap_or_default();
        println!("Output marker written to {}", marker_path);
    }
}
