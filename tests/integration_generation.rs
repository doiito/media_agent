// 端到端集成测试 — 生成流程覆盖
// 测试范围：工作流构建→验证→执行→事件发布 完整链路
// 需要 sd-cli + 模型的测试用 #[ignore] 标记

use comfyui_rust_agent::types::*;
use comfyui_rust_agent::workflow::{WorkflowBuilder, WorkflowValidator};
use comfyui_rust_agent::execution::{EventBus, Event, ExecutionEngine, PromptExecutor};
use comfyui_rust_agent::backend::BackendRouter;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

const MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/v1-5-pruned-emaonly.safetensors";
const SD_CLI_PATH: &str = "/usr/local/bin/sd-cli";
const SVD_MODEL_PATH: &str = "/dev-data/ai-test/media_agent/models/checkpoints/svd.safetensors";
const OUTPUT_DIR: &str = "/dev-data/ai-test/media_agent/output/integration_e2e";

fn model_exists() -> bool { Path::new(MODEL_PATH).exists() }
fn sd_cli_exists() -> bool { Path::new(SD_CLI_PATH).exists() }

// ============================================================================
// 1. 工作流构建与验证测试（不需要 sd-cli）
// ============================================================================

#[test]
fn e2e_text_to_image_workflow_build() {
    let workflow = WorkflowBuilder::text_to_image(
        "a beautiful sunset over mountains".to_string(),
        "blurry, low quality".to_string(),
        512, 512, 20, 7.5, 42,
        "v1-5-pruned-emaonly.safetensors".to_string(),
    )
    .expect("Failed to build text-to-image workflow");

    assert!(!workflow.nodes.is_empty(), "Workflow should have nodes");
    assert!(workflow.nodes.len() >= 6, "T2I workflow should have at least 6 nodes");

    let class_types: Vec<&str> = workflow.nodes.values()
        .map(|n| n.class_type.as_str())
        .collect();
    assert!(class_types.contains(&"CheckpointLoaderSimple"), "Missing CheckpointLoaderSimple");
    assert!(class_types.contains(&"CLIPTextEncode"), "Missing CLIPTextEncode");
    assert!(class_types.contains(&"EmptyLatentImage"), "Missing EmptyLatentImage");
    assert!(class_types.contains(&"KSampler"), "Missing KSampler");
    assert!(class_types.contains(&"VAEDecode"), "Missing VAEDecode");
    assert!(class_types.contains(&"SaveImage"), "Missing SaveImage");

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).expect("Validation should not error");
    assert!(result.valid, "Workflow should be valid: {:?}", result.errors);
    assert!(!result.execution_order.is_empty(), "Should have execution order");

    println!("T2I workflow: {} nodes, execution order: {} steps",
             workflow.nodes.len(), result.execution_order.len());
}

#[test]
fn e2e_image_to_image_workflow_build() {
    let workflow = WorkflowBuilder::image_to_image(
        "transform to watercolor style".to_string(),
        "blurry".to_string(),
        "/dev-data/ai-test/media_agent/test_input.png".to_string(),
        0.65, 20, 7.5, 42,
        "v1-5-pruned-emaonly.safetensors".to_string(),
    )
    .expect("Failed to build image-to-image workflow");

    assert!(!workflow.nodes.is_empty());
    assert!(workflow.nodes.len() >= 6, "I2I workflow should have at least 6 nodes");

    let class_types: Vec<&str> = workflow.nodes.values()
        .map(|n| n.class_type.as_str())
        .collect();
    assert!(class_types.contains(&"LoadImage"), "Missing LoadImage node");
    assert!(class_types.contains(&"KSampler"), "Missing KSampler");

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).expect("Validation failed");
    assert!(result.valid, "I2I workflow should be valid: {:?}", result.errors);

    println!("I2I workflow: {} nodes valid", workflow.nodes.len());
}

#[test]
fn e2e_image_to_video_workflow_build() {
    let workflow = WorkflowBuilder::image_to_video(
        "/dev-data/ai-test/media_agent/test_input.png".to_string(),
        "svd.safetensors".to_string(),
        14, 6, 127, 2.5, 20, 42,
    )
    .expect("Failed to build image-to-video workflow");

    assert!(!workflow.nodes.is_empty(), "I2V workflow should have nodes");

    let class_types: Vec<&str> = workflow.nodes.values()
        .map(|n| n.class_type.as_str())
        .collect();
    assert!(class_types.contains(&"SVDImageToVideo"), "Missing SVDImageToVideo node");

    println!("I2V workflow: {} nodes, class types: {:?}", workflow.nodes.len(), class_types);
}

#[test]
fn e2e_text_to_video_workflow_build() {
    let workflow = WorkflowBuilder::text_to_video(
        "a cat walking in garden".to_string(),
        "blurry".to_string(),
        256, 256, 8, 4, 15, 7.0, 42,
        "v1-5-pruned-emaonly.safetensors".to_string(),
    )
    .expect("Failed to build text-to-video workflow");

    assert!(!workflow.nodes.is_empty(), "T2V workflow should have nodes");

    let class_types: Vec<&str> = workflow.nodes.values()
        .map(|n| n.class_type.as_str())
        .collect();
    println!("T2V workflow: {} nodes, class types: {:?}", workflow.nodes.len(), class_types);
}

// ============================================================================
// 2. 事件总线测试（不需要 sd-cli）
// ============================================================================

#[tokio::test]
async fn e2e_event_bus_execution_events() {
    let bus = EventBus::new();
    let mut rx = bus.subscribe("test-client".to_string()).await;

    bus.publish(Event::ExecutionStart {
        prompt_id: "test-prompt-1".to_string(),
    }).await;

    bus.publish(Event::Executing {
        prompt_id: "test-prompt-1".to_string(),
        node_id: Some("1".to_string()),
    }).await;

    bus.publish(Event::Progress {
        prompt_id: "test-prompt-1".to_string(),
        value: 5,
        max: 20,
    }).await;

    bus.publish(Event::ExecutionSuccess {
        prompt_id: "test-prompt-1".to_string(),
        outputs: std::collections::HashMap::new(),
    }).await;

    let mut received = Vec::new();
    while let Ok(event) = rx.try_recv() {
        received.push(event);
    }

    assert_eq!(received.len(), 4, "Should receive all 4 events");
    assert!(matches!(received[0], Event::ExecutionStart { .. }));
    assert!(matches!(received[1], Event::Executing { .. }));
    assert!(matches!(received[2], Event::Progress { .. }));
    assert!(matches!(received[3], Event::ExecutionSuccess { .. }));

    println!("Event bus: received {} events in order", received.len());
}

#[tokio::test]
async fn e2e_event_bus_agent_pdca_events() {
    let bus = EventBus::new();
    let mut rx = bus.subscribe("agent-client".to_string()).await;

    let phases = ["planning", "doing", "checking", "acting"];
    for phase in &phases {
        bus.publish(Event::AgentPhaseStart {
            prompt_id: "pdca-1".to_string(),
            phase: phase.to_string(),
            description: format!("Starting {} phase", phase),
        }).await;

        bus.publish(Event::AgentPhaseComplete {
            prompt_id: "pdca-1".to_string(),
            phase: phase.to_string(),
            success: true,
        }).await;
    }

    bus.publish(Event::AgentThought {
        prompt_id: "pdca-1".to_string(),
        thought: "Need to generate a landscape image".to_string(),
        action: "text_to_image".to_string(),
    }).await;

    bus.publish(Event::AgentToolCall {
        prompt_id: "pdca-1".to_string(),
        tool_name: "create_node".to_string(),
        status: "completed".to_string(),
        result_summary: "Created KSampler node".to_string(),
    }).await;

    let mut received = Vec::new();
    while let Ok(event) = rx.try_recv() {
        received.push(event);
    }

    assert_eq!(received.len(), 10, "Should receive 8 PDCA + 2 agent events");

    let phase_starts = received.iter()
        .filter(|e| matches!(e, Event::AgentPhaseStart { .. }))
        .count();
    let phase_completes = received.iter()
        .filter(|e| matches!(e, Event::AgentPhaseComplete { .. }))
        .count();
    assert_eq!(phase_starts, 4, "Should have 4 phase start events");
    assert_eq!(phase_completes, 4, "Should have 4 phase complete events");

    println!("Agent PDCA events: {} total, {} starts, {} completes",
             received.len(), phase_starts, phase_completes);
}

#[tokio::test]
async fn e2e_event_bus_multiple_subscribers() {
    let bus = EventBus::new();
    let mut rx1 = bus.subscribe("client-1".to_string()).await;
    let mut rx2 = bus.subscribe("client-2".to_string()).await;

    assert_eq!(bus.subscriber_count().await, 2);

    bus.publish(Event::ExecutionStart {
        prompt_id: "multi-sub".to_string(),
    }).await;

    let recv1 = rx1.try_recv().is_ok();
    let recv2 = rx2.try_recv().is_ok();
    assert!(recv1, "Client 1 should receive event");
    assert!(recv2, "Client 2 should receive event");

    bus.unsubscribe("client-1").await;
    assert_eq!(bus.subscriber_count().await, 1);

    println!("Multi-subscriber: both received, unsubscribe works");
}

// ============================================================================
// 3. 执行引擎事件流测试（不需要 sd-cli）
// ============================================================================

#[tokio::test]
async fn e2e_execution_engine_event_flow() {
    let mut engine = ExecutionEngine::new();
    let mut rx = engine.subscribe("flow-test".to_string()).await;

    let workflow = WorkflowBuilder::text_to_image(
        "test prompt".to_string(),
        "".to_string(),
        256, 256, 5, 7.0, 1,
        "test-model.safetensors".to_string(),
    ).unwrap();

    let prompt_id = engine.submit(workflow, "flow-test".to_string()).await
        .expect("Submit should succeed");

    let result = engine.execute_next().await
        .expect("Execute should not error");

    assert!(result.is_some(), "Should have a result");

    let mut received = Vec::new();
    while let Ok(event) = rx.try_recv() {
        received.push(event);
    }

    assert!(received.len() >= 2, "Should receive at least start and completion event");
    assert!(matches!(received[0], Event::ExecutionStart { .. }),
           "First event should be ExecutionStart");

    let has_success = received.iter().any(|e| matches!(e, Event::ExecutionSuccess { .. }));
    let has_error = received.iter().any(|e| matches!(e, Event::ExecutionError { .. }));
    assert!(has_success || has_error,
           "Should have either success or error event, got: {:?}", received);

    println!("Execution engine flow: prompt_id={}, {} events received",
             prompt_id, received.len());
}

#[tokio::test]
async fn e2e_execution_engine_empty_queue() {
    let mut engine = ExecutionEngine::new();

    let result = engine.execute_next().await
        .expect("Should not error on empty queue");
    assert!(result.is_none(), "Should return None for empty queue");
}

// ============================================================================
// 4. PromptExecutor 无后端执行测试（不需要 sd-cli）
// ============================================================================

#[tokio::test]
async fn e2e_workflow_execution_no_backend() {
    let router = Arc::new(BackendRouter::new());
    let mut executor = PromptExecutor::with_backend(router);

    let workflow = WorkflowBuilder::text_to_image(
        "a test image".to_string(),
        "".to_string(),
        256, 256, 5, 7.0, 42,
        "nonexistent-model.safetensors".to_string(),
    ).unwrap();

    let result = executor.execute(&workflow).await;

    match &result {
        Ok(ExecutionResult::Failure(msg)) => {
            println!("Expected failure without backend: {}", msg);
        }
        Ok(ExecutionResult::Success(_)) => {
            println!("Unexpected success without backend");
        }
        Ok(ExecutionResult::Pending) => {
            println!("Pending result");
        }
        Err(e) => {
            println!("Error during execution (expected without backend): {:?}", e);
        }
    }

    assert!(result.is_ok(), "Execute should return Ok with result, not Err");
}

// ============================================================================
// 5. 真实生成测试（需要 sd-cli + 模型，用 #[ignore] 标记）
// ============================================================================

#[tokio::test]
#[ignore = "requires GPU for real SD generation; run with --ignored"]
async fn e2e_text_to_image_with_sd_cli() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();
    let output_path = format!("{}/e2e_t2i_{}.png", OUTPUT_DIR, chrono::Utc::now().timestamp());

    let mut cmd = Command::new(SD_CLI_PATH);
    cmd.args([
        "-m", MODEL_PATH,
        "-p", "a beautiful landscape with mountains and a lake at sunset",
        "-n", "blurry, low quality, distorted",
        "-W", "512", "-H", "512",
        "--steps", "20",
        "--cfg-scale", "7.5",
        "-s", "42",
        "--sampling-method", "euler",
        "-o", &output_path,
        "--backend", "cpu",
    ]);

    let output = cmd.output().expect("Failed to execute sd-cli");
    assert!(output.status.success(),
            "sd-cli failed: {}", String::from_utf8_lossy(&output.stderr));

    let image_data = std::fs::read(&output_path)
        .expect("Failed to read generated image");
    assert!(image_data.len() > 1000, "Generated image too small: {} bytes", image_data.len());

    println!("E2E T2I: generated {} bytes at {}", image_data.len(), output_path);
}

#[tokio::test]
#[ignore = "requires GPU and SVD model; run with --ignored"]
async fn e2e_image_to_video_with_sd_cli() {
    let svd_exists = Path::new(SVD_MODEL_PATH).exists();
    if !model_exists() || !sd_cli_exists() || !svd_exists {
        println!("SKIP: SVD model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let input_image = format!("{}/e2e_t2i_input.png", OUTPUT_DIR);
    let output_path = format!("{}/e2e_i2v_{}.mp4", OUTPUT_DIR, chrono::Utc::now().timestamp());

    let mut gen_cmd = Command::new(SD_CLI_PATH);
    gen_cmd.args([
        "-m", MODEL_PATH,
        "-p", "a scenic view for video generation",
        "-W", "256", "-H", "256",
        "--steps", "10",
        "--cfg-scale", "7.0",
        "-s", "100",
        "--sampling-method", "euler",
        "-o", &input_image,
        "--backend", "cpu",
    ]);

    let gen_output = gen_cmd.output().expect("Failed to generate input image");
    if !gen_output.status.success() {
        println!("SKIP: Failed to generate input image for I2V");
        return;
    }

    let mut svd_cmd = Command::new(SD_CLI_PATH);
    svd_cmd.args([
        "-m", SVD_MODEL_PATH,
        "-i", &input_image,
        "-W", "256", "-H", "256",
        "--steps", "15",
        "--cfg-scale", "2.5",
        "-s", "42",
        "--sampling-method", "euler",
        "-o", &output_path,
        "--backend", "cpu",
    ]);

    let svd_output = svd_cmd.output().expect("Failed to execute SVD");
    assert!(svd_output.status.success(),
            "SVD failed: {}", String::from_utf8_lossy(&svd_output.stderr));

    let video_data = std::fs::read(&output_path)
        .expect("Failed to read generated video");
    assert!(video_data.len() > 100, "Generated video too small: {} bytes", video_data.len());

    println!("E2E I2V: generated {} bytes at {}", video_data.len(), output_path);
}

#[tokio::test]
#[ignore = "requires GPU for real SD generation; run with --ignored"]
async fn e2e_batch_generation_with_sd_cli() {
    if !model_exists() || !sd_cli_exists() {
        println!("SKIP: Model or sd-cli not available");
        return;
    }

    std::fs::create_dir_all(OUTPUT_DIR).ok();

    let prompts = [
        ("red rose garden", 101),
        ("blue ocean waves", 102),
        ("green forest path", 103),
    ];

    let mut generated = 0;
    for (prompt, seed) in &prompts {
        let output_path = format!("{}/e2e_batch_{}.png", OUTPUT_DIR, seed);

        let mut cmd = Command::new(SD_CLI_PATH);
        cmd.args([
            "-m", MODEL_PATH,
            "-p", prompt,
            "-W", "256", "-H", "256",
            "--steps", "10",
            "--cfg-scale", "7.0",
            "-s", &seed.to_string(),
            "--sampling-method", "euler",
            "-o", &output_path,
            "--backend", "cpu",
        ]);

        if cmd.output().map(|o| o.status.success()).unwrap_or(false) {
            let data = std::fs::read(&output_path).unwrap_or_default();
            if data.len() > 100 {
                generated += 1;
                println!("  Batch: '{}' → {} bytes", prompt, data.len());
            }
        }
    }

    assert!(generated >= 1, "Should generate at least 1 image, got {}", generated);
    println!("E2E batch: {}/{} images generated", generated, prompts.len());
}

// ============================================================================
// 6. 完整工作流链路测试（构建→验证→执行→事件，不需要 sd-cli）
// ============================================================================

#[tokio::test]
async fn e2e_full_workflow_chain_no_backend() {
    let workflow = WorkflowBuilder::text_to_image(
        "integration test prompt".to_string(),
        "negative prompt".to_string(),
        256, 256, 5, 7.0, 42,
        "test.safetensors".to_string(),
    ).expect("Build workflow");

    let validator = WorkflowValidator::new();
    let validation = validator.validate(&workflow).expect("Validate workflow");
    assert!(validation.valid, "Workflow validation failed: {:?}", validation.errors);

    let router = Arc::new(BackendRouter::new());
    let executor = PromptExecutor::with_backend(router);

    let mut engine = ExecutionEngine::new();
    let mut rx = engine.subscribe("chain-test".to_string()).await;

    let prompt_id = engine.submit(workflow, "chain-test".to_string()).await
        .expect("Submit workflow");

    let result = engine.execute_next().await
        .expect("Execute workflow");
    assert!(result.is_some(), "Should have execution result");

    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    assert!(events.len() >= 2, "Should receive start + completion events");
    assert!(matches!(events[0], Event::ExecutionStart { .. }));

    let _ = executor;
    let _ = prompt_id;

    println!("Full chain: build → validate → submit → execute → {} events", events.len());
}
