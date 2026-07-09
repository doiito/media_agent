// 端到端集成测试
// 测试完整的请求处理流程、配置加载、监控、节点执行

use comfyui_rust_agent::api::error::ErrorContext;
use comfyui_rust_agent::backend::{BackendRouter, SdCppConfig, LlamaCppConfig};
use comfyui_rust_agent::config::AppConfig;
use comfyui_rust_agent::monitor::Monitor;
use comfyui_rust_agent::node::core_nodes::*;
use comfyui_rust_agent::node::Node;
use comfyui_rust_agent::types::*;
use comfyui_rust_agent::workflow::{WorkflowBuilder, WorkflowValidator};
use std::collections::HashMap;

#[tokio::test]
async fn test_end_to_end_workflow_pipeline() {
    // 模拟工作流执行流程，跳过 KSampler（需要后端）和 CheckpointLoader 的实际加载

    // 1. 创建 EmptyLatentImage 节点
    let mut empty_latent = EmptyLatentImageNode::new();
    let mut latent_inputs = HashMap::new();
    latent_inputs.insert("width".to_string(), Value::Int(512));
    latent_inputs.insert("height".to_string(), Value::Int(512));
    latent_inputs.insert("batch_size".to_string(), Value::Int(1));

    let latent_output = empty_latent.execute(latent_inputs).await.expect("EmptyLatentImage failed");

    // 验证输出
    let latent_value = &latent_output["LATENT"];
    match latent_value {
        Value::Latent(data) => {
            assert_eq!(data.len(), 64 * 64 * 4);
        }
        _ => panic!("Expected Latent output"),
    }

    // 2. 创建 CLIPTextEncode 节点（正向）
    let mut clip_encode_pos = CLIPTextEncodeNode::new();
    let mut pos_inputs = HashMap::new();
    pos_inputs.insert("text".to_string(), Value::String("a beautiful landscape".to_string()));
    pos_inputs.insert("clip".to_string(), Value::Clip("clip-model".to_string()));

    let pos_output = clip_encode_pos.execute(pos_inputs).await.expect("CLIPTextEncode failed");
    assert!(pos_output.contains_key("CONDITIONING"));

    // 3. 创建 CLIPTextEncode 节点（负向）
    let mut clip_encode_neg = CLIPTextEncodeNode::new();
    let mut neg_inputs = HashMap::new();
    neg_inputs.insert("text".to_string(), Value::String("blurry, low quality".to_string()));
    neg_inputs.insert("clip".to_string(), Value::Clip("clip-model".to_string()));

    let neg_output = clip_encode_neg.execute(neg_inputs).await.expect("CLIPTextEncode failed");
    assert!(neg_output.contains_key("CONDITIONING"));

    // 4. VAE 解码（使用前面生成的 latent）
    let mut vae_decode = VAEDecodeNode::new();
    let mut vae_inputs = HashMap::new();
    vae_inputs.insert("samples".to_string(), latent_output["LATENT"].clone());
    vae_inputs.insert("vae".to_string(), Value::Vae("vae-model".to_string()));

    let image_output = vae_decode.execute(vae_inputs).await.expect("VAEDecode failed");
    assert!(image_output.contains_key("IMAGE"));

    let image_value = &image_output["IMAGE"];
    match image_value {
        Value::Image(data) => assert!(!data.is_empty()),
        _ => panic!("Expected Image output"),
    }
}

#[tokio::test]
async fn test_vae_encode_decode_roundtrip() {
    // 创建测试图像
    let original_image = vec![128u8; 512 * 512 * 3];

    // 1. VAE 编码
    let mut vae_encode = VAEEncodeNode::new();
    let mut encode_inputs = HashMap::new();
    encode_inputs.insert("pixels".to_string(), Value::Image(original_image.clone()));
    encode_inputs.insert("vae".to_string(), Value::Vae("vae-model".to_string()));

    let encode_output = vae_encode.execute(encode_inputs).await.expect("VAEEncode failed");

    let latent_value = &encode_output["LATENT"];
    match latent_value {
        Value::Latent(data) => assert!(!data.is_empty()),
        _ => panic!("Expected Latent output"),
    }

    // 2. VAE 解码（往返）
    let mut vae_decode = VAEDecodeNode::new();
    let mut decode_inputs = HashMap::new();
    decode_inputs.insert("samples".to_string(), latent_value.clone());
    decode_inputs.insert("vae".to_string(), Value::Vae("vae-model".to_string()));

    let decode_output = vae_decode.execute(decode_inputs).await.expect("VAEDecode failed");

    let decoded_image = &decode_output["IMAGE"];
    match decoded_image {
        Value::Image(data) => assert!(!data.is_empty()),
        _ => panic!("Expected Image output"),
    }
}

#[tokio::test]
async fn test_save_image_node() {
    let mut save_node = SaveImageNode::with_output_dir("/tmp/comfyui_test_save".to_string());

    let mut inputs = HashMap::new();
    inputs.insert(
        "images".to_string(),
        Value::Image(vec![200u8; 100]),
    );
    inputs.insert(
        "filename_prefix".to_string(),
        Value::String("integration_test".to_string()),
    );

    let result = save_node.execute(inputs).await;
    assert!(result.is_ok(), "SaveImage failed: {:?}", result.err());

    let output = result.unwrap();
    assert!(output.contains_key("filename"));
    assert!(output.contains_key("subfolder"));

    // 清理
    let _ = std::fs::remove_dir_all("/tmp/comfyui_test_save");
}

#[test]
fn test_config_and_monitor_integration() {
    // 测试配置和监控系统的集成
    let mut config = AppConfig::default();
    config.monitor.collect_interval_secs = 1;
    config.monitor.history_size = 10;

    let monitor = Monitor::new(config.monitor);

    // 在 tokio 运行时中执行异步操作
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // 采集多次
        for _ in 0..3 {
            monitor.collect().await;
        }

        let history = monitor.history(None).await;
        assert_eq!(history.len(), 3);

        let latest = monitor.latest().await;
        assert!(latest.is_some());

        let averages = monitor.averages().await;
        assert!(averages.is_some());
        let avg = averages.unwrap();
        assert_eq!(avg.sample_count, 3);
    });
}

#[tokio::test]
async fn test_backend_router_creation() {
    let sd_config = SdCppConfig::default();
    let llama_config = LlamaCppConfig::default();
    let router = BackendRouter::with_configs(sd_config, llama_config);

    // 健康检查应该在没启动后端时返回 false
    let healthy = router.health_check().await;
    // 不强制断言具体值，因为可能因环境而异
    let _ = healthy;
}

#[test]
fn test_error_context_integration() {
    let ctx = ErrorContext::new("execute_workflow", "ExecutionEngine")
        .with_extra("prompt_id", "test-123")
        .with_extra("node_count", "7");

    let original_err = Error::ExecutionFailed("KSampler failed".to_string());
    let wrapped = ctx.wrap_error(&original_err);

    let msg = wrapped.to_string();
    assert!(msg.contains("ExecutionEngine"));
    assert!(msg.contains("execute_workflow"));
    assert!(msg.contains("KSampler failed"));
    assert!(msg.contains("prompt_id=test-123"));
}

#[test]
fn test_full_config_lifecycle() {
    // 1. 创建配置
    let mut config = AppConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = 19999;
    config.server.output_dir = "/tmp/comfyui_config_test_output".to_string();
    config.log.level = "debug".to_string();

    // 2. 验证
    assert!(config.validate().is_ok());

    // 3. 创建目录
    assert!(config.ensure_directories().is_ok());

    // 4. 序列化
    let json = serde_json::to_string(&config).unwrap();

    // 5. 反序列化
    let parsed: AppConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.server.port, 19999);
    assert_eq!(parsed.log.level, "debug");

    // 清理
    let _ = std::fs::remove_dir_all("/tmp/comfyui_config_test_output");
}

#[tokio::test]
async fn test_checkpoint_loader_caching() {
    let mut loader = CheckpointLoaderNode::new();

    let mut inputs = HashMap::new();
    inputs.insert(
        "ckpt_name".to_string(),
        Value::String("test-model".to_string()),
    );

    // 第一次加载
    let result1 = loader.execute(inputs.clone()).await.unwrap();
    assert!(result1.contains_key("MODEL"));
    assert!(result1.contains_key("CLIP"));
    assert!(result1.contains_key("VAE"));

    // 第二次加载相同模型 - 应该使用缓存
    let result2 = loader.execute(inputs).await.unwrap();
    assert!(result2.contains_key("MODEL"));

    // 验证缓存一致性
    match (&result1["MODEL"], &result2["MODEL"]) {
        (Value::Model(p1), Value::Model(p2)) => assert_eq!(p1, p2),
        _ => panic!("Expected Model values"),
    }
}

#[tokio::test]
async fn test_clip_text_encode_caching() {
    let mut encoder = CLIPTextEncodeNode::new();

    let mut inputs = HashMap::new();
    inputs.insert("text".to_string(), Value::String("test prompt".to_string()));
    inputs.insert("clip".to_string(), Value::Clip("clip".to_string()));

    // 第一次编码
    let result1 = encoder.execute(inputs.clone()).await.unwrap();
    let cond1 = match &result1["CONDITIONING"] {
        Value::Conditioning(c) => c.clone(),
        _ => panic!("Expected Conditioning"),
    };

    // 第二次编码相同文本 - 应该返回相同结果
    let result2 = encoder.execute(inputs).await.unwrap();
    let cond2 = match &result2["CONDITIONING"] {
        Value::Conditioning(c) => c.clone(),
        _ => panic!("Expected Conditioning"),
    };

    // 缓存值应该相同
    assert_eq!(cond1, cond2);
}

#[test]
fn test_workflow_validation_with_all_nodes() {
    let workflow = WorkflowBuilder::text_to_image(
        "test".to_string(),
        "negative".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model.safetensors".to_string(),
    )
    .unwrap();

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).unwrap();

    assert!(result.valid);
    // 验证所有节点都在执行顺序中
    assert_eq!(result.execution_order.len(), 7);

    // 验证所有节点 ID 都在执行顺序中
    for node_id in workflow.nodes.keys() {
        assert!(
            result.execution_order.contains(node_id),
            "Node {} not in execution order",
            node_id
        );
    }
}

#[test]
fn test_value_methods() {
    // 测试 Value 的各种转换方法
    let s = Value::String("hello".to_string());
    assert_eq!(s.as_str().unwrap(), "hello");

    let i = Value::Int(42);
    assert_eq!(i.as_int().unwrap(), 42);
    assert_eq!(i.as_float().unwrap(), 42.0);

    let f = Value::Float(3.14);
    assert!((f.as_float().unwrap() - 3.14).abs() < 0.001);

    // 测试 as_ref_str
    let model = Value::Model("model-path".to_string());
    assert_eq!(model.as_ref_str().unwrap(), "model-path");

    let clip = Value::Clip("clip-path".to_string());
    assert_eq!(clip.as_ref_str().unwrap(), "clip-path");

    let vae = Value::Vae("vae-path".to_string());
    assert_eq!(vae.as_ref_str().unwrap(), "vae-path");

    // 测试错误情况
    let int_val = Value::Int(1);
    assert!(int_val.as_str().is_err());
    assert!(int_val.as_ref_str().is_err());

    let latent = Value::Latent(vec![1.0, 2.0, 3.0]);
    assert!(latent.as_str().is_err());
}

#[tokio::test]
async fn test_empty_latent_various_sizes() {
    let mut node = EmptyLatentImageNode::new();

    // 测试各种有效尺寸（必须是 8 的倍数）
    let valid_sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (512, 768)];

    for (width, height) in valid_sizes.iter() {
        let mut inputs = HashMap::new();
        inputs.insert("width".to_string(), Value::Int(*width as i64));
        inputs.insert("height".to_string(), Value::Int(*height as i64));
        inputs.insert("batch_size".to_string(), Value::Int(1));

        let result = node.execute(inputs).await;
        assert!(result.is_ok(), "Failed for size {}x{}", width, height);

        let output = result.unwrap();
        let latent = &output["LATENT"];
        if let Value::Latent(data) = latent {
            let expected = (width / 8) * (height / 8) * 4;
            assert_eq!(data.len(), expected);
        }
    }
}

#[tokio::test]
async fn test_empty_latent_invalid_sizes() {
    let mut node = EmptyLatentImageNode::new();

    let invalid_sizes = [(7, 512), (512, 7), (100, 100), (513, 513), (0, 0)];

    for (width, height) in invalid_sizes.iter() {
        let mut inputs = HashMap::new();
        inputs.insert("width".to_string(), Value::Int(*width as i64));
        inputs.insert("height".to_string(), Value::Int(*height as i64));
        inputs.insert("batch_size".to_string(), Value::Int(1));

        let result = node.execute(inputs).await;
        assert!(result.is_err(), "Should fail for invalid size {}x{}", width, height);
    }
}
