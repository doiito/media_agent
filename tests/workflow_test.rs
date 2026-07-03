// 工作流测试
// 测试工作流构建、验证、序列化

use comfyui_rust_agent::types::*;
use comfyui_rust_agent::workflow::{WorkflowBuilder, WorkflowValidator, WorkflowManager};

#[test]
fn test_text_to_image_workflow_creation() {
    let workflow = WorkflowBuilder::text_to_image(
        "a cat".to_string(),
        "blurry".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "sd-v1-5".to_string(),
    )
    .expect("Failed to create workflow");

    // 验证节点数量
    assert_eq!(workflow.nodes.len(), 7);

    // 验证关键节点存在
    assert!(workflow.nodes.contains_key("1"));
    assert!(workflow.nodes.contains_key("7"));

    // 验证 CheckpointLoaderSimple
    let loader = &workflow.nodes["1"];
    assert_eq!(loader.class_type, "CheckpointLoaderSimple");
    let ckpt_input = loader.inputs.get("ckpt_name").unwrap();
    match ckpt_input {
        InputValue::Direct(Value::String(s)) => assert_eq!(s, "sd-v1-5"),
        _ => panic!("Expected direct string value"),
    }

    // 验证 KSampler 连接
    let ksampler = &workflow.nodes["5"];
    assert_eq!(ksampler.class_type, "KSampler");
    let model_input = ksampler.inputs.get("model").unwrap();
    match model_input {
        InputValue::Link([node_id, slot]) => {
            assert_eq!(node_id, "1");
            assert_eq!(slot, "0");
        }
        _ => panic!("Expected link value"),
    }
}

#[test]
fn test_image_to_image_workflow_creation() {
    let workflow = WorkflowBuilder::image_to_image(
        "anime style".to_string(),
        "low quality".to_string(),
        "input.png".to_string(),
        0.75,
        25,
        8.0,
        100,
        "sd-v1-5".to_string(),
    )
    .expect("Failed to create i2i workflow");

    // 图生图应该有 8 个节点（比文生图多 LoadImage 和 VAEEncode）
    assert_eq!(workflow.nodes.len(), 8);

    // 验证 LoadImage 节点
    let load_image = &workflow.nodes["2"];
    assert_eq!(load_image.class_type, "LoadImage");
}

#[test]
fn test_workflow_validation_valid() {
    let workflow = WorkflowBuilder::text_to_image(
        "test".to_string(),
        "neg".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model".to_string(),
    )
    .unwrap();

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).expect("Validation failed");

    assert!(result.valid, "Workflow should be valid: {:?}", result.errors);
    assert_eq!(result.execution_order.len(), 7);
    assert!(result.missing_nodes.is_empty());
}

#[test]
fn test_workflow_validation_unknown_node() {
    let mut nodes = std::collections::HashMap::new();
    nodes.insert(
        "1".to_string(),
        WorkflowNode {
            class_type: "UnknownNodeType".to_string(),
            inputs: std::collections::HashMap::new(),
            pos: None,
            size: None,
            is_changed: None,
        },
    );

    let workflow = Workflow {
        nodes,
        links: vec![],
    };

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).unwrap();

    assert!(!result.valid);
    assert!(!result.missing_nodes.is_empty());
    assert!(!result.errors.is_empty());
}

#[test]
fn test_workflow_serialization_roundtrip() {
    let workflow = WorkflowBuilder::text_to_image(
        "test prompt".to_string(),
        "negative".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model.ckpt".to_string(),
    )
    .unwrap();

    let json = serde_json::to_string(&workflow).expect("Failed to serialize");
    let parsed: Workflow = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(workflow.nodes.len(), parsed.nodes.len());
    assert_eq!(workflow.nodes["1"].class_type, parsed.nodes["1"].class_type);
}

#[test]
fn test_workflow_manager() {
    let workflow = WorkflowManager::create_text_to_image_workflow(
        "prompt".to_string(),
        "neg".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model".to_string(),
    )
    .unwrap();

    let manager = WorkflowManager::new();
    let result = manager.validate(&workflow).unwrap();
    assert!(result.valid);
}

#[test]
fn test_workflow_topology_order() {
    // 拓扑顺序：CheckpointLoader(1) 必须在 CLIPTextEncode(2,3) 之前
    // EmptyLatentImage(4) 无依赖
    // KSampler(5) 依赖 1, 2, 3, 4
    // VAEDecode(6) 依赖 5, 1
    // SaveImage(7) 依赖 6
    let workflow = WorkflowBuilder::text_to_image(
        "p".to_string(),
        "n".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "m".to_string(),
    )
    .unwrap();

    let validator = WorkflowValidator::new();
    let result = validator.validate(&workflow).unwrap();

    assert!(result.valid);

    // 验证 1 出现在 2, 3 之前
    let pos_of_1 = result.execution_order.iter().position(|x| x == "1").unwrap();
    let pos_of_2 = result.execution_order.iter().position(|x| x == "2").unwrap();
    let pos_of_3 = result.execution_order.iter().position(|x| x == "3").unwrap();
    let pos_of_5 = result.execution_order.iter().position(|x| x == "5").unwrap();
    let pos_of_7 = result.execution_order.iter().position(|x| x == "7").unwrap();

    assert!(pos_of_1 < pos_of_2, "CheckpointLoader should run before CLIPTextEncode");
    assert!(pos_of_1 < pos_of_3);
    assert!(pos_of_2 < pos_of_5, "CLIPTextEncode should run before KSampler");
    assert!(pos_of_3 < pos_of_5);
    assert!(pos_of_5 < pos_of_7, "KSampler should run before SaveImage");
}

#[test]
fn test_workflow_with_invalid_size() {
    let result = WorkflowBuilder::text_to_image(
        "p".to_string(),
        "n".to_string(),
        0, // 无效尺寸
        0,
        20,
        7.0,
        42,
        "m".to_string(),
    );
    // 应该不会失败（构建器不做尺寸验证，节点执行时才验证）
    assert!(result.is_ok());
}
