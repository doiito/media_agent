// 工作流构建器

use crate::types::*;
use std::collections::HashMap;

/// 工作流构建器
pub struct WorkflowBuilder;

impl WorkflowBuilder {
    /// 创建文生图工作流
    pub fn text_to_image(
        prompt: String,
        negative_prompt: String,
        width: usize,
        height: usize,
        steps: usize,
        cfg: f32,
        seed: usize,
        model: String,
    ) -> Result<Workflow, Error> {
        let mut nodes = HashMap::new();

        // Node 1: CheckpointLoaderSimple
        nodes.insert(
            "1".to_string(),
            WorkflowNode {
                class_type: "CheckpointLoaderSimple".to_string(),
                inputs: HashMap::from([
                    ("ckpt_name".to_string(), InputValue::Direct(Value::String(model))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 2: CLIPTextEncode (positive)
        nodes.insert(
            "2".to_string(),
            WorkflowNode {
                class_type: "CLIPTextEncode".to_string(),
                inputs: HashMap::from([
                    ("text".to_string(), InputValue::Direct(Value::String(prompt))),
                    ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 3: CLIPTextEncode (negative)
        nodes.insert(
            "3".to_string(),
            WorkflowNode {
                class_type: "CLIPTextEncode".to_string(),
                inputs: HashMap::from([
                    ("text".to_string(), InputValue::Direct(Value::String(negative_prompt))),
                    ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 4: EmptyLatentImage
        nodes.insert(
            "4".to_string(),
            WorkflowNode {
                class_type: "EmptyLatentImage".to_string(),
                inputs: HashMap::from([
                    ("width".to_string(), InputValue::Direct(Value::Int(width as i64))),
                    ("height".to_string(), InputValue::Direct(Value::Int(height as i64))),
                    ("batch_size".to_string(), InputValue::Direct(Value::Int(1))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 5: KSampler
        nodes.insert(
            "5".to_string(),
            WorkflowNode {
                class_type: "KSampler".to_string(),
                inputs: HashMap::from([
                    ("model".to_string(), InputValue::Link(["1".to_string(), "0".to_string()])),
                    ("positive".to_string(), InputValue::Link(["2".to_string(), "0".to_string()])),
                    ("negative".to_string(), InputValue::Link(["3".to_string(), "0".to_string()])),
                    ("latent_image".to_string(), InputValue::Link(["4".to_string(), "0".to_string()])),
                    ("seed".to_string(), InputValue::Direct(Value::Int(seed as i64))),
                    ("steps".to_string(), InputValue::Direct(Value::Int(steps as i64))),
                    ("cfg".to_string(), InputValue::Direct(Value::Float(cfg as f64))),
                    ("sampler_name".to_string(), InputValue::Direct(Value::String("euler".to_string()))),
                    ("scheduler".to_string(), InputValue::Direct(Value::String("normal".to_string()))),
                    ("denoise".to_string(), InputValue::Direct(Value::Float(1.0))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 6: VAEDecode
        nodes.insert(
            "6".to_string(),
            WorkflowNode {
                class_type: "VAEDecode".to_string(),
                inputs: HashMap::from([
                    ("samples".to_string(), InputValue::Link(["5".to_string(), "0".to_string()])),
                    ("vae".to_string(), InputValue::Link(["1".to_string(), "2".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 7: SaveImage
        nodes.insert(
            "7".to_string(),
            WorkflowNode {
                class_type: "SaveImage".to_string(),
                inputs: HashMap::from([
                    ("images".to_string(), InputValue::Link(["6".to_string(), "0".to_string()])),
                    ("filename_prefix".to_string(), InputValue::Direct(Value::String("ComfyUI".to_string()))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        Ok(Workflow {
            nodes,
            links: vec![],
        })
    }

    /// 创建图生图工作流
    pub fn image_to_image(
        prompt: String,
        negative_prompt: String,
        input_image: String,
        denoise: f32,
        steps: usize,
        cfg: f32,
        seed: usize,
        model: String,
    ) -> Result<Workflow, Error> {
        let mut nodes = HashMap::new();

        // Node 1: CheckpointLoaderSimple
        nodes.insert(
            "1".to_string(),
            WorkflowNode {
                class_type: "CheckpointLoaderSimple".to_string(),
                inputs: HashMap::from([
                    ("ckpt_name".to_string(), InputValue::Direct(Value::String(model))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 2: LoadImage
        nodes.insert(
            "2".to_string(),
            WorkflowNode {
                class_type: "LoadImage".to_string(),
                inputs: HashMap::from([
                    ("image".to_string(), InputValue::Direct(Value::String(input_image))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 3: VAEEncode
        nodes.insert(
            "3".to_string(),
            WorkflowNode {
                class_type: "VAEEncode".to_string(),
                inputs: HashMap::from([
                    ("pixels".to_string(), InputValue::Link(["2".to_string(), "0".to_string()])),
                    ("vae".to_string(), InputValue::Link(["1".to_string(), "2".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 4: CLIPTextEncode (positive)
        nodes.insert(
            "4".to_string(),
            WorkflowNode {
                class_type: "CLIPTextEncode".to_string(),
                inputs: HashMap::from([
                    ("text".to_string(), InputValue::Direct(Value::String(prompt))),
                    ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 5: CLIPTextEncode (negative)
        nodes.insert(
            "5".to_string(),
            WorkflowNode {
                class_type: "CLIPTextEncode".to_string(),
                inputs: HashMap::from([
                    ("text".to_string(), InputValue::Direct(Value::String(negative_prompt))),
                    ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 6: KSampler
        nodes.insert(
            "6".to_string(),
            WorkflowNode {
                class_type: "KSampler".to_string(),
                inputs: HashMap::from([
                    ("model".to_string(), InputValue::Link(["1".to_string(), "0".to_string()])),
                    ("positive".to_string(), InputValue::Link(["4".to_string(), "0".to_string()])),
                    ("negative".to_string(), InputValue::Link(["5".to_string(), "0".to_string()])),
                    ("latent_image".to_string(), InputValue::Link(["3".to_string(), "0".to_string()])),
                    ("seed".to_string(), InputValue::Direct(Value::Int(seed as i64))),
                    ("steps".to_string(), InputValue::Direct(Value::Int(steps as i64))),
                    ("cfg".to_string(), InputValue::Direct(Value::Float(cfg as f64))),
                    ("sampler_name".to_string(), InputValue::Direct(Value::String("euler".to_string()))),
                    ("scheduler".to_string(), InputValue::Direct(Value::String("normal".to_string()))),
                    ("denoise".to_string(), InputValue::Direct(Value::Float(denoise as f64))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 7: VAEDecode
        nodes.insert(
            "7".to_string(),
            WorkflowNode {
                class_type: "VAEDecode".to_string(),
                inputs: HashMap::from([
                    ("samples".to_string(), InputValue::Link(["6".to_string(), "0".to_string()])),
                    ("vae".to_string(), InputValue::Link(["1".to_string(), "2".to_string()])),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        // Node 8: SaveImage
        nodes.insert(
            "8".to_string(),
            WorkflowNode {
                class_type: "SaveImage".to_string(),
                inputs: HashMap::from([
                    ("images".to_string(), InputValue::Link(["7".to_string(), "0".to_string()])),
                    ("filename_prefix".to_string(), InputValue::Direct(Value::String("ComfyUI".to_string()))),
                ]),
                pos: None,
                size: None,
                is_changed: None,
            },
        );

        Ok(Workflow {
            nodes,
            links: vec![],
        })
    }
}