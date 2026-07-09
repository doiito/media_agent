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
                    ("sampler_name".to_string(), InputValue::Direct(Value::String("dpm++2m".to_string()))),
                    ("scheduler".to_string(), InputValue::Direct(Value::String("karras".to_string()))),
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
                    ("sampler_name".to_string(), InputValue::Direct(Value::String("dpm++2m".to_string()))),
                    ("scheduler".to_string(), InputValue::Direct(Value::String("karras".to_string()))),
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

    /// 创建图生视频工作流（LoadImage → SVDImageToVideo → VideoCombine）
    pub fn image_to_video(
        image_path: String,
        model: String,
        frames: usize,
        fps: usize,
        motion_bucket_id: i32,
        cfg: f32,
        steps: usize,
        seed: i64,
    ) -> Result<Workflow, Error> {
        let mut nodes: HashMap<String, WorkflowNode> = HashMap::new();

        // Node 1: CheckpointLoaderSimple (SVD 模型)
        nodes.insert("1".to_string(), WorkflowNode {
            class_type: "CheckpointLoaderSimple".to_string(),
            inputs: HashMap::from([
                ("ckpt_name".to_string(), InputValue::Direct(Value::String(model))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 2: LoadImage
        nodes.insert("2".to_string(), WorkflowNode {
            class_type: "LoadImage".to_string(),
            inputs: HashMap::from([
                ("image".to_string(), InputValue::Direct(Value::String(image_path))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 3: SVDImageToVideo
        nodes.insert("3".to_string(), WorkflowNode {
            class_type: "SVDImageToVideo".to_string(),
            inputs: HashMap::from([
                ("model".to_string(), InputValue::Link(["1".to_string(), "0".to_string()])),
                ("image".to_string(), InputValue::Link(["2".to_string(), "0".to_string()])),
                ("frames".to_string(), InputValue::Direct(Value::Int(frames as i64))),
                ("fps".to_string(), InputValue::Direct(Value::Int(fps as i64))),
                ("motion_bucket_id".to_string(), InputValue::Direct(Value::Int(motion_bucket_id as i64))),
                ("cfg".to_string(), InputValue::Direct(Value::Float(cfg as f64))),
                ("steps".to_string(), InputValue::Direct(Value::Int(steps as i64))),
                ("seed".to_string(), InputValue::Direct(Value::Int(seed))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 4: VideoCombine
        nodes.insert("4".to_string(), WorkflowNode {
            class_type: "VideoCombine".to_string(),
            inputs: HashMap::from([
                ("frames".to_string(), InputValue::Link(["3".to_string(), "0".to_string()])),
                ("fps".to_string(), InputValue::Direct(Value::Int(fps as i64))),
                ("filename_prefix".to_string(), InputValue::Direct(Value::String("comfyui_video".to_string()))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        Ok(Workflow { nodes, links: vec![] })
    }

    /// 创建文生视频工作流（CLIPTextEncode → EmptyLatentImage → AnimateDiffSampler → VAEDecode → VideoCombine）
    pub fn text_to_video(
        prompt: String,
        negative_prompt: String,
        width: usize,
        height: usize,
        frames: usize,
        fps: usize,
        steps: usize,
        cfg: f32,
        seed: i64,
        model: String,
    ) -> Result<Workflow, Error> {
        let mut nodes: HashMap<String, WorkflowNode> = HashMap::new();

        // Node 1: CheckpointLoaderSimple
        nodes.insert("1".to_string(), WorkflowNode {
            class_type: "CheckpointLoaderSimple".to_string(),
            inputs: HashMap::from([
                ("ckpt_name".to_string(), InputValue::Direct(Value::String(model))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 2: CLIPTextEncode (positive)
        nodes.insert("2".to_string(), WorkflowNode {
            class_type: "CLIPTextEncode".to_string(),
            inputs: HashMap::from([
                ("text".to_string(), InputValue::Direct(Value::String(prompt))),
                ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 3: CLIPTextEncode (negative)
        nodes.insert("3".to_string(), WorkflowNode {
            class_type: "CLIPTextEncode".to_string(),
            inputs: HashMap::from([
                ("text".to_string(), InputValue::Direct(Value::String(negative_prompt))),
                ("clip".to_string(), InputValue::Link(["1".to_string(), "1".to_string()])),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 4: EmptyLatentImage (batch_size = frames)
        nodes.insert("4".to_string(), WorkflowNode {
            class_type: "EmptyLatentImage".to_string(),
            inputs: HashMap::from([
                ("width".to_string(), InputValue::Direct(Value::Int(width as i64))),
                ("height".to_string(), InputValue::Direct(Value::Int(height as i64))),
                ("batch_size".to_string(), InputValue::Direct(Value::Int(frames as i64))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 5: AnimateDiffSampler
        nodes.insert("5".to_string(), WorkflowNode {
            class_type: "AnimateDiffSampler".to_string(),
            inputs: HashMap::from([
                ("model".to_string(), InputValue::Link(["1".to_string(), "0".to_string()])),
                ("positive".to_string(), InputValue::Link(["2".to_string(), "0".to_string()])),
                ("negative".to_string(), InputValue::Link(["3".to_string(), "0".to_string()])),
                ("latent_image".to_string(), InputValue::Link(["4".to_string(), "0".to_string()])),
                ("seed".to_string(), InputValue::Direct(Value::Int(seed))),
                ("steps".to_string(), InputValue::Direct(Value::Int(steps as i64))),
                ("cfg".to_string(), InputValue::Direct(Value::Float(cfg as f64))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 6: VAEDecode
        nodes.insert("6".to_string(), WorkflowNode {
            class_type: "VAEDecode".to_string(),
            inputs: HashMap::from([
                ("samples".to_string(), InputValue::Link(["5".to_string(), "0".to_string()])),
                ("vae".to_string(), InputValue::Link(["1".to_string(), "2".to_string()])),
            ]),
            pos: None, size: None, is_changed: None,
        });

        // Node 7: VideoCombine
        nodes.insert("7".to_string(), WorkflowNode {
            class_type: "VideoCombine".to_string(),
            inputs: HashMap::from([
                ("frames".to_string(), InputValue::Link(["6".to_string(), "0".to_string()])),
                ("fps".to_string(), InputValue::Direct(Value::Int(fps as i64))),
                ("filename_prefix".to_string(), InputValue::Direct(Value::String("comfyui_video".to_string()))),
            ]),
            pos: None, size: None, is_changed: None,
        });

        Ok(Workflow { nodes, links: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_to_video_workflow() {
        let wf = WorkflowBuilder::image_to_video(
            "input/bk_0015.jpg".into(),
            "svd_xt.safetensors".into(),
            25, 8, 127, 2.5, 25, 42,
        ).unwrap();

        assert!(wf.nodes.contains_key("1")); // CheckpointLoaderSimple
        assert!(wf.nodes.contains_key("2")); // LoadImage
        assert!(wf.nodes.contains_key("3")); // SVDImageToVideo
        assert!(wf.nodes.contains_key("4")); // VideoCombine

        // 验证 SVD 节点的 class_type
        assert_eq!(wf.nodes.get("3").unwrap().class_type, "SVDImageToVideo");
        // 验证 LoadImage 的 image 参数
        let load_image = wf.nodes.get("2").unwrap();
        assert_eq!(load_image.class_type, "LoadImage");
        if let InputValue::Direct(Value::String(path)) = load_image.inputs.get("image").unwrap() {
            assert_eq!(path, "input/bk_0015.jpg");
        } else {
            panic!("Expected Direct String for image input");
        }
    }

    #[test]
    fn test_text_to_video_workflow() {
        let wf = WorkflowBuilder::text_to_video(
            "a cat running".into(),
            "blurry".into(),
            512, 512, 16, 8, 20, 7.0, 42, "v1-5.safetensors".into(),
        ).unwrap();

        assert!(wf.nodes.contains_key("5")); // AnimateDiffSampler
        assert!(wf.nodes.contains_key("6")); // VAEDecode
        assert!(wf.nodes.contains_key("7")); // VideoCombine

        // 验证 AnimateDiffSampler 的 class_type
        assert_eq!(wf.nodes.get("5").unwrap().class_type, "AnimateDiffSampler");
        // 验证 EmptyLatentImage 的 batch_size = frames
        let latent = wf.nodes.get("4").unwrap();
        assert_eq!(latent.class_type, "EmptyLatentImage");
        if let InputValue::Direct(Value::Int(batch)) = latent.inputs.get("batch_size").unwrap() {
            assert_eq!(*batch, 16);
        } else {
            panic!("Expected Direct Int for batch_size");
        }
    }
}