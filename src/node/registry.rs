// 节点注册表

use crate::types::*;
use crate::node::{Node, InputType};
use crate::node::core_nodes::*;
use crate::node::extended_nodes::*;
use crate::node::advanced_sampler::*;
use crate::node::image_processing::*;
use crate::node::video_nodes::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// 节点注册表
pub struct NodeRegistry {
    /// 注册的节点类型
    registered_nodes: HashMap<String, Arc<Mutex<dyn Node>>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self {
            registered_nodes: Self::register_default_nodes(),
        }
    }

    /// 使用指定的 BackendRouter 创建注册表（让视频节点能调用实际后端）
    pub fn with_backend(router: Arc<crate::backend::BackendRouter>) -> Self {
        Self {
            registered_nodes: Self::register_default_nodes_with_backend(router),
        }
    }

    /// 创建节点实例
    pub fn create_node(&self, class_type: &str) -> Result<Arc<Mutex<dyn Node>>, Error> {
        if let Some(node) = self.registered_nodes.get(class_type) {
            Ok(node.clone())
        } else {
            Err(Error::NodeNotFound(format!("Node class '{}' not found", class_type)))
        }
    }

    /// 获取所有节点信息
    pub fn get_all_node_info(&self) -> HashMap<String, HashMap<String, InputType>> {
        let mut info = HashMap::new();
        for (class_type, node) in &self.registered_nodes {
            info.insert(class_type.clone(), node.blocking_lock().input_types());
        }
        info
    }

    /// 注册节点
    pub fn register(&mut self, class_type: String, node: Arc<Mutex<dyn Node>>) {
        self.registered_nodes.insert(class_type, node);
    }

    /// 注册默认节点
    fn register_default_nodes() -> HashMap<String, Arc<Mutex<dyn Node>>> {
        Self::register_default_nodes_internal(None)
    }

    /// 使用指定 BackendRouter 注册默认节点
    fn register_default_nodes_with_backend(router: Arc<crate::backend::BackendRouter>) -> HashMap<String, Arc<Mutex<dyn Node>>> {
        Self::register_default_nodes_internal(Some(router))
    }

    fn register_default_nodes_internal(router: Option<Arc<crate::backend::BackendRouter>>) -> HashMap<String, Arc<Mutex<dyn Node>>> {
        let mut nodes: HashMap<String, Arc<Mutex<dyn Node>>>= HashMap::new();

        // 注册核心节点
        nodes.insert(
            "CheckpointLoaderSimple".to_string(),
            Arc::new(Mutex::new(CheckpointLoaderNode::new())),
        );
        nodes.insert(
            "CLIPTextEncode".to_string(),
            Arc::new(Mutex::new(CLIPTextEncodeNode::new())),
        );
        nodes.insert(
            "KSampler".to_string(),
            Arc::new(Mutex::new(KSamplerNode::new())),
        );
        nodes.insert(
            "EmptyLatentImage".to_string(),
            Arc::new(Mutex::new(EmptyLatentImageNode::new())),
        );
        nodes.insert(
            "VAEDecode".to_string(),
            Arc::new(Mutex::new(VAEDecodeNode::new())),
        );
        nodes.insert(
            "VAEEncode".to_string(),
            Arc::new(Mutex::new(VAEEncodeNode::new())),
        );
        nodes.insert(
            "LoadImage".to_string(),
            Arc::new(Mutex::new(LoadImageNode::new())),
        );
        nodes.insert(
            "SaveImage".to_string(),
            Arc::new(Mutex::new(SaveImageNode::new())),
        );

        // 扩展节点 - 模型加载器
        nodes.insert(
            "LoraLoader".to_string(),
            Arc::new(Mutex::new(LoraLoaderNode::new())),
        );
        nodes.insert(
            "ControlNetLoader".to_string(),
            Arc::new(Mutex::new(ControlNetLoaderNode::new())),
        );
        nodes.insert(
            "ControlNetApply".to_string(),
            Arc::new(Mutex::new(ControlNetApplyNode)),
        );
        nodes.insert(
            "CLIPLoader".to_string(),
            Arc::new(Mutex::new(CLIPLoaderNode::new())),
        );
        nodes.insert(
            "VAELoader".to_string(),
            Arc::new(Mutex::new(VAELoaderNode::new())),
        );
        nodes.insert(
            "UNETLoader".to_string(),
            Arc::new(Mutex::new(UNETLoaderNode::new())),
        );
        nodes.insert(
            "DualCLIPLoader".to_string(),
            Arc::new(Mutex::new(DualCLIPLoaderNode)),
        );
        nodes.insert(
            "StyleModelLoader".to_string(),
            Arc::new(Mutex::new(StyleModelLoaderNode)),
        );
        nodes.insert(
            "CLIPVisionLoader".to_string(),
            Arc::new(Mutex::new(CLIPVisionLoaderNode)),
        );
        nodes.insert(
            "CLIPVisionEncode".to_string(),
            Arc::new(Mutex::new(CLIPVisionEncodeNode)),
        );

        // 扩展节点 - Conditioning 处理
        nodes.insert(
            "ConditioningCombine".to_string(),
            Arc::new(Mutex::new(ConditioningCombineNode)),
        );
        nodes.insert(
            "ConditioningConcat".to_string(),
            Arc::new(Mutex::new(ConditioningConcatNode)),
        );

        // 扩展节点 - 图像处理
        nodes.insert(
            "ImageScale".to_string(),
            Arc::new(Mutex::new(ImageScaleNode)),
        );
        nodes.insert(
            "UpscaleImageWithModel".to_string(),
            Arc::new(Mutex::new(UpscaleImageWithModelNode::new())),
        );
        nodes.insert(
            "ImageBlend".to_string(),
            Arc::new(Mutex::new(ImageBlendNode)),
        );
        nodes.insert(
            "ImageCrop".to_string(),
            Arc::new(Mutex::new(ImageCropNode)),
        );
        nodes.insert(
            "ImageRotate".to_string(),
            Arc::new(Mutex::new(ImageRotateNode)),
        );
        nodes.insert(
            "ImageColorAdjust".to_string(),
            Arc::new(Mutex::new(ImageColorAdjustNode)),
        );
        nodes.insert(
            "ImageFilter".to_string(),
            Arc::new(Mutex::new(ImageFilterNode)),
        );
        nodes.insert(
            "ImageFlip".to_string(),
            Arc::new(Mutex::new(ImageFlipNode)),
        );
        nodes.insert(
            "ImageSharpen".to_string(),
            Arc::new(Mutex::new(ImageSharpenNode)),
        );
        nodes.insert(
            "PreviewImage".to_string(),
            Arc::new(Mutex::new(PreviewImageNode)),
        );

        // 高级采样器节点
        nodes.insert(
            "KSamplerAdvanced".to_string(),
            Arc::new(Mutex::new(KSamplerAdvancedNode::new())),
        );
        nodes.insert(
            "SamplerCustom".to_string(),
            Arc::new(Mutex::new(SamplerCustomNode::new())),
        );
        nodes.insert(
            "SchedulerAdvanced".to_string(),
            Arc::new(Mutex::new(SchedulerAdvancedNode)),
        );
        nodes.insert(
            "LatentNoiseInjection".to_string(),
            Arc::new(Mutex::new(LatentNoiseInjectionNode)),
        );

        // 视频节点
        nodes.insert(
            "SVDImageToVideo".to_string(),
            Arc::new(Mutex::new(
                match &router {
                    Some(r) => SVDImageToVideoNode::with_backend(r.clone()),
                    None => SVDImageToVideoNode::new(),
                }
            )),
        );
        nodes.insert(
            "VideoFrameInterpolation".to_string(),
            Arc::new(Mutex::new(VideoFrameInterpolationNode)),
        );
        nodes.insert(
            "VideoCombine".to_string(),
            Arc::new(Mutex::new(VideoCombineNode)),
        );
        nodes.insert(
            "FrameSequenceGenerator".to_string(),
            Arc::new(Mutex::new(FrameSequenceGeneratorNode::new())),
        );
        nodes.insert(
            "LatentInterpolation".to_string(),
            Arc::new(Mutex::new(LatentInterpolationNode)),
        );
        nodes.insert(
            "VideoToFrames".to_string(),
            Arc::new(Mutex::new(VideoToFramesNode)),
        );
        nodes.insert(
            "AnimateDiffSampler".to_string(),
            Arc::new(Mutex::new(
                match &router {
                    Some(r) => AnimateDiffSamplerNode::with_backend(r.clone()),
                    None => AnimateDiffSamplerNode::new(),
                }
            )),
        );

        nodes
    }
}