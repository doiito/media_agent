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
    /// 节点信息缓存（避免在异步环境中调用blocking_lock）
    node_info_cache: HashMap<String, HashMap<String, InputType>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        let (registered_nodes, node_info_cache) = Self::register_default_nodes();
        Self {
            registered_nodes,
            node_info_cache,
        }
    }

    /// 使用指定的 BackendRouter 创建注册表（让视频节点能调用实际后端）
    pub fn with_backend(router: Arc<crate::backend::BackendRouter>) -> Self {
        let (registered_nodes, node_info_cache) = Self::register_default_nodes_with_backend(router);
        Self {
            registered_nodes,
            node_info_cache,
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

    /// 获取所有节点信息（使用缓存，避免blocking_lock）
    pub fn get_all_node_info(&self) -> HashMap<String, HashMap<String, InputType>> {
        self.node_info_cache.clone()
    }

    /// 注册节点
    pub fn register(&mut self, class_type: String, node: Arc<Mutex<dyn Node>>) {
        // 注意：这个方法仍然需要blocking_lock，所以不建议在异步环境中使用
        // 建议在初始化阶段完成所有注册
        use std::sync::MutexGuard;
        // 使用 try_lock 避免 panic，如果失败则不更新缓存
        if let Ok(instance) = node.try_lock() {
            let info = instance.input_types();
            self.node_info_cache.insert(class_type.clone(), info);
        } else {
            log::warn!("Cannot update node info cache for '{}' - mutex is locked", class_type);
        }
        self.registered_nodes.insert(class_type, node);
    }

    /// 注册默认节点
    fn register_default_nodes() -> (HashMap<String, Arc<Mutex<dyn Node>>>, HashMap<String, HashMap<String, InputType>>) {
        Self::register_default_nodes_internal(None)
    }

    /// 使用指定 BackendRouter 注册默认节点
    fn register_default_nodes_with_backend(router: Arc<crate::backend::BackendRouter>) -> (HashMap<String, Arc<Mutex<dyn Node>>>, HashMap<String, HashMap<String, InputType>>) {
        Self::register_default_nodes_internal(Some(router))
    }

    fn register_default_nodes_internal(router: Option<Arc<crate::backend::BackendRouter>>) -> (HashMap<String, Arc<Mutex<dyn Node>>>, HashMap<String, HashMap<String, InputType>>) {
        let mut nodes: HashMap<String, Arc<Mutex<dyn Node>>>= HashMap::new();
        let mut info_cache: HashMap<String, HashMap<String, InputType>> = HashMap::new();

        // Helper: 注册节点并同时缓存其信息（在创建时获取，无需lock）
        macro_rules! register_node {
            ($name:expr, $node_expr:expr) => {
                let node_instance = $node_expr;
                info_cache.insert($name.to_string(), node_instance.input_types());
                nodes.insert($name.to_string(), Arc::new(Mutex::new(node_instance)));
            };
        }

        // 注册核心节点
        register_node!("CheckpointLoaderSimple", CheckpointLoaderNode::new());
        register_node!("CLIPTextEncode", CLIPTextEncodeNode::new());
        register_node!("KSampler", KSamplerNode::new());
        register_node!("EmptyLatentImage", EmptyLatentImageNode::new());
        register_node!("VAEDecode", VAEDecodeNode::new());
        register_node!("VAEEncode", VAEEncodeNode::new());
        register_node!("LoadImage", LoadImageNode::new());
        register_node!("SaveImage", SaveImageNode::new());

        // 扩展节点 - 模型加载器
        register_node!("LoraLoader", LoraLoaderNode::new());
        register_node!("ControlNetLoader", ControlNetLoaderNode::new());
        register_node!("ControlNetApply", ControlNetApplyNode);
        register_node!("CLIPLoader", CLIPLoaderNode::new());
        register_node!("VAELoader", VAELoaderNode::new());
        register_node!("UNETLoader", UNETLoaderNode::new());
        register_node!("DualCLIPLoader", DualCLIPLoaderNode);
        register_node!("StyleModelLoader", StyleModelLoaderNode);
        register_node!("CLIPVisionLoader", CLIPVisionLoaderNode);
        register_node!("CLIPVisionEncode", CLIPVisionEncodeNode);

        // 扩展节点 - Conditioning 处理
        register_node!("ConditioningCombine", ConditioningCombineNode);
        register_node!("ConditioningConcat", ConditioningConcatNode);

        // 扩展节点 - 图像处理
        register_node!("ImageScale", ImageScaleNode);
        register_node!("UpscaleImageWithModel", UpscaleImageWithModelNode::new());
        register_node!("ImageBlend", ImageBlendNode);
        register_node!("ImageCrop", ImageCropNode);
        register_node!("ImageRotate", ImageRotateNode);
        register_node!("ImageColorAdjust", ImageColorAdjustNode);
        register_node!("ImageFilter", ImageFilterNode);
        register_node!("ImageFlip", ImageFlipNode);
        register_node!("ImageSharpen", ImageSharpenNode);
        register_node!("PreviewImage", PreviewImageNode);

        // 高级采样器节点
        register_node!("KSamplerAdvanced", KSamplerAdvancedNode::new());
        register_node!("SamplerCustom", SamplerCustomNode::new());
        register_node!("SchedulerAdvanced", SchedulerAdvancedNode);
        register_node!("LatentNoiseInjection", LatentNoiseInjectionNode);

        // 视频节点（需要特殊处理router参数）
        {
            let svd_node = match &router {
                Some(r) => SVDImageToVideoNode::with_backend(r.clone()),
                None => SVDImageToVideoNode::new(),
            };
            register_node!("SVDImageToVideo", svd_node);
        }
        register_node!("VideoFrameInterpolation", VideoFrameInterpolationNode);
        register_node!("VideoCombine", VideoCombineNode);
        register_node!("FrameSequenceGenerator", FrameSequenceGeneratorNode::new());
        register_node!("LatentInterpolation", LatentInterpolationNode);
        register_node!("VideoToFrames", VideoToFramesNode);
        {
            let ad_node = match &router {
                Some(r) => AnimateDiffSamplerNode::with_backend(r.clone()),
                None => AnimateDiffSamplerNode::new(),
            };
            register_node!("AnimateDiffSampler", ad_node);
        }

        (nodes, info_cache)
    }
}