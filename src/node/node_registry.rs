// 节点类型系统
// 定义每个节点的输入输出类型约束，支持智能连线验证

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataKind {
    /// 模型权重
    MODEL,
    /// CLIP 文本编码器
    CLIP,
    /// VAE 编解码器
    VAE,
    /// 条件编码（正/负提示词）
    CONDITIONING,
    /// Latent 张量
    LATENT,
    /// 图片数据
    IMAGE,
    /// 控制网络
    CONTROL_NET,
    /// LoRA 微调
    LORA,
    /// 整数
    INT,
    /// 浮点数
    FLOAT,
    /// 字符串
    STRING,
    /// 视频数据
    VIDEO,
    /// 帧序列
    FRAMES,
    /// 任意类型
    ANY,
}

impl DataKind {
    /// 检查类型兼容性
    pub fn is_compatible_with(&self, other: &DataKind) -> bool {
        match (self, other) {
            (DataKind::ANY, _) | (_, DataKind::ANY) => true,
            (a, b) => a == b,
        }
    }
    
    /// 类型名称
    pub fn name(&self) -> &'static str {
        match self {
            DataKind::MODEL => "MODEL",
            DataKind::CLIP => "CLIP",
            DataKind::VAE => "VAE",
            DataKind::CONDITIONING => "CONDITIONING",
            DataKind::LATENT => "LATENT",
            DataKind::IMAGE => "IMAGE",
            DataKind::CONTROL_NET => "CONTROL_NET",
            DataKind::LORA => "LORA",
            DataKind::INT => "INT",
            DataKind::FLOAT => "FLOAT",
            DataKind::STRING => "STRING",
            DataKind::VIDEO => "VIDEO",
            DataKind::FRAMES => "FRAMES",
            DataKind::ANY => "ANY",
        }
    }
}

/// 节点端口定义（输入或输出）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePort {
    /// 端口名称
    pub name: String,
    /// 数据类型
    pub data_kind: DataKind,
    /// 是否必需
    pub required: bool,
    /// 默认值（可选）
    pub default: Option<serde_json::Value>,
    /// 描述
    pub description: String,
}

/// 节点类型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    /// 节点类型名称（class_type）
    pub class_type: String,
    /// 显示名称
    pub display_name: String,
    /// 分类
    pub category: String,
    /// 输入端口列表
    pub inputs: Vec<NodePort>,
    /// 输出端口列表
    pub outputs: Vec<NodePort>,
    /// 描述
    pub description: String,
}

/// 节点注册表
pub struct NodeRegistry {
    /// 所有节点规格
    specs: HashMap<String, NodeSpec>,
}

impl NodeRegistry {
    /// 创建节点注册表并注册所有标准节点
    pub fn new() -> Self {
        let mut specs = HashMap::new();
        
        // === 模型加载节点 ===
        specs.insert("CheckpointLoaderSimple".to_string(), NodeSpec {
            class_type: "CheckpointLoaderSimple".to_string(),
            display_name: "加载主模型".to_string(),
            category: "模型加载".to_string(),
            inputs: vec![
                NodePort { name: "ckpt_name".into(), data_kind: DataKind::STRING, required: true, default: None, description: "模型文件名".into() },
            ],
            outputs: vec![
                NodePort { name: "MODEL".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "UNET 模型".into() },
                NodePort { name: "CLIP".into(), data_kind: DataKind::CLIP, required: true, default: None, description: "CLIP 编码器".into() },
                NodePort { name: "VAE".into(), data_kind: DataKind::VAE, required: true, default: None, description: "VAE 编解码器".into() },
            ],
            description: "加载完整的 Stable Diffusion checkpoint，包含 UNET、CLIP、VAE".into(),
        });
        
        specs.insert("LoraLoader".to_string(), NodeSpec {
            class_type: "LoraLoader".to_string(),
            display_name: "加载 LoRA".to_string(),
            category: "模型加载".to_string(),
            inputs: vec![
                NodePort { name: "model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "基础模型".into() },
                NodePort { name: "clip".into(), data_kind: DataKind::CLIP, required: true, default: None, description: "CLIP 编码器".into() },
                NodePort { name: "lora_name".into(), data_kind: DataKind::STRING, required: true, default: None, description: "LoRA 文件名".into() },
                NodePort { name: "strength_model".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(1.0)), description: "模型强度".into() },
                NodePort { name: "strength_clip".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(1.0)), description: "CLIP 强度".into() },
            ],
            outputs: vec![
                NodePort { name: "MODEL".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "增强后的模型".into() },
                NodePort { name: "CLIP".into(), data_kind: DataKind::CLIP, required: true, default: None, description: "增强后的 CLIP".into() },
            ],
            description: "加载 LoRA 微调模型并应用到基础模型".into(),
        });
        
        specs.insert("ControlNetLoader".to_string(), NodeSpec {
            class_type: "ControlNetLoader".to_string(),
            display_name: "加载 ControlNet".to_string(),
            category: "模型加载".to_string(),
            inputs: vec![
                NodePort { name: "control_net_name".into(), data_kind: DataKind::STRING, required: true, default: None, description: "ControlNet 文件名".into() },
            ],
            outputs: vec![
                NodePort { name: "CONTROL_NET".into(), data_kind: DataKind::CONTROL_NET, required: true, default: None, description: "ControlNet 模型".into() },
            ],
            description: "加载 ControlNet 控制模型".into(),
        });
        
        // === 提示词节点 ===
        specs.insert("CLIPTextEncode".to_string(), NodeSpec {
            class_type: "CLIPTextEncode".to_string(),
            display_name: "编码提示词".to_string(),
            category: "提示词".to_string(),
            inputs: vec![
                NodePort { name: "text".into(), data_kind: DataKind::STRING, required: true, default: None, description: "提示词文本".into() },
                NodePort { name: "clip".into(), data_kind: DataKind::CLIP, required: true, default: None, description: "CLIP 编码器".into() },
            ],
            outputs: vec![
                NodePort { name: "CONDITIONING".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "条件编码".into() },
            ],
            description: "使用 CLIP 将文本编码为条件向量".into(),
        });
        
        // === Latent 节点 ===
        specs.insert("EmptyLatentImage".to_string(), NodeSpec {
            class_type: "EmptyLatentImage".to_string(),
            display_name: "空 Latent".to_string(),
            category: "Latent".to_string(),
            inputs: vec![
                NodePort { name: "width".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(512)), description: "宽度".into() },
                NodePort { name: "height".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(512)), description: "高度".into() },
                NodePort { name: "batch_size".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(1)), description: "批次大小".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "空 Latent 张量".into() },
            ],
            description: "创建空的 Latent 张量作为采样起点".into(),
        });
        
        specs.insert("VAEEncode".to_string(), NodeSpec {
            class_type: "VAEEncode".to_string(),
            display_name: "图片编码".to_string(),
            category: "Latent".to_string(),
            inputs: vec![
                NodePort { name: "pixels".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "输入图片".into() },
                NodePort { name: "vae".into(), data_kind: DataKind::VAE, required: true, default: None, description: "VAE 编解码器".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "编码后的 Latent".into() },
            ],
            description: "使用 VAE 将图片编码为 Latent".into(),
        });
        
        // === 采样节点 ===
        specs.insert("KSampler".to_string(), NodeSpec {
            class_type: "KSampler".to_string(),
            display_name: "标准采样器".to_string(),
            category: "采样".to_string(),
            inputs: vec![
                NodePort { name: "model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "UNET 模型".into() },
                NodePort { name: "positive".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "正向条件".into() },
                NodePort { name: "negative".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "负向条件".into() },
                NodePort { name: "latent_image".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "输入 Latent".into() },
                NodePort { name: "seed".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(0)), description: "随机种子".into() },
                NodePort { name: "steps".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(20)), description: "采样步数".into() },
                NodePort { name: "cfg".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(7.0)), description: "CFG Scale".into() },
                NodePort { name: "sampler_name".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("euler")), description: "采样器名称".into() },
                NodePort { name: "scheduler".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("normal")), description: "调度器".into() },
                NodePort { name: "denoise".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(1.0)), description: "去噪强度".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "采样后的 Latent".into() },
            ],
            description: "执行扩散采样，生成新的 Latent".into(),
        });
        
        specs.insert("KSamplerAdvanced".to_string(), NodeSpec {
            class_type: "KSamplerAdvanced".to_string(),
            display_name: "高级采样器".to_string(),
            category: "采样".to_string(),
            inputs: vec![
                NodePort { name: "model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "UNET 模型".into() },
                NodePort { name: "positive".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "正向条件".into() },
                NodePort { name: "negative".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "负向条件".into() },
                NodePort { name: "latent_image".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "输入 Latent".into() },
                NodePort { name: "add_noise".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("enable")), description: "添加噪声".into() },
                NodePort { name: "noise_seed".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(0)), description: "噪声种子".into() },
                NodePort { name: "steps".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(20)), description: "总步数".into() },
                NodePort { name: "start_at_step".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(0)), description: "起始步数".into() },
                NodePort { name: "end_at_step".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(20)), description: "结束步数".into() },
                NodePort { name: "cfg".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(7.0)), description: "CFG Scale".into() },
                NodePort { name: "sampler_name".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("euler")), description: "采样器".into() },
                NodePort { name: "scheduler".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("normal")), description: "调度器".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "采样后的 Latent".into() },
            ],
            description: "高级采样器，支持指定起始和结束步数".into(),
        });
        
        // === 解码节点 ===
        specs.insert("VAEDecode".to_string(), NodeSpec {
            class_type: "VAEDecode".to_string(),
            display_name: "Latent 解码".to_string(),
            category: "解码".to_string(),
            inputs: vec![
                NodePort { name: "samples".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "Latent 张量".into() },
                NodePort { name: "vae".into(), data_kind: DataKind::VAE, required: true, default: None, description: "VAE 编解码器".into() },
            ],
            outputs: vec![
                NodePort { name: "IMAGE".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "解码后的图片".into() },
            ],
            description: "使用 VAE 将 Latent 解码为图片".into(),
        });
        
        // === 图片处理节点 ===
        specs.insert("ImageScale".to_string(), NodeSpec {
            class_type: "ImageScale".to_string(),
            display_name: "图片缩放".to_string(),
            category: "图片处理".to_string(),
            inputs: vec![
                NodePort { name: "image".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "输入图片".into() },
                NodePort { name: "width".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(1024)), description: "目标宽度".into() },
                NodePort { name: "height".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(1024)), description: "目标高度".into() },
                NodePort { name: "upscale_method".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("nearest-exact")), description: "缩放方法".into() },
            ],
            outputs: vec![
                NodePort { name: "IMAGE".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "缩放后的图片".into() },
            ],
            description: "调整图片尺寸".into(),
        });
        
        specs.insert("ImageUpscaleWithModel".to_string(), NodeSpec {
            class_type: "ImageUpscaleWithModel".to_string(),
            display_name: "模型超分".to_string(),
            category: "图片处理".to_string(),
            inputs: vec![
                NodePort { name: "image".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "输入图片".into() },
                NodePort { name: "upscale_model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "超分模型".into() },
            ],
            outputs: vec![
                NodePort { name: "IMAGE".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "超分后的图片".into() },
            ],
            description: "使用超分模型放大图片".into(),
        });
        
        // === ControlNet 节点 ===
        specs.insert("ControlNetApply".to_string(), NodeSpec {
            class_type: "ControlNetApply".to_string(),
            display_name: "应用 ControlNet".to_string(),
            category: "ControlNet".to_string(),
            inputs: vec![
                NodePort { name: "conditioning".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "原始条件".into() },
                NodePort { name: "control_net".into(), data_kind: DataKind::CONTROL_NET, required: true, default: None, description: "ControlNet 模型".into() },
                NodePort { name: "image".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "控制图片".into() },
                NodePort { name: "strength".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(1.0)), description: "控制强度".into() },
            ],
            outputs: vec![
                NodePort { name: "CONDITIONING".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "增强后的条件".into() },
            ],
            description: "将 ControlNet 应用于条件编码".into(),
        });
        
        // === 视频节点 ===
        specs.insert("SVDImageToVideo".to_string(), NodeSpec {
            class_type: "SVDImageToVideo".to_string(),
            display_name: "SVD 图转视频".to_string(),
            category: "视频".to_string(),
            inputs: vec![
                NodePort { name: "image".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "输入图片".into() },
                NodePort { name: "model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "SVD 模型".into() },
                NodePort { name: "frames".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(14)), description: "帧数".into() },
                NodePort { name: "fps".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(6)), description: "帧率".into() },
            ],
            outputs: vec![
                NodePort { name: "FRAMES".into(), data_kind: DataKind::FRAMES, required: true, default: None, description: "帧序列".into() },
            ],
            description: "使用 SVD 将单张图片转换为视频帧序列".into(),
        });
        
        specs.insert("VideoCombine".to_string(), NodeSpec {
            class_type: "VideoCombine".to_string(),
            display_name: "视频合成".to_string(),
            category: "视频".to_string(),
            inputs: vec![
                NodePort { name: "images".into(), data_kind: DataKind::IMAGE, required: false, default: None, description: "图像数据（可选，不提供则扫描 output/ 目录）".into() },
                NodePort { name: "frame_rate".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(8)), description: "帧率".into() },
                NodePort { name: "format".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("mp4")), description: "视频格式 (mp4/gif/webm/avi/mov)".into() },
                NodePort { name: "codec".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("h264")), description: "编码器 (h264/h265/vp8/vp9/gif/raw)".into() },
                NodePort { name: "quality".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(20.0)), description: "编码质量 (1-51, 越低越好)".into() },
                NodePort { name: "filename_prefix".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("video")), description: "输出文件名前缀".into() },
            ],
            outputs: vec![
                NodePort { name: "filename".into(), data_kind: DataKind::STRING, required: true, default: None, description: "输出视频文件路径".into() },
                NodePort { name: "video".into(), data_kind: DataKind::VIDEO, required: false, default: None, description: "输出视频数据".into() },
            ],
            description: "将帧序列或 output/ 目录中的 PNG 文件合成为视频文件".into(),
        });
        
        specs.insert("FrameInterpolation".to_string(), NodeSpec {
            class_type: "FrameInterpolation".to_string(),
            display_name: "帧插值".to_string(),
            category: "视频".to_string(),
            inputs: vec![
                NodePort { name: "frames".into(), data_kind: DataKind::FRAMES, required: true, default: None, description: "输入帧序列".into() },
                NodePort { name: "multiplier".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(2)), description: "插值倍数".into() },
            ],
            outputs: vec![
                NodePort { name: "FRAMES".into(), data_kind: DataKind::FRAMES, required: true, default: None, description: "插值后的帧序列".into() },
            ],
            description: "在帧之间插入中间帧，增加视频流畅度".into(),
        });
        
        // === 输出节点 ===
        specs.insert("SaveImage".to_string(), NodeSpec {
            class_type: "SaveImage".to_string(),
            display_name: "保存图片".to_string(),
            category: "输出".to_string(),
            inputs: vec![
                NodePort { name: "images".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "要保存的图片".into() },
                NodePort { name: "filename_prefix".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("ComfyUI")), description: "文件名前缀".into() },
            ],
            outputs: vec![],
            description: "保存图片到输出目录".into(),
        });
        
        specs.insert("PreviewImage".to_string(), NodeSpec {
            class_type: "PreviewImage".to_string(),
            display_name: "预览图片".to_string(),
            category: "输出".to_string(),
            inputs: vec![
                NodePort { name: "images".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "要预览的图片".into() },
            ],
            outputs: vec![],
            description: "在界面上预览图片，不保存文件".into(),
        });

        specs.insert("LoadImage".to_string(), NodeSpec {
            class_type: "LoadImage".to_string(),
            display_name: "加载图片".to_string(),
            category: "输入".to_string(),
            inputs: vec![
                NodePort { name: "image".into(), data_kind: DataKind::STRING, required: true, default: None, description: "图片文件名或路径（如 input/bk_0015.jpg）".into() },
            ],
            outputs: vec![
                NodePort { name: "IMAGE".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "加载的图片".into() },
                NodePort { name: "MASK".into(), data_kind: DataKind::IMAGE, required: false, default: None, description: "图片的 alpha 蒙版".into() },
            ],
            description: "从 input 目录加载图片文件，用于图生图/图生视频/局部重绘".into(),
        });

        specs.insert("SaveVideo".to_string(), NodeSpec {
            class_type: "SaveVideo".to_string(),
            display_name: "保存视频".to_string(),
            category: "输出".to_string(),
            inputs: vec![
                NodePort { name: "frames".into(), data_kind: DataKind::FRAMES, required: true, default: None, description: "视频帧序列".into() },
                NodePort { name: "filename_prefix".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("ComfyUI_video")), description: "文件名前缀".into() },
                NodePort { name: "fps".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(8)), description: "帧率".into() },
            ],
            outputs: vec![],
            description: "将帧序列保存为视频文件（MP4/GIF）".into(),
        });

        specs.insert("VAEEncodeForInpaint".to_string(), NodeSpec {
            class_type: "VAEEncodeForInpaint".to_string(),
            display_name: "局部重绘编码".to_string(),
            category: "latent".to_string(),
            inputs: vec![
                NodePort { name: "pixels".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "输入图片".into() },
                NodePort { name: "vae".into(), data_kind: DataKind::VAE, required: true, default: None, description: "VAE 模型".into() },
                NodePort { name: "mask".into(), data_kind: DataKind::IMAGE, required: true, default: None, description: "重绘蒙版（白色为重绘区域）".into() },
                NodePort { name: "grow_mask_by".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(6)), description: "蒙版扩展像素".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "编码后的 Latent（含蒙版信息）".into() },
            ],
            description: "将图片编码为 Latent，同时叠加蒙版信息，用于 inpaint 局部重绘".into(),
        });

        specs.insert("AnimateDiffSampler".to_string(), NodeSpec {
            class_type: "AnimateDiffSampler".to_string(),
            display_name: "AnimateDiff 动画采样".to_string(),
            category: "视频".to_string(),
            inputs: vec![
                NodePort { name: "model".into(), data_kind: DataKind::MODEL, required: true, default: None, description: "UNET 模型".into() },
                NodePort { name: "positive".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "正向条件".into() },
                NodePort { name: "negative".into(), data_kind: DataKind::CONDITIONING, required: true, default: None, description: "负向条件".into() },
                NodePort { name: "latent_image".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "输入 Latent（batch_size=帧数）".into() },
                NodePort { name: "seed".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(0)), description: "随机种子".into() },
                NodePort { name: "steps".into(), data_kind: DataKind::INT, required: false, default: Some(serde_json::json!(25)), description: "采样步数".into() },
                NodePort { name: "cfg".into(), data_kind: DataKind::FLOAT, required: false, default: Some(serde_json::json!(8.0)), description: "CFG Scale".into() },
                NodePort { name: "sampler_name".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("euler")), description: "采样器".into() },
                NodePort { name: "scheduler".into(), data_kind: DataKind::STRING, required: false, default: Some(serde_json::json!("normal")), description: "调度器".into() },
            ],
            outputs: vec![
                NodePort { name: "LATENT".into(), data_kind: DataKind::LATENT, required: true, default: None, description: "采样后的 Latent（含多帧）".into() },
            ],
            description: "AnimateDiff 动画采样器，基于 SD 模型生成动画帧序列".into(),
        });

        Self { specs }
    }
    
    /// 获取节点规格
    pub fn get_spec(&self, class_type: &str) -> Option<&NodeSpec> {
        self.specs.get(class_type)
    }
    
    /// 获取所有节点类型列表
    pub fn list_all(&self) -> Vec<&NodeSpec> {
        self.specs.values().collect()
    }
    
    /// 按分类获取节点列表
    pub fn list_by_category(&self, category: &str) -> Vec<&NodeSpec> {
        self.specs.values()
            .filter(|s| s.category == category)
            .collect()
    }
    
    /// 验证连线是否有效
    pub fn validate_connection(
        &self,
        source_node: &str,
        source_output: &str,
        target_node: &str,
        target_input: &str,
    ) -> Result<bool, String> {
        let source_spec = self.get_spec(source_node)
            .ok_or_else(|| format!("源节点类型 '{}' 不存在", source_node))?;
        let target_spec = self.get_spec(target_node)
            .ok_or_else(|| format!("目标节点类型 '{}' 不存在", target_node))?;
        
        // 找到源输出端口
        let source_port = source_spec.outputs.iter()
            .find(|p| p.name == source_output)
            .ok_or_else(|| format!("源节点 '{}' 没有输出端口 '{}'", source_node, source_output))?;
        
        // 找到目标输入端口
        let target_port = target_spec.inputs.iter()
            .find(|p| p.name == target_input)
            .ok_or_else(|| format!("目标节点 '{}' 没有输入端口 '{}'", target_node, target_input))?;
        
        // 检查类型兼容性
        if !source_port.data_kind.is_compatible_with(&target_port.data_kind) {
            return Err(format!(
                "类型不匹配: '{}' 输出 {} 类型，但 '{}' 需要 {} 类型",
                source_node, source_port.data_kind.name(),
                target_node, target_port.data_kind.name()
            ));
        }
        
        Ok(true)
    }
    
    /// 掷取可以连接到指定输入端口的输出端口列表
    pub fn find_compatible_outputs(&self, target_input_kind: DataKind) -> Vec<(String, String)> {
        self.specs.values()
            .flat_map(|spec| {
                spec.outputs.iter()
                    .filter(|port| port.data_kind.is_compatible_with(&target_input_kind))
                    .map(|port| (spec.class_type.clone(), port.name.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// 获取节点类型的 JSON Schema（供 LLM 使用）
    pub fn to_json_schema(&self) -> serde_json::Value {
        let nodes = self.specs.values().map(|spec| {
            serde_json::json!({
                "class_type": spec.class_type,
                "display_name": spec.display_name,
                "category": spec.category,
                "description": spec.description,
                "inputs": spec.inputs.iter().map(|port| {
                    serde_json::json!({
                        "name": port.name,
                        "type": port.data_kind.name(),
                        "required": port.required,
                        "default": port.default,
                        "description": port.description
                    })
                }).collect::<Vec<_>>(),
                "outputs": spec.outputs.iter().map(|port| {
                    serde_json::json!({
                        "name": port.name,
                        "type": port.data_kind.name(),
                        "description": port.description
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>();
        
        serde_json::json!({
            "node_types": nodes,
            "connection_rules": {
                "MODEL": ["MODEL", "ANY"],
                "CLIP": ["CLIP", "ANY"],
                "VAE": ["VAE", "ANY"],
                "CONDITIONING": ["CONDITIONING", "ANY"],
                "LATENT": ["LATENT", "ANY"],
                "IMAGE": ["IMAGE", "ANY"],
                "CONTROL_NET": ["CONTROL_NET", "ANY"],
                "FRAMES": ["FRAMES", "VIDEO", "ANY"],
                "VIDEO": ["VIDEO", "ANY"]
            }
        })
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_registry_creation() {
        let registry = NodeRegistry::new();
        assert!(registry.get_spec("KSampler").is_some());
        assert!(registry.get_spec("CheckpointLoaderSimple").is_some());
    }
    
    #[test]
    fn test_type_compatibility() {
        assert!(DataKind::MODEL.is_compatible_with(&DataKind::MODEL));
        assert!(DataKind::ANY.is_compatible_with(&DataKind::MODEL));
        assert!(!DataKind::IMAGE.is_compatible_with(&DataKind::LATENT));
    }
    
    #[test]
    fn test_validate_connection() {
        let registry = NodeRegistry::new();
        
        // 有效连接: KSampler.MODEL -> VAEDecode.vae（错误例子，应该是 LATENT -> samples）
        // 正确测试: KSampler.LATENT -> VAEDecode.samples
        let result = registry.validate_connection("KSampler", "LATENT", "VAEDecode", "samples");
        assert!(result.is_ok());
        
        // 无效连接: 类型不匹配
        let result = registry.validate_connection("CheckpointLoaderSimple", "MODEL", "VAEDecode", "samples");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_find_compatible_outputs() {
        let registry = NodeRegistry::new();
        let outputs = registry.find_compatible_outputs(DataKind::LATENT);
        assert!(!outputs.is_empty());
        assert!(outputs.iter().any(|(n, p)| n == "KSampler" && p == "LATENT"));
    }
    
    #[test]
    fn test_json_schema() {
        let registry = NodeRegistry::new();
        let schema = registry.to_json_schema();
        assert!(schema["node_types"].is_array());
        assert!(schema["node_types"].as_array().unwrap().len() > 10);
    }
}