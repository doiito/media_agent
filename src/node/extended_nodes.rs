// 扩展节点实现
// 包含 LoRA、ControlNet、Conditioning、ImageScale、Upscale 等常用节点

use crate::types::*;
use crate::node::{Node, InputType, OutputType};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use log::{info, debug, warn};

// ============================================================================
// LoraLoader 节点 - 加载 LoRA 权重
// ============================================================================

pub struct LoraLoaderNode {
    loaded_loras: HashMap<String, LoadedLora>,
}

#[derive(Debug, Clone)]
struct LoadedLora {
    lora_path: String,
    strength_model: f32,
    strength_clip: f32,
}

impl LoraLoaderNode {
    pub fn new() -> Self {
        Self {
            loaded_loras: HashMap::new(),
        }
    }

    fn find_lora_file(name: &str) -> Option<String> {
        let search_dirs = ["models/lora", "models/loras"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for LoraLoaderNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for LoraLoaderNode {
    fn class_type(&self) -> &str {
        "LoraLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("clip".to_string(), InputType {
                data_type: DataType::CLIP,
                required: true,
                default: None,
                choices: None,
            }),
            ("lora_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("strength_model".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("strength_clip".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("MODEL".to_string(), OutputType {
                data_type: DataType::MODEL,
                name: "MODEL".to_string(),
            }),
            ("CLIP".to_string(), OutputType {
                data_type: DataType::CLIP,
                name: "CLIP".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?.to_string();
        let clip = inputs.get("clip")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip".to_string()))?
            .as_ref_str()?.to_string();
        let lora_name = inputs.get("lora_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing lora_name".to_string()))?
            .as_str()?.to_string();
        let strength_model = inputs.get("strength_model")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let strength_clip = inputs.get("strength_clip")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;

        let lora_path = Self::find_lora_file(&lora_name)
            .unwrap_or_else(|| lora_name.clone());

        info!("Loading LoRA: {} (strength_model={}, strength_clip={})",
              lora_path, strength_model, strength_clip);

        let loaded = LoadedLora {
            lora_path: lora_path.clone(),
            strength_model,
            strength_clip,
        };
        self.loaded_loras.insert(lora_name.clone(), loaded);

        // 返回组合后的 MODEL 和 CLIP（带 LoRA 标识）
        let combined_model = format!("{}+lora:{}@{}", model, lora_path, strength_model);
        let combined_clip = format!("{}+lora:{}@{}", clip, lora_path, strength_clip);

        Ok(HashMap::from([
            ("MODEL".to_string(), Value::Model(combined_model)),
            ("CLIP".to_string(), Value::Clip(combined_clip)),
        ]))
    }
}

// ============================================================================
// ControlNetLoader 节点 - 加载 ControlNet 模型
// ============================================================================

pub struct ControlNetLoaderNode {
    loaded_controlnets: HashMap<String, String>,
}

impl ControlNetLoaderNode {
    pub fn new() -> Self {
        Self {
            loaded_controlnets: HashMap::new(),
        }
    }

    fn find_controlnet_file(name: &str) -> Option<String> {
        let search_dirs = ["models/controlnet", "models/controlnets"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for ControlNetLoaderNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for ControlNetLoaderNode {
    fn class_type(&self) -> &str {
        "ControlNetLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("control_net_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CONTROL_NET".to_string(), OutputType {
                data_type: DataType::CONTROL_NET,
                name: "CONTROL_NET".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let control_net_name = inputs.get("control_net_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing control_net_name".to_string()))?
            .as_str()?.to_string();

        let controlnet_path = Self::find_controlnet_file(&control_net_name)
            .unwrap_or_else(|| control_net_name.clone());

        info!("Loading ControlNet: {}", controlnet_path);
        self.loaded_controlnets.insert(control_net_name, controlnet_path.clone());

        Ok(HashMap::from([
            ("CONTROL_NET".to_string(), Value::ControlNet(controlnet_path)),
        ]))
    }
}

// ============================================================================
// ControlNetApply 节点 - 应用 ControlNet 到 conditioning
// ============================================================================

pub struct ControlNetApplyNode;

impl Default for ControlNetApplyNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ControlNetApplyNode {
    fn class_type(&self) -> &str {
        "ControlNetApply"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("conditioning".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("control_net".to_string(), InputType {
                data_type: DataType::CONTROL_NET,
                required: true,
                default: None,
                choices: None,
            }),
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("strength".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CONDITIONING".to_string(), OutputType {
                data_type: DataType::CONDITIONING,
                name: "CONDITIONING".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let conditioning = inputs.get("conditioning")
            .ok_or_else(|| Error::ExecutionFailed("Missing conditioning".to_string()))?;
        let control_net = inputs.get("control_net")
            .ok_or_else(|| Error::ExecutionFailed("Missing control_net".to_string()))?;
        let _image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let strength = inputs.get("strength")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;

        debug!("Applying ControlNet with strength {}", strength);

        // 合并 conditioning 与 controlnet 信息
        let combined = match conditioning {
            Value::Conditioning(data) => {
                let mut new_data = data.clone();
                // 标记 controlnet 应用（追加元数据）
                if let Value::ControlNet(cn_path) = control_net {
                    // 简单地在 conditioning 末尾追加 controlnet 强度信息
                    new_data.push(strength);
                }
                new_data
            }
            _ => return Err(Error::TypeError("Expected Conditioning".to_string())),
        };

        Ok(HashMap::from([
            ("CONDITIONING".to_string(), Value::Conditioning(combined)),
        ]))
    }
}

// ============================================================================
// ImageScale 节点 - 缩放图像
// ============================================================================

pub struct ImageScaleNode;

impl Default for ImageScaleNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageScaleNode {
    fn class_type(&self) -> &str {
        "ImageScale"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("width".to_string(), InputType {
                data_type: DataType::INT,
                required: true,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("height".to_string(), InputType {
                data_type: DataType::INT,
                required: true,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("upscale_method".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("nearest-exact".to_string())),
                choices: Some(vec![
                    "nearest-exact".to_string(),
                    "bilinear".to_string(),
                    "area".to_string(),
                    "bicubic".to_string(),
                    "lanczos".to_string(),
                ]),
            }),
            ("crop".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("disabled".to_string())),
                choices: Some(vec![
                    "disabled".to_string(),
                    "center".to_string(),
                ]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let width = inputs.get("width")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let height = inputs.get("height")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let _upscale_method = inputs.get("upscale_method")
            .unwrap_or(&Value::String("nearest-exact".to_string()))
            .as_str()?;
        let _crop = inputs.get("crop")
            .unwrap_or(&Value::String("disabled".to_string()))
            .as_str()?;

        debug!("Scaling image to {}x{}", width, height);

        // 实际实现会调用图像处理库进行缩放
        // 这里我们生成目标尺寸的图像数据占位符
        let scaled_data = match image {
            Value::Image(data) => {
                // 简单的占位符：根据目标尺寸生成新数据
                // 真实实现应使用 image crate 进行插值
                let target_size = width * height * 3;
                if data.len() >= target_size {
                    data[..target_size].to_vec()
                } else {
                    // 上采样：重复数据
                    let mut result = Vec::with_capacity(target_size);
                    while result.len() + data.len() <= target_size {
                        result.extend_from_slice(data);
                    }
                    if result.len() < target_size {
                        let remaining = target_size - result.len();
                        result.extend_from_slice(&data[..remaining.min(data.len())]);
                    }
                    result
                }
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(scaled_data)),
        ]))
    }
}

// ============================================================================
// UpscaleImageWithModel 节点 - 使用模型放大图像
// ============================================================================

pub struct UpscaleImageWithModelNode {
    loaded_models: HashMap<String, String>,
}

impl UpscaleImageWithModelNode {
    pub fn new() -> Self {
        Self {
            loaded_models: HashMap::new(),
        }
    }

    fn find_upscale_model(name: &str) -> Option<String> {
        let search_dirs = ["models/upscale_models", "models/esrgan"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin", "pth"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for UpscaleImageWithModelNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for UpscaleImageWithModelNode {
    fn class_type(&self) -> &str {
        "UpscaleImageWithModel"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("upscale_model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let upscale_model = inputs.get("upscale_model")
            .ok_or_else(|| Error::ExecutionFailed("Missing upscale_model".to_string()))?
            .as_ref_str()?.to_string();
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;

        let model_path = Self::find_upscale_model(&upscale_model)
            .unwrap_or_else(|| upscale_model.clone());

        info!("Upscaling image with model: {}", model_path);
        self.loaded_models.insert(upscale_model, model_path.clone());

        // 模拟 4x 放大（实际实现调用 ESRGAN/Real-ESRGAN）
        let upscaled_data = match image {
            Value::Image(data) => {
                // 占位符：4x 放大
                let mut result = Vec::with_capacity(data.len() * 4);
                for &byte in data.iter() {
                    result.push(byte);
                    result.push(byte);
                }
                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(upscaled_data)),
        ]))
    }
}

// ============================================================================
// ConditioningCombine 节点 - 合并多个 conditioning
// ============================================================================

pub struct ConditioningCombineNode;

impl Default for ConditioningCombineNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ConditioningCombineNode {
    fn class_type(&self) -> &str {
        "ConditioningCombine"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("conditioning_1".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("conditioning_2".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CONDITIONING".to_string(), OutputType {
                data_type: DataType::CONDITIONING,
                name: "CONDITIONING".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let cond1 = inputs.get("conditioning_1")
            .ok_or_else(|| Error::ExecutionFailed("Missing conditioning_1".to_string()))?;
        let cond2 = inputs.get("conditioning_2")
            .ok_or_else(|| Error::ExecutionFailed("Missing conditioning_2".to_string()))?;

        let mut combined = Vec::new();
        if let Value::Conditioning(data) = cond1 {
            combined.extend_from_slice(data);
        }
        if let Value::Conditioning(data) = cond2 {
            combined.extend_from_slice(data);
        }

        Ok(HashMap::from([
            ("CONDITIONING".to_string(), Value::Conditioning(combined)),
        ]))
    }
}

// ============================================================================
// ConditioningConcat 节点 - 拼接 conditioning
// ============================================================================

pub struct ConditioningConcatNode;

impl Default for ConditioningConcatNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ConditioningConcatNode {
    fn class_type(&self) -> &str {
        "ConditioningConcat"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("conditioning_to".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("conditioning_from".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CONDITIONING".to_string(), OutputType {
                data_type: DataType::CONDITIONING,
                name: "CONDITIONING".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let cond_to = inputs.get("conditioning_to")
            .ok_or_else(|| Error::ExecutionFailed("Missing conditioning_to".to_string()))?;
        let cond_from = inputs.get("conditioning_from")
            .ok_or_else(|| Error::ExecutionFailed("Missing conditioning_from".to_string()))?;

        let mut result = Vec::new();
        if let Value::Conditioning(data) = cond_to {
            result.extend_from_slice(data);
        }
        if let Value::Conditioning(data) = cond_from {
            result.extend_from_slice(data);
        }

        Ok(HashMap::from([
            ("CONDITIONING".to_string(), Value::Conditioning(result)),
        ]))
    }
}

// ============================================================================
// CLIPLoader 节点 - 单独加载 CLIP 模型
// ============================================================================

pub struct CLIPLoaderNode {
    loaded_clips: HashMap<String, String>,
}

impl CLIPLoaderNode {
    pub fn new() -> Self {
        Self {
            loaded_clips: HashMap::new(),
        }
    }

    fn find_clip_file(name: &str) -> Option<String> {
        let search_dirs = ["models/clip", "models/clip_vision"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for CLIPLoaderNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for CLIPLoaderNode {
    fn class_type(&self) -> &str {
        "CLIPLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("clip_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("type".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("stable_diffusion".to_string())),
                choices: Some(vec![
                    "stable_diffusion".to_string(),
                    "stable_cascade".to_string(),
                    "sd3".to_string(),
                    "stable_audio".to_string(),
                    "mochi".to_string(),
                    "ltxv".to_string(),
                ]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CLIP".to_string(), OutputType {
                data_type: DataType::CLIP,
                name: "CLIP".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let clip_name = inputs.get("clip_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip_name".to_string()))?
            .as_str()?.to_string();
        let clip_type = inputs.get("type")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("stable_diffusion");

        let clip_path = Self::find_clip_file(&clip_name)
            .unwrap_or_else(|| clip_name.clone());

        info!("Loading CLIP: {} (type: {})", clip_path, clip_type);
        self.loaded_clips.insert(clip_name, clip_path.clone());

        Ok(HashMap::from([
            ("CLIP".to_string(), Value::Clip(clip_path)),
        ]))
    }
}

// ============================================================================
// VAELoader 节点 - 单独加载 VAE 模型
// ============================================================================

pub struct VAELoaderNode {
    loaded_vaes: HashMap<String, String>,
}

impl VAELoaderNode {
    pub fn new() -> Self {
        Self {
            loaded_vaes: HashMap::new(),
        }
    }

    fn find_vae_file(name: &str) -> Option<String> {
        let search_dirs = ["models/vae", "models/vaes"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for VAELoaderNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for VAELoaderNode {
    fn class_type(&self) -> &str {
        "VAELoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("vae_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("VAE".to_string(), OutputType {
                data_type: DataType::VAE,
                name: "VAE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let vae_name = inputs.get("vae_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing vae_name".to_string()))?
            .as_str()?.to_string();

        let vae_path = Self::find_vae_file(&vae_name)
            .unwrap_or_else(|| vae_name.clone());

        info!("Loading VAE: {}", vae_path);
        self.loaded_vaes.insert(vae_name, vae_path.clone());

        Ok(HashMap::from([
            ("VAE".to_string(), Value::Vae(vae_path)),
        ]))
    }
}

// ============================================================================
// UNETLoader 节点 - 加载 UNET 模型（用于 SDXL/SD3/Flux 等）
// ============================================================================

pub struct UNETLoaderNode {
    loaded_unets: HashMap<String, String>,
}

impl UNETLoaderNode {
    pub fn new() -> Self {
        Self {
            loaded_unets: HashMap::new(),
        }
    }

    fn find_unet_file(name: &str) -> Option<String> {
        let search_dirs = ["models/diffusion_models", "models/unet", "models/checkpoints"];
        for dir in &search_dirs {
            let path = std::path::Path::new(dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
            for ext in &["safetensors", "ckpt", "pt", "bin"] {
                let path_with_ext = std::path::Path::new(dir).join(format!("{}.{}", name, ext));
                if path_with_ext.exists() {
                    return Some(path_with_ext.to_string_lossy().into_owned());
                }
            }
        }
        None
    }
}

impl Default for UNETLoaderNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for UNETLoaderNode {
    fn class_type(&self) -> &str {
        "UNETLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("unet_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("weight_dtype".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("default".to_string())),
                choices: Some(vec![
                    "default".to_string(),
                    "fp8_e4m3fn".to_string(),
                    "fp8_e5m2".to_string(),
                ]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("MODEL".to_string(), OutputType {
                data_type: DataType::MODEL,
                name: "MODEL".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let unet_name = inputs.get("unet_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing unet_name".to_string()))?
            .as_str()?.to_string();
        let weight_dtype = inputs.get("weight_dtype")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("default");

        let unet_path = Self::find_unet_file(&unet_name)
            .unwrap_or_else(|| unet_name.clone());

        info!("Loading UNET: {} (weight_dtype: {})", unet_path, weight_dtype);
        self.loaded_unets.insert(unet_name, unet_path.clone());

        Ok(HashMap::from([
            ("MODEL".to_string(), Value::Model(unet_path)),
        ]))
    }
}

// ============================================================================
// DualCLIPLoader 节点 - 加载双 CLIP 模型（SDXL/SD3 需要）
// ============================================================================

pub struct DualCLIPLoaderNode;

impl Default for DualCLIPLoaderNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for DualCLIPLoaderNode {
    fn class_type(&self) -> &str {
        "DualCLIPLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("clip_name1".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("clip_name2".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
            ("type".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("sdxl".to_string())),
                choices: Some(vec![
                    "sdxl".to_string(),
                    "sd3".to_string(),
                    "flux".to_string(),
                ]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CLIP".to_string(), OutputType {
                data_type: DataType::CLIP,
                name: "CLIP".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let clip_name1 = inputs.get("clip_name1")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip_name1".to_string()))?
            .as_str()?.to_string();
        let clip_name2 = inputs.get("clip_name2")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip_name2".to_string()))?
            .as_str()?.to_string();
        let clip_type = inputs.get("type")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("sdxl");

        info!("Loading DualCLIP: {} + {} (type: {})", clip_name1, clip_name2, clip_type);

        let combined = format!("dual:{}+{}", clip_name1, clip_name2);

        Ok(HashMap::from([
            ("CLIP".to_string(), Value::Clip(combined)),
        ]))
    }
}

// ============================================================================
// StyleModelLoader 节点 - 加载风格模型
// ============================================================================

pub struct StyleModelLoaderNode;

impl Default for StyleModelLoaderNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for StyleModelLoaderNode {
    fn class_type(&self) -> &str {
        "StyleModelLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("style_model_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("STYLE_MODEL".to_string(), OutputType {
                data_type: DataType::MODEL,
                name: "STYLE_MODEL".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let style_model_name = inputs.get("style_model_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing style_model_name".to_string()))?
            .as_str()?.to_string();

        let model_path = format!("models/style_models/{}", style_model_name);
        info!("Loading style model: {}", model_path);

        Ok(HashMap::from([
            ("STYLE_MODEL".to_string(), Value::Model(model_path)),
        ]))
    }
}

// ============================================================================
// CLIPVisionEncode 节点 - 编码视觉 CLIP
// ============================================================================

pub struct CLIPVisionEncodeNode;

impl Default for CLIPVisionEncodeNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for CLIPVisionEncodeNode {
    fn class_type(&self) -> &str {
        "CLIPVisionEncode"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("clip_vision".to_string(), InputType {
                data_type: DataType::CLIP,
                required: true,
                default: None,
                choices: None,
            }),
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CLIP_VISION_OUTPUT".to_string(), OutputType {
                data_type: DataType::CLIP,
                name: "CLIP_VISION_OUTPUT".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let clip_vision = inputs.get("clip_vision")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip_vision".to_string()))?
            .as_ref_str()?.to_string();
        let _image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;

        debug!("Encoding CLIP vision with model: {}", clip_vision);

        // 返回 CLIP vision 编码（占位符）
        Ok(HashMap::from([
            ("CLIP_VISION_OUTPUT".to_string(), Value::Clip(format!("vision_encoded:{}", clip_vision))),
        ]))
    }
}

// ============================================================================
// CLIPVisionLoader 节点 - 加载 CLIP Vision 模型
// ============================================================================

pub struct CLIPVisionLoaderNode;

impl Default for CLIPVisionLoaderNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for CLIPVisionLoaderNode {
    fn class_type(&self) -> &str {
        "CLIPVisionLoader"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("clip_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("CLIP_VISION".to_string(), OutputType {
                data_type: DataType::CLIP,
                name: "CLIP_VISION".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let clip_name = inputs.get("clip_name")
            .ok_or_else(|| Error::ExecutionFailed("Missing clip_name".to_string()))?
            .as_str()?.to_string();

        let clip_path = format!("models/clip_vision/{}", clip_name);
        info!("Loading CLIP Vision: {}", clip_path);

        Ok(HashMap::from([
            ("CLIP_VISION".to_string(), Value::Clip(clip_path)),
        ]))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lora_loader() {
        let mut node = LoraLoaderNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("base_model".to_string()));
        inputs.insert("clip".to_string(), Value::Clip("base_clip".to_string()));
        inputs.insert("lora_name".to_string(), Value::String("test_lora".to_string()));
        inputs.insert("strength_model".to_string(), Value::Float(0.8));
        inputs.insert("strength_clip".to_string(), Value::Float(1.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("MODEL"));
        assert!(result.contains_key("CLIP"));

        if let Value::Model(m) = &result["MODEL"] {
            assert!(m.contains("lora:"));
        }
    }

    #[tokio::test]
    async fn test_controlnet_loader() {
        let mut node = ControlNetLoaderNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("control_net_name".to_string(), Value::String("control_v11p_sd15_canny".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CONTROL_NET"));
    }

    #[tokio::test]
    async fn test_controlnet_apply() {
        let mut node = ControlNetApplyNode;
        let mut inputs = HashMap::new();
        inputs.insert("conditioning".to_string(), Value::Conditioning(vec![0.5; 768]));
        inputs.insert("control_net".to_string(), Value::ControlNet("canny_model".to_string()));
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 100]));
        inputs.insert("strength".to_string(), Value::Float(1.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CONDITIONING"));
    }

    #[tokio::test]
    async fn test_image_scale() {
        let mut node = ImageScaleNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 256 * 256 * 3]));
        inputs.insert("width".to_string(), Value::Int(512));
        inputs.insert("height".to_string(), Value::Int(512));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));

        if let Value::Image(data) = &result["IMAGE"] {
            assert_eq!(data.len(), 512 * 512 * 3);
        }
    }

    #[tokio::test]
    async fn test_upscale_image_with_model() {
        let mut node = UpscaleImageWithModelNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("upscale_model".to_string(), Value::Model("4xUltrasharp".to_string()));
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 100]));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));
    }

    #[tokio::test]
    async fn test_conditioning_combine() {
        let mut node = ConditioningCombineNode;
        let mut inputs = HashMap::new();
        inputs.insert("conditioning_1".to_string(), Value::Conditioning(vec![0.5; 10]));
        inputs.insert("conditioning_2".to_string(), Value::Conditioning(vec![0.3; 10]));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Conditioning(data) = &result["CONDITIONING"] {
            assert_eq!(data.len(), 20);
        }
    }

    #[tokio::test]
    async fn test_clip_loader() {
        let mut node = CLIPLoaderNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("clip_name".to_string(), Value::String("clip_vit_l14".to_string()));
        inputs.insert("type".to_string(), Value::String("stable_diffusion".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CLIP"));
    }

    #[tokio::test]
    async fn test_vae_loader() {
        let mut node = VAELoaderNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("vae_name".to_string(), Value::String("sdxl_vae".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("VAE"));
    }

    #[tokio::test]
    async fn test_unet_loader() {
        let mut node = UNETLoaderNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("unet_name".to_string(), Value::String("sd3_medium".to_string()));
        inputs.insert("weight_dtype".to_string(), Value::String("default".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("MODEL"));
    }

    #[tokio::test]
    async fn test_dual_clip_loader() {
        let mut node = DualCLIPLoaderNode;
        let mut inputs = HashMap::new();
        inputs.insert("clip_name1".to_string(), Value::String("clip_g".to_string()));
        inputs.insert("clip_name2".to_string(), Value::String("clip_l".to_string()));
        inputs.insert("type".to_string(), Value::String("sdxl".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CLIP"));

        if let Value::Clip(c) = &result["CLIP"] {
            assert!(c.starts_with("dual:"));
        }
    }

    #[tokio::test]
    async fn test_style_model_loader() {
        let mut node = StyleModelLoaderNode;
        let mut inputs = HashMap::new();
        inputs.insert("style_model_name".to_string(), Value::String("style_model_test".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("STYLE_MODEL"));
    }

    #[tokio::test]
    async fn test_clip_vision_encode() {
        let mut node = CLIPVisionEncodeNode;
        let mut inputs = HashMap::new();
        inputs.insert("clip_vision".to_string(), Value::Clip("clip_vision_model".to_string()));
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 100]));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CLIP_VISION_OUTPUT"));
    }

    #[tokio::test]
    async fn test_clip_vision_loader() {
        let mut node = CLIPVisionLoaderNode;
        let mut inputs = HashMap::new();
        inputs.insert("clip_name".to_string(), Value::String("clip_vision_vit_h".to_string()));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("CLIP_VISION"));
    }

    #[test]
    fn test_extended_node_class_types() {
        assert_eq!(LoraLoaderNode::new().class_type(), "LoraLoader");
        assert_eq!(ControlNetLoaderNode::new().class_type(), "ControlNetLoader");
        assert_eq!(ControlNetApplyNode.class_type(), "ControlNetApply");
        assert_eq!(ImageScaleNode.class_type(), "ImageScale");
        assert_eq!(UpscaleImageWithModelNode::new().class_type(), "UpscaleImageWithModel");
        assert_eq!(ConditioningCombineNode.class_type(), "ConditioningCombine");
        assert_eq!(ConditioningConcatNode.class_type(), "ConditioningConcat");
        assert_eq!(CLIPLoaderNode::new().class_type(), "CLIPLoader");
        assert_eq!(VAELoaderNode::new().class_type(), "VAELoader");
        assert_eq!(UNETLoaderNode::new().class_type(), "UNETLoader");
        assert_eq!(DualCLIPLoaderNode.class_type(), "DualCLIPLoader");
        assert_eq!(StyleModelLoaderNode.class_type(), "StyleModelLoader");
        assert_eq!(CLIPVisionEncodeNode.class_type(), "CLIPVisionEncode");
        assert_eq!(CLIPVisionLoaderNode.class_type(), "CLIPVisionLoader");
    }
}
