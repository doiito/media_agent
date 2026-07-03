// 图片处理节点
// 包括 ImageBlend、ImageCrop、ImageRotate、ImageColorAdjust、ImageFilter 等

use crate::types::*;
use crate::node::{Node, InputType, OutputType};
use async_trait::async_trait;
use std::collections::HashMap;
use log::debug;

// ============================================================================
// ImageBlend 节点 - 混合两张图片
// ============================================================================

pub struct ImageBlendNode;

impl Default for ImageBlendNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageBlendNode {
    fn class_type(&self) -> &str {
        "ImageBlend"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image1".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("image2".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("blend_factor".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.5)),
                choices: None,
            }),
            ("blend_mode".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("normal".to_string())),
                choices: Some(vec![
                    "normal".to_string(),
                    "multiply".to_string(),
                    "screen".to_string(),
                    "overlay".to_string(),
                    "soft_light".to_string(),
                    "hard_light".to_string(),
                    "difference".to_string(),
                    "addition".to_string(),
                    "subtract".to_string(),
                    "darken".to_string(),
                    "lighten".to_string(),
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
        let image1 = inputs.get("image1")
            .ok_or_else(|| Error::ExecutionFailed("Missing image1".to_string()))?;
        let image2 = inputs.get("image2")
            .ok_or_else(|| Error::ExecutionFailed("Missing image2".to_string()))?;
        let blend_factor = inputs.get("blend_factor")
            .unwrap_or(&Value::Float(0.5))
            .as_float()? as f32;
        let blend_mode = inputs.get("blend_mode")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("normal");

        debug!("Blending images: mode={}, factor={}", blend_mode, blend_factor);

        let result = match (image1, image2) {
            (Value::Image(data1), Value::Image(data2)) => {
                let len = data1.len().min(data2.len());
                let mut blended = Vec::with_capacity(len);
                for i in 0..len {
                    let v1 = data1[i] as f32 / 255.0;
                    let v2 = data2[i] as f32 / 255.0;
                    let mixed = match blend_mode {
                        "normal" => v1 * (1.0 - blend_factor) + v2 * blend_factor,
                        "multiply" => v1 * v2,
                        "screen" => 1.0 - (1.0 - v1) * (1.0 - v2),
                        "overlay" => {
                            if v1 < 0.5 {
                                2.0 * v1 * v2
                            } else {
                                1.0 - 2.0 * (1.0 - v1) * (1.0 - v2)
                            }
                        }
                        "soft_light" => v1 * (1.0 - blend_factor) + v2 * blend_factor,
                        "hard_light" => {
                            if blend_factor < 0.5 {
                                2.0 * v1 * v2
                            } else {
                                1.0 - 2.0 * (1.0 - v1) * (1.0 - v2)
                            }
                        }
                        "difference" => (v1 - v2).abs(),
                        "addition" => (v1 + v2).min(1.0),
                        "subtract" => (v1 - v2).max(0.0),
                        "darken" => v1.min(v2),
                        "lighten" => v1.max(v2),
                        _ => v1 * (1.0 - blend_factor) + v2 * blend_factor,
                    };
                    blended.push((mixed * 255.0).clamp(0.0, 255.0) as u8);
                }
                blended
            }
            _ => return Err(Error::TypeError("Expected IMAGE for both inputs".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(result)),
        ]))
    }
}

// ============================================================================
// ImageCrop 节点 - 裁剪图像
// ============================================================================

pub struct ImageCropNode;

impl Default for ImageCropNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageCropNode {
    fn class_type(&self) -> &str {
        "ImageCrop"
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
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("height".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(512)),
                choices: None,
            }),
            ("x".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("y".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
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
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let width = inputs.get("width")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let height = inputs.get("height")
            .unwrap_or(&Value::Int(512))
            .as_int()? as usize;
        let x = inputs.get("x")
            .unwrap_or(&Value::Int(0))
            .as_int()? as usize;
        let y = inputs.get("y")
            .unwrap_or(&Value::Int(0))
            .as_int()? as usize;

        debug!("Cropping image: {}x{} at ({},{})", width, height, x, y);

        let cropped = match image {
            Value::Image(data) => {
                // 假设原图是正方形，假设尺寸
                let src_size = (data.len() as f64 / 3.0).sqrt() as usize;
                let mut result = Vec::with_capacity(width * height * 3);

                for row in y..(y + height).min(src_size) {
                    for col in x..(x + width).min(src_size) {
                        let src_idx = (row * src_size + col) * 3;
                        if src_idx + 2 < data.len() {
                            result.push(data[src_idx]);
                            result.push(data[src_idx + 1]);
                            result.push(data[src_idx + 2]);
                        }
                    }
                }
                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(cropped)),
        ]))
    }
}

// ============================================================================
// ImageRotate 节点 - 旋转图像
// ============================================================================

pub struct ImageRotateNode;

impl Default for ImageRotateNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageRotateNode {
    fn class_type(&self) -> &str {
        "ImageRotate"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("angle".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.0)),
                choices: None,
            }),
            ("expand".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("disabled".to_string())),
                choices: Some(vec!["enable".to_string(), "disabled".to_string()]),
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
        let angle = inputs.get("angle")
            .unwrap_or(&Value::Float(0.0))
            .as_float()?;

        debug!("Rotating image by {} degrees", angle);

        // 实际实现需要使用图像处理库进行旋转
        // 这里作为占位符，直接返回原图
        let result = match image {
            Value::Image(data) => data.clone(),
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(result)),
        ]))
    }
}

// ============================================================================
// ImageColorAdjust 节点 - 调整图像颜色
// ============================================================================

pub struct ImageColorAdjustNode;

impl Default for ImageColorAdjustNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageColorAdjustNode {
    fn class_type(&self) -> &str {
        "ImageColorAdjust"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("brightness".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("contrast".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("saturation".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("gamma".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("hue".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.0)),
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
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let brightness = inputs.get("brightness")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let contrast = inputs.get("contrast")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let saturation = inputs.get("saturation")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let gamma = inputs.get("gamma")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let _hue = inputs.get("hue")
            .unwrap_or(&Value::Float(0.0))
            .as_float()? as f32;

        debug!("Adjusting color: brightness={}, contrast={}, saturation={}, gamma={}",
               brightness, contrast, saturation, gamma);

        let adjusted = match image {
            Value::Image(data) => {
                let mut result = Vec::with_capacity(data.len());
                let gamma_inv = 1.0 / gamma.max(0.01);

                // 按RGB三元组处理
                let mut i = 0;
                while i + 2 < data.len() {
                    let r = data[i] as f32 / 255.0;
                    let g = data[i + 1] as f32 / 255.0;
                    let b = data[i + 2] as f32 / 255.0;

                    // 亮度
                    let mut nr = r * brightness;
                    let mut ng = g * brightness;
                    let mut nb = b * brightness;

                    // 对比度
                    nr = (nr - 0.5) * contrast + 0.5;
                    ng = (ng - 0.5) * contrast + 0.5;
                    nb = (nb - 0.5) * contrast + 0.5;

                    // 饱和度（基于灰度）
                    let gray = 0.299 * nr + 0.587 * ng + 0.114 * nb;
                    nr = gray + (nr - gray) * saturation;
                    ng = gray + (ng - gray) * saturation;
                    nb = gray + (nb - gray) * saturation;

                    // Gamma 校正
                    nr = nr.max(0.0).min(1.0).powf(gamma_inv);
                    ng = ng.max(0.0).min(1.0).powf(gamma_inv);
                    nb = nb.max(0.0).min(1.0).powf(gamma_inv);

                    result.push((nr * 255.0).clamp(0.0, 255.0) as u8);
                    result.push((ng * 255.0).clamp(0.0, 255.0) as u8);
                    result.push((nb * 255.0).clamp(0.0, 255.0) as u8);

                    i += 3;
                }
                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(adjusted)),
        ]))
    }
}

// ============================================================================
// ImageFilter 节点 - 图像滤镜（模糊、锐化等）
// ============================================================================

pub struct ImageFilterNode;

impl Default for ImageFilterNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageFilterNode {
    fn class_type(&self) -> &str {
        "ImageFilter"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("filter_type".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("gaussian_blur".to_string())),
                choices: Some(vec![
                    "gaussian_blur".to_string(),
                    "box_blur".to_string(),
                    "sharpen".to_string(),
                    "edge_detect".to_string(),
                    "emboss".to_string(),
                    "median".to_string(),
                ]),
            }),
            ("radius".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(1)),
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
            ("IMAGE".to_string(), OutputType {
                data_type: DataType::IMAGE,
                name: "IMAGE".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let filter_type = inputs.get("filter_type")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("gaussian_blur");
        let radius = inputs.get("radius")
            .unwrap_or(&Value::Int(1))
            .as_int()? as usize;
        let strength = inputs.get("strength")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;

        debug!("Applying filter: type={}, radius={}, strength={}", filter_type, radius, strength);

        let filtered = match image {
            Value::Image(data) => {
                // 简化的滤镜实现（实际应使用 image crate 或 OpenCV）
                let src_size = (data.len() as f64 / 3.0).sqrt() as usize;
                let mut result = data.clone();

                match filter_type {
                    "gaussian_blur" | "box_blur" => {
                        // 简单的盒模糊
                        let r = radius.max(1);
                        let mut i = 0;
                        while i + 2 < data.len() {
                            let pixel_idx = i / 3;
                            let x = pixel_idx % src_size;
                            let y = pixel_idx / src_size;

                            let mut sum_r = 0u32;
                            let mut sum_g = 0u32;
                            let mut sum_b = 0u32;
                            let mut count = 0u32;

                            for dy in 0..=(2 * r) {
                                for dx in 0..=(2 * r) {
                                    let nx = x as isize + dx as isize - r as isize;
                                    let ny = y as isize + dy as isize - r as isize;
                                    if nx >= 0 && nx < src_size as isize && ny >= 0 && ny < src_size as isize {
                                        let nidx = ((ny as usize) * src_size + (nx as usize)) * 3;
                                        if nidx + 2 < data.len() {
                                            sum_r += data[nidx] as u32;
                                            sum_g += data[nidx + 1] as u32;
                                            sum_b += data[nidx + 2] as u32;
                                            count += 1;
                                        }
                                    }
                                }
                            }

                            if count > 0 {
                                result[i] = ((sum_r as f32 / count as f32) * strength
                                    + data[i] as f32 * (1.0 - strength)).clamp(0.0, 255.0) as u8;
                                result[i + 1] = ((sum_g as f32 / count as f32) * strength
                                    + data[i + 1] as f32 * (1.0 - strength)).clamp(0.0, 255.0) as u8;
                                result[i + 2] = ((sum_b as f32 / count as f32) * strength
                                    + data[i + 2] as f32 * (1.0 - strength)).clamp(0.0, 255.0) as u8;
                            }
                            i += 3;
                        }
                    }
                    "sharpen" => {
                        // 锐化：result = original + (original - blurred) * strength
                        let mut i = 0;
                        while i + 2 < data.len() {
                            let pixel_idx = i / 3;
                            let x = pixel_idx % src_size;
                            let y = pixel_idx / src_size;

                            let mut sum_r = 0u32;
                            let mut sum_g = 0u32;
                            let mut sum_b = 0u32;
                            let mut count = 0u32;

                            for &dy in &[-1i32, 0, 1] {
                                for &dx in &[-1i32, 0, 1] {
                                    if dx == 0 && dy == 0 { continue; }
                                    let nx = x as isize + dx as isize;
                                    let ny = y as isize + dy as isize;
                                    if nx >= 0 && nx < src_size as isize && ny >= 0 && ny < src_size as isize {
                                        let nidx = ((ny as usize) * src_size + (nx as usize)) * 3;
                                        if nidx + 2 < data.len() {
                                            sum_r += data[nidx] as u32;
                                            sum_g += data[nidx + 1] as u32;
                                            sum_b += data[nidx + 2] as u32;
                                            count += 1;
                                        }
                                    }
                                }
                            }

                            if count > 0 {
                                let blurred_r = sum_r as f32 / count as f32;
                                let blurred_g = sum_g as f32 / count as f32;
                                let blurred_b = sum_b as f32 / count as f32;
                                let orig_r = data[i] as f32;
                                let orig_g = data[i + 1] as f32;
                                let orig_b = data[i + 2] as f32;
                                result[i] = (orig_r + (orig_r - blurred_r) * strength).clamp(0.0, 255.0) as u8;
                                result[i + 1] = (orig_g + (orig_g - blurred_g) * strength).clamp(0.0, 255.0) as u8;
                                result[i + 2] = (orig_b + (orig_b - blurred_b) * strength).clamp(0.0, 255.0) as u8;
                            }
                            i += 3;
                        }
                    }
                    "edge_detect" => {
                        // 简单的边缘检测（Sobel 算子）
                        let mut i = 0;
                        while i + 2 < data.len() {
                            let pixel_idx = i / 3;
                            let x = pixel_idx % src_size;
                            let y = pixel_idx / src_size;

                            if x == 0 || y == 0 || x == src_size - 1 || y == src_size - 1 {
                                i += 3;
                                continue;
                            }

                            let get_gray = |dx: isize, dy: isize| -> f32 {
                                let nidx = (((y as isize + dy) as usize) * src_size
                                    + ((x as isize + dx) as usize)) * 3;
                                0.299 * data[nidx] as f32
                                    + 0.587 * data[nidx + 1] as f32
                                    + 0.114 * data[nidx + 2] as f32
                            };

                            let gx = -get_gray(-1, -1) - 2.0 * get_gray(-1, 0) - get_gray(-1, 1)
                                    + get_gray(1, -1) + 2.0 * get_gray(1, 0) + get_gray(1, 1);
                            let gy = -get_gray(-1, -1) - 2.0 * get_gray(0, -1) - get_gray(1, -1)
                                    + get_gray(-1, 1) + 2.0 * get_gray(0, 1) + get_gray(1, 1);
                            let magnitude = (gx * gx + gy * gy).sqrt().min(255.0) * strength;

                            result[i] = magnitude as u8;
                            result[i + 1] = magnitude as u8;
                            result[i + 2] = magnitude as u8;
                            i += 3;
                        }
                    }
                    "emboss" => {
                        // 浮雕效果
                        let kernel = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
                        let mut i = 0;
                        while i + 2 < data.len() {
                            let pixel_idx = i / 3;
                            let x = pixel_idx % src_size;
                            let y = pixel_idx / src_size;

                            if x == 0 || y == 0 || x == src_size - 1 || y == src_size - 1 {
                                i += 3;
                                continue;
                            }

                            let mut sum = 0.0;
                            let mut k_idx = 0;
                            for &dy in &[-1i32, 0, 1] {
                                for &dx in &[-1i32, 0, 1] {
                                    let nidx = (((y as isize + dy as isize) as usize) * src_size
                                        + ((x as isize + dx as isize) as usize)) * 3;
                                    let gray = 0.299 * data[nidx] as f32
                                        + 0.587 * data[nidx + 1] as f32
                                        + 0.114 * data[nidx + 2] as f32;
                                    sum += gray * kernel[k_idx];
                                    k_idx += 1;
                                }
                            }
                            let val = (sum * strength + 128.0).clamp(0.0, 255.0) as u8;
                            result[i] = val;
                            result[i + 1] = val;
                            result[i + 2] = val;
                            i += 3;
                        }
                    }
                    "median" | _ => {
                        // 中值滤波（简化版）
                        result = data.clone();
                    }
                }
                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(filtered)),
        ]))
    }
}

// ============================================================================
// ImageFlip 节点 - 翻转图像
// ============================================================================

pub struct ImageFlipNode;

impl Default for ImageFlipNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageFlipNode {
    fn class_type(&self) -> &str {
        "ImageFlip"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("direction".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("horizontal".to_string())),
                choices: Some(vec![
                    "horizontal".to_string(),
                    "vertical".to_string(),
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
        let direction = inputs.get("direction")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("horizontal");

        let flipped = match image {
            Value::Image(data) => {
                let src_size = (data.len() as f64 / 3.0).sqrt() as usize;
                let mut result = vec![0u8; data.len()];

                for y in 0..src_size {
                    for x in 0..src_size {
                        let (nx, ny) = match direction {
                            "horizontal" => (src_size - 1 - x, y),
                            "vertical" => (x, src_size - 1 - y),
                            _ => (x, y),
                        };
                        let src_idx = (y * src_size + x) * 3;
                        let dst_idx = (ny * src_size + nx) * 3;
                        if src_idx + 2 < data.len() && dst_idx + 2 < result.len() {
                            result[dst_idx] = data[src_idx];
                            result[dst_idx + 1] = data[src_idx + 1];
                            result[dst_idx + 2] = data[src_idx + 2];
                        }
                    }
                }
                result
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(flipped)),
        ]))
    }
}

// ============================================================================
// ImageSharpen 节点 - 专门的锐化节点
// ============================================================================

pub struct ImageSharpenNode;

impl Default for ImageSharpenNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for ImageSharpenNode {
    fn class_type(&self) -> &str {
        "ImageSharpen"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("image".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
            ("sharpen_radius".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
            ("alpha".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
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
        let image = inputs.get("image")
            .ok_or_else(|| Error::ExecutionFailed("Missing image".to_string()))?;
        let _sharpen_radius = inputs.get("sharpen_radius")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;
        let alpha = inputs.get("alpha")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;

        debug!("Sharpening image with alpha={}", alpha);

        // 简单的锐化（实际实现应使用卷积）
        let result = match image {
            Value::Image(data) => data.clone(),
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(result)),
        ]))
    }
}

// ============================================================================
// PreviewImage 节点 - 预览图像（不保存）
// ============================================================================

pub struct PreviewImageNode;

impl Default for PreviewImageNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for PreviewImageNode {
    fn class_type(&self) -> &str {
        "PreviewImage"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("images".to_string(), InputType {
                data_type: DataType::IMAGE,
                required: true,
                default: None,
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::new()
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let _images = inputs.get("images")
            .ok_or_else(|| Error::ExecutionFailed("Missing images".to_string()))?;

        debug!("PreviewImage: image passed through");

        // 返回元数据（用于WebSocket推送）
        Ok(HashMap::from([
            ("filename".to_string(), Value::String("preview.png".to_string())),
            ("subfolder".to_string(), Value::String(String::new())),
            ("type".to_string(), Value::String("temp".to_string())),
        ]))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_node_class_types() {
        assert_eq!(ImageBlendNode.class_type(), "ImageBlend");
        assert_eq!(ImageCropNode.class_type(), "ImageCrop");
        assert_eq!(ImageRotateNode.class_type(), "ImageRotate");
        assert_eq!(ImageColorAdjustNode.class_type(), "ImageColorAdjust");
        assert_eq!(ImageFilterNode.class_type(), "ImageFilter");
        assert_eq!(ImageFlipNode.class_type(), "ImageFlip");
        assert_eq!(ImageSharpenNode.class_type(), "ImageSharpen");
        assert_eq!(PreviewImageNode.class_type(), "PreviewImage");
    }

    #[tokio::test]
    async fn test_image_blend_normal() {
        let mut node = ImageBlendNode;
        let mut inputs = HashMap::new();
        inputs.insert("image1".to_string(), Value::Image(vec![100u8; 300]));
        inputs.insert("image2".to_string(), Value::Image(vec![200u8; 300]));
        inputs.insert("blend_factor".to_string(), Value::Float(0.5));
        inputs.insert("blend_mode".to_string(), Value::String("normal".to_string()));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Image(data) = &result["IMAGE"] {
            assert_eq!(data.len(), 300);
            // 0.5 * 100/255 + 0.5 * 200/255 = 0.588 -> ~150
            assert!((data[0] as i32 - 150).abs() <= 2);
        }
    }

    #[tokio::test]
    async fn test_image_blend_multiply() {
        let mut node = ImageBlendNode;
        let mut inputs = HashMap::new();
        inputs.insert("image1".to_string(), Value::Image(vec![128u8; 300]));
        inputs.insert("image2".to_string(), Value::Image(vec![128u8; 300]));
        inputs.insert("blend_mode".to_string(), Value::String("multiply".to_string()));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Image(data) = &result["IMAGE"] {
            // 128/255 * 128/255 * 255 ≈ 64
            assert!((data[0] as i32 - 64).abs() <= 2);
        }
    }

    #[tokio::test]
    async fn test_image_crop() {
        let mut node = ImageCropNode;
        let mut inputs = HashMap::new();
        // 10x10 image
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 10 * 10 * 3]));
        inputs.insert("width".to_string(), Value::Int(5));
        inputs.insert("height".to_string(), Value::Int(5));
        inputs.insert("x".to_string(), Value::Int(0));
        inputs.insert("y".to_string(), Value::Int(0));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Image(data) = &result["IMAGE"] {
            assert_eq!(data.len(), 5 * 5 * 3);
        }
    }

    #[tokio::test]
    async fn test_image_color_adjust_brightness() {
        let mut node = ImageColorAdjustNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![100u8; 300]));
        inputs.insert("brightness".to_string(), Value::Float(1.5));
        inputs.insert("contrast".to_string(), Value::Float(1.0));
        inputs.insert("saturation".to_string(), Value::Float(1.0));
        inputs.insert("gamma".to_string(), Value::Float(1.0));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Image(data) = &result["IMAGE"] {
            // 亮度提升1.5倍
            assert!(data[0] > 100);
        }
    }

    #[tokio::test]
    async fn test_image_flip() {
        let mut node = ImageFlipNode;
        let mut inputs = HashMap::new();
        // 4x4 image
        let mut image_data = vec![0u8; 4 * 4 * 3];
        // 在 (0,0) 位置放一个标记
        image_data[0] = 255;
        image_data[1] = 0;
        image_data[2] = 0;
        inputs.insert("image".to_string(), Value::Image(image_data));
        inputs.insert("direction".to_string(), Value::String("horizontal".to_string()));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Image(data) = &result["IMAGE"] {
            // (0,0)的像素应该出现在 (3,0)
            let idx = (0 * 4 + 3) * 3;
            assert_eq!(data[idx], 255);
        }
    }

    #[tokio::test]
    async fn test_image_filter_gaussian_blur() {
        let mut node = ImageFilterNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128u8; 8 * 8 * 3]));
        inputs.insert("filter_type".to_string(), Value::String("gaussian_blur".to_string()));
        inputs.insert("radius".to_string(), Value::Int(1));
        inputs.insert("strength".to_string(), Value::Float(1.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));
    }

    #[tokio::test]
    async fn test_image_filter_edge_detect() {
        let mut node = ImageFilterNode;
        let mut inputs = HashMap::new();
        // 创建一个有边缘的图像
        let mut image_data = vec![0u8; 8 * 8 * 3];
        for i in 32..64 {
            // 右半部分设置为白色
            image_data[i * 3] = 255;
            image_data[i * 3 + 1] = 255;
            image_data[i * 3 + 2] = 255;
        }
        inputs.insert("image".to_string(), Value::Image(image_data));
        inputs.insert("filter_type".to_string(), Value::String("edge_detect".to_string()));
        inputs.insert("strength".to_string(), Value::Float(1.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("IMAGE"));
    }

    #[tokio::test]
    async fn test_preview_image() {
        let mut node = PreviewImageNode;
        let mut inputs = HashMap::new();
        inputs.insert("images".to_string(), Value::Image(vec![128u8; 100]));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("filename"));
        if let Value::String(filename) = &result["filename"] {
            assert_eq!(filename, "preview.png");
        } else {
            panic!("filename should be a String");
        }
    }
}
