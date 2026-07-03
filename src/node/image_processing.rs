// 图片处理节点
// 包括 ImageBlend、ImageCrop、ImageRotate、ImageColorAdjust、ImageFilter 等
// 使用 image crate 进行图像处理

use crate::types::*;
use crate::node::{Node, InputType, OutputType};
use async_trait::async_trait;
use std::collections::HashMap;
use log::debug;
use image::{ImageBuffer, Rgb, DynamicImage, GenericImageView};

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
            .as_int()? as f32;
        let strength = inputs.get("strength")
            .unwrap_or(&Value::Float(1.0))
            .as_float()? as f32;

        debug!("Applying filter: type={}, radius={}, strength={}", filter_type, radius, strength);

        let filtered = match image {
            Value::Image(data) => {
                // 从 RGB 数据创建 image buffer
                let src_size = (data.len() as f64 / 3.0).sqrt() as usize;
                let width = src_size;
                let height = src_size;
                
                // 将 Vec<u8> 转换为 ImageBuffer
                let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = 
                    ImageBuffer::from_raw(width as u32, height as u32, data.clone())
                        .ok_or_else(|| Error::ExecutionFailed("Failed to create image buffer".to_string()))?;
                
                let dynamic_img = DynamicImage::from(img_buffer);
                
                // 使用 image crate 进行滤镜处理
                let result_img = match filter_type {
                    "gaussian_blur" | "box_blur" => {
                        // 高斯模糊/盒模糊 - 使用 image crate 的 blur
                        let blurred = dynamic_img.blur(radius);
                        if strength < 1.0 {
                            blend_images(&dynamic_img, &blurred, strength)
                        } else {
                            blurred
                        }
                    }
                    "sharpen" => {
                        // 锐化 - 先模糊，然后 original + (original - blurred) * strength
                        let blurred = dynamic_img.blur(radius);
                        sharpen_image(&dynamic_img, &blurred, strength)
                    }
                    "edge_detect" => {
                        // 边缘检测 - 使用 Sobel 算子
                        edge_detect_image(&dynamic_img, strength)
                    }
                    "emboss" => {
                        // 浮雕效果
                        emboss_image(&dynamic_img, strength)
                    }
                    "median" => {
                        // 中值滤波 - 用于降噪
                        median_filter(&dynamic_img, radius as usize)
                    }
                    "grayscale" => {
                        // 灰度化
                        DynamicImage::from(dynamic_img.grayscale())
                    }
                    "invert" => {
                        // 反转颜色
                        invert_image(&dynamic_img)
                    }
                    _ => {
                        // 默认：不处理
                        dynamic_img
                    }
                };
                
                // 将结果转换回 Vec<u8>
                result_img.to_rgb8().into_raw()
            }
            _ => return Err(Error::TypeError("Expected IMAGE".to_string())),
        };

        Ok(HashMap::from([
            ("IMAGE".to_string(), Value::Image(filtered)),
        ]))
    }
}

// ============================================================================
// 滤镜辅助函数
// ============================================================================

/// 混合两张图像
fn blend_images(base: &DynamicImage, overlay: &DynamicImage, alpha: f32) -> DynamicImage {
    let (width, height) = base.dimensions();
    let mut result = base.to_rgb8();
    
    for y in 0..height {
        for x in 0..width {
            let base_pixel = base.get_pixel(x, y);
            let overlay_pixel = overlay.get_pixel(x, y);
            
            let r = (base_pixel.0[0] as f32 * (1.0 - alpha) + overlay_pixel.0[0] as f32 * alpha) as u8;
            let g = (base_pixel.0[1] as f32 * (1.0 - alpha) + overlay_pixel.0[1] as f32 * alpha) as u8;
            let b = (base_pixel.0[2] as f32 * (1.0 - alpha) + overlay_pixel.0[2] as f32 * alpha) as u8;
            
            result.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    
    DynamicImage::from(result)
}

/// 锐化图像
fn sharpen_image(original: &DynamicImage, blurred: &DynamicImage, strength: f32) -> DynamicImage {
    let (width, height) = original.dimensions();
    let mut result = original.to_rgb8();
    
    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let blur_pixel = blurred.get_pixel(x, y);
            
            for c in 0..3 {
                let orig = orig_pixel.0[c] as f32;
                let blur = blur_pixel.0[c] as f32;
                let sharpened = orig + (orig - blur) * strength;
                result.get_pixel_mut(x, y).0[c] = sharpened.clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    DynamicImage::from(result)
}

/// 边缘检测（Sobel 算子）
fn edge_detect_image(img: &DynamicImage, strength: f32) -> DynamicImage {
    let (width, height) = img.dimensions();
    let mut result = ImageBuffer::new(width, height);
    
    // Sobel 算子
    let gx_kernel: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let gy_kernel: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let px = x + kx as u32 - 1;
                    let py = y + ky as u32 - 1;
                    let pixel = img.get_pixel(px, py);
                    // 转换为灰度
                    let gray = 0.299 * pixel.0[0] as f32 + 0.587 * pixel.0[1] as f32 + 0.114 * pixel.0[2] as f32;
                    
                    gx += gray * gx_kernel[ky][kx];
                    gy += gray * gy_kernel[ky][kx];
                }
            }
            
            let magnitude = (gx * gx + gy * gy).sqrt() * strength;
            let edge = magnitude.min(255.0) as u8;
            
            result.put_pixel(x, y, Rgb([edge, edge, edge]));
        }
    }
    
    DynamicImage::from(result)
}

/// 浮雕效果
fn emboss_image(img: &DynamicImage, strength: f32) -> DynamicImage {
    let (width, height) = img.dimensions();
    let mut result = ImageBuffer::new(width, height);
    
    // 浮雕核
    let kernel: [[f32; 3]; 3] = [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]];
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let mut sum = 0.0;
            
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let px = x + kx as u32 - 1;
                    let py = y + ky as u32 - 1;
                    let pixel = img.get_pixel(px, py);
                    let gray = 0.299 * pixel.0[0] as f32 + 0.587 * pixel.0[1] as f32 + 0.114 * pixel.0[2] as f32;
                    sum += gray * kernel[ky][kx];
                }
            }
            
            let val = (sum * strength + 128.0).clamp(0.0, 255.0) as u8;
            result.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    
    DynamicImage::from(result)
}

/// 中值滤波
fn median_filter(img: &DynamicImage, radius: usize) -> DynamicImage {
    let (width, height) = img.dimensions();
    let mut result = ImageBuffer::new(width, height);
    let r = radius.max(1);
    
    for y in 0..height {
        for x in 0..width {
            let mut reds = Vec::new();
            let mut greens = Vec::new();
            let mut blues = Vec::new();
            
            for dy in -(r as i32)..=(r as i32) {
                for dx in -(r as i32)..=(r as i32) {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let pixel = img.get_pixel(nx, ny);
                    reds.push(pixel.0[0]);
                    greens.push(pixel.0[1]);
                    blues.push(pixel.0[2]);
                }
            }
            
            reds.sort();
            greens.sort();
            blues.sort();
            
            let mid = reds.len() / 2;
            result.put_pixel(x, y, Rgb([reds[mid], greens[mid], blues[mid]]));
        }
    }
    
    DynamicImage::from(result)
}

/// 反转颜色
fn invert_image(img: &DynamicImage) -> DynamicImage {
    let (width, height) = img.dimensions();
    let mut result = img.to_rgb8();
    
    for y in 0..height {
        for x in 0..width {
            let pixel = result.get_pixel(x, y);
            result.put_pixel(x, y, Rgb([
                255 - pixel.0[0],
                255 - pixel.0[1],
                255 - pixel.0[2],
            ]));
        }
    }
    
    DynamicImage::from(result)
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
