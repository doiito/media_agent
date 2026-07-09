// 后端路由器
// 按模型类型自动选择最优推理后端

use crate::backend::{
    BackendType, StableDiffusionCppBackend, LlamaCppBackend,
    SdCppConfig, LlamaCppConfig,
    T2IParams, I2IParams, T2VParams, I2VParams,
};
use crate::types::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, warn};

/// 后端路由器
/// 根据模型类型自动选择合适的推理后端
pub struct BackendRouter {
    /// stable-diffusion.cpp 后端
    sd_cpp: Option<Arc<StableDiffusionCppBackend>>,
    /// llama.cpp 后端
    llama_cpp: Option<Arc<LlamaCppBackend>>,
    /// 系统状态
    system_stats: Arc<RwLock<SystemStats>>,
}

impl BackendRouter {
    pub fn new() -> Self {
        Self {
            sd_cpp: None,
            llama_cpp: None,
            system_stats: Arc::new(RwLock::new(SystemStats { devices: vec![] })),
        }
    }

    /// 使用配置创建路由器
    pub fn with_configs(sd_config: SdCppConfig, llama_config: LlamaCppConfig) -> Self {
        let sd_cpp = Arc::new(StableDiffusionCppBackend::new(sd_config));
        let llama_cpp = Arc::new(LlamaCppBackend::new(llama_config));

        Self {
            sd_cpp: Some(sd_cpp),
            llama_cpp: Some(llama_cpp),
            system_stats: Arc::new(RwLock::new(SystemStats { devices: vec![] })),
        }
    }

    /// 从环境变量创建路由器
    pub fn from_env() -> Self {
        let sd_config = SdCppConfig::from_env();
        let llama_config = LlamaCppConfig::from_env();
        Self::with_configs(sd_config, llama_config)
    }

    /// 选择后端
    pub fn select_backend(&self, model_type: &str) -> BackendType {
        match model_type {
            "diffusion" | "stable_diffusion" | "sd" => BackendType::StableDiffusionCpp,
            "llm" | "text_encoder" | "t5" | "qwen" | "llama" => BackendType::LlamaCpp,
            "clip" | "vae" => BackendType::LocalProcessor,
            _ => BackendType::StableDiffusionCpp,
        }
    }

    /// 文生图
    pub async fn text_to_image(&self, params: T2IParams) -> Result<Vec<u8>, Error> {
        let backend = self.sd_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("stable-diffusion.cpp backend not configured".to_string())
        })?;

        backend.text_to_image(params).await.map_err(|e| {
            Error::BackendError(format!("T2I failed: {}", e))
        })
    }

    /// 图生图
    pub async fn image_to_image(&self, params: I2IParams) -> Result<Vec<u8>, Error> {
        let backend = self.sd_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("stable-diffusion.cpp backend not configured".to_string())
        })?;

        backend.image_to_image(params).await.map_err(|e| {
            Error::BackendError(format!("I2I failed: {}", e))
        })
    }

    /// 文生视频（组合路径：text_to_image → image_to_video）
    ///
    /// SVD 模型不支持直接的 text_to_video，因此采用两步组合：
    /// 1. text_to_image 生成首帧图像
    /// 2. image_to_video 将首帧动画化为视频
    pub async fn text_to_video(&self, params: T2VParams) -> Result<Vec<u8>, Error> {
        let backend = self.sd_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("stable-diffusion.cpp backend not configured".to_string())
        })?;

        // Step 1: text_to_image 生成首帧
        let t2i_params = crate::backend::T2IParams {
            prompt: params.prompt.clone(),
            negative_prompt: params.negative_prompt.clone(),
            width: params.width,
            height: params.height,
            steps: params.steps,
            cfg: params.cfg,
            sampler: "dpm++2m_karras".to_string(),
            seed: params.seed,
            model_path: params.model_path.clone(),
        };

        info!("T2V combo: Step 1 - generating first frame via T2I");
        let first_frame = backend.text_to_image(t2i_params).await.map_err(|e| {
            Error::BackendError(format!("T2V combo T2I step failed: {}", e))
        })?;

        // Step 2: image_to_video 动画化首帧
        let i2v_params = crate::backend::I2VParams {
            prompt: params.prompt.clone(),
            negative_prompt: params.negative_prompt.clone(),
            input_image: first_frame,
            width: params.width,
            height: params.height,
            frames: params.frames,
            fps: params.fps,
            motion_bucket_id: 127,
            motion_scale: 1024.0,
            steps: params.steps,
            cfg: 2.5, // SVD 推荐 cfg=2.5
            seed: params.seed,
            model_path: params.model_path.clone(),
        };

        info!("T2V combo: Step 2 - animating first frame via I2V (frames={}, fps={})", params.frames, params.fps);
        let video_data = backend.image_to_video(i2v_params).await.map_err(|e| {
            Error::BackendError(format!("T2V combo I2V step failed: {}", e))
        })?;

        Ok(video_data)
    }

    /// 图生视频（SVD）
    pub async fn image_to_video(&self, params: I2VParams) -> Result<Vec<u8>, Error> {
        let backend = self.sd_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("stable-diffusion.cpp backend not configured".to_string())
        })?;

        backend.image_to_video(params).await.map_err(|e| {
            Error::BackendError(format!("I2V failed: {}", e))
        })
    }

    /// 文本编码（使用llama.cpp）
    pub async fn encode_text(&self, text: &str) -> Result<Vec<f32>, Error> {
        let backend = self.llama_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("llama.cpp backend not configured".to_string())
        })?;

        backend.encode_text(text).await.map_err(|e| {
            Error::BackendError(format!("Text encoding failed: {}", e))
        })
    }

    /// 文本生成（使用llama.cpp）
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, Error> {
        let backend = self.llama_cpp.as_ref().ok_or_else(|| {
            Error::BackendError("llama.cpp backend not configured".to_string())
        })?;

        backend.generate_text(prompt, max_tokens, temperature).await.map_err(|e| {
            Error::BackendError(format!("Text generation failed: {}", e))
        })
    }

    /// 执行采样（节点系统调用接口）
    /// 将节点系统的参数转换为后端调用
    pub async fn sample(
        &self,
        model: &str,
        positive: Value,
        negative: Value,
        latent: Value,
        seed: i64,
        steps: i64,
        cfg: f64,
        sampler: &str,
        scheduler: &str,
        denoise: f64,
    ) -> Result<Value, Error> {
        // 从Conditioning提取提示词
        let prompt = extract_prompt_from_conditioning(&positive);
        let negative_prompt = extract_prompt_from_conditioning(&negative);

        // 从Latent获取尺寸信息（如果有）
        let (width, height) = extract_size_from_latent(&latent);

        // 合并 sampler + scheduler → sd-cli 接受的格式（如 "dpm++2m" + "karras" → "dpm++2m_karras"）
        let sampling_method = merge_sampler_scheduler(sampler, scheduler);

        // 检查是否有输入图像（图生图模式）
        let has_input_image = matches!(&latent, Value::Latent(l) if !l.is_empty() && l.len() > 4);

        if has_input_image {
            // 图生图模式
            let input_image_bytes = extract_image_from_latent(&latent);
            let params = I2IParams {
                prompt,
                negative_prompt,
                input_image: input_image_bytes,
                denoise: denoise as f32,
                steps: steps as usize,
                cfg: cfg as f32,
                sampler: sampling_method,
                width,
                height,
                seed: seed as usize,
                model_path: model.to_string(),
            };

            let image_data = self.image_to_image(params).await?;
            // 将图像数据转换回Latent（简化实现）
            Ok(Value::Image(image_data))
        } else {
            // 文生图模式
            let params = T2IParams {
                prompt,
                negative_prompt,
                width,
                height,
                steps: steps as usize,
                cfg: cfg as f32,
                sampler: sampling_method,
                seed: seed as usize,
                model_path: model.to_string(),
            };

            let image_data = self.text_to_image(params).await?;
            Ok(Value::Image(image_data))
        }
    }

    /// 启动所有后端
    pub async fn start_all(&self) -> Result<(), Error> {
        if let Some(sd) = &self.sd_cpp {
            if let Err(e) = sd.start().await {
                warn!("Failed to start stable-diffusion.cpp: {}", e);
            }
        }
        if let Some(llama) = &self.llama_cpp {
            if let Err(e) = llama.start().await {
                warn!("Failed to start llama.cpp: {}", e);
            }
        }
        Ok(())
    }

    /// 停止所有后端
    pub async fn stop_all(&self) -> Result<(), Error> {
        if let Some(sd) = &self.sd_cpp {
            let _ = sd.stop().await;
        }
        if let Some(llama) = &self.llama_cpp {
            let _ = llama.stop().await;
        }
        Ok(())
    }

    /// 释放显存
    pub async fn free_memory(&self) {
        if let Some(sd) = &self.sd_cpp {
            if let Err(e) = sd.free_memory().await {
                warn!("Failed to free sd_cpp memory: {}", e);
            }
        }
        if let Some(llama) = &self.llama_cpp {
            let _ = llama.stop().await;
        }
    }

    /// 健康检查
    pub async fn health_check(&self) -> bool {
        let mut healthy = true;

        if let Some(sd) = &self.sd_cpp {
            match sd.health_check().await {
                Ok(true) => info!("stable-diffusion.cpp: healthy"),
                Ok(false) => {
                    warn!("stable-diffusion.cpp: unhealthy");
                    healthy = false;
                }
                Err(e) => {
                    warn!("stable-diffusion.cpp: health check failed: {}", e);
                    healthy = false;
                }
            }
        }

        healthy
    }

    /// 获取系统状态
    pub async fn get_system_stats(&self) -> SystemStats {
        self.system_stats.read().await.clone()
    }

    /// 更新系统状态
    pub async fn update_system_stats(&self, stats: SystemStats) {
        let mut current = self.system_stats.write().await;
        *current = stats;
    }

    /// 获取stable-diffusion.cpp后端
    pub fn sd_cpp_backend(&self) -> Option<&Arc<StableDiffusionCppBackend>> {
        self.sd_cpp.as_ref()
    }

    /// 获取llama.cpp后端
    pub fn llama_cpp_backend(&self) -> Option<&Arc<LlamaCppBackend>> {
        self.llama_cpp.as_ref()
    }
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// 从Conditioning中提取提示词
fn extract_prompt_from_conditioning(value: &Value) -> String {
    match value {
        Value::Conditioning(text) => text.clone(),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

/// 合并 sampler + scheduler → sd-cli 接受的 --sampling-method 格式
///
/// ComfyUI 的 KSampler 将 sampler_name 和 scheduler 分开传递，而 sd-cli 接受合并格式：
/// - "euler" + "normal" → "euler"
/// - "dpm++2m" + "karras" → "dpm++2m_karras"
/// - "dpm++sde" + "karras" → "dpm++sde_karras"
/// - "euler" + "karras" → "euler_karras"
///
/// 规则：当 scheduler 不是 "normal"/"simple"/"" 时，追加 "_{scheduler}" 后缀
/// （若 sampler 已经以该后缀结尾则不再重复追加）
fn merge_sampler_scheduler(sampler: &str, scheduler: &str) -> String {
    let sampler = sampler.trim();
    let scheduler = scheduler.trim();

    // 不需要后缀的调度器
    let no_suffix = matches!(
        scheduler.to_lowercase().as_str(),
        "" | "normal" | "simple" | "ddim_uniform" | "beta"
    );

    if no_suffix {
        return sampler.to_string();
    }

    let suffix = format!("_{}", scheduler.to_lowercase());
    if sampler.to_lowercase().ends_with(&suffix) {
        sampler.to_string()
    } else {
        format!("{}{}", sampler, suffix)
    }
}

/// 从Latent中提取尺寸信息
/// 
/// Stable Diffusion latent 格式：[4, H/8, W/8]
/// 数据布局：channel-first (NCHW)
fn extract_size_from_latent(value: &Value) -> (usize, usize) {
    match value {
        Value::Latent(data) => {
            // 正确的尺寸计算：SD latent 是 [4, H/8, W/8]
            // 数据长度 = 4 * (H/8) * (W/8)
            if data.is_empty() {
                (512, 512)
            } else {
                let total = data.len();
                let channels = 4; // SD VAE 使用 4 通道
                let latent_pixels = total / channels;
                // latent_pixels = (H/8) * (W/8)
                // 假设正方形，计算 latent 尺寸
                let latent_dim = (latent_pixels as f64).sqrt() as usize;
                // 图像尺寸 = latent 尺寸 * 8
                let image_dim = latent_dim * 8;
                // 确保尺寸合理（SD 标准尺寸：512, 768, 1024）
                let reasonable_dim = match image_dim {
                    d if d <= 512 => 512,
                    d if d <= 768 => 768,
                    d if d <= 1024 => 1024,
                    d => d.clamp(512, 2048),
                };
                (reasonable_dim, reasonable_dim)
            }
        }
        _ => (512, 512),
    }
}

/// 从Latent中提取图像数据（使用 VAE 解码）
///
/// 将 latent [4, H/8, W/8] 解码为 RGB 图像 [3, H, W]
///
/// 特殊情况：如果 Latent 以 IMAGE_LATENT_MAGIC 开头，说明 VAEEncode 将原始
/// Image 字节编码到了 Latent 中，此时直接还原字节（无需 VAE 解码）
fn extract_image_from_latent(value: &Value) -> Vec<u8> {
    match value {
        Value::Latent(data) => {
            // 检测 Image-encoded latent（magic header）
            if data.len() >= 4 && (data[0] - crate::node::core_nodes::IMAGE_LATENT_MAGIC).abs() < 0.5 {
                let width = data[1] as usize;
                let height = data[2] as usize;
                let channels = data[3] as usize;
                let expected_bytes = width * height * channels;
                let available = data.len().saturating_sub(4);
                if available >= expected_bytes && channels == 3 && width > 0 && height > 0 {
                    return data[4..4 + expected_bytes]
                        .iter()
                        .map(|&f| f as u8)
                        .collect();
                }
                // magic header 存在但数据不完整，回退到 VAE 解码
            }

            let (width, height) = extract_size_from_latent(value);
            let vae_scale_factor = 0.18125; // SD VAE 标准缩放因子

            // 使用双线性插值 VAE 解码
            decode_latent_bilinear(data, width, height, vae_scale_factor)
        }
        Value::Image(data) => data.clone(),
        _ => Vec::new(),
    }
}

/// 双线性插值 VAE 解码
/// 
/// 将 latent 数据解码为 RGB 图像，使用双线性插值上采样
fn decode_latent_bilinear(latent: &[f32], width: usize, height: usize, scale_factor: f32) -> Vec<u8> {
    let channels = 4;
    let latent_h = height / 8;
    let latent_w = width / 8;
    let expected = channels * latent_h * latent_w;

    if latent.len() < expected {
        return vec![128u8; width * height * 3]; // 返回灰色图像
    }

    // 第一步：反缩放
    let scaled_latent: Vec<f32> = latent.iter()
        .map(|v| v / scale_factor)
        .collect();

    // 第二步：双线性插值上采样
    let mut image = vec![0u8; width * height * 3];
    
    for y in 0..height {
        for x in 0..width {
            let ly = y as f32 / 8.0;
            let lx = x as f32 / 8.0;
            
            let ly0 = ly.floor() as usize;
            let ly1 = (ly0 + 1).min(latent_h - 1);
            let lx0 = lx.floor() as usize;
            let lx1 = (lx0 + 1).min(latent_w - 1);
            
            let ty = ly - ly0 as f32;
            let tx = lx - lx0 as f32;
            
            let get_latent = |ch: usize, y: usize, x: usize| -> f32 {
                let idx = (y * latent_w + x) * channels + ch;
                if idx < scaled_latent.len() {
                    scaled_latent[idx]
                } else {
                    0.0
                }
            };
            
            // 将 4 通道映射到 3 通道 RGB
            for ch in 0..3 {
                let v00 = get_latent(ch, ly0, lx0);
                let v10 = get_latent(ch, ly1, lx0);
                let v01 = get_latent(ch, ly0, lx1);
                let v11 = get_latent(ch, ly1, lx1);
                
                let value = (1.0 - ty) * (1.0 - tx) * v00
                          + ty * (1.0 - tx) * v10
                          + (1.0 - ty) * tx * v01
                          + ty * tx * v11;
                
                // latent 分布 [-4, 4] 映射到 [0, 1]
                let normalized = (value + 4.0) / 8.0;
                let pixel = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
                image[(y * width + x) * 3 + ch] = pixel;
            }
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_backend() {
        let router = BackendRouter::new();

        assert_eq!(router.select_backend("diffusion"), BackendType::StableDiffusionCpp);
        assert_eq!(router.select_backend("stable_diffusion"), BackendType::StableDiffusionCpp);
        assert_eq!(router.select_backend("sd"), BackendType::StableDiffusionCpp);
        assert_eq!(router.select_backend("llm"), BackendType::LlamaCpp);
        assert_eq!(router.select_backend("text_encoder"), BackendType::LlamaCpp);
        assert_eq!(router.select_backend("t5"), BackendType::LlamaCpp);
        assert_eq!(router.select_backend("clip"), BackendType::LocalProcessor);
        assert_eq!(router.select_backend("vae"), BackendType::LocalProcessor);
        assert_eq!(router.select_backend("unknown"), BackendType::StableDiffusionCpp);
    }

    #[test]
    fn test_extract_prompt() {
        assert_eq!(extract_prompt_from_conditioning(&Value::String("hello".to_string())), "hello");

        let empty_cond = Value::Conditioning(String::new());
        assert_eq!(extract_prompt_from_conditioning(&empty_cond), "");

        let cond = Value::Conditioning("a cute cat".to_string());
        assert_eq!(extract_prompt_from_conditioning(&cond), "a cute cat");
    }

    #[test]
    fn test_extract_size() {
        let empty_latent = Value::Latent(vec![]);
        assert_eq!(extract_size_from_latent(&empty_latent), (512, 512));

        // 4通道，64x64 latent -> 512x512
        let latent = Value::Latent(vec![0.0; 4 * 64 * 64]);
        let (w, h) = extract_size_from_latent(&latent);
        assert_eq!(w, h);
        assert!(w >= 512);
    }

    #[tokio::test]
    async fn test_router_creation() {
        let router = BackendRouter::new();
        assert!(router.sd_cpp.is_none());
        assert!(router.llama_cpp.is_none());

        let router2 = BackendRouter::from_env();
        assert!(router2.sd_cpp.is_some());
        assert!(router2.llama_cpp.is_some());
    }

    #[test]
    fn test_merge_sampler_scheduler() {
        // normal/simple 调度器不加后缀
        assert_eq!(merge_sampler_scheduler("euler", "normal"), "euler");
        assert_eq!(merge_sampler_scheduler("euler", "simple"), "euler");
        assert_eq!(merge_sampler_scheduler("euler", ""), "euler");

        // karras 调度器追加后缀
        assert_eq!(merge_sampler_scheduler("dpm++2m", "karras"), "dpm++2m_karras");
        assert_eq!(merge_sampler_scheduler("dpm++sde", "karras"), "dpm++sde_karras");
        assert_eq!(merge_sampler_scheduler("euler", "karras"), "euler_karras");

        // 大小写不敏感
        assert_eq!(merge_sampler_scheduler("dpm++2m", "Karras"), "dpm++2m_karras");
        assert_eq!(merge_sampler_scheduler("DPM++2M", "karras"), "DPM++2M_karras");

        // 已有后缀不重复追加
        assert_eq!(merge_sampler_scheduler("dpm++2m_karras", "karras"), "dpm++2m_karras");

        // 空格修剪
        assert_eq!(merge_sampler_scheduler(" dpm++2m ", " karras "), "dpm++2m_karras");
    }
}
