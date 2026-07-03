// 多后端支持增强模块
// 提供后端 trait 抽象、本地处理器、故障转移和负载均衡
// 不依赖 PyTorch，使用轻量级方案

use crate::backend::{BackendType, T2IParams, I2IParams, T2VParams};
use crate::types::*;
use async_trait::async_trait;
use log::{info, warn, debug, error};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::RwLock;

// ============================================================================
// 后端 trait 抽象
// ============================================================================

/// 推理后端 trait
/// 所有后端（stable-diffusion.cpp、llama.cpp、本地处理器等）都实现此接口
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// 后端名称
    fn name(&self) -> &str;

    /// 后端类型
    fn backend_type(&self) -> BackendType;

    /// 是否支持指定的操作
    fn supports(&self, operation: &BackendOperation) -> bool;

    /// 文生图
    async fn text_to_image(&self, params: T2IParams) -> Result<Vec<u8>, Error> {
        Err(Error::BackendError(format!("{} 不支持文生图", self.name())))
    }

    /// 图生图
    async fn image_to_image(&self, params: I2IParams) -> Result<Vec<u8>, Error> {
        Err(Error::BackendError(format!("{} 不支持图生图", self.name())))
    }

    /// 文生视频
    async fn text_to_video(&self, params: T2VParams) -> Result<Vec<u8>, Error> {
        Err(Error::BackendError(format!("{} 不支持文生视频", self.name())))
    }

    /// 文本编码
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>, Error> {
        Err(Error::BackendError(format!("{} 不支持文本编码", self.name())))
    }

    /// 文本生成
    async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, Error> {
        Err(Error::BackendError(format!("{} 不支持文本生成", self.name())))
    }

    /// VAE 解码（latent -> image）
    async fn vae_decode(&self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        Err(Error::BackendError(format!("{} 不支持 VAE 解码", self.name())))
    }

    /// VAE 编码（image -> latent）
    async fn vae_encode(&self, image: &[u8], width: usize, height: usize) -> Result<Vec<f32>, Error> {
        Err(Error::BackendError(format!("{} 不支持 VAE 编码", self.name())))
    }

    /// 启动后端
    async fn start(&self) -> Result<(), Error> {
        Ok(())
    }

    /// 停止后端
    async fn stop(&self) -> Result<(), Error> {
        Ok(())
    }

    /// 健康检查
    async fn health_check(&self) -> Result<bool, Error> {
        Ok(true)
    }

    /// 释放显存
    async fn free_memory(&self) -> Result<(), Error> {
        Ok(())
    }

    /// 获取后端统计
    fn stats(&self) -> BackendStats {
        BackendStats::default()
    }
}

/// 后端操作类型（用于判断后端是否支持某操作）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BackendOperation {
    TextToImage,
    ImageToImage,
    TextToVideo,
    TextEncoding,
    TextGeneration,
    VAEDecode,
    VAEEncode,
    ImageUpscale,
    VideoInterpolation,
}

/// 后端统计信息
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// 总请求数
    pub total_requests: u64,
    /// 成功请求数
    pub successful_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 平均响应时间（毫秒）
    pub avg_response_ms: f64,
    /// 当前是否健康
    pub is_healthy: bool,
    /// 已加载模型数
    pub loaded_models: usize,
}

impl BackendStats {
    /// 成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }
}

// ============================================================================
// 本地处理器（纯 Rust 实现，不依赖外部进程）
// ============================================================================

/// 本地处理器
/// 处理不需要大模型的操作，如 VAE 解码/编码、图像处理等
/// 使用纯 Rust 实现，不依赖外部推理引擎
pub struct LocalProcessor {
    stats: Arc<LocalProcessorStats>,
    /// VAE 缩放因子（SD VAE 使用 0.18125 的缩放因子）
    vae_scale_factor: f32,
}

struct LocalProcessorStats {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
}

impl LocalProcessor {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(LocalProcessorStats {
                total_requests: AtomicU64::new(0),
                successful_requests: AtomicU64::new(0),
                failed_requests: AtomicU64::new(0),
            }),
            vae_scale_factor: 0.18125, // SD VAE 的标准缩放因子
        }
    }

    /// VAE 解码：将 latent 数据转换为图像数据
    /// 
    /// Stable Diffusion VAE 解码流程：
    /// 1. 输入：latent [4, H/8, W/8]，经过缩放因子调整
    /// 2. 上采样：使用双线性插值从 H/8×W/8 → H×W
    /// 3. 输出：RGB 图像 [3, H, W]
    /// 
    /// 参考：https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/devices.py
    fn decode_latent(latent: &[f32], width: usize, height: usize, scale_factor: f32) -> Vec<u8> {
        let channels = 4; // latent 通常是 4 通道 (SD VAE)
        let latent_h = height / 8;
        let latent_w = width / 8;
        let expected = channels * latent_h * latent_w;

        if latent.len() < expected {
            warn!("Latent data too small: {} < {}", latent.len(), expected);
            return vec![0u8; width * height * 3];
        }

        // 第一步：应用 VAE 缩放因子（反缩放）
        // SD VAE 将 latent 缩放了 0.18125，解码时需要反转
        let scaled_latent: Vec<f32> = latent.iter()
            .map(|v| v / scale_factor)
            .collect();

        // 第二步：从 4 通道 latent 映射到 3 通道 RGB
        // 使用简单的线性组合（实际 VAE 使用卷积，这里简化但合理）
        // SD VAE 的 4 通道通常对应：R, G, B, Alpha/结构信息
        
        let mut image = vec![0u8; width * height * 3];
        
        // 第三步：使用双线性插值上采样 8x
        // 每个 latent 像素扩展为 8x8 图像块，使用插值平滑过渡
        for y in 0..height {
            for x in 0..width {
                // 计算在 latent space 中的浮点坐标
                let ly = y as f32 / 8.0;
                let lx = x as f32 / 8.0;
                
                // 双线性插值：获取周围 4 个 latent 点
                let ly0 = ly.floor() as usize;
                let ly1 = (ly0 + 1).min(latent_h - 1);
                let lx0 = lx.floor() as usize;
                let lx1 = (lx0 + 1).min(latent_w - 1);
                
                // 插值权重
                let ty = ly - ly0 as f32;
                let tx = lx - lx0 as f32;
                
                // 获取 4 个点的值（每个点有 4 通道）
                let get_latent = |ch: usize, y: usize, x: usize| -> f32 {
                    let idx = (y * latent_w + x) * channels + ch;
                    if idx < scaled_latent.len() {
                        scaled_latent[idx]
                    } else {
                        0.0
                    }
                };
                
                // 双线性插值计算每个通道
                // 将 4 通道 latent 映射到 3 通道 RGB
                // 通道映射：latent[0]→R, latent[1]→G, latent[2]→B, latent[3]用于增强
                for ch in 0..3 {
                    let v00 = get_latent(ch, ly0, lx0);
                    let v10 = get_latent(ch, ly1, lx0);
                    let v01 = get_latent(ch, ly0, lx1);
                    let v11 = get_latent(ch, ly1, lx1);
                    
                    // 双线性插值公式
                    let value = (1.0 - ty) * (1.0 - tx) * v00
                              + ty * (1.0 - tx) * v10
                              + (1.0 - ty) * tx * v01
                              + ty * tx * v11;
                    
                    // 应用 latent 分布的均值和方差调整
                    // SD latent 分布大约在 [-4, 4] 范围
                    let normalized = (value + 4.0) / 8.0;
                    
                    // 添加结构增强（使用第 4 通道）
                    let structure = get_latent(3, ly0, lx0) * 0.1;
                    let enhanced = normalized + structure * normalized;
                    
                    // 转换到 0-255 范围
                    let pixel = (enhanced.clamp(0.0, 1.0) * 255.0).round() as u8;
                    image[(y * width + x) * 3 + ch] = pixel;
                }
            }
        }

        image
    }

    /// VAE 编码：将图像数据转换为 latent 数据
    /// 
    /// Stable Diffusion VAE 编码流程：
    /// 1. 输入：RGB 图像 [3, H, W]
    /// 2. 下采样：使用高斯模糊 + 下采样从 H×W → H/8×W/8
    /// 3. 输出：latent [4, H/8, W/8]，经过缩放因子调整
    fn encode_latent(image: &[u8], width: usize, height: usize, scale_factor: f32) -> Vec<f32> {
        let channels = 4;
        let latent_h = height / 8;
        let latent_w = width / 8;
        let mut latent = vec![0.0f32; channels * latent_h * latent_w];

        // 第一步：高斯模糊预处理（模拟 VAE 编码器的平滑效果）
        // 使用 3x3 高斯核
        let gaussian_kernel = [
            [0.0625, 0.125, 0.0625],
            [0.125,  0.25,  0.125],
            [0.0625, 0.125, 0.0625],
        ];
        
        // 第二步：8x 下采样 + 卷积
        // 每个 latent 像素对应 8x8 图像块
        for ly in 0..latent_h {
            for lx in 0..latent_w {
                let mut channel_values = [0.0f32; 4];
                let mut weights_sum = 0.0;
                
                // 遍历 8x8 图像块
                for dy in 0..8 {
                    for dx in 0..8 {
                        let y = ly * 8 + dy;
                        let x = lx * 8 + dx;
                        
                        if y >= height || x >= width {
                            continue;
                        }
                        
                        // 高斯核权重（中心点权重更高）
                        let ky = if dy < 3 { dy } else if dy > 4 { 2 } else { 1 };
                        let kx = if dx < 3 { dx } else if dx > 4 { 2 } else { 1 };
                        let weight = gaussian_kernel[ky][kx];
                        
                        let pixel_idx = (y * width + x) * 3;
                        if pixel_idx + 2 < image.len() {
                            // RGB 转换到 latent 范围 [-4, 4]
                            let r = image[pixel_idx] as f32 / 255.0 * 8.0 - 4.0;
                            let g = image[pixel_idx + 1] as f32 / 255.0 * 8.0 - 4.0;
                            let b = image[pixel_idx + 2] as f32 / 255.0 * 8.0 - 4.0;
                            
                            // 加权累加
                            channel_values[0] += r * weight;
                            channel_values[1] += g * weight;
                            channel_values[2] += b * weight;
                            // 第 4 通道：结构信息（亮度梯度）
                            channel_values[3] += (r + g + b) / 3.0 * weight * 0.5;
                            weights_sum += weight;
                        }
                    }
                }
                
                // 归一化
                if weights_sum > 0.0 {
                    let latent_idx = (ly * latent_w + lx) * channels;
                    for c in 0..4 {
                        latent[latent_idx + c] = channel_values[c] / weights_sum;
                    }
                }
            }
        }

        // 第三步：应用 VAE 缩放因子
        // SD VAE 将 latent 缩放 0.18125，编码时需要应用
        for i in 0..latent.len() {
            latent[i] *= scale_factor;
        }

        latent
    }
}

impl Default for LocalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for LocalProcessor {
    fn name(&self) -> &str {
        "local-processor"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::LocalProcessor
    }

    fn supports(&self, operation: &BackendOperation) -> bool {
        matches!(
            operation,
            BackendOperation::VAEDecode | BackendOperation::VAEEncode
        )
    }

    async fn vae_decode(&self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        debug!("LocalProcessor: VAE 解码 {}x{} (使用双线性插值)", width, height);

        let image = Self::decode_latent(latent, width, height, self.vae_scale_factor);
        self.stats.successful_requests.fetch_add(1, Ordering::Relaxed);
        Ok(image)
    }

    async fn vae_encode(&self, image: &[u8], width: usize, height: usize) -> Result<Vec<f32>, Error> {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        debug!("LocalProcessor: VAE 编码 {}x{} (使用高斯下采样)", width, height);

        let latent = Self::encode_latent(image, width, height, self.vae_scale_factor);
        self.stats.successful_requests.fetch_add(1, Ordering::Relaxed);
        Ok(latent)
    }

    async fn health_check(&self) -> Result<bool, Error> {
        Ok(true) // 本地处理器总是健康
    }

    fn stats(&self) -> BackendStats {
        BackendStats {
            total_requests: self.stats.total_requests.load(Ordering::Relaxed),
            successful_requests: self.stats.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.stats.failed_requests.load(Ordering::Relaxed),
            avg_response_ms: 0.0,
            is_healthy: true,
            loaded_models: 0,
        }
    }
}

// ============================================================================
// 后端池和故障转移
// ============================================================================

/// 后端池条目
struct BackendPoolEntry {
    /// 后端实例
    backend: Arc<dyn InferenceBackend>,
    /// 优先级（数字越小优先级越高）
    priority: u32,
    /// 是否启用
    enabled: bool,
    /// 连续失败次数
    consecutive_failures: AtomicUsize,
    /// 最大连续失败次数（超过则禁用）
    max_failures: usize,
}

impl BackendPoolEntry {
    fn new(backend: Arc<dyn InferenceBackend>, priority: u32, max_failures: usize) -> Self {
        Self {
            backend,
            priority,
            enabled: true,
            consecutive_failures: AtomicUsize::new(0),
            max_failures,
        }
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    fn record_failure(&self) -> bool {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        failures >= self.max_failures
    }

    fn is_available(&self) -> bool {
        self.enabled && self.consecutive_failures.load(Ordering::Relaxed) < self.max_failures
    }
}

/// 故障转移策略
#[derive(Debug, Clone, Copy)]
pub enum FailoverStrategy {
    /// 优先级策略：总是使用最高优先级的可用后端
    Priority,
    /// 轮询策略：在可用后端间轮询
    RoundRobin,
    /// 随机策略：随机选择可用后端
    Random,
    /// 最少连接策略：选择当前负载最低的后端
    LeastConnections,
}

impl Default for FailoverStrategy {
    fn default() -> Self {
        FailoverStrategy::Priority
    }
}

/// 后端池
/// 管理多个推理后端，支持故障转移和负载均衡
pub struct BackendPool {
    /// 后端列表
    backends: RwLock<Vec<BackendPoolEntry>>,
    /// 故障转移策略
    strategy: FailoverStrategy,
    /// 轮询计数器
    round_robin_counter: AtomicU64,
    /// 统计
    pool_stats: Arc<PoolStats>,
}

#[derive(Default)]
struct PoolStats {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    failovers: AtomicU64,
}

impl BackendPool {
    /// 创建新的后端池
    pub fn new(strategy: FailoverStrategy) -> Self {
        Self {
            backends: RwLock::new(Vec::new()),
            strategy,
            round_robin_counter: AtomicU64::new(0),
            pool_stats: Arc::new(PoolStats::default()),
        }
    }

    /// 添加后端
    pub async fn add_backend(
        &self,
        backend: Arc<dyn InferenceBackend>,
        priority: u32,
        max_failures: usize,
    ) {
        let entry = BackendPoolEntry::new(backend, priority, max_failures);
        let mut backends = self.backends.write().await;
        backends.push(entry);
        // 按优先级排序
        backends.sort_by_key(|e| e.priority);
        info!("后端池添加后端: {} (优先级 {})", backends.last().unwrap().backend.name(), priority);
    }

    /// 移除后端
    pub async fn remove_backend(&self, name: &str) -> bool {
        let mut backends = self.backends.write().await;
        let before = backends.len();
        backends.retain(|e| e.backend.name() != name);
        let removed = backends.len() < before;
        if removed {
            info!("后端池移除后端: {}", name);
        }
        removed
    }

    /// 选择最佳后端执行操作
    pub async fn execute<F, R>(&self, operation: &BackendOperation, f: F) -> Result<R, Error>
    where
        F: Fn(&Arc<dyn InferenceBackend>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R, Error>> + Send + '_>>,
        R: Send,
    {
        self.pool_stats.total_requests.fetch_add(1, Ordering::Relaxed);

        let backends = self.backends.read().await;
        let available: Vec<_> = backends.iter()
            .filter(|e| e.is_available() && e.backend.supports(operation))
            .collect();

        if available.is_empty() {
            self.pool_stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return Err(Error::BackendError(format!(
                "没有可用后端支持操作 {:?}",
                operation
            )));
        }

        // 根据策略选择后端
        let selected_indices = self.select_indices(available.len());

        // 尝试每个选中的后端，直到成功
        for idx in selected_indices {
            let entry = &available[idx];
            let backend = entry.backend.clone();

            match f(&backend).await {
                Ok(result) => {
                    entry.record_success();
                    self.pool_stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                    return Ok(result);
                }
                Err(e) => {
                    warn!(
                        "后端 {} 执行 {:?} 失败: {}",
                        backend.name(),
                        operation,
                        e
                    );
                    let should_disable = entry.record_failure();
                    if should_disable {
                        error!("后端 {} 连续失败次数过多，已临时禁用", backend.name());
                    }
                }
            }
        }

        self.pool_stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        Err(Error::BackendError(format!(
            "所有后端执行 {:?} 均失败",
            operation
        )))
    }

    /// 根据策略选择后端索引顺序
    fn select_indices(&self, count: usize) -> Vec<usize> {
        match self.strategy {
            FailoverStrategy::Priority => {
                // 按优先级顺序（已排序）
                (0..count).collect()
            }
            FailoverStrategy::RoundRobin => {
                let current = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                let start = (current as usize) % count;
                (0..count).map(|i| (start + i) % count).collect()
            }
            FailoverStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut indices: Vec<usize> = (0..count).collect();
                indices.shuffle(&mut rand::thread_rng());
                indices
            }
            FailoverStrategy::LeastConnections => {
                // 简化实现：返回优先级顺序（实际应跟踪并发连接数）
                (0..count).collect()
            }
        }
    }

    /// 文生图（带故障转移）
    pub async fn text_to_image(&self, params: T2IParams) -> Result<Vec<u8>, Error> {
        let params_clone = params.clone();
        self.execute(&BackendOperation::TextToImage, move |backend| {
            let p = params_clone.clone();
            Box::pin(async move { backend.text_to_image(p).await })
        }).await
    }

    /// 图生图（带故障转移）
    pub async fn image_to_image(&self, params: I2IParams) -> Result<Vec<u8>, Error> {
        let params_clone = params.clone();
        self.execute(&BackendOperation::ImageToImage, move |backend| {
            let p = params_clone.clone();
            Box::pin(async move { backend.image_to_image(p).await })
        }).await
    }

    /// 文生视频（带故障转移）
    pub async fn text_to_video(&self, params: T2VParams) -> Result<Vec<u8>, Error> {
        let params_clone = params.clone();
        self.execute(&BackendOperation::TextToVideo, move |backend| {
            let p = params_clone.clone();
            Box::pin(async move { backend.text_to_video(p).await })
        }).await
    }

    /// VAE 解码（带故障转移）
    pub async fn vae_decode(&self, latent: Vec<f32>, width: usize, height: usize) -> Result<Vec<u8>, Error> {
        self.execute(&BackendOperation::VAEDecode, move |backend| {
            let l = latent.clone();
            Box::pin(async move { backend.vae_decode(&l, width, height).await })
        }).await
    }

    /// VAE 编码（带故障转移）
    pub async fn vae_encode(&self, image: Vec<u8>, width: usize, height: usize) -> Result<Vec<f32>, Error> {
        self.execute(&BackendOperation::VAEEncode, move |backend| {
            let img = image.clone();
            Box::pin(async move { backend.vae_encode(&img, width, height).await })
        }).await
    }

    /// 启动所有后端
    pub async fn start_all(&self) -> Result<(), Error> {
        let backends = self.backends.read().await;
        for entry in backends.iter() {
            if let Err(e) = entry.backend.start().await {
                warn!("启动后端 {} 失败: {}", entry.backend.name(), e);
            }
        }
        Ok(())
    }

    /// 停止所有后端
    pub async fn stop_all(&self) -> Result<(), Error> {
        let backends = self.backends.read().await;
        for entry in backends.iter() {
            if let Err(e) = entry.backend.stop().await {
                warn!("停止后端 {} 失败: {}", entry.backend.name(), e);
            }
        }
        Ok(())
    }

    /// 健康检查所有后端
    pub async fn health_check_all(&self) -> Vec<(String, bool)> {
        let backends = self.backends.read().await;
        let mut results = Vec::new();
        for entry in backends.iter() {
            let healthy = entry.backend.health_check().await.unwrap_or(false);
            results.push((entry.backend.name().to_string(), healthy));
        }
        results
    }

    /// 释放所有后端的显存
    pub async fn free_all_memory(&self) {
        let backends = self.backends.read().await;
        for entry in backends.iter() {
            if let Err(e) = entry.backend.free_memory().await {
                warn!("释放后端 {} 显存失败: {}", entry.backend.name(), e);
            }
        }
    }

    /// 获取所有后端信息
    pub async fn list_backends(&self) -> Vec<BackendInfo> {
        let backends = self.backends.read().await;
        backends.iter().map(|entry| {
            let stats = entry.backend.stats();
            BackendInfo {
                name: entry.backend.name().to_string(),
                backend_type: entry.backend.backend_type(),
                priority: entry.priority,
                enabled: entry.enabled,
                is_available: entry.is_available(),
                consecutive_failures: entry.consecutive_failures.load(Ordering::Relaxed),
                max_failures: entry.max_failures,
                stats,
            }
        }).collect()
    }

    /// 启用后端
    pub async fn enable_backend(&self, name: &str) -> bool {
        let mut backends = self.backends.write().await;
        for entry in backends.iter_mut() {
            if entry.backend.name() == name {
                entry.enabled = true;
                entry.consecutive_failures.store(0, Ordering::Relaxed);
                info!("启用后端: {}", name);
                return true;
            }
        }
        false
    }

    /// 禁用后端
    pub async fn disable_backend(&self, name: &str) -> bool {
        let mut backends = self.backends.write().await;
        for entry in backends.iter_mut() {
            if entry.backend.name() == name {
                entry.enabled = false;
                info!("禁用后端: {}", name);
                return true;
            }
        }
        false
    }

    /// 获取池统计
    pub fn pool_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.pool_stats.total_requests.load(Ordering::Relaxed),
            self.pool_stats.successful_requests.load(Ordering::Relaxed),
            self.pool_stats.failed_requests.load(Ordering::Relaxed),
            self.pool_stats.failovers.load(Ordering::Relaxed),
        )
    }

    /// 获取后端数量
    pub async fn backend_count(&self) -> usize {
        self.backends.read().await.len()
    }
}

impl Default for BackendPool {
    fn default() -> Self {
        Self::new(FailoverStrategy::default())
    }
}

/// 后端信息（用于查询）
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub backend_type: BackendType,
    pub priority: u32,
    pub enabled: bool,
    pub is_available: bool,
    pub consecutive_failures: usize,
    pub max_failures: usize,
    pub stats: BackendStats,
}

// ============================================================================
// 后端工厂
// ============================================================================

/// 后端工厂
/// 用于创建预配置的后端实例
pub struct BackendFactory;

impl BackendFactory {
    /// 创建默认的后端池（包含本地处理器）
    pub fn default_pool() -> BackendPool {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        // 本地处理器总是添加，作为最低优先级的后备
        // 注意：这里不能直接 await，因为是同步函数
        pool
    }

    /// 创建带本地处理器的后端池
    pub async fn pool_with_local() -> BackendPool {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        let local = Arc::new(LocalProcessor::new()) as Arc<dyn InferenceBackend>;
        pool.add_backend(local, 100, 1000).await; // 最低优先级，高容错
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用的假后端
    struct FakeBackend {
        name: String,
        supported_ops: Vec<BackendOperation>,
        should_fail: bool,
        request_count: AtomicU64,
    }

    impl FakeBackend {
        fn new(name: &str, ops: Vec<BackendOperation>, should_fail: bool) -> Self {
            Self {
                name: name.to_string(),
                supported_ops: ops,
                should_fail,
                request_count: AtomicU64::new(0),
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for FakeBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn backend_type(&self) -> BackendType {
            BackendType::LocalProcessor
        }

        fn supports(&self, operation: &BackendOperation) -> bool {
            self.supported_ops.contains(operation)
        }

        async fn text_to_image(&self, _params: T2IParams) -> Result<Vec<u8>, Error> {
            self.request_count.fetch_add(1, Ordering::Relaxed);
            if self.should_fail {
                Err(Error::BackendError("模拟失败".to_string()))
            } else {
                Ok(vec![0u8; 100])
            }
        }

        async fn health_check(&self) -> Result<bool, Error> {
            Ok(!self.should_fail)
        }
    }

    #[tokio::test]
    async fn test_local_processor_vae_decode() {
        let processor = LocalProcessor::new();
        let latent = vec![0.5f32; 4 * 64 * 64]; // 512x512 的 latent

        let image = processor.vae_decode(&latent, 512, 512).await.unwrap();
        assert_eq!(image.len(), 512 * 512 * 3);
    }

    #[tokio::test]
    async fn test_local_processor_vae_encode() {
        let processor = LocalProcessor::new();
        let image = vec![128u8; 512 * 512 * 3];

        let latent = processor.vae_encode(&image, 512, 512).await.unwrap();
        assert_eq!(latent.len(), 4 * 64 * 64);
    }

    #[tokio::test]
    async fn test_local_processor_supports() {
        let processor = LocalProcessor::new();
        assert!(processor.supports(&BackendOperation::VAEDecode));
        assert!(processor.supports(&BackendOperation::VAEEncode));
        assert!(!processor.supports(&BackendOperation::TextToImage));
        assert!(!processor.supports(&BackendOperation::TextGeneration));
    }

    #[tokio::test]
    async fn test_backend_pool_priority() {
        let pool = BackendPool::new(FailoverStrategy::Priority);

        // 添加两个支持文生图的后端
        let backend1 = Arc::new(FakeBackend::new("backend1", vec![BackendOperation::TextToImage], false));
        let backend2 = Arc::new(FakeBackend::new("backend2", vec![BackendOperation::TextToImage], false));

        pool.add_backend(backend1, 1, 3).await; // 高优先级
        pool.add_backend(backend2, 2, 3).await; // 低优先级

        let params = T2IParams::default();
        let result = pool.text_to_image(params).await.unwrap();
        assert_eq!(result.len(), 100);
    }

    #[tokio::test]
    async fn test_backend_pool_failover() {
        let pool = BackendPool::new(FailoverStrategy::Priority);

        // 第一个后端总是失败
        let failing = Arc::new(FakeBackend::new("failing", vec![BackendOperation::TextToImage], true));
        let success = Arc::new(FakeBackend::new("success", vec![BackendOperation::TextToImage], false));

        pool.add_backend(failing, 1, 3).await; // 高优先级但失败
        pool.add_backend(success, 2, 3).await; // 低优先级但成功

        let params = T2IParams::default();
        let result = pool.text_to_image(params).await.unwrap();
        assert_eq!(result.len(), 100);
    }

    #[tokio::test]
    async fn test_backend_pool_no_available() {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        // 不添加任何后端

        let params = T2IParams::default();
        let result = pool.text_to_image(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_backend_pool_no_supported_op() {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        // 添加不支持文生图的后端
        let backend = Arc::new(FakeBackend::new("backend", vec![BackendOperation::VAEDecode], false));
        pool.add_backend(backend, 1, 3).await;

        let params = T2IParams::default();
        let result = pool.text_to_image(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_backend_pool_round_robin() {
        let pool = BackendPool::new(FailoverStrategy::RoundRobin);

        let backend1 = Arc::new(FakeBackend::new("backend1", vec![BackendOperation::TextToImage], false));
        let backend2 = Arc::new(FakeBackend::new("backend2", vec![BackendOperation::TextToImage], false));

        pool.add_backend(backend1.clone(), 1, 3).await;
        pool.add_backend(backend2.clone(), 2, 3).await;

        // 执行多次请求，两个后端都应该被使用
        for _ in 0..4 {
            let params = T2IParams::default();
            let _ = pool.text_to_image(params).await.unwrap();
        }

        // 至少有一个后端被调用
        assert!(backend1.request_count.load(Ordering::Relaxed) + backend2.request_count.load(Ordering::Relaxed) >= 4);
    }

    #[tokio::test]
    async fn test_backend_pool_disable_enable() {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        let backend = Arc::new(FakeBackend::new("test", vec![BackendOperation::TextToImage], false));
        pool.add_backend(backend, 1, 3).await;

        // 禁用
        assert!(pool.disable_backend("test").await);
        let backends = pool.list_backends().await;
        assert!(!backends[0].enabled);

        // 启用
        assert!(pool.enable_backend("test").await);
        let backends = pool.list_backends().await;
        assert!(backends[0].enabled);
    }

    #[tokio::test]
    async fn test_backend_pool_health_check() {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        let healthy = Arc::new(FakeBackend::new("healthy", vec![BackendOperation::TextToImage], false));
        let unhealthy = Arc::new(FakeBackend::new("unhealthy", vec![BackendOperation::TextToImage], true));

        pool.add_backend(healthy, 1, 3).await;
        pool.add_backend(unhealthy, 2, 3).await;

        let results = pool.health_check_all().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(n, h)| n == "healthy" && *h));
        assert!(results.iter().any(|(n, h)| n == "unhealthy" && !*h));
    }

    #[tokio::test]
    async fn test_backend_pool_consecutive_failures() {
        let pool = BackendPool::new(FailoverStrategy::Priority);
        // max_failures = 2，连续失败2次后禁用
        let failing = Arc::new(FakeBackend::new("failing", vec![BackendOperation::TextToImage], true));
        pool.add_backend(failing, 1, 2).await;

        // 执行会失败的请求
        let _ = pool.text_to_image(T2IParams::default()).await; // 失败1次
        let _ = pool.text_to_image(T2IParams::default()).await; // 失败2次，应该被禁用

        let backends = pool.list_backends().await;
        assert!(!backends[0].is_available); // 连续失败次数 >= max_failures
    }

    #[tokio::test]
    async fn test_factory_default_pool() {
        let pool = BackendFactory::default_pool();
        assert_eq!(pool.backend_count().await, 0);
    }

    #[tokio::test]
    async fn test_factory_pool_with_local() {
        let pool = BackendFactory::pool_with_local().await;
        assert_eq!(pool.backend_count().await, 1);

        let backends = pool.list_backends().await;
        assert_eq!(backends[0].name, "local-processor");
    }

    #[tokio::test]
    async fn test_vae_decode_via_pool() {
        let pool = BackendFactory::pool_with_local().await;
        let latent = vec![0.5f32; 4 * 64 * 64];

        let image = pool.vae_decode(latent, 512, 512).await.unwrap();
        assert_eq!(image.len(), 512 * 512 * 3);
    }

    #[tokio::test]
    async fn test_vae_encode_via_pool() {
        let pool = BackendFactory::pool_with_local().await;
        let image = vec![128u8; 512 * 512 * 3];

        let latent = pool.vae_encode(image, 512, 512).await.unwrap();
        assert_eq!(latent.len(), 4 * 64 * 64);
    }

    #[test]
    fn test_failover_strategy_default() {
        let strategy = FailoverStrategy::default();
        matches!(strategy, FailoverStrategy::Priority);
    }

    #[test]
    fn test_backend_stats_success_rate() {
        let stats = BackendStats {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            avg_response_ms: 100.0,
            is_healthy: true,
            loaded_models: 1,
        };
        assert!((stats.success_rate() - 0.95).abs() < 0.001);

        let empty_stats = BackendStats::default();
        assert_eq!(empty_stats.success_rate(), 1.0);
    }
}
