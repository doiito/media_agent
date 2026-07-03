// 模型缓存
// LRU 缓存管理已加载的模型，支持显存/内存双层缓存策略

use crate::model_manager::model_info::{LoadState, ModelInfo, ModelType};
use log::{info, debug, warn};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 缓存层级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLayer {
    /// 系统内存（RAM）
    RAM,
    /// 显存（VRAM）
    VRAM,
}

/// 缓存项
#[derive(Debug, Clone)]
struct CacheEntry {
    /// 模型 ID
    model_id: String,
    /// 缓存层级
    layer: CacheLayer,
    /// 占用大小（字节，估算）
    size_bytes: u64,
    /// 最后访问时间（Unix 时间戳，秒）
    last_access: i64,
    /// 访问次数
    access_count: u64,
}

/// 缓存统计
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 当前 RAM 缓存的模型数
    pub ram_count: usize,
    /// 当前 VRAM 缓存的模型数
    pub vram_count: usize,
    /// RAM 缓存总大小（字节）
    pub ram_size_bytes: u64,
    /// VRAM 缓存总大小（字节）
    pub vram_size_bytes: u64,
    /// 总加载次数
    pub total_loads: u64,
    /// 总卸载次数
    pub total_unloads: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// LRU 驱逐次数
    pub evictions: u64,
}

impl CacheStats {
    /// 缓存命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// 获取摘要字符串
    pub fn summary(&self) -> String {
        format!(
            "RAM: {} 个 ({}), VRAM: {} 个 ({}), 命中率: {:.1}%, 加载 {}, 卸载 {}, 驱逐 {}",
            self.ram_count,
            crate::model_manager::model_info::format_size(self.ram_size_bytes),
            self.vram_count,
            crate::model_manager::model_info::format_size(self.vram_size_bytes),
            self.hit_rate() * 100.0,
            self.total_loads,
            self.total_unloads,
            self.evictions
        )
    }
}

/// 模型缓存
/// 双层 LRU 缓存：VRAM（显存）和 RAM（内存）
pub struct ModelCache {
    /// RAM 缓存
    ram_cache: Arc<RwLock<Vec<CacheEntry>>>,
    /// VRAM 缓存
    vram_cache: Arc<RwLock<Vec<CacheEntry>>>,
    /// RAM 最大容量（字节），默认 8GB
    ram_capacity: u64,
    /// VRAM 最大容量（字节），默认 6GB
    vram_capacity: u64,
    /// 统计信息
    stats: Arc<CacheStatsAtomic>,
}

/// 原子统计信息
struct CacheStatsAtomic {
    total_loads: AtomicU64,
    total_unloads: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
}

impl CacheStatsAtomic {
    fn new() -> Self {
        Self {
            total_loads: AtomicU64::new(0),
            total_unloads: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
}

impl ModelCache {
    /// 创建新的模型缓存
    pub fn new() -> Self {
        Self::with_capacity(
            8 * 1024 * 1024 * 1024,  // 8GB RAM
            6 * 1024 * 1024 * 1024,  // 6GB VRAM
        )
    }

    /// 指定容量创建
    pub fn with_capacity(ram_capacity: u64, vram_capacity: u64) -> Self {
        Self {
            ram_cache: Arc::new(RwLock::new(Vec::new())),
            vram_cache: Arc::new(RwLock::new(Vec::new())),
            ram_capacity,
            vram_capacity,
            stats: Arc::new(CacheStatsAtomic::new()),
        }
    }

    /// 从环境变量创建
    pub fn from_env() -> Self {
        let ram_cap = std::env::var("MODEL_CACHE_RAM_GB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(8) * 1024 * 1024 * 1024;
        let vram_cap = std::env::var("MODEL_CACHE_VRAM_GB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(6) * 1024 * 1024 * 1024;
        Self::with_capacity(ram_cap, vram_cap)
    }

    /// 将模型加载到 VRAM 缓存
    pub async fn load_to_vram(&self, model: &ModelInfo) -> Result<bool, String> {
        let model_id = model.id.clone();
        let size = model.size_bytes;

        // 检查是否已在 VRAM 缓存
        {
            let vram = self.vram_cache.read().await;
            if vram.iter().any(|e| e.model_id == model_id) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                debug!("VRAM 缓存命中: {}", model.display_name);
                return Ok(false); // 已缓存，无需加载
            }
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.stats.total_loads.fetch_add(1, Ordering::Relaxed);

        // 如果已在 RAM，先从 RAM 移除（提升到 VRAM）
        {
            let mut ram = self.ram_cache.write().await;
            ram.retain(|e| e.model_id != model_id);
        }

        // 确保 VRAM 有足够空间
        self.ensure_vram_capacity(size).await?;

        // 添加到 VRAM 缓存
        let now = chrono::Utc::now().timestamp();
        let entry = CacheEntry {
            model_id: model_id.clone(),
            layer: CacheLayer::VRAM,
            size_bytes: size,
            last_access: now,
            access_count: 1,
        };

        {
            let mut vram = self.vram_cache.write().await;
            vram.push(entry);
        }

        info!("模型加载到 VRAM: {} ({})", model.display_name, model.size_human);
        Ok(true)
    }

    /// 将模型加载到 RAM 缓存
    pub async fn load_to_ram(&self, model: &ModelInfo) -> Result<bool, String> {
        let model_id = model.id.clone();
        let size = model.size_bytes;

        // 检查是否已在 RAM 或 VRAM 缓存
        {
            let vram = self.vram_cache.read().await;
            if vram.iter().any(|e| e.model_id == model_id) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(false); // 已在 VRAM，无需加载
            }
        }
        {
            let ram = self.ram_cache.read().await;
            if ram.iter().any(|e| e.model_id == model_id) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(false); // 已在 RAM
            }
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.stats.total_loads.fetch_add(1, Ordering::Relaxed);

        // 确保 RAM 有足够空间
        self.ensure_ram_capacity(size).await?;

        let now = chrono::Utc::now().timestamp();
        let entry = CacheEntry {
            model_id: model_id.clone(),
            layer: CacheLayer::RAM,
            size_bytes: size,
            last_access: now,
            access_count: 1,
        };

        {
            let mut ram = self.ram_cache.write().await;
            ram.push(entry);
        }

        info!("模型加载到 RAM: {} ({})", model.display_name, model.size_human);
        Ok(true)
    }

    /// 从缓存中卸载模型
    pub async fn unload(&self, model_id: &str) -> Result<bool, String> {
        let mut unloaded = false;

        {
            let mut vram = self.vram_cache.write().await;
            let before = vram.len();
            vram.retain(|e| e.model_id != model_id);
            if vram.len() < before {
                unloaded = true;
            }
        }

        {
            let mut ram = self.ram_cache.write().await;
            let before = ram.len();
            ram.retain(|e| e.model_id != model_id);
            if ram.len() < before {
                unloaded = true;
            }
        }

        if unloaded {
            self.stats.total_unloads.fetch_add(1, Ordering::Relaxed);
            debug!("模型卸载: {}", model_id);
        }

        Ok(unloaded)
    }

    /// 检查模型是否在缓存中
    pub async fn is_cached(&self, model_id: &str) -> Option<CacheLayer> {
        {
            let vram = self.vram_cache.read().await;
            if vram.iter().any(|e| e.model_id == model_id) {
                return Some(CacheLayer::VRAM);
            }
        }
        {
            let ram = self.ram_cache.read().await;
            if ram.iter().any(|e| e.model_id == model_id) {
                return Some(CacheLayer::RAM);
            }
        }
        None
    }

    /// 更新访问时间（LRU）
    pub async fn touch(&self, model_id: &str) {
        let now = chrono::Utc::now().timestamp();
        let mut updated = false;

        {
            let mut vram = self.vram_cache.write().await;
            for entry in vram.iter_mut() {
                if entry.model_id == model_id {
                    entry.last_access = now;
                    entry.access_count += 1;
                    updated = true;
                    break;
                }
            }
        }

        if !updated {
            let mut ram = self.ram_cache.write().await;
            for entry in ram.iter_mut() {
                if entry.model_id == model_id {
                    entry.last_access = now;
                    entry.access_count += 1;
                    break;
                }
            }
        }
    }

    /// 清空所有缓存
    pub async fn clear(&self) {
        let ram_count = self.ram_cache.read().await.len();
        let vram_count = self.vram_cache.read().await.len();

        self.ram_cache.write().await.clear();
        self.vram_cache.write().await.clear();

        self.stats.total_unloads.fetch_add((ram_count + vram_count) as u64, Ordering::Relaxed);
        info!("清空模型缓存: RAM {} 个, VRAM {} 个", ram_count, vram_count);
    }

    /// 清空 VRAM 缓存（释放显存）
    pub async fn clear_vram(&self) {
        let count = self.vram_cache.read().await.len();
        self.vram_cache.write().await.clear();
        self.stats.total_unloads.fetch_add(count as u64, Ordering::Relaxed);
        info!("清空 VRAM 缓存: {} 个模型", count);
    }

    /// 获取缓存统计
    pub async fn stats(&self) -> CacheStats {
        let ram = self.ram_cache.read().await;
        let vram = self.vram_cache.read().await;

        let ram_size: u64 = ram.iter().map(|e| e.size_bytes).sum();
        let vram_size: u64 = vram.iter().map(|e| e.size_bytes).sum();

        CacheStats {
            ram_count: ram.len(),
            vram_count: vram.len(),
            ram_size_bytes: ram_size,
            vram_size_bytes: vram_size,
            total_loads: self.stats.total_loads.load(Ordering::Relaxed),
            total_unloads: self.stats.total_unloads.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
        }
    }

    /// 获取缓存中的模型 ID 列表
    pub async fn cached_model_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        for entry in self.vram_cache.read().await.iter() {
            ids.push(entry.model_id.clone());
        }
        for entry in self.ram_cache.read().await.iter() {
            ids.push(entry.model_id.clone());
        }
        ids
    }

    /// 确保 VRAM 有足够容量
    async fn ensure_vram_capacity(&self, needed: u64) -> Result<(), String> {
        let mut vram = self.vram_cache.write().await;
        let current_size: u64 = vram.iter().map(|e| e.size_bytes).sum();

        if current_size + needed <= self.vram_capacity {
            return Ok(());
        }

        // LRU 驱逐：按最后访问时间排序，驱逐最久未访问的
        vram.sort_by_key(|e| e.last_access);

        while current_size + needed > self.vram_capacity && !vram.is_empty() {
            let evicted = vram.remove(0);
            warn!(
                "VRAM LRU 驱逐: {} ({}), 释放 {}",
                evicted.model_id,
                evicted.size_bytes,
                crate::model_manager::model_info::format_size(evicted.size_bytes)
            );
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            self.stats.total_unloads.fetch_add(1, Ordering::Relaxed);
        }

        // 如果驱逐后还是不够
        let new_size: u64 = vram.iter().map(|e| e.size_bytes).sum();
        if new_size + needed > self.vram_capacity {
            return Err(format!(
                "VRAM 容量不足: 需要 {}, 可用 {} (容量 {})",
                crate::model_manager::model_info::format_size(needed),
                crate::model_manager::model_info::format_size(self.vram_capacity.saturating_sub(new_size)),
                crate::model_manager::model_info::format_size(self.vram_capacity)
            ));
        }

        Ok(())
    }

    /// 确保 RAM 有足够容量
    async fn ensure_ram_capacity(&self, needed: u64) -> Result<(), String> {
        let mut ram = self.ram_cache.write().await;
        let current_size: u64 = ram.iter().map(|e| e.size_bytes).sum();

        if current_size + needed <= self.ram_capacity {
            return Ok(());
        }

        // LRU 驱逐
        ram.sort_by_key(|e| e.last_access);

        while current_size + needed > self.ram_capacity && !ram.is_empty() {
            let evicted = ram.remove(0);
            warn!(
                "RAM LRU 驱逐: {} ({}), 释放 {}",
                evicted.model_id,
                evicted.size_bytes,
                crate::model_manager::model_info::format_size(evicted.size_bytes)
            );
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            self.stats.total_unloads.fetch_add(1, Ordering::Relaxed);
        }

        let new_size: u64 = ram.iter().map(|e| e.size_bytes).sum();
        if new_size + needed > self.ram_capacity {
            return Err(format!(
                "RAM 容量不足: 需要 {}, 可用 {} (容量 {})",
                crate::model_manager::model_info::format_size(needed),
                crate::model_manager::model_info::format_size(self.ram_capacity.saturating_sub(new_size)),
                crate::model_manager::model_info::format_size(self.ram_capacity)
            ));
        }

        Ok(())
    }

    /// 设置 RAM 容量
    pub fn set_ram_capacity(&self, _capacity: u64) {
        // Note: 由于 ram_capacity 不是 mut，需要通过内部可变性
        // 这里简化实现，实际生产中应使用 RwLock<u64>
        warn!("set_ram_capacity 暂未实现（需要内部可变性改造）");
    }

    /// 设置 VRAM 容量
    pub fn set_vram_capacity(&self, _capacity: u64) {
        warn!("set_vram_capacity 暂未实现（需要内部可变性改造）");
    }

    /// 获取 RAM 容量
    pub fn ram_capacity(&self) -> u64 {
        self.ram_capacity
    }

    /// 获取 VRAM 容量
    pub fn vram_capacity(&self) -> u64 {
        self.vram_capacity
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_test_model(name: &str, size: u64, model_type: ModelType) -> ModelInfo {
        ModelInfo {
            id: format!("test_{}", name),
            name: name.to_string(),
            display_name: name.to_string(),
            path: PathBuf::from(format!("/tmp/{}.safetensors", name)),
            subdir: String::new(),
            model_type,
            format: crate::model_manager::model_info::ModelFormat::Safetensors,
            size_bytes: size,
            size_human: crate::model_manager::model_info::format_size(size),
            modified: chrono::Utc::now(),
            hash: None,
            architecture: None,
            load_state: LoadState::Unloaded,
            last_loaded: None,
            load_count: 0,
            tags: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_load_to_vram() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let model = make_test_model("model1", 100, ModelType::Checkpoint);

        // 首次加载
        let loaded = cache.load_to_vram(&model).await.unwrap();
        assert!(loaded);

        // 统计应该记录加载
        let stats = cache.stats().await;
        assert_eq!(stats.total_loads, 1);
        assert_eq!(stats.vram_count, 1);
        assert_eq!(stats.vram_size_bytes, 100);

        // 再次加载（应命中缓存）
        let loaded2 = cache.load_to_vram(&model).await.unwrap();
        assert!(!loaded2);

        let stats = cache.stats().await;
        assert_eq!(stats.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_load_to_ram() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let model = make_test_model("model1", 100, ModelType::Lora);

        let loaded = cache.load_to_ram(&model).await.unwrap();
        assert!(loaded);

        let stats = cache.stats().await;
        assert_eq!(stats.ram_count, 1);
    }

    #[tokio::test]
    async fn test_is_cached() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let model = make_test_model("model1", 100, ModelType::Checkpoint);

        assert!(cache.is_cached(&model.id).await.is_none());

        cache.load_to_vram(&model).await.unwrap();
        assert_eq!(cache.is_cached(&model.id).await, Some(CacheLayer::VRAM));
    }

    #[tokio::test]
    async fn test_unload() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let model = make_test_model("model1", 100, ModelType::Checkpoint);

        cache.load_to_vram(&model).await.unwrap();
        assert!(cache.is_cached(&model.id).await.is_some());

        let unloaded = cache.unload(&model.id).await.unwrap();
        assert!(unloaded);
        assert!(cache.is_cached(&model.id).await.is_none());
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        // 容量为 300 字节
        let cache = ModelCache::with_capacity(300, 300);

        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);
        let m2 = make_test_model("m2", 100, ModelType::Checkpoint);
        let m3 = make_test_model("m3", 100, ModelType::Checkpoint);

        cache.load_to_vram(&m1).await.unwrap();
        cache.load_to_vram(&m2).await.unwrap();
        cache.load_to_vram(&m3).await.unwrap();

        // m4 需要 100 字节，会驱逐 m1（LRU）
        let m4 = make_test_model("m4", 100, ModelType::Checkpoint);
        cache.load_to_vram(&m4).await.unwrap();

        let stats = cache.stats().await;
        assert!(stats.evictions >= 1);
        assert!(stats.vram_count <= 3);
    }

    #[tokio::test]
    async fn test_clear() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);
        let m2 = make_test_model("m2", 100, ModelType::Lora);

        cache.load_to_vram(&m1).await.unwrap();
        cache.load_to_ram(&m2).await.unwrap();

        assert_eq!(cache.cached_model_ids().await.len(), 2);

        cache.clear().await;
        assert_eq!(cache.cached_model_ids().await.len(), 0);
    }

    #[tokio::test]
    async fn test_clear_vram() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);
        let m2 = make_test_model("m2", 100, ModelType::Lora);

        cache.load_to_vram(&m1).await.unwrap();
        cache.load_to_ram(&m2).await.unwrap();

        cache.clear_vram().await;

        assert!(cache.is_cached(&m1.id).await.is_none());
        assert!(cache.is_cached(&m2.id).await.is_some()); // RAM 中的还在
    }

    #[tokio::test]
    async fn test_touch() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);

        cache.load_to_vram(&m1).await.unwrap();
        cache.touch(&m1.id).await;
        cache.touch(&m1.id).await;

        // touch 不改变缓存大小，只更新访问计数
        let stats = cache.stats().await;
        assert_eq!(stats.vram_count, 1);
    }

    #[tokio::test]
    async fn test_stats_hit_rate() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);

        // 首次加载：miss
        cache.load_to_vram(&m1).await.unwrap();
        // 再次加载：hit
        cache.load_to_vram(&m1).await.unwrap();
        // 再次加载：hit
        cache.load_to_vram(&m1).await.unwrap();

        let stats = cache.stats().await;
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        let rate = stats.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_capacity_too_small() {
        // 容量小于模型大小
        let cache = ModelCache::with_capacity(50, 50);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);

        let result = cache.load_to_vram(&m1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_promote_from_ram_to_vram() {
        let cache = ModelCache::with_capacity(1024, 1024);
        let m1 = make_test_model("m1", 100, ModelType::Checkpoint);

        // 先加载到 RAM
        cache.load_to_ram(&m1).await.unwrap();
        assert_eq!(cache.is_cached(&m1.id).await, Some(CacheLayer::RAM));

        // 加载到 VRAM，应从 RAM 移除
        cache.load_to_vram(&m1).await.unwrap();
        assert_eq!(cache.is_cached(&m1.id).await, Some(CacheLayer::VRAM));

        let stats = cache.stats().await;
        assert_eq!(stats.vram_count, 1);
        assert_eq!(stats.ram_count, 0);
    }

    #[test]
    fn test_cache_stats_summary() {
        let stats = CacheStats {
            ram_count: 2,
            vram_count: 3,
            ram_size_bytes: 1024 * 1024 * 1024,
            vram_size_bytes: 2 * 1024 * 1024 * 1024,
            total_loads: 10,
            total_unloads: 5,
            cache_hits: 7,
            cache_misses: 3,
            evictions: 1,
        };
        let summary = stats.summary();
        assert!(summary.contains("RAM: 2"));
        assert!(summary.contains("VRAM: 3"));
        assert!(summary.contains("命中率"));
    }
}
