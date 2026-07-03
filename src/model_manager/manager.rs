// 模型管理器
// 整合扫描器、缓存和元数据管理，提供完整的模型管理 API

use crate::model_manager::model_info::{
    LoadState, ModelArchitecture, ModelInfo, ModelType, ModelManagerError,
};
use crate::model_manager::scanner::{ModelScanner, ScanResult};
use crate::model_manager::cache::{CacheLayer, CacheStats, ModelCache};
use dashmap::DashMap;
use log::{info, warn};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 模型管理统计
#[derive(Debug, Clone, Default)]
pub struct ManagerStats {
    /// 已索引的模型总数
    pub total_models: usize,
    /// 按类型分组的数量
    pub by_type: std::collections::HashMap<ModelType, usize>,
    /// 总大小（字节）
    pub total_size_bytes: u64,
    /// 缓存统计
    pub cache_stats: CacheStats,
    /// 上次扫描时间
    pub last_scan: Option<chrono::DateTime<chrono::Utc>>,
    /// 上次扫描耗时（毫秒）
    pub last_scan_ms: u64,
}

/// 模型管理器
/// 负责模型的发现、索引、缓存、加载等全生命周期管理
pub struct ModelManager {
    /// 模型根目录
    models_dir: PathBuf,
    /// 模型扫描器
    scanner: ModelScanner,
    /// 模型缓存
    cache: Arc<ModelCache>,
    /// 模型索引（model_id -> ModelInfo）
    index: DashMap<String, ModelInfo>,
    /// 按名称索引（display_name -> model_id），用于快速查找
    name_index: DashMap<String, String>,
    /// 按类型索引（model_type -> [model_id]）
    type_index: DashMap<ModelType, Vec<String>>,
    /// 配置
    config: ManagerConfig,
    /// 上次扫描时间
    last_scan: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    /// 上次扫描耗时
    last_scan_ms: Arc<RwLock<u64>>,
}

/// 管理器配置
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// 是否启用自动扫描（文件系统监听）
    pub auto_scan: bool,
    /// 扫描间隔（秒），0 表示不自动重新扫描
    pub scan_interval_secs: u64,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 是否计算文件哈希（用于去重）
    pub compute_hashes: bool,
    /// 是否读取 safetensors header 元数据
    pub read_metadata: bool,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            auto_scan: false,
            scan_interval_secs: 300, // 5 分钟
            enable_cache: true,
            compute_hashes: false,
            read_metadata: false,
        }
    }
}

impl ModelManager {
    /// 创建新的模型管理器
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        Self {
            scanner: ModelScanner::new(models_dir.clone()),
            cache: Arc::new(ModelCache::new()),
            models_dir,
            index: DashMap::new(),
            name_index: DashMap::new(),
            type_index: DashMap::new(),
            config: ManagerConfig::default(),
            last_scan: Arc::new(RwLock::new(None)),
            last_scan_ms: Arc::new(RwLock::new(0)),
        }
    }

    /// 使用配置创建
    pub fn with_config(models_dir: impl Into<PathBuf>, config: ManagerConfig) -> Self {
        let models_dir = models_dir.into();
        let cache = if config.enable_cache {
            Arc::new(ModelCache::new())
        } else {
            Arc::new(ModelCache::with_capacity(0, 0))
        };
        Self {
            scanner: ModelScanner::new(models_dir.clone()),
            cache,
            models_dir,
            index: DashMap::new(),
            name_index: DashMap::new(),
            type_index: DashMap::new(),
            config,
            last_scan: Arc::new(RwLock::new(None)),
            last_scan_ms: Arc::new(RwLock::new(0)),
        }
    }

    /// 从环境变量创建
    pub fn from_env() -> Self {
        let models_dir = std::env::var("MODELS_DIR")
            .unwrap_or_else(|_| "models".to_string());
        Self::new(models_dir)
    }

    /// 获取模型根目录
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// 获取缓存引用
    pub fn cache(&self) -> &Arc<ModelCache> {
        &self.cache
    }

    /// 扫描所有模型并重建索引
    pub fn scan(&self) -> Result<ScanResult, ModelManagerError> {
        let result = self.scanner.scan()?;

        // 清空旧索引
        self.index.clear();
        self.name_index.clear();
        self.type_index.clear();

        // 构建新索引
        for model in &result.models {
            self.index.insert(model.id.clone(), model.clone());
            self.name_index.insert(model.display_name.clone(), model.id.clone());

            let mut type_entry = self.type_index
                .entry(model.model_type.clone())
                .or_insert_with(Vec::new);
            type_entry.push(model.id.clone());
        }

        // 更新扫描时间
        let now = chrono::Utc::now();
        if let Ok(mut t) = self.last_scan.try_write() {
            *t = Some(now);
        }
        if let Ok(mut ms) = self.last_scan_ms.try_write() {
            *ms = result.elapsed_ms;
        }

        // 如果配置了计算哈希，则在后台计算
        if self.config.compute_hashes {
            self.compute_hashes_background();
        }

        Ok(result)
    }

    /// 异步扫描
    pub async fn scan_async(&self) -> Result<ScanResult, ModelManagerError> {
        let result = self.scanner.scan_async().await?;

        // 清空旧索引
        self.index.clear();
        self.name_index.clear();
        self.type_index.clear();

        for model in &result.models {
            self.index.insert(model.id.clone(), model.clone());
            self.name_index.insert(model.display_name.clone(), model.id.clone());

            let mut type_entry = self.type_index
                .entry(model.model_type.clone())
                .or_insert_with(Vec::new);
            type_entry.push(model.id.clone());
        }

        let now = chrono::Utc::now();
        *self.last_scan.write().await = Some(now);
        *self.last_scan_ms.write().await = result.elapsed_ms;

        if self.config.compute_hashes {
            self.compute_hashes_background();
        }

        Ok(result)
    }

    /// 获取所有模型
    pub fn list_all(&self) -> Vec<ModelInfo> {
        self.index.iter().map(|e| e.value().clone()).collect()
    }

    /// 按类型列出模型
    pub fn list_by_type(&self, model_type: ModelType) -> Vec<ModelInfo> {
        match self.type_index.get(&model_type) {
            Some(ids) => {
                ids.iter()
                    .filter_map(|id| self.index.get(id).map(|e| e.value().clone()))
                    .collect()
            }
            None => Vec::new(),
        }
    }

    /// 获取指定类型的模型名称列表（用于节点 choices）
    pub fn list_names_by_type(&self, model_type: ModelType) -> Vec<String> {
        self.list_by_type(model_type)
            .iter()
            .map(|m| m.display_name.clone())
            .collect()
    }

    /// 通过 ID 获取模型
    pub fn get_by_id(&self, model_id: &str) -> Option<ModelInfo> {
        self.index.get(model_id).map(|e| e.value().clone())
    }

    /// 通过显示名获取模型
    pub fn get_by_name(&self, display_name: &str) -> Option<ModelInfo> {
        self.name_index.get(display_name)
            .and_then(|id| self.get_by_id(&id))
    }

    /// 通过名称或路径查找模型文件路径
    /// 支持以下查找方式：
    /// 1. 完整显示名（如 "sdxl/sdxl_base"）
    /// 2. 模型名称（如 "sdxl_base"）
    /// 3. 文件名（如 "sdxl_base.safetensors"）
    /// 4. 相对路径（如 "checkpoints/sdxl_base.safetensors"）
    pub fn find_model_path(&self, query: &str) -> Option<PathBuf> {
        // 1. 通过显示名查找
        if let Some(model) = self.get_by_name(query) {
            return Some(model.path);
        }

        // 2. 通过名称查找（首个匹配）
        for entry in self.index.iter() {
            if entry.value().name == query {
                return Some(entry.value().path.clone());
            }
        }

        // 3. 通过文件名查找
        for entry in self.index.iter() {
            if let Some(filename) = entry.value().path.file_name().and_then(|n| n.to_str()) {
                if filename == query {
                    return Some(entry.value().path.clone());
                }
            }
        }

        // 4. 作为相对路径查找
        let rel_path = self.models_dir.join(query);
        if rel_path.exists() {
            return Some(rel_path);
        }

        // 5. 尝试补全扩展名
        for ext in crate::model_manager::model_info::ModelFormat::supported_extensions() {
            let with_ext = format!("{}.{}", query, ext);
            let rel_with_ext = self.models_dir.join(&with_ext);
            if rel_with_ext.exists() {
                return Some(rel_with_ext);
            }
        }

        None
    }

    /// 搜索模型
    pub fn search(&self, query: &str) -> Vec<ModelInfo> {
        self.list_all()
            .into_iter()
            .filter(|m| m.matches(query))
            .collect()
    }

    /// 按类型和关键词搜索
    pub fn search_in_type(&self, model_type: ModelType, query: &str) -> Vec<ModelInfo> {
        self.list_by_type(model_type)
            .into_iter()
            .filter(|m| m.matches(query))
            .collect()
    }

    /// 按架构筛选
    pub fn list_by_architecture(&self, arch: ModelArchitecture) -> Vec<ModelInfo> {
        self.list_all()
            .into_iter()
            .filter(|m| m.architecture.as_ref() == Some(&arch))
            .collect()
    }

    /// 加载模型到 VRAM
    pub async fn load_model(&self, model_id: &str) -> Result<(), ModelManagerError> {
        let model = self.get_by_id(model_id)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model_id.to_string()))?;

        if !self.config.enable_cache {
            return Ok(());
        }

        match self.cache.load_to_vram(&model).await {
            Ok(_) => {
                // 更新加载状态
                if let Some(mut entry) = self.index.get_mut(model_id) {
                    entry.load_state = LoadState::LoadedVRAM;
                    entry.last_loaded = Some(chrono::Utc::now());
                    entry.load_count += 1;
                }
                Ok(())
            }
            Err(e) => {
                if let Some(mut entry) = self.index.get_mut(model_id) {
                    entry.load_state = LoadState::Failed(e.clone());
                }
                Err(ModelManagerError::LoadError(e))
            }
        }
    }

    /// 预加载模型到 RAM
    pub async fn preload_model(&self, model_id: &str) -> Result<(), ModelManagerError> {
        let model = self.get_by_id(model_id)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model_id.to_string()))?;

        if !self.config.enable_cache {
            return Ok(());
        }

        match self.cache.load_to_ram(&model).await {
            Ok(_) => {
                if let Some(mut entry) = self.index.get_mut(model_id) {
                    entry.load_state = LoadState::Loaded;
                    entry.last_loaded = Some(chrono::Utc::now());
                    entry.load_count += 1;
                }
                Ok(())
            }
            Err(e) => {
                if let Some(mut entry) = self.index.get_mut(model_id) {
                    entry.load_state = LoadState::Failed(e.clone());
                }
                Err(ModelManagerError::LoadError(e))
            }
        }
    }

    /// 卸载模型
    pub async fn unload_model(&self, model_id: &str) -> Result<(), ModelManagerError> {
        if self.config.enable_cache {
            self.cache.unload(model_id).await
                .map_err(ModelManagerError::LoadError)?;
        }

        if let Some(mut entry) = self.index.get_mut(model_id) {
            entry.load_state = LoadState::Unloaded;
        }
        Ok(())
    }

    /// 检查模型是否已加载
    pub async fn is_model_loaded(&self, model_id: &str) -> Option<CacheLayer> {
        if !self.config.enable_cache {
            return None;
        }
        self.cache.is_cached(model_id).await
    }

    /// 触发模型访问（更新 LRU）
    pub async fn touch_model(&self, model_id: &str) {
        if self.config.enable_cache {
            self.cache.touch(model_id).await;
        }
    }

    /// 释放显存（卸载所有 VRAM 中的模型）
    pub async fn free_vram(&self) {
        if self.config.enable_cache {
            self.cache.clear_vram().await;
            // 更新所有 LoadedVRAM 状态为 Unloaded
            for mut entry in self.index.iter_mut() {
                if entry.load_state == LoadState::LoadedVRAM {
                    entry.load_state = LoadState::Unloaded;
                }
            }
        }
    }

    /// 清空所有缓存
    pub async fn clear_cache(&self) {
        if self.config.enable_cache {
            self.cache.clear().await;
            for mut entry in self.index.iter_mut() {
                if entry.load_state == LoadState::Loaded || entry.load_state == LoadState::LoadedVRAM {
                    entry.load_state = LoadState::Unloaded;
                }
            }
        }
    }

    /// 获取管理器统计信息
    pub async fn stats(&self) -> ManagerStats {
        let mut by_type = std::collections::HashMap::new();
        let mut total_size = 0u64;

        for entry in self.index.iter() {
            *by_type.entry(entry.model_type.clone()).or_insert(0) += 1;
            total_size += entry.size_bytes;
        }

        let cache_stats = if self.config.enable_cache {
            self.cache.stats().await
        } else {
            CacheStats::default()
        };

        let last_scan = *self.last_scan.read().await;
        let last_scan_ms = *self.last_scan_ms.read().await;

        ManagerStats {
            total_models: self.index.len(),
            by_type,
            total_size_bytes: total_size,
            cache_stats,
            last_scan,
            last_scan_ms,
        }
    }

    /// 添加自定义标签到模型
    pub fn add_tag(&self, model_id: &str, tag: &str) -> Result<(), ModelManagerError> {
        let mut entry = self.index.get_mut(model_id)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model_id.to_string()))?;
        if !entry.tags.iter().any(|t| t == tag) {
            entry.tags.push(tag.to_string());
        }
        Ok(())
    }

    /// 移除模型标签
    pub fn remove_tag(&self, model_id: &str, tag: &str) -> Result<(), ModelManagerError> {
        let mut entry = self.index.get_mut(model_id)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model_id.to_string()))?;
        entry.tags.retain(|t| t != tag);
        Ok(())
    }

    /// 获取所有可用模型类型
    pub fn available_types(&self) -> Vec<ModelType> {
        self.type_index.iter()
            .filter(|e| !e.value().is_empty())
            .map(|e| e.key().clone())
            .collect()
    }

    /// 重新加载指定模型的元数据
    pub fn refresh_model(&self, model_id: &str) -> Result<(), ModelManagerError> {
        let model = self.get_by_id(model_id)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model_id.to_string()))?;

        let refreshed = ModelInfo::from_path(model.path, &self.models_dir)?;
        if let Some(mut entry) = self.index.get_mut(model_id) {
            *entry = refreshed;
        }
        Ok(())
    }

    /// 在后台计算所有模型的哈希（异步）
    fn compute_hashes_background(&self) {
        let index = self.index.clone();

        std::thread::spawn(move || {
            for mut entry in index.iter_mut() {
                if entry.hash.is_none() {
                    match ModelScanner::compute_partial_hash(&entry.path) {
                        Ok(hash) => {
                            entry.hash = Some(hash);
                        }
                        Err(e) => {
                            warn!("计算哈希失败 {}: {}", entry.display_name, e);
                        }
                    }
                }
            }
            info!("后台哈希计算完成");
        });
    }

    /// 获取配置引用
    pub fn config(&self) -> &ManagerConfig {
        &self.config
    }

    /// 启动自动扫描后台任务
    pub fn start_auto_scan(self: &Arc<Self>) {
        if !self.config.auto_scan || self.config.scan_interval_secs == 0 {
            return;
        }

        let manager = self.clone();
        let interval = self.config.scan_interval_secs;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(
                tokio::time::Duration::from_secs(interval)
            );
            ticker.tick().await; // 跳过第一次立即触发
            loop {
                ticker.tick().await;
                info!("自动重新扫描模型...");
                if let Err(e) = manager.scan_async().await {
                    warn!("自动扫描失败: {}", e);
                }
            }
        });

        info!("自动扫描已启动，间隔 {} 秒", interval);
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new("models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_manager() -> (PathBuf, ModelManager) {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_model_manager_{}", unique));
        let models_dir = temp_dir.join("models");

        // checkpoints
        let checkpoints = models_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::write(checkpoints.join("v1-5.safetensors"), b"fake sd15").unwrap();
        std::fs::write(checkpoints.join("sdxl_base.safetensors"), b"fake sdxl").unwrap();

        // lora
        let lora = models_dir.join("lora");
        std::fs::create_dir_all(&lora).unwrap();
        std::fs::write(lora.join("style.safetensors"), b"fake lora").unwrap();

        // vae
        let vae = models_dir.join("vae");
        std::fs::create_dir_all(&vae).unwrap();
        std::fs::write(vae.join("sdxl_vae.safetensors"), b"fake vae").unwrap();

        // controlnet
        let controlnet = models_dir.join("controlnet");
        std::fs::create_dir_all(&controlnet).unwrap();
        std::fs::write(controlnet.join("canny.safetensors"), b"fake controlnet").unwrap();

        let manager = ModelManager::new(&models_dir);
        (models_dir, manager)
    }

    #[test]
    fn test_scan_and_index() {
        let (_, manager) = create_test_manager();
        let result = manager.scan().unwrap();

        assert_eq!(result.models.len(), 5);
        assert_eq!(manager.list_all().len(), 5);
    }

    #[test]
    fn test_list_by_type() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let checkpoints = manager.list_by_type(ModelType::Checkpoint);
        assert_eq!(checkpoints.len(), 2);

        let loras = manager.list_by_type(ModelType::Lora);
        assert_eq!(loras.len(), 1);

        let vaes = manager.list_by_type(ModelType::VAE);
        assert_eq!(vaes.len(), 1);

        let controlnets = manager.list_by_type(ModelType::ControlNet);
        assert_eq!(controlnets.len(), 1);
    }

    #[test]
    fn test_list_names_by_type() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let checkpoint_names = manager.list_names_by_type(ModelType::Checkpoint);
        assert!(checkpoint_names.contains(&"v1-5".to_string()));
        assert!(checkpoint_names.contains(&"sdxl_base".to_string()));
    }

    #[test]
    fn test_get_by_id() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let models = manager.list_all();
        let first = &models[0];

        let found = manager.get_by_id(&first.id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, first.id);
    }

    #[test]
    fn test_get_by_name() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let found = manager.get_by_name("sdxl_base");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "sdxl_base");
    }

    #[test]
    fn test_find_model_path() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        // 通过显示名
        let path = manager.find_model_path("sdxl_base");
        assert!(path.is_some());
        assert!(path.unwrap().to_string_lossy().contains("sdxl_base"));

        // 通过文件名
        let path = manager.find_model_path("sdxl_base.safetensors");
        assert!(path.is_some());

        // 通过相对路径
        let path = manager.find_model_path("checkpoints/v1-5.safetensors");
        assert!(path.is_some());

        // 不存在的
        let path = manager.find_model_path("nonexistent");
        assert!(path.is_none());
    }

    #[test]
    fn test_search() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let results = manager.search("sdxl");
        assert!(results.len() >= 2); // sdxl_base + sdxl_vae

        let results = manager.search("lora");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "style");
    }

    #[test]
    fn test_search_in_type() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let results = manager.search_in_type(ModelType::Checkpoint, "sdxl");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "sdxl_base");
    }

    #[test]
    fn test_list_by_architecture() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        // sdxl_base 和 sdxl_vae 都包含 "sdxl" 关键词
        let sdxl_models = manager.list_by_architecture(ModelArchitecture::SDXL);
        assert!(sdxl_models.len() >= 1);
        assert!(sdxl_models.iter().any(|m| m.name == "sdxl_base"));

        let sd15_models = manager.list_by_architecture(ModelArchitecture::SD15);
        assert_eq!(sd15_models.len(), 1);
        assert_eq!(sd15_models[0].name, "v1-5");
    }

    #[test]
    fn test_available_types() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let types = manager.available_types();
        assert!(types.contains(&ModelType::Checkpoint));
        assert!(types.contains(&ModelType::Lora));
        assert!(types.contains(&ModelType::VAE));
        assert!(types.contains(&ModelType::ControlNet));
    }

    #[test]
    fn test_tags() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let model = manager.list_all().into_iter().next().unwrap();
        let model_id = model.id;

        manager.add_tag(&model_id, "favorite").unwrap();
        manager.add_tag(&model_id, "favorite").unwrap(); // 不重复添加
        manager.add_tag(&model_id, "tested").unwrap();

        let updated = manager.get_by_id(&model_id).unwrap();
        assert_eq!(updated.tags.len(), 2);
        assert!(updated.tags.contains(&"favorite".to_string()));
        assert!(updated.tags.contains(&"tested".to_string()));

        manager.remove_tag(&model_id, "favorite").unwrap();
        let updated = manager.get_by_id(&model_id).unwrap();
        assert_eq!(updated.tags.len(), 1);
        assert!(!updated.tags.contains(&"favorite".to_string()));
    }

    #[test]
    fn test_stats() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let stats = rt.block_on(async { manager.stats().await });

        assert_eq!(stats.total_models, 5);
        assert!(stats.total_size_bytes > 0);
        assert_eq!(stats.by_type.get(&ModelType::Checkpoint), Some(&2));
        assert!(stats.last_scan.is_some());
    }

    #[test]
    fn test_refresh_model() {
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let model = manager.list_all().into_iter().next().unwrap();
        let original_size = model.size_bytes;

        // 修改文件内容
        std::fs::write(&model.path, b"updated content that is longer than before for testing").unwrap();

        manager.refresh_model(&model.id).unwrap();
        let refreshed = manager.get_by_id(&model.id).unwrap();
        assert_ne!(refreshed.size_bytes, original_size);
    }

    #[test]
    fn test_load_unload_model() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let model = manager.list_by_type(ModelType::Checkpoint)
            .into_iter().next().unwrap();

        rt.block_on(async {
            // 加载到 VRAM
            manager.load_model(&model.id).await.unwrap();
            assert_eq!(manager.is_model_loaded(&model.id).await, Some(CacheLayer::VRAM));

            // 卸载
            manager.unload_model(&model.id).await.unwrap();
            assert!(manager.is_model_loaded(&model.id).await.is_none());
        });
    }

    #[test]
    fn test_preload_to_ram() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let model = manager.list_by_type(ModelType::Lora)
            .into_iter().next().unwrap();

        rt.block_on(async {
            manager.preload_model(&model.id).await.unwrap();
            assert_eq!(manager.is_model_loaded(&model.id).await, Some(CacheLayer::RAM));
        });
    }

    #[test]
    fn test_free_vram() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        let models = manager.list_by_type(ModelType::Checkpoint);

        rt.block_on(async {
            // 加载多个模型到 VRAM
            for m in &models {
                manager.load_model(&m.id).await.unwrap();
            }

            // 释放 VRAM
            manager.free_vram().await;

            for m in &models {
                assert!(manager.is_model_loaded(&m.id).await.is_none());
            }
        });
    }

    #[test]
    fn test_clear_cache() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = create_test_manager();
        manager.scan().unwrap();

        rt.block_on(async {
            let m1 = manager.list_by_type(ModelType::Checkpoint).into_iter().next().unwrap();
            let m2 = manager.list_by_type(ModelType::Lora).into_iter().next().unwrap();

            manager.load_model(&m1.id).await.unwrap();
            manager.preload_model(&m2.id).await.unwrap();

            manager.clear_cache().await;

            assert!(manager.is_model_loaded(&m1.id).await.is_none());
            assert!(manager.is_model_loaded(&m2.id).await.is_none());
        });
    }

    #[test]
    fn test_manager_with_config() {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_manager_config_{}", unique));
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(models_dir.join("checkpoints")).unwrap();
        std::fs::write(models_dir.join("checkpoints").join("test.safetensors"), b"fake").unwrap();

        let config = ManagerConfig {
            auto_scan: false,
            scan_interval_secs: 0,
            enable_cache: false,
            compute_hashes: false,
            read_metadata: false,
        };

        let manager = ModelManager::with_config(&models_dir, config);
        manager.scan().unwrap();

        assert_eq!(manager.list_all().len(), 1);
        assert!(!manager.config().enable_cache);
    }

    #[test]
    fn test_scan_async() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = create_test_manager();

        let result = rt.block_on(async {
            manager.scan_async().await.unwrap()
        });

        assert_eq!(result.models.len(), 5);
        assert_eq!(manager.list_all().len(), 5);
    }
}
