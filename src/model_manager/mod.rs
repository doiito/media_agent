// 模型管理模块
// 提供模型发现、索引、缓存、加载等全生命周期管理

pub mod model_info;
pub mod scanner;
pub mod cache;
pub mod manager;

// 主要类型重导出
pub use model_info::{
    ModelType, ModelArchitecture, ModelFormat, ModelInfo, LoadState,
    ModelManagerError, format_size,
};
pub use scanner::{ModelScanner, ScanResult};
pub use cache::{ModelCache, CacheLayer, CacheStats};
pub use manager::{ModelManager, ManagerConfig, ManagerStats};

/// 初始化模型管理模块
/// 创建一个默认配置的 ModelManager，扫描指定目录
pub fn init(models_dir: &str) -> Result<ModelManager, ModelManagerError> {
    let manager = ModelManager::new(models_dir);
    manager.scan()?;
    Ok(manager)
}

/// 使用配置初始化模型管理模块
pub fn init_with_config(
    models_dir: &str,
    config: ManagerConfig,
) -> Result<ModelManager, ModelManagerError> {
    let manager = ModelManager::with_config(models_dir, config);
    manager.scan()?;
    Ok(manager)
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn setup_test_env() -> (std::path::PathBuf, ModelManager) {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_model_manager_integration_{}", unique));
        let models_dir = temp_dir.join("models");

        // checkpoints
        let checkpoints = models_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::write(checkpoints.join("v1-5-pruned.safetensors"), b"fake sd15 model data").unwrap();
        std::fs::write(checkpoints.join("sdxl_base_1.0.safetensors"), b"fake sdxl model data").unwrap();

        // lora
        let lora = models_dir.join("lora");
        std::fs::create_dir_all(&lora).unwrap();
        std::fs::write(lora.join("detail_tweaker.safetensors"), b"fake lora").unwrap();
        std::fs::write(lora.join("epic_realism.safetensors"), b"fake lora 2").unwrap();

        // vae
        let vae = models_dir.join("vae");
        std::fs::create_dir_all(&vae).unwrap();
        std::fs::write(vae.join("sdxl_vae.safetensors"), b"fake vae").unwrap();

        // controlnet
        let controlnet = models_dir.join("controlnet");
        std::fs::create_dir_all(&controlnet).unwrap();
        std::fs::write(controlnet.join("control_v11p_sd15_canny.safetensors"), b"fake controlnet").unwrap();
        std::fs::write(controlnet.join("control_v11p_sd15_depth.safetensors"), b"fake controlnet 2").unwrap();

        // clip_vision
        let clip_vision = models_dir.join("clip_vision");
        std::fs::create_dir_all(&clip_vision).unwrap();
        std::fs::write(clip_vision.join("clip_vit_l_14.safetensors"), b"fake clip vision").unwrap();

        let manager = ModelManager::new(&models_dir);
        (models_dir, manager)
    }

    #[test]
    fn test_init_function() {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_init_function_{}", unique));
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(models_dir.join("checkpoints")).unwrap();
        std::fs::write(models_dir.join("checkpoints").join("test.safetensors"), b"fake").unwrap();

        let manager = init(models_dir.to_str().unwrap()).unwrap();
        assert_eq!(manager.list_all().len(), 1);
    }

    #[test]
    fn test_init_with_config_function() {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_init_with_config_{}", unique));
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(models_dir.join("lora")).unwrap();
        std::fs::write(models_dir.join("lora").join("test.safetensors"), b"fake").unwrap();

        let config = ManagerConfig {
            auto_scan: false,
            scan_interval_secs: 0,
            enable_cache: true,
            compute_hashes: false,
            read_metadata: false,
        };

        let manager = init_with_config(models_dir.to_str().unwrap(), config).unwrap();
        assert_eq!(manager.list_all().len(), 1);
        assert_eq!(manager.list_by_type(ModelType::Lora).len(), 1);
    }

    #[test]
    fn test_full_workflow() {
        let (_, manager) = setup_test_env();
        manager.scan().unwrap();

        // 1. 验证扫描结果
        assert_eq!(manager.list_all().len(), 8);
        assert_eq!(manager.list_by_type(ModelType::Checkpoint).len(), 2);
        assert_eq!(manager.list_by_type(ModelType::Lora).len(), 2);
        assert_eq!(manager.list_by_type(ModelType::VAE).len(), 1);
        assert_eq!(manager.list_by_type(ModelType::ControlNet).len(), 2);
        assert_eq!(manager.list_by_type(ModelType::CLIPVision).len(), 1);

        // 2. 验证架构推断
        // sdxl_base_1.0 和 sdxl_vae 都包含 "sdxl" 关键词
        let sdxl_models = manager.list_by_architecture(ModelArchitecture::SDXL);
        assert!(sdxl_models.len() >= 1);
        assert!(sdxl_models.iter().any(|m| m.name == "sdxl_base_1.0"));

        let sd15_models = manager.list_by_architecture(ModelArchitecture::SD15);
        assert_eq!(sd15_models.len(), 1);

        let controlnet_models = manager.list_by_architecture(ModelArchitecture::ControlNet);
        assert_eq!(controlnet_models.len(), 2);

        let clip_models = manager.list_by_architecture(ModelArchitecture::CLIPVITL);
        assert_eq!(clip_models.len(), 1);

        // 3. 验证搜索
        let results = manager.search("control");
        assert_eq!(results.len(), 2);

        // 4. 验证按类型搜索
        let lora_results = manager.search_in_type(ModelType::Lora, "realism");
        assert_eq!(lora_results.len(), 1);
        assert_eq!(lora_results[0].name, "epic_realism");

        // 5. 验证模型查找
        let path = manager.find_model_path("sdxl_base_1.0");
        assert!(path.is_some());
        assert!(path.unwrap().to_string_lossy().ends_with("sdxl_base_1.0.safetensors"));
    }

    #[tokio::test]
    async fn test_cache_integration() {
        let (_, manager) = setup_test_env();
        manager.scan().unwrap();

        let model = manager.list_by_type(ModelType::Checkpoint)
            .into_iter().next().unwrap();

        // 加载到 VRAM
        manager.load_model(&model.id).await.unwrap();
        assert_eq!(manager.is_model_loaded(&model.id).await, Some(CacheLayer::VRAM));

        // 触摸访问
        manager.touch_model(&model.id).await;

        // 释放 VRAM
        manager.free_vram().await;
        assert!(manager.is_model_loaded(&model.id).await.is_none());
    }

    #[test]
    fn test_tags_workflow() {
        let (_, manager) = setup_test_env();
        manager.scan().unwrap();

        let model = manager.list_by_type(ModelType::Lora)
            .into_iter().next().unwrap();

        // 添加标签
        manager.add_tag(&model.id, "favorite").unwrap();
        manager.add_tag(&model.id, "anime").unwrap();

        // 搜索标签
        let results = manager.search("favorite");
        assert_eq!(results.len(), 1);

        let results = manager.search("anime");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_stats_comprehensive() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, manager) = setup_test_env();
        manager.scan().unwrap();

        let stats = rt.block_on(async { manager.stats().await });

        assert_eq!(stats.total_models, 8);
        assert!(stats.total_size_bytes > 0);
        assert_eq!(stats.by_type.get(&ModelType::Checkpoint), Some(&2));
        assert_eq!(stats.by_type.get(&ModelType::Lora), Some(&2));
        assert_eq!(stats.by_type.get(&ModelType::VAE), Some(&1));
        assert_eq!(stats.by_type.get(&ModelType::ControlNet), Some(&2));
        assert_eq!(stats.by_type.get(&ModelType::CLIPVision), Some(&1));
        assert!(stats.last_scan.is_some());
    }
}
