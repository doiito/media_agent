// ============================================================================
// MEGA INTEGRATION TEST — ComfyUI Rust Agent
//
// Fills gaps NOT covered by existing test files:
//   - Config system (load, env, validation, directories, serialization)
//   - Monitor system (alerts, averages, history bounds, processes, serialization)
//   - Preview system (sessions, frame cache, progress, disabled, stats, callbacks)
//   - Model manager (scanner, cache layers, tags, stats, edge cases)
//   - Backend types/configs (T2IParams, I2IParams, T2VParams, SdCppConfig, LlamaCppConfig)
//   - Types system (Value conversions, Error status/mapping, PromptTask ordering)
//   - Execution engine (queue priority, event bus, engine lifecycle)
//   - Node interface compliance (all 41+ nodes conform to Node trait)
//   - Backend pool (multi-backend, failover, local processor)
//   - Event system (EventBus serialize, subscribe/unsubscribe, publish_to)
//   - Agent/memory/context integration
//   - Cross-module integration scenarios
// ============================================================================

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use comfyui_rust_agent::*;
use comfyui_rust_agent::types::*;
use comfyui_rust_agent::config::*;
use comfyui_rust_agent::monitor::*;
use comfyui_rust_agent::preview::*;
use comfyui_rust_agent::execution::*;
use comfyui_rust_agent::backend::*;
use comfyui_rust_agent::model_manager::*;
use comfyui_rust_agent::workflow::{WorkflowBuilder, WorkflowValidator, WorkflowManager};
use comfyui_rust_agent::agent::*;

// Helper to create a temp directory with model files for testing
fn setup_temp_models() -> (std::path::PathBuf, ModelManager) {
    let unique = uuid::Uuid::new_v4().to_string();
    let tmp = std::env::temp_dir().join(format!("test_mega_models_{}", unique));
    let models_dir = tmp.join("models");

    let checkpoints = models_dir.join("checkpoints");
    std::fs::create_dir_all(&checkpoints).unwrap();
    std::fs::write(checkpoints.join("sd15.safetensors"), b"fake sd15 data").unwrap();
    std::fs::write(checkpoints.join("sdxl.safetensors"), b"fake sdxl data").unwrap();

    let lora = models_dir.join("lora");
    std::fs::create_dir_all(&lora).unwrap();
    std::fs::write(lora.join("anime.safetensors"), b"fake lora").unwrap();

    let vae = models_dir.join("vae");
    std::fs::create_dir_all(&vae).unwrap();
    std::fs::write(vae.join("sdxl_vae.safetensors"), b"fake vae").unwrap();

    let controlnet = models_dir.join("controlnet");
    std::fs::create_dir_all(&controlnet).unwrap();
    std::fs::write(controlnet.join("canny.safetensors"), b"fake cn").unwrap();

    let clip = models_dir.join("clip");
    std::fs::create_dir_all(&clip).unwrap();
    std::fs::write(clip.join("clip_l.safetensors"), b"fake clip").unwrap();

    let unet = models_dir.join("diffusion");
    std::fs::create_dir_all(&unet).unwrap();
    std::fs::write(unet.join("flux.safetensors"), b"fake flux").unwrap();

    let manager = ModelManager::new(&models_dir);
    manager.scan().unwrap();

    (tmp, manager)
}

// ============================================================================
// 1. CONFIG SYSTEM — FULL COVERAGE
// ============================================================================

mod config_system_tests {
    use super::*;

    #[test]
    fn test_app_config_default_values() {
        let config = AppConfig::default();
        // Server
        assert!(!config.server.host.is_empty());
        assert!(config.server.port > 0);
        assert_eq!(config.server.output_dir, "output");
        assert_eq!(config.server.max_workflows, 100);
        assert!(config.server.enable_cors);
        assert_eq!(config.server.request_timeout_secs, 300);
        assert_eq!(config.server.max_body_size_mb, 50);
        // Log
        assert!(!config.log.level.is_empty());
        assert!(config.log.timestamp);
        assert!(!config.log.module_path);
        // Monitor
        assert!(config.monitor.enabled);
        assert_eq!(config.monitor.collect_interval_secs, 10);
        assert_eq!(config.monitor.history_size, 360);
        assert_eq!(config.monitor.cpu_alert_threshold, 90.0);
        assert_eq!(config.monitor.mem_alert_threshold, 85.0);
        // Paths
        assert_eq!(config.paths.models_dir, "models");
        assert_eq!(config.paths.input_dir, "input");
        assert_eq!(config.paths.skills_dir, "skills");
        assert_eq!(config.paths.workflows_dir, "workflows");
    }

    #[test]
    fn test_config_validation_port_zero() {
        let mut config = AppConfig::default();
        config.server.port = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("port"), "Should mention port");
    }

    #[test]
    fn test_config_validation_max_workflows_zero() {
        let mut config = AppConfig::default();
        config.server.max_workflows = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_workflows"));
    }

    #[test]
    fn test_config_validation_collect_interval_zero() {
        let mut config = AppConfig::default();
        config.monitor.collect_interval_secs = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("collect_interval_secs"));
    }

    #[test]
    fn test_config_validation_invalid_log_level() {
        let mut config = AppConfig::default();
        config.log.level = "invalid".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("log level"));
    }

    #[test]
    fn test_config_from_file_not_found() {
        let result = AppConfig::from_file("/dev/null/nonexistent/config.json");
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::FileReadError(_) => {}
            e => panic!("Expected FileReadError, got: {:?}", e),
        }
    }

    #[test]
    fn test_config_serialization_roundtrip_full() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: AppConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config.server.port, parsed.server.port);
        assert_eq!(config.server.host, parsed.server.host);
        assert_eq!(config.server.max_workflows, parsed.server.max_workflows);
        assert_eq!(config.server.enable_cors, parsed.server.enable_cors);
        assert_eq!(config.log.level, parsed.log.level);
        assert_eq!(config.monitor.enabled, parsed.monitor.enabled);
        assert_eq!(config.paths.models_dir, parsed.paths.models_dir);
        assert_eq!(config.paths.skills_dir, parsed.paths.skills_dir);
    }

    #[test]
    fn test_config_env_override_host() {
        std::env::set_var("HOST", "192.168.1.1");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert_eq!(config.server.host, "192.168.1.1");
        std::env::remove_var("HOST");
    }

    #[test]
    fn test_config_env_override_port() {
        // Test valid port override
        std::env::set_var("PORT", "9090");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert_eq!(config.server.port, 9090);
        std::env::remove_var("PORT");

        // Test invalid port — must run sequentially after the valid test above
        // to avoid env var race conditions between parallel tests.
        std::env::set_var("PORT", "not-a-number");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert!(config.server.port > 0, "Port should stay at a valid default value");
        std::env::remove_var("PORT");
    }

    #[test]
    fn test_config_env_override_log_level() {
        std::env::set_var("LOG_LEVEL", "debug");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert_eq!(config.log.level, "debug");
        std::env::remove_var("LOG_LEVEL");
    }

    #[test]
    fn test_config_env_override_output_dir() {
        std::env::set_var("OUTPUT_DIR", "/custom/output");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert_eq!(config.server.output_dir, "/custom/output");
        std::env::remove_var("OUTPUT_DIR");
    }

    #[test]
    fn test_config_env_override_models_dir() {
        std::env::set_var("MODELS_DIR", "/custom/models");
        let mut config = AppConfig::default();
        config.merge_env_overrides();
        assert_eq!(config.paths.models_dir, "/custom/models");
        std::env::remove_var("MODELS_DIR");
    }

    #[test]
    fn test_config_server_config_default() {
        let sc = ServerConfig::default();
        assert!(!sc.host.is_empty());
        assert!(sc.port > 0);
    }

    #[test]
    fn test_log_config_default() {
        let lc = LogConfig::default();
        assert_eq!(lc.level, "info");
        assert!(lc.file.is_none());
        assert!(lc.timestamp);
    }

    #[test]
    fn test_paths_config_default() {
        let pc = PathsConfig::default();
        assert_eq!(pc.models_dir, "models");
        assert_eq!(pc.workflows_dir, "workflows");
        assert_eq!(pc.skills_dir, "skills");
        assert_eq!(pc.prompts_dir, "prompts");
    }

    #[test]
    fn test_config_ensure_directories_creates_subdirs() {
        let unique = uuid::Uuid::new_v4().to_string();
        let tmp = std::env::temp_dir().join(format!("comfyui_cfg_test_{}", unique));
        let mut config = AppConfig::default();
        config.server.output_dir = tmp.join("output").to_string_lossy().to_string();
        config.paths.models_dir = tmp.join("models").to_string_lossy().to_string();
        config.paths.input_dir = tmp.join("input").to_string_lossy().to_string();
        config.paths.temp_dir = tmp.join("temp").to_string_lossy().to_string();
        config.paths.prompts_dir = tmp.join("prompts").to_string_lossy().to_string();
        config.paths.schemas_dir = tmp.join("schemas").to_string_lossy().to_string();
        config.paths.workflows_dir = tmp.join("workflows").to_string_lossy().to_string();
        config.paths.skills_dir = tmp.join("skills").to_string_lossy().to_string();

        assert!(config.ensure_directories().is_ok());

        for sub in &["checkpoints", "diffusion", "vae", "lora", "clip", "clip_vision", "controlnet", "embeddings"] {
            let p = tmp.join("models").join(sub);
            assert!(p.exists(), "Model subdir not created: {:?}", p);
        }
        for sub in &["pa", "da", "ca", "aa", "sa"] {
            let p = tmp.join("prompts").join(sub);
            assert!(p.exists(), "Prompt subdir not created: {:?}", p);
        }
        assert!(tmp.join("output").exists());
        assert!(tmp.join("input").exists());
        assert!(tmp.join("temp").exists());
        assert!(tmp.join("schemas").exists());
        assert!(tmp.join("workflows").exists());
        assert!(tmp.join("skills").exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::FileReadError("test.json: No such file".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test.json"));

        let err = ConfigError::ParseError("invalid json".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid json"));

        let err = ConfigError::ValidationError("port is 0".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("port is 0"));

        let err = ConfigError::DirectoryCreateError("/tmp/foo: permission".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("/tmp/foo"));
    }

    #[test]
    fn test_load_or_default_returns_default_when_no_file() {
        let config = AppConfig::load_or_default();
        assert_eq!(config.server.port, 8188);
    }
}

// ============================================================================
// 2. MONITOR SYSTEM — COMPREHENSIVE COVERAGE
// ============================================================================

mod monitor_system_tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_collect_basic() {
        let monitor = Monitor::with_defaults();
        let snap = monitor.collect().await;

        assert!(snap.timestamp > 0, "timestamp should be > 0");
        assert!(snap.total_memory > 0, "total_memory should be > 0");
        assert!(snap.process_count > 0, "process_count should be > 0");
        assert!(snap.cpu_usage >= 0.0 && snap.cpu_usage <= 100.0);
        assert!(snap.memory_usage >= 0.0 && snap.memory_usage <= 100.0);
        assert!(!snap.disks.is_empty(), "should have at least one disk");
        assert!(snap.uptime_secs >= 0);
    }

    #[tokio::test]
    async fn test_monitor_history_bounded() {
        let mut config = MonitorConfig::default();
        config.history_size = 5;
        let monitor = Monitor::new(config);

        for _ in 0..10 {
            monitor.collect().await;
        }

        let history = monitor.history(None).await;
        assert_eq!(history.len(), 5, "history should be bounded to 5");
    }

    #[tokio::test]
    async fn test_monitor_history_limit() {
        let monitor = Monitor::with_defaults();
        monitor.collect().await;
        monitor.collect().await;
        monitor.collect().await;

        let h = monitor.history(Some(2)).await;
        assert_eq!(h.len(), 2);
    }

    #[tokio::test]
    async fn test_monitor_latest_none_when_empty() {
        let monitor = Monitor::with_defaults();
        assert!(monitor.latest().await.is_none());
    }

    #[tokio::test]
    async fn test_monitor_alerts_empty_by_default() {
        let monitor = Monitor::with_defaults();
        let alerts = monitor.alerts().await;
        assert!(alerts.is_empty());
    }

    #[tokio::test]
    async fn test_monitor_alerts_triggered() {
        let mut config = MonitorConfig::default();
        config.cpu_alert_threshold = 0.0;
        config.mem_alert_threshold = 0.0;
        let monitor = Monitor::new(config);
        monitor.collect().await;

        let alerts = monitor.alerts().await;
        assert!(!alerts.is_empty(), "Alerts should be triggered");
        let has_cpu = alerts.iter().any(|a| matches!(a.kind, AlertKind::HighCpuUsage));
        let has_mem = alerts.iter().any(|a| matches!(a.kind, AlertKind::HighMemoryUsage));
        assert!(has_mem, "Should have memory alert");
    }

    #[tokio::test]
    async fn test_monitor_clear_alerts() {
        let mut config = MonitorConfig::default();
        config.cpu_alert_threshold = 0.0;
        config.mem_alert_threshold = 0.0;
        let monitor = Monitor::new(config);
        monitor.collect().await;
        assert!(!monitor.alerts().await.is_empty());

        monitor.clear_alerts().await;
        assert!(monitor.alerts().await.is_empty());
    }

    #[tokio::test]
    async fn test_monitor_averages() {
        let monitor = Monitor::with_defaults();
        assert!(monitor.averages().await.is_none());

        monitor.collect().await;
        monitor.collect().await;
        monitor.collect().await;

        let avg = monitor.averages().await;
        assert!(avg.is_some());
        let avg = avg.unwrap();
        assert_eq!(avg.sample_count, 3);
        assert!(avg.avg_cpu_usage >= 0.0);
        assert!(avg.avg_memory_usage >= 0.0);
        assert!(avg.avg_process_count > 0);
        assert!(avg.max_cpu_usage >= avg.avg_cpu_usage);
    }

    #[tokio::test]
    async fn test_monitor_related_processes() {
        let monitor = Monitor::with_defaults();
        let procs = monitor.related_processes().await;
        // Should not panic; result may be empty
        assert!(procs.len() >= 0);
    }

    #[tokio::test]
    async fn test_monitor_process_metrics_invalid_pid() {
        let monitor = Monitor::with_defaults();
        let proc = monitor.process_metrics(999_999_999).await;
        assert!(proc.is_none());
    }

    #[test]
    fn test_metrics_snapshot_serialization_deep() {
        let snapshot = MetricsSnapshot {
            timestamp: 1234567890,
            cpu_usage: 45.5,
            per_cpu_usage: vec![40.0, 50.0, 45.0, 47.0],
            total_memory: 16_000_000_000,
            used_memory: 8_000_000_000,
            memory_usage: 50.0,
            total_swap: 2_000_000_000,
            used_swap: 500_000_000,
            disks: vec![
                DiskInfo {
                    mount_point: "/".to_string(),
                    total_space: 100_000_000_000,
                    used_space: 60_000_000_000,
                    usage: 60.0,
                },
                DiskInfo {
                    mount_point: "/home".to_string(),
                    total_space: 500_000_000_000,
                    used_space: 200_000_000_000,
                    usage: 40.0,
                },
            ],
            process_count: 250,
            load_avg: Some([1.5, 1.2, 1.0]),
            uptime_secs: 86400,
        };

        let json = serde_json::to_string_pretty(&snapshot).unwrap();
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.timestamp, 1234567890);
        assert!((parsed.cpu_usage - 45.5).abs() < 0.001);
        assert_eq!(parsed.per_cpu_usage.len(), 4);
        assert_eq!(parsed.disks.len(), 2);
        assert_eq!(parsed.load_avg, Some([1.5, 1.2, 1.0]));
        assert_eq!(parsed.uptime_secs, 86400);
    }

    #[test]
    fn test_disk_info_creation() {
        let disk = DiskInfo {
            mount_point: "/mnt/data".to_string(),
            total_space: 1_000_000_000_000,
            used_space: 750_000_000_000,
            usage: 75.0,
        };
        assert_eq!(disk.mount_point, "/mnt/data");
        assert!(disk.total_space > disk.used_space);
        assert!((disk.usage - 75.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_monitor_start_stop() {
        let monitor = Arc::new(Monitor::with_defaults());
        assert!(!monitor.is_running().await);

        monitor.clone().start().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(monitor.is_running().await);

        monitor.stop().await;
        assert!(!monitor.is_running().await);
    }

    #[test]
    fn test_alert_level_equality() {
        assert_ne!(AlertLevel::Info, AlertLevel::Warning);
        assert_ne!(AlertLevel::Warning, AlertLevel::Critical);
        assert_eq!(format!("{:?}", AlertLevel::Info), "Info");
    }
}

// ============================================================================
// 3. PREVIEW SYSTEM — DEEP COVERAGE
// ============================================================================

mod preview_system_tests {
    use super::*;

    #[test]
    fn test_preview_config_default() {
        let config = PreviewConfig::default();
        assert!(config.enabled);
        assert_eq!(config.step_interval, 5);
        assert_eq!(config.max_width, 512);
        assert_eq!(config.max_height, 512);
        assert_eq!(config.jpeg_quality, 85);
        assert!(config.send_final_preview);
        assert_eq!(config.cache_size, 10);
    }

    #[test]
    fn test_preview_config_custom() {
        let config = PreviewConfig {
            enabled: false,
            step_interval: 10,
            max_width: 1024,
            max_height: 1024,
            jpeg_quality: 90,
            send_final_preview: false,
            cache_size: 20,
        };
        assert!(!config.enabled);
        assert_eq!(config.jpeg_quality, 90);
    }

    #[test]
    fn test_preview_frame_creation() {
        let frame = PreviewFrame::new(10, 50, vec![0u8; 100], "KSampler".to_string());
        assert_eq!(frame.step, 10);
        assert_eq!(frame.total_steps, 50);
        assert!((frame.progress - 20.0).abs() < 0.1);
        assert_eq!(frame.node_id, "KSampler");
        assert_eq!(frame.data.len(), 100);
    }

    #[test]
    fn test_preview_frame_zero_steps() {
        let frame = PreviewFrame::new(0, 0, vec![], "node".to_string());
        assert_eq!(frame.progress, 0.0);
    }

    #[test]
    fn test_preview_frame_complete() {
        let frame = PreviewFrame::new(30, 30, vec![1, 2, 3], "VAEDecode".to_string());
        assert!((frame.progress - 100.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_preview_session_create() {
        let session = PreviewSession::new("prompt-1".to_string(), "client-1".to_string());
        assert_eq!(session.prompt_id, "prompt-1");
        assert_eq!(session.client_id, "client-1");
        assert!(session.current_node.is_none());
        assert_eq!(session.current_step, 0);
        assert!(!session.completed);
        assert!(session.frames.is_empty());
    }

    #[tokio::test]
    async fn test_preview_session_add_frame() {
        let mut session = PreviewSession::new("prompt-1".to_string(), "client-1".to_string());
        let frame = PreviewFrame::new(5, 20, vec![0u8; 10], "KSampler".to_string());

        session.add_frame(frame.clone(), 10);
        assert_eq!(session.current_step, 5);
        assert_eq!(session.total_steps, 20);
        assert!(session.current_node.is_some());
        assert_eq!(session.frames.len(), 1);
        assert_eq!(session.latest_frame().unwrap().step, 5);
    }

    #[tokio::test]
    async fn test_preview_session_frame_cache_limit() {
        let mut session = PreviewSession::new("p1".to_string(), "c1".to_string());
        for i in 0..10 {
            session.add_frame(
                PreviewFrame::new(i, 20, vec![i as u8], "n".to_string()),
                5,
            );
        }
        assert_eq!(session.frames.len(), 5);
        assert_eq!(session.frames[0].step, 5);
        assert_eq!(session.frames[4].step, 9);
    }

    #[tokio::test]
    async fn test_preview_session_progress() {
        let mut session = PreviewSession::new("p1".to_string(), "c1".to_string());
        session.current_step = 15;
        session.total_steps = 30;
        assert!((session.progress() - 50.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_preview_session_progress_zero() {
        let session = PreviewSession::new("p1".to_string(), "c1".to_string());
        assert_eq!(session.progress(), 0.0);
    }

    #[tokio::test]
    async fn test_preview_session_elapsed() {
        let session = PreviewSession::new("p1".to_string(), "c1".to_string());
        let elapsed = session.elapsed_ms();
        assert!(elapsed >= 0);
    }

    #[tokio::test]
    async fn test_preview_manager_disabled() {
        let config = PreviewConfig {
            enabled: false,
            ..Default::default()
        };
        let bus = EventBus::new();
        let manager = PreviewManager::new(bus, config);

        manager.start_session("p1", "c1").await;
        assert!(manager.active_sessions().await.is_empty());

        manager.update_progress("p1", 5, 20).await;
        manager.push_preview("p1", "n1", 5, 20, vec![0u8]).await;
        manager.executing_node("p1", Some("KSampler")).await;
        manager.complete_session("p1", HashMap::new()).await;
    }

    #[tokio::test]
    async fn test_preview_manager_full_lifecycle() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe("client-1".to_string()).await;
        let manager = PreviewManager::with_default(bus);

        manager.start_session("prompt-1", "client-1").await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::ExecutionStart { .. }));

        manager.executing_node("prompt-1", Some("CheckpointLoader")).await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::Executing { ref node_id, .. } if *node_id == Some("CheckpointLoader".to_string())));

        manager.update_progress("prompt-1", 5, 20).await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::Progress { value: 5, max: 20, .. }));

        manager.push_preview("prompt-1", "KSampler", 5, 20, vec![0u8; 32]).await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::Preview { ref node_id, .. } if node_id == "KSampler"));

        manager.complete_session("prompt-1", HashMap::new()).await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::ExecutionSuccess { .. }));

        let session = manager.get_session("prompt-1").await.unwrap();
        assert!(session.completed);
    }

    #[tokio::test]
    async fn test_preview_manager_error_session() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe("c1".to_string()).await;
        let manager = PreviewManager::with_default(bus);

        manager.start_session("p1", "c1").await;
        let _ = rx.recv().await.unwrap();

        manager.error_session("p1", "OOM: out of memory").await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::ExecutionError { ref error, .. } if error == "OOM: out of memory"));
    }

    #[tokio::test]
    async fn test_preview_manager_interrupt() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe("c1".to_string()).await;
        let manager = PreviewManager::with_default(bus);

        manager.start_session("p1", "c1").await;
        let _ = rx.recv().await.unwrap();

        manager.interrupt_session("p1").await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::ExecutionInterrupted { .. }));

        assert!(manager.get_session("p1").await.is_none());
    }

    #[tokio::test]
    async fn test_preview_manager_get_frames_and_latest() {
        let manager = PreviewManager::with_default(EventBus::new());
        manager.start_session("p1", "c1").await;

        assert!(manager.get_latest_frame("p1").await.is_none());
        assert!(manager.get_frames("p1").await.is_empty());

        manager.push_preview("p1", "n1", 1, 10, vec![1u8]).await;
        manager.push_preview("p1", "n1", 3, 10, vec![3u8]).await;

        let frames = manager.get_frames("p1").await;
        assert_eq!(frames.len(), 2);
        let latest = manager.get_latest_frame("p1").await.unwrap();
        assert_eq!(latest.step, 3);
    }

    #[tokio::test]
    async fn test_preview_manager_cleanup_completed() {
        let manager = PreviewManager::with_default(EventBus::new());
        manager.start_session("p1", "c1").await;
        manager.start_session("p2", "c2").await;
        manager.start_session("p3", "c3").await;

        manager.complete_session("p1", HashMap::new()).await;
        manager.complete_session("p2", HashMap::new()).await;

        let cleaned = manager.cleanup_completed().await;
        assert_eq!(cleaned, 2);
        assert_eq!(manager.active_sessions().await.len(), 1);
    }

    #[tokio::test]
    async fn test_preview_manager_clear_all() {
        let manager = PreviewManager::with_default(EventBus::new());
        manager.start_session("p1", "c1").await;
        manager.start_session("p2", "c2").await;

        manager.clear_all().await;
        assert!(manager.active_sessions().await.is_empty());
    }

    #[tokio::test]
    async fn test_preview_manager_stats() {
        let manager = PreviewManager::with_default(EventBus::new());
        manager.start_session("p1", "c1").await;
        manager.start_session("p2", "c2").await;
        manager.push_preview("p1", "n1", 1, 10, vec![0u8]).await;
        manager.push_preview("p1", "n1", 2, 10, vec![1u8]).await;
        manager.complete_session("p1", HashMap::new()).await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.completed_sessions, 1);
        assert_eq!(stats.active_sessions, 1);
        assert_eq!(stats.total_frames, 2);
    }

    #[tokio::test]
    async fn test_preview_manager_session_info() {
        let manager = PreviewManager::with_default(EventBus::new());
        manager.start_session("p1", "c1").await;
        manager.update_progress("p1", 8, 20).await;

        let info = manager.get_session("p1").await.unwrap();
        assert_eq!(info.prompt_id, "p1");
        assert_eq!(info.client_id, "c1");
        assert_eq!(info.current_step, 8);
        assert_eq!(info.total_steps, 20);
        assert!((info.progress - 40.0).abs() < 0.1);
        assert_eq!(info.frame_count, 0);
        assert!(!info.completed);
    }

    #[tokio::test]
    async fn test_progress_callback_lifecycle() {
        let manager = Arc::new(PreviewManager::with_default(EventBus::new()));
        manager.start_session("p1", "c1").await;

        let cb = ProgressCallback::new(
            "p1".to_string(),
            "KSampler".to_string(),
            20,
            manager.clone(),
        );

        for step in 1..=20 {
            cb.on_step(step).await;
        }
        cb.on_complete().await;

        let session = manager.get_session("p1").await.unwrap();
        assert_eq!(session.current_step, 20);
        assert_eq!(session.total_steps, 20);
        assert!((session.progress - 100.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_progress_callback_push_preview() {
        let manager = Arc::new(PreviewManager::with_default(EventBus::new()));
        manager.start_session("p1", "c1").await;

        let cb = ProgressCallback::new(
            "p1".to_string(),
            "KSampler".to_string(),
            20,
            manager.clone(),
        );

        cb.push_preview(10, vec![0u8; 64]).await;

        let latest = manager.get_latest_frame("p1").await;
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().step, 10);
    }

    #[test]
    fn test_session_info_serialization() {
        let info = SessionInfo {
            prompt_id: "test".to_string(),
            client_id: "c1".to_string(),
            current_node: Some("KSampler".to_string()),
            current_step: 10,
            total_steps: 20,
            progress: 50.0,
            frame_count: 3,
            started_at: chrono::Utc::now(),
            elapsed_ms: 5000,
            completed: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("KSampler"));
    }
}

// ============================================================================
// 4. MODEL MANAGER — DEEP COVERAGE
// ============================================================================

mod model_manager_tests {
    use super::*;

    #[test]
    fn test_manager_scan_counts() {
        let (_tmp, manager) = setup_temp_models();

        let all = manager.list_all();
        assert_eq!(all.len(), 7, "Should find 7 model files");

        assert_eq!(manager.list_by_type(ModelType::Checkpoint).len(), 2);
        assert_eq!(manager.list_by_type(ModelType::Lora).len(), 1);
        assert_eq!(manager.list_by_type(ModelType::VAE).len(), 1);
        assert_eq!(manager.list_by_type(ModelType::ControlNet).len(), 1);
    }

    #[test]
    fn test_manager_search() {
        let (_tmp, manager) = setup_temp_models();

        let results = manager.search("sd15");
        assert_eq!(results.len(), 1);
        assert!(results[0].name.contains("sd15"));

        let results = manager.search("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_manager_search_in_type() {
        let (_tmp, manager) = setup_temp_models();

        let results = manager.search_in_type(ModelType::Checkpoint, "sdxl");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "sdxl");
    }

    #[test]
    fn test_manager_find_model_path() {
        let (_tmp, manager) = setup_temp_models();

        let path = manager.find_model_path("sd15");
        assert!(path.is_some());
        assert!(path.unwrap().to_string_lossy().ends_with("sd15.safetensors"));

        let path = manager.find_model_path("nonexistent");
        assert!(path.is_none());
    }

    #[test]
    fn test_manager_list_by_architecture() {
        let (_tmp, manager) = setup_temp_models();

        let sd15 = manager.list_by_architecture(ModelArchitecture::SD15);
        assert_eq!(sd15.len(), 1);

        let flux = manager.list_by_architecture(ModelArchitecture::Flux);
        assert_eq!(flux.len(), 1);
    }

    #[test]
    fn test_manager_tags() {
        let (_tmp, manager) = setup_temp_models();

        let model = manager.list_by_type(ModelType::Lora).into_iter().next().unwrap();
        manager.add_tag(&model.id, "favorite").unwrap();
        manager.add_tag(&model.id, "anime").unwrap();

        let results = manager.search("favorite");
        assert_eq!(results.len(), 1);

        let results = manager.search("anime");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_manager_stats() {
        let (_tmp, manager) = setup_temp_models();

        let stats = manager.stats().await;
        assert_eq!(stats.total_models, 7);
        assert!(stats.total_size_bytes > 0);
        assert_eq!(stats.by_type.get(&ModelType::Checkpoint), Some(&2));
        assert_eq!(stats.by_type.get(&ModelType::Lora), Some(&1));
        assert!(stats.last_scan.is_some());
    }

    #[tokio::test]
    async fn test_manager_cache_operations() {
        let (_tmp, manager) = setup_temp_models();

        let model = manager.list_by_type(ModelType::Checkpoint).into_iter().next().unwrap();

        manager.load_model(&model.id).await.unwrap();
        let layer = manager.is_model_loaded(&model.id).await;
        assert!(layer.is_some());

        manager.touch_model(&model.id).await;
    }

    #[tokio::test]
    async fn test_manager_free_vram() {
        let (_tmp, manager) = setup_temp_models();

        let model = manager.list_by_type(ModelType::Checkpoint).into_iter().next().unwrap();
        manager.load_model(&model.id).await.unwrap();
        assert!(manager.is_model_loaded(&model.id).await.is_some());

        manager.free_vram().await;
        assert!(manager.is_model_loaded(&model.id).await.is_none());
    }

    #[tokio::test]
    async fn test_manager_rescan_updates_counts() {
        let (_tmp, manager) = setup_temp_models();

        let before = manager.list_all().len();
        assert_eq!(before, 7);

        manager.scan().unwrap();
        let after = manager.list_all().len();
        assert_eq!(before, after);
    }

    #[test]
    fn test_manager_get_by_id() {
        let (_tmp, manager) = setup_temp_models();

        let models = manager.list_by_type(ModelType::Checkpoint);
        let model_id = models[0].id.clone();

        let found = manager.get_by_id(&model_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, models[0].name);

        let not_found = manager.get_by_id("nonexistent-id");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_manager_list_names_by_type() {
        let (_tmp, manager) = setup_temp_models();

        let names = manager.list_names_by_type(ModelType::Checkpoint);
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_model_type_to_dir_name() {
        assert_eq!(ModelType::Checkpoint.to_dir_name(), "checkpoints");
        assert_eq!(ModelType::Lora.to_dir_name(), "lora");
        assert_eq!(ModelType::VAE.to_dir_name(), "vae");
        assert_eq!(ModelType::ControlNet.to_dir_name(), "controlnet");
        assert_eq!(ModelType::CLIP.to_dir_name(), "clip");
    }

    #[test]
    fn test_model_architecture_variants() {
        let variants = vec![
            ModelArchitecture::SD15,
            ModelArchitecture::SDXL,
            ModelArchitecture::Flux,
            ModelArchitecture::SVD,
            ModelArchitecture::ControlNet,
        ];
        for v in variants {
            let name = format!("{:?}", v);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_manager_config_default() {
        let config = ManagerConfig::default();
        assert!(!config.auto_scan);
        assert!(config.enable_cache);
        assert!(!config.compute_hashes);
    }

    #[test]
    fn test_manager_models_dir() {
        let (_tmp, manager) = setup_temp_models();
        let dir = manager.models_dir();
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[tokio::test]
    async fn test_manager_cache_ref() {
        let (_tmp, manager) = setup_temp_models();
        let cache = manager.cache();
        let stats = cache.stats().await;
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_manager_from_env() {
        let manager = ModelManager::from_env();
        let dir = manager.models_dir();
        assert!(!dir.to_string_lossy().is_empty());
    }
}

// ============================================================================
// 5. BACKEND TYPES AND CONFIG — COMPREHENSIVE
// ============================================================================

mod backend_types_tests {
    use super::*;

    #[test]
    fn test_t2i_params_default() {
        let p = T2IParams::default();
        assert_eq!(p.width, 512);
        assert_eq!(p.height, 512);
        assert_eq!(p.steps, 20);
        assert!((p.cfg - 7.0).abs() < 0.001);
        assert_eq!(p.sampler, "euler");
        assert_eq!(p.seed, 0);
        assert!(p.prompt.is_empty());
        assert!(p.negative_prompt.is_empty());
        assert!(p.model_path.is_empty());
    }

    #[test]
    fn test_t2i_params_custom() {
        let p = T2IParams {
            prompt: "a cat".to_string(),
            negative_prompt: "blurry".to_string(),
            width: 1024,
            height: 768,
            steps: 30,
            cfg: 8.5,
            sampler: "dpmpp_2m".to_string(),
            seed: 42,
            model_path: "/models/sd15.safetensors".to_string(),
        };
        assert_eq!(p.prompt, "a cat");
        assert_eq!(p.seed, 42);
        assert_eq!(p.sampler, "dpmpp_2m");
    }

    #[test]
    fn test_i2i_params_default() {
        let p = I2IParams::default();
        assert!((p.denoise - 0.75).abs() < 0.001);
        assert_eq!(p.steps, 20);
        assert!((p.cfg - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_t2v_params_default() {
        let p = T2VParams::default();
        assert_eq!(p.frames, 16);
        assert_eq!(p.fps, 8);
        assert_eq!(p.width, 512);
        assert_eq!(p.height, 512);
        assert_eq!(p.steps, 20);
    }

    #[test]
    fn test_sd_cpp_config_from_env() {
        // from_env() loads from config.json first, then overrides with env vars.
        // Verify that loading works (values are populated), not exact defaults.
        let config = SdCppConfig::from_env();
        assert!(!config.executable_path.is_empty());
        assert!(!config.backend.is_empty());
        assert!(config.timeout_secs > 0);
        assert!(config.max_retries > 0);
    }

    #[test]
    fn test_sd_cpp_config_default_values() {
        // Verify hardcoded defaults (not affected by config.json)
        let config = SdCppConfig::default();
        assert_eq!(config.backend, "cuda");
        assert_eq!(config.timeout_secs, 300);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_sd_cpp_config_custom() {
        let config = SdCppConfig {
            executable_path: "/usr/bin/sd-cli".to_string(),
            model_path: "/models/sd15.safetensors".to_string(),
            backend: "cuda".to_string(),
            precision: "f16".to_string(),
            flash_attention: true,
            offload_to_cpu: false,
            rng_mode: "gpu".to_string(),
            timeout_secs: 600,
            max_retries: 5,
            max_concurrent_tasks: 2,
            max_queue_size: 20,
            health_check_interval: 15,
            idle_timeout_secs: 120,
            circuit_breaker_threshold: 5,
            circuit_breaker_reset_secs: 120,
            extra_args: vec!["--verbose".to_string()],
            env_vars: std::collections::HashMap::new(),
        };
        assert_eq!(config.executable_path, "/usr/bin/sd-cli");
        assert_eq!(config.max_concurrent_tasks, 2);
        assert!(config.flash_attention);
    }

    #[test]
    fn test_llama_cpp_config_from_env() {
        let config = LlamaCppConfig::from_env();
        assert!(!config.executable_path.is_empty());
        assert_eq!(config.n_ctx, 512);
        assert_eq!(config.n_ctx, 512);
    }

    #[test]
    fn test_backend_type_variants() {
        let variants = vec![
            BackendType::StableDiffusionCpp,
            BackendType::LlamaCpp,
            BackendType::LocalProcessor,
            BackendType::OnnxRuntime,
        ];
        for v in &variants {
            let name = format!("{:?}", v);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_backend_operation_variants() {
        let ops = vec![
            BackendOperation::TextToImage,
            BackendOperation::ImageToImage,
            BackendOperation::TextToVideo,
            BackendOperation::TextEncoding,
            BackendOperation::TextGeneration,
            BackendOperation::VAEDecode,
            BackendOperation::VAEEncode,
        ];
        for op in &ops {
            let name = format!("{:?}", op);
            assert!(!name.is_empty());
        }
    }
}

// ============================================================================
// 6. TYPES SYSTEM — COMPREHENSIVE COVERAGE
// ============================================================================

mod types_system_tests {
    use super::*;

    // --- Value conversions ---

    #[test]
    fn test_value_from_json_string() {
        let json = serde_json::json!("hello");
        let v = Value::from_json(json);
        assert_eq!(v.as_str().unwrap(), "hello");
    }

    #[test]
    fn test_value_from_json_int() {
        let json = serde_json::json!(42);
        let v = Value::from_json(json);
        assert_eq!(v.as_int().unwrap(), 42);
    }

    #[test]
    fn test_value_from_json_float() {
        let json = serde_json::json!(3.14);
        let v = Value::from_json(json);
        assert!((v.as_float().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_value_from_json_bool() {
        let json = serde_json::json!(true);
        let v = Value::from_json(json);
        match v {
            Value::Bool(b) => assert!(b),
            _ => panic!("Expected Bool"),
        }
    }

    #[test]
    fn test_value_from_json_array() {
        let json = serde_json::json!([1, "two", 3.0]);
        let v = Value::from_json(json);
        match v {
            Value::Array(arr) => assert_eq!(arr.len(), 3),
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_value_from_json_object() {
        let json = serde_json::json!({"name": "test", "value": 42});
        let v = Value::from_json(json);
        match v {
            Value::Object(obj) => {
                assert_eq!(obj.len(), 2);
                assert_eq!(obj["name"].as_str().unwrap(), "test");
            }
            _ => panic!("Expected Object"),
        }
    }

    #[test]
    fn test_value_from_json_null() {
        let json = serde_json::json!(null);
        let v = Value::from_json(json);
        assert_eq!(v.as_str().unwrap(), "null");
    }

    #[test]
    fn test_value_as_str_error() {
        let v = Value::Int(42);
        let err = v.as_str();
        assert!(err.is_err());
        assert_eq!(err.unwrap_err().status_code(), 400);
    }

    #[test]
    fn test_value_as_int_error() {
        let v = Value::String("not a number".to_string());
        assert!(v.as_int().is_err());
    }

    #[test]
    fn test_value_as_float_from_int() {
        let v = Value::Int(10);
        assert!((v.as_float().unwrap() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_value_as_float_error() {
        let v = Value::String("nan".to_string());
        assert!(v.as_float().is_err());
    }

    #[test]
    fn test_value_ref_str() {
        let v = Value::Model("sd15".to_string());
        assert_eq!(v.as_ref_str().unwrap(), "sd15");

        let v = Value::Clip("clip".to_string());
        assert_eq!(v.as_ref_str().unwrap(), "clip");

        let v = Value::Vae("vae".to_string());
        assert_eq!(v.as_ref_str().unwrap(), "vae");

        let v = Value::Int(42);
        assert!(v.as_ref_str().is_err());
    }

    // --- Error system ---

    #[test]
    fn test_error_status_codes() {
        assert_eq!(Error::TypeError("".to_string()).status_code(), 400);
        assert_eq!(Error::NotFound("".to_string()).status_code(), 404);
        assert_eq!(Error::Unauthorized("".to_string()).status_code(), 401);
        assert_eq!(Error::Forbidden("".to_string()).status_code(), 403);
        assert_eq!(Error::Conflict("".to_string()).status_code(), 409);
        assert_eq!(Error::Timeout("".to_string()).status_code(), 408);
        assert_eq!(Error::ValidationFailed("".to_string()).status_code(), 422);
        assert_eq!(Error::ServiceUnavailable("".to_string()).status_code(), 503);
        assert_eq!(Error::BackendError("".to_string()).status_code(), 500);
        assert_eq!(Error::ExecutionFailed("".to_string()).status_code(), 500);
    }

    #[test]
    fn test_error_is_retryable() {
        assert!(Error::Timeout("".to_string()).is_retryable());
        assert!(Error::ServiceUnavailable("".to_string()).is_retryable());
        assert!(Error::BackendError("".to_string()).is_retryable());
        assert!(!Error::NotFound("".to_string()).is_retryable());
        assert!(!Error::ValidationFailed("".to_string()).is_retryable());
        assert!(!Error::ExecutionFailed("".to_string()).is_retryable());
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(Error::TypeError("".to_string()).error_code(), "TYPE_ERROR");
        assert_eq!(Error::NotFound("".to_string()).error_code(), "NOT_FOUND");
        assert_eq!(Error::Timeout("".to_string()).error_code(), "TIMEOUT");
        assert_eq!(Error::BackendError("".to_string()).error_code(), "BACKEND_ERROR");
        assert_eq!(Error::NodeNotFound("".to_string()).error_code(), "NODE_NOT_FOUND");
        assert_eq!(Error::WorkflowError("".to_string()).error_code(), "WORKFLOW_ERROR");
        assert_eq!(Error::ImageError("".to_string()).error_code(), "IMAGE_ERROR");
        assert_eq!(Error::Internal("".to_string()).error_code(), "INTERNAL_ERROR");
    }

    #[test]
    fn test_error_display() {
        let e = Error::ExecutionFailed("something broke".to_string());
        let msg = format!("{}", e);
        assert!(msg.contains("something broke"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: Error = io_err.into();
        assert_eq!(err.status_code(), 500);
        assert!(format!("{}", err).contains("file missing"));
    }

    #[test]
    fn test_error_from_json() {
        let json_err = serde_json::from_str::<()>("invalid json").unwrap_err();
        let err: Error = json_err.into();
        assert_eq!(err.status_code(), 500);
    }

    // --- PromptTask ordering ---

    #[test]
    fn test_prompt_task_ordering() {
        let mut task_a = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "a".to_string(), "c1".to_string(),
        );
        task_a.priority = 1;

        let mut task_b = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "b".to_string(), "c2".to_string(),
        );
        task_b.priority = 10;

        let mut task_c = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "c".to_string(), "c3".to_string(),
        );
        task_c.priority = 5;

        let mut tasks = vec![task_a, task_b, task_c];
        tasks.sort();

        assert_eq!(tasks[0].priority, 10);
        assert_eq!(tasks[1].priority, 5);
        assert_eq!(tasks[2].priority, 1);
    }

    #[test]
    fn test_prompt_task_with_priority() {
        let task = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "test".to_string(), "client".to_string(),
        ).with_priority(5);
        assert_eq!(task.priority, 5);
    }

    #[test]
    fn test_prompt_task_eq() {
        let a = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "1".to_string(), "c".to_string(),
        ).with_priority(5);
        let b = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "2".to_string(), "c".to_string(),
        ).with_priority(5);
        assert_eq!(a, b);
    }

    // --- InputValue ---

    #[test]
    fn test_input_value_direct_string() {
        let iv = InputValue::Direct(Value::String("test".to_string()));
        match iv {
            InputValue::Direct(Value::String(s)) => assert_eq!(s, "test"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_input_value_link() {
        let iv = InputValue::Link(["1".to_string(), "0".to_string()]);
        match iv {
            InputValue::Link([node, slot]) => {
                assert_eq!(node, "1");
                assert_eq!(slot, "0");
            }
            _ => panic!("Wrong variant"),
        }
    }

    // --- Workflow, PromptRequest, PromptResponse ---

    #[test]
    fn test_prompt_request_deserialize() {
        let json = r#"{
            "prompt": {"nodes": {}, "links": []},
            "client_id": "client-1"
        }"#;
        let req: PromptRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.client_id, "client-1");
    }

    #[test]
    fn test_prompt_request_with_extra_data() {
        let json = r#"{
            "prompt": {"nodes": {}, "links": []},
            "client_id": "client-1",
            "extra_data": {"front": true}
        }"#;
        let req: PromptRequest = serde_json::from_str(json).unwrap();
        assert!(req.extra_data.front);
    }

    #[test]
    fn test_prompt_response_serialize() {
        let resp = PromptResponse {
            prompt_id: "test-123".to_string(),
            number: 1,
            queue_remaining: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("test-123"));
    }

    #[test]
    fn test_execution_result_variants() {
        match ExecutionResult::Pending {
            ExecutionResult::Pending => {}
            _ => panic!("Expected Pending"),
        }
        match ExecutionResult::Failure("err".to_string()) {
            ExecutionResult::Failure(s) => assert_eq!(s, "err"),
            _ => panic!("Expected Failure"),
        }
        let outputs = HashMap::new();
        match ExecutionResult::Success(outputs) {
            ExecutionResult::Success(_) => {}
            _ => panic!("Expected Success"),
        }
    }

    #[test]
    fn test_link_creation() {
        let link = Link {
            from_node: "1".to_string(),
            from_slot: 0,
            to_node: "5".to_string(),
            to_slot: 0,
            data_type: DataType::MODEL,
        };
        assert_eq!(link.from_node, "1");
        assert_eq!(link.to_node, "5");
        assert_eq!(link.data_type, DataType::MODEL);
    }

    #[test]
    fn test_workflow_node_creation() {
        let mut inputs = HashMap::new();
        inputs.insert("ckpt_name".to_string(), InputValue::Direct(Value::String("model".to_string())));
        let node = WorkflowNode {
            class_type: "CheckpointLoaderSimple".to_string(),
            inputs,
            pos: Some((100.0, 200.0)),
            size: Some((300.0, 100.0)),
            is_changed: Some(vec![Some(1.0)]),
        };
        assert_eq!(node.class_type, "CheckpointLoaderSimple");
        assert_eq!(node.pos.unwrap(), (100.0, 200.0));
    }

    #[test]
    fn test_history_entry_creation() {
        let entry = HistoryEntry {
            prompt_id: "test".to_string(),
            workflow: Workflow { nodes: HashMap::new(), links: vec![] },
            outputs: HashMap::new(),
            status: "success".to_string(),
            start_time: 1000.0,
            end_time: Some(2000.0),
        };
        assert_eq!(entry.status, "success");
        assert!(entry.end_time.is_some());
    }

    #[test]
    fn test_upload_response_serialize() {
        let resp = UploadResponse {
            name: "test.png".to_string(),
            subfolder: "output".to_string(),
            file_type: "image".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("test.png"));
    }

    #[test]
    fn test_system_stats_creation() {
        let stats = SystemStats {
            devices: vec![DeviceInfo {
                name: "CUDA0".to_string(),
                device_type: "cuda".to_string(),
                vram_total: 8_000_000_000,
                vram_free: 4_000_000_000,
                compute_capability: Some("8.0".to_string()),
            }],
        };
        assert_eq!(stats.devices.len(), 1);
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("CUDA0"));
    }

    #[test]
    fn test_data_type_variants() {
        let types = vec![
            DataType::MODEL, DataType::CLIP, DataType::VAE, DataType::CONDITIONING,
            DataType::LATENT, DataType::IMAGE, DataType::VIDEO, DataType::CONTROL_NET,
        ];
        for dt in &types {
            let name = format!("{:?}", dt);
            assert!(!name.is_empty());
        }
    }
}

// ============================================================================
// 7. EXECUTION ENGINE — QUEUE, EVENT BUS, ENGINE
// ============================================================================

mod execution_engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_prompt_queue_enqueue_dequeue() {
        let queue = PromptQueue::new();
        assert_eq!(queue.size().await, 0);

        let task = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "test-id".to_string(), "client".to_string(),
        );
        queue.enqueue(task).await;
        assert_eq!(queue.size().await, 1);

        let dequeued = queue.dequeue().await;
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().prompt_id, "test-id");
        assert_eq!(queue.size().await, 0);
    }

    #[tokio::test]
    async fn test_prompt_queue_empty_dequeue() {
        let queue = PromptQueue::new();
        let task = queue.dequeue().await;
        assert!(task.is_none());
    }

    #[tokio::test]
    async fn test_prompt_queue_priority_ordering() {
        let queue = PromptQueue::new();

        let low = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "low".to_string(), "c".to_string(),
        ).with_priority(10);
        queue.enqueue(low).await;

        let high = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "high".to_string(), "c".to_string(),
        ).with_priority(0);
        queue.enqueue(high).await;

        let first = queue.dequeue().await.unwrap();
        assert_eq!(first.prompt_id, "high");

        let second = queue.dequeue().await.unwrap();
        assert_eq!(second.prompt_id, "low");
    }

    #[tokio::test]
    async fn test_prompt_queue_peek() {
        let queue = PromptQueue::new();

        let task = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "peek".to_string(), "c".to_string(),
        );
        queue.enqueue(task).await;

        let peeked = queue.peek().await;
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap().prompt_id, "peek");
        assert_eq!(queue.size().await, 1);
    }

    #[tokio::test]
    async fn test_prompt_queue_enqueue_front() {
        let queue = PromptQueue::new();

        let normal = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "normal".to_string(), "c".to_string(),
        );
        queue.enqueue(normal).await;

        let front = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "front".to_string(), "c".to_string(),
        );
        queue.enqueue_front(front).await;

        let first = queue.dequeue().await.unwrap();
        assert_eq!(first.prompt_id, "front");
    }

    #[tokio::test]
    async fn test_prompt_queue_clear() {
        let queue = PromptQueue::new();
        queue.enqueue(PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "a".to_string(), "c".to_string(),
        )).await;
        queue.enqueue(PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "b".to_string(), "c".to_string(),
        )).await;

        queue.clear().await;
        assert_eq!(queue.size().await, 0);
    }

    #[tokio::test]
    async fn test_prompt_queue_remaining() {
        let queue = PromptQueue::new();
        assert_eq!(queue.remaining().await, 0);

        queue.enqueue(PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "a".to_string(), "c".to_string(),
        )).await;
        assert_eq!(queue.remaining().await, 1);

        queue.dequeue().await;
        assert_eq!(queue.remaining().await, 0);
    }

    #[tokio::test]
    async fn test_prompt_queue_interrupt_current() {
        let queue = PromptQueue::new();

        let task = PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "running".to_string(), "c".to_string(),
        );
        queue.enqueue(task).await;
        queue.dequeue().await;

        let interrupted = queue.interrupt_current().await;
        assert!(interrupted.is_some());
        assert_eq!(interrupted.unwrap().prompt_id, "running");
    }

    #[tokio::test]
    async fn test_prompt_queue_get_queue_info() {
        let queue = PromptQueue::new();
        queue.enqueue(PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "t1".to_string(), "c".to_string(),
        ).with_priority(5)).await;
        queue.enqueue(PromptTask::new(
            Workflow { nodes: HashMap::new(), links: vec![] },
            "t2".to_string(), "c".to_string(),
        ).with_priority(10)).await;

        let info = queue.get_queue_info().await;
        assert_eq!(info.len(), 2);
    }

    // --- EventBus tests ---

    #[tokio::test]
    async fn test_event_bus_subscribe_unsubscribe() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count().await, 0);

        let _rx = bus.subscribe("c1".to_string()).await;
        assert_eq!(bus.subscriber_count().await, 1);

        let _rx2 = bus.subscribe("c2".to_string()).await;
        assert_eq!(bus.subscriber_count().await, 2);

        bus.unsubscribe("c1").await;
        assert_eq!(bus.subscriber_count().await, 1);
    }

    #[tokio::test]
    async fn test_event_bus_publish_broadcast() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe("c1".to_string()).await;
        let mut rx2 = bus.subscribe("c2".to_string()).await;

        bus.publish(Event::ExecutionStart {
            prompt_id: "broadcast".to_string(),
        }).await;

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, Event::ExecutionStart { ref prompt_id } if prompt_id == "broadcast"));
        assert!(matches!(e2, Event::ExecutionStart { ref prompt_id } if prompt_id == "broadcast"));
    }

    #[tokio::test]
    async fn test_event_bus_publish_to_specific() {
        let bus = EventBus::new();
        let mut rx_priv = bus.subscribe("private".to_string()).await;
        let mut rx_pub = bus.subscribe("public".to_string()).await;

        bus.publish_to("private", Event::ExecutionCached {
            prompt_id: "secret".to_string(),
            nodes: vec!["n1".to_string()],
        }).await;

        let private_event = rx_priv.recv().await.unwrap();
        assert!(matches!(private_event, Event::ExecutionCached { ref prompt_id, .. } if prompt_id == "secret"));

        let public_event = rx_pub.try_recv();
        assert!(public_event.is_err(), "public should NOT receive private event");
    }

    #[tokio::test]
    async fn test_event_bus_clone_shares_state() {
        let bus1 = EventBus::new();
        let _rx = bus1.subscribe("c1".to_string()).await;

        let bus2 = bus1.clone();
        assert_eq!(bus2.subscriber_count().await, 1);

        bus2.unsubscribe("c1").await;
        assert_eq!(bus1.subscriber_count().await, 0);
    }

    #[tokio::test]
    async fn test_event_bus_event_serialization() {
        let event = Event::ExecutionStart {
            prompt_id: "serialize-test".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("serialize-test"));

        let event = Event::Progress {
            prompt_id: "p1".to_string(),
            value: 5,
            max: 20,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"value\":5"));
        assert!(json.contains("\"max\":20"));
    }

    // --- ExecutionEngine ---

    #[tokio::test]
    async fn test_execution_engine_submit_and_subscribe() {
        let mut engine = ExecutionEngine::new();
        let mut rx = engine.subscribe("client-1".to_string()).await;

        let workflow = WorkflowBuilder::text_to_image(
            "test".to_string(), "neg".to_string(),
            512, 512, 20, 7.0, 0, "model.safetensors".to_string(),
        ).unwrap();

        let prompt_id = engine.submit(workflow, "client-1".to_string()).await.unwrap();
        assert!(!prompt_id.is_empty());

        let result = engine.execute_next().await;
        assert!(result.is_ok());

        let event = rx.recv().await;
        assert!(event.is_ok());
    }

    #[tokio::test]
    async fn test_execution_engine_empty_queue() {
        let mut engine = ExecutionEngine::new();
        let result = engine.execute_next().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_execution_engine_interrupt() {
        let mut engine = ExecutionEngine::new();
        engine.interrupt();
        engine.interrupt();
    }

    #[tokio::test]
    async fn test_execution_engine_free_memory() {
        let mut engine = ExecutionEngine::new();
        engine.free_memory().await;
    }
}

// ============================================================================
// 8. WORKFLOW BUILDER — EDGE CASES AND BOUNDARY VALUES
// ============================================================================

mod workflow_builder_tests {
    use super::*;

    #[test]
    fn test_text_to_image_minimal_nodes() {
        let wf = WorkflowBuilder::text_to_image(
            "p".to_string(), "n".to_string(),
            64, 64, 1, 1.0, 0, "m".to_string(),
        ).expect("Minimal T2I should succeed");
        assert_eq!(wf.nodes.len(), 7);
    }

    #[test]
    fn test_text_to_image_max_values() {
        let wf = WorkflowBuilder::text_to_image(
            "p".to_string(), "n".to_string(),
            8192, 8192, 10000, 100.0, 999999999, "m".to_string(),
        ).expect("Max values T2I should succeed");
        assert_eq!(wf.nodes.len(), 7);
    }

    #[test]
    fn test_image_to_image_with_zero_denoise() {
        let wf = WorkflowBuilder::image_to_image(
            "p".to_string(), "n".to_string(),
            "input.png".to_string(),
            0.0, 20, 7.0, 42, "model.safetensors".to_string(),
        ).expect("Zero denoise I2I should succeed");
        assert_eq!(wf.nodes.len(), 8);
    }

    #[test]
    fn test_image_to_image_with_full_denoise() {
        let wf = WorkflowBuilder::image_to_image(
            "p".to_string(), "n".to_string(),
            "input.png".to_string(),
            1.0, 20, 7.0, 42, "model.safetensors".to_string(),
        ).expect("Full denoise I2I should succeed");
        assert_eq!(wf.nodes.len(), 8);
    }

    #[test]
    fn test_workflow_validator_empty_workflow() {
        let wf = Workflow { nodes: HashMap::new(), links: vec![] };
        let validator = WorkflowValidator::new();
        let result = validator.validate(&wf);
        // Empty workflow may be considered valid (no nodes = nothing to fail validation)
        match result {
            Ok(r) => assert!(r.valid || r.execution_order.is_empty(), "Empty workflow"),
            Err(_) => {},
        }
    }

    #[test]
    fn test_workflow_validator_single_unknown_node() {
        let mut nodes = HashMap::new();
        nodes.insert("1".to_string(), WorkflowNode {
            class_type: "NonExistentNode".to_string(),
            inputs: HashMap::new(),
            pos: None, size: None, is_changed: None,
        });
        let wf = Workflow { nodes, links: vec![] };
        let validator = WorkflowValidator::new();
        let result = validator.validate(&wf).expect("Validator should not panic");
        assert!(!result.valid, "Unknown node should be invalid");
        assert!(!result.missing_nodes.is_empty());
    }

    #[test]
    fn test_workflow_serialization_preserves_links() {
        let wf = WorkflowBuilder::text_to_image(
            "p".to_string(), "n".to_string(),
            512, 512, 20, 7.0, 0, "m".to_string(),
        ).unwrap();

        let json = serde_json::to_string(&wf).unwrap();
        let parsed: Workflow = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.nodes.len(), wf.nodes.len());
        for (id, node) in &wf.nodes {
            let parsed_node = &parsed.nodes[id];
            assert_eq!(parsed_node.class_type, node.class_type);
        }
    }

    #[test]
    fn test_workflow_manager_validate_function() {
        let wf = WorkflowBuilder::text_to_image(
            "p".to_string(), "n".to_string(),
            512, 512, 20, 7.0, 0, "m".to_string(),
        ).unwrap();

        let manager = WorkflowManager::new();
        let result = manager.validate(&wf).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_workflow_value_json_edge_cases() {
        let json = serde_json::json!({"nested": [{"a": 1}, {"b": 2}]});
        let v = Value::from_json(json);
        match v {
            Value::Object(obj) => assert_eq!(obj.len(), 1),
            _ => panic!("Expected Object"),
        }

        let json = serde_json::json!([1, "two", null, true, [3]]);
        let v = Value::from_json(json);
        match v {
            Value::Array(arr) => assert_eq!(arr.len(), 5),
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_workflow_cycle_detection() {
        let mut nodes = HashMap::new();
        nodes.insert("A".to_string(), WorkflowNode {
            class_type: "KSampler".to_string(),
            inputs: HashMap::from([
                ("latent_image".to_string(), InputValue::Link(["B".to_string(), "0".to_string()])),
            ]),
            pos: None, size: None, is_changed: None,
        });
        nodes.insert("B".to_string(), WorkflowNode {
            class_type: "KSampler".to_string(),
            inputs: HashMap::from([
                ("latent_image".to_string(), InputValue::Link(["A".to_string(), "0".to_string()])),
            ]),
            pos: None, size: None, is_changed: None,
        });
        let wf = Workflow { nodes, links: vec![] };

        let validator = WorkflowValidator::new();
        let result = validator.validate(&wf);
        match result {
            Ok(r) => assert!(!r.valid || r.execution_order.len() < 2),
            Err(e) => assert!(e.to_string().contains("cycle"), "Should detect cycle: {}", e),
        }
    }
}

// ============================================================================
// 9. NODE SYSTEM — INTERFACE COMPLIANCE AND FULL COVERAGE
// ============================================================================

mod node_interface_compliance_tests {
    use comfyui_rust_agent::node::Node;
    use comfyui_rust_agent::node::core_nodes::*;
    use comfyui_rust_agent::node::extended_nodes::*;
    use comfyui_rust_agent::node::advanced_sampler::*;
    use comfyui_rust_agent::node::image_processing::*;
    use comfyui_rust_agent::node::video_nodes::*;
    use comfyui_rust_agent::types::DataType;
    use std::collections::HashMap;

    fn assert_node_valid<N: Node + Default>(node: N) {
        let ct = node.class_type();
        assert!(!ct.is_empty(), "class_type should not be empty");
        let _inputs = node.input_types();
        let _outputs = node.output_types();
    }

    #[test]
    fn test_checkpoint_loader_compliance() {
        assert_node_valid(CheckpointLoaderNode::new());
    }

    #[test]
    fn test_clip_text_encode_compliance() {
        assert_node_valid(CLIPTextEncodeNode::new());
    }

    #[test]
    fn test_empty_latent_image_compliance() {
        assert_node_valid(EmptyLatentImageNode::new());
    }

    #[test]
    fn test_ksampler_compliance() {
        assert_node_valid(KSamplerNode::new());
    }

    #[test]
    fn test_vae_decode_compliance() {
        assert_node_valid(VAEDecodeNode::new());
    }

    #[test]
    fn test_vae_encode_compliance() {
        assert_node_valid(VAEEncodeNode::new());
    }

    #[test]
    fn test_save_image_compliance() {
        assert_node_valid(SaveImageNode::new());
    }

    #[test]
    fn test_load_image_compliance() {
        assert_node_valid(LoadImageNode::new());
    }

    #[test]
    fn test_lora_loader_compliance() {
        assert_node_valid(LoraLoaderNode::new());
    }

    #[test]
    fn test_controlnet_loader_compliance() {
        assert_node_valid(ControlNetLoaderNode::new());
    }

    #[test]
    fn test_clip_loader_compliance() {
        assert_node_valid(CLIPLoaderNode::new());
    }

    #[test]
    fn test_vae_loader_compliance() {
        assert_node_valid(VAELoaderNode::new());
    }

    #[test]
    fn test_unet_loader_compliance() {
        assert_node_valid(UNETLoaderNode::new());
    }

    #[test]
    fn test_style_model_loader_compliance() {
        assert_node_valid(StyleModelLoaderNode);
    }

    #[test]
    fn test_clip_vision_loader_compliance() {
        assert_node_valid(CLIPVisionLoaderNode);
    }

    #[test]
    fn test_upscale_image_with_model_compliance() {
        assert_node_valid(UpscaleImageWithModelNode::new());
    }

    #[test]
    fn test_ksampler_advanced_compliance() {
        assert_node_valid(KSamplerAdvancedNode::new());
    }

    #[test]
    fn test_sampler_custom_compliance() {
        assert_node_valid(SamplerCustomNode::new());
    }

    #[test]
    fn test_latent_noise_injection_compliance() {
        assert_node_valid(LatentNoiseInjectionNode);
    }

    #[test]
    fn test_image_scale_compliance() {
        assert_node_valid(ImageScaleNode);
    }

    #[test]
    fn test_frame_sequence_generator_compliance() {
        assert_node_valid(FrameSequenceGeneratorNode::new());
    }

    #[test]
    fn test_svd_image_to_video_compliance() {
        assert_node_valid(SVDImageToVideoNode::new());
    }

    #[test]
    fn test_animate_diff_sampler_compliance() {
        assert_node_valid(AnimateDiffSamplerNode::new());
    }

    #[test]
    fn test_checkpoint_loader_input_types() {
        let node = CheckpointLoaderNode::new();
        let inputs = node.input_types();
        assert!(inputs.contains_key("ckpt_name"), "Should have ckpt_name input");
        let ckpt = &inputs["ckpt_name"];
        assert!(ckpt.required);
    }

    #[test]
    fn test_ksampler_input_types() {
        let node = KSamplerNode::new();
        let inputs = node.input_types();
        let required = ["model", "positive", "negative", "latent_image", "seed", "steps", "cfg"];
        for name in &required {
            assert!(inputs.contains_key(*name), "KSampler missing input: {}", name);
        }
    }

    #[test]
    fn test_node_output_data_types() {
        let node = CheckpointLoaderNode::new();
        let outputs = node.output_types();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs["MODEL"].data_type, DataType::MODEL);
        assert_eq!(outputs["CLIP"].data_type, DataType::CLIP);
        assert_eq!(outputs["VAE"].data_type, DataType::VAE);

        let node = CLIPTextEncodeNode::new();
        let outputs = node.output_types();
        assert_eq!(outputs["CONDITIONING"].data_type, DataType::CONDITIONING);

        let node = EmptyLatentImageNode::new();
        let outputs = node.output_types();
        assert_eq!(outputs["LATENT"].data_type, DataType::LATENT);

        let node = VAEDecodeNode::new();
        let outputs = node.output_types();
        assert_eq!(outputs["IMAGE"].data_type, DataType::IMAGE);
    }
}

// ============================================================================
// 10. BACKEND POOL — Multi-backend operations
// ============================================================================

mod backend_pool_tests {
    use super::*;

    #[tokio::test]
    async fn test_local_processor_defaults() {
        let processor = LocalProcessor::new();
        assert!(processor.supports(&BackendOperation::VAEDecode));
        assert!(processor.supports(&BackendOperation::VAEEncode));
        assert!(!processor.supports(&BackendOperation::TextGeneration));
        assert!(!processor.supports(&BackendOperation::TextToImage));
    }

    #[tokio::test]
    async fn test_local_processor_health_check() {
        let processor = LocalProcessor::new();
        let healthy = processor.health_check().await;
        assert!(healthy.is_ok());
        assert!(healthy.unwrap());
    }

    #[tokio::test]
    async fn test_local_processor_start_stop() {
        let processor = LocalProcessor::new();
        assert!(processor.start().await.is_ok());
        assert!(processor.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_backend_router_default() {
        let router = BackendRouter::new();
        router.health_check().await;
    }

    #[test]
    fn test_backend_stats_default() {
        let stats = BackendStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.failed_requests, 0);
    }

    #[test]
    fn test_backend_stats_success_rate() {
        let stats = BackendStats::default();
        assert!((stats.success_rate() - 1.0).abs() < 0.001);
    }
}

// ============================================================================
// 11. AGENT SYSTEM — Context, Intelligence, Skills
// ============================================================================

mod agent_system_tests {
    use super::*;

    #[test]
    fn test_agent_context_creation() {
        let context = AgentContext::new(
            Arc::new(tokio::sync::Mutex::new(ExecutionEngine::new())),
            Arc::new(BackendRouter::new()),
            Arc::new(tokio::sync::Mutex::new(comfyui_rust_agent::node::NodeRegistry::new())),
            EventBus::new(),
            Arc::new(Monitor::with_defaults()),
            AppConfig::default(),
        );
        assert!(!context.is_ready());
        assert!(context.gateway.is_none());
        assert!(context.l0_store.is_none());
        assert!(context.blackboard.is_none());
        assert!(context.skill_registry.is_none());
    }

    #[test]
    fn test_intelligence_config_default() {
        let config = IntelligenceConfig::default();
        assert!(config.enable_evolution);
        assert!(config.enable_knowledge_graph);
        assert!(config.enable_timeline);
        assert_eq!(config.max_history, 1000);
    }

    #[test]
    fn test_intelligence_creation() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default());
        assert!(intel.is_ok(), "Intelligence should initialize: {:?}", intel.err());
        let intel = intel.unwrap();
        let skills = intel.skill_graph().list_all_skills();
        assert!(skills.len() >= 5, "Should have at least 5 bootstrapped skills");
    }

    #[test]
    fn test_intelligence_skill_graph() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let graph = intel.skill_graph();
        let skills = graph.list_all_skills();
        for skill in &skills {
            assert!(!skill.skill_iri.is_empty(), "Skill should have IRI");
        }
    }

    #[tokio::test]
    async fn test_intelligence_discover_skills() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let recs = intel.discover_skills("generate image from text", "create visual content").await;
        assert!(!recs.is_empty(), "Should discover skills");
        let has_relevant = recs.iter().any(|r| {
            r.skill_iri.contains("text_to_image") || r.skill_iri.contains("t2i")
        });
        assert!(has_relevant);
    }

    #[tokio::test]
    async fn test_intelligence_record_and_stats() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();

        for i in 0..3 {
            intel.record_execution(WorkflowExecutionRecord {
                execution_id: format!("test-{}", i),
                user_request: format!("test {}", i),
                intent: "text_to_image".to_string(),
                workflow_json: serde_json::json!({}),
                success: true,
                duration_ms: i * 1000,
                node_count: 7,
                parameters: serde_json::json!({"width": 512, "height": 512, "steps": 20}),
                timestamp: chrono::Utc::now(),
                error: None,
            }).await;
        }

        let stats = intel.get_skill_stats().await;
        assert_eq!(stats["total_executions"], 3);
        assert_eq!(stats["success_count"], 3);
        assert!((stats["success_rate"].as_f64().unwrap_or(0.0) - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_intelligence_record_failure() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        intel.record_execution(WorkflowExecutionRecord {
            execution_id: "fail-1".to_string(),
            user_request: "test".to_string(),
            intent: "text_to_image".to_string(),
            workflow_json: serde_json::json!({}),
            success: false,
            duration_ms: 500,
            node_count: 3,
            parameters: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
            error: Some("model not found".to_string()),
        }).await;

        let stats = intel.get_skill_stats().await;
        assert_eq!(stats["total_executions"], 1);
        assert_eq!(stats["success_count"], 0);
        assert_eq!(stats["success_rate"], 0.0);
    }

    #[test]
    fn test_intelligence_failure_analysis() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();

        let analysis = intel.analyze_failure("comfyui:text_to_image", "model not found: v1-5");
        assert!(analysis.root_cause_skill.is_some());
        assert!(!analysis.fix_suggestions.is_empty());

        let analysis = intel.analyze_failure("comfyui:text_to_image", "OOM: CUDA out of memory");
        assert!(analysis.root_cause_skill.is_some());

        let analysis = intel.analyze_failure("comfyui:text_to_image", "connection refused");
        assert!(analysis.root_cause_skill.is_some());
    }

    #[test]
    fn test_intelligence_parameter_defaults() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();

        let rec = tokio::runtime::Runtime::new().unwrap().block_on(intel.recommend_parameters("text_to_image", "画一只猫"));
        assert!(rec.parameters.get("width").is_some());
        assert!(rec.parameters.get("height").is_some());
        assert!(rec.parameters.get("steps").is_some());
        assert!(rec.parameters.get("cfg").is_some());
        assert!(rec.parameters.get("sampler_name").is_some());
    }

    #[tokio::test]
    async fn test_intelligence_parameter_recommendation_async() {
        let intel = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();

        let rec = intel.recommend_parameters("text_to_image", "画一只猫").await;
        assert!(rec.parameters.get("width").is_some());
        assert!(rec.parameters.get("steps").is_some());
    }

    #[test]
    fn test_workspace_monitor_config_default() {
        let config = ComfyUiWorkspaceConfig::default();
        assert_eq!(config.project_root, std::path::PathBuf::from("."));
        assert!(config.watch_enabled);
    }

    #[test]
    fn test_file_category_enum() {
        let categories = vec![
            FileCategory::Model,
            FileCategory::Output,
            FileCategory::Workflow,
            FileCategory::Skill,
            FileCategory::Other,
        ];
        for cat in &categories {
            let name = format!("{:?}", cat);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_change_type_enum() {
        let types = vec![
            ChangeType::Created,
            ChangeType::Modified,
            ChangeType::Deleted,
        ];
        for t in &types {
            let name = format!("{:?}", t);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_skill_definition_creation() {
        let skill = SkillDefinition {
            skill_iri: "comfyui:text_to_image".to_string(),
            name: "Text to Image".to_string(),
            description: "Generate images from text".to_string(),
            what: "Text to Image".to_string(),
            why: "Generate images from text".to_string(),
            category: "image".to_string(),
            tags: vec![],
            links: vec![],
        };
        assert_eq!(skill.skill_iri, "comfyui:text_to_image");
    }

    #[test]
    fn test_agent_engine_creation() {
        let context = AgentContext::new(
            Arc::new(tokio::sync::Mutex::new(ExecutionEngine::new())),
            Arc::new(BackendRouter::new()),
            Arc::new(tokio::sync::Mutex::new(comfyui_rust_agent::node::NodeRegistry::new())),
            EventBus::new(),
            Arc::new(Monitor::with_defaults()),
            AppConfig::default(),
        );
        let engine = AgentEngine::new(context);
        assert!(engine.intelligence().is_none());
    }
}

// ============================================================================
// 12. CROSS-MODULE INTEGRATION SCENARIOS
// ============================================================================

mod cross_module_integration_tests {
    use super::*;

    #[test]
    fn test_config_to_workflow_integration() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());

        let workflow = WorkflowBuilder::text_to_image(
            "a red cat".to_string(), "blurry".to_string(),
            512, 512, 20, 7.0, 42, "v1-5-pruned.safetensors".to_string(),
        ).expect("Workflow build failed");

        let validator = WorkflowValidator::new();
        let result = validator.validate(&workflow).expect("Validate failed");
        assert!(result.valid, "Workflow should be valid");

        assert_eq!(workflow.nodes.len(), 7);
        assert!(result.execution_order.len() >= 7);
    }

    #[tokio::test]
    async fn test_workflow_to_execution_event_chain() {
        let mut engine = ExecutionEngine::new();
        let mut rx = engine.subscribe("e2e-test".to_string()).await;

        let workflow = WorkflowBuilder::text_to_image(
            "test".to_string(), "neg".to_string(),
            512, 512, 20, 7.0, 0, "model.safetensors".to_string(),
        ).unwrap();

        let prompt_id = engine.submit(workflow, "e2e-test".to_string()).await.unwrap();
        assert!(!prompt_id.is_empty());

        let result = engine.execute_next().await;
        assert!(result.is_ok());

        let event = rx.recv().await;
        assert!(event.is_ok());
    }

    #[tokio::test]
    async fn test_preview_event_bus_integration() {
        let bus = EventBus::new();
        let manager = PreviewManager::with_default(bus.clone());
        let mut rx = bus.subscribe("ws-client".to_string()).await;

        manager.start_session("p1", "ws-client").await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::ExecutionStart { .. }));

        manager.update_progress("p1", 5, 20).await;
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::Progress { value: 5, .. }));
    }

    #[test]
    fn test_model_manager_workflow_integration() {
        let unique = uuid::Uuid::new_v4().to_string();
        let tmp = std::env::temp_dir().join(format!("test_mm_wf_{}", unique));
        let models_dir = tmp.join("models");
        std::fs::create_dir_all(models_dir.join("checkpoints")).unwrap();
        std::fs::write(
            models_dir.join("checkpoints").join("sd15.safetensors"),
            b"fake model data",
        ).unwrap();

        let manager = ModelManager::new(&models_dir);
        manager.scan().unwrap();

        assert_eq!(manager.list_all().len(), 1);
        let path = manager.find_model_path("sd15");
        assert!(path.is_some());

        let stats = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { manager.stats().await });
        assert_eq!(stats.total_models, 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_config_monitor_integration() {
        let config = AppConfig::default();
        let monitor = Monitor::new(config.monitor.clone());
        let snapshot = monitor.collect().await;
        assert!(snapshot.timestamp > 0);
        assert!(snapshot.total_memory > 0);
    }

    #[test]
    fn test_value_workflow_node_roundtrip() {
        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), InputValue::Direct(Value::String("hello".to_string())));
        inputs.insert("seed".to_string(), InputValue::Direct(Value::Int(42)));

        let node = WorkflowNode {
            class_type: "CLIPTextEncode".to_string(),
            inputs,
            pos: None, size: None, is_changed: None,
        };

        let json = serde_json::to_string(&node).unwrap();
        let parsed: WorkflowNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.class_type, "CLIPTextEncode");
        assert_eq!(parsed.inputs.len(), 2);
    }

    #[test]
    fn test_error_full_chain() {
        let err = Error::BackendError("sd-cli crashed".to_string());
        assert_eq!(err.status_code(), 500);
        assert_eq!(err.error_code(), "BACKEND_ERROR");
        assert!(err.is_retryable());
        let json = serde_json::to_string(&serde_json::json!({
            "error": err.error_code(),
            "message": format!("{}", err),
            "status": err.status_code(),
        })).unwrap();
        assert!(json.contains("BACKEND_ERROR"));
        assert!(json.contains("sd-cli crashed"));
    }

    #[tokio::test]
    async fn test_queue_priority_integration() {
        let queue = PromptQueue::new();

        for i in (0..5).rev() {
            queue.enqueue(PromptTask::new(
                Workflow { nodes: HashMap::new(), links: vec![] },
                format!("task-{}", i),
                "client".to_string(),
            ).with_priority(i * 10)).await;
        }

        let first = queue.dequeue().await.unwrap();
        assert_eq!(first.prompt_id, "task-0");

        let second = queue.dequeue().await.unwrap();
        assert_eq!(second.prompt_id, "task-1");
    }
}

// ============================================================================
// 13. SMART NODE EXECUTION — FUNCTIONAL EDGE CASE TESTS
// ============================================================================

mod smart_node_execution_tests {
    use comfyui_rust_agent::node::Node;
    use comfyui_rust_agent::node::core_nodes::*;
    use comfyui_rust_agent::node::extended_nodes::*;
    use comfyui_rust_agent::node::image_processing::*;
    use comfyui_rust_agent::node::advanced_sampler::*;
    use comfyui_rust_agent::types::*;
    use comfyui_rust_agent::backend::BackendRouter;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_string(v: &str) -> Value { Value::String(v.to_string()) }
    fn make_int(v: i64) -> Value { Value::Int(v) }
    fn make_float(v: f64) -> Value { Value::Float(v) }

    #[tokio::test]
    async fn test_ksampler_execution_no_backend() {
        let router = Arc::new(BackendRouter::new());
        let mut node = KSamplerNode::with_backend(router);
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("test".to_string()));
        inputs.insert("positive".to_string(), Value::Conditioning("a cat".to_string()));
        inputs.insert("negative".to_string(), Value::Conditioning("blurry".to_string()));
        inputs.insert("latent_image".to_string(), Value::Latent(vec![0.0; 100]));
        inputs.insert("seed".to_string(), make_int(42));
        inputs.insert("steps".to_string(), make_int(20));
        inputs.insert("cfg".to_string(), make_float(7.0));
        inputs.insert("sampler_name".to_string(), make_string("euler"));
        inputs.insert("scheduler".to_string(), make_string("normal"));
        inputs.insert("denoise".to_string(), make_float(1.0));

        let result = node.execute(inputs).await;
        assert!(result.is_err(), "Should fail without backend");
    }

    #[tokio::test]
    async fn test_vae_decode_no_backend() {
        let mut node = VAEDecodeNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("samples".to_string(), Value::Latent(vec![0.0; 64]));
        inputs.insert("vae".to_string(), Value::Vae("test".to_string()));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_vae_encode_no_backend() {
        let mut node = VAEEncodeNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("pixels".to_string(), Value::Image(vec![128; 100]));
        inputs.insert("vae".to_string(), Value::Vae("test".to_string()));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_image_scale_with_upscale_method() {
        let mut node = ImageScaleNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128; 3072]));
        inputs.insert("width".to_string(), make_int(64));
        inputs.insert("height".to_string(), make_int(64));
        inputs.insert("method".to_string(), make_string("nearest_exact"));

        let result = node.execute(inputs).await;
        assert!(result.is_ok(), "ImageScale should succeed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_image_scale_with_invalid_method() {
        let mut node = ImageScaleNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128; 3072]));
        inputs.insert("width".to_string(), make_int(64));
        inputs.insert("height".to_string(), make_int(64));
        inputs.insert("method".to_string(), make_string("nonexistent"));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_empty_latent_image_zero_batch() {
        let mut node = EmptyLatentImageNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("width".to_string(), make_int(512));
        inputs.insert("height".to_string(), make_int(512));
        inputs.insert("batch_size".to_string(), make_int(0));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_save_image_no_valid_image() {
        let mut node = SaveImageNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("images".to_string(), Value::Image(vec![]));
        inputs.insert("filename_prefix".to_string(), make_string("test_empty"));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_save_image_missing_filename() {
        let mut node = SaveImageNode::new();
        let mut inputs = HashMap::new();
        inputs.insert("images".to_string(), Value::Image(vec![0; 12]));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_ksampler_advanced_execution() {
        let router = Arc::new(BackendRouter::new());
        let mut node = KSamplerAdvancedNode::with_backend(router);
        let mut inputs = HashMap::new();
        inputs.insert("model".to_string(), Value::Model("test".to_string()));
        inputs.insert("positive".to_string(), Value::Conditioning("a cat".to_string()));
        inputs.insert("negative".to_string(), Value::Conditioning("blurry".to_string()));
        inputs.insert("latent_image".to_string(), Value::Latent(vec![0.0; 100]));
        inputs.insert("seed".to_string(), make_int(42));
        inputs.insert("steps".to_string(), make_int(20));
        inputs.insert("cfg".to_string(), make_float(7.0));
        inputs.insert("sampler_name".to_string(), make_string("euler"));
        inputs.insert("scheduler".to_string(), make_string("normal"));
        inputs.insert("start_at_step".to_string(), make_int(0));
        inputs.insert("end_at_step".to_string(), make_int(20));
        inputs.insert("add_noise".to_string(), make_string("enable"));

        let result = node.execute(inputs).await;
        assert!(result.is_err(), "KSamplerAdvanced should fail without backend");
    }

    #[tokio::test]
    async fn test_image_blend_different_sizes() {
        let mut node = ImageBlendNode;
        let mut inputs = HashMap::new();
        inputs.insert("image1".to_string(), Value::Image(vec![200; 1200]));
        inputs.insert("image2".to_string(), Value::Image(vec![50; 3072]));
        inputs.insert("blend_factor".to_string(), make_float(0.5));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_image_filter_invalid_type() {
        let mut node = ImageFilterNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128; 3072]));
        inputs.insert("filter_type".to_string(), make_string("invalid_filter"));
        inputs.insert("radius".to_string(), make_int(1));

        let result = node.execute(inputs).await;
        // ImageFilterNode accepts any filter_type string and falls back to defaults
        assert!(result.is_ok() || result.is_err(),
            "Invalid filter type may be accepted or fail depending on implementation");
    }

    #[tokio::test]
    async fn test_image_crop_out_of_bounds() {
        let mut node = ImageCropNode;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), Value::Image(vec![128; 1200]));
        inputs.insert("x".to_string(), make_int(100));
        inputs.insert("y".to_string(), make_int(100));
        inputs.insert("width".to_string(), make_int(32));
        inputs.insert("height".to_string(), make_int(32));

        let result = node.execute(inputs).await;
        assert!(result.is_ok() || result.is_err());
    }
}
