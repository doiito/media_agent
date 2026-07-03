// ComfyUI Rust Agent Server
// 启动 HTTP/WebSocket API 服务，集成配置管理和监控

use comfyui_rust_agent::*;
use comfyui_rust_agent::api::server::ApiServer;
use comfyui_rust_agent::backend::BackendRouter;
use comfyui_rust_agent::config::AppConfig;
use comfyui_rust_agent::execution::ExecutionEngine;
use comfyui_rust_agent::node::registry::NodeRegistry;
use comfyui_rust_agent::util::logger;
use log::{info, error, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载配置（文件 > 默认值，环境变量最后覆盖）
    let mut config = AppConfig::load_or_default();
    config.merge_env_overrides();

    // 初始化日志（基于配置级别）
    logger::init_logger_with_level(&config.log.level);

    info!("{}", get_project_info());

    // 验证配置
    if let Err(e) = config.validate() {
        error!("Configuration error: {}", e);
        return Err(e.into());
    }

    // 确保所有必要目录存在
    if let Err(e) = config.ensure_directories() {
        warn!("Failed to create directories: {}", e);
    }

    // 打印配置摘要
    info!("Configuration loaded:");
    info!("  Server: {}:{}", config.server.host, config.server.port);
    info!("  Output: {}", config.server.output_dir);
    info!("  Models: {}", config.paths.models_dir);
    info!("  Log level: {}", config.log.level);
    info!("  Monitor enabled: {}", config.monitor.enabled);
    info!(
        "  SD.cpp backend: {} ({})",
        config.sd_cpp.backend, config.sd_cpp.precision
    );

    // 创建后端路由器
    let backend_router = BackendRouter::with_configs(
        config.sd_cpp.clone(),
        config.llama_cpp.clone(),
    );

    // 创建执行引擎和节点注册表
    let engine = ExecutionEngine::new();
    let node_registry = NodeRegistry::new();

    // 从配置创建 API 服务器
    let server = ApiServer::from_config(config, engine, backend_router, node_registry);

    info!("Starting ComfyUI Rust Agent server");
    info!("API endpoints:");
    info!("  GET  /health             - 健康检查");
    info!("  GET  /system_stats       - 系统状态");
    info!("  GET  /stats              - 服务器统计");
    info!("  POST /prompt             - 提交工作流");
    info!("  POST /interrupt          - 中断执行");
    info!("  POST /free               - 释放内存");
    info!("  GET  /queue              - 队列状态");
    info!("  GET  /history            - 执行历史");
    info!("  GET  /object_info        - 节点信息");
    info!("  GET  /models/{{folder}}    - 模型列表");
    info!("  GET  /view               - 查看图像");
    info!("  POST /upload/image       - 上传图像");
    info!("  WS   /ws                 - WebSocket 实时事件");
    info!("Monitor endpoints:");
    info!("  GET  /monitor/latest     - 最新监控指标");
    info!("  GET  /monitor/history    - 历史监控数据");
    info!("  GET  /monitor/averages   - 平均指标");
    info!("  GET  /monitor/processes  - 相关进程信息");
    info!("  GET  /monitor/alerts     - 告警列表");
    info!("  POST /monitor/collect    - 主动采集一次");

    // 启动服务器
    if let Err(e) = server.run().await {
        error!("Server error: {}", e);
        return Err(e);
    }

    Ok(())
}
