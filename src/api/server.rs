// API 服务器
// Axum-based HTTP + WebSocket server

use crate::api::handlers::*;
use crate::api::websocket::{ws_handler, WsState};
use crate::backend::BackendRouter;
use crate::config::AppConfig;
use crate::execution::{EventBus, ExecutionEngine};
use crate::monitor::Monitor;
use crate::node::registry::NodeRegistry;
use crate::agent::{AgentContext, AgentEngine};
use axum::{
    routing::{delete, get, post},
    Router,
};
use log::info;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};

/// API 服务器配置
#[derive(Debug, Clone)]
pub struct ApiServerConfig {
    pub host: String,
    pub port: u16,
    pub output_dir: String,
}

impl Default for ApiServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8188,
            output_dir: "output".to_string(),
        }
    }
}

impl From<&AppConfig> for ApiServerConfig {
    fn from(config: &AppConfig) -> Self {
        Self {
            host: config.server.host.clone(),
            port: config.server.port,
            output_dir: config.server.output_dir.clone(),
        }
    }
}

/// API 服务器
pub struct ApiServer {
    config: ApiServerConfig,
    engine: Arc<Mutex<ExecutionEngine>>,
    backend_router: Arc<BackendRouter>,
    node_registry: Arc<Mutex<NodeRegistry>>,
    event_bus: EventBus,
    monitor: Arc<Monitor>,
    /// Agent Engine（gliding_horse 集成）
    agent: Arc<tokio::sync::Mutex<AgentEngine>>,
}

impl ApiServer {
    pub fn new(
        config: ApiServerConfig,
        engine: ExecutionEngine,
        backend_router: BackendRouter,
        node_registry: NodeRegistry,
        monitor: Monitor,
    ) -> Self {
        let event_bus = EventBus::new();
        let engine_arc = Arc::new(Mutex::new(engine));
        let backend_arc = Arc::new(backend_router);
        let nodes_arc = Arc::new(Mutex::new(node_registry));
        let monitor_arc = Arc::new(monitor);
        
        // 创建 AgentContext 和 AgentEngine
        let agent_context = AgentContext::new(
            engine_arc.clone(),
            backend_arc.clone(),
            nodes_arc.clone(),
            event_bus.clone(),
            monitor_arc.clone(),
            AppConfig::default(),
        );
        let agent = Arc::new(tokio::sync::Mutex::new(AgentEngine::new(agent_context)));
        
        Self {
            config,
            engine: engine_arc,
            backend_router: backend_arc,
            node_registry: nodes_arc,
            event_bus,
            monitor: monitor_arc,
            agent,
        }
    }

    /// 从应用配置创建
    pub fn from_config(
        app_config: AppConfig,
        engine: ExecutionEngine,
        backend_router: BackendRouter,
        node_registry: NodeRegistry,
    ) -> Self {
        let monitor = Monitor::new(app_config.monitor.clone());
        Self::new(
            ApiServerConfig::from(&app_config),
            engine,
            backend_router,
            node_registry,
            monitor,
        )
    }

    /// 启动 API 服务器
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        info!("Starting API server on {}", addr);

        // 启动后台监控采集
        self.monitor.clone().start().await;
        info!("Monitor started in background");

        // 创建共享状态
        let api_state = ApiState {
            engine: self.engine.clone(),
            backend_router: self.backend_router.clone(),
            node_registry: self.node_registry.clone(),
            start_time: Instant::now(),
            stats: Arc::new(RwLock::new(ServerStats::default())),
            history: Arc::new(RwLock::new(std::collections::HashMap::new())),
            output_dir: self.config.output_dir.clone(),
            monitor: self.monitor.clone(),
            agent: self.agent.clone(),
        };

        let ws_state = WsState::new(self.event_bus.clone());

        // 创建路由
        let app = build_router(api_state, ws_state);

        // 启动 HTTP 服务器
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("API server listening on http://{}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// 构建路由（使用两个State需要nest结构）
fn build_router(api_state: ApiState, ws_state: WsState) -> Router {
    // Agent API handlers
    use crate::agent::handlers::{agent_chat, agent_status, agent_workflows, agent_init};
    
    // 主路由 - 使用 ApiState
    let api_routes = Router::new()
        // 健康检查和状态
        .route("/health", get(health_check))
        .route("/system_stats", get(system_stats))
        .route("/stats", get(get_stats))
        // 工作流 API
        .route("/prompt", post(submit_prompt))
        .route("/interrupt", post(interrupt))
        .route("/free", post(free_memory))
        // 队列和历史
        .route("/queue", get(get_queue))
        .route("/history", get(get_history))
        .route("/history", delete(clear_history))
        // 节点信息
        .route("/object_info", get(get_object_info))
        .route("/object_info/{node_class}", get(get_node_info))
        // 模型管理
        .route("/models/{folder}", get(get_models))
        // 图像
        .route("/view", get(view_image))
        .route("/upload/image", post(upload_image))
        // 监控 API
        .route("/monitor/latest", get(monitor_latest))
        .route("/monitor/history", get(monitor_history))
        .route("/monitor/averages", get(monitor_averages))
        .route("/monitor/processes", get(monitor_processes))
        .route("/monitor/alerts", get(monitor_alerts))
        .route("/monitor/alerts", delete(monitor_clear_alerts))
        .route("/monitor/collect", post(monitor_collect))
        // Agent API（gliding_horse 集成）
        .route("/agent/chat", post(agent_chat))
        .route("/agent/status", get(agent_status))
        .route("/agent/workflows", get(agent_workflows))
        .route("/agent/init", post(agent_init))
        .with_state(api_state);

    // WebSocket 路由 - 使用 WsState
    let ws_routes = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(ws_state);

    // 合并路由
    Router::new()
        .merge(api_routes)
        .merge(ws_routes)
        .fallback(get(serve_static))
        .with_state(())
}

/// 请求日志中间件
async fn log_requests(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    if duration.as_millis() > 100 {
        info!(
            "{} {} -> {} (took {:?})",
            method,
            path,
            status,
            duration
        );
    } else {
        info!("{} {} -> {}", method, path, status);
    }

    response
}

/// 提供静态文件服务
async fn serve_static(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> axum::response::Response {
    let file_path = if path.is_empty() {
        std::path::PathBuf::from("web/index.html")
    } else {
        std::path::PathBuf::from("web").join(&path)
    };

    if !file_path.exists() {
        return axum::response::Response::builder()
            .status(axum::http::StatusCode::NOT_FOUND)
            .body(axum::body::Body::from("Not Found"))
            .unwrap();
    }

    let data = match std::fs::read(&file_path) {
        Ok(d) => d,
        Err(_) => {
            return axum::response::Response::builder()
                .status(axum::http::StatusCode::NOT_FOUND)
                .body(axum::body::Body::from("Not Found"))
                .unwrap();
        }
    };

    let mime = file_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| match ext {
            "html" => "text/html",
            "css" => "text/css",
            "js" => "application/javascript",
            "json" => "application/json",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "svg" => "image/svg+xml",
            _ => "application/octet-stream",
        })
        .unwrap_or("application/octet-stream");

    axum::response::Response::builder()
        .header(axum::http::header::CONTENT_TYPE, mime)
        .body(axum::body::Body::from(data))
        .unwrap()
}

impl Default for ApiServer {
    fn default() -> Self {
        let engine = ExecutionEngine::new();
        let backend_router = BackendRouter::from_env();
        let node_registry = NodeRegistry::new();
        let monitor = Monitor::with_defaults();
        Self::new(
            ApiServerConfig::default(),
            engine,
            backend_router,
            node_registry,
            monitor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ApiServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8188);
        assert_eq!(config.output_dir, "output");
    }

    #[test]
    fn test_server_creation() {
        let server = ApiServer::default();
        assert_eq!(server.config.port, 8188);
    }
}
