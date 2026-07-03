// WebSocket 处理模块
// 实时事件推送

use crate::execution::{Event, EventBus};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::IntoResponse,
};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use log::{info, warn};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

/// WebSocket 连接查询参数
#[derive(Debug, Deserialize)]
pub struct WsQuery {
    #[serde(default)]
    pub client_id: Option<String>,
}

/// WebSocket 状态
#[derive(Clone)]
pub struct WsState {
    pub event_bus: EventBus,
    pub active_connections: Arc<RwLock<usize>>,
}

impl WsState {
    pub fn new(event_bus: EventBus) -> Self {
        Self {
            event_bus,
            active_connections: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn get_active_connections(&self) -> usize {
        *self.active_connections.read().await
    }
}

/// WebSocket 升级处理器
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(state): State<WsState>,
) -> impl IntoResponse {
    let client_id = query.client_id.unwrap_or_else(|| {
        uuid::Uuid::new_v4().to_string()
    });

    info!("WebSocket connection from client: {}", client_id);

    ws.on_upgrade(move |socket| handle_ws_connection(socket, client_id, state))
}

/// 处理 WebSocket 连接
async fn handle_ws_connection(socket: WebSocket, client_id: String, state: WsState) {
    // 增加活跃连接数
    {
        let mut count = state.active_connections.write().await;
        *count += 1;
    }

    info!("WebSocket connected: {} (total: {})", client_id, {
        let count = state.active_connections.read().await;
        *count
    });

    // 订阅事件
    let mut event_rx = state.event_bus.subscribe(client_id.clone()).await;

    let (mut sender, mut receiver) = socket.split::<Message>();

    // 为子任务克隆 client_id
    let client_id_for_event = client_id.clone();
    let client_id_for_receive = client_id.clone();

    // 启动事件转发任务
    let event_task = tokio::spawn(async move {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    let json = match serde_json::to_string(&event) {
                        Ok(j) => j,
                        Err(e) => {
                            warn!("Failed to serialize event: {}", e);
                            continue;
                        }
                    };

                    let message = match &event {
                        Event::Preview { data, .. } => {
                            // 预览图使用二进制消息
                            let mut payload = Vec::new();
                            payload.extend_from_slice(b"preview:");
                            payload.extend_from_slice(data);
                            Message::Binary(payload)
                        }
                        _ => Message::Text(json),
                    };

                    if sender.send(message).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("WebSocket client {} lagged by {} events", client_id_for_event, n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // 处理客户端消息（保持连接，直到客户端断开）
    let receive_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    debug!("Received from {}: {}", client_id_for_receive, text);
                    // 处理客户端消息（如心跳、命令等）
                    if text == "ping" {
                        // 客户端发送 ping，无需响应（axum 会自动处理 Ping/Pong 帧）
                    }
                }
                Message::Binary(_) => {
                    // 忽略二进制消息
                }
                Message::Ping(_) | Message::Pong(_) => {
                    // Ping/Pong 由 axum 自动处理
                }
                Message::Close(_) => {
                    info!("WebSocket closed by client: {}", client_id_for_receive);
                    break;
                }
            }
        }
    });

    // 等待任一任务完成
    tokio::select! {
        _ = event_task => {
            info!("Event task ended for client: {}", client_id);
        }
        _ = receive_task => {
            info!("Receive task ended for client: {}", client_id);
        }
    }

    // 减少活跃连接数
    {
        let mut count = state.active_connections.write().await;
        if *count > 0 {
            *count -= 1;
        }
    }

    // 取消订阅
    state.event_bus.unsubscribe(&client_id).await;

    info!("WebSocket disconnected: {} (remaining: {})", client_id, {
        let count = state.active_connections.read().await;
        *count
    });
}

// 引入调试日志
use log::debug;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_query_default() {
        let query = WsQuery { client_id: None };
        assert!(query.client_id.is_none());
    }

    #[tokio::test]
    async fn test_ws_state_connections() {
        let event_bus = EventBus::new();
        let state = WsState::new(event_bus);

        assert_eq!(state.get_active_connections().await, 0);

        {
            let mut count = state.active_connections.write().await;
            *count = 5;
        }

        assert_eq!(state.get_active_connections().await, 5);
    }
}
