// 执行事件系统

use crate::types::*;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use std::collections::HashMap;

/// 执行事件
#[derive(Debug, Clone, Serialize)]
pub enum Event {
    /// 执行开始
    ExecutionStart {
        prompt_id: String,
    },
    /// 节点缓存命中
    ExecutionCached {
        prompt_id: String,
        nodes: Vec<NodeId>,
    },
    /// 节点执行中
    Executing {
        prompt_id: String,
        node_id: Option<NodeId>, // null表示全部完成
    },
    /// 采样进度
    Progress {
        prompt_id: String,
        value: usize,
        max: usize,
    },
    /// 预览图
    Preview {
        prompt_id: String,
        node_id: NodeId,
        data: Vec<u8>, // 图像数据
    },
    /// 执行成功
    ExecutionSuccess {
        prompt_id: String,
        outputs: HashMap<NodeId, HashMap<String, Value>>,
    },
    /// 执行失败
    ExecutionError {
        prompt_id: String,
        error: String,
    },
    /// 执行中断
    ExecutionInterrupted {
        prompt_id: String,
    },
    /// 系统状态
    Status {
        status: SystemStatus,
    },
}

/// 系统状态
#[derive(Debug, Clone, Serialize)]
pub struct SystemStatus {
    pub queue_remaining: usize,
    pub executing: Option<String>,
}

/// 事件总线
pub struct EventBus {
    /// 订阅者列表
    subscribers: Arc<RwLock<HashMap<String, broadcast::Sender<Event>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 订阅事件
    pub async fn subscribe(&self, client_id: String) -> broadcast::Receiver<Event> {
        let mut subs = self.subscribers.write().await;
        let (tx, rx) = broadcast::channel(100);
        subs.insert(client_id, tx);
        rx
    }

    /// 取消订阅
    pub async fn unsubscribe(&self, client_id: &str) {
        let mut subs = self.subscribers.write().await;
        subs.remove(client_id);
    }

    /// 发布事件到所有订阅者
    pub async fn publish(&self, event: Event) {
        let subs = self.subscribers.read().await;
        for (_, tx) in subs.iter() {
            tx.send(event.clone()).ok();
        }
    }

    /// 发布事件到特定订阅者
    pub async fn publish_to(&self, client_id: &str, event: Event) {
        let subs = self.subscribers.read().await;
        if let Some(tx) = subs.get(client_id) {
            tx.send(event).ok();
        }
    }

    /// 获取订阅者数量
    pub async fn subscriber_count(&self) -> usize {
        let subs = self.subscribers.read().await;
        subs.len()
    }
}

impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            subscribers: self.subscribers.clone(),
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_bus_subscribe_unsubscribe() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count().await, 0);

        let _rx1 = bus.subscribe("client-1".to_string()).await;
        assert_eq!(bus.subscriber_count().await, 1);

        bus.unsubscribe("client-1").await;
        assert_eq!(bus.subscriber_count().await, 0);
    }

    #[tokio::test]
    async fn test_event_bus_publish() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe("client-1".to_string()).await;

        bus.publish(Event::ExecutionStart {
            prompt_id: "test-prompt".to_string(),
        })
        .await;

        let event = rx.recv().await;
        assert!(event.is_ok());

        match event.unwrap() {
            Event::ExecutionStart { prompt_id } => {
                assert_eq!(prompt_id, "test-prompt");
            }
            _ => panic!("Expected ExecutionStart event"),
        }
    }

    #[tokio::test]
    async fn test_event_bus_publish_to() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe("c1".to_string()).await;
        let mut rx2 = bus.subscribe("c2".to_string()).await;

        bus.publish_to(
            "c2",
            Event::ExecutionStart {
                prompt_id: "private".to_string(),
            },
        )
        .await;

        let event2 = rx2.recv().await.unwrap();
        match event2 {
            Event::ExecutionStart { prompt_id } => assert_eq!(prompt_id, "private"),
            _ => panic!("Expected ExecutionStart"),
        }

        let event1 = rx1.try_recv();
        assert!(event1.is_err(), "c1 should not receive the event");
    }

    #[tokio::test]
    async fn test_event_bus_multiple_subscribers() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe("c1".to_string()).await;
        let mut rx2 = bus.subscribe("c2".to_string()).await;

        bus.publish(Event::Status {
            status: SystemStatus {
                queue_remaining: 5,
                executing: Some("node-1".to_string()),
            },
        })
        .await;

        for rx in [&mut rx1, &mut rx2] {
            let event = rx.recv().await.unwrap();
            match event {
                Event::Status { status } => {
                    assert_eq!(status.queue_remaining, 5);
                    assert_eq!(status.executing, Some("node-1".to_string()));
                }
                _ => panic!("Expected Status event"),
            }
        }
    }

    #[tokio::test]
    async fn test_event_serialization() {
        let event = Event::ExecutionStart {
            prompt_id: "test-123".to_string(),
        };

        let json = serde_json::to_string(&event).expect("Failed to serialize");
        assert!(json.contains("test-123"));
    }

    #[tokio::test]
    async fn test_event_bus_clone() {
        let bus1 = EventBus::new();
        let _rx = bus1.subscribe("c1".to_string()).await;

        let bus2 = bus1.clone();
        // 克隆后共享同一份数据
        assert_eq!(bus2.subscriber_count().await, 1);
    }
}
