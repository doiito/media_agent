// Prompt任务队列

use crate::types::*;
use std::collections::BinaryHeap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Prompt任务队列 (优先级队列)
pub struct PromptQueue {
    /// 队列数据
    queue: Arc<Mutex<BinaryHeap<PromptTask>>>,
    /// 当前执行的任务
    current_task: Arc<Mutex<Option<PromptTask>>>,
}

impl PromptQueue {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            current_task: Arc::new(Mutex::new(None)),
        }
    }

    /// 入队任务
    pub async fn enqueue(&self, task: PromptTask) {
        let mut queue = self.queue.lock().await;
        queue.push(task);
    }

    /// 入队任务（插队到最前面）
    pub async fn enqueue_front(&self, task: PromptTask) {
        let mut queue = self.queue.lock().await;
        let high_priority_task = task.with_priority(0); // 最高优先级
        queue.push(high_priority_task);
    }

    /// 出队任务
    pub async fn dequeue(&self) -> Option<PromptTask> {
        let mut queue = self.queue.lock().await;
        if let Some(task) = queue.pop() {
            let mut current = self.current_task.lock().await;
            *current = Some(task.clone());
            Some(task)
        } else {
            None
        }
    }

    /// 查看队头任务
    pub async fn peek(&self) -> Option<PromptTask> {
        let queue = self.queue.lock().await;
        queue.peek().cloned()
    }

    /// 队列大小
    pub async fn size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }

    /// 剩余任务数
    pub async fn remaining(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }

    /// 清空队列
    pub async fn clear(&self) {
        let mut queue = self.queue.lock().await;
        queue.clear();
    }

    /// 中断当前任务
    pub async fn interrupt_current(&self) -> Option<PromptTask> {
        let mut current = self.current_task.lock().await;
        current.take()
    }

    /// 获取所有队列任务信息
    pub async fn get_queue_info(&self) -> Vec<(String, usize)> {
        let queue = self.queue.lock().await;
        queue.iter().map(|t| (t.prompt_id.clone(), t.priority)).collect()
    }
}