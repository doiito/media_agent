// 执行引擎模块

mod queue;
mod executor;
mod cache;
pub mod event;

pub use queue::PromptQueue;
pub use executor::PromptExecutor;
pub use cache::HierarchicalCache;
pub use event::{Event, EventBus, SystemStatus};

use crate::types::*;

/// 执行引擎管理器
pub struct ExecutionEngine {
    queue: PromptQueue,
    executor: PromptExecutor,
    event_bus: EventBus,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        Self {
            queue: PromptQueue::new(),
            executor: PromptExecutor::new(),
            event_bus: EventBus::new(),
        }
    }

    /// 提交工作流到队列
    pub async fn submit(&mut self, workflow: Workflow, client_id: String) -> Result<PromptId, Error> {
        let prompt_id = uuid::Uuid::new_v4().to_string();
        let task = PromptTask::new(workflow, prompt_id.clone(), client_id);

        self.queue.enqueue(task).await;

        Ok(prompt_id)
    }

    /// 执行下一个任务
    pub async fn execute_next(&mut self) -> Result<Option<ExecutionResult>, Error> {
        if let Some(task) = self.queue.dequeue().await {
            // 发送执行开始事件
            self.event_bus
                .publish(Event::ExecutionStart {
                    prompt_id: task.prompt_id.clone(),
                })
                .await;

            // 执行工作流
            let result = self.executor.execute(&task.workflow).await?;

            // 发送完成事件
            match &result {
                ExecutionResult::Success(outputs) => {
                    self.event_bus
                        .publish(Event::ExecutionSuccess {
                            prompt_id: task.prompt_id,
                            outputs: outputs.clone(),
                        })
                        .await;
                }
                ExecutionResult::Failure(err) => {
                    self.event_bus
                        .publish(Event::ExecutionError {
                            prompt_id: task.prompt_id,
                            error: err.clone(),
                        })
                        .await;
                }
                _ => {}
            }

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// 获取事件总线订阅
    pub async fn subscribe(&self, client_id: String) -> tokio::sync::broadcast::Receiver<Event> {
        self.event_bus.subscribe(client_id).await
    }

    /// 中断当前任务
    pub fn interrupt(&mut self) {
        self.executor.interrupt();
    }

    /// 释放显存
    pub async fn free_memory(&mut self) {
        self.executor.free_memory().await;
    }
}