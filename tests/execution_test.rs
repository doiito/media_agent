// 执行引擎测试
// 测试队列、事件总线、执行引擎

use comfyui_rust_agent::types::*;
use comfyui_rust_agent::execution::{ExecutionEngine, EventBus, PromptQueue, Event, SystemStatus};
use comfyui_rust_agent::workflow::WorkflowBuilder;

#[tokio::test]
async fn test_prompt_queue_basic() {
    let queue = PromptQueue::new();
    assert_eq!(queue.size().await, 0);

    let workflow = WorkflowBuilder::text_to_image(
        "test".to_string(),
        "neg".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model".to_string(),
    )
    .unwrap();

    let task = PromptTask::new(workflow, "test-1".to_string(), "client-1".to_string());
    queue.enqueue(task).await;
    assert_eq!(queue.size().await, 1);

    let dequeued = queue.dequeue().await;
    assert!(dequeued.is_some());
    assert_eq!(dequeued.unwrap().prompt_id, "test-1");

    assert_eq!(queue.size().await, 0);
}

#[tokio::test]
async fn test_prompt_queue_priority() {
    let queue = PromptQueue::new();

    // 创建两个任务
    let wf = WorkflowBuilder::text_to_image(
        "p1".to_string(),
        "n".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "m".to_string(),
    )
    .unwrap();
    let task1 = PromptTask::new(wf, "normal".to_string(), "c1".to_string());

    let wf2 = WorkflowBuilder::text_to_image(
        "p2".to_string(),
        "n".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "m".to_string(),
    )
    .unwrap();
    let task2 = PromptTask::new(wf2, "priority".to_string(), "c2".to_string());

    queue.enqueue(task1).await;
    queue.enqueue_front(task2).await;

    let first = queue.dequeue().await.unwrap();
    assert_eq!(first.prompt_id, "priority");
}

#[tokio::test]
async fn test_prompt_queue_clear() {
    let queue = PromptQueue::new();

    for i in 0..5 {
        let wf = WorkflowBuilder::text_to_image(
            format!("p{}", i),
            "n".to_string(),
            512,
            512,
            20,
            7.0,
            42,
            "m".to_string(),
        )
        .unwrap();
        let task = PromptTask::new(wf, format!("id-{}", i), "c".to_string());
        queue.enqueue(task).await;
    }

    assert_eq!(queue.size().await, 5);
    queue.clear().await;
    assert_eq!(queue.size().await, 0);
}

#[tokio::test]
async fn test_prompt_queue_interrupt_current() {
    let queue = PromptQueue::new();

    let wf = WorkflowBuilder::text_to_image(
        "p".to_string(),
        "n".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "m".to_string(),
    )
    .unwrap();
    let task = PromptTask::new(wf, "task-1".to_string(), "c".to_string());

    queue.enqueue(task).await;
    let _ = queue.dequeue().await;

    // 中断当前任务
    let interrupted = queue.interrupt_current().await;
    assert!(interrupted.is_some());
    assert_eq!(interrupted.unwrap().prompt_id, "task-1");

    // 再次中断应该返回 None
    let none_result = queue.interrupt_current().await;
    assert!(none_result.is_none());
}

#[tokio::test]
async fn test_event_bus_subscribe_unsubscribe() {
    let bus = EventBus::new();
    assert_eq!(bus.subscriber_count().await, 0);

    let _rx1 = bus.subscribe("client-1".to_string()).await;
    assert_eq!(bus.subscriber_count().await, 1);

    let _rx2 = bus.subscribe("client-2".to_string()).await;
    assert_eq!(bus.subscriber_count().await, 2);

    bus.unsubscribe("client-1").await;
    assert_eq!(bus.subscriber_count().await, 1);
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
    let mut rx1 = bus.subscribe("client-1".to_string()).await;
    let mut rx2 = bus.subscribe("client-2".to_string()).await;

    // 仅发送给 client-2
    bus.publish_to(
        "client-2",
        Event::ExecutionStart {
            prompt_id: "private".to_string(),
        },
    )
    .await;

    // client-2 应该收到
    let event2 = rx2.recv().await.unwrap();
    match event2 {
        Event::ExecutionStart { prompt_id } => assert_eq!(prompt_id, "private"),
        _ => panic!("Expected ExecutionStart"),
    }

    // client-1 不应该收到（recv 会等待，使用 try_recv 立即检查）
    let event1 = rx1.try_recv();
    assert!(event1.is_err(), "client-1 should not receive the event");
}

#[tokio::test]
async fn test_execution_engine_submit() {
    let mut engine = ExecutionEngine::new();

    let workflow = WorkflowBuilder::text_to_image(
        "prompt".to_string(),
        "neg".to_string(),
        512,
        512,
        20,
        7.0,
        42,
        "model".to_string(),
    )
    .unwrap();

    let prompt_id = engine
        .submit(workflow, "client-1".to_string())
        .await
        .expect("Failed to submit workflow");

    assert!(!prompt_id.is_empty());
}

#[tokio::test]
async fn test_execution_engine_empty_queue() {
    let mut engine = ExecutionEngine::new();
    let result = engine.execute_next().await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[tokio::test]
async fn test_event_serialization() {
    let event = Event::ExecutionStart {
        prompt_id: "test-123".to_string(),
    };

    let json = serde_json::to_string(&event).expect("Failed to serialize event");
    assert!(json.contains("test-123"));
}

#[tokio::test]
async fn test_event_progress_serialization() {
    let event = Event::Progress {
        prompt_id: "p1".to_string(),
        value: 5,
        max: 20,
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("5"));
    assert!(json.contains("20"));
}

#[tokio::test]
async fn test_multiple_event_subscribers() {
    let bus = EventBus::new();
    let mut rx1 = bus.subscribe("c1".to_string()).await;
    let mut rx2 = bus.subscribe("c2".to_string()).await;
    let mut rx3 = bus.subscribe("c3".to_string()).await;

    bus.publish(Event::Status {
        status: SystemStatus {
            queue_remaining: 5,
            executing: Some("node-1".to_string()),
        },
    })
    .await;

    // 所有订阅者都应该收到
    for rx in [&mut rx1, &mut rx2, &mut rx3] {
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
