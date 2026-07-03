// 实时预览管理器
// 管理工作流执行过程中的实时预览，包括采样进度、中间结果推送

use crate::execution::event::{Event, EventBus};
use log::{info, debug, warn};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 预览配置
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    /// 是否启用预览
    pub enabled: bool,
    /// 预览推送间隔（步数），每 N 步推送一次
    pub step_interval: usize,
    /// 预览图最大宽度
    pub max_width: usize,
    /// 预览图最大高度
    pub max_height: usize,
    /// 预览图质量（1-100，JPEG）
    pub jpeg_quality: u8,
    /// 是否在执行完成后推送最终预览
    pub send_final_preview: bool,
    /// 预览图缓存大小（每个会话最多缓存的预览数）
    pub cache_size: usize,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            step_interval: 5,
            max_width: 512,
            max_height: 512,
            jpeg_quality: 85,
            send_final_preview: true,
            cache_size: 10,
        }
    }
}

/// 预览帧
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    /// 步数
    pub step: usize,
    /// 总步数
    pub total_steps: usize,
    /// 进度百分比 (0-100)
    pub progress: f32,
    /// 图像数据（JPEG/PNG 编码）
    pub data: Vec<u8>,
    /// 时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 节点 ID
    pub node_id: String,
}

impl PreviewFrame {
    pub fn new(step: usize, total_steps: usize, data: Vec<u8>, node_id: String) -> Self {
        let progress = if total_steps > 0 {
            (step as f32 / total_steps as f32) * 100.0
        } else {
            0.0
        };
        Self {
            step,
            total_steps,
            progress,
            data,
            timestamp: chrono::Utc::now(),
            node_id,
        }
    }
}

/// 预览会话
#[derive(Debug)]
pub struct PreviewSession {
    /// Prompt ID
    pub prompt_id: String,
    /// 客户端 ID
    pub client_id: String,
    /// 当前节点 ID
    pub current_node: Option<String>,
    /// 当前步数
    pub current_step: usize,
    /// 总步数
    pub total_steps: usize,
    /// 预览帧缓存
    pub frames: Vec<PreviewFrame>,
    /// 会话开始时间
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// 是否已完成
    pub completed: bool,
}

impl PreviewSession {
    pub fn new(prompt_id: String, client_id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            prompt_id,
            client_id,
            current_node: None,
            current_step: 0,
            total_steps: 0,
            frames: Vec::new(),
            started_at: now,
            last_updated: now,
            completed: false,
        }
    }

    /// 添加预览帧
    pub fn add_frame(&mut self, frame: PreviewFrame, max_cache: usize) {
        self.current_step = frame.step;
        self.total_steps = frame.total_steps;
        self.current_node = Some(frame.node_id.clone());
        self.last_updated = chrono::Utc::now();

        self.frames.push(frame);
        // 限制缓存大小
        if self.frames.len() > max_cache {
            let excess = self.frames.len() - max_cache;
            self.frames.drain(0..excess);
        }
    }

    /// 获取最新预览帧
    pub fn latest_frame(&self) -> Option<&PreviewFrame> {
        self.frames.last()
    }

    /// 获取当前进度百分比
    pub fn progress(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.current_step as f32 / self.total_steps as f32) * 100.0
        }
    }

    /// 获取会话持续时间（毫秒）
    pub fn elapsed_ms(&self) -> i64 {
        self.last_updated.signed_duration_since(self.started_at).num_milliseconds()
    }
}

/// 实时预览管理器
pub struct PreviewManager {
    /// 预览配置
    config: PreviewConfig,
    /// 事件总线
    event_bus: EventBus,
    /// 活跃预览会话
    sessions: Arc<RwLock<HashMap<String, PreviewSession>>>,
}

impl PreviewManager {
    /// 创建新的预览管理器
    pub fn new(event_bus: EventBus, config: PreviewConfig) -> Self {
        Self {
            config,
            event_bus,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 使用默认配置创建
    pub fn with_default(event_bus: EventBus) -> Self {
        Self::new(event_bus, PreviewConfig::default())
    }

    /// 开始预览会话
    pub async fn start_session(&self, prompt_id: &str, client_id: &str) {
        if !self.config.enabled {
            return;
        }

        let session = PreviewSession::new(prompt_id.to_string(), client_id.to_string());
        self.sessions.write().await.insert(prompt_id.to_string(), session);

        // 发布执行开始事件
        self.event_bus.publish(Event::ExecutionStart {
            prompt_id: prompt_id.to_string(),
        }).await;

        info!("预览会话开始: prompt_id={}", prompt_id);
    }

    /// 更新采样进度
    pub async fn update_progress(&self, prompt_id: &str, step: usize, max: usize) {
        if !self.config.enabled {
            return;
        }

        let should_publish = {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(prompt_id) {
                // 按步数间隔过滤
                step % self.config.step_interval == 0 || step == max
            } else {
                false
            }
        };

        if should_publish {
            self.event_bus.publish(Event::Progress {
                prompt_id: prompt_id.to_string(),
                value: step,
                max,
            }).await;
            debug!("进度更新: prompt_id={}, step={}/{}", prompt_id, step, max);
        }

        // 更新会话状态
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(prompt_id) {
            session.current_step = step;
            session.total_steps = max;
            session.last_updated = chrono::Utc::now();
        }
    }

    /// 推送预览帧
    pub async fn push_preview(
        &self,
        prompt_id: &str,
        node_id: &str,
        step: usize,
        total_steps: usize,
        data: Vec<u8>,
    ) {
        if !self.config.enabled || data.is_empty() {
            return;
        }

        let frame = PreviewFrame::new(step, total_steps, data.clone(), node_id.to_string());

        // 缓存帧
        {
            let mut sessions = self.sessions.write().await;
            let session = sessions.entry(prompt_id.to_string())
                .or_insert_with(|| PreviewSession::new(prompt_id.to_string(), String::new()));
            session.add_frame(frame, self.config.cache_size);
        }

        // 发布预览事件
        self.event_bus.publish(Event::Preview {
            prompt_id: prompt_id.to_string(),
            node_id: node_id.to_string(),
            data,
        }).await;

        debug!("推送预览: prompt_id={}, node={}, step={}/{}", prompt_id, node_id, step, total_steps);
    }

    /// 推送节点执行状态
    pub async fn executing_node(&self, prompt_id: &str, node_id: Option<&str>) {
        if !self.config.enabled {
            return;
        }

        self.event_bus.publish(Event::Executing {
            prompt_id: prompt_id.to_string(),
            node_id: node_id.map(|s| s.to_string()),
        }).await;

        if let Some(nid) = node_id {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(prompt_id) {
                session.current_node = Some(nid.to_string());
            }
        }
    }

    /// 完成预览会话
    pub async fn complete_session(&self, prompt_id: &str, outputs: HashMap<String, HashMap<String, crate::types::Value>>) {
        if !self.config.enabled {
            return;
        }

        // 发布执行完成事件
        let outputs_mapped: HashMap<crate::types::NodeId, HashMap<String, crate::types::Value>> =
            outputs.into_iter().map(|(k, v)| (k, v)).collect();

        self.event_bus.publish(Event::ExecutionSuccess {
            prompt_id: prompt_id.to_string(),
            outputs: outputs_mapped,
        }).await;

        // 更新会话状态
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(prompt_id) {
            session.completed = true;
            session.last_updated = chrono::Utc::now();
        }

        info!("预览会话完成: prompt_id={}", prompt_id);
    }

    /// 报告执行错误
    pub async fn error_session(&self, prompt_id: &str, error: &str) {
        self.event_bus.publish(Event::ExecutionError {
            prompt_id: prompt_id.to_string(),
            error: error.to_string(),
        }).await;

        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(prompt_id) {
            session.last_updated = chrono::Utc::now();
        }

        warn!("预览会话错误: prompt_id={}, error={}", prompt_id, error);
    }

    /// 中断预览会话
    pub async fn interrupt_session(&self, prompt_id: &str) {
        self.event_bus.publish(Event::ExecutionInterrupted {
            prompt_id: prompt_id.to_string(),
        }).await;

        let mut sessions = self.sessions.write().await;
        sessions.remove(prompt_id);

        info!("预览会话中断: prompt_id={}", prompt_id);
    }

    /// 获取会话信息
    pub async fn get_session(&self, prompt_id: &str) -> Option<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.get(prompt_id).map(|s| SessionInfo {
            prompt_id: s.prompt_id.clone(),
            client_id: s.client_id.clone(),
            current_node: s.current_node.clone(),
            current_step: s.current_step,
            total_steps: s.total_steps,
            progress: s.progress(),
            frame_count: s.frames.len(),
            started_at: s.started_at,
            elapsed_ms: s.elapsed_ms(),
            completed: s.completed,
        })
    }

    /// 获取会话的最新预览帧
    pub async fn get_latest_frame(&self, prompt_id: &str) -> Option<PreviewFrame> {
        let sessions = self.sessions.read().await;
        sessions.get(prompt_id).and_then(|s| s.latest_frame().cloned())
    }

    /// 获取会话的所有预览帧
    pub async fn get_frames(&self, prompt_id: &str) -> Vec<PreviewFrame> {
        let sessions = self.sessions.read().await;
        sessions.get(prompt_id)
            .map(|s| s.frames.clone())
            .unwrap_or_default()
    }

    /// 获取所有活跃会话
    pub async fn active_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.values().map(|s| SessionInfo {
            prompt_id: s.prompt_id.clone(),
            client_id: s.client_id.clone(),
            current_node: s.current_node.clone(),
            current_step: s.current_step,
            total_steps: s.total_steps,
            progress: s.progress(),
            frame_count: s.frames.len(),
            started_at: s.started_at,
            elapsed_ms: s.elapsed_ms(),
            completed: s.completed,
        }).collect()
    }

    /// 清理已完成的会话
    pub async fn cleanup_completed(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();
        sessions.retain(|_, s| !s.completed);
        let cleaned = before - sessions.len();
        if cleaned > 0 {
            info!("清理 {} 个已完成预览会话", cleaned);
        }
        cleaned
    }

    /// 清理所有会话
    pub async fn clear_all(&self) {
        let mut sessions = self.sessions.write().await;
        let count = sessions.len();
        sessions.clear();
        info!("清空所有预览会话: {} 个", count);
    }

    /// 获取配置
    pub fn config(&self) -> &PreviewConfig {
        &self.config
    }

    /// 获取统计信息
    pub async fn stats(&self) -> PreviewStats {
        let sessions = self.sessions.read().await;
        let active = sessions.values().filter(|s| !s.completed).count();
        let completed = sessions.values().filter(|s| s.completed).count();
        let total_frames: usize = sessions.values().map(|s| s.frames.len()).sum();

        PreviewStats {
            total_sessions: sessions.len(),
            active_sessions: active,
            completed_sessions: completed,
            total_frames,
        }
    }
}

/// 会话信息（用于查询）
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub prompt_id: String,
    pub client_id: String,
    pub current_node: Option<String>,
    pub current_step: usize,
    pub total_steps: usize,
    pub progress: f32,
    pub frame_count: usize,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub elapsed_ms: i64,
    pub completed: bool,
}

/// 预览统计
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PreviewStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub completed_sessions: usize,
    pub total_frames: usize,
}

/// 采样进度回调
/// 在采样循环中调用，用于推送实时预览
pub struct ProgressCallback {
    prompt_id: String,
    node_id: String,
    step_interval: usize,
    total_steps: usize,
    preview_manager: Arc<PreviewManager>,
    last_pushed_step: std::sync::atomic::AtomicUsize,
}

impl ProgressCallback {
    pub fn new(
        prompt_id: String,
        node_id: String,
        total_steps: usize,
        preview_manager: Arc<PreviewManager>,
    ) -> Self {
        let step_interval = preview_manager.config().step_interval;
        Self {
            prompt_id,
            node_id,
            step_interval,
            total_steps,
            preview_manager,
            last_pushed_step: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 在每步采样后调用
    pub async fn on_step(&self, step: usize) {
        self.preview_manager
            .update_progress(&self.prompt_id, step, self.total_steps)
            .await;

        // 检查是否需要推送预览
        if step % self.step_interval == 0 || step == self.total_steps {
            let last = self.last_pushed_step.load(std::sync::atomic::Ordering::Relaxed);
            if step > last {
                self.last_pushed_step.store(step, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// 推送预览帧
    pub async fn push_preview(&self, step: usize, data: Vec<u8>) {
        self.preview_manager
            .push_preview(
                &self.prompt_id,
                &self.node_id,
                step,
                self.total_steps,
                data,
            )
            .await;
    }

    /// 采样完成
    pub async fn on_complete(&self) {
        self.preview_manager
            .update_progress(&self.prompt_id, self.total_steps, self.total_steps)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::event::EventBus;

    fn make_test_manager() -> (EventBus, PreviewManager) {
        let bus = EventBus::new();
        let manager = PreviewManager::with_default(bus.clone());
        (bus, manager)
    }

    #[tokio::test]
    async fn test_start_session() {
        let (_, manager) = make_test_manager();
        manager.start_session("prompt-1", "client-1").await;

        let sessions = manager.active_sessions().await;
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].prompt_id, "prompt-1");
    }

    #[tokio::test]
    async fn test_update_progress() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        // 收取 ExecutionStart 事件
        let _ = rx.recv().await.unwrap();

        // step 5 应该推送（interval=5）
        manager.update_progress("prompt-1", 5, 20).await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::Progress { value, max, .. } => {
                assert_eq!(value, 5);
                assert_eq!(max, 20);
            }
            _ => panic!("Expected Progress event"),
        }
    }

    #[tokio::test]
    async fn test_update_progress_filtered() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap(); // ExecutionStart

        // step 3 不应该推送（不是5的倍数）
        manager.update_progress("prompt-1", 3, 20).await;

        // 应该没有 Progress 事件
        let result = rx.try_recv();
        assert!(result.is_err(), "step 3 不应推送进度事件");
    }

    #[tokio::test]
    async fn test_push_preview() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap(); // ExecutionStart

        let preview_data = vec![0u8, 1, 2, 3, 4];
        manager.push_preview("prompt-1", "node-1", 5, 20, preview_data.clone()).await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::Preview { node_id, data, .. } => {
                assert_eq!(node_id, "node-1");
                assert_eq!(data, preview_data);
            }
            _ => panic!("Expected Preview event"),
        }

        // 验证帧缓存
        let frames = manager.get_frames("prompt-1").await;
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].step, 5);
        assert_eq!(frames[0].total_steps, 20);
        assert!((frames[0].progress - 25.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_executing_node() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap(); // ExecutionStart

        manager.executing_node("prompt-1", Some("KSampler")).await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::Executing { node_id, .. } => {
                assert_eq!(node_id, Some("KSampler".to_string()));
            }
            _ => panic!("Expected Executing event"),
        }
    }

    #[tokio::test]
    async fn test_complete_session() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap(); // ExecutionStart

        let outputs = HashMap::new();
        manager.complete_session("prompt-1", outputs).await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::ExecutionSuccess { .. } => {}
            _ => panic!("Expected ExecutionSuccess event"),
        }

        let session = manager.get_session("prompt-1").await;
        assert!(session.is_some());
        assert!(session.unwrap().completed);
    }

    #[tokio::test]
    async fn test_error_session() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap();

        manager.error_session("prompt-1", "OOM").await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::ExecutionError { error, .. } => {
                assert_eq!(error, "OOM");
            }
            _ => panic!("Expected ExecutionError event"),
        }
    }

    #[tokio::test]
    async fn test_interrupt_session() {
        let (bus, manager) = make_test_manager();
        let mut rx = bus.subscribe("test-client".to_string()).await;

        manager.start_session("prompt-1", "client-1").await;
        let _ = rx.recv().await.unwrap();

        manager.interrupt_session("prompt-1").await;

        let event = rx.recv().await.unwrap();
        match event {
            Event::ExecutionInterrupted { .. } => {}
            _ => panic!("Expected ExecutionInterrupted event"),
        }

        // 会话应被移除
        let session = manager.get_session("prompt-1").await;
        assert!(session.is_none());
    }

    #[tokio::test]
    async fn test_session_progress() {
        let (_, manager) = make_test_manager();
        manager.start_session("prompt-1", "client-1").await;

        manager.update_progress("prompt-1", 10, 20).await;

        let session = manager.get_session("prompt-1").await.unwrap();
        assert_eq!(session.current_step, 10);
        assert_eq!(session.total_steps, 20);
        assert!((session.progress - 50.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_frame_cache_limit() {
        let config = PreviewConfig {
            cache_size: 3,
            ..Default::default()
        };
        let bus = EventBus::new();
        let manager = PreviewManager::new(bus, config);

        manager.start_session("prompt-1", "client-1").await;

        // 推送5个帧
        for i in 1..=5 {
            manager.push_preview("prompt-1", "node-1", i, 10, vec![i as u8]).await;
        }

        let frames = manager.get_frames("prompt-1").await;
        assert_eq!(frames.len(), 3); // 只保留最后3个
        assert_eq!(frames[0].step, 3);
        assert_eq!(frames[2].step, 5);
    }

    #[tokio::test]
    async fn test_cleanup_completed() {
        let (_, manager) = make_test_manager();

        manager.start_session("prompt-1", "client-1").await;
        manager.start_session("prompt-2", "client-2").await;

        manager.complete_session("prompt-1", HashMap::new()).await;

        assert_eq!(manager.active_sessions().await.len(), 2);

        let cleaned = manager.cleanup_completed().await;
        assert_eq!(cleaned, 1);
        assert_eq!(manager.active_sessions().await.len(), 1);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let (_, manager) = make_test_manager();

        manager.start_session("prompt-1", "client-1").await;
        manager.start_session("prompt-2", "client-2").await;

        manager.clear_all().await;
        assert_eq!(manager.active_sessions().await.len(), 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let (_, manager) = make_test_manager();

        manager.start_session("prompt-1", "client-1").await;
        manager.start_session("prompt-2", "client-2").await;

        manager.push_preview("prompt-1", "node-1", 1, 10, vec![0u8]).await;
        manager.push_preview("prompt-1", "node-1", 2, 10, vec![1u8]).await;
        manager.complete_session("prompt-1", HashMap::new()).await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.active_sessions, 1);
        assert_eq!(stats.completed_sessions, 1);
        assert_eq!(stats.total_frames, 2);
    }

    #[tokio::test]
    async fn test_progress_callback() {
        let (_, manager) = make_test_manager();
        let manager = Arc::new(manager);

        manager.start_session("prompt-1", "client-1").await;

        let callback = ProgressCallback::new(
            "prompt-1".to_string(),
            "KSampler".to_string(),
            20,
            manager.clone(),
        );

        // 模拟采样步骤
        for step in 1..=20 {
            callback.on_step(step).await;
        }

        callback.on_complete().await;

        let session = manager.get_session("prompt-1").await.unwrap();
        assert_eq!(session.current_step, 20);
        assert_eq!(session.total_steps, 20);
        assert!((session.progress - 100.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_disabled_preview() {
        let config = PreviewConfig {
            enabled: false,
            ..Default::default()
        };
        let bus = EventBus::new();
        let manager = PreviewManager::new(bus, config);

        manager.start_session("prompt-1", "client-1").await;

        // 禁用后不应该有会话
        assert_eq!(manager.active_sessions().await.len(), 0);
    }

    #[test]
    fn test_preview_frame() {
        let frame = PreviewFrame::new(5, 20, vec![0u8; 100], "node-1".to_string());
        assert_eq!(frame.step, 5);
        assert_eq!(frame.total_steps, 20);
        assert!((frame.progress - 25.0).abs() < 0.1);
        assert_eq!(frame.node_id, "node-1");
    }

    #[test]
    fn test_preview_config_default() {
        let config = PreviewConfig::default();
        assert!(config.enabled);
        assert_eq!(config.step_interval, 5);
        assert_eq!(config.max_width, 512);
        assert_eq!(config.jpeg_quality, 85);
        assert!(config.send_final_preview);
    }
}
