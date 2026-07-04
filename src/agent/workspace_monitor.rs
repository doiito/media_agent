// ComfyUI 工作区监控 - 基于 gliding_horse WorkspaceMonitor
// 监控 models/outputs/workflows 目录变化，触发技能图更新与知识积累

use std::path::PathBuf;
use std::sync::Arc;
use log::{info, warn, debug};
use serde_json::{json, Value};

use glidinghorse::tools::workspace_monitor::{
    WorkspaceMonitor, WorkspaceMonitorConfig, FileEntry,
};

/// ComfyUI 工作区监控器
///
/// 包装 gliding_horse WorkspaceMonitor，针对 ComfyUI 的目录结构定制：
/// - models/: 模型文件（checkpoint/vae/lora/controlnet 等）
/// - outputs/: 生成输出（images/videos）
/// - workflows/: 工作流模板（JSON-LD）
/// - skills/: 技能定义（JSON-LD）
pub struct ComfyUiWorkspaceMonitor {
    inner: Arc<WorkspaceMonitor>,
    /// 监控的子目录
    watch_dirs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ComfyUiWorkspaceConfig {
    /// 项目根目录
    pub project_root: PathBuf,
    /// 是否启用文件监听
    pub watch_enabled: bool,
    /// 数据库路径（None = 内存模式）
    pub db_path: Option<PathBuf>,
}

impl Default for ComfyUiWorkspaceConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            watch_enabled: true,
            db_path: None,
        }
    }
}

/// 文件变化事件
#[derive(Debug, Clone)]
pub struct FileChangeEvent {
    pub path: String,
    pub category: FileCategory,
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileCategory {
    Model,
    Output,
    Workflow,
    Skill,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
}

impl ComfyUiWorkspaceMonitor {
    /// 初始化工作区监控
    pub fn new(config: ComfyUiWorkspaceConfig) -> Result<Self, String> {
        let watch_dirs = vec![
            "models".to_string(),
            "output".to_string(),
            "outputs".to_string(),
            "workflows".to_string(),
            "skills".to_string(),
        ];

        let ws_config = WorkspaceMonitorConfig {
            workspace_root: config.project_root.clone(),
            exclude_patterns: vec![
                "target/".into(),
                ".git/".into(),
                "node_modules/".into(),
                ".gliding_horse/".into(),
                "__pycache__/".into(),
            ],
            content_store_max_bytes: 32 * 1024 * 1024,
            content_cache_capacity: 500,
            watch_enabled: config.watch_enabled,
            poll_interval_ms: 5000,
            debounce_ms: 500,
            max_debounce_wait_ms: 5000,
            db_path: config.db_path,
        };

        let inner = Arc::new(WorkspaceMonitor::initialize(ws_config, None, None)?);

        info!("ComfyUiWorkspaceMonitor initialized at {:?}", config.project_root);
        Ok(Self { inner, watch_dirs })
    }

    /// 获取内部 WorkspaceMonitor 引用
    pub fn inner(&self) -> &Arc<WorkspaceMonitor> {
        &self.inner
    }

    /// 重新扫描工作区，返回新增文件数
    pub fn rescan(&self) -> usize {
        self.inner.rescan()
    }

    /// 列出指定类别下的所有文件
    pub fn list_files(&self, category: &str) -> Vec<FileEntry> {
        let inventory = self.inner.inventory.read();
        let root = self.inner.config.workspace_root.to_string_lossy().to_string();

        let target_subdir = match category {
            "model" | "models" => vec!["models"],
            "output" | "outputs" => vec!["output", "outputs"],
            "workflow" | "workflows" => vec!["workflows"],
            "skill" | "skills" => vec!["skills"],
            _ => return Vec::new(),
        };

        let mut entries = Vec::new();
        for subdir in target_subdir {
            let prefix = format!("{}/{}", root.trim_end_matches('/'), subdir);
            for entry in inventory.list_dir(&prefix) {
                entries.push(entry);
            }
        }
        entries
    }

    /// 获取模型文件清单
    pub fn list_models(&self) -> Vec<FileEntry> {
        self.list_files("models")
    }

    /// 获取输出文件清单
    pub fn list_outputs(&self) -> Vec<FileEntry> {
        self.list_files("outputs")
    }

    /// 获取工作流模板清单
    pub fn list_workflows(&self) -> Vec<FileEntry> {
        self.list_files("workflows")
    }

    /// 获取技能定义文件清单
    pub fn list_skills(&self) -> Vec<FileEntry> {
        self.list_files("skills")
    }

    /// 获取工作区统计信息
    pub fn stats(&self) -> Value {
        let models = self.list_models();
        let outputs = self.list_outputs();
        let workflows = self.list_workflows();
        let skills = self.list_skills();

        json!({
            "models": {
                "count": models.len(),
                "total_size": models.iter().map(|e| e.file_size).sum::<u64>(),
            },
            "outputs": {
                "count": outputs.len(),
                "total_size": outputs.iter().map(|e| e.file_size).sum::<u64>(),
            },
            "workflows": {
                "count": workflows.len(),
            },
            "skills": {
                "count": skills.len(),
            },
            "watch_dirs": self.watch_dirs,
        })
    }

    /// 标记文件已被 agent 读取
    pub fn mark_read(&self, path: &str) {
        self.inner.mark_file_read_external(path);
    }

    /// 标记文件已被 agent 写入
    pub fn mark_written(&self, path: &str) {
        self.inner.mark_file_written(path);
    }

    /// 根据文件路径推断文件类别
    pub fn categorize(path: &str) -> FileCategory {
        if path.contains("/models/") || path.starts_with("models/") {
            FileCategory::Model
        } else if path.contains("/output") || path.starts_with("output") {
            FileCategory::Output
        } else if path.contains("/workflows/") || path.starts_with("workflows/") {
            FileCategory::Workflow
        } else if path.contains("/skills/") || path.starts_with("skills/") {
            FileCategory::Skill
        } else {
            FileCategory::Other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorize() {
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("models/sd15.safetensors"), FileCategory::Model);
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("output/img_001.png"), FileCategory::Output);
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("outputs/video.mp4"), FileCategory::Output);
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("workflows/t2i.jsonld"), FileCategory::Workflow);
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("skills/text_to_image.jsonld"), FileCategory::Skill);
        assert_eq!(ComfyUiWorkspaceMonitor::categorize("README.md"), FileCategory::Other);
    }

    #[test]
    fn test_workspace_monitor_init() {
        let temp = std::env::temp_dir().join(format!("comfyui-ws-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp).unwrap();
        std::fs::create_dir_all(temp.join("models")).unwrap();
        std::fs::create_dir_all(temp.join("output")).unwrap();
        std::fs::write(temp.join("models/test.safetensors"), b"fake model").unwrap();

        let config = ComfyUiWorkspaceConfig {
            project_root: temp.clone(),
            watch_enabled: false,
            db_path: None,
        };

        let monitor = ComfyUiWorkspaceMonitor::new(config);
        assert!(monitor.is_ok(), "WorkspaceMonitor init failed: {:?}", monitor.err());

        let monitor = monitor.unwrap();
        let models = monitor.list_models();
        assert!(!models.is_empty(), "Should find at least one model file");

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_stats() {
        let temp = std::env::temp_dir().join(format!("comfyui-ws-stats-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp).unwrap();
        std::fs::create_dir_all(temp.join("workflows")).unwrap();
        std::fs::write(temp.join("workflows/test.jsonld"), b"{}").unwrap();

        let config = ComfyUiWorkspaceConfig {
            project_root: temp.clone(),
            watch_enabled: false,
            db_path: None,
        };

        let monitor = ComfyUiWorkspaceMonitor::new(config).unwrap();
        let stats = monitor.stats();
        assert_eq!(stats["workflows"]["count"], 1);

        std::fs::remove_dir_all(&temp).ok();
    }
}
