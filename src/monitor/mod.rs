// 系统监控模块
// 采集系统资源指标（CPU/内存/磁盘）、进程状态、性能数据
// 提供实时监控和历史数据查询

use crate::config::MonitorConfig;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Disks, System};
use tokio::sync::RwLock;
use log::{info, warn, debug};

/// 单次采样的系统指标快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// 采样时间戳（Unix 秒）
    pub timestamp: u64,

    /// 整体 CPU 使用率（百分比，0-100）
    pub cpu_usage: f32,

    /// 每核心 CPU 使用率
    pub per_cpu_usage: Vec<f32>,

    /// 内存总量（字节）
    pub total_memory: u64,

    /// 已用内存（字节）
    pub used_memory: u64,

    /// 内存使用率（百分比）
    pub memory_usage: f32,

    /// 交换分区总量（字节）
    pub total_swap: u64,

    /// 已用交换分区（字节）
    pub used_swap: u64,

    /// 磁盘使用情况
    pub disks: Vec<DiskInfo>,

    /// 进程数
    pub process_count: usize,

    /// 系统负载（仅 Linux，1/5/15 分钟）
    pub load_avg: Option<[f64; 3]>,

    /// 系统运行时间（秒）
    pub uptime_secs: u64,
}

/// 磁盘信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    /// 挂载点
    pub mount_point: String,
    /// 总空间（字节）
    pub total_space: u64,
    /// 已用空间（字节）
    pub used_space: u64,
    /// 使用率（百分比）
    pub usage: f32,
}

/// 进程指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    /// 进程 ID
    pub pid: u32,
    /// 进程名称
    pub name: String,
    /// CPU 使用率（百分比）
    pub cpu_usage: f32,
    /// 内存使用（字节）
    pub memory: u64,
    /// 启动时间戳（Unix 秒）
    pub start_time: u64,
    /// 进程状态
    pub status: String,
}

/// 告警事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// 告警 ID
    pub id: String,
    /// 告警级别
    pub level: AlertLevel,
    /// 告警类型
    pub kind: AlertKind,
    /// 告警消息
    pub message: String,
    /// 触发时间戳
    pub timestamp: u64,
    /// 当前值
    pub current_value: f32,
    /// 阈值
    pub threshold: f32,
}

/// 告警级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    /// 信息
    Info,
    /// 警告
    Warning,
    /// 严重
    Critical,
}

/// 告警类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertKind {
    /// CPU 使用率过高
    HighCpuUsage,
    /// 内存使用率过高
    HighMemoryUsage,
    /// 磁盘空间不足
    LowDiskSpace,
    /// 后端进程崩溃
    BackendProcessCrash,
    /// 自定义
    Custom,
}

/// 监控系统
pub struct Monitor {
    /// 配置
    config: MonitorConfig,
    /// 历史采样数据
    history: Arc<RwLock<VecDeque<MetricsSnapshot>>>,
    /// sysinfo 系统实例
    system: Arc<RwLock<System>>,
    /// 告警历史
    alerts: Arc<RwLock<VecDeque<Alert>>>,
    /// 监控启动时间
    start_time: Instant,
    /// 是否正在运行
    running: Arc<RwLock<bool>>,
}

impl Monitor {
    /// 创建新的监控器
    pub fn new(config: MonitorConfig) -> Self {
        let history_capacity = config.history_size;
        let history = VecDeque::with_capacity(history_capacity);
        let alerts = VecDeque::with_capacity(100);

        Self {
            config,
            history: Arc::new(RwLock::new(history)),
            system: Arc::new(RwLock::new(System::new())),
            alerts: Arc::new(RwLock::new(alerts)),
            start_time: Instant::now(),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults() -> Self {
        Self::new(MonitorConfig::default())
    }

    /// 采集一次指标快照
    pub async fn collect(&self) -> MetricsSnapshot {
        let mut sys = self.system.write().await;
        // 刷新 CPU、内存、进程信息
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        // sysinfo 0.32+ 需要显式刷新进程
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

        // CPU 需要两次刷新才能获取使用率
        let per_cpu_usage: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
        let cpu_usage = if per_cpu_usage.is_empty() {
            0.0
        } else {
            per_cpu_usage.iter().sum::<f32>() / per_cpu_usage.len() as f32
        };

        let total_memory = sys.total_memory();
        let used_memory = sys.used_memory();
        let memory_usage = if total_memory > 0 {
            (used_memory as f32 / total_memory as f32) * 100.0
        } else {
            0.0
        };

        let total_swap = sys.total_swap();
        let used_swap = sys.used_swap();

        // 磁盘信息（sysinfo 0.32+ 中 Disks 已从 System 分离）
        let disks_list = Disks::new_with_refreshed_list();
        let disks: Vec<DiskInfo> = disks_list
            .iter()
            .map(|d| {
                let total = d.total_space();
                let available = d.available_space();
                let used = total.saturating_sub(available);
                let usage = if total > 0 {
                    (used as f32 / total as f32) * 100.0
                } else {
                    0.0
                };
                DiskInfo {
                    mount_point: d.mount_point().to_string_lossy().into_owned(),
                    total_space: total,
                    used_space: used,
                    usage,
                }
            })
            .collect();

        // 系统负载（仅 Unix）
        #[cfg(unix)]
        let load_avg = {
            let loads = System::load_average();
            Some([loads.one, loads.five, loads.fifteen])
        };
        #[cfg(not(unix))]
        let load_avg = None;

        let snapshot = MetricsSnapshot {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            cpu_usage,
            per_cpu_usage,
            total_memory,
            used_memory,
            memory_usage,
            total_swap,
            used_swap,
            disks,
            process_count: sys.processes().len(),
            load_avg,
            uptime_secs: self.start_time.elapsed().as_secs(),
        };

        // 保存到历史
        {
            let mut history = self.history.write().await;
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(snapshot.clone());
        }

        // 检查告警
        self.check_alerts(&snapshot).await;

        snapshot
    }

    /// 检查告警条件
    async fn check_alerts(&self, snapshot: &MetricsSnapshot) {
        let mut new_alerts = Vec::new();

        if snapshot.cpu_usage > self.config.cpu_alert_threshold {
            new_alerts.push(Alert {
                id: format!("cpu-{}", snapshot.timestamp),
                level: if snapshot.cpu_usage > 98.0 {
                    AlertLevel::Critical
                } else {
                    AlertLevel::Warning
                },
                kind: AlertKind::HighCpuUsage,
                message: format!(
                    "CPU usage is {:.1}% (threshold: {:.1}%)",
                    snapshot.cpu_usage, self.config.cpu_alert_threshold
                ),
                timestamp: snapshot.timestamp,
                current_value: snapshot.cpu_usage,
                threshold: self.config.cpu_alert_threshold,
            });
        }

        if snapshot.memory_usage > self.config.mem_alert_threshold {
            new_alerts.push(Alert {
                id: format!("mem-{}", snapshot.timestamp),
                level: if snapshot.memory_usage > 95.0 {
                    AlertLevel::Critical
                } else {
                    AlertLevel::Warning
                },
                kind: AlertKind::HighMemoryUsage,
                message: format!(
                    "Memory usage is {:.1}% (threshold: {:.1}%)",
                    snapshot.memory_usage, self.config.mem_alert_threshold
                ),
                timestamp: snapshot.timestamp,
                current_value: snapshot.memory_usage,
                threshold: self.config.mem_alert_threshold,
            });
        }

        // 磁盘空间告警
        for disk in &snapshot.disks {
            if disk.usage > 90.0 {
                new_alerts.push(Alert {
                    id: format!("disk-{}-{}", disk.mount_point, snapshot.timestamp),
                    level: if disk.usage > 98.0 {
                        AlertLevel::Critical
                    } else {
                        AlertLevel::Warning
                    },
                    kind: AlertKind::LowDiskSpace,
                    message: format!(
                        "Disk {} is {:.1}% full ({} of {} bytes used)",
                        disk.mount_point,
                        disk.usage,
                        disk.used_space,
                        disk.total_space
                    ),
                    timestamp: snapshot.timestamp,
                    current_value: disk.usage,
                    threshold: 90.0,
                });
            }
        }

        if !new_alerts.is_empty() {
            let mut alerts = self.alerts.write().await;
            for alert in new_alerts {
                match alert.level {
                    AlertLevel::Critical => warn!("[ALERT/CRITICAL] {}", alert.message),
                    AlertLevel::Warning => warn!("[ALERT/WARNING] {}", alert.message),
                    AlertLevel::Info => info!("[ALERT/INFO] {}", alert.message),
                }
                if alerts.len() >= 100 {
                    alerts.pop_front();
                }
                alerts.push_back(alert);
            }
        }
    }

    /// 获取最新一次快照
    pub async fn latest(&self) -> Option<MetricsSnapshot> {
        let history = self.history.read().await;
        history.back().cloned()
    }

    /// 获取历史采样数据
    pub async fn history(&self, limit: Option<usize>) -> Vec<MetricsSnapshot> {
        let history = self.history.read().await;
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.iter().cloned().collect(),
        }
    }

    /// 获取所有告警
    pub async fn alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.read().await;
        alerts.iter().cloned().collect()
    }

    /// 清除告警
    pub async fn clear_alerts(&self) {
        let mut alerts = self.alerts.write().await;
        alerts.clear();
    }

    /// 获取指定 PID 的进程指标
    pub async fn process_metrics(&self, pid: u32) -> Option<ProcessMetrics> {
        let sys = self.system.read().await;
        let pid = sysinfo::Pid::from_u32(pid);
        if let Some(process) = sys.process(pid) {
            Some(ProcessMetrics {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().into_owned(),
                cpu_usage: process.cpu_usage(),
                memory: process.memory(),
                start_time: process.start_time(),
                status: format!("{:?}", process.status()),
            })
        } else {
            None
        }
    }

    /// 获取所有 ComfyUI 相关进程（sd-cli、llama、comfyui-server）
    pub async fn related_processes(&self) -> Vec<ProcessMetrics> {
        let sys = self.system.read().await;
        let mut result = Vec::new();

        for (pid, process) in sys.processes() {
            let name = process.name().to_string_lossy().into_owned();
            let is_related = name.contains("sd-cli")
                || name.contains("llama")
                || name.contains("comfyui")
                || name.contains("stable-diffusion");

            if is_related {
                result.push(ProcessMetrics {
                    pid: pid.as_u32(),
                    name,
                    cpu_usage: process.cpu_usage(),
                    memory: process.memory(),
                    start_time: process.start_time(),
                    status: format!("{:?}", process.status()),
                });
            }
        }

        result
    }

    /// 启动后台采集任务
    pub async fn start(self: Arc<Self>) {
        if *self.running.read().await {
            warn!("Monitor is already running");
            return;
        }

        *self.running.write().await = true;
        let interval_secs = self.config.collect_interval_secs;
        let monitor = self.clone();

        tokio::spawn(async move {
            info!(
                "Monitor started (collect every {}s)",
                interval_secs
            );

            // 首次采集需要预热 CPU 数据
            {
                let mut sys = monitor.system.write().await;
                sys.refresh_cpu_usage();
            }

            // 等待一个采集周期，让 CPU 使用率有数据
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;

            while *monitor.running.read().await {
                let snapshot = monitor.collect().await;
                debug!(
                    "Metrics: CPU={:.1}%, MEM={:.1}%, processes={}",
                    snapshot.cpu_usage,
                    snapshot.memory_usage,
                    snapshot.process_count
                );

                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            }

            info!("Monitor stopped");
        });
    }

    /// 停止采集
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    /// 是否正在运行
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// 获取平均指标（基于历史数据）
    pub async fn averages(&self) -> Option<MetricsAverage> {
        let history = self.history.read().await;
        if history.is_empty() {
            return None;
        }

        let count = history.len() as f32;
        let sum_cpu: f32 = history.iter().map(|s| s.cpu_usage).sum();
        let sum_mem: f32 = history.iter().map(|s| s.memory_usage).sum();
        let sum_proc: u64 = history.iter().map(|s| s.process_count as u64).sum();

        Some(MetricsAverage {
            sample_count: history.len(),
            avg_cpu_usage: sum_cpu / count,
            avg_memory_usage: sum_mem / count,
            avg_process_count: sum_proc as usize / history.len(),
            max_cpu_usage: history.iter().map(|s| s.cpu_usage).fold(0.0f32, f32::max),
            max_memory_usage: history
                .iter()
                .map(|s| s.memory_usage)
                .fold(0.0f32, f32::max),
        })
    }
}

/// 平均指标汇总
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAverage {
    pub sample_count: usize,
    pub avg_cpu_usage: f32,
    pub avg_memory_usage: f32,
    pub avg_process_count: usize,
    pub max_cpu_usage: f32,
    pub max_memory_usage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_collect() {
        let monitor = Monitor::with_defaults();
        let snapshot = monitor.collect().await;

        assert!(snapshot.timestamp > 0);
        assert!(snapshot.total_memory > 0);
        assert!(snapshot.process_count > 0);
    }

    #[tokio::test]
    async fn test_monitor_history() {
        let monitor = Monitor::with_defaults();

        // 采集多次
        monitor.collect().await;
        monitor.collect().await;
        monitor.collect().await;

        let history = monitor.history(None).await;
        assert_eq!(history.len(), 3);

        let limited = monitor.history(Some(2)).await;
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_monitor_latest() {
        let monitor = Monitor::with_defaults();
        assert!(monitor.latest().await.is_none());

        monitor.collect().await;
        assert!(monitor.latest().await.is_some());
    }

    #[tokio::test]
    async fn test_alerts() {
        let mut config = MonitorConfig::default();
        config.cpu_alert_threshold = 0.0; // 设置为 0，必然触发
        config.mem_alert_threshold = 0.0;

        let monitor = Monitor::new(config);
        monitor.collect().await;

        let alerts = monitor.alerts().await;
        assert!(!alerts.is_empty());
    }

    #[tokio::test]
    async fn test_clear_alerts() {
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
    async fn test_averages() {
        let monitor = Monitor::with_defaults();
        monitor.collect().await;
        monitor.collect().await;

        let avg = monitor.averages().await;
        assert!(avg.is_some());
        let avg = avg.unwrap();
        assert_eq!(avg.sample_count, 2);
    }

    #[tokio::test]
    async fn test_related_processes() {
        let monitor = Monitor::with_defaults();
        let processes = monitor.related_processes().await;
        // 测试环境可能没有相关进程，但不应 panic
        let _ = processes.len();
    }

    #[test]
    fn test_metrics_snapshot_serialization() {
        let snapshot = MetricsSnapshot {
            timestamp: 1000,
            cpu_usage: 50.0,
            per_cpu_usage: vec![50.0, 60.0],
            total_memory: 1024 * 1024 * 1024,
            used_memory: 512 * 1024 * 1024,
            memory_usage: 50.0,
            total_swap: 0,
            used_swap: 0,
            disks: vec![DiskInfo {
                mount_point: "/".to_string(),
                total_space: 100 * 1024 * 1024 * 1024,
                used_space: 50 * 1024 * 1024 * 1024,
                usage: 50.0,
            }],
            process_count: 100,
            load_avg: Some([0.5, 0.4, 0.3]),
            uptime_secs: 3600,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.timestamp, 1000);
        assert_eq!(parsed.cpu_usage, 50.0);
        assert_eq!(parsed.disks.len(), 1);
    }
}
