// AgentMemory - 4层记忆系统初始化
// 基于 gliding_horse 的 L0Store + Blackboard + ProjectionEngine + MemoryManager

use std::sync::Arc;
use std::path::Path;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Agent 4层记忆系统
///
/// L0: redb 持久化 KV 存储（任务历史、检查点）
/// L1: Session 会话记忆（运行时 HashMap）
/// L2: Blackboard Oxigraph RDF 知识图谱（生成历史、风格偏好）
/// L3: ProjectionEngine SPARQL 投影缓存
///
/// 使用场景：
/// - L0: 跨会话持久化（用户偏好、常用模型、历史任务）
/// - L1: 单次对话上下文（当前任务状态、临时变量）
/// - L2: 知识沉淀（生成历史、风格偏好、质量评估）
/// - L3: 快速查询缓存（SPARQL 结果投影）
pub struct AgentMemory {
    /// L0 持久化存储
    pub l0: Arc<glidinghorse::memory::L0Store>,
    /// L2 知识黑板
    pub blackboard: Arc<glidinghorse::memory::Blackboard>,
    /// L3 投影引擎
    pub projection: Arc<glidinghorse::memory::ProjectionEngine>,
    /// 内存管理器（统一调度 L0-L3）
    pub manager: Arc<tokio::sync::Mutex<glidinghorse::memory::MemoryManager>>,
    /// L1 Session 会话记忆（运行时）
    session: Arc<tokio::sync::RwLock<SessionMemory>>,
}

/// L1 Session 会话记忆
#[derive(Debug, Default)]
struct SessionMemory {
    /// 当前任务 ID
    current_task_id: Option<String>,
    /// 临时变量（工具返回缓存）
    temp_vars: HashMap<String, serde_json::Value>,
    /// 对话历史（最近 N 条）
    conversation_history: Vec<ConversationEntry>,
    /// 最大历史条数
    max_history: usize,
}

/// 对话条目
#[derive(Debug, Clone)]
struct ConversationEntry {
    role: String,  // "user" | "agent" | "tool"
    content: String,
    timestamp: DateTime<Utc>,
}

impl AgentMemory {
    /// 初始化 4层记忆系统
    ///
    /// 参数：
    /// - data_dir: 数据目录路径（用于 L0 redb 文件）
    ///
    /// 返回：
    /// - Ok(AgentMemory): 初始化成功
    /// - Err(String): 初始化失败（目录创建、存储打开等）
    pub fn new(data_dir: &str) -> Result<Self, String> {
        // 确保 L0 目录存在
        let l0_path = Path::new(data_dir).join("agent_l0");
        std::fs::create_dir_all(&l0_path)
            .map_err(|e| format!("Failed to create L0 directory: {}", e))?;

        // 初始化 L0Store（redb）
        let l0 = Arc::new(
            glidinghorse::memory::L0Store::new(l0_path.to_str().unwrap())
                .map_err(|e| format!("Failed to init L0Store: {}", e))?
        );

        // 初始化 L2 Blackboard（Oxigraph）
        let blackboard = Arc::new(
            glidinghorse::memory::Blackboard::new()
                .map_err(|e| format!("Failed to init Blackboard: {}", e))?
        );

        // 初始化 L3 ProjectionEngine
        let projection = Arc::new(
            glidinghorse::memory::ProjectionEngine::new(blackboard.clone(), 500)
        );

        // 初始化 MemoryManager
        let core_config = glidinghorse::core::CoreConfig::default();
        let manager = Arc::new(tokio::sync::Mutex::new(
            glidinghorse::memory::MemoryManager::new(
                l0.clone(),
                blackboard.clone(),
                projection.clone(),
                core_config,
            )
        ));

        // 初始化 L1 Session
        let session = Arc::new(tokio::sync::RwLock::new(SessionMemory {
            max_history: 50,
            ..Default::default()
        }));

        Ok(Self {
            l0,
            blackboard,
            projection,
            manager,
            session,
        })
    }

    /// 设置当前任务 ID
    pub async fn set_current_task(&self, task_id: &str) {
        let mut session = self.session.write().await;
        session.current_task_id = Some(task_id.to_string());
    }

    /// 获取当前任务 ID
    pub async fn get_current_task(&self) -> Option<String> {
        let session = self.session.read().await;
        session.current_task_id.clone()
    }

    /// 存储临时变量
    pub async fn set_temp_var(&self, key: &str, value: serde_json::Value) {
        let mut session = self.session.write().await;
        session.temp_vars.insert(key.to_string(), value);
    }

    /// 获取临时变量
    pub async fn get_temp_var(&self, key: &str) -> Option<serde_json::Value> {
        let session = self.session.read().await;
        session.temp_vars.get(key).cloned()
    }

    /// 添加对话历史条目
    pub async fn add_conversation(&self, role: &str, content: &str) {
        let mut session = self.session.write().await;
        
        // 超出限制时移除最旧的
        if session.conversation_history.len() >= session.max_history {
            session.conversation_history.remove(0);
        }
        
        session.conversation_history.push(ConversationEntry {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
        });
    }

    /// 获取对话历史（最近 N 条）
    pub async fn get_conversation_history(&self, limit: usize) -> Vec<ConversationEntry> {
        let session = self.session.read().await;
        session.conversation_history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// 清除会话记忆（新对话开始时）
    pub async fn clear_session(&self) {
        let mut session = self.session.write().await;
        session.current_task_id = None;
        session.temp_vars.clear();
        session.conversation_history.clear();
    }

    /// 写入生成历史到 Blackboard（RDF 三元组）
    ///
    /// 参数：
    /// - prompt_id: 生成任务 ID
    /// - prompt: 用户提示词
    /// - output_path: 输出图片路径
    ///
    /// 示例 RDF：
    /// ```turtle
    /// <iri://generation/{prompt_id}> <schema:prompt> "{prompt}" .
    /// <iri://generation/{prompt_id}> <schema:outputPath> "{output_path}" .
    /// <iri://generation/{prompt_id}> <schema:timestamp> "{timestamp}" .
    /// ```
    pub async fn record_generation(
        &self,
        prompt_id: &str,
        prompt: &str,
        output_path: &str,
    ) -> Result<(), String> {
        let timestamp = Utc::now().to_rfc3339();
        
        // 构造 RDF 三元组（Turtle 格式）
        let turtle = format!(
            "<iri://generation/{pid}> <https://schema.org/prompt> \"{prompt}\" .\n\
             <iri://generation/{pid}> <https://schema.org/outputPath> \"{output}\" .\n\
             <iri://generation/{pid}> <https://schema.org/timestamp> \"{ts}\" .",
            pid = prompt_id,
            prompt = prompt.replace("\"", "\\\""),
            output = output_path.replace("\"", "\\\""),
            ts = timestamp
        );

        // 写入 Blackboard（通过 write_node）
        // 使用 gliding_horse Blackboard 的 write_node 方法
        let node_iri = format!("iri://generation/{}", prompt_id);
        let config = glidinghorse::core::CoreConfig::default();
        self.blackboard.write_node(&node_iri, &turtle, &config)
            .map_err(|e| format!("Failed to write to Blackboard: {}", e))?;

        log::info!("Recording generation: prompt_id={}, output={}", prompt_id, output_path);
        Ok(())
    }

    /// 查询生成历史（从 Blackboard SPARQL）
    ///
    /// 参数：
    /// - prompt_id: 生成任务 ID（可选，None 表示查询全部）
    /// - limit: 最大返回条数
    ///
    /// 返回：
    /// - 生成记录列表（prompt, output_path, timestamp）
    pub async fn query_generations(
        &self,
        prompt_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GenerationRecord>, String> {
        // SPARQL 查询
        let sparql = if let Some(pid) = prompt_id {
            format!(
                "SELECT ?prompt ?output ?ts WHERE {{
                    <iri://generation/{pid}> <https://schema.org/prompt> ?prompt .
                    <iri://generation/{pid}> <https://schema.org/outputPath> ?output .
                    <iri://generation/{pid}> <https://schema.org/timestamp> ?ts .
                }}",
                pid = pid
            )
        } else {
            format!(
                "SELECT ?gen ?prompt ?output ?ts WHERE {{
                    ?gen <https://schema.org/prompt> ?prompt .
                    ?gen <https://schema.org/outputPath> ?output .
                    ?gen <https://schema.org/timestamp> ?ts .
                }} ORDER BY DESC(?ts) LIMIT {}",
                limit
            )
        };

        // 使用 gliding_horse Blackboard 的 SPARQL 查询
        let results = self.blackboard.query(&sparql)
            .map_err(|e| format!("SPARQL query failed: {}", e))?;

        // 解析 SPARQL 结果
        let records: Vec<GenerationRecord> = results.iter()
            .filter_map(|binding| {
                let gen_iri = binding.get("gen").and_then(|v| v.as_str())?;
                let prompt = binding.get("prompt").and_then(|v| v.as_str())?;
                let output = binding.get("output").and_then(|v| v.as_str())?;
                let ts = binding.get("ts").and_then(|v| v.as_str())?;
                Some(GenerationRecord::from_rdf(gen_iri, prompt, output, ts))
            })
            .collect();

        Ok(records)
    }

    /// 记录用户偏好到 L0（持久化）
    pub async fn record_preference(
        &self,
        key: &str,
        value: &str,
    ) -> Result<(), String> {
        // L0 是 redb KV 存储，使用 store 方法写入
        let iri = format!("iri://preference/{}", key);
        self.l0.store(&iri, value)
            .map_err(|e| format!("Failed to write to L0: {}", e))?;

        log::info!("Recording preference: {}={}", key, value);
        Ok(())
    }

    /// 获取用户偏好
    pub async fn get_preference(&self, key: &str) -> Option<String> {
        let iri = format!("iri://preference/{}", key);
        self.l0.retrieve(&iri)
            .ok()
            .flatten()
            .map(|entry| entry.content)
    }
}

/// 生成记录
#[derive(Debug, Clone)]
pub struct GenerationRecord {
    pub prompt_id: String,
    pub prompt: String,
    pub output_path: String,
    pub timestamp: DateTime<Utc>,
}

impl GenerationRecord {
    /// 从 RDF 三元组解析
    pub fn from_rdf(gen_iri: &str, prompt: &str, output: &str, timestamp: &str) -> Self {
        let prompt_id = gen_iri.replace("iri://generation/", "");
        Self {
            prompt_id,
            prompt: prompt.to_string(),
            output_path: output.to_string(),
            timestamp: DateTime::parse_from_rfc3339(timestamp)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        }
    }
}