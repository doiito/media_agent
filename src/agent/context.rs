// AgentContext - Agent 依赖注入容器
// 持有 media_agent 核心组件和 gliding_horse Agent OS 组件
// 新增 LlmClient（DeepSeek API 集成）

use std::sync::Arc;
use crate::execution::ExecutionEngine;
use crate::backend::BackendRouter;
use crate::node::NodeRegistry;
use crate::execution::EventBus;
use crate::config::AppConfig;
use crate::monitor::Monitor;
use crate::agent::llm::LlmClient;

/// Agent 依赖注入容器
///
/// 持有两层组件：
/// 1. media_agent 原有组件（engine/backend/nodes/event_bus/monitor）
/// 2. LLM 客户端（DeepSeek API 或 llama.cpp）
/// 3. gliding_horse Agent OS 组件（gateway/l0/blackboard/skills/runner）
///
/// gliding_horse 组件在 Batch 4-5 填充，初期为 Option
pub struct AgentContext {
    // === media_agent 原有组件 ===
    /// 执行引擎（提交工作流、执行采样）
    pub engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
    /// 后端路由器（stable-diffusion.cpp / llama.cpp）
    pub backend: Arc<BackendRouter>,
    /// 节点注册表（获取节点信息）
    pub nodes: Arc<tokio::sync::Mutex<NodeRegistry>>,
    /// 事件总线（监听执行进度）
    pub event_bus: EventBus,
    /// 系统监控
    pub monitor: Arc<Monitor>,
    /// 应用配置
    pub app_config: AppConfig,

    // === LLM 客户端（DeepSeek API）===
    /// LLM 客户端（从环境变量初始化）
    pub llm_client: Option<Arc<LlmClient>>,

    // === gliding_horse Agent OS 组件（Batch 2-5 填充）===
    /// LLM 网关（OpenAI 兼容 API）
    pub gateway: Option<Arc<glidinghorse::gateway::UnifiedGateway>>,
    /// L0 持久化存储（redb）
    pub l0_store: Option<Arc<glidinghorse::memory::L0Store>>,
    /// L2 知识黑板（Oxigraph RDF）
    pub blackboard: Option<Arc<glidinghorse::memory::Blackboard>>,
    /// 技能注册表（JSON-LD 技能定义）
    pub skill_registry: Option<Arc<glidinghorse::tools::SkillRegistry>>,
    /// Agent Runner（ReAct 循环执行器）
    pub agent_runner: Option<Arc<glidinghorse::core::AgentRunner>>,
}

impl AgentContext {
    /// 创建空的 AgentContext（仅持有 media_agent 组件）
    ///
    /// gliding_horse 组件通过 `with_gateway` / `with_memory` 等方法逐步填充
    pub fn new(
        engine: Arc<tokio::sync::Mutex<ExecutionEngine>>,
        backend: Arc<BackendRouter>,
        nodes: Arc<tokio::sync::Mutex<NodeRegistry>>,
        event_bus: EventBus,
        monitor: Arc<Monitor>,
        app_config: AppConfig,
    ) -> Self {
        Self {
            engine,
            backend,
            nodes,
            event_bus,
            monitor,
            app_config,
            // LLM 客户端初始化为空（通过 with_llm_client 设置）
            llm_client: None,
            // gliding_horse 组件初始化为空
            gateway: None,
            l0_store: None,
            blackboard: None,
            skill_registry: None,
            agent_runner: None,
        }
    }

    /// 设置 LLM 客户端（DeepSeek API）
    pub fn with_llm_client(mut self, client: Arc<LlmClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// 设置 LLM 网关（Batch 2）
    pub fn with_gateway(mut self, gateway: Arc<glidinghorse::gateway::UnifiedGateway>) -> Self {
        self.gateway = Some(gateway);
        self
    }

    /// 设置记忆系统（Batch 4）
    pub fn with_memory(
        mut self,
        l0: Arc<glidinghorse::memory::L0Store>,
        blackboard: Arc<glidinghorse::memory::Blackboard>,
    ) -> Self {
        self.l0_store = Some(l0);
        self.blackboard = Some(blackboard);
        self
    }

    /// 设置技能注册表（Batch 3）
    pub fn with_skills(mut self, registry: Arc<glidinghorse::tools::SkillRegistry>) -> Self {
        self.skill_registry = Some(registry);
        self
    }

    /// 设置 Agent Runner（Batch 5）
    pub fn with_runner(mut self, runner: Arc<glidinghorse::core::AgentRunner>) -> Self {
        self.agent_runner = Some(runner);
        self
    }

    /// 检查 Agent 是否就绪（所有组件已填充）
    pub fn is_ready(&self) -> bool {
        self.gateway.is_some()
            && self.l0_store.is_some()
            && self.blackboard.is_some()
            && self.skill_registry.is_some()
            && self.agent_runner.is_some()
    }
}

impl Clone for AgentContext {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            backend: self.backend.clone(),
            nodes: self.nodes.clone(),
            event_bus: self.event_bus.clone(),
            monitor: self.monitor.clone(),
            app_config: self.app_config.clone(),
            llm_client: self.llm_client.clone(),
            gateway: self.gateway.clone(),
            l0_store: self.l0_store.clone(),
            blackboard: self.blackboard.clone(),
            skill_registry: self.skill_registry.clone(),
            agent_runner: self.agent_runner.clone(),
        }
    }
}