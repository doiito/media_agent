// AgentEngine - 封装 SupervisorAgent 提供对外 API
// 用户对话入口，调用 gliding_horse 的 PDCA 循环

use std::sync::Arc;
use crate::agent::context::AgentContext;

/// Agent Engine
///
/// 封装 gliding_horse 的 SupervisorAgent，提供：
/// 1. 自然语言对话入口（process_task）
/// 2. 工作流执行入口（execute_workflow）
/// 3. 状态查询（status）
///
/// Batch 1 骨架，Batch 5-6 完善实现
pub struct AgentEngine {
    /// Agent 上下文（依赖注入容器）
    context: AgentContext,
    /// SupervisorAgent（gliding_horse PDCA 核心）
    /// Batch 5 初始化，初期为 None
    supervisor: Option<glidinghorse::core::SupervisorAgent>,
}

impl AgentEngine {
    /// 创建 AgentEngine（骨架模式）
    ///
    /// Batch 1 仅创建骨架，supervisor 为 None
    /// Batch 5 通过 `build_supervisor` 初始化
    pub fn new(context: AgentContext) -> Self {
        Self {
            context,
            supervisor: None,
        }
    }

    /// 构建 SupervisorAgent（Batch 5）
    ///
    /// 需要以下组件已填充：
    /// - gateway: LLM 网关
    /// - l0_store + blackboard: 记忆系统
    /// - skill_registry: 技能注册表
    ///
    /// 返回：
    /// - Ok(supervisor): 构建成功
    /// - Err(String): 缺少必要组件或构建失败
    pub fn build_supervisor(&mut self) -> Result<(), String> {
        // 检查必要组件
        if !self.context.is_ready() {
            return Err("AgentContext not ready: missing gateway/memory/skills".to_string());
        }

        // 从 context 取出组件（clone Arc）
        let gateway = self.context.gateway.clone()
            .ok_or("Missing gateway")?;
        let l0 = self.context.l0_store.clone()
            .ok_or("Missing l0_store")?;
        let blackboard = self.context.blackboard.clone()
            .ok_or("Missing blackboard")?;
        let skills = self.context.skill_registry.clone()
            .ok_or("Missing skill_registry")?;

        // 构建 ProjectionEngine + MemoryManager
        let projection = Arc::new(glidinghorse::memory::ProjectionEngine::new(blackboard.clone(), 500));
        let core_config = glidinghorse::core::CoreConfig::default();
        let memory_manager = Arc::new(tokio::sync::Mutex::new(
            glidinghorse::memory::MemoryManager::new(l0.clone(), blackboard.clone(), projection.clone(), core_config)
        ));

        // 构建 TemplateEngine（使用临时目录）
        let templates_dir = std::env::temp_dir().join("agent_templates");
        std::fs::create_dir_all(&templates_dir)
            .map_err(|e| format!("Failed to create templates dir: {}", e))?;
        let templates = Arc::new(
            glidinghorse::templates::TemplateEngine::new(&templates_dir)
                .map_err(|e| format!("Failed to init TemplateEngine: {}", e))?
        );

        // 构建 AgentSettings
        let agent_settings = glidinghorse::config::AgentSettings::default();

        // 构建 AgentRunner
        let runner = Arc::new(glidinghorse::core::AgentRunner::new(
            gateway,
            skills.clone(),
            blackboard.clone(),
            l0,
            memory_manager,
            templates.clone(),
            agent_settings,
        ));

        // TODO: Batch 2 注册 ComfyUI 工具到 runner.tool_executor

        // 构建 EventBus
        let event_bus = Arc::new(glidinghorse::core::event_bus::EventBus::new(100));

        // 构建 SupervisorAgent
        let max_iterations = 15;
        let supervisor = glidinghorse::core::SupervisorAgent::new(
            runner,
            templates,
            skills,
            event_bus,
            max_iterations,
        )
        .with_memory(Some(blackboard), None, None);

        self.supervisor = Some(supervisor);
        Ok(())
    }

    /// 处理用户任务（自然语言对话）
    ///
    /// 参数：
    /// - message: 用户输入（如"画一只赛博朋克风格的猫"）
    /// - workflow_path: 可选工作流 JSON-LD 路径
    ///
    /// 返回：
    /// - Ok((task_id, TaskResult)): 执行成功
    /// - Err(String): 执行失败
    ///
    /// Batch 6 实现
    pub async fn process_task(
        &mut self,
        message: &str,
        workflow_path: Option<&str>,
    ) -> Result<(String, glidinghorse::core::agent_runner::TaskResult), String> {
        // 检查 supervisor 是否已初始化
        let supervisor = self.supervisor.as_mut()
            .ok_or("SupervisorAgent not initialized. Call build_supervisor() first.")?;

        // 生成任务 ID
        let task_id = uuid::Uuid::new_v4().to_string();
        let task_iri = format!("iri://task/{}", task_id);

        // 如果指定工作流，注入到 TaskContext
        if let Some(path) = workflow_path {
            let jsonld = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read workflow: {}", e))?;

            let ctx = glidinghorse::core::agent_runner::TaskContext::new(&task_iri, message, 15)
                .with_workflow(&jsonld);

            supervisor.process_task_with_context(message, &task_iri, ctx).await
                .map(|result| (task_id, result))
                .map_err(|e| e.to_string())
        } else {
            // 直接处理
            supervisor.process_task(message, &task_iri).await
                .map(|result| (task_id, result))
                .map_err(|e| e.to_string())
        }
    }

    /// 查询 Agent 状态
    pub fn status(&self) -> AgentStatus {
        AgentStatus {
            context_ready: self.context.is_ready(),
            supervisor_ready: self.supervisor.is_some(),
        }
    }

    /// 获取 AgentContext（用于工具注册等）
    pub fn context(&self) -> &AgentContext {
        &self.context
    }
}

impl Clone for AgentEngine {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            supervisor: None, // SupervisorAgent 不 clone，需重新 build
        }
    }
}

/// Agent 状态
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStatus {
    /// AgentContext 是否就绪（gateway/memory/skills 已填充）
    pub context_ready: bool,
    /// SupervisorAgent 是否已初始化
    pub supervisor_ready: bool,
}