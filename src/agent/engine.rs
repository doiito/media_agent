// AgentEngine - 封装 SupervisorAgent 提供对外 API
// 用户对话入口，调用 gliding_horse 的 PDCA 循环

use std::sync::Arc;
use std::path::PathBuf;
use crate::agent::context::AgentContext;
use crate::config::AppConfig;

/// Agent Engine
///
/// 封装 gliding_horse 的 SupervisorAgent，提供：
/// 1. 自然语言对话入口（process_task）
/// 2. 工作流执行入口（execute_workflow）
/// 3. 状态查询（status）
///
/// Agent 引擎实现
pub struct AgentEngine {
    /// Agent 上下文（依赖注入容器）
    context: AgentContext,
    /// SupervisorAgent（gliding_horse PDCA 核心）
    /// Batch 5 初始化，初期为 None
    supervisor: Option<glidinghorse::core::SupervisorAgent>,
    /// 应用配置
    config: Option<AppConfig>,
    /// ComfyUI 智能引擎（SkillGraph + Discovery + Evolution + KnowledgeGraph + Causal + Timeline）
    intelligence: Option<Arc<crate::agent::advanced_intelligence::ComfyUiIntelligence>>,
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
            config: None,
            intelligence: None,
        }
    }

    /// 创建 AgentEngine 并加载配置
    ///
    /// 使用配置中的提示词目录加载模板
    pub fn with_config(context: AgentContext, config: AppConfig) -> Self {
        Self {
            context,
            supervisor: None,
            config: Some(config),
            intelligence: None,
        }
    }

    /// 初始化智能引擎（SkillGraph + Evolution + Causal + KnowledgeGraph）
    pub fn init_intelligence(&mut self) -> Result<(), String> {
        let intel_config = crate::agent::advanced_intelligence::IntelligenceConfig::default();
        let intelligence = crate::agent::advanced_intelligence::ComfyUiIntelligence::new(intel_config)?;
        self.intelligence = Some(Arc::new(intelligence));
        log::info!("ComfyUiIntelligence initialized");
        Ok(())
    }

    /// 获取智能引擎引用
    pub fn intelligence(&self) -> Option<&Arc<crate::agent::advanced_intelligence::ComfyUiIntelligence>> {
        self.intelligence.as_ref()
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

        // 构建 TemplateEngine（使用配置目录或默认目录）
        let templates_dir = self.config.as_ref()
            .map(|c| PathBuf::from(&c.paths.prompts_dir))
            .unwrap_or_else(|| {
                // 尝试多个候选目录
                let candidates = [
                    PathBuf::from("prompts"),
                    PathBuf::from(".gliding_horse/prompts"),
                    std::env::temp_dir().join("agent_templates"),
                ];
                for candidate in candidates {
                    if candidate.exists() {
                        log::info!("Using prompts directory: {:?}", candidate);
                        return candidate;
                    }
                }
                // 默认使用项目根目录下的 prompts
                PathBuf::from("prompts")
            });

        // 确保目录存在
        std::fs::create_dir_all(&templates_dir)
            .map_err(|e| format!("Failed to create prompts dir {:?}: {}", templates_dir, e))?;

        log::info!("Loading prompt templates from: {:?}", templates_dir);

        let templates = Arc::new(
            glidinghorse::templates::TemplateEngine::new(&templates_dir)
                .map_err(|e| format!("Failed to init TemplateEngine: {}", e))?
        );

        // 注册 ComfyUI 专用提示词模板
        self.register_comfyui_prompts(&templates);

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

        // 注册 ComfyUI 工具到 runner.tool_executor
        {
            let mut tool_executor = runner.tool_executor.write()
                .expect("Failed to lock tool_executor");
            crate::agent::tools::register_comfyui_tools(&mut tool_executor, Arc::new(self.context.clone()));

            // 注册智能工具（SkillGraph + Discovery + Evolution）
            if self.intelligence.is_none() {
                self.init_intelligence()?;
            }
            if let Some(ref intel) = self.intelligence {
                crate::agent::tools::register_intelligence_tools(&mut tool_executor, intel.clone());
                log::info!("Registered intelligence tools to AgentRunner");
            }
        }

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

    /// 注册 ComfyUI 专用提示词模板
    ///
    /// 从文件加载或使用内置 fallback
    fn register_comfyui_prompts(&self, templates: &Arc<glidinghorse::templates::TemplateEngine>) {
        // 定义 ComfyUI Agent 角色
        let roles = ["pa", "da", "ca", "aa", "sa"];

        for role in roles {
            // 尝试从文件加载
            let template_path = PathBuf::from("prompts").join(role).join("system.md");
            if template_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&template_path) {
                    log::info!("Loaded prompt template for role '{}' from {:?}", role, template_path);
                    templates.add_template(&format!("{}_system", role), &content, role);
                }
            } else {
                // 使用内置 fallback
                let fallback = self.get_builtin_prompt(role);
                templates.add_template(&format!("{}_system", role), fallback, role);
                log::info!("Using builtin prompt template for role '{}'", role);
            }
        }
    }

    /// 获取内置提示词（当文件不存在时的 fallback）
    fn get_builtin_prompt(&self, role: &str) -> &'static str {
        match role {
            "pa" => "You are the Planning Agent (PA) for ComfyUI image/video generation. Analyze the user request and create a generation plan. Select the appropriate workflow template, recommend parameters and models. Output JSON-formatted results.",
            "da" => "You are the Doing Agent (DA) for ComfyUI. Execute the generation workflow, construct the node graph, and produce the output. Output JSON-formatted results.",
            "ca" => "You are the Checking Agent (CA) for ComfyUI. Verify the generation output quality, check if it matches user requirements. Output JSON-formatted results.",
            "aa" => "You are the Acting Agent (AA) for ComfyUI. Make final decisions based on quality check results, adjust parameters if needed, and provide user summary. Output JSON-formatted results.",
            "sa" => "You are the Supervisor Agent (SA) for ComfyUI Rust Agent. Coordinate PA, DA, CA, AA agents to complete image/video generation tasks using PDCA cycle. Output JSON-formatted results.",
            _ => "",
        }
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
            config: self.config.clone(),
            intelligence: self.intelligence.clone(),
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