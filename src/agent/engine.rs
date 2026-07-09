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
        if !self.context.is_ready() {
            self.init_gliding_horse_components()?;
        }
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
            let mut tool_executor = runner.tool_executor.write().expect("Failed to acquire tool_executor lock");
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

        // 桥接 gliding_horse EventBus → media_agent EventBus
        // 将 PDCA 阶段事件转发到 WebSocket，让前端能展示中间过程
        let mut gh_rx = event_bus.subscribe();
        let media_event_bus = self.context.event_bus.clone();
        tokio::spawn(async move {
            loop {
                match gh_rx.recv().await {
                    Ok(gh_event) => {
                        bridge_gh_event(&media_event_bus, &gh_event).await;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("Agent event bridge lagged by {} events", n);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        log::info!("Agent event bridge closed");
                        break;
                    }
                }
            }
        });
        log::info!("Agent event bridge started (gliding_horse → media_agent EventBus)");

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

    /// 从环境变量初始化 gliding_horse 组件（gateway, memory, skills, runner）
    fn init_gliding_horse_components(&mut self) -> Result<(), String> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_API_KEY"))
            .map_err(|_| "DEEPSEEK_API_KEY or AGENT_OS_GATEWAY_API_KEY not set".to_string())?;
        let base_url = std::env::var("DEEPSEEK_API_URL")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_API_URL"))
            .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
        let default_model = std::env::var("DEEPSEEK_MODEL")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_DEFAULT_MODEL"))
            .unwrap_or_else(|_| "deepseek-v4-flash".to_string());

        let gateway_settings = glidinghorse::config::settings::GatewaySettings {
            base_url,
            api_key,
            default_model,
            timeout_seconds: 120,
            max_retries: 3,
            model_mapping: std::collections::HashMap::new(),
        };
        let gateway = Arc::new(
            glidinghorse::gateway::UnifiedGateway::new(&gateway_settings)
                .map_err(|e| format!("Failed to create UnifiedGateway: {:?}", e))?
        );

        let data_dir = self.config.as_ref()
            .map(|c| PathBuf::from(&c.paths.temp_dir).join("agent_memory"))
            .unwrap_or_else(|| PathBuf::from("/tmp/agent_memory"));
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| format!("Failed to create memory dir: {}", e))?;

        let l0_store = Arc::new(
            glidinghorse::memory::L0Store::new(
                data_dir.to_str().ok_or("Invalid memory path")?
            ).map_err(|e| format!("Failed to create L0Store: {:?}", e))?
        );
        let blackboard = Arc::new(
            glidinghorse::memory::Blackboard::new()
                .map_err(|e| format!("Failed to create Blackboard: {:?}", e))?
        );

        let skill_registry = Arc::new(glidinghorse::tools::SkillRegistry::new());
        let skills_dir = self.config.as_ref()
            .map(|c| PathBuf::from(&c.paths.skills_dir))
            .unwrap_or_else(|| PathBuf::from("skills"));
        if skills_dir.exists() {
            let _ = skill_registry.load_from_jsonld(&skills_dir);
        }

        let projection = Arc::new(glidinghorse::memory::ProjectionEngine::new(blackboard.clone(), 500));
        let core_config = glidinghorse::core::CoreConfig::default();
        let memory_manager = Arc::new(tokio::sync::Mutex::new(
            glidinghorse::memory::MemoryManager::new(
                l0_store.clone(), blackboard.clone(), projection, core_config)
        ));
        let templates_dir = self.config.as_ref()
            .map(|c| PathBuf::from(&c.paths.prompts_dir))
            .unwrap_or_else(|| PathBuf::from("prompts"));
        std::fs::create_dir_all(&templates_dir)
            .map_err(|e| format!("Failed to create prompts dir: {}", e))?;
        let templates = Arc::new(
            glidinghorse::templates::TemplateEngine::new(&templates_dir)
                .map_err(|e| format!("Failed to init TemplateEngine: {:?}", e))?
        );

        let agent_runner = Arc::new(glidinghorse::core::AgentRunner::new(
            gateway.clone(),
            skill_registry.clone(),
            blackboard.clone(),
            l0_store.clone(),
            memory_manager,
            templates,
            glidinghorse::config::AgentSettings::default(),
        ));

        self.context = self.context.clone()
            .with_gateway(gateway)
            .with_memory(l0_store, blackboard)
            .with_skills(skill_registry)
            .with_runner(agent_runner);

        log::info!("gliding_horse components initialized from env (model: {}, url: {})", gateway_settings.default_model, gateway_settings.base_url);
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

        // 发布执行开始事件到 media_agent EventBus（前端可收到）
        self.context.event_bus.publish(
            crate::execution::Event::ExecutionStart {
                prompt_id: task_id.clone(),
            }
        ).await;

        // 如果指定工作流，注入到 TaskContext
        let result = if let Some(path) = workflow_path {
            let jsonld = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read workflow: {}", e))?;

            let ctx = glidinghorse::core::agent_runner::TaskContext::new(&task_iri, message, 15)
                .with_workflow(&jsonld);

            supervisor.process_task_with_context(message, &task_iri, ctx).await
                .map(|result| (task_id.clone(), result))
                .map_err(|e| e.to_string())
        } else {
            // 直接处理
            supervisor.process_task(message, &task_iri).await
                .map(|result| (task_id.clone(), result))
                .map_err(|e| e.to_string())
        };

        // 根据结果发布完成事件
        match &result {
            Ok(_) => {
                self.context.event_bus.publish(
                    crate::execution::Event::ExecutionSuccess {
                        prompt_id: task_id.clone(),
                        outputs: std::collections::HashMap::new(),
                    }
                ).await;
            }
            Err(e) => {
                self.context.event_bus.publish(
                    crate::execution::Event::ExecutionError {
                        prompt_id: task_id.clone(),
                        error: e.clone(),
                    }
                ).await;
            }
        }

        result
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

/// 桥接 gliding_horse 事件到 media_agent EventBus
///
/// 将 PDCA 循环的 THOUGHT 事件映射为前端可展示的 AgentPhaseStart/AgentThought 事件
async fn bridge_gh_event(
    media_bus: &crate::execution::EventBus,
    gh_event: &glidinghorse::core::event_bus::Event,
) {
    // 只处理 THOUGHT 类型事件
    if gh_event.event_type != "THOUGHT" {
        return;
    }

    // 动态解析 payload JSON，避免依赖 gliding_horse 内部类型
    let payload: serde_json::Value = match serde_json::from_str(&gh_event.payload) {
        Ok(v) => v,
        Err(_) => return,
    };

    // 提取 action 和 thought 字段
    let action = payload["event"]["Thought"]["action"]
        .as_str()
        .or_else(|| payload["action"].as_str())
        .unwrap_or("");
    let thought = payload["event"]["Thought"]["thought"]
        .as_str()
        .or_else(|| payload["thought"].as_str())
        .unwrap_or("");

    let prompt_id = gh_event.task_iri.replace("iri://task/", "");

    // 根据 action 映射到 PDCA 阶段
    let (phase, description) = match action {
        "dispatch_plan" | "plan_created" => ("planning", "Planning Agent 正在分析需求并制定生成方案"),
        "dispatch_do" => ("doing", "Doing Agent 正在构建工作流并执行生成"),
        "dispatch_check" => ("checking", "Checking Agent 正在验证生成结果"),
        "dispatch_act" => ("acting", "Acting Agent 正在做最终决策"),
        _ => {
            // 未识别的 action，发布为 AgentThought
            media_bus.publish(crate::execution::Event::AgentThought {
                prompt_id,
                thought: thought.to_string(),
                action: action.to_string(),
            }).await;
            return;
        }
    };

    media_bus.publish(crate::execution::Event::AgentPhaseStart {
        prompt_id,
        phase: phase.to_string(),
        description: description.to_string(),
    }).await;

    // 如果有 thought 内容，同时发布
    if !thought.is_empty() {
        media_bus.publish(crate::execution::Event::AgentThought {
            prompt_id: gh_event.task_iri.replace("iri://task/", ""),
            thought: thought.to_string(),
            action: action.to_string(),
        }).await;
    }
}