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
            let empty_role_tools = glidinghorse::tools::RoleToolConfig::default();
            let role_tools = ["Plan", "Do", "Check", "Act"]
                .into_iter()
                .map(|role| (role.to_string(), empty_role_tools.clone()))
                .collect();
            tool_executor.set_tool_group_manager(glidinghorse::tools::ToolGroupManager::new(Some(
                glidinghorse::tools::ToolGroupSettings {
                    enabled: true,
                    roles: role_tools,
                },
            )));
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
        let max_iterations = self.config.as_ref()
            .map(|config| config.agent.max_iterations)
            .unwrap_or(15);
        let max_iterations = u32::try_from(max_iterations)
            .map_err(|_| "agent.max_iterations exceeds u32 range".to_string())?;
        let max_pdca_cycles = self.config.as_ref()
            .map(|config| config.agent.max_pdca_cycles)
            .unwrap_or(3);
        let max_pdca_cycles = u32::try_from(max_pdca_cycles)
            .map_err(|_| "agent.max_pdca_cycles exceeds u32 range".to_string())?;
        let supervisor = glidinghorse::core::SupervisorAgent::with_pdca_cycles(
            runner,
            templates,
            skills,
            event_bus,
            max_iterations,
            max_pdca_cycles,
        )
        .with_memory(Some(blackboard), None, None);

        self.supervisor = Some(supervisor);
        Ok(())
    }

    /// 注册 ComfyUI 专用提示词模板
    ///
    /// 从文件加载或使用内置 fallback
    /// 注意：模板名必须匹配 build_agent_md 中的查找逻辑
    /// - SA: "prompts/sa/skeleton"
    /// - 其他: "prompts/workers/{role}/skeleton"
    fn register_comfyui_prompts(&self, templates: &Arc<glidinghorse::templates::TemplateEngine>) {
        let roles = ["pa", "da", "ca", "aa", "sa"];

        for role in roles {
            let template_path = PathBuf::from("prompts").join(role).join("system.md");
            let content = if template_path.exists() {
                match std::fs::read_to_string(&template_path) {
                    Ok(c) => {
                        log::info!("Loaded prompt template for role '{}' from {:?}", role, template_path);
                        c
                    }
                    Err(e) => {
                        log::warn!("Failed to read prompt file {:?}: {}, using builtin", template_path, e);
                        self.get_builtin_prompt(role).to_string()
                    }
                }
            } else {
                log::info!("Prompt file not found for role '{}', using builtin", role);
                self.get_builtin_prompt(role).to_string()
            };

            // 注册匹配 build_agent_md 查找逻辑的模板名
            let template_name = if role == "sa" {
                "prompts/sa/skeleton".to_string()
            } else {
                format!("prompts/workers/{}/skeleton", role)
            };
            templates.add_template(&template_name, &content, role);
            log::info!("Registered template '{}' for role '{}'", template_name, role);
        }
    }

    /// 获取内置提示词（当文件不存在时的 fallback）
    fn get_builtin_prompt(&self, role: &str) -> &'static str {
        match role {
            "pa" => "You are PA for a Rust-native media system. Produce one structured text/image/video generation plan. Inspect native runtime evidence when model capability is uncertain. Never propose Python or manual UI operations.",
            "da" => "You are DA. Execute the PA plan with generate_media through Rust and stable-diffusion.cpp. Return the exact real artifact path. Tool failure is task failure; never use Python or fabricate output.",
            "ca" => "You are CA. Call inspect_artifact exactly once on the DA output, then immediately return the audit and reject missing, empty, undecodable, or requirement-mismatched media.",
            "aa" => "You are AA. Accept only a CA-verified artifact. Retry only bounded parameter-adjustable failures; fail immediately for missing, corrupt, or incompatible models.",
            "sa" => "You are the Gliding Horse Supervisor for a Rust-native media system. Coordinate PA, DA, CA, and AA. A task succeeds only with a verified image or video artifact. Python is forbidden.",
            _ => "",
        }
    }

    /// 从环境变量初始化 gliding_horse 组件（gateway, memory, skills, runner）
    fn init_gliding_horse_components(&mut self) -> Result<(), String> {
        let mut llm = self.config.as_ref()
            .map(|config| config.agent.llm.clone())
            .unwrap_or_default();
        llm.merge_env_overrides();

        if llm.provider == crate::config::AgentLlmProvider::Deepseek
            && llm.api_key.trim().is_empty()
        {
            return Err(
                "DeepSeek provider requires AGENT_LLM_API_KEY or DEEPSEEK_API_KEY".to_string()
            );
        }

        let base_url = llm.gateway_base_url();
        let api_key = if llm.api_key.trim().is_empty() {
            "local".to_string()
        } else {
            llm.api_key.clone()
        };
        let default_model = llm.model.clone();

        let gateway_settings = glidinghorse::config::settings::GatewaySettings {
            base_url,
            api_key,
            default_model,
            timeout_seconds: llm.timeout_seconds,
            max_retries: llm.max_retries,
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

        log::info!(
            "gliding_horse components initialized (provider: {:?}, model: {}, url: {})",
            llm.provider,
            gateway_settings.default_model,
            gateway_settings.base_url
        );
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
        let raw_result = if let Some(path) = workflow_path {
            let jsonld = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read workflow: {}", e))?;

            let ctx = glidinghorse::core::agent_runner::TaskContext::new(&task_iri, message, 15)
                .with_workflow(&jsonld);

            supervisor.process_task_with_context(message, &task_iri, ctx).await
                .map(|result| (task_id.clone(), result))
                .map_err(|e| e.to_string())
        } else {
            // Media generation always uses an explicit four-stage Gliding Horse DAG. The
            // user request and UI parameters remain dynamic inputs to the PA/DA agents.
            let workflow = native_media_pdca_workflow_json();
            let ctx = glidinghorse::core::agent_runner::TaskContext::new(
                &task_iri,
                message,
                self.config.as_ref().map(|c| c.agent.max_iterations).unwrap_or(15) as u32,
            )
            .with_workflow(&workflow);
            supervisor.process_task_with_context(message, &task_iri, ctx).await
                .map(|result| (task_id.clone(), result))
                .map_err(|e| e.to_string())
        };

        let result = match raw_result {
            Ok((result_task_id, task_result)) => {
                validate_generation_task_result(&task_result, message)
                    .map(|_| (result_task_id, task_result))
            }
            Err(error) => Err(error),
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

fn native_media_pdca_workflow_json() -> String {
    serde_json::json!({
        "@id": "iri://workflow/native-media-pdca",
        "name": "Native media PA-DA-CA-AA",
        "description": "Rust-native media generation with mandatory planning, execution, verification, and decision",
        "version": "1.0",
        "entry_node": "native-media-plan",
        "nodes": [
            {
                "@id": "native-media-plan",
                "agent_role": "Plan",
                "objective": "Interpret the dynamic user request and UI constraints, then produce an executable native media plan",
                "next": "native-media-do",
                "expected_output": "A concrete plan preserving the user's prompt, intent, input image, dimensions, duration, and explicit parameters",
                "success_criteria": "No fabricated paths, no Python, and all explicit user constraints are retained"
            },
            {
                "@id": "native-media-do",
                "agent_role": "Do",
                "objective": "Execute the current request with generate_media using Rust and stable-diffusion.cpp",
                "next": "native-media-check",
                "dependencies": ["native-media-plan"],
                "expected_output": "A real generated artifact and its exact output path",
                "success_criteria": "generate_media succeeds and returns a non-empty artifact"
            },
            {
                "@id": "native-media-check",
                "agent_role": "Check",
                "objective": "Verify the exact DA artifact with one inspect_artifact call, compare its effective prompt with the original request, then finish immediately",
                "next": "native-media-act",
                "dependencies": ["native-media-do"],
                "expected_output": "A structured artifact validation result",
                "success_criteria": "The artifact exists, decodes or probes successfully, has the requested media type, and the effective prompt preserves the requested visual attributes"
            },
            {
                "@id": "native-media-act",
                "agent_role": "Act",
                "objective": "Accept only the CA-verified artifact and return its exact path with a concise final decision",
                "dependencies": ["native-media-check"],
                "expected_output": "An accept or reject decision preserving the verified artifact path",
                "success_criteria": "Acceptance requires a CA-verified image or video artifact",
                "final_node": true
            }
        ]
    })
    .to_string()
}

fn validate_generation_task_result(
    result: &glidinghorse::core::agent_runner::TaskResult,
    request: &str,
) -> Result<(), String> {
    let status = result.status.to_ascii_lowercase();
    if !matches!(status.as_str(), "success" | "completed" | "accepted") {
        let detail = if result.errors.is_empty() {
            result.summary.clone()
        } else {
            result.errors.join("; ")
        };
        return Err(format!("Gliding Horse task ended with status '{}': {}", result.status, detail));
    }

    let expectation = media_expectation(request);
    validate_pdca_trace(result, &expectation)
}

fn validate_pdca_trace(
    result: &glidinghorse::core::agent_runner::TaskResult,
    expectation: &MediaExpectation,
) -> Result<(), String> {
    let trace = result
        .output
        .as_ref()
        .and_then(|output| output.get("pdca_trace"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Gliding Horse result is missing the mandatory PA-DA-CA-AA trace".to_string())?;

    let roles: Vec<&str> = trace
        .iter()
        .filter_map(|phase| phase.get("role").and_then(serde_json::Value::as_str))
        .collect();
    if roles != ["Plan", "Do", "Check", "Act"] {
        return Err(format!(
            "Gliding Horse phase trace is incomplete or out of order: {:?}",
            roles
        ));
    }

    for phase in trace {
        let role = phase.get("role").and_then(serde_json::Value::as_str).unwrap_or("unknown");
        let phase_status = phase
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_ascii_lowercase();
        if !matches!(phase_status.as_str(), "success" | "completed" | "accepted") {
            return Err(format!("Gliding Horse {} phase did not pass: {}", role, phase_status));
        }
        if matches!(role, "Check" | "Act") && phase_reports_rejection(phase) {
            return Err(format!(
                "Gliding Horse {} phase explicitly rejected or did not complete the artifact",
                role
            ));
        }
    }

    let do_phase = &trace[1];
    let check_phase = &trace[2];
    let mut do_paths = Vec::new();
    if let Some(artifacts) = do_phase.get("artifacts") {
        collect_media_paths(artifacts, &mut do_paths);
    }
    let mut check_paths = Vec::new();
    if let Some(artifacts) = check_phase.get("artifacts") {
        collect_media_paths(artifacts, &mut check_paths);
    }
    if do_paths.is_empty() {
        return Err(
            "Gliding Horse DA did not return a generated media artifact after generate_media"
                .to_string(),
        );
    }
    if check_paths.is_empty() {
        return Err(
            "Gliding Horse CA did not return inspect_artifact evidence for the generated media"
                .to_string(),
        );
    }

    let checked_paths: std::collections::HashSet<String> = check_paths
        .iter()
        .map(|path| comparable_media_path(path))
        .collect();
    let mut errors = Vec::new();
    for path in do_paths {
        if !checked_paths.contains(&comparable_media_path(&path)) {
            errors.push(format!("CA did not inspect DA artifact '{}'", path));
            continue;
        }
        if expectation.is_input(&path) {
            errors.push(format!("input media '{}' cannot be accepted as generated output", path));
            continue;
        }
        if media_kind_from_path(&path) != Some(expectation.kind) {
            errors.push(format!(
                "artifact '{}' is {:?}, but the request requires {:?}",
                path,
                media_kind_from_path(&path),
                expectation.kind
            ));
            continue;
        }
        match crate::agent::tools::inspect_media_artifact(&path) {
            Ok(_) => return Ok(()),
            Err(error) => errors.push(error),
        }
    }
    Err(format!(
        "Gliding Horse CA artifact evidence failed validation: {}",
        errors.join("; ")
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedMediaKind {
    Image,
    Video,
}

#[derive(Debug)]
struct MediaExpectation {
    kind: ExpectedMediaKind,
    input_paths: std::collections::HashSet<String>,
}

impl MediaExpectation {
    fn is_input(&self, path: &str) -> bool {
        self.input_paths.contains(&comparable_media_path(path))
    }
}

fn media_expectation(request: &str) -> MediaExpectation {
    let lower = request.to_ascii_lowercase();
    let explicit_video = lower.contains("\"intent\": \"image_to_video\"")
        || lower.contains("\"intent\": \"text_to_video\"");
    let explicit_image = lower.contains("\"intent\": \"image_to_image\"")
        || lower.contains("\"intent\": \"text_to_image\"");
    let mentions_video = [
        "短视频",
        "视频",
        "文生视频",
        "图生视频",
        "video clip",
        "generate a video",
        "create a video",
        "image to video",
        "text to video",
        "animation",
    ]
    .iter()
    .any(|keyword| lower.contains(keyword));
    let kind = if explicit_video || (!explicit_image && mentions_video) {
        ExpectedMediaKind::Video
    } else {
        ExpectedMediaKind::Image
    };

    let mut input_paths = std::collections::HashSet::new();
    let mut in_input_block = false;
    for line in request.lines() {
        let line = line.trim();
        if line == "<input_image>" {
            in_input_block = true;
        } else if line == "</input_image>" {
            in_input_block = false;
        } else if in_input_block {
            if let Some(path) = line.strip_prefix("path:") {
                input_paths.insert(comparable_media_path(path.trim()));
            }
        }
    }

    MediaExpectation { kind, input_paths }
}

fn media_kind_from_path(path: &str) -> Option<ExpectedMediaKind> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())?
        .to_ascii_lowercase();
    match extension.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" => Some(ExpectedMediaKind::Image),
        "mp4" | "webm" => Some(ExpectedMediaKind::Video),
        _ => None,
    }
}

fn comparable_media_path(path: &str) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| std::path::PathBuf::from(path))
        .to_string_lossy()
        .into_owned()
}

fn phase_reports_rejection(phase: &serde_json::Value) -> bool {
    let summary = phase.get("summary").and_then(serde_json::Value::as_str).unwrap_or("");
    let output = phase.get("output").map(serde_json::Value::to_string).unwrap_or_default();
    let report = format!("{} {}", summary, output).to_ascii_lowercase();
    [
        "wrong type",
        "task incomplete",
        "not the requested",
        "does not match",
        "verification failed",
        "failed verification",
        "reject",
        "未完成",
        "不匹配",
        "错误类型",
        "拒绝",
        "验证失败",
    ]
    .iter()
    .any(|marker| report.contains(marker))
}

fn collect_media_paths(value: &serde_json::Value, paths: &mut Vec<String>) {
    match value {
        serde_json::Value::String(path) => {
            let lower = path.to_ascii_lowercase();
            let is_media = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm"]
                .iter()
                .any(|extension| lower.ends_with(extension));
            if is_media {
                paths.push(path.clone());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_media_paths(value, paths);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values() {
                collect_media_paths(value, paths);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod generation_result_tests {
    use super::*;

    fn result(status: &str, output_path: &str) -> glidinghorse::core::agent_runner::TaskResult {
        let artifact = serde_json::json!({"path": output_path});
        glidinghorse::core::agent_runner::TaskResult {
            task_iri: "iri://task/test".to_string(),
            status: status.to_string(),
            summary: "test".to_string(),
            output: Some(serde_json::json!({
                "final": {"output_path": output_path},
                "pdca_trace": [
                    {"role": "Plan", "status": "success", "artifacts": []},
                    {"role": "Do", "status": "success", "artifacts": [artifact]},
                    {"role": "Check", "status": "success", "artifacts": [artifact]},
                    {"role": "Act", "status": "success", "artifacts": []}
                ]
            })),
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 1,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        }
    }

    #[test]
    fn accepts_only_decodable_media() {
        let path = std::env::temp_dir().join(format!(
            "media-agent-result-{}.png",
            uuid::Uuid::new_v4()
        ));
        let image = image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 31) as u8, ((x + y) * 15) as u8])
        });
        image.save(&path).unwrap();

        let validation = validate_generation_task_result(&result(
            "success",
            path.to_str().unwrap(),
        ), "{\"intent\": \"text_to_image\"}");
        let _ = std::fs::remove_file(path);
        assert!(validation.is_ok());
    }

    #[test]
    fn rejects_corrupt_media_even_when_agent_reports_success() {
        let path = std::env::temp_dir().join(format!(
            "media-agent-corrupt-{}.png",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"not an image").unwrap();

        let error = validate_generation_task_result(&result(
            "success",
            path.to_str().unwrap(),
        ), "{\"intent\": \"text_to_image\"}")
        .unwrap_err();
        let _ = std::fs::remove_file(path);
        assert!(error.contains("CA artifact evidence failed validation"));
    }

    #[test]
    fn rejects_failure_status_even_with_a_media_path() {
        let error = validate_generation_task_result(
            &result("failed", "/tmp/result.png"),
            "{\"intent\": \"text_to_image\"}",
        )
            .unwrap_err();
        assert!(error.contains("status 'failed'"));
    }

    #[test]
    fn rejects_success_without_ca_tool_evidence() {
        let mut task_result = result("success", "/tmp/result.png");
        task_result.output.as_mut().unwrap()["pdca_trace"][2]["artifacts"] =
            serde_json::json!([]);
        let error = validate_generation_task_result(
            &task_result,
            "{\"intent\": \"text_to_image\"}",
        )
        .unwrap_err();
        assert!(error.contains("CA did not return inspect_artifact evidence"));
    }

    #[test]
    fn rejects_input_image_reused_as_generated_output() {
        let path = std::env::temp_dir().join(format!(
            "media-agent-input-{}.png",
            uuid::Uuid::new_v4()
        ));
        image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 31) as u8, ((x + y) * 15) as u8])
        })
        .save(&path)
        .unwrap();
        let request = format!(
            "<input_image>\npath: {}\n</input_image>\n{{\"intent\": \"image_to_image\"}}",
            path.display()
        );

        let error = validate_generation_task_result(
            &result("success", path.to_str().unwrap()),
            &request,
        )
        .unwrap_err();
        let _ = std::fs::remove_file(path);
        assert!(error.contains("cannot be accepted as generated output"));
    }

    #[test]
    fn rejects_image_when_video_was_requested() {
        let path = std::env::temp_dir().join(format!(
            "media-agent-wrong-kind-{}.png",
            uuid::Uuid::new_v4()
        ));
        image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 31) as u8, ((x + y) * 15) as u8])
        })
        .save(&path)
        .unwrap();

        let error = validate_generation_task_result(
            &result("success", path.to_str().unwrap()),
            "{\"intent\": \"image_to_video\"}",
        )
        .unwrap_err();
        let _ = std::fs::remove_file(path);
        assert!(error.contains("request requires Video"));
    }

    #[test]
    fn rejects_explicit_ca_failure_language() {
        let mut task_result = result("success", "/tmp/result.mp4");
        task_result.output.as_mut().unwrap()["pdca_trace"][2]["summary"] =
            serde_json::json!("Artifact verified, wrong type");
        let error = validate_generation_task_result(
            &task_result,
            "{\"intent\": \"image_to_video\"}",
        )
        .unwrap_err();
        assert!(error.contains("Check phase explicitly rejected"));
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
    let payload: serde_json::Value = match serde_json::from_str(&gh_event.payload) {
        Ok(v) => v,
        Err(_) => return,
    };
    let prompt_id = gh_event.task_iri.replace("iri://task/", "");

    if gh_event.event_type == "TOOL_CALL" {
        let event = &payload["event"]["ToolCall"];
        media_bus.publish(crate::execution::Event::AgentToolCall {
            prompt_id,
            tool_name: event["tool_name"].as_str().unwrap_or("unknown").to_string(),
            status: "started".to_string(),
            result_summary: String::new(),
        }).await;
        return;
    }

    if gh_event.event_type == "TOOL_RESULT" {
        let event = &payload["event"]["ToolResult"];
        let success = event["success"].as_bool().unwrap_or(false);
        let summary = event["result"].as_str().unwrap_or("");
        media_bus.publish(crate::execution::Event::AgentToolCall {
            prompt_id,
            tool_name: event["tool_name"].as_str().unwrap_or("unknown").to_string(),
            status: if success { "completed" } else { "failed" }.to_string(),
            result_summary: summary.chars().take(500).collect(),
        }).await;
        return;
    }

    if gh_event.event_type != "THOUGHT" {
        return;
    }

    // 提取 action 和 thought 字段
    let action = payload["event"]["Thought"]["action"]
        .as_str()
        .or_else(|| payload["action"].as_str())
        .unwrap_or("");
    let thought = payload["event"]["Thought"]["thought"]
        .as_str()
        .or_else(|| payload["thought"].as_str())
        .unwrap_or("");

    match action {
        "dispatch_plan" => publish_phase_start(
            media_bus, &prompt_id, "planning", "Planning Agent 正在制定原生生成方案"
        ).await,
        "plan_created" => publish_phase_complete(media_bus, &prompt_id, "planning", true).await,
        "dispatch_do" => publish_phase_start(
            media_bus, &prompt_id, "doing", "Doing Agent 正在调用 Rust 原生推理工具"
        ).await,
        "dispatch_check" => {
            publish_phase_complete(media_bus, &prompt_id, "doing", true).await;
            publish_phase_start(
                media_bus, &prompt_id, "checking", "Checking Agent 正在验证真实媒体产物"
            ).await;
        }
        "dispatch_act" => {
            publish_phase_complete(media_bus, &prompt_id, "checking", true).await;
            publish_phase_start(
                media_bus, &prompt_id, "acting", "Acting Agent 正在决定接受、重试或失败"
            ).await;
        }
        "pdca_cycle_passed" => publish_phase_complete(media_bus, &prompt_id, "acting", true).await,
        "pdca_cycle_failed" | "pdca_cycles_exhausted" => {
            publish_phase_complete(media_bus, &prompt_id, "acting", false).await
        }
        _ => {}
    }

    // 如果有 thought 内容，同时发布
    if !thought.is_empty() {
        media_bus.publish(crate::execution::Event::AgentThought {
            prompt_id: gh_event.task_iri.replace("iri://task/", ""),
            thought: thought.to_string(),
            action: action.to_string(),
        }).await;
    }
}

async fn publish_phase_start(
    media_bus: &crate::execution::EventBus,
    prompt_id: &str,
    phase: &str,
    description: &str,
) {
    media_bus.publish(crate::execution::Event::AgentPhaseStart {
        prompt_id: prompt_id.to_string(),
        phase: phase.to_string(),
        description: description.to_string(),
    }).await;
}

async fn publish_phase_complete(
    media_bus: &crate::execution::EventBus,
    prompt_id: &str,
    phase: &str,
    success: bool,
) {
    media_bus.publish(crate::execution::Event::AgentPhaseComplete {
        prompt_id: prompt_id.to_string(),
        phase: phase.to_string(),
        success,
    }).await;
}
