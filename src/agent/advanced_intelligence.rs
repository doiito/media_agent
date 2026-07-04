// gliding_horse 高级智能功能集成模块
// 整合 SkillGraph / Discovery / Evolution / KnowledgeGraph / Causal / Timeline

use std::sync::Arc;
use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use log::{info, warn, debug};

use glidinghorse::skill_graph::{
    SkillGraphStore, SkillDiscoveryEngine, SkillEvolutionEngine,
    Task5W2H, UsageRecord, EvolutionSuggestion,
    SkillGraphNode, SkillLink, SkillLinkType, LinkStrength,
    SkillGraphMeta, SkillContent, Skill5W2H, SkillRole, SkillTrigger,
    SkillContext, SkillApproach, SkillCost, StorageTier, SkillNodeType,
};
use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;
use glidinghorse::causal::types::{CausalObservation, CausalInference};
use glidinghorse::snapshots::timeline::TimelineStore;
use glidinghorse::memory::{HyperspaceStore, FallbackEmbeddingService, EmbeddingService};
use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::graph_backend::{PetgraphBackend, GraphBackend};
use glidinghorse::CoreError;
use crate::agent::workspace_monitor::{ComfyUiWorkspaceMonitor, ComfyUiWorkspaceConfig};

/// ComfyUI 智能引擎 - 整合所有 gliding_horse 高级功能
pub struct ComfyUiIntelligence {
    skill_graph: Arc<SkillGraphStore>,
    discovery: Arc<tokio::sync::RwLock<SkillDiscoveryEngine>>,
    evolution: Arc<tokio::sync::RwLock<SkillEvolutionEngine>>,
    knowledge_graph: Option<Arc<KnowledgeGraphStore>>,
    timeline: Arc<tokio::sync::RwLock<TimelineStore>>,
    workflow_history: Arc<tokio::sync::RwLock<Vec<WorkflowExecutionRecord>>>,
    /// HNSW 向量搜索引擎（语义搜索工作流历史）
    hyperspace: Option<Arc<HyperspaceStore>>,
    /// 因果推理引擎（根因分析）
    causal: Option<Arc<CausalEngine>>,
    /// 因果模型存储（记录错误观察）
    causal_store: Arc<CausalModelStore>,
    config: IntelligenceConfig,
}

#[derive(Debug, Clone)]
pub struct IntelligenceConfig {
    pub enable_evolution: bool,
    pub enable_knowledge_graph: bool,
    pub enable_timeline: bool,
    pub enable_hyperspace: bool,
    pub enable_causal: bool,
    pub max_history: usize,
    /// HyperspaceStore 持久化目录
    pub hyperspace_data_dir: String,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            enable_evolution: true,
            enable_knowledge_graph: true,
            enable_timeline: true,
            enable_hyperspace: true,
            enable_causal: true,
            max_history: 1000,
            hyperspace_data_dir: ".gliding_horse/hyperspace".to_string(),
        }
    }
}

/// 工作流执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecutionRecord {
    pub execution_id: String,
    pub user_request: String,
    pub intent: String,
    pub workflow_json: Value,
    pub success: bool,
    pub duration_ms: u64,
    pub node_count: usize,
    pub parameters: Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
}

impl WorkflowExecutionRecord {
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::new();
        let intents = ["text_to_image", "image_to_image", "video", "upscale", "inpaint"];
        for intent in &intents {
            features.push(if self.intent == *intent { 1.0 } else { 0.0 });
        }
        let params = &self.parameters;
        features.push(params.get("width").and_then(|v| v.as_f64()).unwrap_or(512.0) as f32 / 1024.0);
        features.push(params.get("height").and_then(|v| v.as_f64()).unwrap_or(512.0) as f32 / 1024.0);
        features.push(params.get("steps").and_then(|v| v.as_f64()).unwrap_or(20.0) as f32 / 50.0);
        features.push(params.get("cfg").and_then(|v| v.as_f64()).unwrap_or(7.0) as f32 / 15.0);
        features.push(params.get("denoise").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32);
        features.push(if self.success { 1.0 } else { 0.0 });
        features.push(self.duration_ms as f32 / 60000.0);
        features.push(self.node_count as f32 / 20.0);
        features
    }

    pub fn cosine_similarity(&self, other: &WorkflowExecutionRecord) -> f32 {
        let v1 = self.to_feature_vector();
        let v2 = other.to_feature_vector();
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
        if norm1 == 0.0 || norm2 == 0.0 { 0.0 } else { dot / (norm1 * norm2) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecommendation {
    pub skill_iri: String,
    pub skill_name: String,
    pub score: f32,
    pub reasons: Vec<String>,
    pub required_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureAnalysis {
    pub failed_skill: String,
    pub root_cause_skill: Option<String>,
    pub root_cause_description: String,
    pub confidence: f32,
    pub fix_suggestions: Vec<String>,
    pub propagation_path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRecommendation {
    pub parameters: Value,
    pub reasoning: String,
    pub confidence: f32,
    pub similar_success_count: usize,
}

#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub skill_iri: String,
    pub name: String,
    pub description: String,
    pub what: String,
    pub why: String,
    pub category: String,
    pub tags: Vec<String>,
    pub links: Vec<SkillLinkDef>,
}

#[derive(Debug, Clone)]
pub struct SkillLinkDef {
    pub link_type: SkillLinkType,
    pub target_iri: String,
    pub strength: LinkStrength,
    pub description: String,
}

impl ComfyUiIntelligence {
    pub fn new(config: IntelligenceConfig) -> Result<Self, String> {
        info!("Initializing ComfyUiIntelligence");

        let skill_graph = Arc::new(SkillGraphStore::new());
        let discovery = Arc::new(tokio::sync::RwLock::new(
            SkillDiscoveryEngine::new(skill_graph.clone())
        ));
        let evolution = Arc::new(tokio::sync::RwLock::new(
            SkillEvolutionEngine::new(skill_graph.clone())
        ));

        let knowledge_graph = if config.enable_knowledge_graph {
            match KnowledgeGraphStore::new() {
                Ok(store) => Some(Arc::new(store)),
                Err(e) => {
                    warn!("Failed to init KnowledgeGraphStore: {}", e);
                    None
                }
            }
        } else { None };

        let timeline = Arc::new(tokio::sync::RwLock::new(TimelineStore::default()));
        let workflow_history = Arc::new(tokio::sync::RwLock::new(Vec::new()));

        let hyperspace = if config.enable_hyperspace {
            match Self::init_hyperspace(&config.hyperspace_data_dir) {
                Ok(store) => Some(Arc::new(store)),
                Err(e) => {
                    warn!("Failed to init HyperspaceStore: {}", e);
                    None
                }
            }
        } else { None };

        let causal_store = Arc::new(CausalModelStore::new());
        let causal = if config.enable_causal {
            let backend: Arc<dyn GraphBackend> = Arc::new(PetgraphBackend::new(skill_graph.clone()));
            Some(Arc::new(CausalEngine::new(causal_store.clone(), backend)))
        } else { None };

        let intelligence = Self {
            skill_graph, discovery, evolution,
            knowledge_graph, timeline, workflow_history,
            hyperspace, causal, causal_store, config,
        };
        intelligence.bootstrap_comfyui_skills();
        intelligence.load_skills_from_dir("skills");
        Ok(intelligence)
    }

    fn init_hyperspace(data_dir: &str) -> Result<HyperspaceStore, String> {
        let path = std::path::Path::new(data_dir);
        std::fs::create_dir_all(path)
            .map_err(|e| format!("Failed to create hyperspace dir: {}", e))?;
        let embed: Arc<dyn EmbeddingService> = Arc::new(FallbackEmbeddingService::with_dimension(128));
        HyperspaceStore::open(path, embed)
            .map_err(|e| format!("HyperspaceStore open: {:?}", e))
    }

    /// 从指定目录加载 JSON-LD 技能定义文件
    pub fn load_skills_from_dir(&self, dir: &str) {
        let path = std::path::Path::new(dir);
        if !path.exists() || !path.is_dir() {
            debug!("Skills directory not found: {}", dir);
            return;
        }

        let entries = match std::fs::read_dir(path) {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to read skills dir {}: {}", dir, e);
                return;
            }
        };

        let mut loaded = 0;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonld") {
                continue;
            }
            if let Some(name) = p.file_stem().and_then(|s| s.to_str()) {
                if name == "comfyui_ontology" {
                    continue;
                }
            }
            if let Ok(content) = std::fs::read_to_string(&p) {
                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                    if let Some(def) = self.parse_skill_jsonld(&json) {
                        let already = self.skill_graph.get_skill(&def.skill_iri).is_some()
                            || self.skill_graph.list_all_skills().iter()
                                .any(|s| s.name == def.name);
                        if already {
                            debug!("Skill {} already registered, skipping", def.skill_iri);
                            continue;
                        }
                        if self.register_skill(def).is_ok() {
                            loaded += 1;
                        }
                    }
                }
            }
        }
        info!("Loaded {} skills from {}", loaded, dir);
    }

    fn parse_skill_jsonld(&self, json: &Value) -> Option<SkillDefinition> {
        let skill_iri = json.get("@id")?.as_str()?.to_string();
        let name = json.get("schema:name").and_then(|v| v.as_str())
            .unwrap_or("unnamed").to_string();
        let description = json.get("schema:description").and_then(|v| v.as_str())
            .unwrap_or("").to_string();
        let category = json.get("skill:category").and_then(|v| v.as_str())
            .unwrap_or("general").to_string();
        let tags: Vec<String> = json.get("skill:tags")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let what = format!("{}", name);
        let why = description.clone();

        Some(SkillDefinition {
            skill_iri,
            name,
            description,
            what,
            why,
            category,
            tags,
            links: vec![],
        })
    }

    fn bootstrap_comfyui_skills(&self) {
        info!("Bootstrapping ComfyUI skills...");
        for def in self.get_builtin_skill_definitions() {
            if let Err(e) = self.register_skill(def) {
                warn!("Failed to register skill: {}", e);
            }
        }
        info!("ComfyUI skills bootstrap completed");
    }

    fn get_builtin_skill_definitions(&self) -> Vec<SkillDefinition> {
        vec![
            SkillDefinition {
                skill_iri: "comfyui:text_to_image".into(),
                name: "文生图".into(),
                description: "从文本描述生成图片".into(),
                what: "generate image from text".into(),
                why: "create visual content".into(),
                category: "generation".into(),
                tags: vec!["t2i".into(), "diffusion".into()],
                links: vec![
                    SkillLinkDef {
                        link_type: SkillLinkType::Related,
                        target_iri: "comfyui:sampler_euler".into(),
                        strength: LinkStrength::Recommended,
                        description: "euler 适合大多数场景".into(),
                    },
                    SkillLinkDef {
                        link_type: SkillLinkType::Related,
                        target_iri: "comfyui:sampler_dpmpp_2m".into(),
                        strength: LinkStrength::Recommended,
                        description: "dpmpp_2m 高质量".into(),
                    },
                ],
            },
            SkillDefinition {
                skill_iri: "comfyui:image_to_image".into(),
                name: "图生图".into(),
                description: "基于参考图片生成新图片".into(),
                what: "generate image from image".into(),
                why: "transform existing image".into(),
                category: "generation".into(),
                tags: vec!["i2i".into(), "transform".into()],
                links: vec![],
            },
            SkillDefinition {
                skill_iri: "comfyui:video_generation".into(),
                name: "视频生成".into(),
                description: "从图片或文本生成短视频".into(),
                what: "generate video".into(),
                why: "create animated content".into(),
                category: "video".into(),
                tags: vec!["video".into(), "svd".into()],
                links: vec![
                    SkillLinkDef {
                        link_type: SkillLinkType::Prerequisite,
                        target_iri: "comfyui:text_to_image".into(),
                        strength: LinkStrength::Required,
                        description: "需要先生成基础图片".into(),
                    },
                ],
            },
            SkillDefinition {
                skill_iri: "comfyui:upscale".into(),
                name: "图片放大".into(),
                description: "使用 AI 模型放大图片".into(),
                what: "upscale image".into(),
                why: "increase resolution".into(),
                category: "postprocess".into(),
                tags: vec!["upscale".into(), "hd".into()],
                links: vec![],
            },
            SkillDefinition {
                skill_iri: "comfyui:sampler_euler".into(),
                name: "Euler 采样器".into(),
                description: "快速通用的采样器".into(),
                what: "euler sampler".into(),
                why: "fast sampling".into(),
                category: "sampler".into(),
                tags: vec!["sampler".into(), "fast".into()],
                links: vec![],
            },
            SkillDefinition {
                skill_iri: "comfyui:sampler_dpmpp_2m".into(),
                name: "DPM++ 2M 采样器".into(),
                description: "高质量采样器".into(),
                what: "dpmpp_2m sampler".into(),
                why: "high quality sampling".into(),
                category: "sampler".into(),
                tags: vec!["sampler".into(), "quality".into()],
                links: vec![],
            },
        ]
    }

    pub fn register_skill(&self, def: SkillDefinition) -> Result<(), String> {
        let mut node = SkillGraphNode::new(&def.skill_iri, &def.name, &def.description);
        node.tags = def.tags.clone();
        node.w2h = Skill5W2H {
            what: def.what.clone(),
            why: def.why.clone(),
            who: SkillRole {
                role_name: "DA".to_string(),
                required_agent_role: Some("DA".to_string()),
            },
            when: SkillTrigger::default(),
            where_: SkillContext::default(),
            how: SkillApproach {
                approach: def.category.clone(),
                plan_iri: None,
            },
            how_much: SkillCost::default(),
        };

        for link in &def.links {
            node = node.with_link(SkillLink {
                link_type: link.link_type,
                target_iri: link.target_iri.clone(),
                strength: link.strength,
                description: link.description.clone(),
            });
        }

        self.skill_graph.register_skill(node).map_err(|e| format!("{:?}", e))?;
        Ok(())
    }

    pub async fn discover_skills(&self, user_request: &str, intent: &str) -> Vec<SkillRecommendation> {
        let task = Task5W2H::new(user_request, intent)
            .with_agent_role("DA")
            .with_phase("execution");

        let discovery = self.discovery.read().await;
        discovery.discover_for_task(&task).into_iter().map(|m| {
            SkillRecommendation {
                skill_iri: m.skill.skill_iri.clone(),
                skill_name: m.skill.name.clone(),
                score: m.relevance_score,
                reasons: m.match_reasons,
                required_dependencies: m.required_dependencies,
            }
        }).collect()
    }

    pub async fn record_execution(&self, record: WorkflowExecutionRecord) {
        let mut history = self.workflow_history.write().await;
        history.push(record.clone());
        if history.len() > self.config.max_history {
            let excess = history.len() - self.config.max_history;
            history.drain(0..excess);
        }
        drop(history);

        if self.config.enable_evolution {
            let usage = UsageRecord::new(
                &format!("comfyui:{}", record.intent),
                &record.execution_id,
                "DA",
                record.success,
            )
            .with_duration((record.duration_ms / 1000) as u32)
            .with_context_tag(&record.intent);

            let mut evolution = self.evolution.write().await;
            if let Err(e) = evolution.record_usage(usage) {
                warn!("Failed to record usage: {:?}", e);
            }
        }

        // 同步写入 HyperspaceStore 进行语义索引
        if let Some(ref hyperspace) = self.hyperspace {
            let iri = format!("iri://workflow/{}", record.execution_id);
            let text = format!(
                "intent={} request={} success={} duration_ms={} node_count={} params={}",
                record.intent, record.user_request, record.success,
                record.duration_ms, record.node_count, record.parameters
            );
            let tags = vec![record.intent.clone(), format!("success:{}", record.success)];
            let importance = if record.success { 0.8 } else { 0.5 };
            let jsonld_types = vec!["comfyui:WorkflowExecution".to_string()];
            if let Err(e) = hyperspace.upsert_with_metadata(
                &iri, &text, &tags,
                Some(importance),
                Some(&jsonld_types),
                Some("comfyui"),
            ).await {
                warn!("Failed to upsert workflow to hyperspace: {:?}", e);
            }
        }

        // 失败时记录 CausalObservation 用于根因分析
        if !record.success {
            if let Some(ref error) = record.error {
                let obs = CausalObservation::new(
                    &record.execution_id,
                    &format!("comfyui:{}", record.intent),
                    "execution_error",
                    error,
                ).with_context("intent", &record.intent)
                 .with_context("duration_ms", &record.duration_ms.to_string());
                self.causal_store.record_observation(&obs);
            }
        }
    }

    pub fn analyze_failure(&self, failed_skill: &str, error: &str) -> FailureAnalysis {
        // 优先使用 CausalEngine 贝叶斯推理（如果有历史观察数据）
        if let Some(ref causal) = self.causal {
            let obs = CausalObservation::new(
                &format!("analysis-{}", chrono::Utc::now().timestamp_millis()),
                failed_skill,
                "execution_error",
                error,
            ).with_context("analyzed_at", &chrono::Utc::now().to_rfc3339());

            let inferences = causal.infer_root_cause(&[obs.clone()], 3);
            if let Some(top) = inferences.first() {
                let path: Vec<String> = top.propagation_paths.iter()
                    .flat_map(|p| p.hops.iter().map(|h| h.skill_iri.clone()))
                    .chain(std::iter::once(failed_skill.to_string()))
                    .collect();
                let mut suggestions = self.heuristic_fix_suggestions(error);
                if !top.alternative_causes.is_empty() {
                    suggestions.push(format!(
                        "其他可能根因: {}",
                        top.alternative_causes.iter()
                            .map(|(iri, conf)| format!("{} ({:.2})", iri, conf))
                            .collect::<Vec<_>>().join(", ")
                    ));
                }
                return FailureAnalysis {
                    failed_skill: failed_skill.to_string(),
                    root_cause_skill: Some(top.root_cause_iri.clone()),
                    root_cause_description: format!(
                        "CausalEngine 推理置信度 {:.2}，解释 {}/{} 个观察",
                        top.confidence, top.observations_explained, top.total_observations
                    ),
                    confidence: top.confidence,
                    fix_suggestions: suggestions,
                    propagation_path: path,
                };
            }
        }
        self.heuristic_failure_analysis(failed_skill, error)
    }

    /// 语义搜索工作流（使用 HyperspaceStore HNSW 引擎）
    pub async fn semantic_search_workflows(&self, query: &str, top_k: u64) -> Vec<(String, f32)> {
        if let Some(ref hyperspace) = self.hyperspace {
            match hyperspace.search(query, top_k).await {
                Ok(results) => results.into_iter()
                    .map(|entry| (entry.iri, entry.score))
                    .collect(),
                Err(e) => {
                    warn!("Semantic search failed: {:?}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    }

    fn heuristic_failure_analysis(&self, failed_skill: &str, error: &str) -> FailureAnalysis {
        let fix_suggestions = self.heuristic_fix_suggestions(error);
        let root_cause = self.heuristic_root_cause(error);

        FailureAnalysis {
            failed_skill: failed_skill.to_string(),
            root_cause_skill: Some(root_cause.clone()),
            root_cause_description: format!("Heuristic analysis: {}", error),
            confidence: 0.6,
            fix_suggestions,
            propagation_path: vec![root_cause, failed_skill.to_string()],
        }
    }

    fn heuristic_fix_suggestions(&self, error: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        if error.contains("model") || error.contains("checkpoint") {
            suggestions.push("检查模型文件是否存在".to_string());
            suggestions.push("验证模型架构是否匹配（SD1.5 vs SDXL）".to_string());
        } else if error.contains("type") || error.contains("mismatch") {
            suggestions.push("检查节点连接类型是否兼容".to_string());
            suggestions.push("使用 validate_workflow 工具验证连接".to_string());
        } else if error.contains("memory") || error.contains("OOM") {
            suggestions.push("降低分辨率或 batch_size".to_string());
            suggestions.push("启用 CPU offload".to_string());
        } else if error.contains("timeout") {
            suggestions.push("减少采样步数".to_string());
            suggestions.push("检查后端进程是否正常运行".to_string());
        } else {
            suggestions.push("查看详细日志获取更多信息".to_string());
        }
        suggestions
    }

    fn heuristic_root_cause(&self, error: &str) -> String {
        if error.contains("model") || error.contains("checkpoint") {
            "model_loading".to_string()
        } else if error.contains("type") || error.contains("mismatch") {
            "type_mismatch".to_string()
        } else if error.contains("memory") || error.contains("OOM") {
            "resource_exhaustion".to_string()
        } else if error.contains("timeout") {
            "timeout".to_string()
        } else {
            "unknown".to_string()
        }
    }

    pub async fn recommend_parameters(&self, intent: &str, user_request: &str) -> ParameterRecommendation {
        let history = self.workflow_history.read().await;
        let similar: Vec<&WorkflowExecutionRecord> = history.iter()
            .filter(|r| r.intent == intent && r.success)
            .collect();

        if similar.is_empty() {
            return ParameterRecommendation {
                parameters: self.default_parameters(intent),
                reasoning: "基于默认规则的推荐（无历史数据）".to_string(),
                confidence: 0.5,
                similar_success_count: 0,
            };
        }

        let mut width_sum = 0.0; let mut height_sum = 0.0;
        let mut steps_sum = 0.0; let mut cfg_sum = 0.0;
        let mut count = 0;

        for record in &similar {
            if let Some(params) = record.parameters.as_object() {
                width_sum += params.get("width").and_then(|v| v.as_f64()).unwrap_or(512.0);
                height_sum += params.get("height").and_then(|v| v.as_f64()).unwrap_or(512.0);
                steps_sum += params.get("steps").and_then(|v| v.as_f64()).unwrap_or(20.0);
                cfg_sum += params.get("cfg").and_then(|v| v.as_f64()).unwrap_or(7.0);
                count += 1;
            }
        }

        if count > 0 {
            let avg_width = (width_sum / count as f64) as u32;
            let avg_height = (height_sum / count as f64) as u32;
            let avg_steps = (steps_sum / count as f64) as u32;
            let avg_cfg = (cfg_sum / count as f64) as f32;

            let query_record = WorkflowExecutionRecord {
                execution_id: "query".to_string(),
                user_request: user_request.to_string(),
                intent: intent.to_string(),
                workflow_json: json!({}),
                success: true,
                duration_ms: 0,
                node_count: 0,
                parameters: json!({}),
                timestamp: chrono::Utc::now(),
                error: None,
            };

            let mut best_score = 0.0;
            let mut best_sampler = "euler".to_string();
            for record in &similar {
                let score = query_record.cosine_similarity(record);
                if score > best_score {
                    best_score = score;
                    if let Some(s) = record.parameters.get("sampler_name").and_then(|v| v.as_str()) {
                        best_sampler = s.to_string();
                    }
                }
            }

            ParameterRecommendation {
                parameters: json!({
                    "width": avg_width,
                    "height": avg_height,
                    "steps": avg_steps,
                    "cfg": avg_cfg,
                    "sampler_name": best_sampler,
                    "scheduler": "normal"
                }),
                reasoning: format!(
                    "基于 {} 个历史成功案例推荐，最相似案例相似度 {:.2}",
                    count, best_score
                ),
                confidence: (0.6 + best_score * 0.4).min(1.0),
                similar_success_count: count,
            }
        } else {
            ParameterRecommendation {
                parameters: self.default_parameters(intent),
                reasoning: "基于默认规则的推荐".to_string(),
                confidence: 0.5,
                similar_success_count: 0,
            }
        }
    }

    fn default_parameters(&self, intent: &str) -> Value {
        match intent {
            "text_to_image" => json!({
                "width": 768, "height": 768, "steps": 25, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0
            }),
            "image_to_image" => json!({
                "width": 768, "height": 768, "steps": 25, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 0.6
            }),
            "video" => json!({
                "width": 512, "height": 512, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal",
                "frames": 14, "fps": 6
            }),
            "upscale" => json!({
                "upscale_factor": 4, "steps": 15, "cfg": 5.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras"
            }),
            _ => json!({
                "width": 512, "height": 512, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal"
            }),
        }
    }

    pub async fn find_similar_workflows(&self, query: &WorkflowExecutionRecord, top_k: usize)
        -> Vec<(WorkflowExecutionRecord, f32)>
    {
        let history = self.workflow_history.read().await;
        let mut scored: Vec<(WorkflowExecutionRecord, f32)> = history.iter()
            .map(|r| (r.clone(), query.cosine_similarity(r)))
            .filter(|(_, score)| *score > 0.1)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub async fn get_skill_stats(&self) -> Value {
        let history = self.workflow_history.read().await;
        let total = history.len();
        let success_count = history.iter().filter(|r| r.success).count();
        let failure_count = total - success_count;
        let success_rate = if total > 0 { success_count as f32 / total as f32 } else { 0.0 };
        let avg_duration = if total > 0 {
            history.iter().map(|r| r.duration_ms).sum::<u64>() / total as u64
        } else { 0 };

        json!({
            "total_executions": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
            "intents": history.iter().map(|r| r.intent.clone()).collect::<std::collections::HashSet<_>>().len(),
        })
    }

    pub fn knowledge_graph(&self) -> Option<&Arc<KnowledgeGraphStore>> {
        self.knowledge_graph.as_ref()
    }

    pub fn skill_graph(&self) -> &Arc<SkillGraphStore> {
        &self.skill_graph
    }

    pub fn hyperspace(&self) -> Option<&Arc<HyperspaceStore>> {
        self.hyperspace.as_ref()
    }

    pub fn causal(&self) -> Option<&Arc<CausalEngine>> {
        self.causal.as_ref()
    }

    /// 自动技能发现：从历史成功执行中识别常见参数模式，
    /// 为高频模式生成参数化技能并注册到技能图。
    ///
    /// 算法：
    /// 1. 按 intent 分组历史成功记录
    /// 2. 在每个 intent 内做简单聚类（基于余弦相似度）
    /// 3. 为规模 >= min_cluster_size 的聚类生成"参数化技能"
    /// 4. 注册新技能到图谱（若 IRI 不存在）
    pub async fn discover_skill_patterns(&self, min_cluster_size: usize, similarity_threshold: f32)
        -> Vec<String>
    {
        let history = self.workflow_history.read().await;
        let success_records: Vec<&WorkflowExecutionRecord> = history.iter()
            .filter(|r| r.success)
            .collect();

        if success_records.len() < min_cluster_size {
            return Vec::new();
        }

        // 按 intent 分组
        let mut groups: HashMap<String, Vec<&WorkflowExecutionRecord>> = HashMap::new();
        for record in &success_records {
            groups.entry(record.intent.clone()).or_default().push(*record);
        }

        let mut discovered = Vec::new();
        for (intent, records) in groups {
            if records.len() < min_cluster_size {
                continue;
            }

            // 简单聚类：贪心法，每个记录与已有聚类中心比较
            let mut clusters: Vec<(WorkflowExecutionRecord, Vec<&WorkflowExecutionRecord>)> = Vec::new();
            for record in &records {
                let mut best_cluster: Option<usize> = None;
                let mut best_score = 0.0;
                for (i, (center, _)) in clusters.iter().enumerate() {
                    let score = record.cosine_similarity(center);
                    if score > best_score {
                        best_score = score;
                        best_cluster = Some(i);
                    }
                }
                if best_score >= similarity_threshold {
                    if let Some(i) = best_cluster {
                        clusters[i].1.push(*record);
                    }
                } else {
                    clusters.push(((*record).clone(), vec![*record]));
                }
            }

            // 为规模达标的聚类生成技能
            for (center, members) in clusters {
                if members.len() < min_cluster_size {
                    continue;
                }
                let skill_iri = format!("comfyui:auto:{}:{}{}", intent, members.len(),
                    center.parameters.get("sampler_name")
                        .and_then(|v| v.as_str())
                        .map(|s| format!(":{}", s))
                        .unwrap_or_default());

                if self.skill_graph.get_skill(&skill_iri).is_some() {
                    continue;
                }

                let def = SkillDefinition {
                    skill_iri: skill_iri.clone(),
                    name: format!("自动发现-{}-{}样本", intent, members.len()),
                    description: format!(
                        "从 {} 个成功 {} 工作流中识别的常见模式",
                        members.len(), intent
                    ),
                    what: format!("auto-discovered {} pattern", intent),
                    why: format!("frequent successful {} workflow", intent),
                    category: format!("auto_discovered:{}", intent),
                    tags: vec![intent.clone(), "auto_discovered".to_string()],
                    links: vec![SkillLinkDef {
                        link_type: SkillLinkType::Related,
                        target_iri: format!("comfyui:{}", intent),
                        strength: LinkStrength::Recommended,
                        description: format!("派生自 {}", intent),
                    }],
                };

                if self.register_skill(def).is_ok() {
                    info!("Auto-discovered skill: {}", skill_iri);
                    discovered.push(skill_iri);
                }
            }
        }

        discovered
    }

    /// 工作区监控：重新扫描工作区并返回统计信息
    pub fn scan_workspace() -> Result<Value, String> {
        let config = ComfyUiWorkspaceConfig {
            project_root: PathBuf::from("."),
            watch_enabled: false,
            db_path: None,
        };
        let monitor = ComfyUiWorkspaceMonitor::new(config)?;
        Ok(monitor.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_intelligence_creation() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default());
        assert!(intelligence.is_ok());
    }

    #[tokio::test]
    async fn test_skill_bootstrap() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let skill = intelligence.skill_graph.get_skill("comfyui:text_to_image");
        assert!(skill.is_some());
        assert_eq!(skill.unwrap().name, "文生图");
    }

    #[tokio::test]
    async fn test_record_execution() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let record = WorkflowExecutionRecord {
            execution_id: "test-1".to_string(),
            user_request: "画一只猫".to_string(),
            intent: "text_to_image".to_string(),
            workflow_json: json!({}),
            success: true,
            duration_ms: 5000,
            node_count: 7,
            parameters: json!({"width": 768, "height": 768, "steps": 25}),
            timestamp: chrono::Utc::now(),
            error: None,
        };
        intelligence.record_execution(record).await;
        let stats = intelligence.get_skill_stats().await;
        assert_eq!(stats["total_executions"], 1);
    }

    #[test]
    fn test_failure_analysis() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let analysis = intelligence.analyze_failure("comfyui:text_to_image", "model not found");
        assert!(analysis.root_cause_skill.is_some());
        assert!(!analysis.fix_suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_parameter_recommendation_no_history() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        let rec = intelligence.recommend_parameters("text_to_image", "画一只猫").await;
        assert!(rec.parameters.is_object());
        assert_eq!(rec.similar_success_count, 0);
    }

    #[tokio::test]
    async fn test_parameter_recommendation_with_history() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        for i in 0..5 {
            intelligence.record_execution(WorkflowExecutionRecord {
                execution_id: format!("test-{}", i),
                user_request: "画猫".to_string(),
                intent: "text_to_image".to_string(),
                workflow_json: json!({}),
                success: true,
                duration_ms: 5000,
                node_count: 7,
                parameters: json!({"width": 1024, "height": 1024, "steps": 30, "cfg": 8.0, "sampler_name": "dpmpp_2m"}),
                timestamp: chrono::Utc::now(),
                error: None,
            }).await;
        }
        let rec = intelligence.recommend_parameters("text_to_image", "画猫").await;
        assert!(rec.similar_success_count >= 5);
        assert!(rec.confidence > 0.5);
        assert_eq!(rec.parameters["width"], 1024);
    }

    #[test]
    fn test_cosine_similarity() {
        let r1 = WorkflowExecutionRecord {
            execution_id: "1".to_string(),
            user_request: "cat".to_string(),
            intent: "text_to_image".to_string(),
            workflow_json: json!({}),
            success: true, duration_ms: 5000, node_count: 7,
            parameters: json!({"width": 768, "height": 768, "steps": 25, "cfg": 7.0, "denoise": 1.0}),
            timestamp: chrono::Utc::now(),
            error: None,
        };
        let r2 = r1.clone();
        let similarity = r1.cosine_similarity(&r2);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_find_similar_workflows() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        for i in 0..3 {
            intelligence.record_execution(WorkflowExecutionRecord {
                execution_id: format!("sim-{}", i),
                user_request: "画猫".to_string(),
                intent: "text_to_image".to_string(),
                workflow_json: json!({}),
                success: true, duration_ms: 5000, node_count: 7,
                parameters: json!({"width": 768, "height": 768, "steps": 25, "cfg": 7.0, "denoise": 1.0}),
                timestamp: chrono::Utc::now(),
                error: None,
            }).await;
        }
        let query = WorkflowExecutionRecord {
            execution_id: "query".to_string(),
            user_request: "画狗".to_string(),
            intent: "text_to_image".to_string(),
            workflow_json: json!({}),
            success: true, duration_ms: 5000, node_count: 7,
            parameters: json!({"width": 768, "height": 768, "steps": 25, "cfg": 7.0, "denoise": 1.0}),
            timestamp: chrono::Utc::now(),
            error: None,
        };
        let similar = intelligence.find_similar_workflows(&query, 5).await;
        assert!(!similar.is_empty());
    }

    #[tokio::test]
    async fn test_auto_discover_skill_patterns() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        for i in 0..10 {
            intelligence.record_execution(WorkflowExecutionRecord {
                execution_id: format!("auto-{}", i),
                user_request: "画猫".to_string(),
                intent: "text_to_image".to_string(),
                workflow_json: json!({}),
                success: true, duration_ms: 5000, node_count: 7,
                parameters: json!({
                    "width": 1024, "height": 1024, "steps": 30, "cfg": 8.0,
                    "sampler_name": "dpmpp_2m", "denoise": 1.0
                }),
                timestamp: chrono::Utc::now(),
                error: None,
            }).await;
        }
        let discovered = intelligence.discover_skill_patterns(3, 0.85).await;
        assert!(!discovered.is_empty(), "Should discover at least one pattern");
        for iri in &discovered {
            assert!(intelligence.skill_graph.get_skill(iri).is_some());
        }
    }

    #[tokio::test]
    async fn test_semantic_search_workflows() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        intelligence.record_execution(WorkflowExecutionRecord {
            execution_id: "semantic-1".to_string(),
            user_request: "画一只猫".to_string(),
            intent: "text_to_image".to_string(),
            workflow_json: json!({}),
            success: true, duration_ms: 5000, node_count: 7,
            parameters: json!({"width": 768, "height": 768, "steps": 25, "cfg": 7.0}),
            timestamp: chrono::Utc::now(),
            error: None,
        }).await;
        // 即使 hyperspace 未启用也能调用（返回空）
        let _results = intelligence.semantic_search_workflows("画猫", 5).await;
    }

    #[test]
    fn test_causal_engine_initialization() {
        let intelligence = ComfyUiIntelligence::new(IntelligenceConfig::default()).unwrap();
        // 因果引擎应已初始化
        assert!(intelligence.causal().is_some(), "CausalEngine should be initialized");
    }
}
