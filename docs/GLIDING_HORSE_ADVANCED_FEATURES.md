# gliding_horse 高级功能集成分析报告

## 1. 当前使用情况分析

### 本系统已使用的 gliding_horse 功能

| 模块 | 使用情况 | 文件 |
|------|---------|------|
| SupervisorAgent | ✅ 使用 | agent/engine.rs |
| AgentRunner | ✅ 使用 | agent/engine.rs, workflow.rs |
| EventBus | ✅ 使用 | agent/engine.rs |
| L0Store | ✅ 使用 | agent/memory.rs |
| Blackboard | ✅ 使用 | agent/memory.rs |
| ProjectionEngine | ✅ 使用 | agent/memory.rs |
| MemoryManager | ✅ 使用 | agent/memory.rs |
| TemplateEngine | ✅ 使用 | agent/engine.rs |
| ToolExecutor | ✅ 使用 | agent/tools.rs |
| SkillRegistry | ✅ 使用 | agent/skills.rs |
| UnifiedGateway | ✅ 使用 | agent/context.rs |

### 本系统缺失的高级功能

| 高级功能 | gliding_horse 模块 | 价值 | 当前状态 |
|---------|-------------------|------|---------|
| **Skill Graph 自进化** | skill_graph/evolution.rs | ❌ 未使用 |
| **Skill Discovery Engine** | skill_graph/discovery.rs | ❌ 未使用 |
| **HyperspaceStore 向量搜索** | memory/hyperspace_store.rs | ❌ 未使用 |
| **Knowledge Graph Store** | knowledge_graph/store.rs | ❌ 未使用 |
| **Causal Engine 因果分析** | causal/engine.rs | ❌ 未使用 |
| **Timeline Store 时序记录** | snapshots/timeline.rs | ❌ 未使用 |
| **Workspace Monitor** | tools/workspace_monitor.rs | ❌ 未使用 |
| **Feature Extractor (GNN)** | graph_features/features.rs | ❌ 未使用 |

---

## 2. 高级功能详解

### 2.1 Skill Graph 自进化系统

**核心概念**：
- Skill Graph 是一个**认知网络**，记录技能之间的关系
- 支持 **6 种链接类型**：prerequisite, composed_of, variant, deprecated_by, conflicts_with, suggests
- **自进化**：根据使用记录自动调整图谱结构

**关键模块**：
```rust
// evolution.rs
pub struct SkillEvolutionEngine {
    graph_store: Arc<SkillGraphStore>,
    usage_history: Vec<UsageRecord>,
    pending_suggestions: Vec<EvolutionSuggestion>,
    causal_model: SkillCausalModel,
}

// 自动进化建议类型
pub enum EvolutionSuggestionType {
    AddLink,              // 添加技能依赖
    UpdateSuccessRate,    // 更新成功率
    CreateFragment,       // 创建技能片段
    Deprecate,            // 废弃低效技能
    Merge,                // 合并相似技能
    Split,                // 分拆复杂技能
}
```

**对 ComfyUI 的价值**：
- 记录工作流节点组合的成功率
- 自动发现最优节点连接模式
- 当某个采样器+调度器组合成功率高时，自动添加 suggests 链接
- 当某个 ControlNet 与特定模型不兼容时，自动添加 conflicts_with 链接

### 2.2 Skill Discovery Engine（技能发现引擎）

**核心概念**：
- 基于 **5W2H 本体** 匹配技能
- 支持 **向量语义搜索**（通过 HyperspaceStore）
- BFS 路径发现 + 组合树构建

**关键模块**：
```rust
// discovery.rs
pub struct Task5W2H {
    pub what: String,         // 任务内容
    pub why: String,          // 任务目标
    pub who: Option<String>,  // Agent 角色
    pub when_phase: Option<String>, // 执行阶段
    pub where_context: Option<String>, // 上下文
    pub how_approach: Option<String>, // 方法
    pub constraints: Vec<String>, // 约束条件
}

pub struct SkillDiscoveryEngine {
    graph_store: Arc<SkillGraphStore>,
    vector_store: Option<Arc<HyperspaceStore>>,
}

// 核心方法
fn discover_for_task(&self, task: &Task5W2H) -> Vec<SkillMatch>;
fn suggest_links(&self, skill_iri: &str) -> Vec<String>;
fn check_conflicts(&self, skills: &[String]) -> Vec<SkillConflict>;
```

**对 ComfyUI 的价值**：
- 用户说 "画一只可爱的猫咪"，SkillDiscovery 自动发现最优技能组合：
  - `text_to_image_base` (What 匹配)
  - `cute_style_lora` (What 包含 "可爱")
  - `euler_sampler` (How_approach 推荐)
- 自动检测技能冲突：SDXL 模型 + SD1.5 LoRA = conflicts_with

### 2.3 HyperspaceStore（向量引擎）

**核心概念**：
- 嵌入式向量存储，支持 HNSW ANN 搜索
- 支持多种度量空间：Poincaré、Cosine、Euclidean、Lorentz
- 混合搜索：文本 × 结构

**关键模块**：
```rust
// hyperspace_store.rs
pub struct HyperspaceStore {
    // HNSW 近似最近邻
    // WAL 预写日志
    // RoaringBitmap 元数据索引
}

// 语义搜索
fn search_similar(&self, query_embedding: &[f32], k: usize) -> Vec<SearchResult>;
```

**对 ComfyUI 的价值**：
- 存储工作流模板的向量表示
- 用户描述相似意图时，快速找到相似的成功工作流
- 存储提示词优化历史的向量，推荐最优提示词模板

### 2.4 Knowledge Graph Store（知识图谱）

**核心概念**：
- 基于 Oxigraph RDF 存储
- 支持 SPARQL 1.1 查询
- JSON-LD 数据格式

**对 ComfyUI 的价值**：
- 存储模型元数据（架构、格式、兼容性）
- 存储节点规格（输入输出类型）
- SPARQL 查询："找出所有支持 SDXL 的 LoRA"

### 2.5 Causal Engine（因果引擎）

**核心概念**：
- 贝叶斯因果推理
- 分析失败根因

**对 ComfyUI 的价值**：
- 当工作流执行失败时，自动分析根因：
  - "KSampler 失败是因为 model 输入类型不匹配"
  - "VAEDecode 失败是因为 vae 与 checkpoint 不兼容"
- 提供修复建议

### 2.6 Workspace Monitor（工作区监控）

**核心概念**：
- 实时文件系统感知
- 10 种事件触发器
- 自动触发技能调用

**对 ComfyUI 的价值**：
- 监控 models/ 目录：新模型下载后自动更新索引
- 监控 outputs/ 目录：生成完成后自动触发后处理
- 监控 workflows/ 目录：新模板添加后自动注册技能

---

## 3. 集成设计方案

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Agent System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ PA (规划)    │  │ DA (执行)    │  │ CA (检查)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
├─────────────────────────────────────────────────────────────┤
│                     AgentEngine                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              gliding_horse 组件集成层                  │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │  │
│  │  │SkillGraph│ │Hyperspace│ │Knowledge│ │ Causal │      │  │
│  │  │  Store  │ │  Store  │ │ Graph  │ │ Engine │      │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │  │
│  │  │Evolution│ │Discovery│ │ Timeline│ │Workspace│      │  │
│  │  │ Engine │ │ Engine │ │  Store │ │ Monitor│      │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     ComfyUI Backend                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ sd.cpp   │  │ llama.cpp│  │ Preview  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ComfyUI 技能图谱设计

```json
{
  "@context": "https://comfyui.ai/skills",
  "@id": "skill:text_to_image_base",
  "@type": "Skill",
  "name": "基础文生图",
  "description": "从文本描述生成图片",
  "w2h": {
    "what": "generate image from text",
    "why": "create visual content",
    "how_approach": "diffusion sampling"
  },
  "nodes": ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode"],
  "success_rate": 0.92,
  "avg_tokens": 1500,
  "links": [
    {"type": "prerequisite", "target": "skill:load_checkpoint"},
    {"type": "composed_of", "target": "skill:encode_prompt"},
    {"type": "composed_of", "target": "skill:sample_latent"},
    {"type": "suggests", "target": "skill:euler_sampler", "weight": 0.85},
    {"type": "suggests", "target": "skill:dpmpp_2m_sampler", "weight": 0.90}
  ]
}
```

### 3.3 技能自进化流程

```
用户请求 → 执行工作流 → 记录 UsageRecord
                           ↓
                    SkillEvolutionEngine
                           ↓
           ┌───────────────────────────────┐
           │ analyze_usage_patterns()      │
           │ - 计算成功率                   │
           │ - 分析失败原因                 │
           │ - 发现技能组合模式             │
           └───────────────────────────────┘
                           ↓
           ┌───────────────────────────────┐
           │ generate_suggestions()        │
           │ - AddLink: 添加优化建议        │
           │ - Deprecate: 废弃低效组合      │
           │ - Merge: 合并相似工作流        │
           └───────────────────────────────┘
                           ↓
           ┌───────────────────────────────┐
           │ apply_evolution()             │
           │ - 更新 Skill Graph            │
           │ - 更新成功率统计               │
           │ - 触发知识积累                 │
           └───────────────────────────────┘
                           ↓
           下次请求时使用优化后的技能组合
```

---

## 4. 实现计划

### Phase 1: 核心集成（高优先级）

| 任务 | 文件 | 依赖 |
|------|------|------|
| 创建 SkillGraphStore | agent/skill_graph.rs | skill_graph/store.rs |
| 创建 SkillDiscoveryEngine | agent/discovery.rs | skill_graph/discovery.rs |
| 创建 SkillEvolutionEngine | agent/evolution.rs | skill_graph/evolution.rs |
| 集成 HyperspaceStore | agent/memory.rs | memory/hyperspace_store.rs |
| 定义 ComfyUI 技能本体 | skills/comfyui_skills.jsonld | - |

### Phase 2: 知识积累

| 任务 | 文件 | 依赖 |
|------|------|------|
| 创建 KnowledgeGraphStore | agent/knowledge.rs | knowledge_graph/store.rs |
| 创建 CausalEngine | agent/causal.rs | causal/engine.rs |
| 创建 TimelineStore | agent/timeline.rs | snapshots/timeline.rs |
| 存储模型元数据到 KG | model_manager/kg_sync.rs | - |

### Phase 3: 智能增强

| 任务 | 文件 | 依赖 |
|------|------|------|
| 集成 Workspace Monitor | agent/monitor.rs | tools/workspace_monitor.rs |
| 实现自动技能发现 | agent/auto_discovery.rs | - |
| 实现失败根因分析 | agent/failure_analysis.rs | - |
| 实现智能参数推荐 | agent/param_recommend.rs | HyperspaceStore |

---

## 5. 预期效果

### 5.1 用户体验改进

| 场景 | 当前 | 集成后 |
|------|------|--------|
| 生成失败 | 手动检查错误日志 | 自动分析根因 + 修复建议 |
| 参数选择 | 需要了解技术细节 | 基于成功率统计自动推荐 |
| 模型选择 | 手动浏览列表 | 语义搜索 + 兼容性检查 |
| 工作流优化 | 手动调整 | 自动进化最优组合 |

### 5.2 系统能力提升

- **知识积累**：每次执行都贡献到 Skill Graph，系统越来越智能
- **自进化**：自动发现最优节点组合，废弃低效配置
- **语义理解**：向量搜索让意图匹配更精准
- **根因分析**：失败时自动定位问题，给出修复建议

---

## 6. 下一步行动

1. 创建 `agent/skill_graph.rs` - 集成 SkillGraphStore
2. 创建 `agent/discovery.rs` - 集成 SkillDiscoveryEngine
3. 创建 `agent/evolution.rs` - 集成 SkillEvolutionEngine
4. 定义 ComfyUI 技能本体 JSON-LD
5. 修改 AgentEngine 初始化流程，添加高级组件