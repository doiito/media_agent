// Agent 智能调度模块
// 基于 gliding_horse 库实现 LLM 编排、工具调用、记忆系统和工作流引擎
// 支持 DeepSeek API（环境变量配置）和本地 llama.cpp server

pub mod context;
pub mod memory;
pub mod engine;
pub mod tools;
pub mod smart_tools;
pub mod skills;
pub mod workflow;
pub mod handlers;
pub mod llm;

pub use context::AgentContext;
pub use engine::AgentEngine;
pub use memory::AgentMemory;
pub use llm::{LlmClient, LlmConfig, LlmError, ChatMessage, ChatRequest, ChatResponse, ToolDefinition};