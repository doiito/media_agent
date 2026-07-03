// ComfyUI Rust Agent - 核心库

pub mod types;
pub mod workflow;
pub mod execution;
pub mod node;
pub mod backend;
pub mod api;
pub mod agent;
pub mod storage;
pub mod util;
pub mod config;
pub mod monitor;
pub mod model_manager;
pub mod preview;

// 导出常用类型
pub use types::*;

// 库版本
pub const VERSION: &str = "0.1.0";

// 项目信息
pub fn get_project_info() -> String {
    format!(
        "ComfyUI Rust Agent v{}\n\
         Enhanced intelligent workflow system\n\
         Backend: stable-diffusion.cpp + llama.cpp",
        VERSION
    )
}