// 节点系统模块

pub mod registry;
pub mod core_nodes;
pub mod extended_nodes;
pub mod advanced_sampler;
pub mod image_processing;
pub mod video_nodes;

pub use registry::NodeRegistry;
pub use core_nodes::*;
pub use extended_nodes::*;
pub use advanced_sampler::*;
pub use image_processing::*;
pub use video_nodes::*;

use crate::types::*;
use async_trait::async_trait;
use std::collections::HashMap;

/// 节点接口
#[async_trait]
pub trait Node: Send + Sync {
    /// 获取节点类型
    fn class_type(&self) -> &str;

    /// 获取输入类型定义
    fn input_types(&self) -> HashMap<String, InputType>;

    /// 获取输出类型定义
    fn output_types(&self) -> HashMap<String, OutputType>;

    /// 执行节点
    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error>;

    /// IS_CHANGED函数（可选）
    fn is_changed(&self, inputs: &HashMap<String, Value>) -> Option<Vec<Option<f64>>> {
        None
    }
}

/// 输入类型定义
#[derive(Debug, Clone)]
pub struct InputType {
    pub data_type: DataType,
    pub required: bool,
    pub default: Option<Value>,
    pub choices: Option<Vec<String>>,
}

/// 输出类型定义
#[derive(Debug, Clone)]
pub struct OutputType {
    pub data_type: DataType,
    pub name: String,
}
