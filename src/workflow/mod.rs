// 工作流模块

mod validator;
pub mod builder;

pub use validator::{WorkflowValidator, ValidationResult};
pub use builder::WorkflowBuilder;

use crate::types::*;

/// 工作流管理器
pub struct WorkflowManager {
    validator: WorkflowValidator,
}

impl WorkflowManager {
    pub fn new() -> Self {
        Self {
            validator: WorkflowValidator::new(),
        }
    }

    /// 验证工作流
    pub fn validate(&self, workflow: &Workflow) -> Result<ValidationResult, Error> {
        self.validator.validate(workflow)
    }

    /// 创建简单文生图工作流
    pub fn create_text_to_image_workflow(
        prompt: String,
        negative_prompt: String,
        width: usize,
        height: usize,
        steps: usize,
        cfg: f32,
        seed: usize,
        model: String,
    ) -> Result<Workflow, Error> {
        WorkflowBuilder::text_to_image(
            prompt, negative_prompt, width, height, steps, cfg, seed, model
        )
    }

    /// 创建图生图工作流
    pub fn create_image_to_image_workflow(
        prompt: String,
        negative_prompt: String,
        input_image: String,
        denoise: f32,
        steps: usize,
        cfg: f32,
        seed: usize,
        model: String,
    ) -> Result<Workflow, Error> {
        WorkflowBuilder::image_to_image(
            prompt, negative_prompt, input_image, denoise, steps, cfg, seed, model
        )
    }
}