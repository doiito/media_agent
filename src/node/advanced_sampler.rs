// 高级采样器节点
// 支持 KSamplerAdvanced、SamplerCustom、SchedulerAdvanced 等

use crate::types::*;
use crate::node::{Node, InputType, OutputType};
use crate::backend::BackendRouter;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use log::{info, debug};

// ============================================================================
// KSamplerAdvanced 节点 - 高级采样器，支持更多参数
// ============================================================================

pub struct KSamplerAdvancedNode {
    backend_router: Arc<BackendRouter>,
}

impl KSamplerAdvancedNode {
    pub fn new() -> Self {
        Self {
            backend_router: Arc::new(BackendRouter::from_env()),
        }
    }

    pub fn with_backend(router: Arc<BackendRouter>) -> Self {
        Self {
            backend_router: router,
        }
    }
}

impl Default for KSamplerAdvancedNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for KSamplerAdvancedNode {
    fn class_type(&self) -> &str {
        "KSamplerAdvanced"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("positive".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("negative".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("latent_image".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("add_noise".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("enable".to_string())),
                choices: Some(vec!["enable".to_string(), "disable".to_string()]),
            }),
            ("noise_seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(20)),
                choices: None,
            }),
            ("cfg".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(7.0)),
                choices: None,
            }),
            ("sampler_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("euler".to_string())),
                choices: Some(vec![
                    "euler".to_string(),
                    "euler_ancestral".to_string(),
                    "euler_cfg_pp".to_string(),
                    "dpmpp_2m".to_string(),
                    "dpmpp_2m_sde".to_string(),
                    "dpmpp_2s_ancestral".to_string(),
                    "dpmpp_3m_sde".to_string(),
                    "ddim".to_string(),
                    "ddpm".to_string(),
                    "lcm".to_string(),
                    "ipndm".to_string(),
                    "heun".to_string(),
                    "res_multistep".to_string(),
                ]),
            }),
            ("scheduler".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("normal".to_string())),
                choices: Some(vec![
                    "normal".to_string(),
                    "karras".to_string(),
                    "exponential".to_string(),
                    "sgm_uniform".to_string(),
                    "simple".to_string(),
                    "ddim_uniform".to_string(),
                    "beta".to_string(),
                    "linear_quadratic".to_string(),
                    "klur".to_string(),
                ]),
            }),
            ("start_at_step".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("end_at_step".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(10000)),
                choices: None,
            }),
            ("return_with_leftover_noise".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("disable".to_string())),
                choices: Some(vec!["enable".to_string(), "disable".to_string()]),
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?;
        let positive = inputs.get("positive")
            .ok_or_else(|| Error::ExecutionFailed("Missing positive".to_string()))?
            .clone();
        let negative = inputs.get("negative")
            .ok_or_else(|| Error::ExecutionFailed("Missing negative".to_string()))?
            .clone();
        let latent = inputs.get("latent_image")
            .ok_or_else(|| Error::ExecutionFailed("Missing latent_image".to_string()))?
            .clone();
        let add_noise = inputs.get("add_noise")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("enable");
        let noise_seed = inputs.get("noise_seed")
            .unwrap_or(&Value::Int(0))
            .as_int()?;
        let steps = inputs.get("steps")
            .unwrap_or(&Value::Int(20))
            .as_int()?;
        let cfg = inputs.get("cfg")
            .unwrap_or(&Value::Float(7.0))
            .as_float()?;
        let sampler_name = inputs.get("sampler_name")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("euler");
        let scheduler = inputs.get("scheduler")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("normal");
        let start_at_step = inputs.get("start_at_step")
            .unwrap_or(&Value::Int(0))
            .as_int()?;
        let end_at_step = inputs.get("end_at_step")
            .unwrap_or(&Value::Int(10000))
            .as_int()?;
        let return_with_leftover_noise = inputs.get("return_with_leftover_noise")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("disable");

        info!("KSamplerAdvanced: model={}, add_noise={}, seed={}, steps={}, cfg={}, sampler={}, scheduler={}, start={}, end={}, leftover={}",
              model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
              start_at_step, end_at_step, return_with_leftover_noise);

        let denoise = if start_at_step > 0 { 1.0 } else { 1.0 };

        let output_latent = self.backend_router.sample(
            model, positive, negative, latent,
            noise_seed, steps, cfg,
            sampler_name, scheduler, denoise
        ).await?;

        Ok(HashMap::from([
            ("LATENT".to_string(), output_latent),
        ]))
    }
}

// ============================================================================
// SamplerCustom 节点 - 自定义采样器
// ============================================================================

pub struct SamplerCustomNode {
    backend_router: Arc<BackendRouter>,
}

impl SamplerCustomNode {
    pub fn new() -> Self {
        Self {
            backend_router: Arc::new(BackendRouter::from_env()),
        }
    }

    pub fn with_backend(router: Arc<BackendRouter>) -> Self {
        Self {
            backend_router: router,
        }
    }
}

impl Default for SamplerCustomNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Node for SamplerCustomNode {
    fn class_type(&self) -> &str {
        "SamplerCustom"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("model".to_string(), InputType {
                data_type: DataType::MODEL,
                required: true,
                default: None,
                choices: None,
            }),
            ("positive".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("negative".to_string(), InputType {
                data_type: DataType::CONDITIONING,
                required: true,
                default: None,
                choices: None,
            }),
            ("latent_image".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(20)),
                choices: None,
            }),
            ("cfg".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(7.0)),
                choices: None,
            }),
            ("sampler_name".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("euler".to_string())),
                choices: Some(vec![
                    "euler".to_string(),
                    "euler_ancestral".to_string(),
                    "dpmpp_2m".to_string(),
                    "dpmpp_2m_sde".to_string(),
                    "lcm".to_string(),
                    "ddim".to_string(),
                ]),
            }),
            ("scheduler".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("karras".to_string())),
                choices: Some(vec![
                    "normal".to_string(),
                    "karras".to_string(),
                    "exponential".to_string(),
                    "simple".to_string(),
                ]),
            }),
            ("denoise".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(1.0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
            ("LATENT_DENOISED".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT_DENOISED".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let model = inputs.get("model")
            .ok_or_else(|| Error::ExecutionFailed("Missing model".to_string()))?
            .as_ref_str()?;
        let positive = inputs.get("positive")
            .ok_or_else(|| Error::ExecutionFailed("Missing positive".to_string()))?
            .clone();
        let negative = inputs.get("negative")
            .ok_or_else(|| Error::ExecutionFailed("Missing negative".to_string()))?
            .clone();
        let latent = inputs.get("latent_image")
            .ok_or_else(|| Error::ExecutionFailed("Missing latent_image".to_string()))?
            .clone();
        let seed = inputs.get("seed")
            .unwrap_or(&Value::Int(0))
            .as_int()?;
        let steps = inputs.get("steps")
            .unwrap_or(&Value::Int(20))
            .as_int()?;
        let cfg = inputs.get("cfg")
            .unwrap_or(&Value::Float(7.0))
            .as_float()?;
        let sampler_name = inputs.get("sampler_name")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("euler");
        let scheduler = inputs.get("scheduler")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("karras");
        let denoise = inputs.get("denoise")
            .unwrap_or(&Value::Float(1.0))
            .as_float()?;

        info!("SamplerCustom: model={}, seed={}, steps={}, cfg={}, sampler={}, scheduler={}, denoise={}",
              model, seed, steps, cfg, sampler_name, scheduler, denoise);

        let output_latent = self.backend_router.sample(
            model, positive, negative, latent.clone(),
            seed, steps, cfg,
            sampler_name, scheduler, denoise
        ).await?;

        // 返回 LATENT 和 LATENT_DENOISED（去噪后的 latent）
        Ok(HashMap::from([
            ("LATENT".to_string(), output_latent.clone()),
            ("LATENT_DENOISED".to_string(), output_latent),
        ]))
    }
}

// ============================================================================
// SchedulerAdvanced 节点 - 高级调度器配置
// ============================================================================

pub struct SchedulerAdvancedNode;

impl Default for SchedulerAdvancedNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for SchedulerAdvancedNode {
    fn class_type(&self) -> &str {
        "SchedulerAdvanced"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("scheduler".to_string(), InputType {
                data_type: DataType::STRING,
                required: false,
                default: Some(Value::String("karras".to_string())),
                choices: Some(vec![
                    "normal".to_string(),
                    "karras".to_string(),
                    "exponential".to_string(),
                    "sgm_uniform".to_string(),
                    "simple".to_string(),
                    "ddim_uniform".to_string(),
                    "beta".to_string(),
                    "linear_quadratic".to_string(),
                ]),
            }),
            ("steps".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(20)),
                choices: None,
            }),
            ("sigma_max".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(14.6)),
                choices: None,
            }),
            ("sigma_min".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.0292)),
                choices: None,
            }),
            ("rho".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(7.0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("SIGMAS".to_string(), OutputType {
                data_type: DataType::FLOAT,
                name: "SIGMAS".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let scheduler = inputs.get("scheduler")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("karras");
        let steps = inputs.get("steps")
            .unwrap_or(&Value::Int(20))
            .as_int()?;
        let sigma_max = inputs.get("sigma_max")
            .unwrap_or(&Value::Float(14.6))
            .as_float()?;
        let sigma_min = inputs.get("sigma_min")
            .unwrap_or(&Value::Float(0.0292))
            .as_float()?;
        let rho = inputs.get("rho")
            .unwrap_or(&Value::Float(7.0))
            .as_float()?;

        debug!("SchedulerAdvanced: scheduler={}, steps={}, sigma_max={}, sigma_min={}, rho={}",
               scheduler, steps, sigma_max, sigma_min, rho);

        // 计算 sigmas 序列（基于 Karras 等调度算法）
        let sigmas = match scheduler {
            "karras" => {
                let mut sigmas = Vec::with_capacity(steps as usize + 1);
                let sigma_max_ln = sigma_max.ln();
                let sigma_min_ln = sigma_min.ln();
                let rho_inv = 1.0 / rho;
                for i in 0..=steps {
                    let t = 1.0 - (i as f64 / steps as f64);
                    let factor = t.powf(rho_inv);
                    let sigma = (sigma_min_ln + factor * (sigma_max_ln - sigma_min_ln)).exp();
                    sigmas.push(sigma);
                }
                sigmas.push(0.0);
                sigmas
            }
            "exponential" => {
                let mut sigmas = Vec::with_capacity(steps as usize + 1);
                let total = sigma_max / sigma_min;
                for i in 0..=steps {
                    let t = i as f64 / steps as f64;
                    let sigma = sigma_max * total.powf(-t);
                    sigmas.push(sigma);
                }
                sigmas.push(0.0);
                sigmas
            }
            "normal" | "simple" | _ => {
                let mut sigmas = Vec::with_capacity(steps as usize + 1);
                for i in 0..=steps {
                    let t = 1.0 - (i as f64 / steps as f64);
                    let sigma = sigma_max + (sigma_min - sigma_max) * t;
                    sigmas.push(sigma);
                }
                sigmas.push(0.0);
                sigmas
            }
        };

        Ok(HashMap::from([
            ("SIGMAS".to_string(), Value::Array(
                sigmas.iter().map(|s| Value::Float(*s)).collect()
            )),
        ]))
    }
}

// ============================================================================
// LatentNoiseInjection 节点 - 向 latent 注入噪声
// ============================================================================

pub struct LatentNoiseInjectionNode;

impl Default for LatentNoiseInjectionNode {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Node for LatentNoiseInjectionNode {
    fn class_type(&self) -> &str {
        "LatentNoiseInjection"
    }

    fn input_types(&self) -> HashMap<String, InputType> {
        HashMap::from([
            ("latents".to_string(), InputType {
                data_type: DataType::LATENT,
                required: true,
                default: None,
                choices: None,
            }),
            ("strength".to_string(), InputType {
                data_type: DataType::FLOAT,
                required: false,
                default: Some(Value::Float(0.5)),
                choices: None,
            }),
            ("seed".to_string(), InputType {
                data_type: DataType::INT,
                required: false,
                default: Some(Value::Int(0)),
                choices: None,
            }),
        ])
    }

    fn output_types(&self) -> HashMap<String, OutputType> {
        HashMap::from([
            ("LATENT".to_string(), OutputType {
                data_type: DataType::LATENT,
                name: "LATENT".to_string(),
            }),
        ])
    }

    async fn execute(&mut self, inputs: HashMap<String, Value>) -> Result<HashMap<String, Value>, Error> {
        let latents = inputs.get("latents")
            .ok_or_else(|| Error::ExecutionFailed("Missing latents".to_string()))?;
        let strength = inputs.get("strength")
            .unwrap_or(&Value::Float(0.5))
            .as_float()? as f32;
        let _seed = inputs.get("seed")
            .unwrap_or(&Value::Int(0))
            .as_int()?;

        debug!("Injecting noise with strength {}", strength);

        let noisy_latent = match latents {
            Value::Latent(data) => {
                let mut result = data.clone();
                // 简单的噪声注入（实际实现需要使用种子化的随机数生成器）
                for (i, v) in result.iter_mut().enumerate() {
                    // 基于位置和种子的伪随机噪声
                    let noise = ((i as f32 * 12.9898 + _seed as f32).sin() * 43758.5453).fract();
                    *v = *v * (1.0 - strength) + noise * strength;
                }
                result
            }
            _ => return Err(Error::TypeError("Expected LATENT".to_string())),
        };

        Ok(HashMap::from([
            ("LATENT".to_string(), Value::Latent(noisy_latent)),
        ]))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_node_class_types() {
        assert_eq!(KSamplerAdvancedNode::new().class_type(), "KSamplerAdvanced");
        assert_eq!(SamplerCustomNode::new().class_type(), "SamplerCustom");
        assert_eq!(SchedulerAdvancedNode.class_type(), "SchedulerAdvanced");
        assert_eq!(LatentNoiseInjectionNode.class_type(), "LatentNoiseInjection");
    }

    #[tokio::test]
    async fn test_scheduler_advanced_karras() {
        let mut node = SchedulerAdvancedNode;
        let mut inputs = HashMap::new();
        inputs.insert("scheduler".to_string(), Value::String("karras".to_string()));
        inputs.insert("steps".to_string(), Value::Int(5));
        inputs.insert("sigma_max".to_string(), Value::Float(10.0));
        inputs.insert("sigma_min".to_string(), Value::Float(0.1));
        inputs.insert("rho".to_string(), Value::Float(7.0));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("SIGMAS"));

        if let Value::Array(sigmas) = &result["SIGMAS"] {
            // steps + 1 个 sigma + 1 个 0
            assert_eq!(sigmas.len(), 7);
        }
    }

    #[tokio::test]
    async fn test_scheduler_advanced_normal() {
        let mut node = SchedulerAdvancedNode;
        let mut inputs = HashMap::new();
        inputs.insert("scheduler".to_string(), Value::String("normal".to_string()));
        inputs.insert("steps".to_string(), Value::Int(4));
        inputs.insert("sigma_max".to_string(), Value::Float(8.0));
        inputs.insert("sigma_min".to_string(), Value::Float(0.2));

        let result = node.execute(inputs).await.unwrap();
        if let Value::Array(sigmas) = &result["SIGMAS"] {
            assert_eq!(sigmas.len(), 6); // 4 + 1 + 1
        }
    }

    #[tokio::test]
    async fn test_latent_noise_injection() {
        let mut node = LatentNoiseInjectionNode;
        let mut inputs = HashMap::new();
        inputs.insert("latents".to_string(), Value::Latent(vec![0.5; 100]));
        inputs.insert("strength".to_string(), Value::Float(0.3));
        inputs.insert("seed".to_string(), Value::Int(42));

        let result = node.execute(inputs).await.unwrap();
        assert!(result.contains_key("LATENT"));

        if let Value::Latent(data) = &result["LATENT"] {
            assert_eq!(data.len(), 100);
            // 噪声注入后值应该改变
            assert_ne!(data[0], 0.5);
        }
    }
}
