// 工作流验证器

use crate::types::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否合法
    pub valid: bool,
    /// 拓扑排序后的执行顺序
    pub execution_order: Vec<NodeId>,
    /// 缺失的节点ID列表
    pub missing_nodes: Vec<NodeId>,
    /// 错误信息
    pub errors: Vec<String>,
}

/// 工作流验证器
pub struct WorkflowValidator {
    /// 节点类型注册表
    registered_nodes: HashMap<String, NodeDefinition>,
}

#[derive(Debug, Clone)]
struct NodeDefinition {
    class_type: String,
    input_types: HashMap<String, DataType>,
    output_types: HashMap<String, DataType>,
    required_inputs: HashSet<String>,
}

impl WorkflowValidator {
    pub fn new() -> Self {
        Self {
            registered_nodes: Self::register_default_nodes(),
        }
    }

    /// 验证工作流
    pub fn validate(&self, workflow: &Workflow) -> Result<ValidationResult, Error> {
        let mut errors = Vec::new();
        let mut missing_nodes = Vec::new();

        // 1. 检查节点是否存在
        for (node_id, node) in &workflow.nodes {
            if !self.registered_nodes.contains_key(&node.class_type) {
                missing_nodes.push(node_id.clone());
                errors.push(format!("Node class '{}' not found", node.class_type));
            }
        }

        if !missing_nodes.is_empty() {
            return Ok(ValidationResult {
                valid: false,
                execution_order: vec![],
                missing_nodes,
                errors,
            });
        }

        // 2. 检查输入是否合法
        for (node_id, node) in &workflow.nodes {
            if let Some(def) = self.registered_nodes.get(&node.class_type) {
                for required_input in &def.required_inputs {
                    if !node.inputs.contains_key(required_input) {
                        errors.push(format!(
                            "Node '{}' missing required input '{}'",
                            node_id, required_input
                        ));
                    }
                }
            }
        }

        // 3. 检查连接类型是否匹配
        for link in &workflow.links {
            if let Some(from_node) = workflow.nodes.get(&link.from_node) {
                if let Some(to_node) = workflow.nodes.get(&link.to_node) {
                    if let Some(from_def) = self.registered_nodes.get(&from_node.class_type) {
                        if let Some(to_def) = self.registered_nodes.get(&to_node.class_type) {
                            // 检查源输出槽是否有效
                            // 检查目标输入槽是否有效
                            // 检查数据类型是否匹配
                        }
                    }
                } else {
                    errors.push(format!("Link to missing node '{}'", link.to_node));
                }
            } else {
                errors.push(format!("Link from missing node '{}'", link.from_node));
            }
        }

        // 4. 检查是否成环 (拓扑排序)
        let execution_order = self.topological_sort(workflow)?;

        // 5. 返回验证结果
        Ok(ValidationResult {
            valid: errors.is_empty(),
            execution_order,
            missing_nodes,
            errors,
        })
    }

    /// 拓扑排序
    ///
    /// 依赖关系来源：
    /// 1. 节点 inputs 中的 `InputValue::Link`（ComfyUI prompt API 格式的主要来源）
    /// 2. `workflow.links`（显式连接列表，补充来源）
    ///
    /// 为保证结果确定性，入度为 0 的节点按 NodeId 排序后入队。
    fn topological_sort(&self, workflow: &Workflow) -> Result<Vec<NodeId>, Error> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut result: Vec<NodeId> = Vec::new();

        // 初始化所有节点的入度
        for node_id in workflow.nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
            adjacency.insert(node_id.clone(), Vec::new());
        }

        // 从 InputValue::Link 解析依赖关系
        for (node_id, node) in &workflow.nodes {
            for input_value in node.inputs.values() {
                if let InputValue::Link([from_node, _]) = input_value {
                    if workflow.nodes.contains_key(from_node) {
                        adjacency.get_mut(from_node).unwrap().push(node_id.clone());
                        *in_degree.get_mut(node_id).unwrap() += 1;
                    }
                }
            }
        }

        // 从 workflow.links 解析依赖关系（补充来源）
        for link in &workflow.links {
            if workflow.nodes.contains_key(&link.from_node)
                && workflow.nodes.contains_key(&link.to_node)
            {
                adjacency.get_mut(&link.from_node).unwrap().push(link.to_node.clone());
                *in_degree.get_mut(&link.to_node).unwrap() += 1;
            }
        }

        // 找到所有入度为0的节点，按 NodeId 排序保证确定性
        let mut zero_degree_nodes: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| id.clone())
            .collect();
        zero_degree_nodes.sort();
        for node_id in zero_degree_nodes {
            queue.push_back(node_id);
        }

        // BFS拓扑排序
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id.clone());

            if let Some(neighbors) = adjacency.get(&node_id) {
                // 收集本次解除依赖后入度变 0 的节点，排序后入队以保证确定性
                let mut next_zero: Vec<NodeId> = Vec::new();
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        next_zero.push(neighbor.clone());
                    }
                }
                next_zero.sort();
                for n in next_zero {
                    queue.push_back(n);
                }
            }
        }

        // 检查是否有环
        if result.len() != workflow.nodes.len() {
            return Err(Error::ValidationFailed("Workflow contains cycles".to_string()));
        }

        Ok(result)
    }

    /// 注册默认节点类型
    fn register_default_nodes() -> HashMap<String, NodeDefinition> {
        let mut nodes = HashMap::new();

        // CheckpointLoaderSimple
        nodes.insert(
            "CheckpointLoaderSimple".to_string(),
            NodeDefinition {
                class_type: "CheckpointLoaderSimple".to_string(),
                input_types: HashMap::from([
                    ("ckpt_name".to_string(), DataType::STRING),
                ]),
                output_types: HashMap::from([
                    ("MODEL".to_string(), DataType::MODEL),
                    ("CLIP".to_string(), DataType::CLIP),
                    ("VAE".to_string(), DataType::VAE),
                ]),
                required_inputs: HashSet::from(["ckpt_name".to_string()]),
            },
        );

        // CLIPTextEncode
        nodes.insert(
            "CLIPTextEncode".to_string(),
            NodeDefinition {
                class_type: "CLIPTextEncode".to_string(),
                input_types: HashMap::from([
                    ("text".to_string(), DataType::STRING),
                    ("clip".to_string(), DataType::CLIP),
                ]),
                output_types: HashMap::from([
                    ("CONDITIONING".to_string(), DataType::CONDITIONING),
                ]),
                required_inputs: HashSet::from(["text".to_string(), "clip".to_string()]),
            },
        );

        // KSampler
        nodes.insert(
            "KSampler".to_string(),
            NodeDefinition {
                class_type: "KSampler".to_string(),
                input_types: HashMap::from([
                    ("model".to_string(), DataType::MODEL),
                    ("positive".to_string(), DataType::CONDITIONING),
                    ("negative".to_string(), DataType::CONDITIONING),
                    ("latent_image".to_string(), DataType::LATENT),
                    ("seed".to_string(), DataType::INT),
                    ("steps".to_string(), DataType::INT),
                    ("cfg".to_string(), DataType::FLOAT),
                    ("sampler_name".to_string(), DataType::STRING),
                    ("scheduler".to_string(), DataType::STRING),
                    ("denoise".to_string(), DataType::FLOAT),
                ]),
                output_types: HashMap::from([
                    ("LATENT".to_string(), DataType::LATENT),
                ]),
                required_inputs: HashSet::from([
                    "model".to_string(),
                    "positive".to_string(),
                    "negative".to_string(),
                    "latent_image".to_string(),
                ]),
            },
        );

        // EmptyLatentImage
        nodes.insert(
            "EmptyLatentImage".to_string(),
            NodeDefinition {
                class_type: "EmptyLatentImage".to_string(),
                input_types: HashMap::from([
                    ("width".to_string(), DataType::INT),
                    ("height".to_string(), DataType::INT),
                    ("batch_size".to_string(), DataType::INT),
                ]),
                output_types: HashMap::from([
                    ("LATENT".to_string(), DataType::LATENT),
                ]),
                required_inputs: HashSet::from(["width".to_string(), "height".to_string()]),
            },
        );

        // VAEDecode
        nodes.insert(
            "VAEDecode".to_string(),
            NodeDefinition {
                class_type: "VAEDecode".to_string(),
                input_types: HashMap::from([
                    ("samples".to_string(), DataType::LATENT),
                    ("vae".to_string(), DataType::VAE),
                ]),
                output_types: HashMap::from([
                    ("IMAGE".to_string(), DataType::IMAGE),
                ]),
                required_inputs: HashSet::from(["samples".to_string(), "vae".to_string()]),
            },
        );

        // SaveImage
        nodes.insert(
            "SaveImage".to_string(),
            NodeDefinition {
                class_type: "SaveImage".to_string(),
                input_types: HashMap::from([
                    ("images".to_string(), DataType::IMAGE),
                    ("filename_prefix".to_string(), DataType::STRING),
                ]),
                output_types: HashMap::new(),
                required_inputs: HashSet::from(["images".to_string()]),
            },
        );

        // LoadImage
        nodes.insert(
            "LoadImage".to_string(),
            NodeDefinition {
                class_type: "LoadImage".to_string(),
                input_types: HashMap::from([
                    ("image".to_string(), DataType::STRING),
                ]),
                output_types: HashMap::from([
                    ("IMAGE".to_string(), DataType::IMAGE),
                ]),
                required_inputs: HashSet::from(["image".to_string()]),
            },
        );

        // VAEEncode
        nodes.insert(
            "VAEEncode".to_string(),
            NodeDefinition {
                class_type: "VAEEncode".to_string(),
                input_types: HashMap::from([
                    ("pixels".to_string(), DataType::IMAGE),
                    ("vae".to_string(), DataType::VAE),
                ]),
                output_types: HashMap::from([
                    ("LATENT".to_string(), DataType::LATENT),
                ]),
                required_inputs: HashSet::from(["pixels".to_string(), "vae".to_string()]),
            },
        );

        nodes.insert(
            "VideoCombine".to_string(),
            NodeDefinition {
                class_type: "VideoCombine".to_string(),
                input_types: HashMap::from([
                    ("images".to_string(), DataType::IMAGE),
                    ("frame_rate".to_string(), DataType::INT),
                    ("format".to_string(), DataType::STRING),
                    ("codec".to_string(), DataType::STRING),
                    ("quality".to_string(), DataType::FLOAT),
                    ("filename_prefix".to_string(), DataType::STRING),
                ]),
                output_types: HashMap::new(),
                required_inputs: HashSet::new(),
            },
        );

        nodes
    }
}