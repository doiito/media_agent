// Prompt执行器

use crate::types::*;
use crate::workflow::WorkflowValidator;
use crate::node::NodeRegistry;
use crate::backend::BackendRouter;
use crate::execution::cache::HierarchicalCache;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;

/// Prompt执行器
pub struct PromptExecutor {
    /// 节点注册表
    node_registry: Arc<NodeRegistry>,
    /// 后端路由器
    backend_router: Arc<Mutex<BackendRouter>>,
    /// 缓存系统
    cache: HierarchicalCache,
    /// 是否中断
    interrupted: Arc<StdMutex<bool>>,
}

impl PromptExecutor {
    pub fn new() -> Self {
        Self {
            node_registry: Arc::new(NodeRegistry::new()),
            backend_router: Arc::new(Mutex::new(BackendRouter::new())),
            cache: HierarchicalCache::new(),
            interrupted: Arc::new(StdMutex::new(false)),
        }
    }

    /// 执行工作流
    pub async fn execute(&mut self, workflow: &Workflow) -> Result<ExecutionResult, Error> {
        // 1. 验证工作流
        let validator = WorkflowValidator::new();
        let validation_result = validator.validate(workflow)?;

        if !validation_result.valid {
            return Ok(ExecutionResult::Failure(
                validation_result.errors.join(", ")
            ));
        }

        // 2. 获取执行顺序
        let execution_order = validation_result.execution_order;

        // 3. 存储节点输出
        let mut outputs: HashMap<NodeId, HashMap<String, Value>> = HashMap::new();

        // 4. 逐节点执行
        for node_id in execution_order {
            // 检查是否中断
            if *self.interrupted.lock().unwrap() {
                return Ok(ExecutionResult::Failure("Interrupted".to_string()));
            }

            let node = workflow.nodes.get(&node_id).unwrap();

            // 检查缓存
            let fingerprint = self.compute_fingerprint(node, &outputs);
            if let Some(cached_output) = self.cache.get(&fingerprint) {
                outputs.insert(node_id.clone(), cached_output);
                continue; // 跳过执行，使用缓存
            }

            // 准备输入数据
            let inputs = self.prepare_inputs(node, &outputs)?;

            // 执行节点
            let node_instance = match self.node_registry.create_node(&node.class_type) {
                Ok(n) => n,
                Err(e) => {
                    return Ok(ExecutionResult::Failure(format!("Node creation failed: {}", e)));
                }
            };
            let result = match node_instance.lock().await.execute(inputs).await {
                Ok(r) => r,
                Err(e) => {
                    return Ok(ExecutionResult::Failure(format!("Node '{}' execution failed: {}", node_id, e)));
                }
            };

            // 存储输出
            outputs.insert(node_id.clone(), result.clone());

            // 缓存输出
            self.cache.put(fingerprint, result);
        }

        Ok(ExecutionResult::Success(outputs))
    }

    /// 计算节点指纹（用于缓存）
    fn compute_fingerprint(
        &self,
        node: &WorkflowNode,
        outputs: &HashMap<NodeId, HashMap<String, Value>>,
    ) -> NodeFingerprint {
        // 哈希节点类型
        let mut hasher = blake3::Hasher::new();
        hasher.update(node.class_type.as_bytes());

        // 哈希输入值（不包括连接输入）
        for (key, value) in &node.inputs {
            match value {
                InputValue::Direct(v) => {
                    hasher.update(key.as_bytes());
                    hasher.update(serde_json::to_string(v).unwrap().as_bytes());
                }
                InputValue::Link(_) => {}, // 连接输入不参与指纹
            }
        }

        NodeFingerprint {
            class_type: node.class_type.clone(),
            input_hash: hasher.finalize(),
        }
    }

    /// 准备节点输入数据
    fn prepare_inputs(
        &self,
        node: &WorkflowNode,
        outputs: &HashMap<NodeId, HashMap<String, Value>>,
    ) -> Result<HashMap<String, Value>, Error> {
        let mut inputs = HashMap::new();

        for (key, value) in &node.inputs {
            match value {
                InputValue::Direct(v) => {
                    inputs.insert(key.clone(), v.clone());
                }
                InputValue::Link([from_node_id, output_slot]) => {
                    // 从输出中获取值
                    if let Some(node_outputs) = outputs.get(from_node_id) {
                        let slot_idx: usize = output_slot.parse()?;
                        let output_keys: Vec<String> = node_outputs.keys().cloned().collect();
                        if let Some(output_key) = output_keys.get(slot_idx) {
                            if let Some(output_value) = node_outputs.get(output_key) {
                                inputs.insert(key.clone(), output_value.clone());
                            }
                        }
                    } else {
                        return Err(Error::InvalidConnection(format!(
                            "Node '{}' output not found for input '{}'",
                            from_node_id, key
                        )));
                    }
                }
            }
        }

        Ok(inputs)
    }

    /// 中断当前执行
    pub fn interrupt(&mut self) {
        let mut interrupted = self.interrupted.lock().unwrap();
        *interrupted = true;
    }

    /// 释放显存
    pub async fn free_memory(&mut self) {
        self.cache.clear();
        // 清空后端路由器的显存
        let router = self.backend_router.lock().await;
        router.free_memory().await;
    }
}