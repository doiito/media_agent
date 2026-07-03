// JSON-LD 工作流适配 + DagEngine 执行
// 多步生成任务编排

use std::sync::Arc;
use std::path::Path;
use glidinghorse::core::workflow::{load_workflow_jsonld, build_dag, DagEngine, WorkflowDag};
use glidinghorse::core::agent_runner::TaskResult;

/// 执行 Agent 工作流
pub async fn execute_agent_workflow(
    runner: Arc<glidinghorse::core::AgentRunner>,
    workflow_path: &str,
    user_input: &str,
    max_iterations: u32,
) -> Result<TaskResult, String> {
    // 读取 JSON-LD 工作流文件
    let jsonld = std::fs::read_to_string(workflow_path)
        .map_err(|e| format!("Failed to read workflow {}: {}", workflow_path, e))?;

    // 解析 JSON-LD 工作流定义
    let workflow_def = load_workflow_jsonld(&jsonld)
        .map_err(|e| format!("Failed to parse workflow JSON-LD: {}", e))?;

    // 构建 DAG
    let dag = build_dag(&workflow_def)
        .map_err(|e| format!("Failed to build DAG: {}", e))?;

    // 创建 DagEngine
    let engine = DagEngine::new(runner, max_iterations);

    // 生成任务 IRI
    let task_iri = format!("iri://task/{}", uuid::Uuid::new_v4());

    // 执行 DAG
    let results = engine.execute(&dag, &task_iri, user_input).await
        .map_err(|e| format!("DAG execution failed: {}", e))?;

    // 转换为 TaskResult
    Ok(convert_to_task_result(results, &task_iri))
}

/// 转换执行结果为 TaskResult
fn convert_to_task_result(
    results: Vec<glidinghorse::core::workflow::NodeResult>,
    task_iri: &str,
) -> TaskResult {
    let mut final_outputs = serde_json::Map::new();
    let mut turn_count = 0;
    let mut tool_call_count = 0;
    let mut errors = vec![];

    for result in results {
        turn_count += result.turn_count;
        tool_call_count += result.tool_call_count;

        if let Some(err) = result.error {
            errors.push(err);
        }

        if let Some(output) = result.output {
            if let serde_json::Value::Object(map) = output {
                for (k, v) in map {
                    final_outputs.insert(k, v);
                }
            }
        }
    }

    let status = if errors.is_empty() { "success" } else { "partial" };

    TaskResult {
        task_iri: task_iri.to_string(),
        status: status.to_string(),
        summary: "Workflow completed".to_string(),
        output: Some(serde_json::Value::Object(final_outputs)),
        jsonld_output: None,
        artifacts: vec![],
        errors,
        turn_count,
        tool_call_count,
        five_w2h_updates: None,
        tracked_actions: vec![],
        archive_iri: None,
    }
}

/// 加载工作流模板
pub fn load_workflow_templates(templates_dir: &str) -> Result<Vec<String>, String> {
    let dir = Path::new(templates_dir);

    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut templates = vec![];

    for entry in std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read templates directory: {}", e))?
    {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "jsonld") {
            templates.push(path.to_string_lossy().into_owned());
        }
    }

    Ok(templates)
}

/// 列出可用的工作流模板
pub fn list_workflows(templates_dir: &str) -> Vec<WorkflowInfo> {
    match load_workflow_templates(templates_dir) {
        Ok(paths) => {
            paths.iter().filter_map(|path| {
                let name = Path::new(path).file_stem()?.to_str()?.to_string();
                Some(WorkflowInfo {
                    name: name.clone(),
                    path: path.clone(),
                    description: format!("Workflow: {}", name),
                })
            }).collect()
        }
        Err(_) => vec![],
    }
}

/// 工作流信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkflowInfo {
    pub name: String,
    pub path: String,
    pub description: String,
}
