// JSON-LD 工作流适配 + DagEngine 执行
// 多步生成任务编排

use std::sync::Arc;
use std::path::Path;
use glidinghorse::core::workflow::{load_workflow_jsonld, build_dag};
use glidinghorse::core::workflow::adapter::dag_to_execution_plan;
use glidinghorse::core::agent_runner::TaskResult;

/// 执行 Agent 工作流（新版 API，通过 SupervisorAgent）
pub async fn execute_agent_workflow(
    supervisor: &mut glidinghorse::core::SupervisorAgent,
    workflow_path: &str,
    user_input: &str,
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

    // 将 DAG 转换为 ExecutionPlan（使用 adapter）
    let task_iri = format!("iri://task/{}", uuid::Uuid::new_v4());
    let plan = dag_to_execution_plan(&dag, &workflow_def, &task_iri);

    // 初始化 5W2H
    let five_w2h_iri = format!("iri://5w2h/{}", uuid::Uuid::new_v4());
    let five_w2h = glidinghorse::core::five_w2h::Task5W2H::default();

    // 通过 SupervisorAgent 执行计划
    supervisor.execute_plan(plan, &task_iri, user_input, five_w2h, &five_w2h_iri, None, None)
        .await
        .map_err(|e| format!("Workflow execution failed: {:?}", e))
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
