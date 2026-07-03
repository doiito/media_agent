// 生成技能定义 + SkillRegistry 注册
// JSON-LD 技能文件加载和注册

use std::path::Path;
use serde_json::Value;
use glidinghorse::tools::skill_registry::SkillMeta;

/// 加载 ComfyUI 技能到 SkillRegistry
pub fn load_comfyui_skills(
    registry: &glidinghorse::tools::skill_registry::SkillRegistry,
) -> Result<usize, String> {
    let skills_dir = Path::new("skills");

    if !skills_dir.exists() {
        log::warn!("Skills directory not found: skills/");
        return Ok(0);
    }

    let mut count = 0;

    for entry in std::fs::read_dir(skills_dir)
        .map_err(|e| format!("Failed to read skills directory: {}", e))?
    {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "jsonld") {
            let skill_name = path.file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");

            let jsonld_content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let skill_def: Value = serde_json::from_str(&jsonld_content)
                .map_err(|e| format!("Invalid JSON-LD in {}: {}", path.display(), e))?;

            let skill_meta = build_skill_meta(skill_name, &skill_def);

            registry.register_skill(skill_meta);
            count += 1;
            log::info!("Loaded skill: {} from {}", skill_name, path.display());
        }
    }

    log::info!("Loaded {} skills from skills/", count);
    Ok(count)
}

/// 从 JSON-LD 定义构建 SkillMeta
fn build_skill_meta(name: &str, jsonld: &Value) -> SkillMeta {
    let skill_iri = jsonld.get("@id")
        .and_then(|v| v.as_str())
        .unwrap_or(&format!("iri://skills/comfyui/{}", name))
        .to_string();

    let description = jsonld.get("schema:description")
        .and_then(|v| v.as_str())
        .unwrap_or(&format!("ComfyUI {} skill", name))
        .to_string();

    let version = jsonld.get("skill:version")
        .and_then(|v| v.as_str())
        .unwrap_or("1.0.0")
        .to_string();

    let category = jsonld.get("skill:category")
        .and_then(|v| v.as_str())
        .unwrap_or("ai")
        .to_string();

    let security_level = jsonld.get("skill:securityLevel")
        .and_then(|v| v.as_str())
        .unwrap_or("normal")
        .to_string();

    let allowed_roles = jsonld.get("skill:allowedRoles")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_else(|| vec!["DA".to_string()]);

    let input_schema = jsonld.get("skill:inputSchema")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

    let output_schema = jsonld.get("skill:outputSchema")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

    SkillMeta {
        skill_iri,
        name: name.to_string(),
        description,
        version,
        category,
        security_level,
        allowed_roles,
        input_schema,
        output_schema,
        compiled_template: String::new(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    }
}

/// 注册内置 ComfyUI 技能
pub fn register_builtin_skills(
    registry: &glidinghorse::tools::skill_registry::SkillRegistry,
) -> Result<usize, String> {
    // text_to_image
    registry.register_skill(SkillMeta {
        skill_iri: "iri://skills/comfyui/text_to_image".to_string(),
        name: "text_to_image".to_string(),
        description: "Generate an image from a text prompt using Stable Diffusion".to_string(),
        version: "1.0.0".to_string(),
        category: "ai".to_string(),
        security_level: "normal".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "steps": {"type": "integer", "default": 20},
                "cfg": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": 0}
            },
            "required": ["prompt"]
        }),
        output_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt_id": {"type": "string"}
            }
        }),
        compiled_template: String::new(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    // image_to_image
    registry.register_skill(SkillMeta {
        skill_iri: "iri://skills/comfyui/image_to_image".to_string(),
        name: "image_to_image".to_string(),
        description: "Transform an input image based on a text prompt".to_string(),
        version: "1.0.0".to_string(),
        category: "ai".to_string(),
        security_level: "normal".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt": {"type": "string"},
                "strength": {"type": "number", "default": 0.75}
            },
            "required": ["image_path", "prompt"]
        }),
        output_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "image_path": {"type": "string"}
            }
        }),
        compiled_template: String::new(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    log::info!("Registered 2 builtin ComfyUI skills");
    Ok(2)
}
