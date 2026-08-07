use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Debug, Parser)]
#[command(name = "media-agent-eval", about = "Evaluate the real Gliding Horse media path")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8188")]
    base_url: String,

    #[arg(long, default_value = "evaluation/gliding_horse_suite.json")]
    suite: String,

    #[arg(long, default_value = "output/gliding_horse_evaluation.json")]
    output: String,
}

#[derive(Debug, Deserialize)]
struct EvaluationSuite {
    cases: Vec<EvaluationCase>,
}

#[derive(Debug, Deserialize)]
struct EvaluationCase {
    id: String,
    message: String,
    image_path: Option<String>,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    required_prompt_terms: Vec<String>,
    required_pipeline: Option<String>,
    expected_width: Option<u64>,
    expected_height: Option<u64>,
    #[serde(default = "default_true")]
    expected_success: bool,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    id: String,
    passed: bool,
    http_status: u16,
    execution_mode: Option<String>,
    artifact_verified: bool,
    pdca_roles: Vec<String>,
    ca_artifact_verified: bool,
    turn_count: u64,
    tool_calls: u64,
    prompt_alignment: bool,
    pipeline_alignment: bool,
    dimension_alignment: bool,
    pipeline: Option<String>,
    effective_prompt: Option<String>,
    media_paths: Vec<String>,
    quality_score: Option<f64>,
    gpu_tier: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    generated_at: String,
    total: usize,
    passed: usize,
    false_successes: usize,
    artifact_pass_rate: f64,
    average_turns: f64,
    average_tool_calls: f64,
    average_quality_score: Option<f64>,
    cases: Vec<CaseResult>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let suite: EvaluationSuite = serde_json::from_slice(&std::fs::read(&args.suite)?)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;
    let mut results = Vec::new();

    for case in suite.cases {
        let mut body = json!({
            "message": case.message,
            "params": case.params
        });
        if let Some(image_path) = case.image_path {
            body["image_path"] = Value::String(image_path);
        }

        let response = client
            .post(format!("{}/agent/chat", args.base_url.trim_end_matches('/')))
            .json(&body)
            .send()
            .await?;
        let http_status = response.status().as_u16();
        let payload: Value = response.json().await.unwrap_or_else(|error| {
            json!({"error": format!("invalid response JSON: {}", error)})
        });
        let execution_mode = payload["execution_mode"].as_str().map(str::to_string);
        let artifact_verified = payload["artifact_verified"].as_bool().unwrap_or(false);
        let trace = payload["output"]["result"]["pdca_trace"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let pdca_roles: Vec<String> = trace
            .iter()
            .filter_map(|phase| phase["role"].as_str().map(str::to_string))
            .collect();
        let ca_artifact_verified = trace
            .iter()
            .find(|phase| phase["role"].as_str() == Some("Check"))
            .map(|phase| {
                let mut paths = Vec::new();
                collect_media_paths(&phase["artifacts"], &mut paths);
                !paths.is_empty()
                    && paths.iter().all(|path| {
                        std::fs::metadata(path)
                            .map(|metadata| metadata.is_file() && metadata.len() > 0)
                            .unwrap_or(false)
                    })
            })
            .unwrap_or(false);
        let mut media_paths = Vec::new();
        collect_media_paths(&payload["output"], &mut media_paths);
        media_paths.sort();
        media_paths.dedup();
        let artifacts_exist = !media_paths.is_empty()
            && media_paths.iter().all(|path| {
                std::fs::metadata(path)
                    .map(|metadata| metadata.is_file() && metadata.len() > 0)
                    .unwrap_or(false)
            });
        let reported_success = response_status_is_success(&payload) && http_status < 400;
        let effective_prompt = payload["generation_audit"]["effective_prompt"]
            .as_str()
            .map(str::to_string);
        let pipeline = payload["generation_audit"]["pipeline"]
            .as_str()
            .map(str::to_string);
        let prompt_alignment = required_terms_match(
            effective_prompt.as_deref(),
            &case.required_prompt_terms,
        );
        let pipeline_alignment = case
            .required_pipeline
            .as_deref()
            .is_none_or(|required| pipeline.as_deref() == Some(required));
        let actual_width = payload["generation_audit"]["parameters"]["width"].as_u64();
        let actual_height = payload["generation_audit"]["parameters"]["height"].as_u64();
        let dimension_alignment = case
            .expected_width
            .is_none_or(|expected| actual_width == Some(expected))
            && case
                .expected_height
                .is_none_or(|expected| actual_height == Some(expected));
        let passed = if case.expected_success {
            reported_success
                && execution_mode.as_deref() == Some("gliding_horse")
                && artifact_verified
                && pdca_roles == ["Plan", "Do", "Check", "Act"]
                && ca_artifact_verified
                && artifacts_exist
                && prompt_alignment
                && pipeline_alignment
                && dimension_alignment
        } else {
            !reported_success
        };
        results.push(CaseResult {
            id: case.id,
            passed,
            http_status,
            execution_mode,
            artifact_verified,
            pdca_roles,
            ca_artifact_verified,
            turn_count: payload["turn_count"].as_u64().unwrap_or(0),
            tool_calls: payload["tool_calls"].as_u64().unwrap_or(0),
            prompt_alignment,
            pipeline_alignment,
            dimension_alignment,
            pipeline,
            effective_prompt,
            media_paths,
            quality_score: payload["generation_audit"]["quality_score"].as_f64(),
            gpu_tier: payload["generation_audit"]["parameters"]["gpu_tier"]
                .as_str()
                .map(str::to_string),
            error: payload["error"]
                .as_str()
                .or_else(|| payload["message"].as_str())
                .map(str::to_string),
        });
    }

    let total = results.len();
    let passed = results.iter().filter(|result| result.passed).count();
    let false_successes = results
        .iter()
        .filter(|result| {
            result.http_status < 400
                && !result.artifact_verified
                && result.error.is_none()
        })
        .count();
    let sum_turns: u64 = results.iter().map(|result| result.turn_count).sum();
    let sum_tool_calls: u64 = results.iter().map(|result| result.tool_calls).sum();
    let denominator = total.max(1) as f64;
    let scored: Vec<f64> = results
        .iter()
        .filter_map(|result| result.quality_score)
        .collect();
    let average_quality_score = if scored.is_empty() {
        None
    } else {
        Some(scored.iter().sum::<f64>() / scored.len() as f64)
    };
    let report = EvaluationReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        total,
        passed,
        false_successes,
        artifact_pass_rate: passed as f64 / denominator,
        average_turns: sum_turns as f64 / denominator,
        average_tool_calls: sum_tool_calls as f64 / denominator,
        average_quality_score,
        cases: results,
    };

    if let Some(parent) = Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    if report.passed != report.total || report.false_successes != 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn response_status_is_success(payload: &Value) -> bool {
    matches!(
        payload["status"].as_str().unwrap_or("").to_ascii_lowercase().as_str(),
        "success" | "completed" | "accepted"
    )
}

fn collect_media_paths(value: &Value, paths: &mut Vec<String>) {
    match value {
        Value::String(path) => {
            if is_exact_media_path(path) {
                paths.push(path.clone());
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_media_paths(value, paths);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_media_paths(value, paths);
            }
        }
        _ => {}
    }
}

fn is_exact_media_path(value: &str) -> bool {
    if value.trim() != value || value.chars().any(char::is_control) {
        return false;
    }
    let path = Path::new(value);
    if !path.is_absolute() && !value.starts_with("output/") {
        return false;
    }
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "gif" | "mp4" | "webm")
    )
}

fn default_true() -> bool {
    true
}

fn required_terms_match(prompt: Option<&str>, required_terms: &[String]) -> bool {
    if required_terms.is_empty() {
        return true;
    }
    let Some(prompt) = prompt else {
        return false;
    };
    let prompt = prompt.to_ascii_lowercase();
    required_terms
        .iter()
        .all(|term| prompt.contains(&term.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_alignment_requires_every_configured_term() {
        let terms = vec!["rain".to_string(), "neon".to_string(), "wet".to_string()];
        assert!(required_terms_match(
            Some("rainy night with neon reflections on wet pavement"),
            &terms
        ));
        assert!(!required_terms_match(Some("sunny city street"), &terms));
        assert!(!required_terms_match(None, &terms));
    }

    #[test]
    fn media_path_parser_rejects_natural_language_summary() {
        assert!(!is_exact_media_path("Accepted: /tmp/generated.png"));
        assert!(!is_exact_media_path("video saved to output/generated.mp4"));
    }

    #[test]
    fn media_path_parser_accepts_exact_output_paths() {
        assert!(is_exact_media_path("/tmp/generated image.png"));
        assert!(is_exact_media_path("output/generated.mp4"));
    }

    #[test]
    fn required_pipeline_rejects_a_different_execution_path() {
        let required = Some("native_t2i_keyframe_to_i2v");
        let actual = Some("native_t2v");
        assert!(!required.is_none_or(|value| actual == Some(value)));
    }
}
