// ComfyUI Rust Agent 系统初始化程序
// 创建部署目录、从 HuggingFace 下载模型、验证 sd-cli
//
// 用法:
//   cargo run --bin comfyui-init -- --check-only          # 仅检查环境
//   cargo run --bin comfyui-init -- --required-only       # 仅下载必需模型
//   cargo run --bin comfyui-init -- --force               # 强制重新下载
//   cargo run --bin comfyui-init                           # 完整初始化

use anyhow::{bail, Context, Result};
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const BANNER: &str = r#"
   ___                                ___ _    ___
  / __| ___  ___    __ _ _ _ __ _ __ | __| |  | _ )
 | (_ |/ _ \/ _ \  / _` | '_/ _` / /_/ _` |__| _ \
  \___|\___/\___/  \__,_|_| \__,_\__/\__,_|__|___/
   Rust Agent — 系统初始化程序
"#;

// ============================================================================
// CLI 参数定义
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "comfyui-init", version, about = "ComfyUI Rust Agent 系统初始化：创建目录、下载模型、验证 sd-cli")]
struct Cli {
    /// 配置文件路径
    #[arg(long, default_value = "config/models.yaml")]
    config: PathBuf,

    /// 仅下载必需模型（required: true 的模型）
    #[arg(long)]
    required_only: bool,

    /// 强制重新下载（即使目标文件已存在）
    #[arg(long)]
    force: bool,

    /// 跳过 sd-cli 验证
    #[arg(long)]
    skip_sd_check: bool,

    /// 仅检查环境与配置，不执行下载和目录创建
    #[arg(long)]
    check_only: bool,
}

// ============================================================================
// 配置结构（对应 config/models.yaml）
// ============================================================================

#[derive(Debug, Deserialize)]
struct ModelsConfig {
    #[serde(default)]
    huggingface: HfConfig,
    #[serde(default)]
    models: Vec<ModelEntry>,
    #[serde(default)]
    directories: Vec<String>,
    #[serde(default)]
    sd_cli: SdCliConfig,
}

#[derive(Debug, Deserialize, Default)]
struct HfConfig {
    #[serde(default)]
    token: String,
    #[serde(default)]
    endpoint: String,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    name: String,
    repo: String,
    #[serde(default = "default_revision")]
    revision: String,
    filename: String,
    target_dir: String,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    size_hint: String,
}

fn default_revision() -> String {
    "main".to_string()
}

#[derive(Debug, Deserialize)]
struct SdCliConfig {
    #[serde(default = "default_sd_path")]
    expected_path: String,
    #[serde(default = "default_version_arg")]
    version_arg: String,
    #[serde(default = "default_min_version")]
    #[allow(dead_code)]
    min_version: String,
}

fn default_sd_path() -> String {
    "/usr/local/bin/sd-cli".to_string()
}
fn default_version_arg() -> String {
    "--version".to_string()
}
fn default_min_version() -> String {
    "0.0.1".to_string()
}

impl Default for SdCliConfig {
    fn default() -> Self {
        Self {
            expected_path: default_sd_path(),
            version_arg: default_version_arg(),
            min_version: default_min_version(),
        }
    }
}

// ============================================================================
// 环境变量替换
// ============================================================================

/// 替换字符串中的 ${VAR} 和 ${VAR:-default} 环境变量引用
fn substitute_env_vars(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut result = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1] == '{' {
            let rest: String = chars[i + 2..].iter().collect();
            if let Some(end) = rest.find('}') {
                let expr = &rest[..end];
                let (var, default_val) = match expr.find(":-") {
                    Some(pos) => (&expr[..pos], Some(&expr[pos + 2..])),
                    None => (expr, None),
                };
                let value = std::env::var(var)
                    .ok()
                    .filter(|v| !v.is_empty())
                    .or_else(|| default_val.map(|d| d.to_string()));
                result.push_str(&value.unwrap_or_default());
                i += 2 + end + 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

// ============================================================================
// 配置加载
// ============================================================================

fn load_config(path: &Path) -> Result<ModelsConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("无法读取配置文件: {}", path.display()))?;
    let substituted = substitute_env_vars(&content);
    let config: ModelsConfig = serde_yaml::from_str(&substituted)
        .with_context(|| format!("配置文件解析失败: {}", path.display()))?;
    Ok(config)
}

// ============================================================================
// 目录创建
// ============================================================================

fn create_directories(dirs: &[String]) -> Result<usize> {
    let mut created = 0;
    for dir in dirs {
        let path = Path::new(dir);
        if !path.exists() {
            std::fs::create_dir_all(path)
                .with_context(|| format!("创建目录失败: {}", dir))?;
            println!("  + 创建目录: {}", dir);
            created += 1;
        } else {
            println!("  ✓ 目录已存在: {}", dir);
        }
    }
    Ok(created)
}

// ============================================================================
// 模型下载
// ============================================================================

struct DownloadOutcome {
    name: String,
    status: DownloadStatus,
}

enum DownloadStatus {
    Skipped(String),
    Downloaded(u64),
    Failed(String),
}

/// 格式化字节数为人类可读字符串
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// 构建进度条样式
fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
    )
    .unwrap()
    .progress_chars("=>-")
}

/// 下载单个模型（支持断点续传）
async fn download_model(
    client: &reqwest::Client,
    model: &ModelEntry,
    hf: &HfConfig,
    force: bool,
) -> Result<DownloadOutcome> {
    let target_dir = PathBuf::from(&model.target_dir);
    let target_path = target_dir.join(&model.filename);
    let part_path = target_dir.join(format!("{}.part", model.filename));

    // 已存在则跳过（除非 --force）
    if target_path.exists() && !force {
        let size = std::fs::metadata(&target_path).map(|m| m.len()).unwrap_or(0);
        return Ok(DownloadOutcome {
            name: model.name.clone(),
            status: DownloadStatus::Skipped(format!("已存在 ({})", format_bytes(size))),
        });
    }

    // 构建 HuggingFace 下载 URL
    let endpoint = if hf.endpoint.is_empty() {
        "https://huggingface.co"
    } else {
        hf.endpoint.trim_end_matches('/')
    };
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint, model.repo, model.revision, model.filename
    );

    // 断点续传：检查 .part 文件已有大小
    let mut start_offset: u64 = 0;
    if part_path.exists() && !force {
        start_offset = std::fs::metadata(&part_path)
            .map(|m| m.len())
            .unwrap_or(0);
    }

    // 构建请求
    let mut req = client
        .get(&url)
        .header("Accept", "application/octet-stream");

    if !hf.token.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", hf.token));
    }
    if start_offset > 0 {
        req = req.header("Range", format!("bytes={}-", start_offset));
    }

    let resp = req.send().await.with_context(|| format!("请求失败: {}", url))?;
    let status = resp.status();

    // 206 = 续传成功; 200 = 服务器忽略 Range（从头开始）
    let is_resume = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !status.is_success() {
        return Ok(DownloadOutcome {
            name: model.name.clone(),
            status: DownloadStatus::Failed(format!("HTTP {} — 检查 repo/token 是否正确", status)),
        });
    }

    let chunk_size = resp.content_length().unwrap_or(0);
    let total_size = if is_resume {
        chunk_size + start_offset
    } else {
        chunk_size
    };

    let display_total = if total_size > 0 {
        total_size
    } else if !model.size_hint.is_empty() {
        0 // 未知大小，进度条用非确定模式
    } else {
        0
    };

    let pb = if display_total > 0 {
        let pb = ProgressBar::new(display_total);
        pb.set_style(progress_style());
        pb.set_message(model.name.clone());
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("  {spinner:.green} {msg} {bytes}")
                .unwrap(),
        );
        pb.set_message(format!("下载 {} (大小未知{})", model.name, if !model.size_hint.is_empty() { format!(", 预计 {}", model.size_hint) } else { String::new() }));
        pb
    };

    // 打开 .part 文件（续传则追加，否则覆盖）
    let mut file = if is_resume && start_offset > 0 {
        OpenOptions::new().append(true).open(&part_path)
    } else {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&part_path)
    }
    .with_context(|| format!("无法写入临时文件: {}", part_path.display()))?;

    // 流式写入
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.with_context(|| "下载流中断")?;
        file.write_all(&chunk).with_context(|| "写入文件失败")?;
        let len = chunk.len() as u64;
        downloaded += len;
        if display_total > 0 {
            pb.inc(len);
        } else {
            pb.inc(len);
        }
    }
    file.flush()?;
    drop(file);
    pb.finish_and_clear();

    // 重命名 .part → 最终文件名
    std::fs::rename(&part_path, &target_path)
        .with_context(|| format!("重命名失败: {} → {}", part_path.display(), target_path.display()))?;

    let actual_size = if is_resume {
        start_offset + downloaded
    } else {
        downloaded
    };

    println!(
        "  ✓ 下载完成: {} ({})",
        model.name,
        format_bytes(actual_size)
    );

    Ok(DownloadOutcome {
        name: model.name.clone(),
        status: DownloadStatus::Downloaded(actual_size),
    })
}

// ============================================================================
// sd-cli 验证
// ============================================================================

struct SdCliCheckResult {
    #[allow(dead_code)]
    path_exists: bool,
    version_output: Option<String>,
    error: Option<String>,
}

fn check_sd_cli(sd_cli: &SdCliConfig) -> SdCliCheckResult {
    let path = Path::new(&sd_cli.expected_path);

    if !path.exists() {
        return SdCliCheckResult {
            path_exists: false,
            version_output: None,
            error: Some(format!("sd-cli 不存在于: {}", sd_cli.expected_path)),
        };
    }

    let output = Command::new(&sd_cli.expected_path)
        .arg(&sd_cli.version_arg)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if out.status.success() {
                let version_text = if !stdout.is_empty() { stdout } else { stderr };
                SdCliCheckResult {
                    path_exists: true,
                    version_output: Some(version_text),
                    error: None,
                }
            } else {
                SdCliCheckResult {
                    path_exists: true,
                    version_output: None,
                    error: Some(format!(
                        "sd-cli 退出码 {:?}: {}",
                        out.status.code(),
                        if stderr.is_empty() { &stdout } else { &stderr }
                    )),
                }
            }
        }
        Err(e) => SdCliCheckResult {
            path_exists: true,
            version_output: None,
            error: Some(format!("无法执行 sd-cli: {}", e)),
        },
    }
}

// ============================================================================
// 主函数
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", BANNER);

    // --- 加载配置 ---
    println!("\n[1/4] 加载配置...");
    let config = load_config(&cli.config)?;
    let hf_endpoint = if config.huggingface.endpoint.is_empty() {
        "https://huggingface.co".to_string()
    } else {
        config.huggingface.endpoint.clone()
    };
    let has_token = !config.huggingface.token.is_empty();

    println!("  配置文件: {}", cli.config.display());
    println!("  HF Endpoint: {}", hf_endpoint);
    println!("  HF Token: {}", if has_token { "已配置" } else { "未配置（部分模型可能无法下载）" });
    println!("  模型总数: {} (必需: {})", config.models.len(), config.models.iter().filter(|m| m.required).count());
    println!("  目录总数: {}", config.directories.len());
    println!("  sd-cli 路径: {}", config.sd_cli.expected_path);

    // --- 检查模式 ---
    if cli.check_only {
        println!("\n[检查模式] 仅验证环境，不执行修改\n");

        println!("[目录检查]");
        for dir in &config.directories {
            let exists = Path::new(dir).exists();
            println!("  {} {}", if exists { "✓" } else { "✗" }, dir);
        }

        println!("\n[模型检查]");
        for model in &config.models {
            if cli.required_only && !model.required {
                continue;
            }
            let target = PathBuf::from(&model.target_dir).join(&model.filename);
            let exists = target.exists();
            let size_str = if exists {
                format_bytes(std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0))
            } else if !model.size_hint.is_empty() {
                model.size_hint.clone()
            } else {
                "?".to_string()
            };
            let required_tag = if model.required { "必需" } else { "可选" };
            println!(
                "  {} {} [{}] — {} ({})",
                if exists { "✓" } else { "✗" },
                model.name,
                required_tag,
                model.repo,
                size_str
            );
        }

        println!("\n[sd-cli 检查]");
        let sd_result = check_sd_cli(&config.sd_cli);
        match (&sd_result.version_output, &sd_result.error) {
            (Some(ver), _) => println!("  ✓ sd-cli 可用: {}", ver),
            (_, Some(err)) => println!("  ✗ {}", err),
            _ => println!("  ? sd-cli 状态未知"),
        }

        println!("\n检查完成。使用不带 --check-only 运行以执行初始化。");
        return Ok(());
    }

    // --- 创建目录 ---
    println!("\n[2/4] 创建系统目录...");
    let dirs_created = create_directories(&config.directories)?;
    println!("  完成: 新建 {} 个目录", dirs_created);

    // --- 下载模型 ---
    println!("\n[3/4] 下载模型...");

    let client = reqwest::Client::builder()
        .user_agent("comfyui-init/0.1")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("无法创建 HTTP 客户端")?;

    let models_to_download: Vec<&ModelEntry> = config.models.iter()
        .filter(|m| !cli.required_only || m.required)
        .collect();

    let mut outcomes: Vec<DownloadOutcome> = Vec::new();

    for model in &models_to_download {
        println!("\n  模型: {} ({})", model.name, model.repo);
        let required_tag = if model.required { "必需" } else { "可选" };
        println!("  类型: {} | 目标: {}/{}", required_tag, model.target_dir, model.filename);

        match download_model(&client, model, &config.huggingface, cli.force).await {
            Ok(outcome) => outcomes.push(outcome),
            Err(e) => {
                let msg = format!("{:#}", e);
                eprintln!("  ✗ 下载失败: {}", msg);
                outcomes.push(DownloadOutcome {
                    name: model.name.clone(),
                    status: DownloadStatus::Failed(msg),
                });
            }
        }
    }

    // --- sd-cli 验证 ---
    let sd_cli_ok = if cli.skip_sd_check {
        println!("\n[4/4] sd-cli 验证已跳过 (--skip-sd-check)");
        None
    } else {
        println!("\n[4/4] 验证 sd-cli...");
        let result = check_sd_cli(&config.sd_cli);
        match (&result.version_output, &result.error) {
            (Some(ver), _) => {
                println!("  ✓ sd-cli 可用: {}", ver);
                Some(true)
            }
            (_, Some(err)) => {
                eprintln!("  ✗ {}", err);
                Some(false)
            }
            _ => Some(false),
        }
    };

    // --- 汇总报告 ---
    println!("\n{}", "=".repeat(60));
    println!("初始化汇总报告");
    println!("{}", "=".repeat(60));

    let downloaded = outcomes.iter().filter(|o| matches!(o.status, DownloadStatus::Downloaded(_))).count();
    let skipped = outcomes.iter().filter(|o| matches!(o.status, DownloadStatus::Skipped(_))).count();
    let failed = outcomes.iter().filter(|o| matches!(o.status, DownloadStatus::Failed(_))).count();

    for outcome in &outcomes {
        match &outcome.status {
            DownloadStatus::Downloaded(size) => {
                println!("  ✓ {} — 下载完成 ({})", outcome.name, format_bytes(*size));
            }
            DownloadStatus::Skipped(reason) => {
                println!("  → {} — 跳过 ({})", outcome.name, reason);
            }
            DownloadStatus::Failed(reason) => {
                println!("  ✗ {} — 失败: {}", outcome.name, reason);
            }
        }
    }

    println!("\n  目录创建: {} 个", dirs_created);
    println!("  模型下载: {} 成功 / {} 跳过 / {} 失败", downloaded, skipped, failed);

    if let Some(ok) = sd_cli_ok {
        println!("  sd-cli: {}", if ok { "可用" } else { "不可用" });
    }

    let overall_ok = failed == 0 && sd_cli_ok != Some(false);
    println!("\n  总体状态: {}", if overall_ok { "✓ 初始化成功" } else { "✗ 存在问题（见上方详情）" });

    if !overall_ok {
        // 返回非零退出码但提供有用信息
        eprintln!("\n提示:");
        if failed > 0 {
            eprintln!("  - 检查 HF_TOKEN 环境变量是否已设置（部分模型需要授权）");
            eprintln!("  - 检查网络连接和 HF_ENDPOINT 是否可达");
        }
        if sd_cli_ok == Some(false) {
            eprintln!("  - 安装 sd-cli: 参考 stable-diffusion.cpp 项目构建说明");
        }
        bail!("初始化未完全成功");
    }

    println!("\n初始化完成。现在可以运行 comfyui-server 启动服务。");
    Ok(())
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_env_vars_no_var() {
        let result = substitute_env_vars("hello world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_substitute_env_vars_simple() {
        std::env::set_var("TEST_INIT_VAR", "myvalue");
        let result = substitute_env_vars("value: ${TEST_INIT_VAR}");
        assert_eq!(result, "value: myvalue");
        std::env::remove_var("TEST_INIT_VAR");
    }

    #[test]
    fn test_substitute_env_vars_with_default() {
        std::env::remove_var("TEST_INIT_MISSING");
        let result = substitute_env_vars("ep: ${TEST_INIT_MISSING:-https://default.com}");
        assert_eq!(result, "ep: https://default.com");
    }

    #[test]
    fn test_substitute_env_vars_empty_default() {
        std::env::remove_var("TEST_INIT_MISSING2");
        let result = substitute_env_vars("token: ${TEST_INIT_MISSING2:-}");
        assert_eq!(result, "token: ");
    }

    #[test]
    fn test_substitute_env_vars_env_overrides_default() {
        std::env::set_var("TEST_INIT_OVERRIDE", "actual");
        let result = substitute_env_vars("v: ${TEST_INIT_OVERRIDE:-fallback}");
        assert_eq!(result, "v: actual");
        std::env::remove_var("TEST_INIT_OVERRIDE");
    }

    #[test]
    fn test_substitute_env_vars_multiple() {
        std::env::set_var("TEST_INIT_A", "AAA");
        let result = substitute_env_vars("${TEST_INIT_A}-${TEST_INIT_B:-BBB}");
        assert_eq!(result, "AAA-BBB");
        std::env::remove_var("TEST_INIT_A");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 4 + 300 * 1024 * 1024), "4.3 GB");
    }

    #[test]
    fn test_parse_models_config() {
        let yaml = r#"
huggingface:
  token: "test-token"
  endpoint: "https://hf-mirror.com"
models:
  - name: "test-model"
    repo: "org/repo"
    revision: "main"
    filename: "model.safetensors"
    target_dir: "models/test"
    required: true
    size_hint: "1.0GB"
directories:
  - "models/test"
  - "output"
sd_cli:
  expected_path: "/usr/bin/sd"
  version_arg: "--version"
  min_version: "0.0.1"
"#;
        let config: ModelsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.huggingface.token, "test-token");
        assert_eq!(config.huggingface.endpoint, "https://hf-mirror.com");
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].name, "test-model");
        assert!(config.models[0].required);
        assert_eq!(config.models[0].revision, "main");
        assert_eq!(config.directories.len(), 2);
        assert_eq!(config.sd_cli.expected_path, "/usr/bin/sd");
    }

    #[test]
    fn test_parse_models_config_with_defaults() {
        let yaml = r#"
huggingface:
  token: ""
  endpoint: ""
models:
  - name: "m"
    repo: "r"
    filename: "f.bin"
    target_dir: "d"
"#;
        let config: ModelsConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.huggingface.token.is_empty());
        assert!(config.models[0].revision == "main"); // default
        assert!(!config.models[0].required); // default false
        assert!(config.models[0].size_hint.is_empty()); // default empty
        // sd_cli uses defaults
        assert_eq!(config.sd_cli.expected_path, "/usr/local/bin/sd-cli");
    }

    #[test]
    fn test_parse_empty_config() {
        let yaml = "";
        let config: ModelsConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.models.is_empty());
        assert!(config.directories.is_empty());
    }
}
