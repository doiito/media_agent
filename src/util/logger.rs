// 日志工具
// 支持基于配置的日志级别、模块过滤、格式化

use crate::config::LogConfig;
use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;

/// 初始化日志（使用默认级别 info）
pub fn init_logger() {
    init_logger_with_level("info");
}

/// 使用指定级别初始化日志
pub fn init_logger_with_level(level: &str) {
    let config = LogConfig {
        level: level.to_string(),
        ..Default::default()
    };
    init_logger_with_config(&config);
}

/// 使用配置初始化日志
pub fn init_logger_with_config(config: &LogConfig) {
    let level_filter = parse_level(&config.level);

    let mut builder = Builder::from_default_env();
    builder.filter_level(level_filter);

    // 模块过滤：RUST_LOG=module=level
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        // RUST_LOG 已经被 from_default_env 处理，但显式覆盖级别需要重新设置
        for part in rust_log.split(',') {
            if let Some(idx) = part.find('=') {
                let module = &part[..idx];
                let level_str = &part[idx + 1..];
                if let Some(level) = parse_level_optional(level_str) {
                    builder.filter_module(module, level);
                }
            }
        }
    }

    // 格式化
    if config.timestamp && config.module_path {
        builder.format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] {} - {}",
                buf.timestamp_seconds(),
                record.level(),
                record.target(),
                record.args()
            )
        });
    } else if config.timestamp {
        builder.format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] {}",
                buf.timestamp_seconds(),
                record.level(),
                record.args()
            )
        });
    } else if config.module_path {
        builder.format(|buf, record| {
            writeln!(
                buf,
                "[{}] {} - {}",
                record.level(),
                record.target(),
                record.args()
            )
        });
    } else {
        builder.format(|buf, record| {
            writeln!(buf, "[{}] {}", record.level(), record.args())
        });
    }

    // 日志文件（可选）
    if let Some(file_path) = &config.file {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
        {
            Ok(file) => {
                builder.target(env_logger::Target::Pipe(Box::new(file)));
            }
            Err(e) => {
                eprintln!("Failed to open log file {}: {}", file_path, e);
            }
        }
    }

    builder.try_init().ok();
}

/// 解析日志级别字符串
fn parse_level(s: &str) -> LevelFilter {
    parse_level_optional(s).unwrap_or(LevelFilter::Info)
}

fn parse_level_optional(s: &str) -> Option<LevelFilter> {
    match s.to_lowercase().as_str() {
        "trace" => Some(LevelFilter::Trace),
        "debug" => Some(LevelFilter::Debug),
        "info" => Some(LevelFilter::Info),
        "warn" | "warning" => Some(LevelFilter::Warn),
        "error" => Some(LevelFilter::Error),
        "off" => Some(LevelFilter::Off),
        _ => None,
    }
}

/// 获取当前日志级别（运行时检查）
pub fn current_level() -> LevelFilter {
    log::max_level()
}

/// 检查指定级别是否启用
pub fn is_level_enabled(level: LevelFilter) -> bool {
    log::max_level() >= level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_level() {
        assert_eq!(parse_level("trace"), LevelFilter::Trace);
        assert_eq!(parse_level("DEBUG"), LevelFilter::Debug);
        assert_eq!(parse_level("info"), LevelFilter::Info);
        assert_eq!(parse_level("warning"), LevelFilter::Warn);
        assert_eq!(parse_level("error"), LevelFilter::Error);
        assert_eq!(parse_level("off"), LevelFilter::Off);
        assert_eq!(parse_level("invalid"), LevelFilter::Info);
    }

    #[test]
    fn test_parse_level_optional() {
        assert_eq!(parse_level_optional("debug"), Some(LevelFilter::Debug));
        assert_eq!(parse_level_optional("invalid"), None);
    }
}
