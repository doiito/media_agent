//! Native runtime discovery and model-container validation.
//!
//! This module deliberately performs only cheap, deterministic checks. Model
//! capability checks still happen inside stable-diffusion.cpp/llama.cpp when a
//! context is created.

use crate::config::{AgentLlmProvider, AppConfig};
use serde::Serialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_SAFETENSORS_HEADER: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelContainer {
    Safetensors,
    Gguf,
    TorchZip,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelFileInfo {
    pub path: String,
    pub size: u64,
    pub container: ModelContainer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCheckStatus {
    Ready,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCheck {
    pub id: String,
    pub status: RuntimeCheckStatus,
    pub path: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeRuntimeReport {
    pub ready: bool,
    pub python_required: bool,
    pub agent_llm_provider: AgentLlmProvider,
    pub checks: Vec<RuntimeCheck>,
}

impl NativeRuntimeReport {
    pub fn inspect(config: &AppConfig) -> Self {
        let mut checks = Vec::new();

        checks.push(check_directory(
            "stable_diffusion_source",
            &config.sd_cpp.source_path,
            "include/stable-diffusion.h",
        ));
        checks.push(check_file(
            "stable_diffusion_executable",
            &config.sd_cpp.executable_path,
            config.sd_cpp.execution_mode != "cli",
        ));
        if config.sd_cpp.execution_mode == "native_worker" {
            checks.push(check_file(
                "stable_diffusion_rust_worker",
                &config.sd_cpp.worker_path,
                false,
            ));
            checks.push(check_file(
                "stable_diffusion_c_bridge",
                &config.sd_cpp.bridge_library_path,
                false,
            ));
        }
        checks.push(check_directory(
            "llama_cpp_source",
            &config.llama_cpp.source_path,
            "include/llama.h",
        ));
        checks.push(check_file(
            "llama_server",
            &config.llama_cpp.server_path,
            config.agent.llm.provider != AgentLlmProvider::LlamaCpp,
        ));

        if !config.sd_cpp.model_path.trim().is_empty() {
            checks.push(check_model(
                "default_diffusion_model",
                &config.sd_cpp.model_path,
                ModelContainer::Safetensors,
                false,
            ));
        }

        if !config.sd_cpp.video_model_path.trim().is_empty() {
            checks.push(check_native_model(
                "default_video_model",
                &config.sd_cpp.video_model_path,
                false,
            ));
        }

        if !config.sd_cpp.svd_model_path.trim().is_empty() {
            checks.push(check_native_model(
                "svd_video_model",
                &config.sd_cpp.svd_model_path,
                true,
            ));
        }

        if !config.sd_cpp.video_t5xxl_path.trim().is_empty() {
            checks.push(check_native_model(
                "video_t5xxl_model",
                &config.sd_cpp.video_t5xxl_path,
                false,
            ));
        }

        if !config.sd_cpp.video_vae_path.trim().is_empty() {
            checks.push(check_native_model(
                "video_vae_model",
                &config.sd_cpp.video_vae_path,
                false,
            ));
        }

        if !config.sd_cpp.video_high_noise_model_path.trim().is_empty() {
            checks.push(check_native_model(
                "video_high_noise_model",
                &config.sd_cpp.video_high_noise_model_path,
                false,
            ));
        }

        if !config.sd_cpp.clip_vision_path.trim().is_empty() {
            checks.push(check_model(
                "clip_vision_model",
                &config.sd_cpp.clip_vision_path,
                ModelContainer::Safetensors,
                true,
            ));
        }

        if config.agent.llm.provider == AgentLlmProvider::LlamaCpp {
            if config.llama_cpp.model_path.trim().is_empty() {
                checks.push(RuntimeCheck {
                    id: "llama_model".to_string(),
                    status: RuntimeCheckStatus::Warning,
                    path: None,
                    detail: "No llama.cpp model_path configured; llama-server must be started with an external GGUF model".to_string(),
                });
            } else {
                checks.push(check_model(
                    "llama_model",
                    &config.llama_cpp.model_path,
                    ModelContainer::Gguf,
                    false,
                ));
            }
        }

        let ready = checks
            .iter()
            .all(|check| check.status != RuntimeCheckStatus::Error);

        Self {
            ready,
            python_required: false,
            agent_llm_provider: config.agent.llm.provider,
            checks,
        }
    }
}

pub fn inspect_model_file(path: impl AsRef<Path>) -> Result<ModelFileInfo, String> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .map_err(|e| format!("cannot open model '{}': {}", path.display(), e))?;
    let size = file
        .metadata()
        .map_err(|e| format!("cannot stat model '{}': {}", path.display(), e))?
        .len();
    if size == 0 {
        return Err(format!("model '{}' is empty", path.display()));
    }

    let mut magic = [0_u8; 8];
    let read = file
        .read(&mut magic)
        .map_err(|e| format!("cannot read model '{}': {}", path.display(), e))?;

    let container = if read >= 4 && &magic[..4] == b"GGUF" {
        if valid_gguf_header(&mut file, size, &magic) {
            ModelContainer::Gguf
        } else {
            ModelContainer::Unknown
        }
    } else if read >= 4 && &magic[..4] == b"PK\x03\x04" {
        ModelContainer::TorchZip
    } else if read == 8 {
        let header_len = u64::from_le_bytes(magic);
        if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER || header_len + 8 > size {
            ModelContainer::Unknown
        } else {
            let mut header = vec![0_u8; header_len as usize];
            file.seek(SeekFrom::Start(8))
                .and_then(|_| file.read_exact(&mut header))
                .map_err(|e| format!("cannot read safetensors header '{}': {}", path.display(), e))?;
            let data_size = size - 8 - header_len;
            if valid_safetensors_header(&header, data_size) {
                ModelContainer::Safetensors
            } else {
                ModelContainer::Unknown
            }
        }
    } else {
        ModelContainer::Unknown
    };

    Ok(ModelFileInfo {
        path: path.display().to_string(),
        size,
        container,
    })
}

fn valid_safetensors_header(header: &[u8], data_size: u64) -> bool {
    let Ok(serde_json::Value::Object(entries)) =
        serde_json::from_slice::<serde_json::Value>(header)
    else {
        return false;
    };

    let mut tensor_count = 0_u64;
    for (name, value) in entries {
        if name == "__metadata__" {
            continue;
        }
        let Some(tensor) = value.as_object() else {
            return false;
        };
        let Some(offsets) = tensor.get("data_offsets").and_then(|value| value.as_array()) else {
            return false;
        };
        if offsets.len() != 2 {
            return false;
        }
        let (Some(start), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
            return false;
        };
        if end < start || end > data_size {
            return false;
        }
        if tensor.get("dtype").and_then(|value| value.as_str()).is_none()
            || tensor.get("shape").and_then(|value| value.as_array()).is_none()
        {
            return false;
        }
        tensor_count += 1;
    }
    tensor_count > 0
}

fn valid_gguf_header(file: &mut File, size: u64, first_eight: &[u8; 8]) -> bool {
    if size < 24 {
        return false;
    }
    let version = u32::from_le_bytes(first_eight[4..8].try_into().unwrap());
    if !matches!(version, 2 | 3) {
        return false;
    }
    let mut counts = [0_u8; 16];
    if file.read_exact(&mut counts).is_err() {
        return false;
    }
    let tensor_count = u64::from_le_bytes(counts[..8].try_into().unwrap());
    tensor_count > 0
}

pub fn require_model_container(
    path: impl AsRef<Path>,
    expected: ModelContainer,
) -> Result<ModelFileInfo, String> {
    let info = inspect_model_file(path)?;
    if info.container != expected {
        return Err(format!(
            "model '{}' has {:?} container, expected {:?}",
            info.path, info.container, expected
        ));
    }
    Ok(info)
}

fn check_directory(id: &str, root: &str, required_child: &str) -> RuntimeCheck {
    let child = PathBuf::from(root).join(required_child);
    if child.is_file() {
        RuntimeCheck {
            id: id.to_string(),
            status: RuntimeCheckStatus::Ready,
            path: Some(root.to_string()),
            detail: format!("found {}", child.display()),
        }
    } else {
        RuntimeCheck {
            id: id.to_string(),
            status: RuntimeCheckStatus::Error,
            path: Some(root.to_string()),
            detail: format!("missing {}", child.display()),
        }
    }
}

fn check_file(id: &str, path: &str, optional: bool) -> RuntimeCheck {
    if Path::new(path).is_file() {
        RuntimeCheck {
            id: id.to_string(),
            status: RuntimeCheckStatus::Ready,
            path: Some(path.to_string()),
            detail: "file is available".to_string(),
        }
    } else {
        RuntimeCheck {
            id: id.to_string(),
            status: if optional {
                RuntimeCheckStatus::Warning
            } else {
                RuntimeCheckStatus::Error
            },
            path: Some(path.to_string()),
            detail: "file is missing".to_string(),
        }
    }
}

fn check_model(
    id: &str,
    path: &str,
    expected: ModelContainer,
    optional: bool,
) -> RuntimeCheck {
    match require_model_container(path, expected) {
        Ok(info) => RuntimeCheck {
            id: id.to_string(),
            status: RuntimeCheckStatus::Ready,
            path: Some(path.to_string()),
            detail: format!("{:?}, {} bytes", info.container, info.size),
        },
        Err(error) => RuntimeCheck {
            id: id.to_string(),
            status: if optional {
                RuntimeCheckStatus::Warning
            } else {
                RuntimeCheckStatus::Error
            },
            path: Some(path.to_string()),
            detail: error,
        },
    }
}

fn check_native_model(id: &str, path: &str, optional: bool) -> RuntimeCheck {
    match inspect_model_file(path) {
        Ok(info)
            if matches!(
                info.container,
                ModelContainer::Safetensors | ModelContainer::Gguf
            ) =>
        {
            RuntimeCheck {
                id: id.to_string(),
                status: RuntimeCheckStatus::Ready,
                path: Some(path.to_string()),
                detail: format!("{:?}, {} bytes", info.container, info.size),
            }
        }
        Ok(info) => RuntimeCheck {
            id: id.to_string(),
            status: if optional {
                RuntimeCheckStatus::Warning
            } else {
                RuntimeCheckStatus::Error
            },
            path: Some(path.to_string()),
            detail: format!("unsupported {:?} model container", info.container),
        },
        Err(error) => RuntimeCheck {
            id: id.to_string(),
            status: if optional {
                RuntimeCheckStatus::Warning
            } else {
                RuntimeCheckStatus::Error
            },
            path: Some(path.to_string()),
            detail: error,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_torch_zip_renamed_as_safetensors() {
        let path = std::env::temp_dir().join(format!("model-{}.safetensors", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"PK\x03\x04not-a-safetensors-file").unwrap();
        let info = inspect_model_file(&path).unwrap();
        assert_eq!(info.container, ModelContainer::TorchZip);
        assert!(require_model_container(&path, ModelContainer::Safetensors).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn detects_minimal_safetensors_header() {
        let path = std::env::temp_dir().join(format!("model-{}.safetensors", uuid::Uuid::new_v4()));
        let header = br#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0_u8; 4]).unwrap();
        drop(file);
        let info = inspect_model_file(&path).unwrap();
        assert_eq!(info.container, ModelContainer::Safetensors);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_truncated_safetensors_tensor_data() {
        let path = std::env::temp_dir().join(format!("model-{}.safetensors", uuid::Uuid::new_v4()));
        let header = br#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0_u8; 2]).unwrap();
        drop(file);
        let info = inspect_model_file(&path).unwrap();
        assert_eq!(info.container, ModelContainer::Unknown);
        let _ = std::fs::remove_file(path);
    }
}
