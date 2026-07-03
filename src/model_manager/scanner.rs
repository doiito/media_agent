// 模型扫描器
// 自动发现和索引 models/ 目录下的所有模型文件

use crate::model_manager::model_info::{ModelInfo, ModelFormat, ModelType};
use log::{info, debug, warn};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// 模型扫描结果
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// 发现的模型列表
    pub models: Vec<ModelInfo>,
    /// 扫描的目录数
    pub directories_scanned: usize,
    /// 跳过的文件数（不支持的格式等）
    pub files_skipped: usize,
    /// 扫描耗时（毫秒）
    pub elapsed_ms: u64,
    /// 总大小（字节）
    pub total_size_bytes: u64,
}

impl ScanResult {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            directories_scanned: 0,
            files_skipped: 0,
            elapsed_ms: 0,
            total_size_bytes: 0,
        }
    }

    /// 按类型分组模型
    pub fn group_by_type(&self) -> std::collections::HashMap<ModelType, Vec<&ModelInfo>> {
        let mut groups: std::collections::HashMap<ModelType, Vec<&ModelInfo>> =
            std::collections::HashMap::new();
        for model in &self.models {
            groups.entry(model.model_type.clone())
                .or_default()
                .push(model);
        }
        groups
    }

    /// 获取扫描摘要
    pub fn summary(&self) -> String {
        let groups = self.group_by_type();
        let mut lines = Vec::new();
        lines.push(format!(
            "扫描完成: {} 个模型, {} 个目录, 跳过 {} 个文件, 耗时 {}ms",
            self.models.len(),
            self.directories_scanned,
            self.files_skipped,
            self.elapsed_ms
        ));
        lines.push(format!("总大小: {}", crate::model_manager::model_info::format_size(self.total_size_bytes)));
        for (model_type, models) in groups {
            let size: u64 = models.iter().map(|m| m.size_bytes).sum();
            lines.push(format!(
                "  {}: {} 个, {}",
                model_type,
                models.len(),
                crate::model_manager::model_info::format_size(size)
            ));
        }
        lines.join("\n")
    }
}

impl Default for ScanResult {
    fn default() -> Self {
        Self::new()
    }
}

/// 模型扫描器
pub struct ModelScanner {
    /// 模型根目录
    models_dir: PathBuf,
    /// 支持的文件扩展名
    supported_extensions: Vec<String>,
    /// 要忽略的文件名模式
    ignore_patterns: Vec<String>,
}

impl ModelScanner {
    /// 创建新的扫描器
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            supported_extensions: ModelFormat::supported_extensions()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ignore_patterns: vec![
                ".tmp".to_string(),
                ".bak".to_string(),
                ".lock".to_string(),
                "temp_".to_string(),
            ],
        }
    }

    /// 获取模型根目录
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// 添加支持的扩展名
    pub fn with_extension(mut self, ext: impl Into<String>) -> Self {
        self.supported_extensions.push(ext.into());
        self
    }

    /// 添加忽略模式
    pub fn with_ignore_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.ignore_patterns.push(pattern.into());
        self
    }

    /// 扫描所有模型
    pub fn scan(&self) -> Result<ScanResult, crate::model_manager::model_info::ModelManagerError> {
        let start = std::time::Instant::now();
        let mut result = ScanResult::new();

        if !self.models_dir.exists() {
            warn!("Models directory does not exist: {}", self.models_dir.display());
            result.elapsed_ms = start.elapsed().as_millis() as u64;
            return Ok(result);
        }

        info!("开始扫描模型目录: {}", self.models_dir.display());

        for entry in WalkDir::new(&self.models_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_dir() {
                result.directories_scanned += 1;
                continue;
            }

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(f) => f,
                None => {
                    result.files_skipped += 1;
                    continue;
                }
            };

            // 检查忽略模式
            if self.should_ignore(filename) {
                debug!("跳过忽略文件: {}", filename);
                result.files_skipped += 1;
                continue;
            }

            // 检查扩展名
            let ext = path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !self.is_supported_extension(ext) {
                debug!("跳过不支持的格式: {} ({})", filename, ext);
                result.files_skipped += 1;
                continue;
            }

            // 创建模型信息
            match ModelInfo::from_path(path.to_path_buf(), &self.models_dir) {
                Ok(info) => {
                    debug!(
                        "发现模型: {} ({}, {})",
                        info.display_name,
                        info.model_type,
                        info.size_human
                    );
                    result.total_size_bytes += info.size_bytes;
                    result.models.push(info);
                }
                Err(e) => {
                    warn!("无法解析模型 {}: {}", path.display(), e);
                    result.files_skipped += 1;
                }
            }
        }

        // 按名称排序
        result.models.sort_by(|a, b| a.display_name.cmp(&b.display_name));
        result.elapsed_ms = start.elapsed().as_millis() as u64;

        info!("{}", result.summary());

        Ok(result)
    }

    /// 扫描指定类型的模型
    pub fn scan_type(&self, model_type: ModelType) -> Result<Vec<ModelInfo>, crate::model_manager::model_info::ModelManagerError> {
        let type_dir = self.models_dir.join(model_type.to_dir_name());
        if !type_dir.exists() {
            return Ok(Vec::new());
        }

        let mut models = Vec::new();
        for entry in WalkDir::new(&type_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(f) => f,
                None => continue,
            };

            if self.should_ignore(filename) {
                continue;
            }

            let ext = path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !self.is_supported_extension(ext) {
                continue;
            }

            if let Ok(info) = ModelInfo::from_path(path.to_path_buf(), &self.models_dir) {
                models.push(info);
            }
        }

        models.sort_by(|a, b| a.display_name.cmp(&b.display_name));
        Ok(models)
    }

    /// 异步扫描（在 blocking 任务中执行）
    pub async fn scan_async(&self) -> Result<ScanResult, crate::model_manager::model_info::ModelManagerError> {
        let scanner = Self {
            models_dir: self.models_dir.clone(),
            supported_extensions: self.supported_extensions.clone(),
            ignore_patterns: self.ignore_patterns.clone(),
        };
        tokio::task::spawn_blocking(move || scanner.scan())
            .await
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(format!("Scan task failed: {}", e)))?
    }

    /// 检查文件是否应该忽略
    fn should_ignore(&self, filename: &str) -> bool {
        let lower = filename.to_lowercase();
        for pattern in &self.ignore_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }
        // 忽略隐藏文件
        if filename.starts_with('.') {
            return true;
        }
        false
    }

    /// 检查扩展名是否支持
    fn is_supported_extension(&self, ext: &str) -> bool {
        if ext.is_empty() {
            return false;
        }
        self.supported_extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
    }

    /// 计算 safetensors 文件的部分哈希（前 1MB + 中间 1MB + 末尾 1MB）
    /// 这是 ComfyUI 使用的快速哈希算法，用于模型去重和验证
    pub fn compute_partial_hash(path: &Path) -> Result<String, crate::model_manager::model_info::ModelManagerError> {
        use std::io::{Read, Seek, SeekFrom};

        let file = std::fs::File::open(path)
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(e.to_string()))?;
        let file_size = file.metadata()?.len();
        let mut hasher = blake3::Hasher::new();

        // 小文件直接全量哈希
        if file_size < 3 * 1024 * 1024 {
            let mut reader = std::io::BufReader::new(file);
            let mut buffer = [0u8; 65536];
            loop {
                let n = reader.read(&mut buffer)
                    .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
            }
        } else {
            // 大文件：头部 1MB + 中间 1MB + 尾部 1MB
            let mut reader = std::io::BufReader::new(file);
            let mut buffer = vec![0u8; 1024 * 1024];

            // 头部
            reader.read_exact(&mut buffer)
                .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
            hasher.update(&buffer);

            // 中间
            reader.seek(SeekFrom::Start(file_size / 2))
                .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
            reader.read_exact(&mut buffer)
                .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
            hasher.update(&buffer);

            // 尾部
            reader.seek(SeekFrom::End(-(1024 * 1024) as i64))
                .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
            reader.read_exact(&mut buffer)
                .map_err(|e| crate::model_manager::model_info::ModelManagerError::HashError(e.to_string()))?;
            hasher.update(&buffer);
        }

        Ok(hasher.finalize().to_hex().to_string())
    }

    /// 读取 safetensors header（前 8 字节是 header 长度，后面是 JSON）
    pub fn read_safetensors_header(path: &Path) -> Result<serde_json::Value, crate::model_manager::model_info::ModelManagerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(e.to_string()))?;

        // 读取 header 长度（8 字节小端序）
        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(format!("Read header length: {}", e)))?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        // 限制 header 大小（防止恶意文件）
        if header_len > 100 * 1024 * 1024 {
            return Err(crate::model_manager::model_info::ModelManagerError::IoError(
                format!("Safetensors header too large: {} bytes", header_len)
            ));
        }

        // 读取 header JSON
        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(format!("Read header: {}", e)))?;

        let header: serde_json::Value = serde_json::from_slice(&header_buf)
            .map_err(|e| crate::model_manager::model_info::ModelManagerError::IoError(format!("Parse header JSON: {}", e)))?;

        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manager::model_info::ModelType;

    fn create_test_models_dir() -> PathBuf {
        let unique = uuid::Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("test_model_scanner_{}", unique));
        let models_dir = temp_dir.join("models");

        // checkpoints
        let checkpoints = models_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints).unwrap();
        std::fs::write(checkpoints.join("model1.safetensors"), b"fake model 1").unwrap();
        std::fs::write(checkpoints.join("model2.ckpt"), b"fake model 2").unwrap();

        // sdxl 子目录
        let sdxl_dir = checkpoints.join("sdxl");
        std::fs::create_dir_all(&sdxl_dir).unwrap();
        std::fs::write(sdxl_dir.join("sdxl_base.safetensors"), b"fake sdxl").unwrap();

        // lora
        let lora = models_dir.join("lora");
        std::fs::create_dir_all(&lora).unwrap();
        std::fs::write(lora.join("style.safetensors"), b"fake lora").unwrap();
        std::fs::write(lora.join("detail.safetensors"), b"fake lora 2").unwrap();

        // vae
        let vae = models_dir.join("vae");
        std::fs::create_dir_all(&vae).unwrap();
        std::fs::write(vae.join("sdxl_vae.safetensors"), b"fake vae").unwrap();

        // 不支持的格式应该被跳过
        std::fs::write(checkpoints.join("readme.txt"), b"readme").unwrap();
        std::fs::write(checkpoints.join(".hidden"), b"hidden").unwrap();

        models_dir
    }

    #[test]
    fn test_scan_all() {
        let models_dir = create_test_models_dir();
        let scanner = ModelScanner::new(&models_dir);
        let result = scanner.scan().unwrap();

        // 应该扫描到 6 个模型文件（model1, model2, sdxl_base, style, detail, sdxl_vae）
        assert_eq!(result.models.len(), 6);
        assert!(result.directories_scanned > 0);
        // readme.txt 和 .hidden 应该被跳过
        assert!(result.files_skipped >= 2);
        assert!(result.total_size_bytes > 0);
    }

    #[test]
    fn test_scan_by_type() {
        let models_dir = create_test_models_dir();
        let scanner = ModelScanner::new(&models_dir);

        let checkpoints = scanner.scan_type(ModelType::Checkpoint).unwrap();
        assert_eq!(checkpoints.len(), 3); // model1, model2, sdxl_base

        let loras = scanner.scan_type(ModelType::Lora).unwrap();
        assert_eq!(loras.len(), 2); // style, detail

        let vaes = scanner.scan_type(ModelType::VAE).unwrap();
        assert_eq!(vaes.len(), 1); // sdxl_vae
    }

    #[test]
    fn test_scan_result_grouping() {
        let models_dir = create_test_models_dir();
        let scanner = ModelScanner::new(&models_dir);
        let result = scanner.scan().unwrap();

        let groups = result.group_by_type();
        assert!(groups.contains_key(&ModelType::Checkpoint));
        assert!(groups.contains_key(&ModelType::Lora));
        assert!(groups.contains_key(&ModelType::VAE));
        assert_eq!(groups[&ModelType::Checkpoint].len(), 3);
        assert_eq!(groups[&ModelType::Lora].len(), 2);
        assert_eq!(groups[&ModelType::VAE].len(), 1);
    }

    #[test]
    fn test_scan_result_summary() {
        let models_dir = create_test_models_dir();
        let scanner = ModelScanner::new(&models_dir);
        let result = scanner.scan().unwrap();

        let summary = result.summary();
        assert!(summary.contains("扫描完成"));
        assert!(summary.contains("6 个模型"));
        assert!(summary.contains("Checkpoint"));
        assert!(summary.contains("LoRA"));
        assert!(summary.contains("VAE"));
    }

    #[test]
    fn test_scan_nonexistent_dir() {
        let scanner = ModelScanner::new("/nonexistent/path/models");
        let result = scanner.scan().unwrap();
        assert_eq!(result.models.len(), 0);
        assert_eq!(result.directories_scanned, 0);
    }

    #[test]
    fn test_should_ignore() {
        let scanner = ModelScanner::new("/tmp");
        assert!(scanner.should_ignore(".hidden"));
        assert!(scanner.should_ignore("model.tmp"));
        assert!(scanner.should_ignore("model.bak"));
        assert!(!scanner.should_ignore("model.safetensors"));
    }

    #[test]
    fn test_is_supported_extension() {
        let scanner = ModelScanner::new("/tmp");
        assert!(scanner.is_supported_extension("safetensors"));
        assert!(scanner.is_supported_extension("ckpt"));
        assert!(scanner.is_supported_extension("pt"));
        assert!(scanner.is_supported_extension("onnx"));
        assert!(!scanner.is_supported_extension("txt"));
        assert!(!scanner.is_supported_extension(""));
    }

    #[test]
    fn test_compute_partial_hash_small_file() {
        let temp = std::env::temp_dir().join("test_hash_small.bin");
        std::fs::write(&temp, b"small file content").unwrap();

        let hash = ModelScanner::compute_partial_hash(&temp).unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // blake3 hex is 64 chars

        // 相同文件应该产生相同哈希
        let hash2 = ModelScanner::compute_partial_hash(&temp).unwrap();
        assert_eq!(hash, hash2);

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_compute_partial_hash_large_file() {
        let temp = std::env::temp_dir().join("test_hash_large.bin");
        // 创建大于 3MB 的文件
        let data: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 256) as u8).collect();
        std::fs::write(&temp, &data).unwrap();

        let hash = ModelScanner::compute_partial_hash(&temp).unwrap();
        assert!(!hash.is_empty());

        // 相同文件应该产生相同哈希
        let hash2 = ModelScanner::compute_partial_hash(&temp).unwrap();
        assert_eq!(hash, hash2);

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_safetensors_header() {
        // 创建一个最小的 safetensors 文件（只有空 header）
        let temp = std::env::temp_dir().join("test_safetensors.safetensors");
        let header_json = b"{}";
        let header_len = header_json.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_json);
        data.extend_from_slice(&[0u8; 100]); // 一些假数据
        std::fs::write(&temp, &data).unwrap();

        let header = ModelScanner::read_safetensors_header(&temp).unwrap();
        assert!(header.is_object());

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_scan_async() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let models_dir = create_test_models_dir();
        let scanner = ModelScanner::new(models_dir.clone());

        let result = rt.block_on(async {
            scanner.scan_async().await.unwrap()
        });

        assert_eq!(result.models.len(), 6);
    }
}
