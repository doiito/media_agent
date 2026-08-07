//! GPU 显存档位策略。
//!
//! 目标:16GB 以下显存均可用,运行期零外部运行时依赖。
//!
//! 档位决定图片模型、原生生成画布、hires-fix 参数与视频路由。
//! 探测顺序(逐级降级,探测失败不影响启动):
//! 1. `GPU_TIER` 环境变量(tier4g/tier8g/tier12g/tier16g)
//! 2. `nvidia-smi`(NVIDIA 驱动自带,非额外依赖)
//! 3. 配置文件 `sd_cpp.gpu_tier` 覆盖
//! 4. 默认 Tier8G(保守)

use serde::{Deserialize, Serialize};
use std::process::Command;

/// 显存可用档位(按可用显存而不是显卡型号划分)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuTier {
    /// 约 4-6 GiB:SD1.5 精修链路(512-640 画布 + LoRA + hires)
    Tier4G,
    /// 约 6-10 GiB:SDXL Q4_K_S GGUF @768(视频回退 SVD)
    Tier8G,
    /// 约 10-14 GiB:SDXL Q5_K_M GGUF @1024(视频 SVD 全帧/Wan 可选)
    Tier12G,
    /// 约 14 GiB 以上:SDXL f16 @1024 + hires(视频 Wan2.2)
    Tier16G,
}

impl GpuTier {
    pub const ALL: [GpuTier; 4] = [GpuTier::Tier4G, GpuTier::Tier8G, GpuTier::Tier12G, GpuTier::Tier16G];

    pub fn name(self) -> &'static str {
        match self {
            GpuTier::Tier4G => "tier4g",
            GpuTier::Tier8G => "tier8g",
            GpuTier::Tier12G => "tier12g",
            GpuTier::Tier16G => "tier16g",
        }
    }

    fn parse_name(value: &str) -> Option<GpuTier> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tier4g" | "4g" | "4" => Some(GpuTier::Tier4G),
            "tier8g" | "8g" | "8" => Some(GpuTier::Tier8G),
            "tier12g" | "12g" | "12" => Some(GpuTier::Tier12G),
            "tier16g" | "16g" | "16" => Some(GpuTier::Tier16G),
            _ => None,
        }
    }

    /// 探测当前显存档位。任何一步失败都降级,绝不阻塞启动。
    pub fn detect(override_value: Option<&str>) -> GpuTier {
        if let Some(value) = override_value {
            if let Some(tier) = Self::parse_name(value) {
                return tier;
            }
        }
        if let Ok(value) = std::env::var("GPU_TIER") {
            if let Some(tier) = Self::parse_name(&value) {
                return tier;
            }
        }
        match Self::query_nvidia_vram_mib() {
            Some(mib) => Self::from_vram_mib(mib),
            None => GpuTier::Tier8G,
        }
    }

    pub fn from_vram_mib(mib: u64) -> GpuTier {
        match mib {
            0..=6_143 => GpuTier::Tier4G,
            6_144..=10_239 => GpuTier::Tier8G,
            10_240..=14_335 => GpuTier::Tier12G,
            _ => GpuTier::Tier16G,
        }
    }

    /// 通过 NVIDIA 驱动自带的 nvidia-smi 查询显存总量(MiB)。
    fn query_nvidia_vram_mib() -> Option<u64> {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines().next()?.trim().parse().ok()
    }
}

/// 图片模型种类,决定默认 cfg 与分辨率甜点。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModelKind {
    Sd15,
    Sdxl,
}

impl ImageModelKind {
    pub fn default_cfg(self) -> f32 {
        match self {
            ImageModelKind::Sd15 => 7.0,
            ImageModelKind::Sdxl => 6.0,
        }
    }
}

/// 图片模型选择结果。
#[derive(Debug, Clone)]
pub struct ImageModelChoice {
    pub path: String,
    pub kind: ImageModelKind,
}

/// hires-fix 参数(原生 latent upscale,无外部模型依赖)。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HiresPolicy {
    pub scale: f32,
    pub steps: u32,
    pub denoising_strength: f32,
}

impl GpuTier {
    /// 图片原生画布短边(px)。
    fn image_short_edge(self, kind: ImageModelKind, quality: &str) -> u32 {
        match (kind, quality) {
            (ImageModelKind::Sd15, "fast") => 512,
            (ImageModelKind::Sd15, _) => match self {
                GpuTier::Tier4G => 576,
                _ => 640,
            },
            (ImageModelKind::Sdxl, _) => match self {
                GpuTier::Tier8G => 768,
                GpuTier::Tier12G | GpuTier::Tier16G => 1024,
                GpuTier::Tier4G => 640, // 理论不可达,SDXL 不用于 4G 档
            },
        }
    }

    /// 图片原生画布长边上限(px),防止超高纵横比爆显存。
    fn image_long_edge_cap(self) -> u32 {
        match self {
            GpuTier::Tier4G => 832,
            GpuTier::Tier8G => 1280,
            GpuTier::Tier12G => 1536,
            GpuTier::Tier16G => 1536,
        }
    }

    /// 计算原生生成画布:保持请求纵横比,短边取档位甜点,长边受上限约束。
    /// 生成尺寸与交付尺寸分离:交付缩放由上层用纯 Rust image 完成。
    pub fn generation_canvas(
        self,
        kind: ImageModelKind,
        quality: &str,
        requested_width: u32,
        requested_height: u32,
    ) -> (u32, u32) {
        let width = requested_width.max(64);
        let height = requested_height.max(64);
        let aspect = width as f64 / height as f64;
        let short = self.image_short_edge(kind, quality);
        let long_cap = self.image_long_edge_cap() as f64;
        let (gen_width, gen_height) = if width >= height {
            let long = (short as f64 * aspect).min(long_cap);
            (long, long / aspect)
        } else {
            let long = (short as f64 / aspect).min(long_cap);
            (long * aspect, long)
        };
        (
            align_dimension(gen_width.max(64.0), 8),
            align_dimension(gen_height.max(64.0), 8),
        )
    }

    /// 交付缩放:把生成画布结果缩放到请求尺寸(纯 Rust image,零依赖)。
    /// 生成尺寸大于交付尺寸时为降采样(清晰),小于时为升采样(尽力)。
    pub fn delivery_dimensions(self, requested_width: u32, requested_height: u32) -> (u32, u32) {
        (
            requested_width.max(64).max(8),
            requested_height.max(64).max(8),
        )
    }

    /// hires-fix 策略:仅 high 档启用;其余档位返回 None(零显存开销)。
    pub fn hires_policy(self, quality: &str) -> Option<HiresPolicy> {
        if quality != "high" {
            return None;
        }
        let policy = match self {
            GpuTier::Tier4G => HiresPolicy { scale: 1.3, steps: 10, denoising_strength: 0.42 },
            GpuTier::Tier8G => HiresPolicy { scale: 1.25, steps: 12, denoising_strength: 0.40 },
            GpuTier::Tier12G => HiresPolicy { scale: 1.5, steps: 14, denoising_strength: 0.36 },
            GpuTier::Tier16G => HiresPolicy { scale: 1.5, steps: 16, denoising_strength: 0.34 },
        };
        Some(policy)
    }

    /// 视频最大帧数(时序激活显存随帧数增长)。
    pub fn video_max_frames(self) -> u32 {
        match self {
            GpuTier::Tier4G => 14,
            GpuTier::Tier8G => 25,
            GpuTier::Tier12G => 25,
            GpuTier::Tier16G => 49,
        }
    }

    /// 低显存档位优先 SVD(轻量)而非 Wan(时序模型显存大)。
    pub fn prefer_svd_video(self) -> bool {
        matches!(self, GpuTier::Tier4G | GpuTier::Tier8G)
    }

    /// 显存不足档位(4G/8G)在生成时启用 sd-cpp 图分段,防止 OOM。
    pub fn segmented_vram(self) -> bool {
        matches!(self, GpuTier::Tier4G | GpuTier::Tier8G)
    }

    /// 质量档默认步数(与模型无关的档位基线,模型相关调整在调用方)。
    pub fn default_steps(self, quality: &str) -> u32 {
        match (self, quality) {
            (_, "fast") => 14,
            (GpuTier::Tier4G | GpuTier::Tier8G, _) => 26,
            (_, "high") => 36,
            _ => 30,
        }
    }
}

fn align_dimension(value: f64, alignment: u32) -> u32 {
    let alignment = alignment as f64;
    (((value / alignment).round() as u32).max(1) * alignment as u32).clamp(64, 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_from_env_override() {
        assert_eq!(GpuTier::detect(Some("tier8g")), GpuTier::Tier8G);
        assert_eq!(GpuTier::detect(Some("16")), GpuTier::Tier16G);
        assert_eq!(GpuTier::detect(Some("bogus")), GpuTier::detect(None));
    }

    #[test]
    fn vram_mapping_uses_giB_thresholds() {
        assert_eq!(GpuTier::from_vram_mib(4 * 1024), GpuTier::Tier4G);
        assert_eq!(GpuTier::from_vram_mib(8 * 1024), GpuTier::Tier8G);
        assert_eq!(GpuTier::from_vram_mib(12 * 1024), GpuTier::Tier12G);
        assert_eq!(GpuTier::from_vram_mib(16 * 1024), GpuTier::Tier16G);
        assert_eq!(GpuTier::from_vram_mib(2 * 1024), GpuTier::Tier4G);
    }

    #[test]
    fn tier16_sdxl_canvas_preserves_aspect() {
        let (w, h) = GpuTier::Tier16G.generation_canvas(ImageModelKind::Sdxl, "balanced", 1366, 780);
        assert_eq!((w, h), (1536, 880));
        let (w, h) = GpuTier::Tier16G.generation_canvas(ImageModelKind::Sdxl, "balanced", 512, 512);
        assert_eq!((w, h), (1024, 1024));
    }

    #[test]
    fn tier8_sdxl_canvas_caps_long_edge() {
        let (w, h) = GpuTier::Tier8G.generation_canvas(ImageModelKind::Sdxl, "balanced", 4096, 1024);
        assert_eq!(w, 1280);
        assert!(h >= 64 && h % 8 == 0);
    }

    #[test]
    fn tier4_sd15_canvas_stays_small() {
        let (w, h) = GpuTier::Tier4G.generation_canvas(ImageModelKind::Sd15, "high", 1024, 1024);
        assert!(w <= 832);
        assert!(h <= 832);
    }

    #[test]
    fn hires_only_for_high_quality() {
        assert!(GpuTier::Tier16G.hires_policy("high").is_some());
        assert!(GpuTier::Tier16G.hires_policy("balanced").is_none());
        assert!(GpuTier::Tier4G.hires_policy("high").is_some());
    }

    #[test]
    fn video_frames_scaled_by_tier() {
        assert_eq!(GpuTier::Tier4G.video_max_frames(), 14);
        assert_eq!(GpuTier::Tier8G.video_max_frames(), 25);
        assert_eq!(GpuTier::Tier16G.video_max_frames(), 49);
    }

    #[test]
    fn sdxl_default_cfg_lower_than_sd15() {
        assert_eq!(ImageModelKind::Sd15.default_cfg(), 7.0);
        assert_eq!(ImageModelKind::Sdxl.default_cfg(), 6.0);
    }
}
