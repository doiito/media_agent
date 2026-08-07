//! 启发式质量评估(纯 Rust,零外部依赖)。
//!
//! 评分由清晰度(边缘能量)、对比度(动态范围)、细节(亮度方差)组成,
//! 用于候选选优与失败模式重试。本地 VLM 评分是可选的网络增强,
//! 不参与默认运行路径,不占用本地显存。

use image::GenericImageView;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureMode {
    /// 近空白或近均匀画面
    Blank,
    /// 边缘能量低(模糊)
    Blurry,
    /// 动态范围窄(发灰/过曝)
    LowContrast,
    /// 亮度方差低(细节平)
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityScore {
    /// 综合分 0..1
    pub overall: f32,
    /// 边缘能量归一化 0..1
    pub sharpness: f32,
    /// 动态范围归一化 0..1
    pub contrast: f32,
    /// 亮度方差归一化 0..1
    pub detail: f32,
}

impl QualityScore {
    pub fn failure_modes(self) -> Vec<FailureMode> {
        let mut modes = Vec::new();
        if self.overall <= 0.02 && self.contrast < 0.04 {
            modes.push(FailureMode::Blank);
        }
        if self.sharpness < 0.18 {
            modes.push(FailureMode::Blurry);
        }
        if self.contrast < 0.30 {
            modes.push(FailureMode::LowContrast);
        }
        if self.detail < 0.16 {
            modes.push(FailureMode::Flat);
        }
        modes
    }

    /// 是否达到可接受线(候选选优的及格线)。
    pub fn acceptable(self) -> bool {
        self.overall >= 0.45 && self.sharpness >= 0.18
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "overall": round_metric(self.overall),
            "sharpness": round_metric(self.sharpness),
            "contrast": round_metric(self.contrast),
            "detail": round_metric(self.detail),
            "acceptable": self.acceptable(),
            "failure_modes": self.failure_modes().iter()
                .map(|mode| format!("{:?}", mode))
                .collect::<Vec<_>>(),
        })
    }
}

/// 对内存中的图片字节做质量评分(解码失败返回 Err,由调用方降级)。
pub fn score_image_bytes(data: &[u8]) -> Result<QualityScore, String> {
    let image = image::load_from_memory(data)
        .map_err(|error| format!("cannot decode image for quality scoring: {}", error))?;
    let luma = image.to_luma8();
    let (width, height) = luma.dimensions();
    let pixels = luma.as_raw();
    let count = pixels.len();
    if count == 0 || width == 0 || height == 0 {
        return Err("quality scoring received an empty image".to_string());
    }

    let mut sum = 0_u64;
    let mut sum_squares = 0_u64;
    let mut histogram = [0_u64; 256];
    for &value in pixels {
        histogram[value as usize] += 1;
        sum += u64::from(value);
        sum_squares += u64::from(value) * u64::from(value);
    }
    let mean = sum as f64 / count as f64;
    let variance = (sum_squares as f64 / count as f64 - mean * mean).max(0.0);
    let stddev = variance.sqrt();

    let mut edge_sum = 0_u64;
    let mut edge_count = 0_u64;
    for y in 0..height {
        let row = y as usize * width as usize;
        for x in 0..width - 1 {
            let left = pixels[row + x as usize];
            let right = pixels[row + x as usize + 1];
            edge_sum += u64::from(left.abs_diff(right));
            edge_count += 1;
        }
    }
    for y in 0..height - 1 {
        let top = y as usize * width as usize;
        let bottom = (y as usize + 1) * width as usize;
        for x in 0..width {
            let above = pixels[top + x as usize];
            let below = pixels[bottom + x as usize];
            edge_sum += u64::from(above.abs_diff(below));
            edge_count += 1;
        }
    }
    let edge_energy = if edge_count > 0 {
        edge_sum as f64 / edge_count as f64
    } else {
        0.0
    };

    let (p01, p99) = histogram_percentiles(&histogram, count);
    let dynamic_range = p99.saturating_sub(p01);

    Ok(QualityScore {
        overall: compose_score(stddev, dynamic_range, edge_energy),
        sharpness: (edge_energy / 22.0).min(1.0) as f32,
        contrast: dynamic_range as f32 / 255.0,
        detail: (stddev / 70.0).min(1.0) as f32,
    })
}

fn compose_score(stddev: f64, dynamic_range: u8, edge_energy: f64) -> f32 {
    let contrast = dynamic_range as f32 / 255.0;
    let detail = (stddev / 70.0).min(1.0) as f32;
    let sharpness = (edge_energy / 22.0).min(1.0) as f32;
    (0.45 * sharpness + 0.35 * contrast + 0.20 * detail).clamp(0.0, 1.0)
}

fn histogram_percentiles(histogram: &[u64; 256], count: usize) -> (u8, u8) {
    let mut p01 = 0_u8;
    let mut p99 = 255_u8;
    let mut cumulative = 0_u64;
    let target_01 = ((count.saturating_sub(1)) as f64 * 0.01).round() as u64;
    for (value, &bucket) in histogram.iter().enumerate() {
        cumulative += bucket;
        if cumulative > target_01 {
            p01 = value as u8;
            break;
        }
    }
    cumulative = 0;
    let target_99 = ((count.saturating_sub(1)) as f64 * 0.99).round() as u64;
    for (value, &bucket) in histogram.iter().enumerate() {
        cumulative += bucket;
        if cumulative > target_99 {
            p99 = value as u8;
            break;
        }
    }
    (p01, p99)
}

/// 失败模式 → 参数调整。返回是否发生了调整。
/// 规则只调整步数与 CFG,不改变模型与提示词,保证重试可预期。
pub fn adjust_for_retry(steps: &mut usize, cfg: &mut f32, modes: &[FailureMode]) -> bool {
    let mut changed = false;
    for mode in modes {
        match mode {
            FailureMode::Blurry => {
                *steps = ((*steps as f32 * 1.25).round() as usize).min(60);
                *cfg = (*cfg + 0.4).min(12.0);
                changed = true;
            }
            FailureMode::LowContrast => {
                *cfg = (*cfg + 0.3).min(12.0);
                changed = true;
            }
            FailureMode::Flat => {
                *steps = ((*steps as f32 * 1.15).round() as usize).min(60);
                changed = true;
            }
            FailureMode::Blank => {}
        }
    }
    changed
}

fn round_metric(value: f32) -> f32 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn encode(image: &RgbImage) -> Vec<u8> {
        let mut buffer = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image.clone())
            .write_to(&mut buffer, image::ImageFormat::Png)
            .unwrap();
        buffer.into_inner()
    }

    #[test]
    fn rejects_blank_image() {
        let image = RgbImage::from_pixel(64, 64, image::Rgb([255, 255, 255]));
        let score = score_image_bytes(&encode(&image)).unwrap();
        assert!(score.overall < 0.1);
        assert!(score.failure_modes().contains(&FailureMode::Blank));
    }

    #[test]
    fn accepts_high_detail_image() {
        let image = RgbImage::from_fn(256, 256, |x, y| {
            let v = ((x * 31 + y * 17) % 256) as u8;
            image::Rgb([v, v, v])
        });
        let score = score_image_bytes(&encode(&image)).unwrap();
        assert!(score.overall > 0.5);
        assert!(score.acceptable());
        assert!(!score.failure_modes().contains(&FailureMode::Blank));
    }

    #[test]
    fn blurry_detection_triggers_retry_adjustment() {
        let mut steps = 20;
        let mut cfg = 6.0;
        assert!(adjust_for_retry(&mut steps, &mut cfg, &[FailureMode::Blurry]));
        assert_eq!(steps, 25);
        assert!((cfg - 6.4).abs() < 0.001);
    }

    #[test]
    fn blank_is_not_param_adjustable() {
        let mut steps = 20;
        let mut cfg = 6.0;
        assert!(!adjust_for_retry(&mut steps, &mut cfg, &[FailureMode::Blank]));
        assert_eq!(steps, 20);
    }

    #[test]
    fn adjustment_caps_steps_and_cfg() {
        let mut steps = 58;
        let mut cfg = 11.8;
        adjust_for_retry(&mut steps, &mut cfg, &[FailureMode::Blurry]);
        assert_eq!(steps, 60);
        assert!((cfg - 12.0).abs() < 0.001);
    }

    #[test]
    fn checkerboard_image_scores_sharp() {
        let image = RgbImage::from_fn(256, 256, |x, y| {
            let v = if ((x / 2) + (y / 2)) % 2 == 0 { 255 } else { 0 };
            image::Rgb([v, v, v])
        });
        let score = score_image_bytes(&encode(&image)).unwrap();
        assert!(score.sharpness > 0.8);
        assert!(score.contrast > 0.9);
    }
}
