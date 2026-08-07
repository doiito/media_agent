use comfyui_rust_agent::backend::sd_worker_protocol::{SdWorkerRequest, SdWorkerResponse};
use comfyui_rust_agent::native_runtime::{inspect_model_file, ModelContainer};
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use libloading::Library;
use std::ffi::{c_char, c_void, CStr, CString};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

const BRIDGE_ABI_VERSION: i32 = 7;

#[repr(C)]
struct MediaSdLora {
    path: *const c_char,
    multiplier: f32,
}

#[repr(C)]
struct ContextConfig {
    model_path: *const c_char,
    diffusion_model_path: *const c_char,
    high_noise_diffusion_model_path: *const c_char,
    clip_vision_path: *const c_char,
    t5xxl_path: *const c_char,
    vae_path: *const c_char,
    backend: *const c_char,
    params_backend: *const c_char,
    max_vram: *const c_char,
    weight_type: *const c_char,
    rng_type: *const c_char,
    threads: i32,
    flash_attention: bool,
    diffusion_flash_attention: bool,
    enable_mmap: bool,
    stream_layers: bool,
}

#[repr(C)]
struct GenerationRequest {
    prompt: *const c_char,
    negative_prompt: *const c_char,
    sampler: *const c_char,
    scheduler: *const c_char,
    init_image_data: *const u8,
    init_image_width: u32,
    init_image_height: u32,
    init_image_channels: u32,
    width: i32,
    height: i32,
    steps: i32,
    cfg: f32,
    flow_shift: f32,
    min_cfg: f32,
    noise_aug_strength: f32,
    strength: f32,
    seed: i64,
    frame_count: i32,
    fps: i32,
    motion_bucket_id: i32,
    loras: *const MediaSdLora,
    lora_count: u32,
    hires_enabled: i32,
    hires_scale: f32,
    hires_steps: i32,
    hires_denoising_strength: f32,
}

#[repr(C)]
struct ImageBuffer {
    data: *mut u8,
    length: usize,
    count: i32,
    width: u32,
    height: u32,
    channels: u32,
}

type AbiVersionFn = unsafe extern "C" fn() -> i32;
type StringFn = unsafe extern "C" fn() -> *const c_char;
type CreateFn = unsafe extern "C" fn(*const ContextConfig) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type SupportsFn = unsafe extern "C" fn(*const c_void) -> bool;
type GenerateFn =
    unsafe extern "C" fn(*mut c_void, *const GenerationRequest, *mut ImageBuffer) -> bool;
type FreeBufferFn = unsafe extern "C" fn(*mut ImageBuffer);

struct Bridge {
    _library: Library,
    abi_version: AbiVersionFn,
    version: StringFn,
    commit: StringFn,
    last_error: StringFn,
    create: CreateFn,
    destroy: DestroyFn,
    supports_image: SupportsFn,
    supports_video: SupportsFn,
    generate_image: GenerateFn,
    generate_video: GenerateFn,
    vae_roundtrip: GenerateFn,
    free_buffer: FreeBufferFn,
}

impl Bridge {
    unsafe fn load(path: &str) -> Result<Self, String> {
        let library = unsafe { Library::new(path) }
            .map_err(|e| format!("cannot load stable-diffusion.cpp bridge '{}': {}", path, e))?;

        macro_rules! symbol {
            ($name:literal, $type:ty) => {{
                *unsafe { library.get::<$type>(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| format!("bridge symbol {} is unavailable: {}", $name, e))?
            }};
        }

        Ok(Self {
            abi_version: symbol!("media_sd_bridge_abi_version", AbiVersionFn),
            version: symbol!("media_sd_bridge_version", StringFn),
            commit: symbol!("media_sd_bridge_commit", StringFn),
            last_error: symbol!("media_sd_bridge_last_error", StringFn),
            create: symbol!("media_sd_bridge_create", CreateFn),
            destroy: symbol!("media_sd_bridge_destroy", DestroyFn),
            supports_image: symbol!("media_sd_bridge_supports_image", SupportsFn),
            supports_video: symbol!("media_sd_bridge_supports_video", SupportsFn),
            generate_image: symbol!("media_sd_bridge_generate_image", GenerateFn),
            generate_video: symbol!("media_sd_bridge_generate_video", GenerateFn),
            vae_roundtrip: symbol!("media_sd_bridge_vae_roundtrip", GenerateFn),
            free_buffer: symbol!("media_sd_bridge_free_buffer", FreeBufferFn),
            _library: library,
        })
    }

    unsafe fn string(&self, function: StringFn) -> String {
        let value = unsafe { function() };
        if value.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }
    }

    unsafe fn error(&self) -> String {
        let message = unsafe { self.string(self.last_error) };
        if message.trim().is_empty() {
            "stable-diffusion.cpp native call failed without an error message".to_string()
        } else {
            message
        }
    }
}

struct ContextGuard<'a> {
    bridge: &'a Bridge,
    value: *mut c_void,
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        unsafe { (self.bridge.destroy)(self.value) };
    }
}

struct BufferGuard<'a> {
    bridge: &'a Bridge,
    value: ImageBuffer,
}

impl Drop for BufferGuard<'_> {
    fn drop(&mut self) {
        unsafe { (self.bridge.free_buffer)(&mut self.value) };
    }
}

fn main() {
    let request = match read_request() {
        Ok(request) => request,
        Err(error) => {
            write_response(SdWorkerResponse::failure(String::new(), error));
            return;
        }
    };
    let request_id = request.request_id.clone();
    let result = match StdoutRedirect::to_stderr() {
        Ok(redirect) => {
            let result = run(request);
            drop(redirect);
            result
        }
        Err(error) => Err(error),
    };
    let response = match result {
        Ok(response) => response,
        Err(error) => SdWorkerResponse::failure(request_id, error),
    };
    write_response(response);
}

struct StdoutRedirect {
    saved_stdout: i32,
}

impl StdoutRedirect {
    fn to_stderr() -> Result<Self, String> {
        unsafe {
            libc::fflush(std::ptr::null_mut());
            let saved_stdout = libc::dup(libc::STDOUT_FILENO);
            if saved_stdout < 0 {
                return Err(format!(
                    "cannot preserve worker stdout: {}",
                    std::io::Error::last_os_error()
                ));
            }
            if libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) < 0 {
                let error = std::io::Error::last_os_error();
                libc::close(saved_stdout);
                return Err(format!("cannot redirect native logs to stderr: {}", error));
            }
            Ok(Self { saved_stdout })
        }
    }
}

impl Drop for StdoutRedirect {
    fn drop(&mut self) {
        unsafe {
            libc::fflush(std::ptr::null_mut());
            libc::dup2(self.saved_stdout, libc::STDOUT_FILENO);
            libc::close(self.saved_stdout);
        }
    }
}

fn read_request() -> Result<SdWorkerRequest, String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("cannot read worker request: {}", e))?;
    serde_json::from_str(&input).map_err(|e| format!("invalid worker request JSON: {}", e))
}

fn write_response(response: SdWorkerResponse) {
    let serialized = serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(
            "{{\"request_id\":\"\",\"status\":\"failed\",\"error\":\"response serialization failed: {}\"}}",
            error
        )
    });
    let _ = std::io::stdout().write_all(serialized.as_bytes());
}

fn run(request: SdWorkerRequest) -> Result<SdWorkerResponse, String> {
    validate_request(&request)?;
    if !request.model_path.is_empty() {
        validate_model(&request.model_path, "full diffusion model")?;
    }
    if !request.diffusion_model_path.is_empty() {
        validate_model(&request.diffusion_model_path, "standalone diffusion model")?;
    }
    if !request.high_noise_diffusion_model_path.is_empty() {
        validate_model(
            &request.high_noise_diffusion_model_path,
            "high-noise diffusion model",
        )?;
    }
    if !request.clip_vision_path.is_empty() {
        validate_model(&request.clip_vision_path, "CLIP Vision model")?;
    }
    if !request.t5xxl_path.is_empty() {
        validate_model(&request.t5xxl_path, "T5 text encoder")?;
    }
    if !request.vae_path.is_empty() {
        validate_model(&request.vae_path, "VAE model")?;
    }

    let bridge = unsafe { Bridge::load(&request.bridge_library_path) }?;
    let abi = unsafe { (bridge.abi_version)() };
    if abi != BRIDGE_ABI_VERSION {
        return Err(format!(
            "stable-diffusion.cpp bridge ABI mismatch: worker={}, bridge={}",
            BRIDGE_ABI_VERSION, abi
        ));
    }

    let model = c_string(&request.model_path, "model_path")?;
    let diffusion_model = c_string(&request.diffusion_model_path, "diffusion_model_path")?;
    let high_noise_diffusion_model = c_string(
        &request.high_noise_diffusion_model_path,
        "high_noise_diffusion_model_path",
    )?;
    let clip_vision = c_string(&request.clip_vision_path, "clip_vision_path")?;
    let t5xxl = c_string(&request.t5xxl_path, "t5xxl_path")?;
    let vae = c_string(&request.vae_path, "vae_path")?;
    let backend = c_string(&request.backend, "backend")?;
    let params_backend = c_string(&request.params_backend, "params_backend")?;
    let max_vram = c_string(&request.max_vram, "max_vram")?;
    let weight_type = c_string(&request.weight_type, "weight_type")?;
    let rng_type = c_string(&request.rng_type, "rng_type")?;

    let context_config = ContextConfig {
        model_path: model.as_ptr(),
        diffusion_model_path: diffusion_model.as_ptr(),
        high_noise_diffusion_model_path: high_noise_diffusion_model.as_ptr(),
        clip_vision_path: clip_vision.as_ptr(),
        t5xxl_path: t5xxl.as_ptr(),
        vae_path: vae.as_ptr(),
        backend: backend.as_ptr(),
        params_backend: params_backend.as_ptr(),
        max_vram: max_vram.as_ptr(),
        weight_type: weight_type.as_ptr(),
        rng_type: rng_type.as_ptr(),
        threads: request.threads,
        flash_attention: request.flash_attention,
        diffusion_flash_attention: request.flash_attention,
        enable_mmap: true,
        stream_layers: request.stream_layers,
    };
    let context_value = unsafe { (bridge.create)(&context_config) };
    if context_value.is_null() {
        return Err(unsafe { bridge.error() });
    }
    let context = ContextGuard {
        bridge: &bridge,
        value: context_value,
    };

    let supports_operation = unsafe {
        if request.operation.is_vae_roundtrip() {
            true
        } else if request.operation.is_video() {
            (bridge.supports_video)(context.value)
        } else {
            (bridge.supports_image)(context.value)
        }
    };
    if !supports_operation {
        return Err(format!(
            "loaded model '{}' does not support {:?}",
            request.model_path, request.operation
        ));
    }

    let input_image = load_input_image(&request)?;
    let (input_data, input_width, input_height) = match &input_image {
        Some(image) => (image.as_raw().as_ptr(), image.width(), image.height()),
        None => (std::ptr::null(), 0, 0),
    };

    let prompt = c_string(&request.prompt, "prompt")?;
    let negative_prompt = c_string(&request.negative_prompt, "negative_prompt")?;
    let sampler = c_string(&request.sampler, "sampler")?;
    let scheduler = c_string(&request.scheduler, "scheduler")?;

    let lora_paths: Vec<CString> = request
        .loras
        .iter()
        .map(|lora| c_string(&lora.path, "lora path"))
        .collect::<Result<_, _>>()?;
    let loras: Vec<MediaSdLora> = lora_paths
        .iter()
        .zip(request.loras.iter())
        .map(|(path, lora)| MediaSdLora {
            path: path.as_ptr(),
            multiplier: lora.multiplier,
        })
        .collect();

    let generation_request = GenerationRequest {
        prompt: prompt.as_ptr(),
        negative_prompt: negative_prompt.as_ptr(),
        sampler: sampler.as_ptr(),
        scheduler: scheduler.as_ptr(),
        init_image_data: input_data,
        init_image_width: input_width,
        init_image_height: input_height,
        init_image_channels: if input_image.is_some() { 3 } else { 0 },
        width: request.width as i32,
        height: request.height as i32,
        steps: request.steps,
        cfg: request.cfg,
        flow_shift: request.flow_shift,
        min_cfg: request.min_cfg,
        noise_aug_strength: request.noise_aug_strength,
        strength: request.strength,
        seed: request.seed,
        frame_count: request.frames,
        fps: request.fps,
        motion_bucket_id: request.motion_bucket_id,
        loras: loras.as_ptr(),
        lora_count: loras.len() as u32,
        hires_enabled: if request.hires.is_some() { 1 } else { 0 },
        hires_scale: request.hires.map(|hires| hires.scale).unwrap_or(0.0),
        hires_steps: request.hires.map(|hires| hires.steps).unwrap_or(0),
        hires_denoising_strength: request
            .hires
            .map(|hires| hires.denoising_strength)
            .unwrap_or(0.0),
    };
    let mut output = ImageBuffer {
        data: std::ptr::null_mut(),
        length: 0,
        count: 0,
        width: 0,
        height: 0,
        channels: 0,
    };
    let generated = unsafe {
        if request.operation.is_vae_roundtrip() {
            (bridge.vae_roundtrip)(context.value, &generation_request, &mut output)
        } else if request.operation.is_video() {
            (bridge.generate_video)(context.value, &generation_request, &mut output)
        } else {
            (bridge.generate_image)(context.value, &generation_request, &mut output)
        }
    };
    if !generated {
        return Err(unsafe { bridge.error() });
    }
    let output = BufferGuard {
        bridge: &bridge,
        value: output,
    };
    validate_output_buffer(&output.value)?;

    if request.operation.is_video() || (request.operation.is_vae_roundtrip() && request.frames > 1)
    {
        encode_video(
            &output.value,
            &request.output_path,
            request.fps,
            request.output_width,
            request.output_height,
        )?;
    } else {
        encode_image(&output.value, &request.output_path)?;
    }

    let version = unsafe { bridge.string(bridge.version) };
    let commit = unsafe { bridge.string(bridge.commit) };
    Ok(SdWorkerResponse::success(
        request.request_id,
        request.output_path,
        version,
        commit,
    ))
}

fn validate_request(request: &SdWorkerRequest) -> Result<(), String> {
    if request.model_path.is_empty() && request.diffusion_model_path.is_empty() {
        return Err("model_path or diffusion_model_path is required".to_string());
    }
    if request.width == 0 || request.height == 0 {
        return Err("generation dimensions must be greater than zero".to_string());
    }
    if !(1..=100).contains(&request.steps) {
        return Err("sampling steps must be between 1 and 100".to_string());
    }
    if !request.cfg.is_finite() || !(0.1..=30.0).contains(&request.cfg) {
        return Err("cfg must be a finite value between 0.1 and 30".to_string());
    }
    if request.operation.is_video() {
        if !(1..=161).contains(&request.frames) {
            return Err("video frames must be between 1 and 161".to_string());
        }
        if !(1..=60).contains(&request.fps) {
            return Err("video fps must be between 1 and 60".to_string());
        }
        if !request.min_cfg.is_finite() || !(0.1..=30.0).contains(&request.min_cfg) {
            return Err("video min_cfg must be a finite value between 0.1 and 30".to_string());
        }
        if request.min_cfg > request.cfg {
            return Err("video min_cfg must not exceed final-frame cfg".to_string());
        }
        if !request.noise_aug_strength.is_finite()
            || !(0.0..=1.0).contains(&request.noise_aug_strength)
        {
            return Err("video noise_aug_strength must be between 0 and 1".to_string());
        }
        if !(0..=1023).contains(&request.motion_bucket_id) {
            return Err("motion_bucket_id must be between 0 and 1023".to_string());
        }
    }
    Ok(())
}

fn c_string(value: &str, name: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("{} contains an embedded NUL byte", name))
}

fn validate_model(path: &str, role: &str) -> Result<(), String> {
    let info = inspect_model_file(path).map_err(|e| format!("{}: {}", role, e))?;
    if matches!(
        info.container,
        ModelContainer::Safetensors | ModelContainer::Gguf
    ) {
        Ok(())
    } else {
        Err(format!(
            "{} '{}' uses unsupported {:?} container",
            role, path, info.container
        ))
    }
}

fn load_input_image(request: &SdWorkerRequest) -> Result<Option<image::RgbImage>, String> {
    let Some(path) = &request.input_path else {
        return Ok(None);
    };
    let image =
        image::open(path).map_err(|e| format!("cannot decode input image '{}': {}", path, e))?;
    let prepared = if request.operation.is_video() {
        fit_video_conditioning_image(image, request.width, request.height)
    } else {
        image
            .resize_to_fill(request.width, request.height, FilterType::Lanczos3)
            .to_rgb8()
    };
    Ok(Some(prepared))
}

fn fit_video_conditioning_image(image: DynamicImage, width: u32, height: u32) -> RgbImage {
    let mut canvas = image
        .resize_exact(width, height, FilterType::Triangle)
        .blur(24.0)
        .to_rgb8();
    let (fitted_width, fitted_height) =
        fit_within_dimensions(image.width(), image.height(), width, height);
    let foreground = image
        .resize_exact(fitted_width, fitted_height, FilterType::Lanczos3)
        .to_rgb8();
    let x = (width.saturating_sub(foreground.width()) / 2) as i64;
    let y = (height.saturating_sub(foreground.height()) / 2) as i64;
    image::imageops::overlay(&mut canvas, &foreground, x, y);
    canvas
}

fn fit_within_dimensions(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> (u32, u32) {
    let width_scale = target_width as f64 / source_width.max(1) as f64;
    let height_scale = target_height as f64 / source_height.max(1) as f64;
    let scale = width_scale.min(height_scale);
    (
        (source_width as f64 * scale).round().max(1.0) as u32,
        (source_height as f64 * scale).round().max(1.0) as u32,
    )
}

fn validate_output_buffer(output: &ImageBuffer) -> Result<(), String> {
    if output.data.is_null() || output.count <= 0 || output.width == 0 || output.height == 0 {
        return Err("native bridge returned an empty output buffer".to_string());
    }
    let frame_size = output.width as usize * output.height as usize * output.channels as usize;
    let expected = frame_size
        .checked_mul(output.count as usize)
        .ok_or_else(|| "native output size overflow".to_string())?;
    if expected != output.length {
        return Err(format!(
            "native output length mismatch: expected {}, received {}",
            expected, output.length
        ));
    }
    Ok(())
}

fn encode_image(output: &ImageBuffer, output_path: &str) -> Result<(), String> {
    ensure_parent(output_path)?;
    let frame_size = output.width as usize * output.height as usize * output.channels as usize;
    let data = unsafe { std::slice::from_raw_parts(output.data, frame_size) };
    let color = color_type(output.channels)?;
    image::save_buffer(output_path, data, output.width, output.height, color)
        .map_err(|e| format!("cannot encode output image '{}': {}", output_path, e))
}

fn encode_video(
    output: &ImageBuffer,
    output_path: &str,
    fps: i32,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    if fps <= 0 {
        return Err("video fps must be greater than zero".to_string());
    }
    ensure_parent(output_path)?;
    let frame_dir = std::env::temp_dir().join(format!("media-sd-frames-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&frame_dir)
        .map_err(|e| format!("cannot create temporary frame directory: {}", e))?;

    let result = (|| {
        let frame_size = output.width as usize * output.height as usize * output.channels as usize;
        let all_data = unsafe { std::slice::from_raw_parts(output.data, output.length) };
        let color = color_type(output.channels)?;
        for index in 0..output.count as usize {
            let frame = &all_data[index * frame_size..(index + 1) * frame_size];
            let path = frame_dir.join(format!("frame_{:06}.png", index));
            image::save_buffer(&path, frame, output.width, output.height, color)
                .map_err(|e| format!("cannot encode generated frame {}: {}", index, e))?;
        }

        let input_pattern = frame_dir.join("frame_%06d.png");
        let mut command = Command::new("ffmpeg");
        command
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-framerate")
            .arg(fps.to_string())
            .arg("-i")
            .arg(&input_pattern);

        let target_width = even_dimension(output_width, output.width);
        let target_height = even_dimension(output_height, output.height);
        if target_width != output.width || target_height != output.height {
            command.arg("-vf").arg(format!(
                "scale={}:{}:force_original_aspect_ratio=decrease:flags=lanczos,pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black",
                target_width, target_height, target_width, target_height
            ));
        }

        let process = command
            .arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg("slow")
            .arg("-crf")
            .arg("17")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-movflags")
            .arg("+faststart")
            .arg(output_path)
            .output()
            .map_err(|e| format!("cannot start ffmpeg: {}", e))?;
        if !process.status.success() {
            return Err(format!(
                "ffmpeg failed: {}",
                String::from_utf8_lossy(&process.stderr).trim()
            ));
        }
        Ok(())
    })();

    let _ = std::fs::remove_dir_all(&frame_dir);
    result
}

fn even_dimension(requested: u32, fallback: u32) -> u32 {
    let value = if requested == 0 { fallback } else { requested };
    value.max(2) & !1
}

fn ensure_parent(path: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "cannot create output directory '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    Ok(())
}

fn color_type(channels: u32) -> Result<image::ColorType, String> {
    match channels {
        1 => Ok(image::ColorType::L8),
        3 => Ok(image::ColorType::Rgb8),
        4 => Ok(image::ColorType::Rgba8),
        _ => Err(format!(
            "unsupported native output channel count: {}",
            channels
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_conditioning_preserves_portrait_content() {
        let source = DynamicImage::ImageRgb8(RgbImage::new(400, 500));
        let prepared = fit_video_conditioning_image(source, 1024, 576);
        assert_eq!(prepared.dimensions(), (1024, 576));
        assert_eq!(fit_within_dimensions(400, 500, 1024, 576), (461, 576));
    }
}
