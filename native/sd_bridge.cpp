#include "sd_bridge.h"

#include "stable-diffusion.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <vector>

struct media_sd_context {
    sd_ctx_t* value;
};

namespace {

thread_local std::string last_error;

const char* nullable(const char* value) {
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

void log_callback(enum sd_log_level_t level, const char* text, void*) {
    if (level >= SD_LOG_WARN && text != nullptr) {
        last_error.append(text);
    }
}

void reset_error() {
    last_error.clear();
}

void set_error(const char* message) {
    if (last_error.empty()) {
        last_error = message;
    }
}

bool requests_cuda(const char* backend) {
    return backend != nullptr && std::strstr(backend, "cuda") != nullptr;
}

bool initialize_cuda_driver(const char* backend) {
    if (!requests_cuda(backend)) {
        return true;
    }

    // WSL's libcuda shim can report cudaErrorInsufficientDriver when the CUDA
    // runtime initializes before the real host driver has been opened.
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (driver == nullptr) {
        last_error = "cannot load CUDA driver: ";
        last_error.append(dlerror());
        return false;
    }

    using cu_init_fn = int (*)(unsigned int);
    auto* cu_init = reinterpret_cast<cu_init_fn>(dlsym(driver, "cuInit"));
    if (cu_init == nullptr) {
        last_error = "CUDA driver does not export cuInit";
        dlclose(driver);
        return false;
    }

    const int result = cu_init(0);
    if (result != 0) {
        last_error = "CUDA driver initialization failed with code ";
        last_error.append(std::to_string(result));
        dlclose(driver);
        return false;
    }

    dlclose(driver);
    return true;
}

void apply_sample_params(
    sd_sample_params_t* params,
    const media_sd_generation_request* request) {
    params->sample_steps = request->steps;
    params->guidance.txt_cfg = request->cfg;
    params->flow_shift = request->flow_shift;

    if (const char* sampler = nullable(request->sampler)) {
        enum sample_method_t value = str_to_sample_method(sampler);
        if (value != SAMPLE_METHOD_COUNT) {
            params->sample_method = value;
        }
    }
    if (const char* scheduler = nullable(request->scheduler)) {
        enum scheduler_t value = str_to_scheduler(scheduler);
        if (value != SCHEDULER_COUNT) {
            params->scheduler = value;
        }
    }
}

sd_image_t make_image(const media_sd_generation_request* request) {
    sd_image_t image{};
    image.width = request->init_image_width;
    image.height = request->init_image_height;
    image.channel = request->init_image_channels;
    image.data = const_cast<uint8_t*>(request->init_image_data);
    return image;
}

std::vector<sd_lora_t> make_loras(const media_sd_generation_request* request) {
    std::vector<sd_lora_t> loras;
    if (request->loras == nullptr || request->lora_count == 0) {
        return loras;
    }
    loras.reserve(request->lora_count);
    for (uint32_t index = 0; index < request->lora_count; ++index) {
        const char* path = nullable(request->loras[index].path);
        if (path == nullptr) {
            continue;
        }
        sd_lora_t lora{};
        lora.path = path;
        lora.multiplier = request->loras[index].multiplier;
        loras.push_back(lora);
    }
    return loras;
}

void apply_hires(sd_hires_params_t* hires, const media_sd_generation_request* request) {
    if (request->hires_enabled == 0) {
        return;
    }
    hires->enabled = true;
    hires->upscaler = SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED;
    hires->scale = request->hires_scale;
    hires->steps = request->hires_steps;
    hires->denoising_strength = request->hires_denoising_strength;
}

bool copy_images(sd_image_t* images, int count, media_sd_image_buffer* output) {
    if (images == nullptr || count <= 0 || output == nullptr) {
        set_error("native inference returned no frames");
        return false;
    }

    const uint32_t width = images[0].width;
    const uint32_t height = images[0].height;
    const uint32_t channels = images[0].channel;
    if (width == 0 || height == 0 || channels == 0) {
        set_error("native inference returned an invalid frame shape");
        return false;
    }

    const size_t frame_size = static_cast<size_t>(width) * height * channels;
    if (frame_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(count)) {
        set_error("native inference frame buffer overflow");
        return false;
    }
    const size_t total_size = frame_size * static_cast<size_t>(count);
    auto* data = static_cast<uint8_t*>(std::malloc(total_size));
    if (data == nullptr) {
        set_error("failed to allocate native inference output buffer");
        return false;
    }

    for (int index = 0; index < count; ++index) {
        if (images[index].width != width || images[index].height != height
            || images[index].channel != channels || images[index].data == nullptr) {
            std::free(data);
            set_error("native inference returned inconsistent frame shapes");
            return false;
        }
        std::memcpy(data + frame_size * static_cast<size_t>(index), images[index].data, frame_size);
    }

    output->data = data;
    output->length = total_size;
    output->count = count;
    output->width = width;
    output->height = height;
    output->channels = channels;
    return true;
}

}  // namespace

extern "C" {

int32_t media_sd_bridge_abi_version(void) {
    return MEDIA_SD_BRIDGE_ABI_VERSION;
}

const char* media_sd_bridge_version(void) {
    return sd_version();
}

const char* media_sd_bridge_commit(void) {
    return sd_commit();
}

const char* media_sd_bridge_last_error(void) {
    return last_error.c_str();
}

media_sd_context* media_sd_bridge_create(const media_sd_context_config* config) {
    reset_error();
    if (config == nullptr
        || (nullable(config->model_path) == nullptr
            && nullable(config->diffusion_model_path) == nullptr)) {
        set_error("model_path or diffusion_model_path is required");
        return nullptr;
    }

    sd_set_log_callback(log_callback, nullptr);

    if (!initialize_cuda_driver(config->backend)) {
        return nullptr;
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = nullable(config->model_path);
    params.diffusion_model_path = nullable(config->diffusion_model_path);
    params.high_noise_diffusion_model_path = nullable(config->high_noise_diffusion_model_path);
    params.clip_vision_path = nullable(config->clip_vision_path);
    params.t5xxl_path = nullable(config->t5xxl_path);
    params.vae_path = nullable(config->vae_path);
    params.backend = nullable(config->backend);
    params.params_backend = nullable(config->params_backend);
    params.max_vram = nullable(config->max_vram);
    params.n_threads = config->threads;
    params.flash_attn = config->flash_attention;
    params.diffusion_flash_attn = config->diffusion_flash_attention;
    params.enable_mmap = config->enable_mmap;
    params.stream_layers = config->stream_layers;

    if (const char* weight_type = nullable(config->weight_type)) {
        params.wtype = str_to_sd_type(weight_type);
    }
    if (const char* rng_type = nullable(config->rng_type)) {
        params.rng_type = str_to_rng_type(rng_type);
        params.sampler_rng_type = params.rng_type;
    }

    sd_ctx_t* value = new_sd_ctx(&params);
    if (value == nullptr) {
        set_error("stable-diffusion.cpp failed to create a model context");
        return nullptr;
    }

    auto* context = new media_sd_context{value};
    return context;
}

void media_sd_bridge_destroy(media_sd_context* context) {
    if (context == nullptr) {
        return;
    }
    free_sd_ctx(context->value);
    delete context;
}

bool media_sd_bridge_supports_image(const media_sd_context* context) {
    return context != nullptr && sd_ctx_supports_image_generation(context->value);
}

bool media_sd_bridge_supports_video(const media_sd_context* context) {
    return context != nullptr && sd_ctx_supports_video_generation(context->value);
}

void media_sd_bridge_cancel(media_sd_context* context) {
    if (context != nullptr) {
        sd_cancel_generation(context->value, SD_CANCEL_ALL);
    }
}

bool media_sd_bridge_generate_image(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output) {
    reset_error();
    if (context == nullptr || request == nullptr || output == nullptr) {
        set_error("invalid image generation arguments");
        return false;
    }
    *output = {};
    if (!sd_ctx_supports_image_generation(context->value)) {
        set_error("loaded model does not support image generation");
        return false;
    }

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = nullable(request->prompt);
    params.negative_prompt = nullable(request->negative_prompt);
    params.width = request->width;
    params.height = request->height;
    params.strength = request->strength;
    params.seed = request->seed;
    params.batch_count = 1;
    if (request->init_image_data != nullptr) {
        params.init_image = make_image(request);
    }
    apply_sample_params(&params.sample_params, request);

    std::vector<sd_lora_t> loras = make_loras(request);
    if (!loras.empty()) {
        params.loras = loras.data();
        params.lora_count = static_cast<uint32_t>(loras.size());
    }
    apply_hires(&params.hires, request);

    sd_image_t* images = nullptr;
    int count = 0;
    if (!generate_image(context->value, &params, &images, &count)) {
        set_error("stable-diffusion.cpp image generation failed");
        return false;
    }
    const bool copied = copy_images(images, count, output);
    free_sd_images(images, count);
    return copied;
}

bool media_sd_bridge_generate_video(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output) {
    reset_error();
    if (context == nullptr || request == nullptr || output == nullptr) {
        set_error("invalid video generation arguments");
        return false;
    }
    *output = {};
    if (!sd_ctx_supports_video_generation(context->value)) {
        set_error("loaded model does not support video generation");
        return false;
    }

    sd_vid_gen_params_t params;
    sd_vid_gen_params_init(&params);
    params.prompt = nullable(request->prompt);
    params.negative_prompt = nullable(request->negative_prompt);
    params.width = request->width;
    params.height = request->height;
    params.strength = request->strength;
    params.seed = request->seed;
    params.video_frames = request->frame_count;
    params.fps = request->fps;
    params.motion_bucket_id = request->motion_bucket_id;
    params.min_guidance_scale = request->min_cfg;
    params.noise_aug_strength = request->noise_aug_strength;
    if (request->frame_count > 2) {
        // The SVD temporal decoder otherwise builds one very large CUDA graph for
        // every frame. Spatial tiling preserves temporal context while bounding
        // the per-tile VAE compute buffer on 16 GB GPUs.
        params.vae_tiling_params.enabled = true;
        params.vae_tiling_params.tile_size_x = 32;
        params.vae_tiling_params.tile_size_y = 32;
        params.vae_tiling_params.target_overlap = 0.5f;
    }
    if (request->init_image_data != nullptr) {
        params.init_image = make_image(request);
    }
    apply_sample_params(&params.sample_params, request);

    std::vector<sd_lora_t> loras = make_loras(request);
    if (!loras.empty()) {
        params.loras = loras.data();
        params.lora_count = static_cast<uint32_t>(loras.size());
    }

    sd_image_t* frames = nullptr;
    sd_audio_t* audio = nullptr;
    int count = 0;
    if (!generate_video(context->value, &params, &frames, &count, &audio)) {
        set_error("stable-diffusion.cpp video generation failed");
        return false;
    }
    const bool copied = copy_images(frames, count, output);
    free_sd_images(frames, count);
    if (audio != nullptr) {
        free_sd_audio(audio);
    }
    return copied;
}

bool media_sd_bridge_vae_roundtrip(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output) {
    reset_error();
    if (context == nullptr || request == nullptr || output == nullptr
        || request->init_image_data == nullptr) {
        set_error("VAE roundtrip requires a context, input image, and output buffer");
        return false;
    }
    *output = {};

    sd_image_t* images = nullptr;
    int count = 0;
    if (!vae_roundtrip(
            context->value,
            make_image(request),
            request->width,
            request->height,
            request->frame_count,
            &images,
            &count)) {
        set_error("stable-diffusion.cpp VAE roundtrip failed");
        return false;
    }
    const bool copied = copy_images(images, count, output);
    free_sd_images(images, count);
    return copied;
}

void media_sd_bridge_free_buffer(media_sd_image_buffer* buffer) {
    if (buffer == nullptr) {
        return;
    }
    std::free(buffer->data);
    *buffer = {};
}

}  // extern "C"
