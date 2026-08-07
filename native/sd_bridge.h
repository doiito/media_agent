#ifndef MEDIA_AGENT_SD_BRIDGE_H
#define MEDIA_AGENT_SD_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEDIA_SD_BRIDGE_ABI_VERSION 7

typedef struct media_sd_context media_sd_context;

typedef struct {
    const char* model_path;
    const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* vae_path;
    const char* backend;
    const char* params_backend;
    const char* max_vram;
    const char* weight_type;
    const char* rng_type;
    int32_t threads;
    bool flash_attention;
    bool diffusion_flash_attention;
    bool enable_mmap;
    bool stream_layers;
} media_sd_context_config;

typedef struct {
    const char* path;
    float multiplier;
} media_sd_lora_t;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    const char* sampler;
    const char* scheduler;
    const uint8_t* init_image_data;
    uint32_t init_image_width;
    uint32_t init_image_height;
    uint32_t init_image_channels;
    int32_t width;
    int32_t height;
    int32_t steps;
    float cfg;
    float flow_shift;
    float min_cfg;
    float noise_aug_strength;
    float strength;
    int64_t seed;
    int32_t frame_count;
    int32_t fps;
    int32_t motion_bucket_id;
    const media_sd_lora_t* loras;
    uint32_t lora_count;
    int32_t hires_enabled;
    float hires_scale;
    int32_t hires_steps;
    float hires_denoising_strength;
} media_sd_generation_request;

typedef struct {
    uint8_t* data;
    size_t length;
    int32_t count;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
} media_sd_image_buffer;

int32_t media_sd_bridge_abi_version(void);
const char* media_sd_bridge_version(void);
const char* media_sd_bridge_commit(void);
const char* media_sd_bridge_last_error(void);

media_sd_context* media_sd_bridge_create(const media_sd_context_config* config);
void media_sd_bridge_destroy(media_sd_context* context);
bool media_sd_bridge_supports_image(const media_sd_context* context);
bool media_sd_bridge_supports_video(const media_sd_context* context);
void media_sd_bridge_cancel(media_sd_context* context);

bool media_sd_bridge_generate_image(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output);

bool media_sd_bridge_generate_video(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output);

bool media_sd_bridge_vae_roundtrip(
    media_sd_context* context,
    const media_sd_generation_request* request,
    media_sd_image_buffer* output);

void media_sd_bridge_free_buffer(media_sd_image_buffer* buffer);

#ifdef __cplusplus
}
#endif

#endif
