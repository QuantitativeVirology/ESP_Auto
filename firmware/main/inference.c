#include "inference.h"
#include "ternary_ops.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char *TAG = "inference";

// Generated model data (populated by export_packed.py)
// #include "model_data.h"

// Ping-pong activation buffers in SRAM (16-byte aligned)
// Largest activation: 48*48*16 = 36864 bytes (after first DW-separable block)
#define MAX_ACTIVATION_SIZE (48 * 48 * 16)
static int8_t __attribute__((aligned(16))) buf_a[MAX_ACTIVATION_SIZE];
static int8_t __attribute__((aligned(16))) buf_b[MAX_ACTIVATION_SIZE];

int classify_image(const int8_t *input_96x96x3)
{
    // TODO: implement layer-by-layer dispatch once model_data.h is generated
    // Placeholder: copy input to buf_a, run through model_layers[], swap buffers
    (void)input_96x96x3;
    ESP_LOGW(TAG, "Inference not yet implemented — returning CAT");
    return CLASS_CAT;
}

void inference_print_memory_map(void)
{
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t min_internal = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    size_t free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG, "MEMORY sram_free=%u sram_min_free=%u psram_free=%u "
             "buf_a=%u buf_b=%u",
             (unsigned)free_internal, (unsigned)min_internal,
             (unsigned)free_spiram,
             (unsigned)sizeof(buf_a), (unsigned)sizeof(buf_b));
}
