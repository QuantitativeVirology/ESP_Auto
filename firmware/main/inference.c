#include "inference.h"
#include "ternary_ops.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char *TAG = "inference";

#include "model_data.h"

/*
 * Ping-pong activation buffers in SRAM (16-byte aligned).
 * Max activation: 48*48*16 = 36864 bytes (features.0.pw output).
 * Input (96*96*3 = 27648) also fits.
 *
 * INT32 accumulator reuses buf_b's memory (cast to int32_t*) since
 * we write INT32 accumulators then requantize back to INT8 into buf_b
 * before the next layer reads it.
 */
#define MAX_ACTIVATION_SIZE (48 * 48 * 16)
static int8_t __attribute__((aligned(16))) buf_a[MAX_ACTIVATION_SIZE];
static int8_t __attribute__((aligned(16))) buf_b[MAX_ACTIVATION_SIZE];

/*
 * Accumulator buffer — must hold largest single-layer output as INT32.
 * Heap-allocated at init to avoid static DRAM pressure.
 */
static int32_t *acc_buf = NULL;

/*
 * Compute output spatial dimensions for a conv/depthwise layer.
 */
static inline int out_dim(int in_dim, int kernel, int stride, int padding)
{
    return (in_dim + 2 * padding - kernel) / stride + 1;
}

void inference_init(void)
{
    if (acc_buf) return;
    acc_buf = heap_caps_malloc(MAX_ACTIVATION_SIZE * sizeof(int32_t),
                               MALLOC_CAP_INTERNAL);
    if (!acc_buf) {
        ESP_LOGE(TAG, "Failed to allocate accumulator buffer (%d bytes)",
                 (int)(MAX_ACTIVATION_SIZE * sizeof(int32_t)));
    }
}

int classify_image(const int8_t *input_96x96x3)
{
    int8_t *in = buf_a;
    int8_t *out = buf_b;

    /* Copy input (HWC format, 96×96×3) */
    memcpy(in, input_96x96x3, 96 * 96 * 3);

    int cur_h = 96, cur_w = 96;

    for (int i = 0; i < NUM_LAYERS; i++) {
        const layer_config_t *L = &model_layers[i];

        switch (L->type) {

        case LAYER_CONV2D: {
            int h_out = out_dim(cur_h, L->kernel, L->stride, L->padding);
            int w_out = out_dim(cur_w, L->kernel, L->stride, L->padding);
            int out_count = h_out * w_out * L->out_c;

            if (L->quant == QUANT_TERNARY) {
                ternary_conv2d_simd(
                    in, (const uint8_t *)L->weights, acc_buf,
                    L->scale_pos, L->scale_neg,
                    cur_h, cur_w, L->in_c,
                    L->out_c, L->kernel,
                    L->stride, L->padding);
            } else {
                int8_conv2d(
                    in, (const int8_t *)L->weights, L->bias, acc_buf,
                    cur_h, cur_w, L->in_c,
                    L->out_c, L->kernel,
                    L->stride, L->padding);
            }

            requantize_i32_to_i8(acc_buf, out, out_count,
                                 L->requant_scale, L->requant_zp);
            relu_i8(out, out_count);

            cur_h = h_out;
            cur_w = w_out;
            break;
        }

        case LAYER_DEPTHWISE_CONV2D: {
            int h_out = out_dim(cur_h, L->kernel, L->stride, L->padding);
            int w_out = out_dim(cur_w, L->kernel, L->stride, L->padding);
            int out_count = h_out * w_out * L->out_c;

            if (L->quant == QUANT_TERNARY) {
                /* Depthwise ternary: use conv2d_ref with C_out == C_in, groups=C */
                ternary_conv2d_simd(
                    in, (const uint8_t *)L->weights, acc_buf,
                    L->scale_pos, L->scale_neg,
                    cur_h, cur_w, L->in_c,
                    L->out_c, L->kernel,
                    L->stride, L->padding);
            } else {
                int8_depthwise_conv2d(
                    in, (const int8_t *)L->weights, L->bias, acc_buf,
                    cur_h, cur_w, L->in_c,
                    L->kernel, L->stride, L->padding);
            }

            requantize_i32_to_i8(acc_buf, out, out_count,
                                 L->requant_scale, L->requant_zp);
            relu_i8(out, out_count);

            cur_h = h_out;
            cur_w = w_out;
            break;
        }

        case LAYER_DENSE: {
            int n_in = L->in_c;
            int n_out = L->out_c;

            if (L->quant == QUANT_TERNARY) {
                ternary_dense_simd(
                    in, (const uint8_t *)L->weights, acc_buf,
                    L->scale_pos, L->scale_neg,
                    n_in, n_out);
            } else {
                int8_dense(
                    in, (const int8_t *)L->weights, L->bias, acc_buf,
                    n_in, n_out);
            }

            requantize_i32_to_i8(acc_buf, out, n_out,
                                 L->requant_scale, L->requant_zp);
            /* No ReLU on final dense layer */
            cur_h = 1;
            cur_w = 1;
            break;
        }

        case LAYER_GLOBAL_AVG_POOL: {
            global_avg_pool(in, out, cur_h, cur_w, L->in_c);
            cur_h = 1;
            cur_w = 1;
            break;
        }
        }

        /* Swap ping-pong buffers */
        int8_t *tmp = in;
        in = out;
        out = tmp;
    }

    /* 'in' now points to final output: [2] softmax logits as INT8 */
    return (in[0] > in[1]) ? CLASS_CAT : CLASS_DOG;
}

void inference_print_memory_map(void)
{
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t min_internal = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    size_t free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG, "MEMORY sram_free=%u sram_min_free=%u psram_free=%u "
             "buf_a=%u buf_b=%u acc_buf=%u",
             (unsigned)free_internal, (unsigned)min_internal,
             (unsigned)free_spiram,
             (unsigned)sizeof(buf_a), (unsigned)sizeof(buf_b),
             (unsigned)sizeof(acc_buf));
}
