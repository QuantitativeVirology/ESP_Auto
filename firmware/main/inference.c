#include "inference.h"
#include "ternary_ops.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>

static const char *TAG = "inference";

#include "model_data.h"

#define INPUT_H 96
#define INPUT_W 96
#define INPUT_C 3

static int8_t *buf_sram = NULL;
static int8_t *buf_psram = NULL;
static int32_t *acc_row = NULL;
static int max_act_size = 0;
static int max_row_i32 = 0;

static inline int out_dim(int in_dim, int kernel, int stride, int padding)
{
    return (in_dim + 2 * padding - kernel) / stride + 1;
}

void inference_init(void)
{
    if (buf_sram) return;

    int cur_h = INPUT_H, cur_w = INPUT_W;
    max_act_size = INPUT_H * INPUT_W * INPUT_C;
    max_row_i32 = 0;

    for (int i = 0; i < NUM_LAYERS; i++) {
        const layer_config_t *L = &model_layers[i];

        switch (L->type) {
        case LAYER_CONV2D:
        case LAYER_DEPTHWISE_CONV2D: {
            int h_out = out_dim(cur_h, L->kernel, L->stride, L->padding);
            int w_out = out_dim(cur_w, L->kernel, L->stride, L->padding);
            int act = h_out * w_out * L->out_c;
            if (act > max_act_size) max_act_size = act;
            int row = w_out * L->out_c;
            if (row > max_row_i32) max_row_i32 = row;
            cur_h = h_out;
            cur_w = w_out;
            break;
        }
        case LAYER_DENSE:
            if (L->out_c > max_row_i32) max_row_i32 = L->out_c;
            cur_h = 1; cur_w = 1;
            break;
        case LAYER_GLOBAL_AVG_POOL:
            cur_h = 1; cur_w = 1;
            break;
        }
    }

    buf_sram  = heap_caps_aligned_alloc(16, max_act_size, MALLOC_CAP_INTERNAL);
    buf_psram = heap_caps_aligned_alloc(16, max_act_size, MALLOC_CAP_SPIRAM);
    acc_row   = heap_caps_aligned_alloc(16, max_row_i32 * sizeof(int32_t), MALLOC_CAP_INTERNAL);

    if (!buf_sram || !buf_psram || !acc_row) {
        ESP_LOGE(TAG, "Buffer allocation failed: sram=%p psram=%p acc=%p",
                 buf_sram, buf_psram, acc_row);
        return;
    }

    ESP_LOGI(TAG, "Buffers: sram=%d psram=%d acc_row=%d bytes",
             max_act_size, max_act_size, (int)(max_row_i32 * sizeof(int32_t)));
}

void inference_deinit(void)
{
    heap_caps_free(buf_sram);  buf_sram = NULL;
    heap_caps_free(buf_psram); buf_psram = NULL;
    heap_caps_free(acc_row);   acc_row = NULL;
}

int classify_image(const int8_t *input_96x96x3)
{
    int8_t *in = buf_sram;
    int8_t *out = buf_psram;

    memcpy(in, input_96x96x3, INPUT_H * INPUT_W * INPUT_C);

    int cur_h = INPUT_H, cur_w = INPUT_W;

    for (int i = 0; i < NUM_LAYERS; i++) {
        const layer_config_t *L = &model_layers[i];

        switch (L->type) {

        case LAYER_CONV2D: {
            int h_out = out_dim(cur_h, L->kernel, L->stride, L->padding);
            int w_out = out_dim(cur_w, L->kernel, L->stride, L->padding);
            int row_elems = w_out * L->out_c;

            for (int y = 0; y < h_out; y++) {
                if (L->quant == QUANT_TERNARY) {
                    ternary_conv2d_simd(
                        in, (const uint8_t *)L->weights, acc_row,
                        L->scale_pos, L->scale_neg,
                        cur_h, cur_w, L->in_c,
                        L->out_c, L->kernel,
                        L->stride, L->padding, y, 1);
                } else {
                    int8_conv2d(
                        in, (const int8_t *)L->weights, L->bias, acc_row,
                        cur_h, cur_w, L->in_c,
                        L->out_c, L->kernel,
                        L->stride, L->padding, y, 1);
                }

                requantize_i32_to_i8(acc_row, &out[y * row_elems], row_elems,
                                     L->requant_scale, L->requant_zp);
                relu_i8(&out[y * row_elems], row_elems);
            }

            cur_h = h_out;
            cur_w = w_out;
            break;
        }

        case LAYER_DEPTHWISE_CONV2D: {
            int h_out = out_dim(cur_h, L->kernel, L->stride, L->padding);
            int w_out = out_dim(cur_w, L->kernel, L->stride, L->padding);
            int row_elems = w_out * L->out_c;

            for (int y = 0; y < h_out; y++) {
                if (L->quant == QUANT_TERNARY) {
                    ternary_conv2d_simd(
                        in, (const uint8_t *)L->weights, acc_row,
                        L->scale_pos, L->scale_neg,
                        cur_h, cur_w, L->in_c,
                        L->out_c, L->kernel,
                        L->stride, L->padding, y, 1);
                } else {
                    int8_depthwise_conv2d(
                        in, (const int8_t *)L->weights, L->bias, acc_row,
                        cur_h, cur_w, L->in_c,
                        L->kernel, L->stride, L->padding, y, 1);
                }

                requantize_i32_to_i8(acc_row, &out[y * row_elems], row_elems,
                                     L->requant_scale, L->requant_zp);
                relu_i8(&out[y * row_elems], row_elems);
            }

            cur_h = h_out;
            cur_w = w_out;
            break;
        }

        case LAYER_DENSE: {
            int n_in = L->in_c;
            int n_out = L->out_c;

            if (L->quant == QUANT_TERNARY) {
                ternary_dense_simd(
                    in, (const uint8_t *)L->weights, acc_row,
                    L->scale_pos, L->scale_neg,
                    n_in, n_out);
            } else {
                int8_dense(
                    in, (const int8_t *)L->weights, L->bias, acc_row,
                    n_in, n_out);
            }

            requantize_i32_to_i8(acc_row, out, n_out,
                                 L->requant_scale, L->requant_zp);
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

    return (in[0] > in[1]) ? CLASS_CAT : CLASS_DOG;
}

void inference_print_memory_map(void)
{
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t min_internal = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    size_t free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG, "MEMORY sram_free=%u sram_min_free=%u psram_free=%u "
             "buf_sram=%d buf_psram=%d acc_row=%d",
             (unsigned)free_internal, (unsigned)min_internal,
             (unsigned)free_spiram,
             max_act_size, max_act_size,
             (int)(max_row_i32 * sizeof(int32_t)));
}
