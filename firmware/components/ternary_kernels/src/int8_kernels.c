#include "ternary_ops.h"
#include <string.h>

/*
 * INT8 convolution and dense kernels (reference C implementations).
 * Used for first/last layers (INT8) and depthwise layers.
 * All weights in NHWC layout: [C_out, K, K, C_in] for conv, [N_out, N_in] for dense.
 */

void int8_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding,
    int y_out_start, int y_out_count)
{
    int W_out = (W + 2 * padding - K) / stride + 1;

    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = y_out_start; oh < y_out_start + y_out_count; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int32_t acc = bias ? bias[oc] : 0;

                for (int kh = 0; kh < K; kh++) {
                    int ih = oh * stride - padding + kh;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow * stride - padding + kw;
                        if (iw < 0 || iw >= W) continue;

                        for (int ic = 0; ic < C_in; ic++) {
                            int8_t a = input[ih * W * C_in + iw * C_in + ic];
                            int8_t w = weights[oc * K * K * C_in + kh * K * C_in + kw * C_in + ic];
                            acc += (int32_t)a * (int32_t)w;
                        }
                    }
                }

                output[(oh - y_out_start) * W_out * C_out + ow * C_out + oc] = acc;
            }
        }
    }
}

void int8_depthwise_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C,
    int K, int stride, int padding,
    int y_out_start, int y_out_count)
{
    int W_out = (W + 2 * padding - K) / stride + 1;

    for (int c = 0; c < C; c++) {
        for (int oh = y_out_start; oh < y_out_start + y_out_count; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int32_t acc = bias ? bias[c] : 0;

                for (int kh = 0; kh < K; kh++) {
                    int ih = oh * stride - padding + kh;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow * stride - padding + kw;
                        if (iw < 0 || iw >= W) continue;

                        int8_t a = input[ih * W * C + iw * C + c];
                        // Depthwise weights: [C, K, K, 1] but stored as [C, K, K]
                        int8_t w = weights[c * K * K + kh * K + kw];
                        acc += (int32_t)a * (int32_t)w;
                    }
                }

                output[(oh - y_out_start) * W_out * C + ow * C + c] = acc;
            }
        }
    }
}

void int8_dense(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int N_in, int N_out)
{
    for (int o = 0; o < N_out; o++) {
        int32_t acc = bias ? bias[o] : 0;
        for (int i = 0; i < N_in; i++) {
            acc += (int32_t)input[i] * (int32_t)weights[o * N_in + i];
        }
        output[o] = acc;
    }
}

void global_avg_pool(
    const int8_t *input,
    int8_t *output,
    int H, int W, int C)
{
    int count = H * W;
    for (int c = 0; c < C; c++) {
        int32_t sum = 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[h * W * C + w * C + c];
            }
        }
        int32_t avg = (sum + count / 2) / count;
        if (avg < -128) avg = -128;
        if (avg > 127) avg = 127;
        output[c] = (int8_t)avg;
    }
}

void relu_i8(int8_t *data, int count)
{
    for (int i = 0; i < count; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}
