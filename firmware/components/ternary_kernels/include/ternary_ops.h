#pragma once
#include <stdint.h>

/**
 * Ternary Conv2d — C reference implementation (correctness oracle).
 *
 * Weights are packed as 2 bits per weight:
 *   Per 128-bit (16-byte) block covering 64 weights:
 *     bytes [0:7]  = zero_mask (1 = non-zero)
 *     bytes [8:15] = sign_bits (0 = positive, 1 = negative)
 *
 * Activations are INT8, accumulators are INT32.
 */
void ternary_conv2d_ref(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding);

/** Ternary Conv2d — SIMD-optimized (same interface as ref). */
void ternary_conv2d_simd(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding);

/**
 * Ternary Dense (fully-connected) — C reference implementation.
 */
void ternary_dense_ref(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int N_in, int N_out);

/** Ternary Dense — SIMD-optimized. */
void ternary_dense_simd(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int N_in, int N_out);

/**
 * INT8 Conv2d (standard, for first/last layers and depthwise).
 * Weights: [C_out, K, K, C_in] int8_t (NHWC order).
 * For depthwise: C_out == C_in, each filter operates on one channel.
 */
void int8_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding);

void int8_depthwise_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C,
    int K, int stride, int padding);

/** INT8 Dense (fully-connected). Weights: [N_out, N_in] int8_t. */
void int8_dense(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int N_in, int N_out);

/** Global average pooling: [H, W, C] int8 → [C] int8. */
void global_avg_pool(
    const int8_t *input,
    int8_t *output,
    int H, int W, int C);

/**
 * Requantize INT32 accumulators to INT8 activations.
 * out[i] = clamp(round(in[i] * scale) + zero_point, -128, 127)
 */
void requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    int count,
    float scale,
    int8_t zero_point);

/** ReLU in-place on INT8 activations. */
void relu_i8(int8_t *data, int count);
