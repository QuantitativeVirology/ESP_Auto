#include "ternary_ops.h"
#include <string.h>

/*
 * Ternary Conv2d reference implementation.
 *
 * Weight packing (per 16-byte / 128-bit block, covering 64 weights):
 *   bytes [0:7]  = zero_mask bits (1 = non-zero, 0 = zero)
 *   bytes [8:15] = sign_bits (0 = positive, 1 = negative)
 *
 * Weight traversal order: [C_out, K, K, C_in] (NHWC-style, C_in innermost).
 * Weights are padded so C_in is a multiple of 64.
 */

static inline void unpack_weight(const uint8_t *packed, int idx,
                                 int *is_nonzero, int *is_negative)
{
    // Each 128-bit block covers 64 weights
    int block = idx / 64;
    int bit = idx % 64;
    int byte_in_block = bit / 8;
    int bit_in_byte = bit % 8;

    // zero_mask is in bytes [0:7] of the block, sign is in bytes [8:15]
    const uint8_t *block_ptr = packed + block * 16;
    *is_nonzero = (block_ptr[byte_in_block] >> bit_in_byte) & 1;
    *is_negative = (block_ptr[8 + byte_in_block] >> bit_in_byte) & 1;
}

void ternary_conv2d_ref(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding,
    int y_out_start, int y_out_count)
{
    int W_out = (W + 2 * padding - K) / stride + 1;

    // C_in padded to multiple of 64 for packing
    int C_in_padded = (C_in + 63) & ~63;

    memset(output, 0, y_out_count * W_out * C_out * sizeof(int32_t));

    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = y_out_start; oh < y_out_start + y_out_count; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int32_t acc = 0;

                for (int kh = 0; kh < K; kh++) {
                    int ih = oh * stride - padding + kh;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow * stride - padding + kw;
                        if (iw < 0 || iw >= W) continue;

                        for (int ic = 0; ic < C_in; ic++) {
                            // Weight index in [C_out, K, K, C_in_padded] order
                            int w_idx = oc * K * K * C_in_padded
                                      + kh * K * C_in_padded
                                      + kw * C_in_padded
                                      + ic;

                            int nz, neg;
                            unpack_weight(weights, w_idx, &nz, &neg);

                            if (nz) {
                                int8_t act = input[ih * W * C_in + iw * C_in + ic];
                                if (neg) {
                                    acc -= act;
                                } else {
                                    acc += act;
                                }
                            }
                        }
                    }
                }

                output[(oh - y_out_start) * W_out * C_out + ow * C_out + oc] = acc;
            }
        }
    }
}

void requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    int count,
    float scale,
    int8_t zero_point)
{
    for (int i = 0; i < count; i++) {
        int32_t val = (int32_t)(input[i] * scale + 0.5f) + zero_point;
        if (val < -128) val = -128;
        if (val > 127) val = 127;
        output[i] = (int8_t)val;
    }
}
