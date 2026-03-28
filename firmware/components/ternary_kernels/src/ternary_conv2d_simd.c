#include "ternary_ops.h"
#include <string.h>

/*
 * SIMD-optimized ternary conv2d for ESP32-S3.
 *
 * Strategy: expand packed 2-bit ternary weights to INT8 {-1, 0, +1}
 * per-row, then use EE.VMULAS.S8.ACCX for 16-wide MAC.
 *
 * The expansion trades a small amount of computation for the ability
 * to reuse the proven INT8 MAC pipeline (same instruction as esp-nn).
 *
 * Weight packing (per 16-byte block, 64 weights):
 *   bytes [0:7]  = zero_mask (1 = non-zero)
 *   bytes [8:15] = sign_bits (0 = positive, 1 = negative)
 */

/* Max expanded row size: largest C_in_padded we'll see */
#define MAX_EXPANDED_ROW 512
static int8_t __attribute__((aligned(16))) expanded_weights[MAX_EXPANDED_ROW];

/*
 * Expand a row of packed ternary weights to INT8 {-1, 0, +1}.
 * n_weights must be a multiple of 64 (C_in_padded).
 */
static void expand_ternary_row(const uint8_t *packed, int8_t *out, int n_weights)
{
    for (int i = 0; i < n_weights; i += 64) {
        const uint8_t *block = packed + (i / 64) * 16;
        const uint8_t *zm = block;       /* zero_mask: bytes 0-7 */
        const uint8_t *sg = block + 8;   /* sign_bits: bytes 8-15 */

        for (int j = 0; j < 64; j++) {
            int byte_idx = j / 8;
            int bit_pos = j % 8;
            int nz = (zm[byte_idx] >> bit_pos) & 1;
            int neg = (sg[byte_idx] >> bit_pos) & 1;

            /* nz=0 → 0, nz=1 & neg=0 → +1, nz=1 & neg=1 → -1 */
            out[i + j] = nz ? (neg ? -1 : 1) : 0;
        }
    }
}

/*
 * Compute dot product of n INT8 values using SIMD.
 * n must be a multiple of 16. Both arrays must be 16-byte aligned.
 * Returns the INT32 accumulator result.
 */
static int32_t simd_dot_i8(const int8_t *a, const int8_t *b, int n)
{
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
    int32_t result;
    /* Use PIE SIMD: 16-wide INT8 MAC */
    __asm__ __volatile__(
        "ee.zero.accx\n"

        /* Process 16 bytes per iteration */
        "loopnez %[n16], 1f\n"
        "ee.vld.128.ip q0, %[pa], 16\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "1:\n"

        /* Extract accumulator */
        "rur.accx_0 %[res]\n"
        : [res] "=r" (result),
          [pa] "+r" (a),
          [pb] "+r" (b)
        : [n16] "r" (n / 16)
        : "memory"
    );
    return result;
#else
    /* Fallback for non-S3 targets or host compilation */
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
#endif
}

void ternary_conv2d_simd(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding)
{
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int C_in_padded = (C_in + 63) & ~63;
    int row_packed_bytes = (C_in_padded / 64) * 16;

    /* Temporary aligned buffer for one row of input (with padding handling) */
    static int8_t __attribute__((aligned(16))) input_row[MAX_EXPANDED_ROW];

    memset(output, 0, H_out * W_out * C_out * sizeof(int32_t));

    for (int oc = 0; oc < C_out; oc++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int32_t acc = 0;

                for (int kh = 0; kh < K; kh++) {
                    int ih = oh * stride - padding + kh;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow * stride - padding + kw;
                        if (iw < 0 || iw >= W) continue;

                        /* Weight index for this (oc, kh, kw) row */
                        int w_row = oc * K * K + kh * K + kw;
                        const uint8_t *packed_row = weights + w_row * row_packed_bytes;

                        /* Expand ternary weights for this row */
                        expand_ternary_row(packed_row, expanded_weights, C_in_padded);

                        /* Copy and align input row */
                        memset(input_row, 0, C_in_padded);
                        memcpy(input_row, &input[ih * W * C_in + iw * C_in], C_in);

                        /* SIMD dot product */
                        acc += simd_dot_i8(input_row, expanded_weights, C_in_padded);
                    }
                }

                output[oh * W_out * C_out + ow * C_out + oc] = acc;
            }
        }
    }
}
