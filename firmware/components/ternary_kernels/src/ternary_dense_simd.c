#include "ternary_ops.h"
#include <string.h>

/*
 * SIMD-optimized ternary dense (fully-connected) for ESP32-S3.
 * Same expand-to-INT8 + VMULAS strategy as conv2d.
 */

#define MAX_EXPANDED_ROW 512
static int8_t __attribute__((aligned(16))) expanded_weights_dense[MAX_EXPANDED_ROW];

static void expand_ternary_row(const uint8_t *packed, int8_t *out, int n_weights)
{
    for (int i = 0; i < n_weights; i += 64) {
        const uint8_t *block = packed + (i / 64) * 16;
        const uint8_t *zm = block;
        const uint8_t *sg = block + 8;

        for (int j = 0; j < 64; j++) {
            int byte_idx = j / 8;
            int bit_pos = j % 8;
            int nz = (zm[byte_idx] >> bit_pos) & 1;
            int neg = (sg[byte_idx] >> bit_pos) & 1;
            out[i + j] = nz ? (neg ? -1 : 1) : 0;
        }
    }
}

static int32_t simd_dot_i8(const int8_t *a, const int8_t *b, int n)
{
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
    int32_t result;
    __asm__ __volatile__(
        "ee.zero.accx\n"
        "loopnez %[n16], 1f\n"
        "ee.vld.128.ip q0, %[pa], 16\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "1:\n"
        "rur.accx_0 %[res]\n"
        : [res] "=r" (result),
          [pa] "+r" (a),
          [pb] "+r" (b)
        : [n16] "r" (n / 16)
        : "memory"
    );
    return result;
#else
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
#endif
}

void ternary_dense_simd(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int N_in, int N_out)
{
    int N_in_padded = (N_in + 63) & ~63;
    int row_packed_bytes = (N_in_padded / 64) * 16;

    /* Align input */
    static int8_t __attribute__((aligned(16))) input_aligned[MAX_EXPANDED_ROW];
    memset(input_aligned, 0, N_in_padded);
    memcpy(input_aligned, input, N_in);

    for (int o = 0; o < N_out; o++) {
        const uint8_t *packed_row = weights + o * row_packed_bytes;
        expand_ternary_row(packed_row, expanded_weights_dense, N_in_padded);
        output[o] = simd_dot_i8(input_aligned, expanded_weights_dense, N_in_padded);
    }
}
