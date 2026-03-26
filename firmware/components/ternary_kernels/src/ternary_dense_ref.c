#include "ternary_ops.h"
#include <string.h>

/*
 * Ternary Dense (fully-connected) reference implementation.
 *
 * Weight packing same as conv2d:
 *   Per 16-byte block (64 weights): bytes[0:7] = zero_mask, bytes[8:15] = sign.
 *
 * Weight layout: [N_out, N_in_padded] where N_in_padded = ceil(N_in / 64) * 64.
 */

static inline void unpack_weight(const uint8_t *packed, int idx,
                                 int *is_nonzero, int *is_negative)
{
    int block = idx / 64;
    int bit = idx % 64;
    int byte_in_block = bit / 8;
    int bit_in_byte = bit % 8;

    const uint8_t *block_ptr = packed + block * 16;
    *is_nonzero = (block_ptr[byte_in_block] >> bit_in_byte) & 1;
    *is_negative = (block_ptr[8 + byte_in_block] >> bit_in_byte) & 1;
}

void ternary_dense_ref(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int N_in, int N_out)
{
    int N_in_padded = (N_in + 63) & ~63;

    for (int o = 0; o < N_out; o++) {
        int32_t acc = 0;

        for (int i = 0; i < N_in; i++) {
            int w_idx = o * N_in_padded + i;

            int nz, neg;
            unpack_weight(weights, w_idx, &nz, &neg);

            if (nz) {
                if (neg) {
                    acc -= input[i];
                } else {
                    acc += input[i];
                }
            }
        }

        output[o] = acc;
    }
}
