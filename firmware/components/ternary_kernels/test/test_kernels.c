#include "ternary_ops.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const char *TAG = "test_kernels";

// Simple pseudo-random for reproducible tests
static uint32_t rng_state = 0xDEADBEEF;
static uint32_t xorshift32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int8_t rand_int8(void)
{
    return (int8_t)(xorshift32() & 0xFF);
}

static uint8_t rand_uint8(void)
{
    return (uint8_t)(xorshift32() & 0xFF);
}

// ---------------------------------------------------------------------------
// Conv2d tests
// ---------------------------------------------------------------------------

static int test_conv2d(int H, int W, int C_in, int C_out,
                       int K, int stride, int padding)
{
    int C_in_padded = (C_in + 63) & ~63;
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    int input_size = H * W * C_in;
    int weight_blocks = C_out * K * K * C_in_padded / 64;
    int weight_bytes = weight_blocks * 16;
    int output_size = H_out * W_out * C_out;

    int8_t *input = (int8_t *)calloc(input_size, 1);
    uint8_t *weights = (uint8_t *)calloc(weight_bytes, 1);
    int32_t *out_ref = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_simd = (int32_t *)calloc(output_size, sizeof(int32_t));

    if (!input || !weights || !out_ref || !out_simd) {
        ESP_LOGE(TAG, "Allocation failed for conv2d test");
        free(input); free(weights); free(out_ref); free(out_simd);
        return -1;
    }

    // Fill with random data
    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_bytes; i++) weights[i] = rand_uint8();

    // Run both implementations
    ternary_conv2d_ref(input, weights, out_ref,
                       1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                       0, H_out);
    ternary_conv2d_simd(input, weights, out_simd,
                        1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                        0, H_out);

    // Compare (must be bit-exact)
    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_ref[i] != out_simd[i]) {
            ESP_LOGE(TAG, "FAIL conv2d [%dx%dx%d->%d K=%d s=%d p=%d] "
                     "idx=%d ref=%d simd=%d",
                     H, W, C_in, C_out, K, stride, padding,
                     i, (int)out_ref[i], (int)out_simd[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_ref); free(out_simd);
    return pass ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Conv2d rowwise equivalence test
// ---------------------------------------------------------------------------

static int test_conv2d_rowwise(int H, int W, int C_in, int C_out,
                                int K, int stride, int padding)
{
    int C_in_padded = (C_in + 63) & ~63;
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    int input_size = H * W * C_in;
    int weight_blocks = C_out * K * K * C_in_padded / 64;
    int weight_bytes = weight_blocks * 16;
    int output_size = H_out * W_out * C_out;
    int row_size = W_out * C_out;

    int8_t *input = (int8_t *)calloc(input_size, 1);
    uint8_t *weights = (uint8_t *)calloc(weight_bytes, 1);
    int32_t *out_full = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_rows = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *row_buf = (int32_t *)calloc(row_size, sizeof(int32_t));

    if (!input || !weights || !out_full || !out_rows || !row_buf) {
        ESP_LOGE(TAG, "Allocation failed for conv2d rowwise test");
        free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
        return -1;
    }

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_bytes; i++) weights[i] = rand_uint8();

    // Full output in one call
    ternary_conv2d_ref(input, weights, out_full,
                       1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                       0, H_out);

    // Row-by-row, concatenate into out_rows
    for (int row = 0; row < H_out; row++) {
        ternary_conv2d_ref(input, weights, row_buf,
                           1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                           row, 1);
        memcpy(&out_rows[row * row_size], row_buf, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_rows[i]) {
            ESP_LOGE(TAG, "FAIL conv2d rowwise [%dx%dx%d->%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C_in, C_out, K, stride, padding,
                     i, (int)out_full[i], (int)out_rows[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
    return pass ? 0 : -1;
}

// ---------------------------------------------------------------------------
// INT8 conv2d rowwise equivalence test
// ---------------------------------------------------------------------------

static int test_int8_conv2d_rowwise(int H, int W, int C_in, int C_out,
                                     int K, int stride, int padding)
{
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    int input_size = H * W * C_in;
    int weight_size = C_out * K * K * C_in;
    int output_size = H_out * W_out * C_out;
    int row_size = W_out * C_out;

    int8_t *input = (int8_t *)calloc(input_size, 1);
    int8_t *weights = (int8_t *)calloc(weight_size, 1);
    int32_t *out_full = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_rows = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *row_buf = (int32_t *)calloc(row_size, sizeof(int32_t));

    if (!input || !weights || !out_full || !out_rows || !row_buf) {
        ESP_LOGE(TAG, "Allocation failed for int8 conv2d rowwise test");
        free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
        return -1;
    }

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_size; i++) weights[i] = rand_int8();

    // Full output in one call
    int8_conv2d(input, weights, NULL, out_full,
                H, W, C_in, C_out, K, stride, padding,
                0, H_out);

    // Row-by-row, concatenate into out_rows
    for (int row = 0; row < H_out; row++) {
        int8_conv2d(input, weights, NULL, row_buf,
                    H, W, C_in, C_out, K, stride, padding,
                    row, 1);
        memcpy(&out_rows[row * row_size], row_buf, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_rows[i]) {
            ESP_LOGE(TAG, "FAIL int8 conv2d rowwise [%dx%dx%d->%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C_in, C_out, K, stride, padding,
                     i, (int)out_full[i], (int)out_rows[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
    return pass ? 0 : -1;
}

// ---------------------------------------------------------------------------
// INT8 depthwise conv2d rowwise equivalence test
// ---------------------------------------------------------------------------

static int test_int8_dw_conv2d_rowwise(int H, int W, int C,
                                        int K, int stride, int padding)
{
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    int input_size = H * W * C;
    int weight_size = C * K * K;
    int output_size = H_out * W_out * C;
    int row_size = W_out * C;

    int8_t *input = (int8_t *)calloc(input_size, 1);
    int8_t *weights = (int8_t *)calloc(weight_size, 1);
    int32_t *out_full = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_rows = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *row_buf = (int32_t *)calloc(row_size, sizeof(int32_t));

    if (!input || !weights || !out_full || !out_rows || !row_buf) {
        ESP_LOGE(TAG, "Allocation failed for int8 dw conv2d rowwise test");
        free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
        return -1;
    }

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_size; i++) weights[i] = rand_int8();

    // Full output in one call
    int8_depthwise_conv2d(input, weights, NULL, out_full,
                          H, W, C, K, stride, padding,
                          0, H_out);

    // Row-by-row, concatenate into out_rows
    for (int row = 0; row < H_out; row++) {
        int8_depthwise_conv2d(input, weights, NULL, row_buf,
                              H, W, C, K, stride, padding,
                              row, 1);
        memcpy(&out_rows[row * row_size], row_buf, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_rows[i]) {
            ESP_LOGE(TAG, "FAIL int8 dw conv2d rowwise [%dx%dx%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C, K, stride, padding,
                     i, (int)out_full[i], (int)out_rows[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_full); free(out_rows); free(row_buf);
    return pass ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Dense tests
// ---------------------------------------------------------------------------

static int test_dense(int N_in, int N_out)
{
    int N_in_padded = (N_in + 63) & ~63;
    int weight_blocks = N_out * N_in_padded / 64;
    int weight_bytes = weight_blocks * 16;

    int8_t *input = (int8_t *)calloc(N_in, 1);
    uint8_t *weights = (uint8_t *)calloc(weight_bytes, 1);
    int32_t *out_ref = (int32_t *)calloc(N_out, sizeof(int32_t));
    int32_t *out_simd = (int32_t *)calloc(N_out, sizeof(int32_t));

    if (!input || !weights || !out_ref || !out_simd) {
        ESP_LOGE(TAG, "Allocation failed for dense test");
        free(input); free(weights); free(out_ref); free(out_simd);
        return -1;
    }

    for (int i = 0; i < N_in; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_bytes; i++) weights[i] = rand_uint8();

    ternary_dense_ref(input, weights, out_ref, 1.0f, 1.0f, N_in, N_out);
    ternary_dense_simd(input, weights, out_simd, 1.0f, 1.0f, N_in, N_out);

    int pass = 1;
    for (int i = 0; i < N_out; i++) {
        if (out_ref[i] != out_simd[i]) {
            ESP_LOGE(TAG, "FAIL dense [%d->%d] idx=%d ref=%d simd=%d",
                     N_in, N_out, i, (int)out_ref[i], (int)out_simd[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_ref); free(out_simd);
    return pass ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

int run_kernel_tests(void)
{
    int failures = 0;

    ESP_LOGI(TAG, "=== Kernel Correctness Tests ===");

    // Conv2d tests: various C_in, K, stride, padding combinations
    struct { int H, W, C_in, C_out, K, stride, pad; } conv_cases[] = {
        { 8, 8, 64, 16, 1, 1, 0 },   // 1x1 pointwise, C_in=64
        { 8, 8, 64, 32, 3, 1, 1 },   // 3x3 conv, same padding
        { 8, 8, 128, 64, 3, 2, 1 },  // 3x3 conv, stride 2
        { 12, 12, 64, 64, 3, 1, 1 }, // larger spatial
        { 6, 6, 128, 128, 1, 1, 0 }, // big pointwise
        { 4, 4, 256, 128, 3, 1, 1 }, // deep channels
    };

    for (int i = 0; i < (int)(sizeof(conv_cases) / sizeof(conv_cases[0])); i++) {
        rng_state = 0xDEADBEEF + i;  // reproducible per test
        int r = test_conv2d(conv_cases[i].H, conv_cases[i].W,
                            conv_cases[i].C_in, conv_cases[i].C_out,
                            conv_cases[i].K, conv_cases[i].stride,
                            conv_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS conv2d [%dx%dx%d->%d K=%d s=%d p=%d]",
                      conv_cases[i].H, conv_cases[i].W,
                      conv_cases[i].C_in, conv_cases[i].C_out,
                      conv_cases[i].K, conv_cases[i].stride,
                      conv_cases[i].pad);
    }

    // Dense tests
    struct { int N_in, N_out; } dense_cases[] = {
        { 64, 16 },
        { 128, 64 },
        { 256, 128 },
        { 256, 2 },
    };

    for (int i = 0; i < (int)(sizeof(dense_cases) / sizeof(dense_cases[0])); i++) {
        rng_state = 0xCAFEBABE + i;
        int r = test_dense(dense_cases[i].N_in, dense_cases[i].N_out);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS dense [%d->%d]",
                      dense_cases[i].N_in, dense_cases[i].N_out);
    }

    // Ternary conv2d rowwise equivalence tests
    for (int i = 0; i < (int)(sizeof(conv_cases) / sizeof(conv_cases[0])); i++) {
        rng_state = 0xDEADBEEF + i;
        int r = test_conv2d_rowwise(conv_cases[i].H, conv_cases[i].W,
                                    conv_cases[i].C_in, conv_cases[i].C_out,
                                    conv_cases[i].K, conv_cases[i].stride,
                                    conv_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS conv2d rowwise [%dx%dx%d->%d K=%d s=%d p=%d]",
                      conv_cases[i].H, conv_cases[i].W,
                      conv_cases[i].C_in, conv_cases[i].C_out,
                      conv_cases[i].K, conv_cases[i].stride,
                      conv_cases[i].pad);
    }

    // INT8 conv2d rowwise equivalence tests
    struct { int H, W, C_in, C_out, K, stride, pad; } int8_conv_cases[] = {
        { 8, 8, 3, 16, 3, 2, 1 },
        { 12, 12, 16, 32, 1, 1, 0 },
        { 6, 6, 32, 64, 3, 1, 1 },
    };

    for (int i = 0; i < (int)(sizeof(int8_conv_cases) / sizeof(int8_conv_cases[0])); i++) {
        rng_state = 0xFEEDFACE + i;
        int r = test_int8_conv2d_rowwise(int8_conv_cases[i].H, int8_conv_cases[i].W,
                                          int8_conv_cases[i].C_in, int8_conv_cases[i].C_out,
                                          int8_conv_cases[i].K, int8_conv_cases[i].stride,
                                          int8_conv_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS int8 conv2d rowwise [%dx%dx%d->%d K=%d s=%d p=%d]",
                      int8_conv_cases[i].H, int8_conv_cases[i].W,
                      int8_conv_cases[i].C_in, int8_conv_cases[i].C_out,
                      int8_conv_cases[i].K, int8_conv_cases[i].stride,
                      int8_conv_cases[i].pad);
    }

    // INT8 depthwise conv2d rowwise equivalence tests
    struct { int H, W, C, K, stride, pad; } dw_cases[] = {
        { 8, 8, 16, 3, 1, 1 },
        { 12, 12, 32, 3, 2, 1 },
        { 6, 6, 64, 3, 1, 1 },
    };

    for (int i = 0; i < (int)(sizeof(dw_cases) / sizeof(dw_cases[0])); i++) {
        rng_state = 0xBADC0DE0 + i;
        int r = test_int8_dw_conv2d_rowwise(dw_cases[i].H, dw_cases[i].W,
                                             dw_cases[i].C, dw_cases[i].K,
                                             dw_cases[i].stride, dw_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS int8 dw conv2d rowwise [%dx%dx%d K=%d s=%d p=%d]",
                      dw_cases[i].H, dw_cases[i].W, dw_cases[i].C,
                      dw_cases[i].K, dw_cases[i].stride, dw_cases[i].pad);
    }

    if (failures == 0) {
        printf("TESTS PASSED\n");
    } else {
        printf("TESTS FAILED (%d failures)\n", failures);
    }

    return failures;
}
