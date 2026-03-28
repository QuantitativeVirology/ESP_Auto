# PSRAM Ping-Pong + Per-Row Accumulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the inference engine to use PSRAM for one activation buffer and per-row accumulator processing, enabling autoresearch to explore models up to alpha=1.0+.

**Architecture:** Conv2d/depthwise kernels gain `y_out_start`/`y_out_count` parameters to compute partial output rows. inference.c allocates one buffer in SRAM, one in PSRAM, and a small per-row INT32 accumulator in SRAM. The inference loop processes conv layers one output row at a time.

**Tech Stack:** C (ESP-IDF), Xtensa SIMD assembly (unchanged inner loops)

**Spec:** `docs/superpowers/specs/2026-03-28-psram-ping-pong-design.md`

---

### Task 1: Add row-range parameters to kernel header

**Files:**
- Modify: `firmware/components/ternary_kernels/include/ternary_ops.h:14-22` (ternary_conv2d_ref)
- Modify: `firmware/components/ternary_kernels/include/ternary_ops.h:24-33` (ternary_conv2d_simd)
- Modify: `firmware/components/ternary_kernels/include/ternary_ops.h:60-67` (int8_conv2d)
- Modify: `firmware/components/ternary_kernels/include/ternary_ops.h:69-75` (int8_depthwise_conv2d)

- [ ] **Step 1: Update ternary_conv2d_ref signature**

Add `int y_out_start, int y_out_count` as the last two parameters:

```c
void ternary_conv2d_ref(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding,
    int y_out_start, int y_out_count);
```

- [ ] **Step 2: Update ternary_conv2d_simd signature**

Same parameters added:

```c
void ternary_conv2d_simd(
    const int8_t *input,
    const uint8_t *weights,
    int32_t *output,
    float scale_pos,
    float scale_neg,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding,
    int y_out_start, int y_out_count);
```

- [ ] **Step 3: Update int8_conv2d signature**

```c
void int8_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C_in,
    int C_out, int K,
    int stride, int padding,
    int y_out_start, int y_out_count);
```

- [ ] **Step 4: Update int8_depthwise_conv2d signature**

```c
void int8_depthwise_conv2d(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    int H, int W, int C,
    int K, int stride, int padding,
    int y_out_start, int y_out_count);
```

---

### Task 2: Update ternary_conv2d_ref with row-range support

**Files:**
- Modify: `firmware/components/ternary_kernels/src/ternary_conv2d_ref.c:30-87`

- [ ] **Step 1: Update function signature and memset**

Change the signature to accept the new params. Replace the full-tensor memset with a row-range memset. Change the `oh` loop bounds and output indexing:

```c
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
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

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
```

---

### Task 3: Update ternary_conv2d_simd with row-range support

**Files:**
- Modify: `firmware/components/ternary_kernels/src/ternary_conv2d_simd.c:84-137`

- [ ] **Step 1: Update function signature, memset, loop bounds, and output index**

```c
void ternary_conv2d_simd(
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
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int C_in_padded = (C_in + 63) & ~63;
    int row_packed_bytes = (C_in_padded / 64) * 16;

    static int8_t __attribute__((aligned(16))) input_row[MAX_EXPANDED_ROW];

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

                        int w_row = oc * K * K + kh * K + kw;
                        const uint8_t *packed_row = weights + w_row * row_packed_bytes;

                        expand_ternary_row(packed_row, expanded_weights, C_in_padded);

                        memset(input_row, 0, C_in_padded);
                        memcpy(input_row, &input[ih * W * C_in + iw * C_in], C_in);

                        acc += simd_dot_i8(input_row, expanded_weights, C_in_padded);
                    }
                }

                output[(oh - y_out_start) * W_out * C_out + ow * C_out + oc] = acc;
            }
        }
    }
}
```

---

### Task 4: Update int8_conv2d and int8_depthwise_conv2d with row-range support

**Files:**
- Modify: `firmware/components/ternary_kernels/src/int8_kernels.c:10-47` (int8_conv2d)
- Modify: `firmware/components/ternary_kernels/src/int8_kernels.c:49-84` (int8_depthwise_conv2d)

- [ ] **Step 1: Update int8_conv2d**

Change signature, loop bounds from `oh = 0..H_out` to `oh = y_out_start..y_out_start+y_out_count`, output index uses `(oh - y_out_start)`:

```c
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
```

- [ ] **Step 2: Update int8_depthwise_conv2d**

```c
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
                        int8_t w = weights[c * K * K + kh * K + kw];
                        acc += (int32_t)a * (int32_t)w;
                    }
                }

                output[(oh - y_out_start) * W_out * C + ow * C + c] = acc;
            }
        }
    }
}
```

---

### Task 5: Update existing tests + add row-range equivalence tests

**Files:**
- Modify: `firmware/components/ternary_kernels/test/test_kernels.c`

- [ ] **Step 1: Update existing test_conv2d to use new API**

Add `0, H_out` as last two args to both `ternary_conv2d_ref` and `ternary_conv2d_simd` calls in the existing `test_conv2d` function:

```c
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

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_bytes; i++) weights[i] = rand_uint8();

    ternary_conv2d_ref(input, weights, out_ref,
                       1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                       0, H_out);
    ternary_conv2d_simd(input, weights, out_simd,
                        1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                        0, H_out);

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
```

- [ ] **Step 2: Add per-row equivalence test for ternary conv2d**

This test verifies that computing row-by-row (y_out_count=1) and concatenating matches a single full call:

```c
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
    int32_t *out_row = (int32_t *)calloc(row_size, sizeof(int32_t));
    int32_t *out_concat = (int32_t *)calloc(output_size, sizeof(int32_t));

    if (!input || !weights || !out_full || !out_row || !out_concat) {
        ESP_LOGE(TAG, "Allocation failed for conv2d rowwise test");
        free(input); free(weights); free(out_full); free(out_row); free(out_concat);
        return -1;
    }

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_bytes; i++) weights[i] = rand_uint8();

    // Full computation
    ternary_conv2d_ref(input, weights, out_full,
                       1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                       0, H_out);

    // Row-by-row computation
    for (int y = 0; y < H_out; y++) {
        ternary_conv2d_ref(input, weights, out_row,
                           1.0f, 1.0f, H, W, C_in, C_out, K, stride, padding,
                           y, 1);
        memcpy(&out_concat[y * row_size], out_row, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_concat[i]) {
            ESP_LOGE(TAG, "FAIL conv2d_rowwise [%dx%dx%d->%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C_in, C_out, K, stride, padding,
                     i, (int)out_full[i], (int)out_concat[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(out_full); free(out_row); free(out_concat);
    return pass ? 0 : -1;
}
```

- [ ] **Step 3: Add per-row equivalence test for int8 conv2d**

```c
static int test_int8_conv2d_rowwise(int H, int W, int C_in, int C_out,
                                    int K, int stride, int padding)
{
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int output_size = H_out * W_out * C_out;
    int row_size = W_out * C_out;

    int input_size = H * W * C_in;
    int weight_size = C_out * K * K * C_in;

    int8_t *input = (int8_t *)calloc(input_size, 1);
    int8_t *weights = (int8_t *)calloc(weight_size, 1);
    int32_t *bias = (int32_t *)calloc(C_out, sizeof(int32_t));
    int32_t *out_full = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_row = (int32_t *)calloc(row_size, sizeof(int32_t));
    int32_t *out_concat = (int32_t *)calloc(output_size, sizeof(int32_t));

    if (!input || !weights || !bias || !out_full || !out_row || !out_concat) {
        ESP_LOGE(TAG, "Allocation failed for int8 conv2d rowwise test");
        free(input); free(weights); free(bias);
        free(out_full); free(out_row); free(out_concat);
        return -1;
    }

    for (int i = 0; i < input_size; i++) input[i] = rand_int8();
    for (int i = 0; i < weight_size; i++) weights[i] = rand_int8();
    for (int i = 0; i < C_out; i++) bias[i] = (int32_t)(xorshift32() & 0xFFFF) - 0x8000;

    int8_conv2d(input, weights, bias, out_full,
                H, W, C_in, C_out, K, stride, padding, 0, H_out);

    for (int y = 0; y < H_out; y++) {
        int8_conv2d(input, weights, bias, out_row,
                    H, W, C_in, C_out, K, stride, padding, y, 1);
        memcpy(&out_concat[y * row_size], out_row, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_concat[i]) {
            ESP_LOGE(TAG, "FAIL int8_conv2d_rowwise [%dx%dx%d->%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C_in, C_out, K, stride, padding,
                     i, (int)out_full[i], (int)out_concat[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(bias);
    free(out_full); free(out_row); free(out_concat);
    return pass ? 0 : -1;
}
```

- [ ] **Step 4: Add per-row equivalence test for int8 depthwise conv2d**

```c
static int test_int8_dw_conv2d_rowwise(int H, int W, int C,
                                       int K, int stride, int padding)
{
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int output_size = H_out * W_out * C;
    int row_size = W_out * C;

    int8_t *input = (int8_t *)calloc(H * W * C, 1);
    int8_t *weights = (int8_t *)calloc(C * K * K, 1);
    int32_t *bias = (int32_t *)calloc(C, sizeof(int32_t));
    int32_t *out_full = (int32_t *)calloc(output_size, sizeof(int32_t));
    int32_t *out_row = (int32_t *)calloc(row_size, sizeof(int32_t));
    int32_t *out_concat = (int32_t *)calloc(output_size, sizeof(int32_t));

    if (!input || !weights || !bias || !out_full || !out_row || !out_concat) {
        ESP_LOGE(TAG, "Allocation failed for int8 dw conv2d rowwise test");
        free(input); free(weights); free(bias);
        free(out_full); free(out_row); free(out_concat);
        return -1;
    }

    for (int i = 0; i < H * W * C; i++) input[i] = rand_int8();
    for (int i = 0; i < C * K * K; i++) weights[i] = rand_int8();
    for (int i = 0; i < C; i++) bias[i] = (int32_t)(xorshift32() & 0xFFFF) - 0x8000;

    int8_depthwise_conv2d(input, weights, bias, out_full,
                          H, W, C, K, stride, padding, 0, H_out);

    for (int y = 0; y < H_out; y++) {
        int8_depthwise_conv2d(input, weights, bias, out_row,
                              H, W, C, K, stride, padding, y, 1);
        memcpy(&out_concat[y * row_size], out_row, row_size * sizeof(int32_t));
    }

    int pass = 1;
    for (int i = 0; i < output_size; i++) {
        if (out_full[i] != out_concat[i]) {
            ESP_LOGE(TAG, "FAIL int8_dw_conv2d_rowwise [%dx%dx%d K=%d s=%d p=%d] "
                     "idx=%d full=%d rowwise=%d",
                     H, W, C, K, stride, padding,
                     i, (int)out_full[i], (int)out_concat[i]);
            pass = 0;
            break;
        }
    }

    free(input); free(weights); free(bias);
    free(out_full); free(out_row); free(out_concat);
    return pass ? 0 : -1;
}
```

- [ ] **Step 5: Register new tests in run_kernel_tests**

Add the rowwise tests after the existing test loops in `run_kernel_tests()`:

```c
    // Row-wise equivalence tests (ternary conv2d)
    ESP_LOGI(TAG, "--- Row-wise equivalence tests ---");
    for (int i = 0; i < (int)(sizeof(conv_cases) / sizeof(conv_cases[0])); i++) {
        rng_state = 0xDEADBEEF + i;
        int r = test_conv2d_rowwise(conv_cases[i].H, conv_cases[i].W,
                                    conv_cases[i].C_in, conv_cases[i].C_out,
                                    conv_cases[i].K, conv_cases[i].stride,
                                    conv_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS conv2d_rowwise [%dx%dx%d->%d K=%d s=%d p=%d]",
                      conv_cases[i].H, conv_cases[i].W,
                      conv_cases[i].C_in, conv_cases[i].C_out,
                      conv_cases[i].K, conv_cases[i].stride,
                      conv_cases[i].pad);
    }

    // Row-wise equivalence tests (int8 conv2d)
    struct { int H, W, C_in, C_out, K, stride, pad; } int8_conv_cases[] = {
        { 8, 8, 3, 16, 3, 2, 1 },
        { 12, 12, 16, 32, 1, 1, 0 },
        { 6, 6, 32, 64, 3, 1, 1 },
    };

    for (int i = 0; i < (int)(sizeof(int8_conv_cases) / sizeof(int8_conv_cases[0])); i++) {
        rng_state = 0xFEEDFACE + i;
        int r = test_int8_conv2d_rowwise(
            int8_conv_cases[i].H, int8_conv_cases[i].W,
            int8_conv_cases[i].C_in, int8_conv_cases[i].C_out,
            int8_conv_cases[i].K, int8_conv_cases[i].stride,
            int8_conv_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS int8_conv2d_rowwise [%dx%dx%d->%d K=%d s=%d p=%d]",
                      int8_conv_cases[i].H, int8_conv_cases[i].W,
                      int8_conv_cases[i].C_in, int8_conv_cases[i].C_out,
                      int8_conv_cases[i].K, int8_conv_cases[i].stride,
                      int8_conv_cases[i].pad);
    }

    // Row-wise equivalence tests (int8 depthwise conv2d)
    struct { int H, W, C, K, stride, pad; } dw_cases[] = {
        { 8, 8, 16, 3, 1, 1 },
        { 12, 12, 32, 3, 2, 1 },
        { 6, 6, 64, 3, 1, 1 },
    };

    for (int i = 0; i < (int)(sizeof(dw_cases) / sizeof(dw_cases[0])); i++) {
        rng_state = 0xBADC0DE0 + i;
        int r = test_int8_dw_conv2d_rowwise(
            dw_cases[i].H, dw_cases[i].W, dw_cases[i].C,
            dw_cases[i].K, dw_cases[i].stride, dw_cases[i].pad);
        if (r != 0) failures++;
        else ESP_LOGI(TAG, "PASS int8_dw_conv2d_rowwise [%dx%dx%d K=%d s=%d p=%d]",
                      dw_cases[i].H, dw_cases[i].W, dw_cases[i].C,
                      dw_cases[i].K, dw_cases[i].stride, dw_cases[i].pad);
    }
```

- [ ] **Step 6: Commit kernel changes**

```bash
git add firmware/components/ternary_kernels/
git commit -m "Add row-range params to conv2d/depthwise kernels + rowwise tests"
```

---

### Task 6: Rework inference.c — PSRAM ping-pong + per-row accumulator

**Files:**
- Modify: `firmware/main/inference.c`
- Modify: `firmware/main/inference.h:32-34`

- [ ] **Step 1: Add inference_deinit to header**

In `firmware/main/inference.h`, add after `inference_init`:

```c
void inference_init(void);
void inference_deinit(void);
int classify_image(const int8_t *input_96x96x3);
void inference_print_memory_map(void);
```

- [ ] **Step 2: Replace buffer declarations in inference.c**

Remove the old static arrays and acc_buf. Replace with:

```c
#define INPUT_H 96
#define INPUT_W 96
#define INPUT_C 3

static int8_t *buf_sram = NULL;
static int8_t *buf_psram = NULL;
static int32_t *acc_row = NULL;
static int max_act_size = 0;
static int max_row_i32 = 0;
```

- [ ] **Step 3: Rewrite inference_init to scan layers and allocate**

```c
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

    buf_sram = heap_caps_aligned_alloc(16, max_act_size,
                                        MALLOC_CAP_INTERNAL);
    buf_psram = heap_caps_aligned_alloc(16, max_act_size,
                                         MALLOC_CAP_SPIRAM);
    acc_row = heap_caps_aligned_alloc(16, max_row_i32 * sizeof(int32_t),
                                       MALLOC_CAP_INTERNAL);

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
```

- [ ] **Step 4: Rewrite classify_image with per-row loop**

```c
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
```

- [ ] **Step 5: Update inference_print_memory_map**

```c
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
```

- [ ] **Step 6: Commit inference rework**

```bash
git add firmware/main/inference.c firmware/main/inference.h
git commit -m "Rework inference: PSRAM ping-pong buffers + per-row accumulator"
```

---

### Task 7: Build verification

**Files:** None (build-only)

- [ ] **Step 1: Source ESP-IDF and build**

```bash
bash -c 'source /Users/jensbosse/Documents/Cowork/ESP_webclock_temp_humidity/esp-idf/export.sh && cd firmware && idf.py build'
```

Expected: Build succeeds with no errors. Warnings about unused H_out variables are acceptable (the kernels compute it but may not use it directly anymore).

- [ ] **Step 2: Fix any build errors**

If the build fails, fix compilation errors. Common issues:
- Missing `#include <stdlib.h>` in test_kernels.c for the new test functions
- `heap_caps_aligned_alloc` needs `#include "esp_heap_caps.h"` (already included)
- `heap_caps_free` used instead of `free` for aligned allocs (correct for ESP-IDF)

- [ ] **Step 3: Final commit if fixes were needed**

```bash
git add firmware/
git commit -m "Fix build issues from PSRAM ping-pong rework"
```
