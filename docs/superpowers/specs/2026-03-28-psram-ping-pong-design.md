# Per-Row Accumulator + PSRAM Ping-Pong Inference Engine

## Goal

Maximize the model envelope that autoresearch can explore on ESP32-S3-WROOM (512KB SRAM, 8MB PSRAM, 8MB flash) by reworking inference.c to use PSRAM for one activation buffer and per-row accumulator processing. Zero changes to the export pipeline or model format.

## Current State

- `inference.c` allocates two static INT8 ping-pong buffers (`buf_a`, `buf_b`) in SRAM, plus a heap-allocated INT32 accumulator (`acc_buf`) in SRAM.
- All three scale with the largest layer activation: `H_out * W_out * C_out`.
- At alpha=0.25: 72KB (ping-pong) + 144KB (acc) = 216KB. Fits.
- At alpha=0.5: 146KB (ping-pong) + 294KB (acc) = 440KB. Does not fit.
- Kernels compute the full output tensor in one call, then a separate `requantize_i32_to_i8()` pass converts to INT8.

## Design

### 1. Buffer allocation

Replace static buffers with heap allocation at `inference_init()`:

```c
static int8_t *buf_sram = NULL;  // heap_caps_aligned_alloc(16, ..., MALLOC_CAP_INTERNAL)
static int8_t *buf_psram = NULL; // heap_caps_aligned_alloc(16, ..., MALLOC_CAP_SPIRAM)
static int32_t *acc_row = NULL;  // heap_caps_aligned_alloc(16, ..., MALLOC_CAP_INTERNAL)
```

Buffer sizes are derived from the model's layer configs at init time by scanning `model_layers[]` for the maximum activation size. `MAX_ACTIVATION_SIZE` becomes a runtime variable, not a compile-time constant.

The SRAM buffer holds the largest single-layer activation as INT8. The PSRAM buffer holds the same size. The accumulator holds one output row: `max(W_out * C_out)` across all layers, as INT32.

### 2. Ping-pong with PSRAM

Layers alternate between SRAM and PSRAM:

```
Layer 0: input=buf_sram,  output=buf_psram  (write to PSRAM)
Layer 1: input=buf_psram, output=buf_sram   (read from PSRAM)
Layer 2: input=buf_sram,  output=buf_psram
...
```

The initial `memcpy` of the input image goes into `buf_sram`. The first layer reads from SRAM (fast) and writes to PSRAM. Every other layer pays the PSRAM read penalty.

### 3. Per-row accumulator processing

Instead of computing the full output tensor into acc_buf, the inference loop processes one output row at a time:

```
for y_out in range(H_out):
    kernel_row(..., y_out) → acc_row[W_out * C_out]  (INT32, in SRAM)
    requantize_i32_to_i8(acc_row, &out[y_out * W_out * C_out], ...)
    relu_i8(&out[y_out * W_out * C_out], ...)
```

Accumulator size: `W_out * C_out * 4` bytes. Worst case at alpha=1.0, 96x96 input: `48 * 64 * 4 = 12KB`.

### 4. Kernel API changes

Current signatures compute full output. New signatures add row-range parameters:

```c
// Before
void ternary_conv2d_ref(
    const int8_t *input, const uint8_t *weights, int32_t *output,
    float scale_pos, float scale_neg,
    int H, int W, int C_in, int C_out, int K, int stride, int padding);

// After — adds y_out_start, y_out_count
void ternary_conv2d_ref(
    const int8_t *input, const uint8_t *weights, int32_t *output,
    float scale_pos, float scale_neg,
    int H, int W, int C_in, int C_out, int K, int stride, int padding,
    int y_out_start, int y_out_count);
```

Same change for: `ternary_conv2d_simd`, `int8_conv2d`, `int8_depthwise_conv2d`.

Kernels that already produce small outputs need no row-range support:
- `ternary_dense_ref` / `ternary_dense_simd` / `int8_dense` — output is `[N_out]`, no spatial dims
- `global_avg_pool` — output is `[C]`

Backward compatibility: passing `y_out_start=0, y_out_count=H_out` produces identical output to the current API (useful for testing).

### 5. Kernel inner loop changes

The spatial iteration loops change from:

```c
for (int y_out = 0; y_out < H_out; y_out++)
    for (int x_out = 0; x_out < W_out; x_out++)
        // MAC over kernel window
```

to:

```c
for (int y_out = y_out_start; y_out < y_out_start + y_out_count; y_out++)
    for (int x_out = 0; x_out < W_out; x_out++)
        // MAC over kernel window (identical)
```

The inner MAC loop, SIMD assembly, weight unpacking — all unchanged. Only the outer y-loop bounds change.

Output pointer offset: the kernel writes to `output[(y_out - y_out_start) * W_out * C_out + ...]` so the output buffer starts at index 0 regardless of y_out_start.

### 6. Test strategy

The existing `test_kernels.c` (SIMD vs C reference) remains the correctness oracle. Extended with:

1. **Row-range equivalence**: For each kernel, verify that calling with `y_out_start=0, y_out_count=H_out` produces bit-identical output to the full-tensor reference.
2. **Per-row equivalence**: Verify that calling row-by-row (y_out_count=1 for each row) and concatenating produces bit-identical output to a single full call.
3. **PSRAM buffer test**: Run inference with buf_psram in SPIRAM, compare output class to SRAM-only baseline across all 20 test images.

### 7. Memory budget (alpha=1.0, 96x96 input)

| Buffer | Location | Size |
|--------|----------|------|
| buf_sram (INT8 activation) | SRAM | 48 * 48 * 64 = 144 KB |
| acc_row (INT32, one row) | SRAM | 48 * 64 * 4 = 12 KB |
| Stack, RTOS, DMA descriptors | SRAM | ~50 KB |
| **Total SRAM** | | **~206 KB** |
| buf_psram (INT8 activation) | PSRAM | 144 KB |
| Camera DMA buffers | PSRAM | ~400 KB |
| **Total PSRAM** | | **~544 KB of 8 MB** |

Fits comfortably. Leaves ~110KB SRAM headroom for future use.

### 8. Latency estimate

PSRAM octal-SPI: ~100 MB/s sequential read. MobileNetV1 has 13 conv layers; ~7 read from PSRAM. Largest PSRAM read per layer: 147KB = ~1.5ms. Total PSRAM overhead: ~10ms. Current alpha=0.25 inference: ~50ms. Expected alpha=1.0: ~200ms compute + ~10ms PSRAM = ~210ms per frame. Acceptable for "some latency is fine."

## Files changed

| File | Change |
|------|--------|
| `firmware/main/inference.c` | Buffer allocation, per-row loop, PSRAM ping-pong |
| `firmware/main/inference.h` | No struct changes. Add `inference_deinit()` for cleanup. |
| `firmware/components/ternary_kernels/include/ternary_ops.h` | Add y_out_start/y_out_count params to conv2d and depthwise signatures |
| `firmware/components/ternary_kernels/ternary_conv2d_ref.c` | Row-range outer loop |
| `firmware/components/ternary_kernels/ternary_conv2d_simd.c` | Row-range outer loop |
| `firmware/components/ternary_kernels/int8_conv2d.c` | Row-range outer loop |
| `firmware/components/ternary_kernels/int8_depthwise_conv2d.c` | Row-range outer loop |
| `firmware/components/ternary_kernels/test_kernels.c` | Row-range equivalence tests |
| `model/export_packed.py` | Emit `max_activation_size` constant in model_data.h (convenience, not required) |

## Not changed

- `model/train_baseline.py`, `model/quantize.py`, `model/export_packed.py` — model training and export pipeline unchanged
- `autoresearch/prepare.py`, `autoresearch/train.py` — autoresearch loop unchanged
- Weight packing format, `layer_config_t` struct, UART protocol — unchanged
- SIMD inner loops (MAC assembly) — unchanged
