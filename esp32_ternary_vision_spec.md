# ESP32 Ternary Vision Inference — Claude Code Implementation Spec

## Project Overview

**Goal**: Build an autonomous optimization loop (based on Karpathy's `autoresearch`) that co-optimizes a ternary-weight vision network and hand-tuned Xtensa SIMD assembly kernels for real-time cat/dog classification on an ESP32-S3-WROOM-CAM.

**Target hardware**: ESP32-S3-WROOM (dual-core Xtensa LX7 @ 240 MHz, 512 KB SRAM, 2–8 MB PSRAM via OSPI, OV2640 camera)

**Target performance**: 93–97% accuracy, 10–30 FPS, model + tensor arena ≤ 300 KB SRAM

**Repository structure**:
```
esp32-ternary-vision/
├── program.md                    # autoresearch direction document (immutable)
├── prepare.py                    # autoresearch harness: compile → flash → measure (immutable)
├── train.py                      # autoresearch mutable target: model def + quantization
├── results.tsv                   # autoresearch experiment log
├── model/
│   ├── train_baseline.py         # PyTorch training script (cat/dog, 96×96)
│   ├── quantize.py               # Ternary quantization (TTQ-style QAT)
│   ├── export_onnx.py            # Export to ONNX for ESP-DL or custom runtime
│   └── datasets/                 # Cat/dog dataset management
├── firmware/
│   ├── CMakeLists.txt            # ESP-IDF project root
│   ├── main/
│   │   ├── main.c                # Entry point: camera init → inference loop → GPIO signal
│   │   ├── inference.c           # Inference engine (calls into kernels)
│   │   ├── inference.h
│   │   ├── model_data.h          # Compiled model weights (generated)
│   │   └── camera.c              # OV2640 capture + downscale to 96×96
│   ├── components/
│   │   ├── ternary_kernels/      # Custom Xtensa SIMD assembly kernels
│   │   │   ├── CMakeLists.txt
│   │   │   ├── include/
│   │   │   │   └── ternary_ops.h # C API for ternary convolution, dense, etc.
│   │   │   ├── src/
│   │   │   │   ├── ternary_conv2d.S    # SIMD ternary convolution
│   │   │   │   ├── ternary_dense.S     # SIMD ternary fully-connected
│   │   │   │   ├── ternary_conv2d_ref.c # C reference (correctness oracle)
│   │   │   │   └── ternary_dense_ref.c
│   │   │   └── test/
│   │   │       └── test_kernels.c      # Correctness tests: SIMD vs C reference
│   │   └── esp_dl_bridge/        # Optional: thin wrapper if using ESP-DL for non-ternary layers
│   └── sdkconfig.defaults        # ESP-IDF config (PSRAM, CPU freq, etc.)
├── harness/
│   ├── flash_and_measure.py      # Cross-compile → flash → read UART metrics
│   ├── pico_timer/               # Raspberry Pi Pico PIO timing firmware
│   │   ├── timer.pio             # PIO program: edge-to-edge cycle counter
│   │   └── main.c                # Pico firmware: report timing over USB serial
│   └── test_images/              # Validation image set (cat/dog, 96×96 raw)
├── autoresearch/
│   ├── program.md                # Research direction for model optimization
│   ├── program_kernels.md        # Research direction for kernel optimization
│   ├── prepare.py                # Immutable harness (wraps harness/flash_and_measure.py)
│   ├── train.py                  # Mutable: model architecture + quantization config
│   └── kernel_target.S           # Mutable: assembly kernel under optimization
└── docs/
    ├── ARCHITECTURE.md           # System design overview
    └── XTENSA_SIMD_REFERENCE.md  # Cheat sheet for PIE instructions
```

---

## Phase 0: Environment and Toolchain Setup

### 0.1 — Host development environment

```bash
# Install ESP-IDF v5.3+ (latest stable with full ESP32-S3 support)
# Follow https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/
mkdir -p ~/esp
cd ~/esp
git clone -b v5.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3
source export.sh

# Verify toolchain
idf.py --version
xtensa-esp32s3-elf-gcc --version
```

### 0.2 — Python environment for model training

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision onnx onnxruntime
pip install esptool pyserial    # for flash_and_measure.py
pip install datasets            # for Hugging Face cat/dog datasets
```

### 0.3 — Clone and study key references

```bash
# Autoresearch framework (study pattern, do not use as submodule)
git clone https://github.com/karpathy/autoresearch.git ../ref/autoresearch

# ESP-DL v3.0 (reference for SIMD kernel patterns)
git clone https://github.com/espressif/esp-dl.git ../ref/esp-dl

# ESP-NN (optimized NN kernels — critical SIMD assembly reference)
git clone https://github.com/espressif/esp-nn.git ../ref/esp-nn

# esp_simd (community SIMD library — additional patterns)
git clone https://github.com/zliu43/esp_simd.git ../ref/esp_simd

# ESP TFLite Micro (person detection example as reference)
git clone https://github.com/espressif/esp-tflite-micro.git ../ref/esp-tflite-micro
```

### 0.4 — ESP-IDF project skeleton

Create the ESP-IDF project in `firmware/`:

```
idf.py create-project esp32_ternary_vision
```

Critical `sdkconfig.defaults`:
```ini
# CPU
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y

# PSRAM (for camera framebuffer; model must NOT use this)
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y

# Camera
CONFIG_CAMERA_TASK_PINNED_TO_CORE_0=y

# Inference pinned to Core 1
CONFIG_FREERTOS_UNICORE=n

# Optimization
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_SYSTEM_MEMPROT_FEATURE=n  # needed for IRAM execution of custom kernels
```

---

## Phase 1: INT8 Baseline Model (Target: 2 weeks)

### 1.1 — Dataset preparation

**Task**: Download and preprocess cat/dog dataset to 96×96 RGB images.

```python
# model/train_baseline.py
# Use Oxford-IIIT Pets or Kaggle Cats vs Dogs
# Output: train/ and val/ directories with 96×96 JPEG images
# Split: 80/20 train/val
# Augmentation: random horizontal flip, random crop (resize to 112 then crop 96),
#               color jitter (brightness=0.2, contrast=0.2)
# Normalize to [0, 1] then quantize-aware: scale to INT8 range [-128, 127]
```

Implementation notes:
- Use `torchvision.datasets.OxfordIIITPets` or download Kaggle cats-vs-dogs
- Binary labels: 0 = cat, 1 = dog
- Store validation set also as raw 96×96×3 uint8 arrays (`.bin` format) for on-device testing
- Generate a C header (`test_images.h`) with 10 cat + 10 dog images embedded as `const uint8_t[]` for firmware-side validation without camera

### 1.2 — Train baseline model in PyTorch

**Architecture**: MobileNetV1 α=0.25 (the proven ESP32-S3 sweet spot)

```python
# model/train_baseline.py
# Key architecture decisions:
# - Input: 96×96×3
# - MobileNetV1 with width multiplier 0.25
# - Final: Global Average Pooling → Dense(64) → Dense(2) with softmax
# - Total params: ~25K–50K (must fit in ~200KB with INT8)
#
# Training:
# - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
# - Scheduler: CosineAnnealingLR, 100 epochs
# - Quantization-aware training (QAT) from epoch 50
#   using torch.quantization with INT8 weights and activations
# - Target: >95% val accuracy before ternarization
#
# Output: model_int8.onnx (quantized ONNX for ESP-DL)
```

### 1.3 — Deploy INT8 model via ESP-DL v3.0

**Task**: Convert ONNX → ESP-DL format → flash → run inference.

Steps:
1. Use ESP-DL's model converter (`esp-ppq` or the built-in ONNX converter) to produce a `.espdl` model file
2. Configure ESP-DL's memory planner with `internal_ram_budget = 300000` (300KB, leaving ~50KB for stack and system)
3. If model doesn't fit, reduce width multiplier or resolution
4. Build and flash:

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### 1.4 — Implement camera capture pipeline

```c
// firmware/main/camera.c
// OV2640 initialization at QVGA (320×240), JPEG output
// Capture frame → decode JPEG (or use RGB565 mode) → bilinear downscale to 96×96
// Convert to INT8 (subtract 128 from each uint8 pixel)
// Store result in a 96×96×3 int8_t buffer in SRAM
//
// CRITICAL: Camera DMA buffers go in PSRAM. Model tensors stay in SRAM.
// Use esp_camera with:
//   .fb_location = CAMERA_FB_IN_PSRAM,
//   .grab_mode = CAMERA_GRAB_LATEST,  // drop old frames
```

### 1.5 — Implement GPIO signaling

```c
// firmware/main/main.c
// GPIO configuration:
#define PIN_DOG   GPIO_NUM_7
#define PIN_CAT   GPIO_NUM_8
#define PIN_FRAME GPIO_NUM_9   // toggles at start of each inference (for external timing)

// Before inference:
gpio_set_level(PIN_FRAME, 1);  // rising edge = inference start
uint32_t cycles_start = esp_cpu_get_cycle_count();

// After inference:
uint32_t cycles_end = esp_cpu_get_cycle_count();
uint32_t latency_us = (cycles_end - cycles_start) / 240;  // 240 MHz → µs

if (result == DOG) gpio_set_level(PIN_DOG, 1);
else               gpio_set_level(PIN_CAT, 1);

gpio_set_level(PIN_FRAME, 0);  // falling edge = inference complete

// Report over UART (for autoresearch harness):
printf("METRIC latency_us=%u accuracy=%.4f sram_bytes=%u\n",
       latency_us, accuracy, heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
```

### 1.6 — Establish baseline metrics

Run 100 inference iterations on the embedded test images. Record:
- Median inference latency (µs)
- Accuracy on 20 embedded test images
- Peak SRAM usage (via `heap_caps_get_minimum_free_size`)
- Model binary size (bytes in flash)

Expected baseline: **~87–93% accuracy, 140–200ms per frame (5–7 FPS), ~200KB SRAM**

---

## Phase 2: Measurement Harness (Target: 1 week)

### 2.1 — UART-based metric reporting protocol

The ESP32-S3 prints structured lines to UART after each inference:

```
METRIC latency_us=142000 accuracy=0.9000 sram_free=102400 flash_size=98304
```

The harness script parses these to extract the optimization scalar.

### 2.2 — `harness/flash_and_measure.py`

```python
# This script is called by autoresearch's prepare.py
# It performs: compile → flash → collect metrics → return scalar
#
# Interface:
#   python flash_and_measure.py --port /dev/ttyUSB0 --timeout 120
#   Reads firmware/ project, runs idf.py build, flashes, waits for METRIC lines
#
# Steps:
# 1. Run: subprocess.run(["idf.py", "build"], cwd="firmware/", check=True)
#    - If build fails → return metric = -1.0 (autoresearch reverts)
# 2. Run: subprocess.run(["idf.py", "-p", port, "flash"], cwd="firmware/", check=True)
#    - If flash fails → return metric = -1.0
# 3. Open serial port, wait for "READY" line (firmware booted)
# 4. Send "RUN_BENCHMARK\n" over serial
# 5. Collect 100 METRIC lines, parse latency_us and accuracy
# 6. Compute composite scalar:
#    if sram_free < 0:  # SRAM budget exceeded
#        score = -1.0
#    elif accuracy < 0.90:  # accuracy gate
#        score = accuracy * 0.1  # heavily penalized
#    else:
#        score = 1.0 / (latency_us / 1000.0)  # maximize FPS, gated by accuracy
# 7. Print: f"SCORE: {score:.6f}"
# 8. Return
```

### 2.3 — Raspberry Pi Pico PIO external timer (optional, for validation)

```c
// harness/pico_timer/timer.pio
// PIO program:
// 1. Wait for rising edge on GPIO input (connected to ESP32 PIN_FRAME)
// 2. Count cycles at 125 MHz until falling edge
// 3. Push count to FIFO
// 4. main.c reads FIFO, converts to µs, prints over USB serial
//
// Resolution: 8 ns per tick
// This validates that CCOUNT-based measurements are accurate
// Not in the critical path for autoresearch; used for spot-checking
```

### 2.4 — Test image injection (without camera)

For deterministic benchmarking, firmware should support two modes:
1. **Camera mode**: Live OV2640 capture (for demo)
2. **Benchmark mode**: Iterate over 20 embedded test images from `test_images.h`

Trigger via UART command: `RUN_BENCHMARK` uses embedded images, `RUN_LIVE` uses camera.

---

## Phase 3: Ternary Quantization Pipeline (Target: 3 weeks)

### 3.1 — Implement Trained Ternary Quantization (TTQ)

```python
# model/quantize.py
#
# Implement TTQ (Li et al., ICLR 2017) adapted for vision CNNs:
#
# During training, each layer has:
# - Full-precision shadow weights W_fp (gradient target)
# - Ternary weights W_t ∈ {-s_n, 0, +s_p} where s_n, s_p are learned per-layer scales
# - Threshold t (typically 0.05 * max(|W_fp|)) determines zero vs non-zero
#
# Ternarization function:
#   W_t[i] = +s_p  if W_fp[i] > t
#   W_t[i] = 0     if |W_fp[i]| <= t
#   W_t[i] = -s_n  if W_fp[i] < -t
#
# Forward pass uses W_t (ternary), backward pass updates W_fp (full-precision)
# s_p and s_n are learned via gradient descent (STE for ternary function)
#
# CRITICAL: Keep first conv layer and final dense layer at INT8.
# Only ternarize intermediate depthwise and pointwise conv layers.
# This is the proven "mixed-precision" sweet spot.
#
# Training schedule:
# - Epochs 1–30: Full-precision warmup
# - Epochs 31–100: Enable TTQ with STE, anneal threshold
# - Epochs 101–120: Fine-tune with frozen ternary assignments (optional)
#
# Output: ternary_model.pt containing:
#   - Per-layer ternary weight masks: sign_mask (1 bit), zero_mask (1 bit)
#   - Per-layer scales: s_p, s_n (float16, applied during inference as INT8 post-scale)
#   - First/last layer weights in INT8
```

### 3.2 — Weight packing for ESP32-S3

```python
# model/export_packed.py
#
# Pack ternary weights for SIMD consumption:
# Each ternary weight needs 2 bits: 1 bit for sign, 1 bit for zero-mask
# Pack into bytes: 4 weights per byte, 64 weights per 128-bit SIMD register
#
# Packing format per 128-bit register:
#   Bits [0:63]  = zero_mask (1 = non-zero, 0 = zero) for 64 weights
#   Bits [64:127] = sign_bits (0 = positive, 1 = negative) for 64 weights
#
# This allows the SIMD kernel to:
#   1. Load 128-bit packed weights
#   2. Load 64 INT8 activations (two 128-bit loads, or process in two halves)
#   3. Use AND (zero_mask) + XOR (sign_bits) to compute conditional add/subtract
#
# Export as C header:
#   const uint8_t layer1_weights[] ALIGNED(16) = { ... };
#   const float layer1_scale_pos = 0.342f;
#   const float layer1_scale_neg = 0.298f;
#   const int8_t layer0_weights_int8[] = { ... };  // first layer, INT8
#
# Also export layer metadata struct:
#   typedef struct {
#       const uint8_t *packed_weights;  // ternary packed
#       float scale_pos, scale_neg;
#       int in_channels, out_channels, kernel_h, kernel_w;
#       int stride, padding;
#       bool is_ternary;  // false for first/last layer
#   } layer_config_t;
```

### 3.3 — Validate ternary model accuracy

```python
# Run full validation set through ternary model in PyTorch
# Compare against INT8 baseline
# Target: <3% accuracy drop (e.g., 95% INT8 → 92%+ ternary)
# If accuracy is too low:
#   - Increase width multiplier (0.25 → 0.35)
#   - Add knowledge distillation from INT8 teacher
#   - Keep more layers at INT8 (only ternarize the deepest layers)
```

---

## Phase 4: Custom Ternary SIMD Kernels (Target: 4–6 weeks)

### 4.1 — Xtensa PIE SIMD cheat sheet

Document in `docs/XTENSA_SIMD_REFERENCE.md`:

```
Registers:
  q0–q7: 128-bit SIMD vector registers
  ACCX, ACCY: accumulator registers (used by some MAC instructions)
  SAR: shift amount register

Key instructions for ternary inference:
  EE.VLD.128.IP    qX, aY, imm   # Load 128 bits from memory, post-increment
  EE.VST.128.IP    qX, aY, imm   # Store 128 bits to memory, post-increment
  EE.ANDQ          qZ, qX, qY    # 128-bit bitwise AND
  EE.ORQ           qZ, qX, qY    # 128-bit bitwise OR
  EE.XORQ          qZ, qX, qY    # 128-bit bitwise XOR
  EE.NOTQ          qZ, qX        # 128-bit bitwise NOT
  EE.MOVI.32.A     qX, aY, sel   # Move 32-bit lane from q-reg to scalar
  EE.MOVI.32.Q     qX, aY, sel   # Move scalar 32-bit into q-reg lane

  # INT8 SIMD (for first/last layer, INT8 path):
  EE.VMULAS.S8.ACCX        qX, qY        # 16×INT8 multiply-accumulate
  EE.VMULAS.S8.ACCX.LD.IP  qZ, aW, imm, qX, qY  # MAC + preload next vector

Memory alignment: ALL vector loads/stores require 16-byte alignment.
No unaligned access exists. Use __attribute__((aligned(16))) on all buffers.

There is NO hardware popcount. Must implement in software:
  // Software popcount for 32-bit word:
  // Option A: Shift-and-mask (Hamming weight, ~12 instructions per word)
  // Option B: 256-byte lookup table (2 loads per word, ~6 instructions)
  // For 128-bit register: extract 4 × 32-bit lanes, popcount each, sum
```

### 4.2 — C reference implementation (correctness oracle)

```c
// firmware/components/ternary_kernels/src/ternary_conv2d_ref.c
//
// void ternary_conv2d_ref(
//     const int8_t *input,       // [H, W, C_in] INT8 activations
//     const uint8_t *weights,    // packed ternary: [C_out, K, K, C_in/4] (2 bits per weight)
//     int32_t *output,           // [H_out, W_out, C_out] INT32 accumulators
//     float scale_pos,
//     float scale_neg,
//     int H, int W, int C_in,
//     int C_out, int K,          // kernel size (assume square)
//     int stride, int padding
// );
//
// For each output position (oh, ow, oc):
//   acc = 0
//   For each (kh, kw, ic):
//     weight_idx = oc * K * K * C_in + kh * K * C_in + kw * C_in + ic
//     byte_idx = weight_idx / 4
//     bit_pos = (weight_idx % 4) * 2
//     zero_bit = (weights[byte_idx] >> bit_pos) & 1
//     sign_bit = (weights[byte_idx] >> (bit_pos + 1)) & 1
//     if (zero_bit):
//       if (sign_bit):
//         acc -= input[ih * W * C_in + iw * C_in + ic]
//       else:
//         acc += input[ih * W * C_in + iw * C_in + ic]
//   // Post-processing: scale by s_p/s_n, add bias, ReLU, requantize to INT8
//   output[...] = acc
```

### 4.3 — SIMD ternary convolution kernel

```asm
# firmware/components/ternary_kernels/src/ternary_conv2d.S
#
# High-level approach:
# Process 64 input channels at a time (one 128-bit packed weight register = 64 ternary weights)
#
# For each output position:
#   1. Load 128-bit packed weights into q0:
#      - q0[0:63] = zero_mask, q0[64:127] = sign_mask
#   2. Load 64 INT8 activations into q1 (first 16) and q2 (next 16) etc.
#      - Actually: process 16 activations at a time (128 bits = 16 × INT8)
#   3. For each group of 16 activations:
#      a. Extract corresponding 16 zero-mask bits and 16 sign bits
#      b. For non-zero weights: conditionally add or subtract activation
#      c. Accumulate into scalar register
#
# ALTERNATIVE APPROACH (likely faster):
# Treat ternary as two binary masks: positive_mask and negative_mask
# positive_mask = zero_mask AND (NOT sign_mask)
# negative_mask = zero_mask AND sign_mask
#
# For 16 INT8 activations in q1:
#   - Multiply by positive_mask (select only positive-weight activations) → sum
#   - Multiply by negative_mask (select only negative-weight activations) → sum
#   - acc += pos_sum - neg_sum
#
# This avoids popcount entirely if we use EE.VMULAS.S8.ACCX with constructed
# +1/-1/0 weight vectors. The trick:
#   - Expand 2-bit ternary weights back to INT8 {-1, 0, +1} in a q-register
#   - Use standard INT8 MAC instruction
#   - This uses more registers but avoids the popcount bottleneck
#
# REGISTER BUDGET per inner loop iteration (16 activations):
#   q0: packed ternary weights (source)
#   q1: INT8 activations (16 values)
#   q2: expanded INT8 weights {-1, 0, +1} (constructed from q0)
#   q3: scratch
#   ACCX: running accumulator
#
# Estimated cycles per 16-activation MAC: ~8–12 cycles
# vs INT8 MAC: ~4–6 cycles
# But model is 8–16x smaller, so fewer total iterations and no PSRAM stalls
```

### 4.4 — Ternary dense (fully-connected) kernel

Same approach as conv2d but simpler (no spatial dimensions):

```asm
# firmware/components/ternary_kernels/src/ternary_dense.S
# Input: [N_in] INT8 activations, [N_out, N_in] packed ternary weights
# Output: [N_out] INT32 accumulators
# Inner loop: process 16 inputs at a time using VMULAS with expanded ternary weights
```

### 4.5 — Correctness test suite

```c
// firmware/components/ternary_kernels/test/test_kernels.c
//
// For each kernel (conv2d, dense):
// 1. Generate random INT8 input
// 2. Generate random packed ternary weights
// 3. Run C reference implementation → expected output
// 4. Run SIMD kernel → actual output
// 5. Assert: max(|expected - actual|) == 0 (exact match, no floating point)
//
// Test cases:
// - C_in = 16, 32, 64, 128 (boundary cases for SIMD register packing)
// - K = 1, 3, 5
// - stride = 1, 2
// - padding = 0, same
// - All-zero weights, all-positive, all-negative, mixed
//
// Run on device via UART command: "RUN_TESTS"
// Output: "TESTS PASSED" or "TEST FAILED at ..."
```

---

## Phase 5: Inference Engine Integration (Target: 2 weeks)

### 5.1 — Layer-by-layer inference engine

```c
// firmware/main/inference.c
//
// typedef struct {
//     layer_type_t type;        // CONV2D, DEPTHWISE_CONV2D, DENSE, RELU, POOL, BATCHNORM
//     quant_mode_t quant;       // INT8 or TERNARY
//     const void *weights;      // pointer to weight data (INT8 or packed ternary)
//     const int32_t *bias;
//     float scale_pos, scale_neg;  // ternary per-layer scales
//     float requant_scale;      // output requantization scale (to go back to INT8 activations)
//     int8_t requant_zero_point;
//     // Geometry:
//     int in_c, out_c, kernel, stride, padding;
// } layer_t;
//
// static const layer_t model_layers[] = {
//     // Generated by export_packed.py
//     { .type = CONV2D, .quant = INT8, ... },       // first layer: INT8
//     { .type = DEPTHWISE_CONV2D, .quant = TERNARY, ... },
//     { .type = CONV2D, .quant = TERNARY, ... },      // pointwise
//     // ...
//     { .type = DENSE, .quant = INT8, ... },           // final layer: INT8
// };
//
// int classify_image(const int8_t *input_96x96x3) {
//     // Allocate two SRAM buffers (ping-pong):
//     static int8_t buf_a[MAX_ACTIVATION_SIZE] __attribute__((aligned(16)));
//     static int8_t buf_b[MAX_ACTIVATION_SIZE] __attribute__((aligned(16)));
//
//     int8_t *in = buf_a, *out = buf_b;
//     memcpy(in, input_96x96x3, 96*96*3);
//
//     for (int i = 0; i < NUM_LAYERS; i++) {
//         const layer_t *L = &model_layers[i];
//         switch (L->type) {
//             case CONV2D:
//                 if (L->quant == TERNARY)
//                     ternary_conv2d(in, L->weights, out, ...);
//                 else
//                     int8_conv2d(in, L->weights, out, ...);  // ESP-DL or ESP-NN kernel
//                 break;
//             // ... other layer types
//         }
//         // Requantize INT32 accumulator → INT8 for next layer
//         requantize(out, L->requant_scale, L->requant_zero_point, out_size);
//         // Swap buffers
//         int8_t *tmp = in; in = out; out = tmp;
//     }
//     // in now points to final [2] softmax output
//     return (in[0] > in[1]) ? CLASS_CAT : CLASS_DOG;
// }
```

### 5.2 — Memory budget verification

```c
// At compile time (or early runtime), verify:
// 1. Total weight data (flash): sum of all layer weight sizes
// 2. Peak activation buffer: max over all layers of (input_size + output_size)
//    This determines buf_a and buf_b sizes
// 3. Must satisfy: peak_activation * 2 + weight_data_in_sram < 300KB
//
// Print memory map on boot:
// "MEMORY flash_model=48320 sram_peak_act=73728 sram_total=196576 sram_free=115424"
```

### 5.3 — Dual-core task pinning

```c
// Core 0: WiFi, BLE, camera capture (managed by ESP-IDF)
// Core 1: Inference loop (pinned, maximum priority)
//
// xTaskCreatePinnedToCore(inference_task, "infer", 8192, NULL, configMAX_PRIORITIES - 1, NULL, 1);
//
// inference_task:
//   while (1) {
//       camera_frame_t *frame = camera_get_latest();  // non-blocking, from Core 0 queue
//       if (frame) {
//           preprocess(frame, input_buffer);
//           gpio_set_level(PIN_FRAME, 1);
//           int result = classify_image(input_buffer);
//           gpio_set_level(PIN_FRAME, 0);
//           signal_result(result);
//           camera_return_frame(frame);
//       }
//       vTaskDelay(1);  // yield briefly
//   }
```

---

## Phase 6: Autoresearch Integration (Target: 4–6 weeks)

### 6.1 — `autoresearch/program.md` (immutable research direction)

```markdown
# Research Direction: ESP32-S3 Ternary Cat/Dog Classifier

## Objective
Minimize inference latency (maximize FPS) for binary cat/dog classification
on ESP32-S3-WROOM-CAM while maintaining ≥92% accuracy.

## Constraints (HARD, never violate)
- Total SRAM usage (weights + activations) ≤ 300 KB
- Model flash size ≤ 512 KB
- Input resolution: 96×96×3 INT8
- Output: 2-class softmax
- First and last layers must remain INT8
- Intermediate layers: ternary (preferred) or INT8

## Optimization levers (search space for train.py)
- Width multiplier: [0.15, 0.20, 0.25, 0.30, 0.35, 0.50]
- Depth: number of depthwise-separable blocks [3, 4, 5, 6, 7, 8]
- Kernel sizes per block: [1, 3, 5]
- Skip connections: [none, residual] per block
- Channel expansion ratio: [1, 2, 3, 4, 6]
- Ternary vs INT8 per-layer decision
- Input resolution: [64, 80, 96] (96 preferred, 64 as fallback)
- Knowledge distillation temperature: [1, 3, 5, 10]

## Metric
SCORE = 1000 / latency_ms   (i.e., FPS × 1000)
Gated by: accuracy ≥ 0.92 AND sram ≤ 300KB
If gate fails: SCORE = accuracy × 0.1  (drives optimization toward gate)

## Strategy hints
- Wider + shallower models often beat narrow + deep on MCU (memory-bound)
- Ternary layers compress 8x, freeing SRAM for wider channels
- Depthwise convolutions are activation-bound; pointwise are weight-bound
- Ternary benefits most on pointwise layers (many parameters, few activations)
- Keep depthwise layers at INT8 (few parameters, high sensitivity to quantization)
```

### 6.2 — `autoresearch/prepare.py` (immutable harness)

```python
# autoresearch/prepare.py
#
# This file is NEVER modified by the LLM agent.
# It orchestrates the evaluation pipeline:
#
# 1. Import train.py's model definition
# 2. Train for N epochs (or load cached weights if architecture unchanged)
# 3. Quantize (TTQ for ternary layers, standard INT8 for others)
# 4. Export packed weights to firmware/main/model_data.h
# 5. Run flash_and_measure.py → parse SCORE
# 6. Print final SCORE to stdout
#
# Timeout: 300 seconds (5 minutes, matching autoresearch default)
#
# Caching strategy:
# - Hash the model architecture definition (not weights)
# - If architecture unchanged from last run, skip training, only re-export
# - If architecture changed, retrain from scratch (or from last checkpoint)
#
# Fast-path for kernel-only optimization:
# - If train.py only changed assembly kernels (not model architecture):
#   skip training, rebuild firmware only, re-measure
```

### 6.3 — `autoresearch/train.py` (mutable target)

```python
# autoresearch/train.py
#
# THIS FILE IS EDITED BY THE LLM AGENT.
# It defines the model architecture and quantization configuration.
#
# The LLM agent will modify:
# - The model class definition (layers, channels, kernel sizes)
# - The quantization config (which layers are ternary vs INT8)
# - Hyperparameters (learning rate, epochs, etc.)
#
# Required interface (called by prepare.py):
#   model = build_model()          # returns nn.Module
#   config = get_quant_config()    # returns dict of layer_name → "ternary" | "int8"
#   hparams = get_hparams()        # returns dict with lr, epochs, etc.
```

### 6.4 — Kernel optimization loop (AutoKernel pattern)

A second autoresearch instance optimizes assembly kernels:

```python
# autoresearch/program_kernels.md (second research direction)
#
# Objective: Minimize cycle count of ternary_conv2d.S for fixed model architecture.
#
# Constraints:
# - Output must exactly match ternary_conv2d_ref.c (bit-exact)
# - Must not exceed IRAM budget (32 KB)
# - Must maintain 16-byte alignment on all memory accesses
#
# Optimization levers:
# - Loop unrolling factor
# - Register allocation (q0–q7 assignment)
# - Software pipelining (overlap load/compute)
# - Lookup table vs shift-and-mask for popcount (if used)
# - Weight expansion strategy (on-the-fly vs pre-expanded)
# - Dual-issue instruction scheduling (Xtensa can dual-issue some pairs)
#
# Metric: Cycle count for Conv2d(C_in=32, C_out=64, K=3, H=48, W=48)
# Correctness gate: Must pass test_kernels.c (TESTS PASSED)
```

### 6.5 — Running the optimization

```bash
# Terminal 1: Model architecture search
cd autoresearch/
python ../ref/autoresearch/autoresearch.py \
    --program program.md \
    --prepare prepare.py \
    --target train.py \
    --timeout 300 \
    --runs 50

# Terminal 2 (optional, after model converges): Kernel optimization
python ../ref/autoresearch/autoresearch.py \
    --program program_kernels.md \
    --prepare prepare_kernels.py \
    --target ../firmware/components/ternary_kernels/src/ternary_conv2d.S \
    --timeout 300 \
    --runs 30
```

---

## Phase 7: Wokwi / Simulation (Parallel with Phase 4–6)

### 7.1 — Scope and limitations

Wokwi supports ESP32-S3 but is NOT cycle-accurate. Use it ONLY for:
- Functional correctness of the inference pipeline (does the model produce correct outputs?)
- CI smoke tests (does the firmware compile and boot?)
- GPIO signal protocol validation (do pins toggle correctly?)

Do NOT use Wokwi for:
- Performance benchmarking (timing is unreliable)
- SIMD kernel validation (custom PIE instructions may not be emulated)

### 7.2 — Wokwi project setup

```json
// wokwi.toml (in firmware/ directory)
[wokwi]
version = 1
firmware = "build/esp32_ternary_vision.bin"
elf = "build/esp32_ternary_vision.elf"

// diagram.json
{
  "version": 1,
  "author": "Bosse Lab",
  "editor": "wokwi",
  "parts": [
    { "type": "board-esp32-s3-devkitc-1", "id": "esp32s3" },
    { "type": "wokwi-logic-analyzer", "id": "la1" }
  ],
  "connections": [
    ["esp32s3:7", "la1:D0", "green", []],
    ["esp32s3:8", "la1:D1", "blue", []],
    ["esp32s3:9", "la1:D2", "red", []]
  ]
}
```

### 7.3 — QEMU for GDB debugging

```bash
# Install Espressif QEMU fork
pip install qemu-esp32
# Or build from source: https://github.com/espressif/qemu

# Run with GDB server
idf.py qemu monitor --gdb
# In another terminal:
xtensa-esp32s3-elf-gdb build/esp32_ternary_vision.elf -ex "target remote :1234"
```

---

## Phase 8: Real Hardware Validation and Demo (Target: 1 week)

### 8.1 — Flash and validate

```bash
cd firmware/
idf.py -p /dev/ttyUSB0 flash monitor
# Verify:
# 1. CCOUNT-based latency matches Pico PIO external measurement (within 5%)
# 2. Accuracy on embedded test images matches PyTorch validation
# 3. GPIO signals toggle correctly (verify with logic analyzer or oscilloscope)
# 4. Camera capture works and feeds real images through the pipeline
```

### 8.2 — Live demo mode

```c
// Firmware mode: continuous camera capture + inference + GPIO + UART output
// UART prints human-readable results:
// "CAT  confidence=0.94  latency=68ms  fps=14.7"
// "DOG  confidence=0.87  latency=71ms  fps=14.1"
//
// Optional: stream JPEG + classification result over WiFi for a web dashboard
// (low priority, only if time permits)
```

### 8.3 — Final metrics collection

Run 1000 frames, report:
- Median / P90 / P99 latency
- Accuracy on full validation set (streamed over UART as raw images or pre-loaded from SPIFFS)
- FPS (sustained, including camera capture overhead)
- SRAM usage breakdown (model weights, activation buffers, camera DMA, system)
- Flash usage breakdown
- Power consumption (if measurable)

---

## Key Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base architecture | MobileNetV1 | Best ternarization response; proven on ESP32-S3 |
| Width multiplier | Start 0.25, search 0.15–0.50 | Autoresearch will find optimal |
| Input resolution | 96×96 | TinyML standard; fits SRAM budget |
| Quantization | W2A8 (ternary weights, INT8 activations) | 16× weight compression; INT8 activations use standard SIMD |
| First/last layers | INT8 | Ternary input/output layers lose too much accuracy |
| Inference runtime | Custom C engine + SIMD assembly | ESP-DL/TFLite don't support ternary; need custom kernels |
| Ternary method | TTQ (Trained Ternary Quantization) | Best accuracy retention; learned per-layer scales |
| SIMD strategy | Expand ternary → INT8 {-1,0,+1}, use EE.VMULAS | Avoids popcount; reuses existing INT8 SIMD MAC |
| Memory strategy | All model + activations in SRAM; camera in PSRAM | PSRAM 3–10× slower; this is the main perf lever |
| Measurement | CCOUNT (primary) + Pico PIO (validation) | CCOUNT is 4.17ns resolution; Pico validates externally |
| Optimization loop | Karpathy autoresearch, two instances | Model arch search + kernel optimization, decoupled |
| Simulation | Wokwi for CI, QEMU for debug, real HW for perf | Wokwi is not cycle-accurate; real HW is non-negotiable |

---

## Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Ternary SIMD kernels are too slow (expand-to-INT8 overhead negates compression benefit) | High | Medium | Fallback: pure INT8 model, rely on autoresearch to find smallest INT8 architecture that fits SRAM |
| Autoresearch compile-flash cycle too slow (>60s per experiment) | Medium | Medium | Implement UART-based weight-only update (send new weights over serial without full reflash) |
| Camera + inference don't fit in SRAM together | Medium | Low | Camera outputs to PSRAM, memcpy 96×96 frame to SRAM before inference; 27KB per frame is small |
| TTQ training instability | Medium | Low | Start from strong INT8 checkpoint; use conservative threshold schedule |
| Xtensa PIE instructions behave unexpectedly | Medium | Medium | Test every instruction individually before building kernels; esp-nn source is the ground truth |
| Autoresearch LLM generates invalid assembly | High | High | 5-stage verification: syntax check → assemble → link → correctness test → cycle count. Revert on any failure. |

---

## Deliverables Checklist

- [ ] Repository with full source code and build instructions
- [ ] Trained ternary model (PyTorch checkpoint + exported C header)
- [ ] Custom Xtensa SIMD ternary kernels (conv2d, dense) with correctness tests
- [ ] ESP-IDF firmware for ESP32-S3-WROOM-CAM
- [ ] Autoresearch configuration (program.md, prepare.py, train.py)
- [ ] Measurement harness (flash_and_measure.py + Pico PIO firmware)
- [ ] results.tsv with full experiment history
- [ ] ARCHITECTURE.md documenting the system
- [ ] Performance report: accuracy, latency, memory, FPS
