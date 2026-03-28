# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Autonomous optimization loop (Karpathy's autoresearch pattern) for ternary-weight cat/dog classification on ESP32-S3-WROOM-CAM. MobileNetV1 α=0.25, 96×96 input, Trained Ternary Quantization (TTQ) for middle layers, custom Xtensa SIMD assembly kernels.

## Hard Constraints

- Model + tensor arena ≤ 300 KB SRAM (camera DMA goes in PSRAM, model tensors in SRAM only)
- Model flash ≤ 512 KB
- First conv and classifier: INT8. Depthwise layers: INT8. Pointwise layers: ternary.
- All SIMD buffers and weight arrays must be 16-byte aligned (`__attribute__((aligned(16)))`)
- Weight packing: 2 bits per ternary weight, 64 weights per 128-bit block (bytes [0:7] = zero_mask, bytes [8:15] = sign_bits)
- Weight layout is NHWC `[C_out, K, K, C_in]` with C_in padded to multiples of 64

## ESP-IDF Setup

```bash
source /Users/jensbosse/Documents/Cowork/ESP_webclock_temp_humidity/esp-idf/export.sh
```

## Build & Flash

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbserial-* flash monitor
```

## Python Environment

```bash
source .venv/bin/activate
python model/train_baseline.py --export-test-images  # train INT8 baseline + generate test_images.h
python model/quantize.py                              # TTQ quantization (loads baseline weights)
python model/export_packed.py                         # export model_data.h for firmware
python harness/flash_and_measure.py --port /dev/cu.usbserial-*  # build + flash + measure score
```

## Architecture: 4-Stage Pipeline

**Training → Quantization → Export → Firmware+Measurement**

1. `model/train_baseline.py` trains MobileNetV1 α=0.25 on Oxford-IIIT Pets (binary cat/dog, 96×96). Saves `checkpoints/best_model.pt`. Can also generate `firmware/main/test_images.h` (20 embedded images for on-device benchmarking).

2. `model/quantize.py` loads the baseline, applies TTQ wrappers (`TernaryQuantWrapper`) to configured layers, runs 3-phase training (warmup → TTQ with STE → frozen fine-tune). The `TernarizeFunction` autograd function handles forward ternary quantization and STE backward pass. Per-layer learned scales (`scale_pos`, `scale_neg`) are separate `nn.Parameter`s trained with 0.1× base LR.

3. `model/export_packed.py` extracts all layers from the quantized model, packs ternary weights into 2-bit format, transposes NCHW→NHWC, folds BatchNorm into bias, computes requantization parameters, and generates `firmware/main/model_data.h`. Round-trip validation unpacks and verifies bit-exactness.

4. Firmware runs layer-by-layer inference with ping-pong INT8 activation buffers. Reports `METRIC latency_us=X accuracy=X sram_free=X` over UART. The harness parses these and computes `SCORE = 1000 / latency_ms` (gated by accuracy ≥ 0.90 and SRAM budget).

## Cross-File Dependencies

- `quantize.py` imports `MobileNetV1` and `get_loaders` from `train_baseline.py`
- `export_packed.py` imports from both `train_baseline.py` and `quantize.py` (needs `TernaryQuantWrapper` to detect ternary layers)
- `autoresearch/train.py` imports from `model/train_baseline.py` (via sys.path manipulation)
- `autoresearch/prepare.py` dynamically loads `train.py` and imports from `model/quantize.py` and `model/export_packed.py`
- Generated `model_data.h` bridges Python export → C firmware (contains `layer_config_t` array referencing weight arrays)
- `inference.c` dispatches to kernels in `ternary_kernels/` component based on `layer_config_t.quant` field

## Autoresearch Loop

`autoresearch/train.py` is the **mutable target** (LLM-editable). It exposes:
- `build_model()` → `nn.Module`
- `get_quant_config()` → `dict` mapping layer names to `"ternary"` or `"int8"`
- `get_hparams()` → `dict` with lr, epochs, threshold_ratio, etc.

`autoresearch/prepare.py` is **immutable**. It hashes `build_model()` source to cache trained weights per architecture variant. On cache hit, skips training and only rebuilds firmware.

## Ternary Kernel Design

SIMD assembly stubs currently fall through to C reference implementations. The intended SIMD strategy:
- Expand 2-bit ternary weights to INT8 {-1, 0, +1} in a q-register
- Use `EE.VMULAS.S8.ACCX` (16-wide INT8 MAC) — avoids popcount bottleneck
- Register budget: q0 (packed weights), q1 (activations), q2 (expanded weights), q3 (scratch), ACCX (accumulator)
- C reference in `ternary_conv2d_ref.c` / `ternary_dense_ref.c` is the correctness oracle — SIMD must match bit-exact
- `test_kernels.c` runs random-input comparisons (SIMD vs C ref) triggered by UART command `RUN_TESTS`

## GPIO Pins

- PIN_DOG = GPIO 45, PIN_CAT = GPIO 46, PIN_FRAME = GPIO 47 (timing)
- Camera uses GPIOs 4-18, 43, 44 — do not conflict

## Firmware Modes

- `RUN_BENCHMARK` — iterate 20 embedded test images, report aggregate METRIC
- `RUN_TESTS` — run kernel correctness tests (SIMD vs C reference)
- `MEMORY` — print SRAM/PSRAM usage map
- Default: live camera inference on Core 1
