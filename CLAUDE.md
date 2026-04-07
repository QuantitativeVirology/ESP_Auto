# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Autonomous optimization loop (Karpathy's autoresearch pattern) for ternary-weight cat/dog classification on ESP32-S3-WROOM-CAM. MobileNetV1 α=0.25, 96×96 input, Trained Ternary Quantization (TTQ) for pointwise layers, custom Xtensa SIMD assembly kernels.

## Hard Constraints

- Model + tensor arena <= 300 KB SRAM (camera DMA goes in PSRAM, model tensors in SRAM only)
- Model flash <= 512 KB
- First conv and classifier: INT8. Depthwise layers: INT8. Pointwise layers: ternary.
- All SIMD buffers and weight arrays must be 16-byte aligned (`__attribute__((aligned(16)))`)
- Weight packing: 2 bits per ternary weight, 64 weights per 128-bit block (bytes [0:7] = zero_mask, bytes [8:15] = sign_bits)
- Weight layout is NHWC `[C_out, K, K, C_in]` with C_in padded to multiples of 64

## ESP-IDF Setup

```bash
source /Users/bosselab/Projects/esp-idf/export.sh
```

**Local paths:** Project lives at `/Users/bosselab/Projects/ESP_Auto` (off iCloud). Datasets at `/tmp/esp_datasets` (symlinked from `model/datasets`). Venv at `/tmp/esp_auto_venv3`.

## Build & Flash

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbserial-* flash monitor
```

## Python Environment

```bash
source /tmp/esp_auto_venv3/bin/activate
python model/train_baseline.py --export-test-images  # train INT8 baseline + generate test_images.h
python model/quantize.py                              # TTQ quantization (loads baseline weights)
python model/export_packed.py                         # export model_data.h for firmware
python harness/flash_and_measure.py --port /dev/cu.usbserial-*  # build + flash + measure score
```

## Architecture: 4-Stage Pipeline

**Training -> Quantization -> Export -> Firmware+Measurement**

1. `model/train_baseline.py` trains MobileNetV1 on Oxford-IIIT Pets (binary cat/dog, 96x96). Saves `checkpoints/best_model.pt`. Can also generate `firmware/main/test_images.h` (20 embedded images for on-device benchmarking).

2. `model/quantize.py` loads the baseline, applies TTQ wrappers (`TernaryQuantWrapper`) to configured layers, inserts `FakeQuantize` modules after each ReLU for QAT, runs 3-phase training (warmup -> TTQ+QAT -> frozen fine-tune). Per-layer learned scales (`scale_pos`, `scale_neg`) are separate `nn.Parameter`s trained with 0.1x base LR.

3. `model/export_packed.py` extracts all layers, packs ternary weights into 2-bit format, transposes NCHW->NHWC, folds BatchNorm, computes analytical requantization from float activation ranges (per-tensor output scaling with s_in chain), and generates `firmware/main/model_data.h`. Includes simulated INT8 verification pipeline.

4. Firmware runs layer-by-layer inference with ping-pong INT8 activation buffers (SRAM + PSRAM). Reports `METRIC latency_us=X accuracy=X sram_free=X` over UART. The harness parses these and computes `SCORE = 1000 / latency_ms` (gated by accuracy >= 0.90 and SRAM budget).

## Quantization & Requantization Math

### FakeQuantize (QAT)

Uses **per-tensor** fake INT8 quantization after each ReLU: `absmax = x.abs().max()`, then `scale = absmax / 127`, quantize-dequantize with STE backward. Per-tensor matches the firmware's per-tensor activation scaling. Tracks `running_absmax` (scalar) for export.

### Export Requant Formulas

The export pipeline chains per-layer float scales:
- `s_in` = float value per int8 unit of the input (starts at 2.64/127 for ImageNet-normalized input)
- `s_out` = max(float_activation_range_per_channel) / 127 (per-tensor output scale)

**INT8 layers (BN folded into weights):**
- `acc_scale[c] = s_in * w_scale_per_ch[c]` (float per accumulator unit)
- `bias_i32[c] = round(bn_folded_bias[c] / acc_scale[c])`
- `requant[c] = acc_scale[c] / s_out`

**Ternary layers (BN NOT folded, in requant+bias instead):**
- `acc_scale[c] = bn_s[c] * s_in * avg(scale_pos, scale_neg)`
- `bias_i32[c] = round(bn_folded_bias[c] / acc_scale[c])`
- `requant[c] = |acc_scale[c]| / s_out`

After analytical init, multi-pass calibration refines requant to `127 / max_abs(acc+bias)` per channel using simulated int8 inference on 64 calibration images.

### Firmware Requantization

```c
// Per-channel: out[c] = clamp(round((acc[c] + bias[c]) * requant_per_ch[c]) + zp, -128, 127)
// Rounding: symmetric (fval >= 0 ? floor(fval+0.5) : ceil(fval-0.5))
```

## Cross-File Dependencies

- `quantize.py` imports `MobileNetV1` and `get_loaders` from `train_baseline.py`
- `export_packed.py` imports from both `train_baseline.py` and `quantize.py` (needs `TernaryQuantWrapper` to detect ternary layers)
- `autoresearch/train.py` imports from `model/train_baseline.py` (via sys.path manipulation)
- `autoresearch/prepare.py` dynamically loads `train.py` and imports from `model/quantize.py` and `model/export_packed.py`
- Generated `model_data.h` bridges Python export -> C firmware (contains `layer_config_t` array referencing weight arrays)
- `inference.c` dispatches to kernels in `ternary_kernels/` component based on `layer_config_t.quant` field
- Saved checkpoints include FakeQuantize `running_absmax` buffers; loading into models without `apply_fake_quantize()` requires `strict=False`
- Cached weights in `model/.cache/<arch_hash>/` keyed by hash of `build_model()` + `get_quant_config()` source

## Autoresearch Loop

`autoresearch/train.py` is the **mutable target** (LLM-editable). It exposes:
- `build_model()` -> `nn.Module`
- `get_quant_config()` -> `dict` mapping layer names to `"ternary"` or `"int8"`
- `get_hparams()` -> `dict` with lr, epochs, threshold_ratio, quant_config, etc.

`autoresearch/prepare.py` is **immutable**. It hashes `build_model()` source to cache trained weights per architecture variant. On cache hit, skips training and only rebuilds firmware.

Search space (`autoresearch/program.md`): width [0.15-0.50], depth [3-8], kernel sizes [1,3,5], input resolution [64,80,96]. Wider+shallower models generally outperform on MCU.

## Ternary Kernel Design

SIMD assembly stubs currently fall through to C reference implementations. The intended SIMD strategy:
- Expand 2-bit ternary weights to INT8 {-1, 0, +1} in a q-register
- Use `EE.VMULAS.S8.ACCX` (16-wide INT8 MAC) -- avoids popcount bottleneck
- Register budget: q0 (packed weights), q1 (activations), q2 (expanded weights), q3 (scratch), ACCX (accumulator)
- C reference in `ternary_conv2d_ref.c` / `ternary_dense_ref.c` is the correctness oracle -- SIMD must match bit-exact
- `test_kernels.c` runs random-input comparisons (SIMD vs C ref) triggered by UART command `RUN_TESTS`

## GPIO Pins

- PIN_DOG = GPIO 45, PIN_CAT = GPIO 46, PIN_FRAME = GPIO 47 (timing)
- Camera uses GPIOs 4-18, 43, 44 -- do not conflict

## Firmware Modes

- `RUN_BENCHMARK` -- iterate 20 embedded test images, report aggregate METRIC
- `RUN_TESTS` -- run kernel correctness tests (SIMD vs C reference)
- `MEMORY` -- print SRAM/PSRAM usage map
- Default: live camera inference on Core 1
