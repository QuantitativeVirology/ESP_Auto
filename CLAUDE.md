# ESP32 Ternary Vision Inference

## Project

Autonomous optimization loop for ternary-weight cat/dog classifier on ESP32-S3-WROOM-CAM.
MobileNetV1 α=0.25, 96×96 input, ternary middle layers, custom Xtensa SIMD kernels.

## Constraints

- Model + tensor arena ≤ 300 KB SRAM
- Model flash ≤ 512 KB
- First/last layers INT8, middle layers ternary
- Camera DMA in PSRAM, model tensors in SRAM only

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
python model/train_baseline.py      # train INT8 baseline
python model/quantize.py            # TTQ quantization
python model/export_packed.py       # export C header
python harness/flash_and_measure.py # build + flash + measure
```

## Directory Layout

- `model/` — PyTorch training, quantization, export
- `firmware/` — ESP-IDF project (main.c, camera.c, inference.c)
- `firmware/components/ternary_kernels/` — SIMD assembly + C reference kernels
- `harness/` — flash-and-measure automation
- `autoresearch/` — Karpathy autoresearch integration
- `docs/` — architecture docs, Xtensa SIMD reference

## GPIO Pins

- PIN_DOG = GPIO 7
- PIN_CAT = GPIO 8
- PIN_FRAME = GPIO 9 (toggles for inference timing)
