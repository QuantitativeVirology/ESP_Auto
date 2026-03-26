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
SCORE = 1000 / latency_ms   (i.e., FPS * 1000)
Gated by: accuracy ≥ 0.92 AND sram ≤ 300KB
If gate fails: SCORE = accuracy * 0.1  (drives optimization toward gate)

## Strategy hints
- Wider + shallower models often beat narrow + deep on MCU (memory-bound)
- Ternary layers compress 8x, freeing SRAM for wider channels
- Depthwise convolutions are activation-bound; pointwise are weight-bound
- Ternary benefits most on pointwise layers (many parameters, few activations)
- Keep depthwise layers at INT8 (few parameters, high sensitivity to quantization)
