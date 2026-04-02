# Close INT8 Export Accuracy Gap

**Date:** 2026-04-02
**Status:** Approved

## Problem

The float model with TTQ (Trained Ternary Quantization) achieves 81% validation accuracy, but the simulated INT8 export pipeline produces 50% accuracy (random chance for binary classification). The 31% gap makes the exported model useless on-device.

### Root Cause

Per-tensor activation quantization between layers destroys channel-level information. When layer N outputs INT8 with per-channel requant, each channel maps to a different float range. Layer N+1 treats all input channels as having the same float-per-INT8 scale (per-tensor `s_in`). Channels with small activation ranges get few effective INT8 quantization levels.

Measured dynamic range ratios (weakest channel / strongest channel):
- Depthwise layers: 0.06-0.18 (worst channels get ~3-4 effective bits)
- Pointwise layers: 0.20-0.37 (better but still lossy)
- By layer 12: 6 completely dead channels (ratio = 0.0)

This compounds multiplicatively through 27 layers.

Ternary weights ({-1, 0, +1}) amplify the problem because they have only ~1.6 bits of precision per weight, producing high per-channel variance in output activations. INT8 weights (7 bits) produce much more uniform outputs.

### Prior Fixes Already Applied

During this investigation, several bugs were fixed:
1. `get_default_quant_config` now keeps depthwise layers as INT8 (were incorrectly ternarized)
2. `train_ternary` resets `best_acc` when TTQ activates (was saving warmup-phase checkpoints with untrained ternary weights)
3. `FakeQuantize` now uses per-tensor scaling (was per-channel, mismatching firmware)
4. Export uses analytical requant with `s_in` chain (calibration was breaking bias/requant coupling)
5. Export handles ternary depthwise weights correctly in verification/calibration

## Design

### Approach B-light: Reduce Ternary Layer Count (Primary)

Keep early pointwise layers (pw0-pw4, in_channels < 128) as INT8. Only ternarize the large pointwise layers (pw5-pw12, in_channels >= 128) where ternary saves the most memory.

**Rationale:** The early pointwise layers form the feature extraction backbone. They have few parameters (7.8K total) so INT8 costs negligible extra flash (+6 KB). But INT8 weights give 40x more weight precision than ternary, dramatically reducing per-channel output variance in the critical early layers.

**Weight budget:**

| Config | Total weights | Flash delta |
|--------|--------------|-------------|
| All PW ternary (current) | ~81 KB | baseline |
| B-light (pw0-4 INT8, pw5-12 ternary) | ~87 KB | +6 KB |

Both well under the 512 KB flash limit.

**Change:** One modification to `get_default_quant_config` in `model/quantize.py`:

```python
def get_default_quant_config(model):
    config = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            if name == "first_conv.0":
                config[name] = "int8"
            elif mod.groups == mod.in_channels and mod.groups > 1:
                config[name] = "int8"  # depthwise
            elif mod.in_channels >= 128:
                config[name] = "ternary"  # large pointwise only
            else:
                config[name] = "int8"  # small pointwise
        elif isinstance(mod, nn.Linear):
            config[name] = "int8"
    return config
```

Layers affected:

| Layer | Shape | Old | New |
|-------|-------|-----|-----|
| pw0 | 16x8 | ternary | **INT8** |
| pw1 | 32x16 | ternary | **INT8** |
| pw2 | 32x32 | ternary | **INT8** |
| pw3 | 64x32 | ternary | **INT8** |
| pw4 | 64x64 | ternary | **INT8** |
| pw5-pw12 | 128+ channels | ternary | ternary (unchanged) |

### Approach A: Learned Activation Clamps — PACT (Conditional)

Only applied if B-light INT8 verification accuracy < 70%.

Replace `ReLU` with `PACTReLU(init_alpha)` during QAT. The learned `alpha` parameter bounds the maximum activation per layer, reducing per-tensor quantization waste.

```python
class PACTReLU(nn.Module):
    def __init__(self, init_alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        return torch.clamp(x, 0, self.alpha.abs())
```

Inserted by `apply_fake_quantize` replacing ReLU + FakeQuantize with PACTReLU + FakeQuantize. Alpha params use the scale_params optimizer group (0.1x base LR, no weight decay).

## Files Changed

| File | Change |
|------|--------|
| `model/quantize.py` | `get_default_quant_config`: ternary cutoff at `in_channels >= 128` |
| `model/quantize.py` | (conditional) `PACTReLU` class + insertion in `apply_fake_quantize` |
| No firmware changes | Export already handles mixed INT8/ternary correctly |

## Validation

### B-light success criteria
1. Float val accuracy with TTQ >= 78%
2. INT8 simulated verification accuracy >= 70%
3. If INT8 >= 75%: ship. If 70-75%: acceptable. If < 70%: escalate to PACT.

### PACT success criteria (if triggered)
1. INT8 simulated verification accuracy >= 75%

## Escalation Path

If B-light + PACT still < 70%:
- B-heavy: extend INT8 to pw0-pw6 (+24 KB flash)
- Firmware per-channel rescale between DW-PW pairs
- Reduce model depth (fewer blocks)
