# Close INT8 Export Accuracy Gap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 31% accuracy gap between float TTQ model (81%) and INT8 export (50%) by reducing ternary layer count, with optional PACT learned clamps if needed.

**Architecture:** Change `get_default_quant_config` so only pointwise layers with `in_channels >= 128` use ternary quantization. Early pointwise layers (pw0-pw4) become INT8, preserving channel-level information through the critical feature extraction stages. Retrain, export, and measure. If INT8 verification < 70%, add PACT learned activation clamps and retrain again.

**Tech Stack:** PyTorch (MPS), custom MobileNetV1, TTQ quantization, numpy INT8 simulation pipeline

---

### Task 1: Update quant config to B-light cutoff

**Files:**
- Modify: `model/quantize.py:263-277`

- [ ] **Step 1: Change the ternary cutoff in `get_default_quant_config`**

In `model/quantize.py`, replace lines 263-277:

```python
def get_default_quant_config(model):
    """Default mixed-precision config: first conv + classifier + depthwise INT8, pointwise ternary."""
    config = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            if name == "first_conv.0":
                config[name] = "int8"
            elif mod.groups == mod.in_channels and mod.groups > 1:
                # Depthwise conv: keep INT8 (only K*K weights per channel, too few for ternary)
                config[name] = "int8"
            else:
                config[name] = "ternary"
        elif isinstance(mod, nn.Linear):
            config[name] = "int8"
    return config
```

With:

```python
def get_default_quant_config(model):
    """Mixed-precision: first conv + classifier + depthwise + small pointwise INT8, large pointwise ternary."""
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
                config[name] = "int8"  # small pointwise: precision > compression
        elif isinstance(mod, nn.Linear):
            config[name] = "int8"
    return config
```

- [ ] **Step 2: Verify the config produces the expected layer assignment**

Run:
```bash
cd model && source /tmp/esp_auto_venv2/bin/activate && python -c "
from train_baseline import MobileNetV1
from quantize import get_default_quant_config
model = MobileNetV1(alpha=0.25, num_classes=2)
config = get_default_quant_config(model)
for name, qt in sorted(config.items()):
    print(f'  {name:30s} {qt}')
"
```

Expected output — pw0-pw4 are `int8`, pw5-pw12 are `ternary`:
```
  classifier                     int8
  features.0.dw.0                int8
  features.0.pw.0                int8       # was ternary
  features.1.dw.0                int8
  features.1.pw.0                int8       # was ternary
  features.2.dw.0                int8
  features.2.pw.0                int8       # was ternary
  features.3.dw.0                int8
  features.3.pw.0                int8       # was ternary
  features.4.dw.0                int8
  features.4.pw.0                int8       # was ternary
  features.5.dw.0                int8
  features.5.pw.0                ternary    # unchanged
  ...
  features.12.pw.0               ternary    # unchanged
  first_conv.0                   int8
```

- [ ] **Step 3: Commit**

```bash
git add model/quantize.py
git commit -m "quant: B-light cutoff — ternary only for pw with in_c >= 128

Early pointwise layers (pw0-pw4) now use INT8 to reduce per-channel
activation variance through the feature extraction backbone. Adds ~6 KB
flash but gives 40x more weight precision in critical early layers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Retrain with B-light config

**Files:**
- No code changes — this is a training run
- Output: `checkpoints/best_ternary.pt`

- [ ] **Step 1: Run training**

```bash
cd /Users/bosselab/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Cowork/ESP_Auto
source /tmp/esp_auto_venv2/bin/activate
python model/quantize.py \
  --data-dir /tmp/esp_datasets \
  --epochs-warmup 15 \
  --epochs-ttq 50 \
  --epochs-freeze 15
```

This takes ~35 minutes on MPS. The TTQ phase now only ternarizes 8 pointwise layers (pw5-pw12) instead of 13.

Expected: val accuracy should reach >= 78% during TTQ/freeze phases. Watch for the "Best ternary val accuracy" line at the end.

- [ ] **Step 2: Verify the checkpoint has the right layer structure**

```bash
cd model && python -c "
import torch
ckpt = torch.load('../checkpoints/best_ternary.pt', map_location='cpu', weights_only=False)
ternary = [k for k in ckpt if 'weight_fp' in k]
int8 = [k.rsplit('.', 1)[0] for k in ckpt if k.endswith('.weight') and 'features' in k and '.0.weight' in k and k.rsplit('.', 1)[0] + '.weight_fp' not in [t.rsplit('.', 1)[0] + '.weight_fp' for t in ternary]]
print('Ternary layers:', sorted(set(k.split('.weight_fp')[0] for k in ternary)))
print('Count:', len(set(k.split('.weight_fp')[0] for k in ternary)))
"
```

Expected: 8 ternary layers (features.5.pw.0 through features.12.pw.0). NOT features.0-4.pw.0.

---

### Task 3: Export and measure INT8 verification accuracy

**Files:**
- No code changes — runs existing export pipeline
- Output: `firmware/main/model_data.h`

- [ ] **Step 1: Run the export**

```bash
cd /Users/bosselab/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Cowork/ESP_Auto
source /tmp/esp_auto_venv2/bin/activate
python model/export_packed.py
```

Look at the output for:
1. Layer summary: pw0-pw4 should show `QUANT_INT8`, pw5-pw12 should show `QUANT_TERNARY`
2. `Verification accuracy: X/64 = Y%` — this is the key metric
3. `Round-trip validation PASSED`
4. `Generated firmware/main/model_data.h`

- [ ] **Step 2: Evaluate against success criteria**

Check the verification accuracy:
- **>= 75%**: Success. Skip PACT. Proceed to Task 5 (commit and done).
- **70-75%**: Acceptable. Consider skipping PACT. Proceed to Task 5.
- **< 70%**: Escalate to PACT. Proceed to Task 4.

- [ ] **Step 3: Check model size is within budget**

The export prints model size. Verify:
- Total weights < 100 KB (expected ~87 KB)
- Well under the 512 KB flash limit

---

### Task 4: Add PACT learned activation clamps (CONDITIONAL — only if Task 3 accuracy < 70%)

**Files:**
- Modify: `model/quantize.py:45-101` (add PACTReLU class, modify apply_fake_quantize and train_ternary)

- [ ] **Step 1: Add the PACTReLU class**

In `model/quantize.py`, add this class right after the `FakeQuantize` class (after line 72):

```python
class PACTReLU(nn.Module):
    """Learned activation clamp: clamp(x, 0, |alpha|).

    Replaces ReLU during QAT to bound per-layer activation range,
    reducing per-tensor quantization waste. Alpha is learned.
    """

    def __init__(self, init_alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.alpha.abs())
```

- [ ] **Step 2: Update `apply_fake_quantize` to use PACTReLU**

Replace the `apply_fake_quantize` function (lines 75-94) with:

```python
def apply_fake_quantize(model, use_pact=False):
    """Insert FakeQuantize (and optionally PACTReLU) after every ReLU in the model for QAT."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            new_children = []
            for child in module:
                if isinstance(child, nn.ReLU):
                    child.inplace = False
                    if use_pact:
                        new_children.append(PACTReLU(init_alpha=6.0))
                    else:
                        new_children.append(child)
                    new_children.append(FakeQuantize())
                else:
                    new_children.append(child)
            if len(new_children) != len(list(module)):
                parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                attr_name = name.split(".")[-1]
                if parent_name:
                    parent = model
                    for p in parent_name.split("."):
                        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                else:
                    parent = model
                setattr(parent, attr_name, nn.Sequential(*new_children))
```

- [ ] **Step 3: Update `train_ternary` to include alpha params in optimizer**

In `train_ternary` (around line 354), change the param group logic from:

```python
    scale_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "scale_" in name:
            scale_params.append(p)
        else:
            other_params.append(p)
```

To:

```python
    scale_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "scale_" in name or "alpha" in name:
            scale_params.append(p)
        else:
            other_params.append(p)
```

- [ ] **Step 4: Add `--pact` CLI flag to `main()`**

In the `main()` function (around line 425), add after the existing arguments:

```python
    parser.add_argument("--pact", action="store_true",
                        help="Use PACT learned activation clamps during QAT")
```

And pass it to `train_ternary` by adding to hparams (around line 468):

```python
        "use_pact": args.pact,
```

Then in `train_ternary`, change the `apply_fake_quantize` call (line 350) from:

```python
    apply_fake_quantize(model)
```

To:

```python
    apply_fake_quantize(model, use_pact=hparams.get("use_pact", False))
```

- [ ] **Step 5: Retrain with PACT enabled**

```bash
cd /Users/bosselab/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Cowork/ESP_Auto
source /tmp/esp_auto_venv2/bin/activate
python model/quantize.py \
  --data-dir /tmp/esp_datasets \
  --epochs-warmup 15 \
  --epochs-ttq 50 \
  --epochs-freeze 15 \
  --pact
```

Expected: val accuracy >= 78%. The PACT clamps may slightly reduce float accuracy but should improve INT8 deployment accuracy.

- [ ] **Step 6: Re-run export and measure**

```bash
python model/export_packed.py
```

Check `Verification accuracy` — target >= 75%.

- [ ] **Step 7: Commit**

```bash
git add model/quantize.py
git commit -m "quant: add PACT learned activation clamps for QAT

PACTReLU replaces ReLU during QAT when --pact flag is used. Learned
alpha parameter bounds per-layer activation range, reducing per-tensor
INT8 quantization waste. Alpha params trained with 0.1x base LR.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final commit of trained checkpoint and exported model

**Files:**
- No code changes — commit artifacts and verify

- [ ] **Step 1: Verify final state**

```bash
cd /Users/bosselab/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Cowork/ESP_Auto
source /tmp/esp_auto_venv2/bin/activate

# Re-run export to confirm
python model/export_packed.py

# Check the output for:
# 1. "Verification accuracy: X/64 = Y%" where Y >= 70
# 2. "Round-trip validation PASSED"
# 3. "Generated firmware/main/model_data.h"
```

- [ ] **Step 2: Verify model size**

```bash
ls -la firmware/main/model_data.h
```

Expected: model_data.h should be < 200 KB (C header with hex arrays).

- [ ] **Step 3: Commit all remaining changes**

```bash
git add model/quantize.py model/export_packed.py CLAUDE.md firmware/main/model_data.h
git commit -m "B-light: INT8 early pointwise + retrained ternary model

Quant config: pw0-pw4 INT8, pw5-pw12 ternary, all DW INT8.
Float val accuracy: XX%, INT8 verification: YY/64 = ZZ%.
Total model weights: ~87 KB (+6 KB vs all-ternary).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

Fill in XX, YY, ZZ with actual measured values.
