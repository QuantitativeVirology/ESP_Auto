#!/usr/bin/env python3
"""Trained Ternary Quantization (TTQ) for MobileNetV1.

Implements Li et al. (ICLR 2017) with per-layer learned scales.
Three-phase training: warmup → TTQ with STE → frozen fine-tune.
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train_baseline import MobileNetV1, get_loaders


# ---------------------------------------------------------------------------
# Fake INT8 quantization for QAT
# ---------------------------------------------------------------------------

class FakeQuantizeFunction(torch.autograd.Function):
    """Simulate per-tensor INT8 quantization matching deployment requant.

    Uses per-tensor scaling (single scale for all channels) to match the
    firmware's per-tensor activation quantization. This ensures QAT training
    produces a model robust to the actual deployment quantization scheme.
    """

    @staticmethod
    def forward(ctx, x):
        # Per-tensor quantize-dequantize (single scale for all channels)
        absmax = x.abs().max().clamp(min=1e-8)
        scale = absmax / 127.0
        x_q = torch.clamp(torch.round(x / scale), -128, 127)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # STE: pass gradient through unchanged


class FakeQuantize(nn.Module):
    """Module wrapper for fake INT8 quantization (inserted after ReLU)."""

    def __init__(self):
        super().__init__()
        self.enabled = False
        # Track running absmax per channel for export
        self.register_buffer('running_absmax', None)
        self.momentum = 0.1

    def forward(self, x):
        if self.enabled:
            result = FakeQuantizeFunction.apply(x)
            # Track running per-tensor absmax for export
            with torch.no_grad():
                absmax = x.abs().max().unsqueeze(0)
                if self.running_absmax is None or self.running_absmax.shape != absmax.shape:
                    self.running_absmax = absmax.clone()
                else:
                    self.running_absmax = (1 - self.momentum) * self.running_absmax + self.momentum * absmax
            return result
        return x


class PACTReLU(nn.Module):
    """Learned activation clamp: clamp(x, 0, |alpha|).

    Replaces ReLU during QAT to bound per-layer activation range,
    reducing per-tensor quantization waste. Alpha is learned.
    """

    def __init__(self, init_alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return torch.min(F.relu(x), self.alpha.abs())


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


def set_fake_quantize_enabled(model, enabled):
    """Enable/disable all FakeQuantize modules in the model."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.enabled = enabled


# ---------------------------------------------------------------------------
# Ternary quantization core
# ---------------------------------------------------------------------------

class TernarizeFunction(torch.autograd.Function):
    """Ternary quantization with Straight-Through Estimator (STE)."""

    @staticmethod
    def forward(ctx, w_fp, scale_pos, scale_neg, threshold_ratio):
        threshold = threshold_ratio * w_fp.abs().max()

        pos_mask = w_fp > threshold
        neg_mask = w_fp < -threshold
        zero_mask = ~(pos_mask | neg_mask)

        w_ternary = torch.zeros_like(w_fp)
        w_ternary[pos_mask] = scale_pos
        w_ternary[neg_mask] = -scale_neg

        ctx.save_for_backward(pos_mask, neg_mask, zero_mask)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output):
        pos_mask, neg_mask, zero_mask = ctx.saved_tensors

        # STE: pass gradient through for non-zero positions only
        grad_w_fp = grad_output.clone()
        grad_w_fp[zero_mask] = 0

        # Gradients for per-layer scales
        grad_scale_pos = grad_output[pos_mask].sum() if pos_mask.any() else torch.tensor(0.0)
        grad_scale_neg = -grad_output[neg_mask].sum() if neg_mask.any() else torch.tensor(0.0)

        return grad_w_fp, grad_scale_pos, grad_scale_neg, None


class TernaryQuantWrapper(nn.Module):
    """Wraps a Conv2d or Linear layer with TTQ quantization."""

    def __init__(self, module, threshold_ratio=0.05):
        super().__init__()
        self.threshold_ratio = threshold_ratio

        # Store module config for functional dispatch
        self._is_conv = isinstance(module, nn.Conv2d)
        if self._is_conv:
            self._stride = module.stride
            self._padding = module.padding
            self._dilation = module.dilation
            self._groups = module.groups
        self._bias = module.bias  # may be None

        # Full-precision shadow weights (gradient target)
        w = module.weight.data.clone()
        self.weight_fp = nn.Parameter(w)

        # Per-layer learned scales (initialized from weight statistics)
        pos_vals = w[w > 0]
        neg_vals = w[w < 0].abs()
        self.scale_pos = nn.Parameter(
            torch.tensor(pos_vals.mean().item() if len(pos_vals) > 0 else 0.1)
        )
        self.scale_neg = nn.Parameter(
            torch.tensor(neg_vals.mean().item() if len(neg_vals) > 0 else 0.1)
        )

        # Keep reference for export (geometry info)
        self.module = module

        self.enabled = False
        self.frozen = False
        self._frozen_ternary = None

    def forward(self, x):
        if self.frozen and self._frozen_ternary is not None:
            w = self._frozen_ternary * self.scale_pos
        elif self.enabled:
            w = TernarizeFunction.apply(
                self.weight_fp, self.scale_pos, self.scale_neg,
                self.threshold_ratio
            )
        else:
            w = self.weight_fp

        if self._is_conv:
            return F.conv2d(x, w, self._bias,
                            self._stride, self._padding,
                            self._dilation, self._groups)
        return F.linear(x, w, self._bias)

    def freeze_ternary(self):
        """Freeze the ternary mask but keep scales trainable."""
        with torch.no_grad():
            threshold = self.threshold_ratio * self.weight_fp.abs().max()
            pos_mask = self.weight_fp > threshold
            neg_mask = self.weight_fp < -threshold

            frozen = torch.zeros_like(self.weight_fp)
            frozen[pos_mask] = 1.0
            frozen[neg_mask] = -1.0
            self._frozen_ternary = frozen

        self.weight_fp.requires_grad_(False)
        self.frozen = True
        self.enabled = False

    def get_ternary_stats(self):
        """Return sparsity and scale info for this layer."""
        with torch.no_grad():
            threshold = self.threshold_ratio * self.weight_fp.abs().max()
            total = self.weight_fp.numel()
            zero = ((self.weight_fp.abs() <= threshold).sum()).item()
            return {
                "total": total,
                "zero": zero,
                "sparsity": zero / total,
                "scale_pos": self.scale_pos.item(),
                "scale_neg": self.scale_neg.item(),
            }


# ---------------------------------------------------------------------------
# Apply TTQ to model
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model, name):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def apply_ttq(model, quant_config, threshold_ratio=0.05):
    """Wrap specified layers with TernaryQuantWrapper.

    Args:
        quant_config: dict mapping module name -> "ternary" | "int8"
    """
    for name, mode in quant_config.items():
        if mode != "ternary":
            continue

        parent, attr = _get_parent_and_attr(model, name)
        module = getattr(parent, attr)

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        wrapper = TernaryQuantWrapper(module, threshold_ratio)
        setattr(parent, attr, wrapper)

    return model


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_ttq_enabled(model, enabled):
    for m in model.modules():
        if isinstance(m, TernaryQuantWrapper):
            m.enabled = enabled


def freeze_ternary(model):
    for m in model.modules():
        if isinstance(m, TernaryQuantWrapper):
            m.freeze_ternary()


def print_ternary_stats(model):
    for name, m in model.named_modules():
        if isinstance(m, TernaryQuantWrapper):
            stats = m.get_ternary_stats()
            print(f"  {name}: sparsity={stats['sparsity']:.3f} "
                  f"s+={stats['scale_pos']:.4f} s-={stats['scale_neg']:.4f} "
                  f"params={stats['total']}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.train(False)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train_ternary(model, train_loader, val_loader, device, hparams):
    """Three-phase TTQ training."""
    # Load pre-trained baseline weights if provided
    baseline_weights = hparams.get("baseline_weights")
    if baseline_weights and Path(baseline_weights).exists():
        model.load_state_dict(
            torch.load(baseline_weights, weights_only=True, map_location="cpu")
        )
        model = model.to(device)
        print(f"Loaded baseline weights from {baseline_weights}")

    quant_config = hparams.get("quant_config", get_default_quant_config(model))
    apply_ttq(model, quant_config, hparams.get("threshold_ratio", 0.05))

    # Insert fake INT8 quantization after each ReLU for QAT
    apply_fake_quantize(model, use_pact=hparams.get("use_pact", False))
    set_fake_quantize_enabled(model, False)  # Disabled during warmup

    # Separate param groups: scales get lower LR
    scale_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "scale_" in name or "alpha" in name:
            scale_params.append(p)
        else:
            other_params.append(p)

    optimizer = optim.AdamW([
        {"params": other_params, "lr": hparams["lr"], "weight_decay": hparams["wd"]},
        {"params": scale_params, "lr": hparams["lr"] * 0.1, "weight_decay": 0},
    ])

    total_epochs = (hparams["epochs_warmup"] + hparams["epochs_ttq"]
                    + hparams["epochs_freeze"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Class-balanced loss (Oxford Pets has ~2x more dogs than cats)
    cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
    dog_count = len(train_loader.dataset) - cat_count
    if cat_count > 0 and dog_count > 0:
        weight = torch.tensor([dog_count / cat_count, 1.0]).to(device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_acc = 0.0
    save_dir = Path(hparams.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    ttq_active = False

    for epoch in range(1, total_epochs + 1):
        # Phase transitions
        if epoch == 1:
            print("Phase 1: Full-precision warmup")
            set_ttq_enabled(model, False)

        if epoch == hparams["epochs_warmup"] + 1:
            print("Phase 2: TTQ + QAT (fake INT8 quantization)")
            set_ttq_enabled(model, True)
            set_fake_quantize_enabled(model, True)
            ttq_active = True
            best_acc = 0.0  # Reset: warmup accuracy isn't comparable

        if epoch == hparams["epochs_warmup"] + hparams["epochs_ttq"] + 1:
            print("Phase 3: Frozen ternary fine-tune (QAT still active)")
            freeze_ternary(model)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{total_epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if epoch % 10 == 0:
            print_ternary_stats(model)

        # Only save best from TTQ/freeze phases (warmup weights are not ternary-trained)
        if ttq_active and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_ternary.pt")

    print(f"Best ternary val accuracy: {best_acc:.4f}")
    model.load_state_dict(
        torch.load(save_dir / "best_ternary.pt", weights_only=True),
        strict=False  # FakeQuantize running_absmax may be None in checkpoint
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="model/datasets")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--baseline-weights", default="checkpoints/best_model.pt",
                        help="Path to pre-trained INT8 baseline weights")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs-warmup", type=int, default=30)
    parser.add_argument("--epochs-ttq", type=int, default=70)
    parser.add_argument("--epochs-freeze", type=int, default=20)
    parser.add_argument("--threshold-ratio", type=float, default=0.05)
    parser.add_argument("--pact", action="store_true",
                        help="Use PACT learned activation clamps during QAT")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained baseline
    model = MobileNetV1(alpha=args.alpha, num_classes=2)
    if Path(args.baseline_weights).exists():
        model.load_state_dict(
            torch.load(args.baseline_weights, weights_only=True, map_location="cpu")
        )
        print(f"Loaded baseline weights from {args.baseline_weights}")
    else:
        print("WARNING: No baseline weights found, training from scratch")

    model = model.to(device)

    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, args.size, args.workers
    )

    hparams = {
        "lr": args.lr,
        "wd": args.wd,
        "epochs_warmup": args.epochs_warmup,
        "epochs_ttq": args.epochs_ttq,
        "epochs_freeze": args.epochs_freeze,
        "threshold_ratio": args.threshold_ratio,
        "use_pact": args.pact,
        "save_dir": args.save_dir,
    }

    model = train_ternary(model, train_loader, val_loader, device, hparams)
    print("Ternary quantization complete.")
    print_ternary_stats(model)


if __name__ == "__main__":
    main()
