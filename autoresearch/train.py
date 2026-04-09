#!/usr/bin/env python3
"""Mutable target for autoresearch — defines model architecture and quantization config.

THIS FILE IS EDITED BY THE LLM AGENT during autoresearch optimization.
It must expose three functions: build_model(), get_quant_config(), get_hparams().

Current config: alpha=0.35, all-INT8 (no ternary). Targeting 90%+ on-device accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

import torch.nn as nn
from train_baseline import MobileNetV1, DepthwiseSeparableConv


def build_model():
    """Return an untrained model."""
    return MobileNetV1(alpha=0.35, num_classes=2)


def get_quant_config():
    """All-INT8 quantization — maximum precision, fits in 512KB flash (398KB weights)."""
    return {
        "first_conv.0": "int8",
        "features.0.dw.0": "int8",
        "features.0.pw.0": "int8",
        "features.1.dw.0": "int8",
        "features.1.pw.0": "int8",
        "features.2.dw.0": "int8",
        "features.2.pw.0": "int8",
        "features.3.dw.0": "int8",
        "features.3.pw.0": "int8",
        "features.4.dw.0": "int8",
        "features.4.pw.0": "int8",
        "features.5.dw.0": "int8",
        "features.5.pw.0": "int8",
        "features.6.dw.0": "int8",
        "features.6.pw.0": "int8",
        "features.7.dw.0": "int8",
        "features.7.pw.0": "int8",
        "features.8.dw.0": "int8",
        "features.8.pw.0": "int8",
        "features.9.dw.0": "int8",
        "features.9.pw.0": "int8",
        "features.10.dw.0": "int8",
        "features.10.pw.0": "int8",
        "features.11.dw.0": "int8",
        "features.11.pw.0": "int8",
        "features.12.dw.0": "int8",
        "features.12.pw.0": "int8",
        "classifier": "int8",
    }


def get_hparams():
    """Training hyperparameters."""
    return {
        "lr": 5e-4,
        "wd": 1e-4,
        "epochs_warmup": 15,
        "epochs_ttq": 50,
        "epochs_freeze": 15,
        "batch_size": 64,
        "threshold_ratio": 0.05,
        "quant_config": get_quant_config(),
        "baseline_weights": "checkpoints/alpha035/best_model.pt",
    }
