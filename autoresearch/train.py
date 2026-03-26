#!/usr/bin/env python3
"""Mutable target for autoresearch — defines model architecture and quantization config.

THIS FILE IS EDITED BY THE LLM AGENT during autoresearch optimization.
It must expose three functions: build_model(), get_quant_config(), get_hparams().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

import torch.nn as nn
from train_baseline import MobileNetV1, DepthwiseSeparableConv


def build_model():
    """Return an untrained model. The LLM agent modifies this."""
    return MobileNetV1(alpha=0.25, num_classes=2)


def get_quant_config():
    """Per-layer quantization mode.

    Keys are module names from model.named_modules().
    Values: "ternary" or "int8".
    First conv and classifier MUST be "int8".
    """
    return {
        "first_conv.0": "int8",
        # Depthwise layers: INT8 (few params, high quantization sensitivity)
        "features.0.dw.0": "int8",
        "features.1.dw.0": "int8",
        "features.2.dw.0": "int8",
        "features.3.dw.0": "int8",
        "features.4.dw.0": "int8",
        "features.5.dw.0": "int8",
        "features.6.dw.0": "int8",
        "features.7.dw.0": "int8",
        "features.8.dw.0": "int8",
        "features.9.dw.0": "int8",
        "features.10.dw.0": "int8",
        "features.11.dw.0": "int8",
        "features.12.dw.0": "int8",
        # Pointwise layers: ternary (many params, benefit from compression)
        "features.0.pw.0": "ternary",
        "features.1.pw.0": "ternary",
        "features.2.pw.0": "ternary",
        "features.3.pw.0": "ternary",
        "features.4.pw.0": "ternary",
        "features.5.pw.0": "ternary",
        "features.6.pw.0": "ternary",
        "features.7.pw.0": "ternary",
        "features.8.pw.0": "ternary",
        "features.9.pw.0": "ternary",
        "features.10.pw.0": "ternary",
        "features.11.pw.0": "ternary",
        "features.12.pw.0": "ternary",
        # Classifier: INT8
        "classifier": "int8",
    }


def get_hparams():
    """Training hyperparameters."""
    return {
        "lr": 5e-4,
        "wd": 1e-4,
        "epochs_warmup": 20,
        "epochs_ttq": 40,
        "epochs_freeze": 10,
        "batch_size": 64,
        "threshold_ratio": 0.05,
    }
