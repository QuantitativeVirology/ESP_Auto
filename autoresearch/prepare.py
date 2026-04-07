#!/usr/bin/env python3
"""Immutable autoresearch harness — orchestrates train → quantize → export → flash → measure.

This file is NEVER modified by the LLM agent. It calls train.py's interface functions
and coordinates the full evaluation pipeline.
"""

import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

TIMEOUT_SECONDS = 7200  # 2 hours — training ~80 min on MPS
CACHE_DIR = PROJECT_ROOT / "model" / ".cache"
# Try /tmp first (avoids iCloud Drive I/O hangs), fall back to in-repo
DATA_DIR = Path("/tmp/esp_datasets") if Path("/tmp/esp_datasets").exists() else PROJECT_ROOT / "model" / "datasets"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"


def timeout_handler(signum, frame):
    raise TimeoutError(f"prepare.py exceeded {TIMEOUT_SECONDS}s timeout")


def load_train_module():
    """Dynamically load train.py (the mutable target)."""
    train_path = Path(__file__).parent / "train.py"
    spec = importlib.util.spec_from_file_location("train_target", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_arch_hash(train_module):
    """Hash model architecture to detect changes."""
    import inspect
    src = inspect.getsource(train_module.build_model)
    src += inspect.getsource(train_module.get_quant_config)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def get_cached_weights(arch_hash):
    """Return path to cached weights if they exist."""
    cache = CACHE_DIR / arch_hash
    weights = cache / "best_ternary.pt"
    if weights.exists():
        return weights
    return None


def train_and_quantize(train_module, arch_hash):
    """Train model with TTQ and return path to saved weights."""
    import torch
    from train_baseline import get_loaders
    from quantize import apply_ttq, train_ternary

    hparams = train_module.get_hparams()

    # Check cache
    cached = get_cached_weights(arch_hash)
    if cached:
        print(f"[prepare] Using cached weights: {cached}")
        return cached

    # Build and train
    model = train_module.build_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    train_loader, val_loader = get_loaders(
        str(DATA_DIR), hparams.get("batch_size", 64), 96, 4
    )

    cache_dir = CACHE_DIR / arch_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    hparams["save_dir"] = str(cache_dir)

    model = train_ternary(model, train_loader, val_loader, device, hparams)

    weights_path = cache_dir / "best_ternary.pt"
    return weights_path


def export_model(train_module, weights_path):
    """Export packed weights to C header. Returns (success, int8_accuracy)."""
    import torch
    from quantize import apply_ttq, set_ttq_enabled
    from export_packed import extract_layers, validate_packing, generate_header
    from export_packed import _verify_quantized_pipeline

    model = train_module.build_model()
    config = train_module.get_quant_config()
    hparams = train_module.get_hparams()
    threshold = hparams.get("threshold_ratio", 0.05)

    apply_ttq(model, config, threshold)
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location="cpu"),
        strict=False  # Checkpoint may include FakeQuantize buffers
    )
    set_ttq_enabled(model, True)

    layers = extract_layers(model, threshold)
    if not validate_packing(layers):
        print("[prepare] Packing validation FAILED")
        return False, 0.0

    # Simulated INT8 verification (pre-filter before flash)
    int8_acc = _verify_quantized_pipeline(layers) or 0.0
    print(f"[prepare] Simulated INT8 accuracy: {int8_acc:.1f}%")

    output = FIRMWARE_DIR / "main" / "model_data.h"
    generate_header(layers, str(output))
    return True, int8_acc


def flash_and_measure(port):
    """Build, flash, collect metrics, return score."""
    harness = PROJECT_ROOT / "harness" / "flash_and_measure.py"
    result = subprocess.run(
        [sys.executable, str(harness), "--port", port],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )

    # Parse SCORE line from output
    for line in result.stdout.splitlines():
        if line.startswith("SCORE:"):
            return float(line.split(":")[1].strip())

    print(f"[prepare] No SCORE in output:\n{result.stdout[-500:]}")
    return -1.0


def find_esp_port():
    """Auto-detect ESP32 serial port."""
    import glob
    for pattern in ["/dev/cu.usbserial-*", "/dev/cu.usbmodem*", "/dev/cu.SLAB*"]:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    return None


MIN_INT8_ACCURACY = 60.0  # Skip flash if simulated accuracy below this


def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    port = os.environ.get("ESP_PORT") or find_esp_port()
    if not port:
        print("[prepare] WARNING: No ESP32 port found, will skip flash")

    try:
        train_module = load_train_module()
        arch_hash = compute_arch_hash(train_module)
        print(f"[prepare] Architecture hash: {arch_hash}")

        weights_path = train_and_quantize(train_module, arch_hash)

        result = export_model(train_module, weights_path)
        if isinstance(result, tuple):
            success, int8_acc = result
        else:
            success, int8_acc = result, 0.0

        if not success:
            print("SCORE: -1.000000")
            return

        # Pre-filter: skip expensive flash if simulated accuracy too low
        if int8_acc < MIN_INT8_ACCURACY:
            print(f"[prepare] Simulated INT8 accuracy {int8_acc:.1f}% < {MIN_INT8_ACCURACY}%, skipping flash")
            score = int8_acc * 0.001  # Low score proportional to accuracy
            print(f"SCORE: {score:.6f}")
            return

        if not port:
            print(f"[prepare] No ESP32 port — using simulated accuracy as score proxy")
            print(f"SCORE: {int8_acc * 0.01:.6f}")
            return

        score = flash_and_measure(port)
        print(f"SCORE: {score:.6f}")

    except TimeoutError as e:
        print(f"[prepare] {e}")
        print("SCORE: -1.000000")
    except Exception as e:
        print(f"[prepare] Error: {e}")
        import traceback
        traceback.print_exc()
        print("SCORE: -1.000000")


if __name__ == "__main__":
    main()
