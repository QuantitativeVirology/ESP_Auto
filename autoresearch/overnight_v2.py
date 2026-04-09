#!/usr/bin/env python3
"""Overnight autoresearch v2: KD baselines + all-INT8 QAT + percentile export.

Strategy: Train KD baselines for each alpha, then QAT with all-INT8,
export with percentile scaling, flash, measure.

Usage:
    source /tmp/esp_auto_venv3/bin/activate
    python autoresearch/overnight_v2.py
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

RESULTS_TSV = Path(__file__).parent / "search_results_v2.tsv"
ESP_PORT = os.environ.get("ESP_PORT", "/dev/cu.usbmodem5B414820541")
IDF_PATH = Path.home() / "Projects" / "esp-idf"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
DATA_DIR = Path("/tmp/esp_datasets")

VARIANTS = [
    # Already have KD baseline for 0.35
    {"name": "int8_a035_kd", "alpha": 0.35, "kd_epochs": 0, "qat_epochs": (15, 50, 15), "lr": 5e-4},
    # Longer QAT for 0.35
    {"name": "int8_a035_kd_long", "alpha": 0.35, "kd_epochs": 0, "qat_epochs": (20, 80, 20), "lr": 3e-4},
    # Lower LR QAT for 0.35
    {"name": "int8_a035_kd_lowlr", "alpha": 0.35, "kd_epochs": 0, "qat_epochs": (15, 50, 15), "lr": 1e-4},
    # Train NEW KD baseline for 0.25 and QAT
    {"name": "int8_a025_kd", "alpha": 0.25, "kd_epochs": 200, "qat_epochs": (15, 50, 15), "lr": 5e-4},
    # Train NEW KD baseline for 0.30
    {"name": "int8_a030_kd", "alpha": 0.30, "kd_epochs": 200, "qat_epochs": (15, 50, 15), "lr": 5e-4},
    # 0.35 with stronger QAT (more warmup)
    {"name": "int8_a035_kd_warmup", "alpha": 0.35, "kd_epochs": 0, "qat_epochs": (30, 50, 20), "lr": 5e-4},
]


def train_kd_baseline(alpha, epochs=200):
    """Train a KD baseline using pretrained MobileNetV2 teacher."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    from train_baseline import MobileNetV1, get_loaders

    save_dir = PROJECT_ROOT / "checkpoints" / "alpha{}".format(str(alpha).replace('.',''))
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = save_dir / "best_model.pt"

    if baseline_path.exists():
        model = MobileNetV1(alpha=alpha, num_classes=2)
        try:
            model.load_state_dict(torch.load(baseline_path, map_location='cpu', weights_only=True))
            print("[kd] Using existing baseline for alpha={}: {}".format(alpha, baseline_path))
            return baseline_path
        except Exception:
            print("[kd] Existing baseline corrupt, retraining")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Teacher
    teacher_path = PROJECT_ROOT / "checkpoints" / "teacher_mobilenetv2.pt"
    teacher = models.mobilenet_v2(weights=None if teacher_path.exists() else 'DEFAULT')
    teacher.classifier[-1] = nn.Linear(teacher.classifier[-1].in_features, 2)

    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location='cpu', weights_only=True))
        print("[kd] Loaded cached teacher")
    else:
        teacher = teacher.to(device)
        train_loader, val_loader = get_loaders(str(DATA_DIR), 64, 96, 4)
        cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
        dog_count = len(train_loader.dataset) - cat_count
        weight = torch.tensor([dog_count / cat_count, 1.0]).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss(weight=weight)
        for ep in range(15):
            teacher.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = crit(teacher(imgs), labels)
                opt.zero_grad(); loss.backward(); opt.step()
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher.state_dict(), teacher_path)
        print("[kd] Teacher trained and saved")

    teacher = teacher.to(device)
    teacher.requires_grad_(False)
    teacher.train(False)

    # Student KD training
    student = MobileNetV1(alpha=alpha, num_classes=2).to(device)
    train_loader, val_loader = get_loaders(str(DATA_DIR), 64, 96, 4)

    cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
    dog_count = len(train_loader.dataset) - cat_count
    weight = torch.tensor([dog_count / cat_count, 1.0]).to(device)
    hard_crit = nn.CrossEntropyLoss(weight=weight)

    T, alpha_kd = 5.0, 0.7
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        student.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean') * (T * T)
            hard_loss = hard_crit(student_logits, labels)
            loss = alpha_kd * soft_loss + (1 - alpha_kd) * hard_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        student.train(False)
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (student(imgs).argmax(1) == labels).sum().item()
                total += len(labels)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), baseline_path)
        if (epoch + 1) % 20 == 0:
            print("[kd] alpha={} epoch {}/{}: val_acc={:.1f}% best={:.1f}%".format(
                alpha, epoch+1, epochs, 100*val_acc, 100*best_acc))

    print("[kd] alpha={} KD complete. Best: {:.1f}%".format(alpha, 100*best_acc))
    return baseline_path


def run_qat(alpha, baseline_path, qat_epochs, lr):
    """Run QAT with all-INT8 config. Returns path to best_ternary.pt."""
    import torch
    import torch.nn as nn
    from train_baseline import MobileNetV1, get_loaders
    from quantize import train_ternary

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = MobileNetV1(alpha=alpha, num_classes=2).to(device)

    config = {}
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            config[n] = 'int8'

    train_loader, val_loader = get_loaders(str(DATA_DIR), 64, 96, 4)

    save_dir = str(PROJECT_ROOT / "checkpoints" / "alpha{}_int8_{}".format(
        str(alpha).replace('.',''), int(sum(qat_epochs))))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    hparams = {
        'lr': lr, 'wd': 1e-4,
        'epochs_warmup': qat_epochs[0],
        'epochs_ttq': qat_epochs[1],
        'epochs_freeze': qat_epochs[2],
        'threshold_ratio': 0.05,
        'quant_config': config,
        'baseline_weights': str(baseline_path),
        'save_dir': save_dir,
    }

    model = train_ternary(model, train_loader, val_loader, device, hparams)
    return Path(save_dir) / "best_ternary.pt"


def export_and_verify(alpha, weights_path):
    """Export all-INT8 model. Returns (sim_accuracy, model_kb)."""
    import torch
    import torch.nn as nn
    from train_baseline import MobileNetV1
    from quantize import apply_ttq, set_ttq_enabled
    from export_packed import extract_layers, generate_header, validate_packing
    from export_packed import _verify_quantized_pipeline

    model = MobileNetV1(alpha=alpha, num_classes=2)
    config = {}
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            config[n] = 'int8'
    apply_ttq(model, config, 0.05)
    model.load_state_dict(
        torch.load(weights_path, map_location='cpu', weights_only=True), strict=False)
    set_ttq_enabled(model, True)
    model.train(False)

    layers = extract_layers(model, 0.05)
    if not validate_packing(layers):
        return -1, -1

    sim_acc = _verify_quantized_pipeline(layers) or 0.0

    total_weights = 0
    for L in layers:
        if 'weights_int8' in L:
            total_weights += len(L['weights_int8'])
        elif 'weights_packed' in L:
            total_weights += len(L['weights_packed'])
    model_kb = total_weights / 1024

    generate_header(layers, str(FIRMWARE_DIR / "main" / "model_data.h"))
    return sim_acc, model_kb


def build_and_flash():
    """Build and flash firmware."""
    result = subprocess.run(
        ["bash", "-c",
         "cd {} && source export.sh 2>/dev/null && "
         "cd {} && idf.py build 2>&1 && "
         "idf.py -p {} flash 2>&1".format(IDF_PATH, FIRMWARE_DIR, ESP_PORT)],
        capture_output=True, text=True, timeout=300)
    success = result.returncode == 0
    if not success:
        # Check for partition error specifically
        if 'app partition is too small' in result.stdout:
            print("[flash] App too large for partition")
        else:
            print("[flash] Failed: {}".format(result.stdout[-200:]))
    return success


def read_benchmark():
    """Read on-device benchmark."""
    import serial
    try:
        ser = serial.Serial(ESP_PORT, 115200, timeout=2)
        ser.dtr = False; time.sleep(0.1); ser.dtr = True; time.sleep(4)
        while ser.in_waiting:
            ser.readline()
        ser.write(b"RUN_BENCHMARK\r\n")
        start = time.time()
        while time.time() - start < 120:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            m = re.match(
                r"METRIC\s+latency_us=(\d+)\s+accuracy=([\d.]+)\s+sram_free=(\d+)", line)
            if m:
                ser.close()
                return {
                    "latency_us": int(m.group(1)),
                    "accuracy": float(m.group(2)),
                    "sram_free": int(m.group(3)),
                }
        ser.close()
    except Exception as e:
        print("[bench] Error: {}".format(e))
    return None


def log_result(name, float_acc, sim_acc, device_acc, latency_us, sram_free, model_kb, train_min):
    header = "timestamp\tvariant\tfloat_acc\tsim_int8\tdevice_acc\tlatency_ms\tsram_free\tmodel_kb\ttrain_min\n"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lat_ms = latency_us / 1000 if latency_us > 0 else -1
    with open(RESULTS_TSV, "a") as f:
        f.write("{}\t{}\t{:.1f}\t{:.1f}\t{:.2f}\t{:.1f}\t{}\t{:.1f}\t{:.1f}\n".format(
            ts, name, float_acc, sim_acc, device_acc, lat_ms, sram_free, model_kb, train_min))
    print("\n=== RESULT: {} ===".format(name))
    print("  Float: {:.1f}%  Sim INT8: {:.1f}%  Device: {:.0f}%  Latency: {:.0f}ms  Model: {:.0f}KB".format(
        float_acc, sim_acc, device_acc*100, lat_ms, model_kb))


def main():
    print("[v2] Overnight search: {} variants".format(len(VARIANTS)))
    print("[v2] Results: {}".format(RESULTS_TSV))

    for i, v in enumerate(VARIANTS):
        name = v["name"]
        alpha = v["alpha"]
        print("\n" + "#" * 60)
        print("# [{}/{}] {} (alpha={})".format(i+1, len(VARIANTS), name, alpha))
        print("#" * 60)

        start = time.time()

        try:
            # Step 1: KD baseline
            if v["kd_epochs"] > 0:
                print("[v2] Training KD baseline for alpha={}...".format(alpha))
                baseline_path = train_kd_baseline(alpha, v["kd_epochs"])
            else:
                baseline_path = PROJECT_ROOT / "checkpoints" / "alpha{}".format(
                    str(alpha).replace('.','')) / "best_model.pt"
                if not baseline_path.exists():
                    print("[v2] No baseline, training KD...")
                    baseline_path = train_kd_baseline(alpha, 200)

            # Step 2: QAT
            print("[v2] Running QAT...")
            qat_path = run_qat(alpha, baseline_path, v["qat_epochs"], v["lr"])

            # Step 3: Measure float accuracy
            import torch
            import torch.nn as nn
            from train_baseline import MobileNetV1, get_loaders
            from quantize import apply_ttq, set_ttq_enabled

            model = MobileNetV1(alpha=alpha, num_classes=2)
            config = {}
            for n, m in model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    config[n] = 'int8'
            apply_ttq(model, config, 0.05)
            model.load_state_dict(
                torch.load(qat_path, map_location='cpu', weights_only=True), strict=False)
            set_ttq_enabled(model, True)
            model.train(False)
            _, val_loader = get_loaders(str(DATA_DIR), 64, 96, 4)
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    correct += (model(imgs).argmax(1) == labels).sum().item()
                    total += len(labels)
            float_acc = 100 * correct / total
            print("[v2] Float accuracy: {:.1f}%".format(float_acc))

            # Step 4: Export
            print("[v2] Exporting...")
            sim_acc, model_kb = export_and_verify(alpha, qat_path)
            print("[v2] Simulated INT8: {:.1f}%".format(sim_acc))

            train_min = (time.time() - start) / 60

            # Step 5: Flash + measure
            device_acc = -1.0
            latency_us = -1
            sram_free = -1

            if sim_acc >= 50:
                print("[v2] Building and flashing...")
                if build_and_flash():
                    time.sleep(2)
                    metric = read_benchmark()
                    if metric:
                        device_acc = metric["accuracy"]
                        latency_us = metric["latency_us"]
                        sram_free = metric["sram_free"]
                        print("[v2] On-device: {:.0f}% @ {:.0f}ms".format(
                            device_acc*100, latency_us/1000))

            log_result(name, float_acc, sim_acc, device_acc, latency_us, sram_free, model_kb, train_min)

        except Exception as e:
            import traceback
            print("[v2] ERROR in {}: {}".format(name, e))
            traceback.print_exc()
            log_result(name, -1, -1, -1, -1, -1, -1, (time.time() - start) / 60)

    print("\n" + "=" * 60)
    print("[v2] ALL DONE")
    if RESULTS_TSV.exists():
        print(RESULTS_TSV.read_text())


if __name__ == "__main__":
    main()
