#!/usr/bin/env python3
"""Weekend continuous optimization loop.

Runs multiple autoresearch sweeps back-to-back, adapting strategy
based on results from each sweep. Targets 90%+ on-device accuracy.

Usage:
    source /tmp/esp_auto_venv3/bin/activate
    nohup python -u autoresearch/weekend_loop.py > autoresearch/weekend.log 2>&1 &
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

RESULTS_TSV = Path(__file__).parent / "weekend_results.tsv"
ESP_PORT = os.environ.get("ESP_PORT", "/dev/cu.usbmodem5B414820541")
IDF_PATH = Path.home() / "Projects" / "esp-idf"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
DATA_DIR = Path("/tmp/esp_datasets")

# Stop on Monday morning Berlin time (UTC+2)
STOP_TIME = datetime(2026, 4, 13, 7, 0)  # Monday 7:00 AM


def train_kd_baseline(alpha, epochs=200):
    """Train KD baseline if not cached."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    from train_baseline import MobileNetV1, get_loaders

    save_dir = PROJECT_ROOT / "checkpoints" / "alpha{}".format(str(alpha).replace('.', ''))
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = save_dir / "best_model.pt"

    if baseline_path.exists():
        model = MobileNetV1(alpha=alpha, num_classes=2)
        try:
            model.load_state_dict(torch.load(baseline_path, map_location='cpu', weights_only=True))
            print("[kd] Cached baseline for alpha={}".format(alpha))
            return baseline_path
        except Exception:
            pass

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    teacher_path = PROJECT_ROOT / "checkpoints" / "teacher_mobilenetv2.pt"
    teacher = models.mobilenet_v2(weights=None if teacher_path.exists() else 'DEFAULT')
    teacher.classifier[-1] = nn.Linear(teacher.classifier[-1].in_features, 2)
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location='cpu', weights_only=True))
    else:
        teacher = teacher.to(device)
        train_loader, _ = get_loaders(str(DATA_DIR), 64, 96, 4)
        cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
        dog_count = len(train_loader.dataset) - cat_count
        w = torch.tensor([dog_count / cat_count, 1.0]).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss(weight=w)
        for ep in range(15):
            teacher.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = crit(teacher(imgs), labels)
                opt.zero_grad(); loss.backward(); opt.step()
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher.state_dict(), teacher_path)

    teacher = teacher.to(device)
    teacher.requires_grad_(False)
    teacher.train(False)

    student = MobileNetV1(alpha=alpha, num_classes=2).to(device)
    train_loader, val_loader = get_loaders(str(DATA_DIR), 64, 96, 4)
    cat_count = sum(1 for _, l in train_loader.dataset if l == 0)
    dog_count = len(train_loader.dataset) - cat_count
    w = torch.tensor([dog_count / cat_count, 1.0]).to(device)
    hard_crit = nn.CrossEntropyLoss(weight=w)
    T, alpha_kd = 5.0, 0.7
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        student.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                tl = teacher(imgs)
            sl = student(imgs)
            soft = F.kl_div(F.log_softmax(sl/T, dim=1), F.softmax(tl/T, dim=1), reduction='batchmean') * T*T
            hard = hard_crit(sl, labels)
            loss = alpha_kd * soft + (1-alpha_kd) * hard
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        student.train(False)
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (student(imgs).argmax(1) == labels).sum().item()
                total += len(labels)
        if correct/total > best_acc:
            best_acc = correct/total
            torch.save(student.state_dict(), baseline_path)
        if (epoch+1) % 50 == 0:
            print("[kd] a={} ep={}/{} acc={:.1f}% best={:.1f}%".format(
                alpha, epoch+1, epochs, 100*correct/total, 100*best_acc))

    print("[kd] a={} done. Best={:.1f}%".format(alpha, 100*best_acc))
    return baseline_path


def run_qat(alpha, baseline_path, epochs, lr, save_suffix=""):
    """Run all-INT8 QAT. Returns (qat_path, float_acc)."""
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
    sd = str(PROJECT_ROOT / "checkpoints" / "wknd_{}".format(save_suffix))
    Path(sd).mkdir(parents=True, exist_ok=True)

    hparams = {
        'lr': lr, 'wd': 1e-4,
        'epochs_warmup': epochs[0], 'epochs_ttq': epochs[1], 'epochs_freeze': epochs[2],
        'threshold_ratio': 0.05, 'quant_config': config,
        'baseline_weights': str(baseline_path), 'save_dir': sd,
    }
    model = train_ternary(model, train_loader, val_loader, device, hparams)

    # Measure float acc
    from quantize import apply_ttq, set_ttq_enabled
    model2 = MobileNetV1(alpha=alpha, num_classes=2)
    config2 = {}
    for n, m in model2.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            config2[n] = 'int8'
    apply_ttq(model2, config2, 0.05)
    qat_path = Path(sd) / "best_ternary.pt"
    model2.load_state_dict(torch.load(qat_path, map_location='cpu', weights_only=True), strict=False)
    set_ttq_enabled(model2, True); model2.train(False)
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            correct += (model2(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    return qat_path, 100*correct/total


def export_verify(alpha, qat_path):
    """Export and get simulated accuracy."""
    import torch
    import torch.nn as nn
    from train_baseline import MobileNetV1
    from quantize import apply_ttq, set_ttq_enabled
    from export_packed import extract_layers, generate_header, validate_packing, _verify_quantized_pipeline

    model = MobileNetV1(alpha=alpha, num_classes=2)
    config = {}
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            config[n] = 'int8'
    apply_ttq(model, config, 0.05)
    model.load_state_dict(torch.load(qat_path, map_location='cpu', weights_only=True), strict=False)
    set_ttq_enabled(model, True); model.train(False)
    layers = extract_layers(model, 0.05)
    if not validate_packing(layers):
        return -1, -1
    sim_acc = _verify_quantized_pipeline(layers) or 0.0
    total_w = sum(len(L.get('weights_int8', L.get('weights_packed', b''))) for L in layers)
    generate_header(layers, str(FIRMWARE_DIR / "main" / "model_data.h"))
    return sim_acc, total_w / 1024


def build_flash_measure():
    """Build, flash, measure. Returns (acc, lat_us, sram) or None."""
    result = subprocess.run(
        ["bash", "-c",
         "cd {} && source export.sh 2>/dev/null && cd {} && idf.py build 2>&1 && idf.py -p {} flash 2>&1".format(
             IDF_PATH, FIRMWARE_DIR, ESP_PORT)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        if 'app partition is too small' in result.stdout:
            print("[bfm] App too large")
        return None

    time.sleep(2)
    import serial
    try:
        ser = serial.Serial(ESP_PORT, 115200, timeout=2)
        ser.dtr = False; time.sleep(0.1); ser.dtr = True; time.sleep(4)
        while ser.in_waiting: ser.readline()
        ser.write(b"RUN_BENCHMARK\r\n")
        start = time.time()
        while time.time() - start < 120:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            m = re.match(r"METRIC\s+latency_us=(\d+)\s+accuracy=([\d.]+)\s+sram_free=(\d+)", line)
            if m:
                ser.close()
                return float(m.group(2)), int(m.group(1)), int(m.group(3))
        ser.close()
    except Exception as e:
        print("[bfm] {}".format(e))
    return None


def log(name, float_acc, sim_acc, dev_acc, lat_us, sram, model_kb, mins):
    header = "timestamp\tvariant\tfloat_acc\tsim_int8\tdevice_acc\tlatency_ms\tsram_free\tmodel_kb\ttrain_min\n"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lat_ms = lat_us / 1000 if lat_us > 0 else -1
    with open(RESULTS_TSV, "a") as f:
        f.write("{}\t{}\t{:.1f}\t{:.1f}\t{:.2f}\t{:.1f}\t{}\t{:.1f}\t{:.1f}\n".format(
            ts, name, float_acc, sim_acc, dev_acc, lat_ms, sram, model_kb, mins))
    print("\n=== {} === float={:.1f}% sim={:.1f}% dev={:.0f}% lat={:.0f}ms".format(
        name, float_acc, sim_acc, dev_acc*100, lat_ms))


def run_variant(name, alpha, epochs, lr):
    """Run one variant end-to-end."""
    print("\n" + "#"*60)
    print("# {} (a={} ep={} lr={})".format(name, alpha, epochs, lr))
    print("#"*60)
    t0 = time.time()

    try:
        baseline = train_kd_baseline(alpha)
        qat_path, float_acc = run_qat(alpha, baseline, epochs, lr, save_suffix=name)
        print("[v] Float: {:.1f}%".format(float_acc))

        sim_acc, model_kb = export_verify(alpha, qat_path)
        print("[v] Sim INT8: {:.1f}%".format(sim_acc))

        dev_acc, lat_us, sram = -1, -1, -1
        if sim_acc >= 50:
            r = build_flash_measure()
            if r:
                dev_acc, lat_us, sram = r
                print("[v] Device: {:.0f}% @ {:.0f}ms".format(dev_acc*100, lat_us/1000))

        log(name, float_acc, sim_acc, dev_acc, lat_us, sram, model_kb, (time.time()-t0)/60)
        return sim_acc, dev_acc
    except Exception as e:
        import traceback
        print("[v] ERROR: {}".format(e))
        traceback.print_exc()
        log(name, -1, -1, -1, -1, -1, -1, (time.time()-t0)/60)
        return -1, -1


def main():
    print("[wknd] Weekend optimization loop starting")
    print("[wknd] Stop time: {} Berlin".format(STOP_TIME))
    print("[wknd] Results: {}".format(RESULTS_TSV))

    sweep = 0
    best_sim = 0
    best_dev = 0
    best_config = None

    # Sweep 1: Systematic search over alpha and LR
    sweep += 1
    print("\n" + "="*60)
    print("SWEEP {}: Alpha x LR grid".format(sweep))
    print("="*60)

    configs = [
        # (name, alpha, epochs, lr)
        ("s{}_a025_lr5e4".format(sweep), 0.25, (15, 50, 15), 5e-4),
        ("s{}_a025_lr3e4".format(sweep), 0.25, (15, 50, 15), 3e-4),
        ("s{}_a025_lr1e3".format(sweep), 0.25, (15, 50, 15), 1e-3),
        ("s{}_a035_lr5e4".format(sweep), 0.35, (15, 50, 15), 5e-4),
        ("s{}_a035_lr3e4".format(sweep), 0.35, (15, 50, 15), 3e-4),
        ("s{}_a035_lr1e3".format(sweep), 0.35, (15, 50, 15), 1e-3),
    ]

    for name, alpha, epochs, lr in configs:
        if datetime.now() >= STOP_TIME:
            print("[wknd] Stop time reached")
            break
        sim, dev = run_variant(name, alpha, epochs, lr)
        if sim > best_sim:
            best_sim = sim
            best_config = (alpha, epochs, lr)
        if dev > best_dev:
            best_dev = dev

    # Sweep 2+: Zoom in on best config with variations
    while datetime.now() < STOP_TIME:
        sweep += 1
        print("\n" + "="*60)
        print("SWEEP {}: Refine around best (sim={:.1f}% dev={:.0f}%)".format(
            sweep, best_sim, best_dev*100 if best_dev > 0 else -1))
        print("="*60)

        if best_config is None:
            best_config = (0.25, (15, 50, 15), 5e-4)

        ba, be, blr = best_config

        # Try: longer training, different LR multipliers, different warmup
        refinements = [
            ("s{}_best_long".format(sweep), ba, (20, 80, 20), blr),
            ("s{}_best_lr_up".format(sweep), ba, be, blr * 1.5),
            ("s{}_best_lr_down".format(sweep), ba, be, blr * 0.67),
            ("s{}_best_warmup30".format(sweep), ba, (30, be[1], be[2]), blr),
            ("s{}_best_freeze30".format(sweep), ba, (be[0], be[1], 30), blr),
        ]

        for name, alpha, epochs, lr in refinements:
            if datetime.now() >= STOP_TIME:
                print("[wknd] Stop time reached")
                break
            sim, dev = run_variant(name, alpha, epochs, lr)
            if sim > best_sim:
                best_sim = sim
                best_config = (alpha, epochs, lr)
                print("[wknd] NEW BEST: sim={:.1f}%".format(best_sim))
            if dev > best_dev and dev > 0:
                best_dev = dev
                print("[wknd] NEW BEST DEVICE: {:.0f}%".format(best_dev*100))

    print("\n" + "="*60)
    print("[wknd] WEEKEND COMPLETE")
    print("[wknd] Best simulated: {:.1f}%".format(best_sim))
    print("[wknd] Best on-device: {:.0f}%".format(best_dev*100 if best_dev > 0 else -1))
    print("[wknd] Best config: {}".format(best_config))
    if RESULTS_TSV.exists():
        print("\nAll results:")
        print(RESULTS_TSV.read_text())


if __name__ == "__main__":
    main()
