#!/usr/bin/env python3
"""Build, flash, and measure ESP32-S3 firmware. Returns optimization score.

Called by autoresearch's prepare.py to evaluate model+kernel configurations.

Usage:
    python flash_and_measure.py --port /dev/cu.usbserial-*
    python flash_and_measure.py --port /dev/ttyUSB0 --timeout 120
"""

import argparse
import re
import subprocess
import sys
import time

import serial


FIRMWARE_DIR = "firmware"
METRIC_RE = re.compile(
    r"METRIC\s+latency_us=(\d+)\s+accuracy=([\d.]+)\s+sram_free=(\d+)"
)


def run_build():
    """Run idf.py build. Returns True on success."""
    print("[harness] Building firmware...")
    result = subprocess.run(
        ["idf.py", "build"],
        cwd=FIRMWARE_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"[harness] Build FAILED:\n{result.stderr[-500:]}")
        return False
    print("[harness] Build OK")
    return True


def run_flash(port):
    """Flash firmware. Returns True on success."""
    print(f"[harness] Flashing to {port}...")
    result = subprocess.run(
        ["idf.py", "-p", port, "flash"],
        cwd=FIRMWARE_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"[harness] Flash FAILED:\n{result.stderr[-500:]}")
        return False
    print("[harness] Flash OK")
    return True


def collect_metrics(port, timeout=120, baudrate=115200):
    """Open serial, send RUN_BENCHMARK, collect METRIC lines.

    Returns dict with latency_us, accuracy, sram_free or None on failure.
    """
    print(f"[harness] Collecting metrics from {port}...")
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
    except serial.SerialException as e:
        print(f"[harness] Serial open failed: {e}")
        return None

    # Wait for READY
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line == "READY":
            ready = True
            break

    if not ready:
        print("[harness] Timeout waiting for READY")
        ser.close()
        return None

    # Send benchmark command
    ser.write(b"RUN_BENCHMARK\n")
    ser.flush()

    # Collect METRIC line
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        match = METRIC_RE.search(line)
        if match:
            ser.close()
            return {
                "latency_us": int(match.group(1)),
                "accuracy": float(match.group(2)),
                "sram_free": int(match.group(3)),
            }

    print("[harness] Timeout waiting for METRIC")
    ser.close()
    return None


def compute_score(metrics):
    """Compute optimization scalar from metrics.

    Score = 1000 / latency_ms (i.e., FPS * 1000)
    Gated by: accuracy >= 0.92 AND sram_free > 0
    """
    if metrics is None:
        return -1.0

    latency_us = metrics["latency_us"]
    accuracy = metrics["accuracy"]
    sram_free = metrics["sram_free"]

    # SRAM budget exceeded
    if sram_free <= 0:
        print(f"[harness] SRAM budget exceeded (free={sram_free})")
        return -1.0

    # Accuracy gate
    if accuracy < 0.90:
        score = accuracy * 0.1  # heavily penalized
        print(f"[harness] Accuracy below gate: {accuracy:.4f} -> score={score:.6f}")
        return score

    # Normal scoring: maximize FPS
    latency_ms = latency_us / 1000.0
    score = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    print(f"[harness] latency={latency_ms:.1f}ms accuracy={accuracy:.4f} "
          f"sram_free={sram_free} -> score={score:.6f}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        if not run_build():
            print(f"SCORE: -1.000000")
            sys.exit(1)

    if not args.skip_flash:
        if not run_flash(args.port):
            print(f"SCORE: -1.000000")
            sys.exit(1)

    # Brief pause for device to boot after flash
    time.sleep(2)

    metrics = collect_metrics(args.port, args.timeout)
    score = compute_score(metrics)
    print(f"SCORE: {score:.6f}")


if __name__ == "__main__":
    main()
