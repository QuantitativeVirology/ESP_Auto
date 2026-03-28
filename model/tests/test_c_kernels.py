#!/usr/bin/env python3
"""Cross-validate C reference kernels against Python implementation.

Compiles ternary_conv2d_ref.c and ternary_dense_ref.c as a native shared
library, then runs identical inputs through both Python and C to verify
bit-exact match.
"""

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
KERNEL_SRC = PROJECT_ROOT / "firmware" / "components" / "ternary_kernels"


def build_native_lib():
    """Compile C reference kernels as a native shared library."""
    src_files = [
        KERNEL_SRC / "src" / "ternary_conv2d_ref.c",
        KERNEL_SRC / "src" / "ternary_dense_ref.c",
        KERNEL_SRC / "src" / "int8_kernels.c",
    ]
    include_dir = KERNEL_SRC / "include"
    out_dir = Path(tempfile.mkdtemp())
    lib_path = out_dir / "libkernels.dylib"

    cmd = [
        "cc", "-shared", "-fPIC", "-O2",
        f"-I{include_dir}",
        "-o", str(lib_path),
    ] + [str(f) for f in src_files]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)

    return ctypes.CDLL(str(lib_path))


def pack_ternary_weights_np(flat_weights, threshold_ratio=0.05):
    """Pack weights matching the C unpacking format."""
    threshold = threshold_ratio * np.abs(flat_weights).max()
    pos_mask = flat_weights > threshold
    neg_mask = flat_weights < -threshold
    non_zero = pos_mask | neg_mask

    pad_len = (64 - len(flat_weights) % 64) % 64
    if pad_len > 0:
        non_zero = np.concatenate([non_zero, np.zeros(pad_len, dtype=bool)])
        neg_mask = np.concatenate([neg_mask, np.zeros(pad_len, dtype=bool)])

    packed = bytearray()
    for i in range(0, len(non_zero), 64):
        for byte_idx in range(8):
            byte_val = 0
            for bit in range(8):
                if non_zero[i + byte_idx * 8 + bit]:
                    byte_val |= (1 << bit)
            packed.append(byte_val)
        for byte_idx in range(8):
            byte_val = 0
            for bit in range(8):
                if neg_mask[i + byte_idx * 8 + bit]:
                    byte_val |= (1 << bit)
            packed.append(byte_val)

    return bytes(packed)


def py_ternary_conv2d(inp, packed_weights, H, W, C_in, C_out, K, stride, padding):
    """Python reference matching the C implementation."""
    C_in_padded = (C_in + 63) & ~63
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    output = np.zeros((H_out, W_out, C_out), dtype=np.int32)

    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = 0
                for kh in range(K):
                    ih = oh * stride - padding + kh
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(K):
                        iw = ow * stride - padding + kw
                        if iw < 0 or iw >= W:
                            continue
                        for ic in range(C_in):
                            w_idx = oc * K * K * C_in_padded + kh * K * C_in_padded + kw * C_in_padded + ic
                            block = w_idx // 64
                            bit = w_idx % 64
                            byte_in_block = bit // 8
                            bit_in_byte = bit % 8
                            block_ptr = block * 16
                            nz = (packed_weights[block_ptr + byte_in_block] >> bit_in_byte) & 1
                            neg = (packed_weights[block_ptr + 8 + byte_in_block] >> bit_in_byte) & 1
                            if nz:
                                a = int(inp[ih * W * C_in + iw * C_in + ic])
                                acc += -a if neg else a
                output[oh, ow, oc] = acc
    return output


def py_ternary_dense(inp, packed_weights, N_in, N_out):
    """Python reference matching the C dense implementation."""
    N_in_padded = (N_in + 63) & ~63
    output = np.zeros(N_out, dtype=np.int32)

    for o in range(N_out):
        acc = 0
        for i in range(N_in):
            w_idx = o * N_in_padded + i
            block = w_idx // 64
            bit = w_idx % 64
            byte_in_block = bit // 8
            bit_in_byte = bit % 8
            block_ptr = block * 16
            nz = (packed_weights[block_ptr + byte_in_block] >> bit_in_byte) & 1
            neg = (packed_weights[block_ptr + 8 + byte_in_block] >> bit_in_byte) & 1
            if nz:
                a = int(inp[i])
                acc += -a if neg else a
        output[o] = acc
    return output


def test_conv2d(lib, H, W, C_in, C_out, K, stride, padding, seed=42):
    """Test conv2d: C vs Python, bit-exact."""
    rng = np.random.RandomState(seed)
    C_in_padded = (C_in + 63) & ~63
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    inp = rng.randint(-128, 128, size=H * W * C_in, dtype=np.int8)

    # Random float weights → pack as ternary
    float_weights = rng.randn(C_out * K * K * C_in_padded).astype(np.float32)
    packed = pack_ternary_weights_np(float_weights)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)

    # Python
    py_out = py_ternary_conv2d(inp, packed, H, W, C_in, C_out, K, stride, padding)

    # C
    out_size = H_out * W_out * C_out
    c_out = np.zeros(out_size, dtype=np.int32)

    lib.ternary_conv2d_ref(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        packed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_float(1.0), ctypes.c_float(1.0),
        ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(C_in),
        ctypes.c_int(C_out), ctypes.c_int(K),
        ctypes.c_int(stride), ctypes.c_int(padding),
    )

    c_out_reshaped = c_out.reshape(H_out, W_out, C_out)
    if np.array_equal(py_out, c_out_reshaped):
        print(f"  PASS conv2d [{H}x{W}x{C_in}->{C_out} K={K} s={stride} p={padding}]")
        return True
    else:
        diff = np.abs(py_out - c_out_reshaped)
        print(f"  FAIL conv2d [{H}x{W}x{C_in}->{C_out} K={K} s={stride} p={padding}] "
              f"max_diff={diff.max()} at {np.unravel_index(diff.argmax(), diff.shape)}")
        return False


def test_dense(lib, N_in, N_out, seed=42):
    """Test dense: C vs Python, bit-exact."""
    rng = np.random.RandomState(seed)
    N_in_padded = (N_in + 63) & ~63

    inp = rng.randint(-128, 128, size=N_in, dtype=np.int8)
    float_weights = rng.randn(N_out * N_in_padded).astype(np.float32)
    packed = pack_ternary_weights_np(float_weights)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)

    py_out = py_ternary_dense(inp, packed, N_in, N_out)

    c_out = np.zeros(N_out, dtype=np.int32)
    lib.ternary_dense_ref(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        packed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_float(1.0), ctypes.c_float(1.0),
        ctypes.c_int(N_in), ctypes.c_int(N_out),
    )

    if np.array_equal(py_out, c_out):
        print(f"  PASS dense [{N_in}->{N_out}]")
        return True
    else:
        diff = np.abs(py_out - c_out)
        print(f"  FAIL dense [{N_in}->{N_out}] max_diff={diff.max()}")
        return False


def main():
    print("Building native kernel library...")
    lib = build_native_lib()
    print("Build OK\n")

    failures = 0
    print("=== Conv2d Tests ===")
    for H, W, Ci, Co, K, s, p in [
        (8, 8, 64, 16, 1, 1, 0),
        (8, 8, 64, 32, 3, 1, 1),
        (12, 12, 64, 64, 3, 2, 1),
        (6, 6, 128, 128, 1, 1, 0),
    ]:
        if not test_conv2d(lib, H, W, Ci, Co, K, s, p):
            failures += 1

    print("\n=== Dense Tests ===")
    for Ni, No in [(64, 16), (128, 64), (256, 2)]:
        if not test_dense(lib, Ni, No):
            failures += 1

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
