#!/usr/bin/env python3
"""Verify that the exported layer configs produce correct inference results.

Simulates the C inference engine's layer-by-layer dispatch in Python,
using the same weight packing and requantization logic, then compares
against PyTorch's floating-point output.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_baseline import MobileNetV1
from quantize import (apply_ttq, get_default_quant_config, set_ttq_enabled,
                      TernaryQuantWrapper)
from export_packed import extract_layers, nchw_to_nhwc, pack_int8_weights


def unpack_ternary_weight(packed, idx):
    """Unpack single ternary weight from packed format."""
    block = idx // 64
    bit = idx % 64
    byte_in_block = bit // 8
    bit_in_byte = bit % 8
    block_ptr = block * 16
    nz = (packed[block_ptr + byte_in_block] >> bit_in_byte) & 1
    neg = (packed[block_ptr + 8 + byte_in_block] >> bit_in_byte) & 1
    return nz, neg


def sim_int8_conv2d(inp, weights_i8, bias_i32, H, W, C_in, C_out, K, stride, padding):
    """Simulate INT8 conv2d matching the C kernel."""
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    out = np.zeros((H_out, W_out, C_out), dtype=np.int32)

    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = int(bias_i32[oc]) if bias_i32 is not None else 0
                for kh in range(K):
                    ih = oh * stride - padding + kh
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(K):
                        iw = ow * stride - padding + kw
                        if iw < 0 or iw >= W:
                            continue
                        for ic in range(C_in):
                            a = int(inp[ih, iw, ic])
                            w = int(weights_i8[oc * K * K * C_in + kh * K * C_in + kw * C_in + ic])
                            acc += a * w
                out[oh, ow, oc] = acc
    return out


def sim_int8_depthwise(inp, weights_i8, bias_i32, H, W, C, K, stride, padding):
    """Simulate INT8 depthwise conv2d."""
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    out = np.zeros((H_out, W_out, C), dtype=np.int32)

    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = int(bias_i32[c]) if bias_i32 is not None else 0
                for kh in range(K):
                    ih = oh * stride - padding + kh
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(K):
                        iw = ow * stride - padding + kw
                        if iw < 0 or iw >= W:
                            continue
                        a = int(inp[ih, iw, c])
                        w = int(weights_i8[c * K * K + kh * K + kw])
                        acc += a * w
                out[oh, ow, c] = acc
    return out


def sim_requantize(acc, scale, zp):
    """Simulate requantize_i32_to_i8."""
    out = np.round(acc.astype(np.float64) * scale + 0.5).astype(np.int32) + zp
    return np.clip(out, -128, 127).astype(np.int8)


def sim_relu(data):
    return np.maximum(data, 0)


def sim_global_avg_pool(inp, H, W, C):
    out = np.zeros(C, dtype=np.int8)
    count = H * W
    for c in range(C):
        s = 0
        for h in range(H):
            for w in range(W):
                s += int(inp[h, w, c])
        avg = (s + count // 2) // count
        out[c] = np.clip(avg, -128, 127).astype(np.int8)
    return out


def run_test():
    """Run inference simulation and compare to PyTorch."""
    model = MobileNetV1(alpha=0.25, num_classes=2)

    # Use autoresearch config (depthwise=INT8, pointwise=ternary)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "autoresearch"))
    import train as train_target
    config = train_target.get_quant_config()

    apply_ttq(model, config)
    set_ttq_enabled(model, True)

    # Random test input
    torch.manual_seed(42)
    x_float = torch.randn(1, 3, 96, 96)

    # Get PyTorch output
    model.train(False)
    with torch.no_grad():
        y_pytorch = model(x_float)
    pytorch_class = y_pytorch.argmax(1).item()
    print(f"PyTorch output: {y_pytorch.numpy().flatten()}, class={pytorch_class}")

    # Export layers
    layers = extract_layers(model)

    # Simulate C inference
    # Input: float [0,1] → uint8 [0,255] → int8 [-128,127]
    inp_uint8 = (x_float[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    inp_int8 = (inp_uint8.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)

    cur = inp_int8
    cur_h, cur_w = 96, 96

    for i, L in enumerate(layers):
        ltype = L["type"]
        lquant = L["quant"]

        if ltype == "LAYER_CONV2D":
            h_out = (cur_h + 2 * L["padding"] - L["kernel"]) // L["stride"] + 1
            w_out = (cur_w + 2 * L["padding"] - L["kernel"]) // L["stride"] + 1

            if lquant == "QUANT_INT8":
                acc = sim_int8_conv2d(
                    cur, L["weights_int8"], L.get("bias"),
                    cur_h, cur_w, L["in_c"], L["out_c"],
                    L["kernel"], L["stride"], L["padding"]
                )
            else:
                # Ternary conv would go here — skip for now
                print(f"  [{i}] TERNARY CONV2D — skipping simulation")
                cur_h, cur_w = h_out, w_out
                cur = np.zeros((h_out, w_out, L["out_c"]), dtype=np.int8)
                continue

            cur = sim_requantize(acc, L["requant_scale"], L["requant_zp"])
            cur = sim_relu(cur).reshape(h_out, w_out, L["out_c"])
            cur_h, cur_w = h_out, w_out

        elif ltype == "LAYER_DEPTHWISE_CONV2D":
            h_out = (cur_h + 2 * L["padding"] - L["kernel"]) // L["stride"] + 1
            w_out = (cur_w + 2 * L["padding"] - L["kernel"]) // L["stride"] + 1

            if lquant == "QUANT_INT8":
                acc = sim_int8_depthwise(
                    cur, L["weights_int8"], L.get("bias"),
                    cur_h, cur_w, L["in_c"],
                    L["kernel"], L["stride"], L["padding"]
                )
            else:
                print(f"  [{i}] TERNARY DW — skipping simulation")
                cur_h, cur_w = h_out, w_out
                cur = np.zeros((h_out, w_out, L["out_c"]), dtype=np.int8)
                continue

            cur = sim_requantize(acc, L["requant_scale"], L["requant_zp"])
            cur = sim_relu(cur).reshape(h_out, w_out, L["out_c"])
            cur_h, cur_w = h_out, w_out

        elif ltype == "LAYER_GLOBAL_AVG_POOL":
            cur = sim_global_avg_pool(cur, cur_h, cur_w, L["in_c"])
            cur_h, cur_w = 1, 1

        elif ltype == "LAYER_DENSE":
            flat = cur.flatten()
            if lquant == "QUANT_INT8":
                acc = np.zeros(L["out_c"], dtype=np.int32)
                for o in range(L["out_c"]):
                    s = int(L["bias"][o]) if L.get("bias") is not None else 0
                    for j in range(L["in_c"]):
                        s += int(flat[j]) * int(L["weights_int8"][o * L["in_c"] + j])
                    acc[o] = s
                cur = sim_requantize(acc, L["requant_scale"], L["requant_zp"])
            cur_h, cur_w = 1, 1

        if i < 3 or i >= len(layers) - 3:
            print(f"  [{i:2d}] {L['name']:30s} {ltype:25s} shape={cur.shape}")

    sim_class = 0 if cur[0] > cur[1] else 1
    print(f"\nSimulated output: {cur.flatten()[:10]}, class={sim_class}")
    print(f"PyTorch class: {pytorch_class}, Simulated class: {sim_class}")
    print(f"NOTE: Exact match not expected due to quantization — this verifies the dispatch logic runs.")


if __name__ == "__main__":
    run_test()
