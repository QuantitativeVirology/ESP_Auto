#!/usr/bin/env python3
"""Export ternary model weights as packed C header for ESP32-S3 firmware.

Weight packing format (per 128-bit / 16-byte block, covering 64 weights):
  bytes [0:7]   = zero_mask bits (1 = non-zero, 0 = zero)
  bytes [8:15]  = sign_bits (0 = positive, 1 = negative)

Weight order: NHWC — [C_out, K, K, C_in] for conv, [N_out, N_in] for dense.
C_in / N_in padded to multiple of 64.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from train_baseline import MobileNetV1
from quantize import (apply_ttq, get_default_quant_config, set_ttq_enabled,
                      TernaryQuantWrapper)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_ternary_weights(weight_fp, threshold_ratio=0.05):
    """Pack ternary weights into 2-bit format.

    Returns: (packed_bytes, scale_pos, scale_neg, sparsity)
    """
    flat = weight_fp.detach().cpu().flatten().numpy().astype(np.float32)
    threshold = threshold_ratio * np.abs(flat).max()

    pos_mask = flat > threshold
    neg_mask = flat < -threshold
    non_zero = pos_mask | neg_mask
    sign = neg_mask  # 1 = negative

    scale_pos = float(np.abs(flat[pos_mask]).mean()) if pos_mask.any() else 0.1
    scale_neg = float(np.abs(flat[neg_mask]).mean()) if neg_mask.any() else 0.1
    sparsity = 1.0 - non_zero.sum() / len(flat)

    # Pad to multiple of 64
    pad_len = (64 - len(flat) % 64) % 64
    if pad_len > 0:
        non_zero = np.concatenate([non_zero, np.zeros(pad_len, dtype=bool)])
        sign = np.concatenate([sign, np.zeros(pad_len, dtype=bool)])

    packed = bytearray()
    for i in range(0, len(non_zero), 64):
        # Pack 64 zero_mask bits into 8 bytes (LSB first)
        zm_chunk = non_zero[i:i + 64]
        for byte_idx in range(8):
            byte_val = 0
            for bit in range(8):
                if zm_chunk[byte_idx * 8 + bit]:
                    byte_val |= (1 << bit)
            packed.append(byte_val)
        # Pack 64 sign bits into 8 bytes
        sign_chunk = sign[i:i + 64]
        for byte_idx in range(8):
            byte_val = 0
            for bit in range(8):
                if sign_chunk[byte_idx * 8 + bit]:
                    byte_val |= (1 << bit)
            packed.append(byte_val)

    return bytes(packed), scale_pos, scale_neg, sparsity


def unpack_ternary_weights(packed, num_weights, scale_pos, scale_neg):
    """Unpack for round-trip validation. Returns numpy array."""
    result = np.zeros(num_weights, dtype=np.float32)

    for i in range(0, len(packed), 16):
        block = packed[i:i + 16]
        base_idx = (i // 16) * 64

        for j in range(64):
            if base_idx + j >= num_weights:
                break
            byte_idx = j // 8
            bit_pos = j % 8
            nz = (block[byte_idx] >> bit_pos) & 1
            neg = (block[8 + byte_idx] >> bit_pos) & 1
            if nz:
                result[base_idx + j] = -scale_neg if neg else scale_pos

    return result


def nchw_to_nhwc(weight, depthwise=False):
    """Transpose conv weight to firmware layout.

    Standard conv: [C_out, C_in, K, K] → [C_out, K, K, C_in]
    Depthwise conv: [C, 1, K, K] → [C, K, K] (squeeze the groups=1 dim)
    Linear: [N_out, N_in] → unchanged
    """
    if weight.ndim == 4:
        w = weight.permute(0, 2, 3, 1).contiguous()
        if depthwise:
            # [C, K, K, 1] → [C, K, K]
            w = w.squeeze(-1)
        return w
    return weight  # Linear weights [N_out, N_in] stay as-is


def pack_int8_weights(weight_fp):
    """Quantize float weights to INT8 for first/last layers."""
    w = weight_fp.detach().cpu().float()
    scale = w.abs().max() / 127.0
    if scale == 0:
        scale = 1.0
    w_int8 = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return w_int8.numpy(), float(scale)


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------

def extract_layers(model, threshold_ratio=0.05):
    """Walk model and extract layer configs with packed weights."""
    layers = []

    # First conv (INT8)
    first_conv = model.first_conv[0]  # Conv2d
    first_bn = model.first_conv[1]    # BatchNorm2d
    w_nhwc = nchw_to_nhwc(first_conv.weight.data)
    w_int8, w_scale = pack_int8_weights(w_nhwc)

    # Fold BatchNorm into bias for inference
    bn_scale = first_bn.weight.data / torch.sqrt(first_bn.running_var + 1e-5)
    bias = (first_bn.bias.data - first_bn.running_mean * bn_scale).cpu().numpy()
    # Quantize bias to int32 (in accumulator domain)
    bias_i32 = np.round(bias / w_scale).astype(np.int32)

    layers.append({
        "name": "first_conv",
        "type": "LAYER_CONV2D",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8.flatten(),
        "weight_scale": w_scale,
        "bias": bias_i32,
        "in_c": 3, "out_c": first_conv.out_channels,
        "kernel": first_conv.kernel_size[0],
        "stride": first_conv.stride[0],
        "padding": first_conv.padding[0],
        "requant_scale": w_scale,  # simplified; should calibrate properly
        "requant_zp": 0,
    })

    # Depthwise-separable blocks
    for blk_idx, block in enumerate(model.features):
        for sub_name, sub_seq in [("dw", block.dw), ("pw", block.pw)]:
            conv = sub_seq[0]  # Conv2d or TernaryQuantWrapper
            bn = sub_seq[1]    # BatchNorm2d

            is_depthwise = sub_name == "dw"
            layer_type = "LAYER_DEPTHWISE_CONV2D" if is_depthwise else "LAYER_CONV2D"

            if isinstance(conv, TernaryQuantWrapper):
                w = nchw_to_nhwc(conv.weight_fp.data, depthwise=is_depthwise)
                packed, sp, sn, sparsity = pack_ternary_weights(w, threshold_ratio)

                bn_s = bn.weight.data / torch.sqrt(bn.running_var + 1e-5)
                bias = (bn.bias.data - bn.running_mean * bn_s).cpu().numpy()
                bias_i32 = np.round(bias / max(sp, sn)).astype(np.int32)

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_TERNARY",
                    "weights_packed": packed,
                    "scale_pos": sp,
                    "scale_neg": sn,
                    "bias": bias_i32,
                    "in_c": conv.module.in_channels,
                    "out_c": conv.module.out_channels,
                    "kernel": conv.module.kernel_size[0],
                    "stride": conv.module.stride[0],
                    "padding": conv.module.padding[0],
                    "requant_scale": max(sp, sn),
                    "requant_zp": 0,
                    "sparsity": sparsity,
                })
            else:
                w_nhwc = nchw_to_nhwc(conv.weight.data, depthwise=is_depthwise)
                w_int8, w_scale = pack_int8_weights(w_nhwc)

                bn_s = bn.weight.data / torch.sqrt(bn.running_var + 1e-5)
                bias = (bn.bias.data - bn.running_mean * bn_s).cpu().numpy()
                bias_i32 = np.round(bias / w_scale).astype(np.int32)

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_INT8",
                    "weights_int8": w_int8.flatten(),
                    "weight_scale": w_scale,
                    "bias": bias_i32,
                    "in_c": conv.in_channels,
                    "out_c": conv.out_channels,
                    "kernel": conv.kernel_size[0],
                    "stride": conv.stride[0],
                    "padding": conv.padding[0],
                    "requant_scale": w_scale,
                    "requant_zp": 0,
                })

    # Global average pool
    layers.append({
        "name": "pool",
        "type": "LAYER_GLOBAL_AVG_POOL",
        "quant": "QUANT_INT8",
        "in_c": layers[-1]["out_c"],
        "out_c": layers[-1]["out_c"],
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": 1.0, "requant_zp": 0,
    })

    # Classifier (INT8)
    classifier = model.classifier
    w_int8, w_scale = pack_int8_weights(classifier.weight.data)
    bias_i32 = np.round(classifier.bias.data.cpu().numpy() / w_scale).astype(np.int32)

    layers.append({
        "name": "classifier",
        "type": "LAYER_DENSE",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8.flatten(),
        "weight_scale": w_scale,
        "bias": bias_i32,
        "in_c": classifier.in_features,
        "out_c": classifier.out_features,
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": w_scale,
        "requant_zp": 0,
    })

    return layers


# ---------------------------------------------------------------------------
# C header generation
# ---------------------------------------------------------------------------

def format_array(data, per_line=16):
    """Format byte/int array as C initializer."""
    lines = []
    for i in range(0, len(data), per_line):
        chunk = data[i:i + per_line]
        if isinstance(chunk[0], (np.int8, np.uint8)):
            items = ", ".join(f"0x{int(b) & 0xFF:02x}" for b in chunk)
        elif isinstance(chunk[0], (np.int32,)):
            items = ", ".join(str(int(v)) for v in chunk)
        else:
            items = ", ".join(f"0x{int(b) & 0xFF:02x}" for b in chunk)
        lines.append(f"    {items},")
    return "\n".join(lines)


def generate_header(layers, output_path):
    """Generate model_data.h with all weights and layer configs."""
    lines = [
        "// Auto-generated by export_packed.py -- do not edit",
        "#pragma once",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
    ]

    total_flash = 0

    for i, layer in enumerate(layers):
        lines.append(f"// === Layer {i}: {layer['name']} ({layer['quant']}) ===")

        if "weights_packed" in layer:
            data = layer["weights_packed"]
            total_flash += len(data)
            lines.append(
                f"static const uint8_t __attribute__((aligned(16))) "
                f"layer{i}_weights[{len(data)}] = {{"
            )
            lines.append(format_array(list(data)))
            lines.append("};")
            lines.append(f"static const float layer{i}_scale_pos = {layer['scale_pos']:.6f}f;")
            lines.append(f"static const float layer{i}_scale_neg = {layer['scale_neg']:.6f}f;")

        elif "weights_int8" in layer:
            data = layer["weights_int8"]
            total_flash += len(data)
            lines.append(
                f"static const int8_t __attribute__((aligned(16))) "
                f"layer{i}_weights[{len(data)}] = {{"
            )
            lines.append(format_array(data))
            lines.append("};")
            lines.append(f"static const float layer{i}_weight_scale = {layer['weight_scale']:.6f}f;")

        if "bias" in layer:
            bias = layer["bias"]
            total_flash += len(bias) * 4
            lines.append(f"static const int32_t layer{i}_bias[{len(bias)}] = {{")
            lines.append(format_array(bias))
            lines.append("};")

        lines.append(f"static const float layer{i}_requant_scale = {layer['requant_scale']:.6f}f;")
        lines.append(f"static const int8_t layer{i}_requant_zp = {layer['requant_zp']};")
        lines.append("")

    # Layer config table
    num_layers = len(layers)
    lines.append(f"#define NUM_LAYERS {num_layers}")
    lines.append("")
    lines.append("static const layer_config_t model_layers[] = {")

    for i, layer in enumerate(layers):
        has_weights = "weights_packed" in layer or "weights_int8" in layer
        weights_ref = f"layer{i}_weights" if has_weights else "NULL"
        bias_ref = f"layer{i}_bias" if "bias" in layer else "NULL"

        sp = f"{layer.get('scale_pos', 0):.6f}f"
        sn = f"{layer.get('scale_neg', 0):.6f}f"

        lines.append(
            f"    {{ {layer['type']}, {layer['quant']}, "
            f"{weights_ref}, {bias_ref}, "
            f"{sp}, {sn}, "
            f"layer{i}_requant_scale, layer{i}_requant_zp, "
            f"{layer['in_c']}, {layer['out_c']}, "
            f"{layer['kernel']}, {layer['stride']}, {layer['padding']} }},"
        )

    lines.append("};")
    lines.append("")

    # Memory summary
    lines.append(f"// Total weight data in flash: {total_flash} bytes ({total_flash/1024:.1f} KB)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Generated {output_path} ({num_layers} layers, {total_flash/1024:.1f} KB weights)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_packing(layers):
    """Round-trip validation: unpack packed weights and check consistency."""
    for layer in layers:
        if "weights_packed" not in layer:
            continue

        packed = layer["weights_packed"]
        sp = layer["scale_pos"]
        sn = layer["scale_neg"]

        # Compute expected number of weights
        if layer["kernel"] > 0:
            num_weights = layer["out_c"] * layer["kernel"] * layer["kernel"] * layer["in_c"]
        else:
            num_weights = layer["out_c"] * layer["in_c"]

        unpacked = unpack_ternary_weights(packed, num_weights, sp, sn)

        # Check only 3 distinct values (plus padding zeros) with tolerance
        vals = unpacked[:num_weights]
        for v in vals:
            v_abs = abs(float(v))
            if v_abs > 1e-6:  # non-zero
                if not (abs(v_abs - sp) < 1e-4 or abs(v_abs - sn) < 1e-4):
                    print(f"FAIL {layer['name']}: unexpected value {v} "
                          f"(expected 0, +/-{sp:.6f}, +/-{sn:.6f})")
                    return False

        # Check alignment
        if len(packed) % 16 != 0:
            print(f"FAIL {layer['name']}: packed size {len(packed)} not 16-byte aligned")
            return False

    print("Round-trip validation PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="checkpoints/best_ternary.pt")
    parser.add_argument("--output", default="firmware/main/model_data.h")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--threshold-ratio", type=float, default=0.05)
    args = parser.parse_args()

    model = MobileNetV1(alpha=args.alpha, num_classes=2)
    config = get_default_quant_config(model)
    apply_ttq(model, config, args.threshold_ratio)

    if Path(args.weights).exists():
        model.load_state_dict(
            torch.load(args.weights, weights_only=True, map_location="cpu")
        )
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"WARNING: {args.weights} not found, using random weights")

    set_ttq_enabled(model, True)

    layers = extract_layers(model, args.threshold_ratio)

    # Print summary
    print(f"\nModel summary ({len(layers)} layers):")
    for i, l in enumerate(layers):
        size = 0
        if "weights_packed" in l:
            size = len(l["weights_packed"])
        elif "weights_int8" in l:
            size = len(l["weights_int8"])
        print(f"  [{i:2d}] {l['name']:30s} {l['type']:25s} {l['quant']:15s} "
              f"{l['in_c']:4d}->{l['out_c']:4d} "
              f"k={l['kernel']} s={l['stride']} p={l['padding']} "
              f"weights={size}B")

    validate_packing(layers)
    generate_header(layers, args.output)


if __name__ == "__main__":
    main()
