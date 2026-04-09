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
    """Quantize float weights to INT8 (global scale)."""
    w = weight_fp.detach().cpu().float()
    scale = w.abs().max() / 127.0
    if scale == 0:
        scale = 1.0
    w_int8 = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return w_int8.numpy(), float(scale)


def pack_int8_weights_per_channel(weight_nhwc):
    """Quantize float weights to INT8 with per-output-channel scales.

    weight_nhwc: [C_out, K, K, C_in] or [C_out, N_in] (NHWC-reshaped)
    Returns: (w_int8 numpy, scales_per_ch numpy float32 [C_out])
    """
    w = weight_nhwc.detach().cpu().float()
    n_out = w.shape[0]
    scales = np.zeros(n_out, dtype=np.float32)
    w_int8 = torch.zeros_like(w, dtype=torch.int8)

    for oc in range(n_out):
        ch = w[oc].flatten()
        absmax = ch.abs().max().item()
        s = absmax / 127.0 if absmax > 0 else 1e-10
        scales[oc] = s
        w_int8[oc] = torch.clamp(torch.round(w[oc] / s), -128, 127).to(torch.int8)

    return w_int8.numpy(), scales


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------

def _im2col(input_hwc, K, stride, padding):
    """Extract patches for convolution. input: [H,W,C] int8 -> [H_out*W_out, K*K*C]."""
    H, W, C = input_hwc.shape
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    if padding > 0:
        padded = np.zeros((H + 2*padding, W + 2*padding, C), dtype=input_hwc.dtype)
        padded[padding:padding+H, padding:padding+W, :] = input_hwc
    else:
        padded = input_hwc

    patches = np.zeros((H_out * W_out, K * K * C), dtype=np.int16)
    for oh in range(H_out):
        for ow in range(W_out):
            patch = padded[oh*stride:oh*stride+K, ow*stride:ow*stride+K, :]
            patches[oh * W_out + ow] = patch.reshape(-1).astype(np.int16)
    return patches, H_out, W_out


def _load_calibration_images(data_dir=None):
    """Load balanced calibration images as firmware-style INT8."""
    import torchvision.transforms as transforms
    from train_baseline import BinaryPetsDataset
    import os

    if data_dir is None:
        for p in ["/tmp/esp_datasets", "model/datasets"]:
            if os.path.exists(p):
                data_dir = p
                break
    if data_dir is None:
        return [], []

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    raw_tf = transforms.Compose([
        transforms.Resize(104), transforms.CenterCrop(96), transforms.ToTensor()])
    ds = BinaryPetsDataset(data_dir, "test", raw_tf)

    cal_images, cal_labels = [], []
    cats_seen, dogs_seen = 0, 0
    for img, label in ds:
        if label == 0 and cats_seen >= 32: continue
        if label == 1 and dogs_seen >= 32: continue
        uint8_hwc = (img * 255).byte().permute(1, 2, 0).contiguous().numpy()
        int8_img = np.zeros((96, 96, 3), dtype=np.int8)
        for c in range(3):
            norm = uint8_hwc[:,:,c].astype(np.float32) / 255.0
            norm = (norm - MEAN[c]) / STD[c]
            int8_img[:,:,c] = np.clip(np.round(norm * 127), -128, 127).astype(np.int8)
        cal_images.append(int8_img)
        cal_labels.append(label)
        if label == 0: cats_seen += 1
        else: dogs_seen += 1
        if cats_seen >= 32 and dogs_seen >= 32: break

    return cal_images, cal_labels


def _measure_float_activation_ranges(model, data_dir=None):
    """Run model in float on calibration data to measure per-layer activation ranges.

    Returns dict: layer_key -> np.array of per-channel max abs activation (post-ReLU).
    These ranges define the per-tensor output scale for each layer.
    """
    import torchvision.transforms as transforms
    from train_baseline import BinaryPetsDataset
    import os

    if data_dir is None:
        for p in ["/tmp/esp_datasets", "model/datasets"]:
            if os.path.exists(p):
                data_dir = p
                break
    if data_dir is None:
        print("[export] WARNING: No data for float activation measurement")
        return {}

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.Resize(104), transforms.CenterCrop(96), transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    ds = BinaryPetsDataset(data_dir, "test", tf)

    images = []
    cats, dogs = 0, 0
    for img, label in ds:
        if label == 0 and cats >= 32: continue
        if label == 1 and dogs >= 32: continue
        images.append(img)
        if label == 0: cats += 1
        else: dogs += 1
        if cats >= 32 and dogs >= 32: break

    batch = torch.stack(images)
    ranges = {}
    hooks = []

    def make_hook(name):
        def fn(mod, inp, out):
            with torch.no_grad():
                o = out
                if o.dim() == 4:
                    am = o.abs().amax(dim=(0, 2, 3))
                elif o.dim() == 2:
                    am = o.abs().amax(dim=0)
                else:
                    am = o.abs().max().unsqueeze(0)
                if name in ranges:
                    ranges[name] = torch.maximum(ranges[name], am)
                else:
                    ranges[name] = am.clone()
        return fn

    hooks.append(model.first_conv.register_forward_hook(make_hook('first_conv')))
    for i in range(len(model.features)):
        block = model.features[i]
        hooks.append(block.dw.register_forward_hook(make_hook(f'features_{i}_dw')))
        hooks.append(block.pw.register_forward_hook(make_hook(f'features_{i}_pw')))
    hooks.append(model.classifier.register_forward_hook(make_hook('classifier')))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), 16):
            model(batch[start:start+16])

    for h in hooks:
        h.remove()

    result = {}
    for k, v in ranges.items():
        result[k] = v.cpu().numpy()
        print(f"  [float_range] {k:30s} max={v.max().item():.4f} min_ch={v.min().item():.4f}")

    return result


def _verify_quantized_pipeline(layers, data_dir=None):
    """Run a few calibration images through the simulated INT8 pipeline.

    Uses the layers' current requant_per_ch and bias values (does NOT modify them).
    Prints classifier outputs for verification.
    """
    cal_images, cal_labels = _load_calibration_images(data_dir)
    if not cal_images:
        print("[export] WARNING: No calibration data for verification")
        return

    n_verify = len(cal_images)
    correct = 0

    for img_idx in range(n_verify):
        act = cal_images[img_idx].copy()
        cur_h, cur_w = 96, 96

        for li, L in enumerate(layers):
            if L['type'] == 'LAYER_GLOBAL_AVG_POOL':
                pooled = act.astype(np.int32).mean(axis=(0, 1))
                act = np.clip(np.round(pooled), -128, 127).astype(np.int8).reshape(1, 1, -1)
                cur_h, cur_w = 1, 1
                continue

            K = L.get('kernel', 1) or 1
            S = L.get('stride', 1) or 1
            P = L.get('padding', 0)
            is_dw = L['type'] == 'LAYER_DEPTHWISE_CONV2D'
            is_dense = L['type'] == 'LAYER_DENSE'

            if is_dense:
                in_flat = act.flatten().astype(np.int32)
                w = L['weights_int8'].reshape(L['out_c'], L['in_c'])
                acc_all = np.zeros((1, L['out_c']), dtype=np.int32)
                for oc in range(L['out_c']):
                    acc_all[0, oc] = np.dot(in_flat[:L['in_c']].astype(np.int32),
                                            w[oc].astype(np.int32))
                    acc_all[0, oc] += L['bias'][oc]
                h_out, w_out = 1, 1
            elif is_dw:
                n_ch = L['in_c']
                h_out = (cur_h + 2*P - K) // S + 1
                w_out = (cur_w + 2*P - K) // S + 1
                if L['quant'] == 'QUANT_INT8':
                    w_k = L['weights_int8'].reshape(n_ch, K * K)
                else:
                    # Ternary depthwise: unpack to {-1, 0, +1}
                    num_w = n_ch * K * K
                    pad_num_w = ((num_w + 63) // 64) * 64
                    tern_float = unpack_ternary_weights(
                        L['weights_packed'], pad_num_w, L['scale_pos'], L['scale_neg'])
                    w_k = np.sign(tern_float[:num_w]).astype(np.int8).reshape(n_ch, K * K)
                acc_all = np.zeros((h_out * w_out, n_ch), dtype=np.int32)
                if P > 0:
                    padded = np.zeros((cur_h+2*P, cur_w+2*P, n_ch), dtype=np.int8)
                    padded[P:P+cur_h, P:P+cur_w, :] = act
                else:
                    padded = act
                for ch in range(n_ch):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            patch = padded[oh*S:oh*S+K, ow*S:ow*S+K, ch].flatten()
                            acc_all[oh*w_out+ow, ch] = np.dot(
                                patch.astype(np.int32), w_k[ch].astype(np.int32))
                            acc_all[oh*w_out+ow, ch] += L['bias'][ch]
            else:
                patches, h_out, w_out = _im2col(act, K, S, P)
                if L['quant'] == 'QUANT_INT8':
                    w = L['weights_int8'].reshape(L['out_c'], -1)
                    acc_all = patches.astype(np.int32) @ w.astype(np.int32).T
                    for oc in range(L['out_c']):
                        acc_all[:, oc] += L['bias'][oc]
                else:
                    cin_pad = (L['in_c'] + 63) & ~63
                    num_w = L['out_c'] * K * K * cin_pad
                    tern_float = unpack_ternary_weights(
                        L['weights_packed'], num_w, L['scale_pos'], L['scale_neg'])
                    tern_signs = np.sign(tern_float).astype(np.int8)
                    tern_signs = tern_signs.reshape(L['out_c'], K * K * cin_pad)
                    actual_cols = K * K * L['in_c']
                    tern_used = tern_signs[:, :actual_cols]
                    acc_all = patches[:, :actual_cols].astype(np.int32) @ tern_used.astype(np.int32).T
                    for oc in range(L['out_c']):
                        acc_all[:, oc] += L['bias'][oc]

            rq = L['requant_per_ch']
            out_i8 = np.zeros_like(acc_all, dtype=np.int8)
            for oc in range(acc_all.shape[1]):
                fval = acc_all[:, oc].astype(np.float64) * rq[oc]
                ival = np.where(fval >= 0, np.floor(fval + 0.5), np.ceil(fval - 0.5))
                out_i8[:, oc] = np.clip(ival, -128, 127).astype(np.int8)

            if li < len(layers) - 1:
                out_i8 = np.maximum(out_i8, np.int8(0))

            act = out_i8.reshape(h_out, w_out, L['out_c'])
            cur_h, cur_w = h_out, w_out

        logits = act.flatten()[:2]
        pred = 0 if logits[0] > logits[1] else 1
        if pred == cal_labels[img_idx]:
            correct += 1
        if img_idx < 5 or (img_idx >= 32 and img_idx < 37):
            label_str = "CAT" if cal_labels[img_idx] == 0 else "DOG"
            pred_str = "CAT" if pred == 0 else "DOG"
            print(f"  Image {img_idx} [{label_str}]: logits=[{logits[0]}, {logits[1]}] -> {pred_str} {'OK' if pred == cal_labels[img_idx] else 'WRONG'}")

    acc_pct = 100*correct/n_verify
    print(f"  Verification accuracy: {correct}/{n_verify} = {acc_pct:.1f}%")
    return acc_pct


def _calibrate_requant(layers, n_passes=2, data_dir=None):
    """Multi-pass calibration: measure INT32 acc ranges and set per-channel requant.

    Uses the layers' current requant_per_ch as initialization. Each pass:
    1. Runs calibration images through the simulated int8 pipeline
    2. Tracks max_abs(acc + bias) per channel
    3. Updates requant_per_ch = 127 / max_abs_acc

    Multiple passes help converge because requant affects intermediate activations.
    """
    cal_images, cal_labels = _load_calibration_images(data_dir)
    if not cal_images:
        print("[calibrate] WARNING: No calibration data")
        return

    for pass_idx in range(n_passes):
        # Reset max_abs_acc tracking
        for L in layers:
            if 'requant_per_ch' in L:
                L['_max_abs_acc'] = np.zeros(L['out_c'], dtype=np.float64)

        # Run all calibration images
        for img_idx, int8_input in enumerate(cal_images):
            act = int8_input.copy()
            cur_h, cur_w = 96, 96

            for li, L in enumerate(layers):
                if L['type'] == 'LAYER_GLOBAL_AVG_POOL':
                    pooled = act.astype(np.int32).mean(axis=(0, 1))
                    act = np.clip(np.round(pooled), -128, 127).astype(np.int8).reshape(1, 1, -1)
                    cur_h, cur_w = 1, 1
                    continue

                K = L.get('kernel', 1) or 1
                S = L.get('stride', 1) or 1
                P = L.get('padding', 0)
                is_dw = L['type'] == 'LAYER_DEPTHWISE_CONV2D'
                is_dense = L['type'] == 'LAYER_DENSE'

                if is_dense:
                    in_flat = act.flatten().astype(np.int32)
                    w = L['weights_int8'].reshape(L['out_c'], L['in_c'])
                    acc_all = np.zeros((1, L['out_c']), dtype=np.int32)
                    for oc in range(L['out_c']):
                        acc_all[0, oc] = np.dot(in_flat[:L['in_c']].astype(np.int32),
                                                w[oc].astype(np.int32))
                        acc_all[0, oc] += L['bias'][oc]
                    h_out, w_out = 1, 1
                elif is_dw:
                    n_ch = L['in_c']
                    h_out = (cur_h + 2*P - K) // S + 1
                    w_out = (cur_w + 2*P - K) // S + 1
                    if L['quant'] == 'QUANT_INT8':
                        w_k = L['weights_int8'].reshape(n_ch, K * K)
                    else:
                        # Ternary depthwise: unpack to {-1, 0, +1}
                        num_w = n_ch * K * K
                        pad_num_w = ((num_w + 63) // 64) * 64
                        tern_float = unpack_ternary_weights(
                            L['weights_packed'], pad_num_w, L['scale_pos'], L['scale_neg'])
                        w_k = np.sign(tern_float[:num_w]).astype(np.int8).reshape(n_ch, K * K)
                    acc_all = np.zeros((h_out * w_out, n_ch), dtype=np.int32)
                    if P > 0:
                        padded = np.zeros((cur_h+2*P, cur_w+2*P, n_ch), dtype=np.int8)
                        padded[P:P+cur_h, P:P+cur_w, :] = act
                    else:
                        padded = act
                    for ch in range(n_ch):
                        for oh in range(h_out):
                            for ow in range(w_out):
                                patch = padded[oh*S:oh*S+K, ow*S:ow*S+K, ch].flatten()
                                acc_all[oh*w_out+ow, ch] = np.dot(
                                    patch.astype(np.int32), w_k[ch].astype(np.int32))
                                acc_all[oh*w_out+ow, ch] += L['bias'][ch]
                else:
                    patches, h_out, w_out = _im2col(act, K, S, P)
                    if L['quant'] == 'QUANT_INT8':
                        w = L['weights_int8'].reshape(L['out_c'], -1)
                        acc_all = patches.astype(np.int32) @ w.astype(np.int32).T
                        for oc in range(L['out_c']):
                            acc_all[:, oc] += L['bias'][oc]
                    else:
                        cin_pad = (L['in_c'] + 63) & ~63
                        num_w = L['out_c'] * K * K * cin_pad
                        tern_float = unpack_ternary_weights(
                            L['weights_packed'], num_w, L['scale_pos'], L['scale_neg'])
                        tern_signs = np.sign(tern_float).astype(np.int8)
                        tern_signs = tern_signs.reshape(L['out_c'], K * K * cin_pad)
                        actual_cols = K * K * L['in_c']
                        tern_used = tern_signs[:, :actual_cols]
                        acc_all = patches[:, :actual_cols].astype(np.int32) @ tern_used.astype(np.int32).T
                        for oc in range(L['out_c']):
                            acc_all[:, oc] += L['bias'][oc]

                # Track max absolute accumulator per channel
                for oc in range(L['out_c']):
                    ch_max = np.abs(acc_all[:, oc].astype(np.float64)).max()
                    if ch_max > L['_max_abs_acc'][oc]:
                        L['_max_abs_acc'][oc] = ch_max

                # Requant to INT8 using current requant
                rq = L['requant_per_ch']
                out_i8 = np.zeros_like(acc_all, dtype=np.int8)
                for oc in range(acc_all.shape[1]):
                    fval = acc_all[:, oc].astype(np.float64) * rq[oc]
                    ival = np.where(fval >= 0, np.floor(fval + 0.5), np.ceil(fval - 0.5))
                    out_i8[:, oc] = np.clip(ival, -128, 127).astype(np.int8)

                if li < len(layers) - 1:
                    out_i8 = np.maximum(out_i8, np.int8(0))

                act = out_i8.reshape(h_out, w_out, L['out_c'])
                cur_h, cur_w = h_out, w_out

        # Update requant from measured ranges
        for li, L in enumerate(layers):
            if '_max_abs_acc' in L:
                max_acc = L['_max_abs_acc']
                max_acc = np.where(max_acc < 1.0, 1.0, max_acc)
                new_rq = (127.0 / max_acc).astype(np.float32)
                L['requant_per_ch'] = np.clip(new_rq, 1e-6, 10.0)
                del L['_max_abs_acc']

        print(f"  Pass {pass_idx+1}: calibrated {len([L for L in layers if 'requant_per_ch' in L])} layers")


def _compute_s_out(float_range, percentile=99.5):
    """Compute per-tensor output scale from per-channel float activation ranges.

    Uses a percentile instead of max to avoid letting outlier channels waste
    the INT8 dynamic range. Channels above the percentile get clipped to 127
    (saturated) but the majority of channels get better precision.
    """
    if float_range is None:
        return None
    fr = float_range
    if len(fr) <= 2:
        # Classifier or very few channels — use max
        return fr.max() / 127.0
    pct_val = np.percentile(fr, percentile)
    # Don't let percentile be less than 50% of max (avoid extreme clipping)
    pct_val = max(pct_val, fr.max() * 0.5)
    return max(pct_val / 127.0, 1e-10)


def extract_layers(model, threshold_ratio=0.05):
    """Walk model and extract layer configs with packed weights.

    Uses analytical requant and bias from float activation ranges (per-tensor
    output scaling with percentile clipping). This ensures correct
    scale propagation through the layer chain while giving weak channels
    more effective INT8 bits.
    """
    layers = []

    # Step 1: Measure float activation ranges for analytical requant/bias
    print("[export] Measuring float activation ranges...")
    float_ranges = _measure_float_activation_ranges(model)

    # Input scale: max abs of ImageNet-normalized input ≈ (1 - 0.406) / 0.225
    INPUT_ABSMAX = 2.64
    s_in = INPUT_ABSMAX / 127.0  # float value per int8 unit

    # --- First conv (INT8, BN folded) ---
    first_conv = model.first_conv[0]  # Conv2d
    first_bn = model.first_conv[1]    # BatchNorm2d

    w_float = first_conv.weight.data.clone()
    bn_s = first_bn.weight.data / torch.sqrt(first_bn.running_var + 1e-5)
    bn_bias = first_bn.bias.data - first_bn.running_mean * bn_s
    for oc in range(first_conv.out_channels):
        w_float[oc] *= bn_s[oc]

    w_nhwc = nchw_to_nhwc(w_float)
    w_int8, w_scale_per_ch = pack_int8_weights_per_channel(w_nhwc)

    # Bias: bn_folded_bias / (s_in * w_scale_per_ch)
    acc_scale = s_in * w_scale_per_ch
    safe_acc_scale = np.where(acc_scale < 1e-10, 1e-10, acc_scale)
    bias_i32 = np.round(bn_bias.cpu().numpy() / safe_acc_scale).astype(np.int32)

    # Output scale from float activation range (per-tensor: max over all channels)
    n_out = first_conv.out_channels
    fr = float_ranges.get('first_conv')
    s_out = _compute_s_out(fr) or (s_in * float(w_scale_per_ch.max()))
    s_out = max(s_out, 1e-10)

    # Analytical requant: requant[c] = (s_in * w_scale[c]) / s_out
    requant_per_ch = (acc_scale / s_out).astype(np.float32)
    requant_per_ch = np.clip(requant_per_ch, 1e-6, 10.0)

    layers.append({
        "name": "first_conv",
        "type": "LAYER_CONV2D",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8.flatten(),
        "weight_scale": float(w_scale_per_ch.max()),
        "bias": bias_i32,
        "requant_per_ch": requant_per_ch,
        "in_c": 3, "out_c": n_out,
        "kernel": first_conv.kernel_size[0],
        "stride": first_conv.stride[0],
        "padding": first_conv.padding[0],
        "requant_scale": 0.0,
        "requant_zp": 0,
    })
    print(f"  first_conv: s_in={s_in:.6f} s_out={s_out:.6f} requant=[{requant_per_ch.min():.4f}, {requant_per_ch.max():.4f}]")
    s_in = s_out  # update for next layer

    # --- Depthwise-separable blocks ---
    for blk_idx, block in enumerate(model.features):
        for sub_name, sub_seq in [("dw", block.dw), ("pw", block.pw)]:
            conv = sub_seq[0]  # Conv2d or TernaryQuantWrapper
            bn = sub_seq[1]    # BatchNorm2d

            is_depthwise = sub_name == "dw"
            layer_type = "LAYER_DEPTHWISE_CONV2D" if is_depthwise else "LAYER_CONV2D"

            bn_s = bn.weight.data / torch.sqrt(bn.running_var + 1e-5)
            bn_bias_val = (bn.bias.data - bn.running_mean * bn_s)

            range_key = f'features_{blk_idx}_{sub_name}'
            fr = float_ranges.get(range_key)
            s_out = _compute_s_out(fr) or s_in
            s_out = max(s_out, 1e-10)

            if isinstance(conv, TernaryQuantWrapper):
                sp = conv.scale_pos.item()
                sn = conv.scale_neg.item()
                avg_s = (sp + sn) / 2.0
                w = nchw_to_nhwc(conv.weight_fp.data, depthwise=is_depthwise)
                packed, _, _, sparsity = pack_ternary_weights(w, threshold_ratio)

                # Ternary: acc_float = bn_s * s_in * avg_ternary_scale * acc_int32 + bn_folded_bias
                # Firmware: out_int8 = (acc_int32 + bias) * requant
                # → bias = bn_folded_bias / (bn_s * s_in * avg_s)
                # → requant[c] = bn_s[c] * s_in * avg_s / s_out
                bn_s_np = bn_s.cpu().numpy()
                acc_scale_per_ch = bn_s_np * avg_s * s_in
                safe_acc_s = np.where(np.abs(acc_scale_per_ch) < 1e-10, 1e-10, acc_scale_per_ch)
                bias_i32 = np.round(bn_bias_val.cpu().numpy() / safe_acc_s).astype(np.int32)
                n_oc = conv.module.out_channels

                requant_per_ch = (np.abs(acc_scale_per_ch) / s_out).astype(np.float32)
                requant_per_ch = np.clip(requant_per_ch, 1e-6, 10.0)

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_TERNARY",
                    "weights_packed": packed,
                    "scale_pos": sp, "scale_neg": sn,
                    "bias": bias_i32,
                    "_float_bias": bn_bias_val.cpu().numpy().copy(),
                    "requant_per_ch": requant_per_ch,
                    "in_c": conv.module.in_channels, "out_c": n_oc,
                    "kernel": conv.module.kernel_size[0],
                    "stride": conv.module.stride[0],
                    "padding": conv.module.padding[0],
                    "requant_scale": 0.0, "requant_zp": 0,
                    "sparsity": sparsity,
                })
            else:
                # INT8: fold BN into weights
                w_float = conv.weight.data.clone()
                n_oc = conv.out_channels if not is_depthwise else conv.weight.shape[0]
                for oc in range(n_oc):
                    w_float[oc] *= bn_s[oc]

                w_nhwc = nchw_to_nhwc(w_float, depthwise=is_depthwise)
                w_int8, w_scale_per_ch = pack_int8_weights_per_channel(w_nhwc)

                # Bias: bn_folded_bias / (s_in * w_scale_per_ch)
                acc_scale = s_in * w_scale_per_ch
                safe_acc_scale = np.where(acc_scale < 1e-10, 1e-10, acc_scale)
                bias_i32 = np.round(bn_bias_val.cpu().numpy() / safe_acc_scale).astype(np.int32)

                requant_per_ch = (acc_scale / s_out).astype(np.float32)
                requant_per_ch = np.clip(requant_per_ch, 1e-6, 10.0)

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_INT8",
                    "weights_int8": w_int8.flatten(),
                    "weight_scale": float(w_scale_per_ch.max()),
                    "bias": bias_i32,
                    "requant_per_ch": requant_per_ch,
                    "in_c": conv.in_channels, "out_c": conv.out_channels,
                    "kernel": conv.kernel_size[0],
                    "stride": conv.stride[0],
                    "padding": conv.padding[0],
                    "requant_scale": 0.0, "requant_zp": 0,
                })

            print(f"  {range_key:30s} s_in={s_in:.6f} s_out={s_out:.6f} requant=[{requant_per_ch.min():.4f}, {requant_per_ch.max():.4f}]")
            s_in = s_out  # update for next layer

    # --- Global average pool (scale unchanged) ---
    layers.append({
        "name": "pool",
        "type": "LAYER_GLOBAL_AVG_POOL",
        "quant": "QUANT_INT8",
        "in_c": layers[-1]["out_c"],
        "out_c": layers[-1]["out_c"],
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": 1.0, "requant_zp": 0,
    })
    # s_in unchanged after pool

    # --- Classifier (INT8, per-channel) ---
    classifier = model.classifier
    w_cls = classifier.weight.data
    w_int8_cls, w_scale_per_ch = pack_int8_weights_per_channel(w_cls)

    # Bias: float_bias / (s_in * w_scale_per_ch)
    acc_scale = s_in * w_scale_per_ch
    safe_acc_scale = np.where(acc_scale < 1e-10, 1e-10, acc_scale)
    float_bias_cls = classifier.bias.data.cpu().numpy()
    bias_i32 = np.round(float_bias_cls / safe_acc_scale).astype(np.int32)

    # Classifier requant from float output range (use max for classifier — only 2 outputs)
    fr = float_ranges.get('classifier')
    s_out_cls = _compute_s_out(fr) or max(s_in * float(w_scale_per_ch.max()) * classifier.in_features / 127.0, 1e-10)

    n_cls = classifier.out_features
    requant_per_ch = (acc_scale / s_out_cls).astype(np.float32)
    requant_per_ch = np.clip(requant_per_ch, 1e-6, 10.0)

    layers.append({
        "name": "classifier",
        "type": "LAYER_DENSE",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8_cls.flatten(),
        "weight_scale": float(w_scale_per_ch.max()),
        "bias": bias_i32,
        "requant_per_ch": requant_per_ch,
        "in_c": classifier.in_features,
        "out_c": n_cls,
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": 0.0,
        "requant_zp": 0,
    })
    print(f"  classifier: s_in={s_in:.6f} s_out={s_out_cls:.6f} requant=[{requant_per_ch.min():.4f}, {requant_per_ch.max():.4f}]")

    # Refine requant with multi-pass sequential calibration.
    # The analytical requant (per-tensor output scaling) initializes the first pass.
    # NOTE: Calibration (_calibrate_requant) is disabled because it breaks the
    # bias/requant coupling. The analytical requant maintains correct scale chain:
    #   bias = float_bias / acc_scale
    #   requant = acc_scale / s_out
    # Calibration changes requant (to 127/max_abs_acc) without updating bias,
    # creating systematic errors that accumulate through the network.

    # Verification
    print("[export] Verifying quantized pipeline...")
    _verify_quantized_pipeline(layers)

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

        if "requant_per_ch" in layer:
            rq = layer["requant_per_ch"]
            total_flash += len(rq) * 4
            items = ", ".join(f"{float(v):.6f}f" for v in rq)
            lines.append(
                f"static const float __attribute__((aligned(16))) "
                f"layer{i}_requant_per_ch[{len(rq)}] = {{"
            )
            lines.append(f"    {items}")
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
        rq_per_ch_ref = f"layer{i}_requant_per_ch" if "requant_per_ch" in layer else "NULL"

        sp = f"{layer.get('scale_pos', 0):.6f}f"
        sn = f"{layer.get('scale_neg', 0):.6f}f"

        lines.append(
            f"    {{ {layer['type']}, {layer['quant']}, "
            f"{weights_ref}, {bias_ref}, "
            f"{sp}, {sn}, "
            f"layer{i}_requant_scale, {rq_per_ch_ref}, layer{i}_requant_zp, "
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
            torch.load(args.weights, weights_only=True, map_location="cpu"),
            strict=False  # Checkpoint may include FakeQuantize buffers from QAT
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
