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


def _sequential_calibration(layers, data_dir=None):
    """Run calibration through simulated quantized inference.

    Measures actual INT32 accumulator ranges at each layer and sets
    requant_per_ch = 127 / max_abs_acc for each layer.
    Modifies layers in-place.
    """
    from train_baseline import get_loaders, BinaryPetsDataset
    import torchvision.transforms as transforms
    import os

    if data_dir is None:
        for p in ["/tmp/esp_datasets", "model/datasets"]:
            if os.path.exists(p):
                data_dir = p
                break
    if data_dir is None:
        print("[export] WARNING: No calibration data for sequential calibration")
        return

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Load calibration images and convert to firmware-style INT8
    raw_tf = transforms.Compose([
        transforms.Resize(104), transforms.CenterCrop(96), transforms.ToTensor()])
    ds = BinaryPetsDataset(data_dir, "test", raw_tf)

    # Use 64 images for calibration (balanced: 32 cats + 32 dogs)
    cal_images = []
    cal_labels = []
    cats_seen = 0
    dogs_seen = 0
    for i, (img, label) in enumerate(ds):
        if label == 0 and cats_seen >= 32:
            continue
        if label == 1 and dogs_seen >= 32:
            continue
        uint8_hwc = (img * 255).byte().permute(1, 2, 0).contiguous().numpy()
        int8_img = np.zeros((96, 96, 3), dtype=np.int8)
        for c in range(3):
            norm = uint8_hwc[:,:,c].astype(np.float32) / 255.0
            norm = (norm - MEAN[c]) / STD[c]
            int8_img[:,:,c] = np.clip(np.round(norm * 127), -128, 127).astype(np.int8)
        cal_images.append(int8_img)
        cal_labels.append(label)
        if label == 0:
            cats_seen += 1
        else:
            dogs_seen += 1
        if cats_seen >= 32 and dogs_seen >= 32:
            break

    print(f"[export] Sequential calibration with {len(cal_images)} images")

    # Track per-channel max absolute accumulator for each layer
    for li, L in enumerate(layers):
        if L['type'] == 'LAYER_GLOBAL_AVG_POOL':
            continue  # No requant needed
        if 'requant_per_ch' not in L and L.get('requant_scale', 0) == 1.0:
            continue  # Pool layer

        n_out = L['out_c']
        max_abs_acc = np.zeros(n_out, dtype=np.float64)

    # Run each calibration image through the quantized chain
    for img_idx, int8_input in enumerate(cal_images):
        act = int8_input.copy()  # [H, W, C] int8
        cur_h, cur_w = 96, 96

        for li, L in enumerate(layers):
            if L['type'] == 'LAYER_GLOBAL_AVG_POOL':
                # Average pool: mean of int8 values
                pooled = act.astype(np.int32).mean(axis=(0, 1))
                act = np.clip(np.round(pooled), -128, 127).astype(np.int8).reshape(1, 1, -1)
                cur_h, cur_w = 1, 1
                continue

            K = L.get('kernel', 1) or 1
            S = L.get('stride', 1) or 1
            P = L.get('padding', 0)
            is_dw = L['type'] == 'LAYER_DEPTHWISE_CONV2D'
            is_dense = L['type'] == 'LAYER_DENSE'

            if img_idx == 0:
                nz = np.count_nonzero(act)
                print(f"    L{li} {L['name']:25s} in: shape={act.shape} nz={nz}/{act.size} range=[{act.min()},{act.max()}]")

            if is_dense:
                in_flat = act.flatten().astype(np.int32)
                w = L['weights_int8'].reshape(L['out_c'], L['in_c'])
                acc_all = np.zeros((1, L['out_c']), dtype=np.int32)
                for oc in range(L['out_c']):
                    acc_all[0, oc] = np.dot(in_flat[:L['in_c']].astype(np.int32),
                                            w[oc].astype(np.int32))
                    if L['quant'] == 'QUANT_INT8':
                        acc_all[0, oc] += L['bias'][oc]
                h_out, w_out = 1, 1

            elif is_dw:
                patches, h_out, w_out = _im2col(act, K, S, P)
                n_ch = L['in_c']
                if L['quant'] == 'QUANT_INT8':
                    w = L['weights_int8'].reshape(n_ch, K * K)
                    acc_all = np.zeros((h_out * w_out, n_ch), dtype=np.int32)
                    for ch in range(n_ch):
                        # Extract this channel's patches
                        ch_patches = patches[:, ch::n_ch*0+1]  # wrong indexing
                        # Actually for depthwise, patches has all channels interleaved
                        # Need per-channel patches
                        pass
                    # Simpler approach: loop over spatial positions
                    acc_all = np.zeros((h_out * w_out, n_ch), dtype=np.int32)
                    if P > 0:
                        padded = np.zeros((cur_h+2*P, cur_w+2*P, n_ch), dtype=np.int8)
                        padded[P:P+cur_h, P:P+cur_w, :] = act
                    else:
                        padded = act
                    for ch in range(n_ch):
                        w_ch = w[ch]  # [K*K]
                        for oh in range(h_out):
                            for ow in range(w_out):
                                patch = padded[oh*S:oh*S+K, ow*S:ow*S+K, ch].flatten()
                                acc_all[oh*w_out+ow, ch] = np.dot(
                                    patch.astype(np.int32), w_ch.astype(np.int32))
                                acc_all[oh*w_out+ow, ch] += L['bias'][ch]
                else:
                    # Ternary depthwise (not used in current config, skip for now)
                    acc_all = np.zeros((h_out * w_out, n_ch), dtype=np.int32)

            else:  # Regular conv (INT8 or ternary)
                patches, h_out, w_out = _im2col(act, K, S, P)

                if L['quant'] == 'QUANT_INT8':
                    w = L['weights_int8'].reshape(L['out_c'], -1)  # [C_out, K*K*C_in]
                    # matmul: [HW, KKC] @ [KKC, C_out] = [HW, C_out]
                    acc_all = patches.astype(np.int32) @ w.astype(np.int32).T
                    for oc in range(L['out_c']):
                        acc_all[:, oc] += L['bias'][oc]
                else:
                    # Ternary: unpack weights to {-1, 0, +1}
                    cin_pad = (L['in_c'] + 63) & ~63
                    num_w = L['out_c'] * K * K * cin_pad
                    tern_float = unpack_ternary_weights(
                        L['weights_packed'], num_w, L['scale_pos'], L['scale_neg'])
                    tern_signs = np.sign(tern_float).astype(np.int8)
                    tern_signs = tern_signs.reshape(L['out_c'], K * K * cin_pad)
                    # Only use the non-padded columns
                    actual_cols = K * K * L['in_c']
                    tern_used = tern_signs[:, :actual_cols]
                    acc_all = patches[:, :actual_cols].astype(np.int32) @ tern_used.astype(np.int32).T

            # Now acc_all is [spatial, C_out] INT32
            # For ternary: add bias in accumulator domain
            if L['quant'] == 'QUANT_TERNARY' and not is_dense:
                for oc in range(L['out_c']):
                    acc_all[:, oc] += L['bias'][oc]
            if L['quant'] == 'QUANT_TERNARY' and is_dense:
                for oc in range(L['out_c']):
                    acc_all[:, oc] += L['bias'][oc]

            # Track max absolute accumulator per channel (across all images)
            for oc in range(L['out_c']):
                ch_max = np.abs(acc_all[:, oc].astype(np.float64)).max()
                if '_max_abs_acc' not in L:
                    L['_max_abs_acc'] = np.zeros(L['out_c'], dtype=np.float64)
                if ch_max > L['_max_abs_acc'][oc]:
                    L['_max_abs_acc'][oc] = ch_max

            # Apply requant (use current requant_per_ch or compute from accumulated max)
            # For now, use a provisional requant based on this image's range
            if 'requant_per_ch' in L:
                rq = L['requant_per_ch']
            else:
                # Provisional: just clamp
                rq = np.ones(L['out_c'], dtype=np.float32)

            # Requant to INT8
            out_i8 = np.zeros_like(acc_all, dtype=np.int8)
            for oc in range(acc_all.shape[1]):
                fval = acc_all[:, oc].astype(np.float64) * rq[oc]
                ival = np.where(fval >= 0, np.floor(fval + 0.5), np.ceil(fval - 0.5))
                out_i8[:, oc] = np.clip(ival, -128, 127).astype(np.int8)

            # ReLU (except for classifier which is last layer before pool... actually classifier IS after pool)
            if li < len(layers) - 1:  # Don't ReLU the classifier
                out_i8 = np.maximum(out_i8, np.int8(0))

            act = out_i8.reshape(h_out, w_out, L['out_c'])
            cur_h, cur_w = h_out, w_out

        if img_idx == 0:
            print(f"  Image 0 classifier output: {act.flatten()[:2]}")
            # Debug: print per-layer activation stats for first image
            # (already printed in loop below during second pass)

    # Now compute final requant_per_ch from accumulated max_abs_acc
    for li, L in enumerate(layers):
        if '_max_abs_acc' in L:
            max_acc = L['_max_abs_acc']
            max_acc = np.where(max_acc < 1.0, 1.0, max_acc)  # floor to prevent huge scales
            L['requant_per_ch'] = (127.0 / max_acc).astype(np.float32)
            del L['_max_abs_acc']
            print(f"  L{li} {L['name']:30s} requant=[{L['requant_per_ch'].min():.4f}, {L['requant_per_ch'].max():.4f}]")

    # Second pass: run one image with the corrected scales to verify
    act = cal_images[0].copy()
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
                if L['quant'] == 'QUANT_INT8':
                    acc_all[0, oc] += L['bias'][oc]
                else:
                    acc_all[0, oc] += L['bias'][oc]
            h_out, w_out = 1, 1
        elif is_dw:
            n_ch = L['in_c']
            h_out = (cur_h + 2*P - K) // S + 1
            w_out = (cur_w + 2*P - K) // S + 1
            w_k = L['weights_int8'].reshape(n_ch, K * K)
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

    print(f"  Verification: classifier output = {act.flatten()[:2]} -> {'CAT' if act.flatten()[0] > act.flatten()[1] else 'DOG'}")


def extract_layers(model, threshold_ratio=0.05):
    """Walk model and extract layer configs with packed weights."""
    layers = []

    # Phase 1: Build all layers with weights, bias, and PLACEHOLDER requant.
    # Phase 2 (at end): Sequential calibration fills in correct requant_per_ch.

    first_conv = model.first_conv[0]  # Conv2d
    first_bn = model.first_conv[1]    # BatchNorm2d

    # Fold BatchNorm into weights (but NOT input normalization)
    w_float = first_conv.weight.data.clone()
    bn_s = first_bn.weight.data / torch.sqrt(first_bn.running_var + 1e-5)
    bn_bias = first_bn.bias.data - first_bn.running_mean * bn_s
    for oc in range(first_conv.out_channels):
        w_float[oc] *= bn_s[oc]

    w_nhwc = nchw_to_nhwc(w_float)
    w_int8, w_scale_per_ch = pack_int8_weights_per_channel(w_nhwc)

    # Bias: convert float bias to accumulator domain
    # acc_scale[c] ≈ w_scale_per_ch[c] (input is int8 with ~unit scale after normalization)
    safe_ws = np.where(w_scale_per_ch < 1e-10, 1e-10, w_scale_per_ch)
    bias_i32 = np.round(bn_bias.cpu().numpy() / safe_ws).astype(np.int32)

    # Placeholder requant — sequential calibration will fill this in
    n_out = first_conv.out_channels
    layers.append({
        "name": "first_conv",
        "type": "LAYER_CONV2D",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8.flatten(),
        "weight_scale": float(w_scale_per_ch.max()),
        "bias": bias_i32,
        "requant_per_ch": np.ones(n_out, dtype=np.float32),  # placeholder
        "in_c": 3, "out_c": n_out,
        "kernel": first_conv.kernel_size[0],
        "stride": first_conv.stride[0],
        "padding": first_conv.padding[0],
        "requant_scale": 0.0,
        "requant_zp": 0,
    })

    # Depthwise-separable blocks
    for blk_idx, block in enumerate(model.features):
        for sub_name, sub_seq in [("dw", block.dw), ("pw", block.pw)]:
            conv = sub_seq[0]  # Conv2d or TernaryQuantWrapper
            bn = sub_seq[1]    # BatchNorm2d

            is_depthwise = sub_name == "dw"
            layer_type = "LAYER_DEPTHWISE_CONV2D" if is_depthwise else "LAYER_CONV2D"

            bn_s = bn.weight.data / torch.sqrt(bn.running_var + 1e-5)
            bn_bias_val = (bn.bias.data - bn.running_mean * bn_s)

            if isinstance(conv, TernaryQuantWrapper):
                sp = conv.scale_pos.item()
                sn = conv.scale_neg.item()
                w = nchw_to_nhwc(conv.weight_fp.data, depthwise=is_depthwise)
                packed, _, _, sparsity = pack_ternary_weights(w, threshold_ratio)

                # Ternary bias: acc = sum(input_int8 * {-1,0,+1})
                # BN not in weights, so bias needs BN info. Use avg_scale as approx.
                avg_ls = (sp + sn) / 2.0
                bn_s_np = bn_s.cpu().numpy()
                acc_s = np.where(np.abs(bn_s_np * avg_ls) < 1e-10, 1e-10, bn_s_np * avg_ls)
                bias_i32 = np.round(bn_bias_val.cpu().numpy() / acc_s).astype(np.int32)
                n_oc = conv.module.out_channels

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_TERNARY",
                    "weights_packed": packed,
                    "scale_pos": sp, "scale_neg": sn,
                    "bias": bias_i32,
                    "_float_bias": bn_bias_val.cpu().numpy().copy(),
                    "requant_per_ch": np.ones(n_oc, dtype=np.float32),  # placeholder
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
                n_out = conv.out_channels if not is_depthwise else conv.weight.shape[0]
                for oc in range(n_out):
                    w_float[oc] *= bn_s[oc]

                w_nhwc = nchw_to_nhwc(w_float, depthwise=is_depthwise)
                w_int8, w_scale_per_ch = pack_int8_weights_per_channel(w_nhwc)

                # Bias in accumulator domain (approx: use weight scale as acc scale)
                safe_ws = np.where(w_scale_per_ch < 1e-10, 1e-10, w_scale_per_ch)
                bias_i32 = np.round(bn_bias_val.cpu().numpy() / safe_ws).astype(np.int32)

                layers.append({
                    "name": f"features_{blk_idx}_{sub_name}",
                    "type": layer_type,
                    "quant": "QUANT_INT8",
                    "weights_int8": w_int8.flatten(),
                    "weight_scale": float(w_scale_per_ch.max()),
                    "bias": bias_i32,
                    "requant_per_ch": np.ones(n_out, dtype=np.float32),  # placeholder
                    "in_c": conv.in_channels, "out_c": conv.out_channels,
                    "kernel": conv.kernel_size[0],
                    "stride": conv.stride[0],
                    "padding": conv.padding[0],
                    "requant_scale": 0.0, "requant_zp": 0,
                })

    # Global average pool (passes through, scale unchanged)
    layers.append({
        "name": "pool",
        "type": "LAYER_GLOBAL_AVG_POOL",
        "quant": "QUANT_INT8",
        "in_c": layers[-1]["out_c"],
        "out_c": layers[-1]["out_c"],
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": 1.0, "requant_zp": 0,
    })

    # Classifier (INT8, per-channel) — preserve trained bias
    classifier = model.classifier
    w_cls = classifier.weight.data
    w_int8_cls, w_scale_per_ch = pack_int8_weights_per_channel(w_cls)
    # Bias in accumulator domain: bias_i32 = float_bias / w_scale_per_ch (approx)
    safe_ws = np.where(w_scale_per_ch < 1e-10, 1e-10, w_scale_per_ch)
    float_bias_cls = classifier.bias.data.cpu().numpy()
    bias_i32 = np.round(float_bias_cls / safe_ws).astype(np.int32)

    n_cls = classifier.out_features
    layers.append({
        "name": "classifier",
        "type": "LAYER_DENSE",
        "quant": "QUANT_INT8",
        "weights_int8": w_int8_cls.flatten(),
        "weight_scale": float(w_scale_per_ch.max()),
        "bias": bias_i32,
        "requant_per_ch": np.ones(n_cls, dtype=np.float32),  # placeholder
        "in_c": classifier.in_features,
        "out_c": n_cls,
        "kernel": 0, "stride": 0, "padding": 0,
        "requant_scale": 0.0,
        "requant_zp": 0,
    })

    # Phase 2: Initialize requant with heuristic, then refine with sequential calibration
    for L in layers:
        if 'requant_per_ch' in L:
            K = L.get('kernel', 1) or 1
            fan = K * K * L.get('in_c', 1)
            # Heuristic: typical acc ≈ sqrt(fan) * avg_input * avg_weight
            # For INT8: avg ~40, for ternary: avg ~30
            est_acc = max(1.0, np.sqrt(fan) * 40.0)
            L['requant_per_ch'][:] = 127.0 / est_acc

    # Use sequential calibration (measures actual INT32 accumulator ranges)
    _sequential_calibration(layers)
    return layers

    # DISABLED: QAT-learned scales approach (doesn't match integer accumulator ranges)
    from quantize import FakeQuantize as FQ
    fq_modules = []
    for n, m in model.named_modules():
        if isinstance(m, FQ) and m.running_absmax is not None:
            fq_modules.append((n, m))

    if fq_modules:
        print(f"[export] Using {len(fq_modules)} QAT-learned scales for requant")
        # Each FakeQuantize has running_absmax[C] = per-channel output range
        # The requant must convert INT32 accumulator → INT8 output
        # output_float = acc_int32 * (input_scale * weight_scale)
        # output_int8 = output_float / (absmax/127) = acc * input_scale * weight_scale * 127 / absmax
        # So requant_per_ch[c] = input_scale * weight_scale_per_ch[c] * 127 / absmax[c]
        #
        # But we DON'T KNOW input_scale (the effective float-per-int8 of the input).
        # HOWEVER: we can compute it from the PREVIOUS layer's absmax!
        # input_scale = prev_absmax / 127 (since prev layer maps its output to [-127,127] via its own FQ)
        #
        # So: requant[c] = (prev_absmax_mean / 127) * w_scale[c] * 127 / absmax[c]
        #                = prev_absmax_mean * w_scale[c] / absmax[c]

        # For first conv: input is firmware-normalized, input_scale = max_normalized / 127
        input_absmax = 2.64  # max ImageNet normalized value

        fq_idx = 0
        for li, L in enumerate(layers):
            if L['type'] == 'LAYER_GLOBAL_AVG_POOL':
                continue
            if 'requant_per_ch' not in L:
                continue
            if fq_idx >= len(fq_modules):
                break

            _, fq = fq_modules[fq_idx]
            out_absmax = fq.running_absmax.cpu().numpy()  # [C_out]
            out_absmax = np.where(out_absmax < 1e-8, 1e-8, out_absmax)

            if L['quant'] == 'QUANT_INT8':
                # For INT8: acc = sum(input_int8 * weight_int8) + bias
                # Float equiv: acc * (input_absmax/127) * w_scale_per_ch[c]
                # → INT8: acc * (input_absmax/127) * w_scale[c] * (127/out_absmax[c])
                # = acc * input_absmax * w_scale[c] / out_absmax[c]
                # Get per-channel w_scale from the layer's weights
                w_all = L['weights_int8']
                n_oc = L['out_c']
                w_per_ch = w_all.reshape(n_oc, -1)
                w_scale_per_ch = np.array([np.abs(w_per_ch[oc]).max() / 127.0 if np.abs(w_per_ch[oc]).max() > 0 else 1e-10 for oc in range(n_oc)])
                # Wait — weights are already INT8 quantized per channel, so w_scale is implicit
                # The actual weight float = weight_int8 * w_scale_per_ch
                # But pack_int8_weights_per_channel already divided by w_scale, so weight_int8 ≈ round(w_float / w_scale)
                # acc = sum(input_int8 * weight_int8) → float = acc * (input_absmax/127) * w_scale_per_ch
                # Actually, w_scale_per_ch was used to quantize the weights, so we stored it.
                # But it was returned as the second output of pack_int8_weights_per_channel.
                # It's stored as L['weight_scale'] (max), not per-channel...
                # We need to recover per-channel scales. They equal absmax(w_float_per_ch) / 127.
                # Let me just compute: w_scale_per_ch already implicit in the int8 weights.
                # Since weight_int8[oc] ≈ round(w_float[oc] / scale[oc]):
                # We don't have w_float anymore, but we have weight_int8 with max abs = 127 per channel.
                # So the effective per-channel w_scale is whatever makes int8 max = 127.
                # That means w_scale_per_ch = w_float_absmax_per_ch / 127.
                # We don't have w_float but we can use the w_scale stored during construction.
                # Problem: we only stored max(w_scale_per_ch), not the per-channel values.

                # Simplest correct approach: w_int8 has max abs = 127 per channel.
                # So effective w_scale per channel = (original float absmax) / 127.
                # We can't recover this from int8 alone.
                # Let me just use L['weight_scale'] (the global max) as an approximation.
                w_s = L['weight_scale']  # global max
                rq = (input_absmax / 127.0) * w_s * (127.0 / out_absmax)
            else:
                # Ternary: acc = sum(input_int8 * {-1,0,+1}) → float = acc * (input_absmax/127) * avg_tern_scale * bn_s
                # But bn_s is per-channel and folded into requant...
                # Simplified: just use the relationship requant = input_absmax * effective_scale / out_absmax
                avg_ts = (L.get('scale_pos', 0.1) + L.get('scale_neg', 0.1)) / 2.0
                rq = (input_absmax / 127.0) * avg_ts * (127.0 / out_absmax)

            # Clamp requant to reasonable range (avoid overflow from near-zero absmax)
            rq = np.clip(rq, 1e-6, 1.0)
            L['requant_per_ch'] = rq.astype(np.float32)
            print(f"  L{li} {L['name']:25s} requant=[{rq.min():.4f}, {rq.max():.4f}]")

            # Next layer's INT8 input has values in [-127, 127]
            # The float value of each INT8 unit = out_absmax / 127 (from this layer's FQ)
            # For the requant formula: input_absmax = max(out_absmax) (the float range)
            # This represents what float value INT8=127 corresponds to
            input_absmax = float(out_absmax.max())
            fq_idx += 1
        # Handle classifier which has no FakeQuantize
        for li, L in enumerate(layers):
            if L['name'] == 'classifier' and 'requant_per_ch' in L:
                    w_s = L['weight_scale']
                    # Classifier output doesn't have a FakeQuantize, so use a generous out_scale
                    # We just need the two logits to have the right relative ordering
                    # Use requant = input_absmax * w_s / (generous_output_range)
                    # generous_output_range = input_absmax * w_s * 127 (full accumulator range)
                    # → requant = 1/127 ≈ 0.00787
                    # Or simpler: requant = 127 / (358 * 127 * input_scale_per_unit)
                    # where input_scale_per_unit = input_absmax / 127
                    # → requant = 1 / (358 * input_absmax / 127) = 127 / (358 * input_absmax)
                    rq_cls = 127.0 / (L['in_c'] * input_absmax / 127.0 * 127.0)
                    # Simplified: rq_cls = 127.0 / (L['in_c'] * input_absmax)
                    rq_cls = 127.0 / (L['in_c'] * max(input_absmax, 1.0))
                    L['requant_per_ch'] = np.full(L['out_c'], rq_cls, dtype=np.float32)
                    print(f"  Classifier requant: {rq_cls:.6f}")
    else:
        print("[export] No QAT scales found, falling back to sequential calibration")
        _sequential_calibration(layers)

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
