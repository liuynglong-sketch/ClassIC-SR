#!/usr/bin/env python3
"""Evaluate ClassIC-SR / Version A under FP32, BF16, and INT8 fake quant.

This script keeps the ClassSR classifier architecture and ClassIC-SR checkpoint
unchanged.  In the default fixed-route mode it uses FP32 classifier decisions for
all precisions, so BF16/INT8 results isolate reconstruction-branch numerical
error rather than router drift.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CLASS_ROOT = ROOT
CODES = ROOT / "codes"
if str(CODES) not in sys.path:
    sys.path.insert(0, str(CODES))

from data.util import bgr2ycbcr  # noqa: E402
from models.archs.classSR_fsrcnn_arch import (  # noqa: E402
    classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net,
)
from utils import util  # noqa: E402

MODEL_KEY = "classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net"
DEFAULT_CKPT = CLASS_ROOT / "pretrained/classic_sr_version_a.pth"
DEFAULT_RESULTS = CLASS_ROOT / "results"
DATE = "2026-05-21"

# Repository FLOPs convention for Version A: Conv/Deconv MACx2, bilinear arithmetic excluded.
# These constants reproduce the existing benchmark values in classic_sr_contribution1_summary_2026-05-07.md.
BRANCH_FLOPS_M = [7.9, 102.9, 468.0]
CLASSIFIER_FLOPS_M = 7.4

EXPECTED_FP32 = {
    "DIV2K_valid": {"psnr": 27.8349, "psnr_y": 29.2792, "flops_m": 153.11},
    "Test2K": {"psnr": 25.6057, "psnr_y": 27.0205, "flops_m": 165.73},
    "Test4K": {"psnr": 26.9061, "psnr_y": 28.2673, "flops_m": 157.09},
    "Test8K": {"psnr": 32.6440, "psnr_y": 34.1137, "flops_m": 145.52},
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y"}


def image_paths(root):
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    if not out:
        raise FileNotFoundError("No image files under {}".format(root))
    return out


def resolve_dataset(data_root, name):
    """Resolve benchmark datasets using README-first paths plus historical fallbacks."""
    data_root = Path(data_root)
    candidates = {
        "DIV2K_valid": [
            (
                data_root / "datasets/DIV2K/DIV2K_valid_HR",
                data_root / "datasets/DIV2K/DIV2K_valid_LR_bicubic/X4",
            ),
            (
                data_root / "DIV2K/DIV2K_valid_HR",
                data_root / "DIV2K/DIV2K_valid_LR_bicubic/X4",
            ),
            (
                data_root / "DIV2K/DIV2K_valid_HR",
                data_root / "data/valid/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4",
            ),
        ],
        "Test2K": [
            (data_root / "datasets/Test2K/HR/X4", data_root / "datasets/Test2K/LR/X4"),
            (data_root / "Test2K/HR/X4", data_root / "Test2K/LR/X4"),
            (data_root / "Test2K4K8K/test2k/HR/X4", data_root / "Test2K4K8K/test2k/LR/X4"),
        ],
        "Test4K": [
            (data_root / "datasets/Test4K/HR/X4", data_root / "datasets/Test4K/LR/X4"),
            (data_root / "Test4K/HR/X4", data_root / "Test4K/LR/X4"),
            (data_root / "Test2K4K8K/test4k/HR/X4", data_root / "Test2K4K8K/test4k/LR/X4"),
        ],
        "Test8K": [
            (data_root / "datasets/Test8K/HR/X4", data_root / "datasets/Test8K/LR/X4"),
            (data_root / "Test8K/HR/X4", data_root / "Test8K/LR/X4"),
            (data_root / "Test2K4K8K/test8k/HR/X4", data_root / "Test2K4K8K/test8k/LR/X4"),
        ],
    }
    if name not in candidates:
        raise ValueError("Unsupported dataset {}".format(name))

    expected = []
    selected = None
    for gt_root, lr_root in candidates[name]:
        expected.append("GT={} LR={}".format(gt_root, lr_root))
        if gt_root.is_dir() and lr_root.is_dir():
            selected = (gt_root, lr_root)
            break
    if selected is None:
        raise FileNotFoundError(
            "Dataset {} missing. Expected one of:\n  {}".format(name, "\n  ".join(expected))
        )

    gt_root, lr_root = selected
    gt_paths = image_paths(gt_root)
    lr_by_name = {p.name: p for p in image_paths(lr_root)}
    pairs = []
    for gt in gt_paths:
        lr = lr_by_name.get(gt.name)
        if lr is None:
            raise FileNotFoundError(
                "Missing LR pair for {} in {}. LR/HR file names must match.".format(gt.name, lr_root)
            )
        pairs.append((lr, gt))
    return {"name": name, "gt_root": str(gt_root), "lr_root": str(lr_root), "pairs": pairs}


def load_model(checkpoint, device):
    model = classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net(in_nc=3, out_nc=3)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    model.to(device)
    model.eval()
    return model, {"missing_keys": missing, "unexpected_keys": unexpected}


def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, _ = img.shape
    else:
        raise ValueError("Wrong image shape: {}".format(img.shape))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    lr_list = []
    num_h = 0
    num_w = 0
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            lr_list.append(img[x:x + crop_sz, y:y + crop_sz, ...] if n_channels == 3 else img[x:x + crop_sz, y:y + crop_sz])
    if not lr_list:
        raise ValueError("Image too small for crop {} step {}: {}".format(crop_sz, step, img.shape))
    h_end = int(h_space[-1] + crop_sz)
    w_end = int(w_space[-1] + crop_sz)
    return lr_list, num_h, num_w, h_end, w_end


def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    sr_img = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    index = 0
    for i in range(num_h):
        for j in range(num_w):
            sr_img[
                i * step * scale:i * step * scale + patch_size * scale,
                j * step * scale:j * step * scale + patch_size * scale,
                :,
            ] += sr_list[index]
            index += 1
    for j in range(1, num_w):
        sr_img[:, j * step * scale:j * step * scale + (patch_size - step) * scale, :] /= 2
    for i in range(1, num_h):
        sr_img[i * step * scale:i * step * scale + (patch_size - step) * scale, :, :] /= 2
    return sr_img


def patches_to_tensor(patches, device):
    arr = []
    for p in patches:
        img = p.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = img[:, :, [2, 1, 0]]  # BGR -> RGB
        arr.append(np.transpose(img, (2, 0, 1)))
    x = torch.from_numpy(np.ascontiguousarray(np.stack(arr, axis=0))).float().to(device)
    return x


def tensor_to_bgr_uint8_batch(tensor):
    tensor = tensor.detach().float().cpu().clamp_(0.0, 1.0)
    arr = tensor.numpy()[:, [2, 1, 0], :, :]  # RGB -> BGR
    arr = np.transpose(arr, (0, 2, 3, 1))
    arr = np.rint(arr * 255.0).astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def calc_metrics(sr_img, gt_img, scale):
    sr_crop, gt_crop = util.crop_border([sr_img, gt_img], scale)
    psnr = util.calculate_psnr(sr_crop, gt_crop)
    if gt_crop.ndim == 3 and gt_crop.shape[2] == 3:
        sr_y = bgr2ycbcr(sr_crop / 255.0, only_y=True)
        gt_y = bgr2ycbcr(gt_crop / 255.0, only_y=True)
        psnr_y = util.calculate_psnr(sr_y * 255.0, gt_y * 255.0)
    else:
        psnr_y = float("nan")
    return float(psnr), float(psnr_y)


def weighted_flops(route_counts):
    total = float(sum(route_counts))
    if total <= 0:
        return 0.0
    branch = sum(BRANCH_FLOPS_M[i] * route_counts[i] for i in range(3)) / total
    return float(branch + CLASSIFIER_FLOPS_M)


def fake_quant_tensor(x, scale=None, qmax=127.0):
    if scale is None or scale <= 0:
        max_abs = x.detach().abs().max()
        scale = float(max_abs.item()) / qmax if max_abs.item() > 0 else 1.0
    y = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return y


def fake_quant_weight_per_out_channel(module):
    w = module.weight.data
    if isinstance(module, nn.ConvTranspose2d):
        # ConvTranspose2d stores weight as [in_channels, out_channels/groups, kH, kW].
        reduce_dims = (0, 2, 3)
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
    elif isinstance(module, (nn.Conv2d, nn.Linear)):
        reduce_dims = tuple(range(1, w.dim()))
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        return
    scale = max_abs / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    module.weight.data = torch.clamp(torch.round(w / scale), -127, 127) * scale


class QuantSim:
    def __init__(self, model, act_percentile=None, quantize_classifier=False):
        self.model = model
        self.act_percentile = act_percentile
        self.quantize_classifier = quantize_classifier
        self.ranges = defaultdict(float)
        self.samples = defaultdict(list)
        self.handles = []
        self.notes = []
        self._orig_interpolate = None
        self.enabled = False
        self.collecting = False

    def _want_module(self, name):
        if self.quantize_classifier:
            return True
        return not name.startswith("classifier")

    def quantize_weights(self):
        count = 0
        skipped_classifier = 0
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if name.startswith("classifier") and not self.quantize_classifier:
                    skipped_classifier += 1
                    continue
                fake_quant_weight_per_out_channel(m)
                count += 1
        self.notes.append("INT8_sim weights: per-output-channel symmetric q-deq for {} Conv/Deconv/Linear modules; bias kept FP32.".format(count))
        if skipped_classifier:
            self.notes.append("INT8_sim main run: classifier weights/activations left FP32; fixed FP32 route labels are used.")

    def _record(self, key, x):
        if not torch.is_tensor(x):
            return
        if self.act_percentile is None:
            val = float(x.detach().abs().max().item())
            if val > self.ranges[key]:
                self.ranges[key] = val
        else:
            flat = x.detach().abs().float().flatten()
            if flat.numel() > 4096:
                idx = torch.linspace(0, flat.numel() - 1, steps=4096, device=flat.device).long()
                flat = flat[idx]
            self.samples[key].append(flat.cpu())

    def _scale(self, key):
        max_abs = self.ranges.get(key, 0.0)
        if max_abs <= 0:
            return None
        return max_abs / 127.0

    def _collect_hook(self, name):
        def hook(_module, _inp, out):
            if self._want_module(name):
                self._record(name, out)
        return hook

    def _quant_hook(self, name):
        def hook(_module, _inp, out):
            if not self._want_module(name):
                return out
            return fake_quant_tensor(out, self._scale(name))
        return hook

    def _patch_interpolate(self, mode):
        if self._orig_interpolate is None:
            self._orig_interpolate = F.interpolate
        orig = self._orig_interpolate
        q = self

        def wrapped(input, *args, **kwargs):
            if mode == "collect":
                q._record("F.interpolate.input", input)
                out = orig(input, *args, **kwargs)
                q._record("F.interpolate.output", out)
                return out
            inp = fake_quant_tensor(input, q._scale("F.interpolate.input"))
            out = orig(inp, *args, **kwargs)
            return fake_quant_tensor(out, q._scale("F.interpolate.output"))

        F.interpolate = wrapped

    def _restore_interpolate(self):
        if self._orig_interpolate is not None:
            F.interpolate = self._orig_interpolate
            self._orig_interpolate = None

    @contextmanager
    def collect_context(self):
        self.clear_hooks()
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.PReLU)):
                self.handles.append(m.register_forward_hook(self._collect_hook(name)))
        self._patch_interpolate("collect")
        try:
            yield
        finally:
            self._restore_interpolate()
            self.clear_hooks()

    @contextmanager
    def quant_context(self):
        self.clear_hooks()
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.PReLU)):
                self.handles.append(m.register_forward_hook(self._quant_hook(name)))
        self._patch_interpolate("quant")
        try:
            yield
        finally:
            self._restore_interpolate()
            self.clear_hooks()

    def finalize_ranges(self):
        if self.act_percentile is not None:
            for key, vals in self.samples.items():
                if not vals:
                    continue
                cat = torch.cat(vals)
                qv = torch.quantile(cat, float(self.act_percentile) / 100.0).item()
                self.ranges[key] = float(qv)
        self.notes.append("INT8_sim activations: per-tensor symmetric q-deq after Conv/Deconv/Linear/PReLU, around interpolation input/output, and before branch output.")
        if self.act_percentile is None:
            self.notes.append("Activation calibration: min-max absolute range.")
        else:
            self.notes.append("Activation calibration: percentile {} absolute range.".format(self.act_percentile))
        self.notes.append("Bilinear interpolation coefficients are not quantized; only interpolation input/output tensors are q-deq simulated.")

    def clear_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def autocast_context(device, precision, notes):
    if precision != "bf16":
        return nullcontext()
    if device.type == "cuda":
        supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not supported:
            notes.append("CUDA BF16 not reported as supported; using autocast BF16 anyway may fall back internally.")
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        notes.append("torch.autocast unavailable; using torch.cuda.amp.autocast(dtype=bf16).")
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    notes.append("BF16 CPU autocast requested; unsupported ops may fall back to FP32.")
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()



@contextmanager
def prelu_fp32_fallback(model, notes):
    """PyTorch 1.10 CUDA lacks BF16 PReLU; compute PReLU in FP32 and cast back."""
    patched = []
    for m in model.modules():
        if isinstance(m, nn.PReLU):
            orig_forward = m.forward

            def make_forward(module):
                def forward(input):
                    with torch.cuda.amp.autocast(enabled=False) if input.is_cuda else nullcontext():
                        out = F.prelu(input.float(), module.weight.float())
                    return out.to(dtype=input.dtype)
                return forward

            m.forward = make_forward(m)
            patched.append((m, orig_forward))
    if patched:
        notes.append("BF16 fallback: PReLU executed in FP32 because torch 1.10 CUDA has no BF16 PReLU kernel; outputs cast back to BF16/autocast dtype.")
    try:
        yield
    finally:
        for m, orig_forward in patched:
            m.forward = orig_forward


@contextmanager
def interpolate_fp32_fallback(notes):
    """PyTorch 1.10 CUDA lacks BF16 bilinear upsample; compute interpolate in FP32."""
    orig = F.interpolate

    def wrapped(input, *args, **kwargs):
        if torch.is_tensor(input) and input.dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False) if input.is_cuda else nullcontext():
                out = orig(input.float(), *args, **kwargs)
            return out.to(dtype=input.dtype)
        return orig(input, *args, **kwargs)

    F.interpolate = wrapped
    notes.append("BF16 fallback: bilinear interpolation executed in FP32 because torch 1.10 CUDA has no BF16 upsample kernel; outputs cast back to BF16/autocast dtype.")
    try:
        yield
    finally:
        F.interpolate = orig

def run_classifier_fp32(model, x, batch_size):
    labels = []
    probs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size]
            logits = model.classifier(xb)
            p = torch.softmax(logits.float(), dim=1)
            labels.append(torch.argmax(logits, dim=1).cpu())
            probs.append(p.cpu())
    return torch.cat(labels, dim=0).numpy().astype(np.int64), torch.cat(probs, dim=0)


def run_branches(model, x, labels, precision, device, batch_size, quant_sim=None, notes=None):
    out_tensors = [None] * int(x.shape[0])
    notes = notes if notes is not None else []
    ctx = autocast_context(device, precision, notes) if precision == "bf16" else nullcontext()
    qctx = quant_sim.quant_context() if precision == "int8_sim" and quant_sim is not None else nullcontext()
    branch_modules = [model.net1, model.net2, model.net3]
    with torch.no_grad(), ctx, qctx, (prelu_fp32_fallback(model, notes) if precision == "bf16" else nullcontext()), (interpolate_fp32_fallback(notes) if precision == "bf16" else nullcontext()):
        for branch_id, branch in enumerate(branch_modules):
            idx = np.where(labels == branch_id)[0]
            if idx.size == 0:
                continue
            for start in range(0, idx.size, batch_size):
                sub = idx[start:start + batch_size]
                xb = x[torch.from_numpy(sub).to(device)]
                y = branch(xb)
                if precision == "int8_sim":
                    scale = quant_sim._scale("branch_output") if quant_sim is not None else None
                    y = fake_quant_tensor(y, scale)
                ys = list(y.detach().float().cpu())
                for j, original_idx in enumerate(sub):
                    out_tensors[int(original_idx)] = ys[j]
    return torch.stack(out_tensors, dim=0)


def evaluate_image(model, lr_path, gt_path, device, scale, patch_size, step, batch_size, precision,
                   fixed_routes=True, fp32_route_cache=None, quant_sim=None, notes=None):
    lr_img = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if lr_img is None:
        raise FileNotFoundError(lr_path)
    if gt_img is None:
        raise FileNotFoundError(gt_path)
    if lr_img.ndim == 2:
        lr_img = np.expand_dims(lr_img, axis=2)
    if gt_img.ndim == 2:
        gt_img = np.expand_dims(gt_img, axis=2)
    if lr_img.shape[2] > 3:
        lr_img = lr_img[:, :, :3]
    if gt_img.shape[2] > 3:
        gt_img = gt_img[:, :, :3]

    patches, num_h, num_w, h, w = crop_cpu(lr_img, patch_size, step)
    x = patches_to_tensor(patches, device)

    if fixed_routes:
        if fp32_route_cache is None:
            labels, _ = run_classifier_fp32(model, x, batch_size)
        else:
            labels = fp32_route_cache
    else:
        labels, _ = run_classifier_fp32(model, x, batch_size)

    y = run_branches(model, x, labels, precision, device, batch_size, quant_sim=quant_sim, notes=notes)
    sr_list = tensor_to_bgr_uint8_batch(y)
    sr_img = combine(sr_list, num_h, num_w, h, w, patch_size, step, scale)
    gt_crop = gt_img[0:h * scale, 0:w * scale, :]
    psnr, psnr_y = calc_metrics(sr_img, gt_crop, scale)
    counts = [int((labels == i).sum()) for i in range(3)]
    return {
        "image": Path(gt_path).stem,
        "psnr": psnr,
        "psnr_y": psnr_y,
        "route_counts": counts,
        "num_patches": int(len(labels)),
        "sr_shape": list(sr_img.shape),
        "gt_shape": list(gt_crop.shape),
    }


def calibrate_int8(model, dataset_info, device, scale, patch_size, step, batch_size, calib_patches,
                   act_percentile=None, quantize_classifier=False):
    qsim = QuantSim(model, act_percentile=act_percentile, quantize_classifier=quantize_classifier)
    seen = 0
    model.eval()
    with torch.no_grad(), qsim.collect_context():
        for lr_path, _gt_path in dataset_info["pairs"]:
            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
            if lr_img is None:
                continue
            if lr_img.ndim == 2:
                lr_img = np.expand_dims(lr_img, axis=2)
            if lr_img.shape[2] > 3:
                lr_img = lr_img[:, :, :3]
            patches, _, _, _, _ = crop_cpu(lr_img, patch_size, step)
            for start in range(0, len(patches), batch_size):
                if seen >= calib_patches:
                    break
                batch_patches = patches[start:start + min(batch_size, calib_patches - seen)]
                x = patches_to_tensor(batch_patches, device)
                if quantize_classifier:
                    _ = model.classifier(x)
                # Calibrate all branch specialists, not only the current router path.
                y1 = model.net1(x)
                y2 = model.net2(x)
                y3 = model.net3(x)
                qsim._record("branch_output", y1)
                qsim._record("branch_output", y2)
                qsim._record("branch_output", y3)
                seen += x.shape[0]
            if seen >= calib_patches:
                break
    qsim.finalize_ranges()
    qsim.quantize_weights()
    qsim.notes.append("Calibration dataset: {}; calibration patches used: {}.".format(dataset_info["name"], seen))
    return qsim


def evaluate_dataset(model, dataset_info, device, scale, patch_size, step, batch_size, precision,
                     fixed_routes=True, quant_sim=None, max_images=None):
    rows = []
    all_psnr = []
    all_psnr_y = []
    route = [0, 0, 0]
    notes = []
    total = len(dataset_info["pairs"]) if max_images is None else min(max_images, len(dataset_info["pairs"]))
    t0 = time.time()
    for idx, (lr_path, gt_path) in enumerate(dataset_info["pairs"][:total]):
        # Always use FP32 classifier labels for the main fixed-route experiment.
        fp32_labels = None
        if fixed_routes:
            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED)
            patches, _, _, _, _ = crop_cpu(lr_img, patch_size, step)
            x = patches_to_tensor(patches, device)
            fp32_labels, _ = run_classifier_fp32(model, x, batch_size)
            del x
        item = evaluate_image(
            model, lr_path, gt_path, device, scale, patch_size, step, batch_size,
            precision=precision, fixed_routes=fixed_routes, fp32_route_cache=fp32_labels,
            quant_sim=quant_sim, notes=notes,
        )
        rows.append(item)
        all_psnr.append(item["psnr"])
        all_psnr_y.append(item["psnr_y"])
        route = [route[i] + item["route_counts"][i] for i in range(3)]
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            elapsed = time.time() - t0
            print("[{}][{}] {}/{} images, PSNR {:.4f}, PSNR_Y {:.4f}, route {}/{}/{}, {:.1f}s".format(
                dataset_info["name"], precision, idx + 1, total,
                sum(all_psnr) / len(all_psnr), sum(all_psnr_y) / len(all_psnr_y),
                route[0], route[1], route[2], elapsed
            ), flush=True)
    flops_m = weighted_flops(route)
    nroute = float(sum(route))
    summary = {
        "dataset": dataset_info["name"],
        "precision": precision,
        "psnr_rgb": float(sum(all_psnr) / len(all_psnr)),
        "psnr_y": float(sum(all_psnr_y) / len(all_psnr_y)),
        "avg_flops_per_subregion_m": flops_m,
        "simple_count": route[0],
        "medium_count": route[1],
        "hard_count": route[2],
        "simple_ratio": route[0] / nroute if nroute else 0.0,
        "medium_ratio": route[1] / nroute if nroute else 0.0,
        "hard_ratio": route[2] / nroute if nroute else 0.0,
        "num_images": len(rows),
        "num_subregions": int(sum(route)),
        "notes": "; ".join(sorted(set(notes))) if notes else "",
    }
    return summary, rows


def write_outputs(summary_rows, per_image_rows, output_csv, output_md, output_json, env_info, run_notes):
    output_csv = Path(output_csv)
    output_md = Path(output_md)
    output_json = Path(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "Dataset", "Precision", "PSNR_RGB", "Delta_PSNR_RGB_vs_FP32", "PSNR_Y", "Delta_PSNR_Y_vs_FP32",
        "Avg_FLOPs_per_Subregion_M", "Simple_Ratio", "Medium_Ratio", "Hard_Ratio", "Route_e/m/h", "Notes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({
                "Dataset": r["dataset"],
                "Precision": r["precision"],
                "PSNR_RGB": "{:.6f}".format(r["psnr_rgb"]),
                "Delta_PSNR_RGB_vs_FP32": "{:.6f}".format(r["delta_psnr_rgb_vs_fp32"]),
                "PSNR_Y": "{:.6f}".format(r["psnr_y"]),
                "Delta_PSNR_Y_vs_FP32": "{:.6f}".format(r["delta_psnr_y_vs_fp32"]),
                "Avg_FLOPs_per_Subregion_M": "{:.2f}".format(r["avg_flops_per_subregion_m"]),
                "Simple_Ratio": "{:.6f}".format(r["simple_ratio"]),
                "Medium_Ratio": "{:.6f}".format(r["medium_ratio"]),
                "Hard_Ratio": "{:.6f}".format(r["hard_ratio"]),
                "Route_e/m/h": "{}/{}/{}".format(r["simple_count"], r["medium_count"], r["hard_count"]),
                "Notes": r.get("notes", ""),
            })

    per_image_csv = output_csv.with_name(output_csv.stem + "_per_image.csv")
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        fields_img = ["dataset", "precision", "image", "psnr", "psnr_y", "route_easy", "route_medium", "route_hard", "num_patches"]
        writer = csv.DictWriter(f, fieldnames=fields_img)
        writer.writeheader()
        for r in per_image_rows:
            writer.writerow(r)

    payload = {
        "date": DATE,
        "environment": env_info,
        "model_key": MODEL_KEY,
        "branch_flops_m": BRANCH_FLOPS_M,
        "classifier_flops_m": CLASSIFIER_FLOPS_M,
        "notes": run_notes,
        "summary_rows": summary_rows,
        "per_image_csv": str(per_image_csv),
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# ClassIC-SR Quantization Robustness")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Model: `{}` / ClassIC-SR Version A.".format(MODEL_KEY))
    lines.append("- Checkpoint: `{}`.".format(env_info.get("checkpoint", "")))
    lines.append("- No retraining; architecture and ClassSR classifier are unchanged.")
    lines.append("- Main table uses fixed FP32 classifier branch labels for BF16 and INT8-sim to isolate SR branch numerical error.")
    lines.append("- Metrics use repository PSNR/PSNR-Y path: crop border = scale = 4; RGB PSNR on BGR image arrays; Y via `data.util.bgr2ycbcr`.")
    lines.append("- FLOPs use repository Version A convention: weighted branch FLOPs plus 7.4M classifier overhead; bilinear interpolation arithmetic is excluded.")
    lines.append("- INT8-sim uses q-deq fake quantization, not backend INT8 kernels; Conv/Deconv weights per-output-channel symmetric INT8, activations per-tensor symmetric INT8, bias FP32.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for k, v in env_info.items():
        lines.append("- {}: `{}`".format(k, v))
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Dataset | Precision | PSNR_RGB | Delta RGB | PSNR_Y | Delta Y | FLOPs/subregion(M) | Route e/m/h | Simple % | Medium % | Hard % | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in summary_rows:
        lines.append(
            "| {dataset} | {precision} | {psnr_rgb:.6f} | {delta_psnr_rgb_vs_fp32:+.6f} | "
            "{psnr_y:.6f} | {delta_psnr_y_vs_fp32:+.6f} | {avg_flops_per_subregion_m:.2f} | "
            "{simple_count}/{medium_count}/{hard_count} | {s:.2f}% | {m:.2f}% | {h:.2f}% | {notes} |".format(
                s=100 * r["simple_ratio"], m=100 * r["medium_ratio"], h=100 * r["hard_ratio"], **r
            )
        )
    lines.append("")
    lines.append("## FP32 Sanity Against Existing Paper Numbers")
    lines.append("")
    lines.append("| Dataset | Expected PSNR_RGB | Actual PSNR_RGB | Diff | Expected PSNR_Y | Actual PSNR_Y | Diff | Expected FLOPs(M) | Actual FLOPs(M) | Diff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in [x for x in summary_rows if x["precision"] == "fp32"]:
        exp = EXPECTED_FP32.get(r["dataset"])
        if exp:
            lines.append("| {ds} | {ep:.4f} | {ap:.6f} | {dp:+.6f} | {ey:.4f} | {ay:.6f} | {dy:+.6f} | {ef:.2f} | {af:.2f} | {df:+.2f} |".format(
                ds=r["dataset"], ep=exp["psnr"], ap=r["psnr_rgb"], dp=r["psnr_rgb"] - exp["psnr"],
                ey=exp["psnr_y"], ay=r["psnr_y"], dy=r["psnr_y"] - exp["psnr_y"],
                ef=exp["flops_m"], af=r["avg_flops_per_subregion_m"], df=r["avg_flops_per_subregion_m"] - exp["flops_m"]
            ))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for n in run_notes:
        lines.append("- " + n)
    lines.append("")
    lines.append("## Pass/Fail")
    lines.append("")
    lines.append("- PASS if FP32 sanity matches existing benchmark within small numerical tolerance and BF16/INT8-sim deltas remain acceptable for the target paper/hardware claim.")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(per_image_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16", "int8_sim", "all"], default="all")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--data_root", default=str(ROOT))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=["DIV2K_valid", "Test2K", "Test4K", "Test8K"])
    parser.add_argument("--fixed_routes", type=str2bool, default=True)
    parser.add_argument("--quantize_classifier", type=str2bool, default=False)
    parser.add_argument("--calib_dataset", default="DIV2K_valid")
    parser.add_argument("--calib_patches", type=int, default=1000)
    parser.add_argument("--act_percentile", type=float, default=None)
    parser.add_argument("--interp_coeff_bits", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--step", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=None, help="Debug only: limit images per dataset.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", default=str(DEFAULT_RESULTS / "classic_sr_quantization_results.csv"))
    parser.add_argument("--output_md", default=str(DEFAULT_RESULTS / "classic_sr_quantization_table.md"))
    parser.add_argument("--output_json", default=str(DEFAULT_RESULTS / "classic_sr_quantization_results.json"))
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    precisions = [args.precision] if args.precision != "all" else ["fp32", "bf16", "int8_sim"]
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            "Checkpoint not found: {}. Please download the approved pretrained checkpoint "
            "and place it under pretrained/.".format(args.checkpoint)
        )
    if "bf16" in precisions and not torch.cuda.is_available():
        raise RuntimeError(
            "BF16 evaluation requires CUDA autocast support. Falling back is not enabled. "
            "Please use FP32 or run on a CUDA-enabled GPU."
        )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    env_info = {
        "cwd": os.getcwd(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    print(json.dumps(env_info, indent=2, ensure_ascii=False), flush=True)

    datasets = {name: resolve_dataset(args.data_root, name) for name in args.datasets}
    calib_info = resolve_dataset(args.data_root, args.calib_dataset) if "int8_sim" in precisions else None
    summary_rows = []
    per_image_rows = []
    run_notes = []
    if args.interp_coeff_bits is not None:
        run_notes.append("--interp_coeff_bits={} requested; coefficient quantization is not implemented, so bilinear coefficients remain FP32 while interpolation input/output tensors are q-deq quantized.".format(args.interp_coeff_bits))

    fp32_by_dataset = {}
    int8_qsim = None
    for precision in precisions:
        model, load_info = load_model(args.checkpoint, device)
        if load_info["missing_keys"] or load_info["unexpected_keys"]:
            run_notes.append("Load checkpoint with strict=False: missing={} unexpected={}".format(
                len(load_info["missing_keys"]), len(load_info["unexpected_keys"])
            ))
        if precision == "int8_sim":
            print("Calibrating INT8_sim on {} patches from {}...".format(args.calib_patches, args.calib_dataset), flush=True)
            int8_qsim = calibrate_int8(
                model, calib_info, device, args.scale, args.patch_size, args.step, args.batch_size,
                args.calib_patches, act_percentile=args.act_percentile,
                quantize_classifier=args.quantize_classifier,
            )
            run_notes.extend(int8_qsim.notes)
        for ds_name, ds_info in datasets.items():
            summary, images = evaluate_dataset(
                model, ds_info, device, args.scale, args.patch_size, args.step, args.batch_size,
                precision=precision, fixed_routes=args.fixed_routes,
                quant_sim=int8_qsim if precision == "int8_sim" else None,
                max_images=args.max_images,
            )
            if precision == "fp32":
                fp32_by_dataset[ds_name] = summary
            base = fp32_by_dataset.get(ds_name)
            if base is None and precision != "fp32":
                # If the user runs a single non-FP32 mode, use existing expected values only for display deltas.
                base = {"psnr_rgb": EXPECTED_FP32.get(ds_name, {}).get("psnr", summary["psnr_rgb"]),
                        "psnr_y": EXPECTED_FP32.get(ds_name, {}).get("psnr_y", summary["psnr_y"])}
            summary["delta_psnr_rgb_vs_fp32"] = 0.0 if precision == "fp32" else summary["psnr_rgb"] - base["psnr_rgb"]
            summary["delta_psnr_y_vs_fp32"] = 0.0 if precision == "fp32" else summary["psnr_y"] - base["psnr_y"]
            summary_rows.append(summary)
            for im in images:
                per_image_rows.append({
                    "dataset": ds_name,
                    "precision": precision,
                    "image": im["image"],
                    "psnr": "{:.6f}".format(im["psnr"]),
                    "psnr_y": "{:.6f}".format(im["psnr_y"]),
                    "route_easy": im["route_counts"][0],
                    "route_medium": im["route_counts"][1],
                    "route_hard": im["route_counts"][2],
                    "num_patches": im["num_patches"],
                })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_image_csv = write_outputs(
        summary_rows, per_image_rows, args.output_csv, args.output_md, args.output_json, env_info, run_notes
    )
    print("Wrote CSV: {}".format(args.output_csv), flush=True)
    print("Wrote per-image CSV: {}".format(per_image_csv), flush=True)
    print("Wrote Markdown: {}".format(args.output_md), flush=True)
    print("Wrote JSON: {}".format(args.output_json), flush=True)


if __name__ == "__main__":
    main()
