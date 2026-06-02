# Reproducibility

This document describes algorithm-level inference and evaluation reproduction for ClassIC-SR. Circuit-level CIM macro files, SRAM compiler data, PDK files, chip measurement logs, and hardware power-estimation scripts are outside the scope of this repository.


## Release Scope

This repository is an inference/evaluation-only release. It includes network definitions, pretrained-checkpoint loading, PSNR/FLOPs/parameter profiling, and FP32/BF16/INT8 evaluation utilities. Training scripts and training configs are intentionally not included.

## Recommended Public Entry Point

Use `tester.py` from the repository root. It wraps the underlying evaluation utilities and writes compact output files under `results/`.

FP32 Test8K example:

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision fp32 \
  --output_dir results/test8k_fp32
```

BF16 Test8K example:

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision bf16 \
  --output_dir results/test8k_bf16
```

INT8 Test8K example:

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision int8 \
  --output_dir results/test8k_int8 \
  --fixed_routes true
```

All precision modes:

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision all \
  --output_dir results/test8k_all \
  --fixed_routes true
```

## Dataset Layout

Expected dataset root:

```text
datasets/
  DIV2K/
    DIV2K_valid_HR/
    DIV2K_valid_LR_bicubic/X4/
  Test2K/
    HR/X4/
    LR/X4/
  Test4K/
    HR/X4/
    LR/X4/
  Test8K/
    HR/X4/
    LR/X4/
```

The scripts also support the historical layout documented in `docs/dataset_preparation.md`.

## Fixed-route Evaluation

With `--fixed_routes true`, the FP32 classifier is run first and its selected simple/medium/hard branch labels are reused for BF16 and INT8 evaluation. This isolates reconstruction-branch numerical precision effects from router changes.

## Metrics

- RGB PSNR is computed after cropping the scale border.
- PSNR-Y converts images to the Y channel using the repository `bgr2ycbcr` implementation, then computes PSNR after border crop.
- The default x4 crop border is 4 pixels.
- SSIM is not part of the default public release table unless explicitly implemented and reported by a script.

## FLOPs Convention

- Conv2d, ConvTranspose2d, and Linear multiply-adds are counted as 2 FLOPs.
- Route-weighted FLOPs use the measured simple/medium/hard route distribution.
- Bilinear interpolation arithmetic is excluded from the headline FLOPs table, matching the current project convention.

## Expected Values

Test8K x4:

- FP32 PSNR: approximately 32.64 dB.
- Average FLOPs: approximately 145.52 M.
- FLOPs reduction over ClassSR-FSRCNN: approximately 43.26%.

Quantization robustness:

| Dataset | FP32 | BF16 | INT8 |
|---|---:|---:|---:|
| Test2K | 25.6058 | 25.6025 | 24.9282 |
| Test4K | 26.9062 | 26.9012 | 26.0767 |
| Test8K | 32.6444 | 32.6238 | 30.5352 |

## Profiling

```bash
python tools/profile_classic_sr_params.py --output results/param_profile
python tools/compute_classic_sr_flops.py --route 89561 154952 51899
```
