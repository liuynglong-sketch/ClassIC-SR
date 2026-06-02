# Reproducibility

This document describes algorithm-level reproduction for ClassIC-SR. Circuit-level CIM macro files, SRAM compiler data, PDK files, chip measurement logs, and hardware power-estimation scripts are outside the scope of this repository.

## Evaluation Protocol

- Task: x4 single-image super-resolution.
- Datasets: `DIV2K_valid`, `Test2K`, `Test4K`, and `Test8K`.
- Patch protocol: ClassIC-SR evaluates LR sub-regions and routes each region to the simple, medium, or hard branch.
- Classifier: the ClassSR-FSRCNN classifier architecture is unchanged.
- Checkpoint location: `pretrained/classic_sr_version_a.pth`.

Example command:

```bash
python codes/eval_classic_sr.py -opt configs/test_classic_sr_x4.yml
```

## Metrics

- RGB PSNR is computed after cropping the scale border.
- PSNR-Y converts RGB/BGR images to the Y channel using the repository `bgr2ycbcr` implementation, then computes PSNR after border crop.
- The default x4 crop border is 4 pixels.
- SSIM is not part of the default release table unless explicitly implemented and reported by a script.

## FLOPs Convention

- Conv2d, ConvTranspose2d, and Linear multiply-adds are counted as 2 FLOPs.
- Route-weighted FLOPs use the measured simple/medium/hard route distribution.
- Bilinear interpolation arithmetic is excluded from the headline FLOPs table, matching the current project convention.
- If interpolation arithmetic is included in future work, report it as a separate column to avoid mixing conventions.

## FP32 / BF16 / INT8 Evaluation

Run all precision modes:

```bash
python tools/eval_classic_sr_quantization.py \
  --precision all \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root . \
  --scale 4 \
  --datasets DIV2K_valid Test2K Test4K Test8K \
  --fixed_routes true \
  --calib_dataset DIV2K_valid \
  --calib_patches 1000 \
  --output_csv results/classic_sr_quantization_results.csv \
  --output_md results/classic_sr_quantization_table.md
```

### Fixed-route evaluation

With `--fixed_routes true`, the FP32 classifier is run first and its selected simple/medium/hard branch labels are reused for BF16 and INT8 evaluation. This isolates reconstruction-branch numerical precision effects from router changes.

### Precision modes

- `fp32`: standard floating-point inference.
- `bf16`: PyTorch BF16 autocast where supported.
- `int8_sim`: fake quantize/dequantize network-level simulation. It does not require backend INT8 kernels.

## Parameter and FLOPs Utilities

```bash
python tools/profile_classic_sr_params.py --output results/param_profile
python tools/compute_classic_sr_flops.py --route 89561 154952 51899
```
