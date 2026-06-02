# Model Card: ClassIC-SR

## Model Name

ClassIC-SR: Classification-Interpolation-CNN Super-Resolution.

## Task

x4 image super-resolution.

## Architecture Summary

ClassIC-SR uses region-aware dynamic inference with three reconstruction branches:

- Simple branch: x4 bilinear interpolation plus lightweight CNN residual compensation.
- Medium branch: FSRCNN-style LR feature body with interpolation-CNN upsampling compensation.
- Hard branch: original FSRCNN-style hard branch with ConvTranspose2d upsampling.

The classifier architecture is unchanged from the ClassSR-FSRCNN reference and selects one branch per sub-region.

## Supported Evaluation Precision

- FP32 inference.
- BF16 inference on CUDA devices with autocast support.
- INT8 network-level fake quantization simulation.

## Intended Use

This repository is intended for research and algorithm reproduction of ClassIC-SR network behavior, including PSNR, PSNR-Y, route distribution, parameter count, FLOPs, and network-level quantization robustness.

## Not Included

This repository does not include chip design files, SRAM compiler files, PDK files, foundry files, chip measurement data, oscilloscope logs, FPGA raw logs, or power-estimation scripts.

## Limitations

- Public reproduction requires users to provide datasets and approved checkpoints separately.
- INT8 results are network-level fake quantization results and are not a substitute for backend-specific integer kernel benchmarking.
- Headline FLOPs exclude bilinear interpolation arithmetic under the current project convention.
