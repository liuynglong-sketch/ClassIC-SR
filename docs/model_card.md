# Model Card: ClassIC-SR

## Model Name

ClassIC-SR: Classification-Interpolation-CNN Super-Resolution.

## Task

x4 single-image super-resolution.

## Architecture Summary

ClassIC-SR uses region-aware dynamic inference with three reconstruction branches:

- Simple branch: x4 bilinear interpolation plus lightweight CNN residual compensation.
- Medium branch: FSRCNN-style LR feature body with interpolation-CNN upsampling compensation.
- Hard branch: original FSRCNN-style hard branch with ConvTranspose2d upsampling.

The classifier architecture is unchanged from the ClassSR-FSRCNN reference and selects one branch per sub-region.

## Precision Modes

The repository supports algorithm-level network evaluation for:

- FP32 inference.
- BF16 inference where supported by PyTorch/CUDA.
- INT8 network-level fake quantization simulation.

## Intended Use

This repository is intended for algorithm-level reproduction, research comparison, and network evaluation of ClassIC-SR.

## Not Included

This repository does not include CIM macro RTL, Verilog, netlist, layout, SPICE, SRAM compiler data, PDK files, chip measurement data, oscilloscope logs, FPGA raw logs, power-estimation scripts, or hardware energy calculation notebooks.

## Limitations

- Public reproduction requires users to provide datasets and approved checkpoints separately.
- INT8 results are network-level fake quantization results and are not a substitute for backend-specific integer kernel benchmarking.
- Headline FLOPs exclude bilinear interpolation arithmetic under the current project convention.
