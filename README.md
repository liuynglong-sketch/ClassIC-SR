# ClassIC-SR

ClassIC-SR is the algorithm-level implementation of the proposed Classification-Interpolation-CNN Super-Resolution network. It follows a region-aware evaluation protocol and provides the ClassSR-FSRCNN reference implementation for controlled comparison.

This repository provides the algorithm-level ClassIC-SR implementation for network evaluation and reproducibility. Circuit-level is not included.

## Scope

Included:

- ClassIC-SR  network definition.
- ClassSR-FSRCNN reference network definition for controlled algorithm comparison.
- Dataset loaders and preprocessing-compatible utilities.
- Training and evaluation entry points.
- PSNR, PSNR-Y, route distribution, FLOPs, and parameter-count utilities.
- FP32, BF16, and INT8 network-level inference evaluation.
- Example configs for DIV2K_valid, Test2K, Test4K, and Test8K x4 evaluation.

Not included:

- Benchmark image files or downloaded datasets.
- Large trained weights by default. Put approved checkpoints under `pretrained/`.
- CIM macro RTL, Verilog, netlist, layout, SPICE, SRAM compiler files, PDK files, chip measurement logs, power-estimation scripts, or hardware power calculation notebooks.

## Model

Main model key:

```text
classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net
```

Branch structure:

```text
Simple branch:
  x4 bilinear interpolation
  Conv1x1 3->8
  PReLU
  Conv3x3 8->3
  residual add to bilinear HR base

Medium branch:
  FSRCNN LR body, d=36, s=12, m=4
  x2 bilinear interpolation
  Conv3x3 36->24
  PReLU
  x2 bilinear interpolation
  Conv3x3 24->3

Hard branch:
  Original FSRCNN branch, d=56, s=12, m=4
  ConvTranspose2d 9x9, stride=4

Classifier:
  Original ClassSR classifier, unchanged.
```

## Installation

```bash
conda create -n classic-sr python=3.8 -y
conda activate classic-sr
pip install -r requirements.txt
```

The original experiments used PyTorch 1.10.x with CUDA. Newer PyTorch versions should work for most inference tasks, but exact numeric reproduction should be checked against the provided metric scripts.

## Dataset Layout

This repository does not include datasets. Prepare x4 LR/HR pairs using this layout or edit the YAML paths:

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

The quantization evaluation script also supports the historical local layout:

```text
Test2K4K8K/
  test2k/HR/X4/
  test2k/LR/X4/
  test4k/HR/X4/
  test4k/LR/X4/
  test8k/HR/X4/
  test8k/LR/X4/
```

See `docs/dataset_preparation.md` for details.

## Checkpoints

Approved public checkpoints can be placed under:

```text
pretrained/classic_sr_version_a.pth
```

If no checkpoint is provided, the architecture and profiling scripts still run, but benchmark PSNR will not reproduce trained results.

## Evaluation

Run from the repository root:

```bash
python codes/eval_classic_sr.py -opt configs/test_classic_sr_x4.yml
```

Outputs are written to `results/test_classic_sr_x4/`.

## Quantization Robustness Evaluation

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

Notes:

- `fp32` is the baseline inference path.
- `bf16` uses PyTorch autocast when supported.
- `int8_sim` uses fake quantize/dequantize simulation for network weights and activations; it does not depend on backend INT8 kernels.
- With `--fixed_routes true`, BF16 and INT8-sim reuse FP32 classifier-selected branches to isolate reconstruction-branch numerical error.

## Parameter Count

```bash
python tools/profile_classic_sr_params.py --output results/param_profile
```

## FLOPs Calculation

Route-weighted FLOPs use the same convention as the project experiments: Conv/Deconv multiply-adds are counted as 2 FLOPs, and bilinear interpolation arithmetic is excluded from the headline FLOPs table.

```bash
python tools/compute_classic_sr_flops.py --route 89561 154952 51899
```

## Expected Results

Test8K x4:

- PSNR: 32.64 dB
- Average FLOPs: 145.52 M
- FLOPs reduction over ClassSR-FSRCNN: 43.26%

Quantization robustness:

| Dataset | FP32 | BF16 | INT8 |
|---|---:|---:|---:|
| Test2K | 25.6058 | 25.6025 | 24.9282 |
| Test4K | 26.9062 | 26.9012 | 26.0767 |
| Test8K | 32.6444 | 32.6238 | 30.5352 |

INT8 denotes the repository's network-level INT8 evaluation path. Hardware power estimation, SRAM compiler characterization, and chip measurement data are not included in this repository.

## Training

Example:

```bash
python codes/train_classic_sr.py -opt configs/train_classic_sr_x4.yml
```

The training config expects pre-trained branch checkpoints if you follow the staged training workflow. Paths are placeholders and should be changed to your local approved checkpoints.

## Documentation

- `docs/reproducibility.md`: evaluation protocol and precision modes.
- `docs/dataset_preparation.md`: dataset directory requirements.
- `docs/model_card.md`: model scope and intended use.

## Citation

See `CITATION.cff`.

## License

This repository is released under the MIT License unless a dependency or imported upstream component states otherwise.
