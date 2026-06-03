# ClassIC-SR

ClassIC-SR is the algorithm-level implementation of the proposed Classification-Interpolation-CNN Super-Resolution network. It follows a region-aware evaluation protocol and provides the ClassSR-FSRCNN reference implementation for controlled comparison.

This repository provides the algorithm-level ClassIC-SR implementation for network evaluation and reproducibility.

## Scope

Included:

- ClassIC-SR / Version A network definition.
- ClassSR-FSRCNN reference network definition for controlled algorithm comparison.
- Dataset loaders and preprocessing-compatible utilities.
- Inference and evaluation entry points.
- PSNR, PSNR-Y, route distribution, FLOPs, and parameter-count utilities.
- FP32, BF16, and INT8 network-level inference evaluation.
- Example configs for DIV2K_valid, Test2K, Test4K, and Test8K x4 evaluation.

Not included:

- Benchmark image files or downloaded datasets.
- Training scripts, training configs, private datasets, and large trained weights by default. Put approved public/reviewer checkpoints under `pretrained/`.
- CIM macro RTL, Verilog, netlist.

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

## How to Test

### 1. Clone this repository

```bash
git clone https://github.com/liuynglong-sketch/ClassIC-SR.git
cd ClassIC-SR
```

### 2. Create environment

```bash
conda create -n classic-sr python=3.8 -y
conda activate classic-sr
pip install -r requirements.txt
```

The original experiments used PyTorch 1.10.x with CUDA. Newer PyTorch versions should work for most inference tasks, but exact numeric reproduction should be checked against the provided metric scripts.

### 3. Download pretrained checkpoint

Default expected path:

```text
pretrained/classic_sr_version_a.pth
```

For peer-review or public reproduction, download the approved checkpoint from the release page or supplementary review link and place it at the default path. The v0.1.0 release provides the checkpoint used by the reported Test8K x4 results.

```bash
mkdir -p pretrained
wget -O pretrained/classic_sr_version_a.pth https://github.com/YunlongLiu-code/ClassIC-SR/releases/download/v0.1.0/classic_sr_version_a.pth
```

If `wget` fails in a restricted network environment, use GitHub CLI as a fallback:

```bash
mkdir -p pretrained
gh release download v0.1.0 \
  --repo YunlongLiu-code/ClassIC-SR \
  --pattern classic_sr_version_a.pth \
  --dir pretrained
```

If both commands are unavailable, manually download the approved checkpoint asset and save it as `pretrained/classic_sr_version_a.pth`.

### 4. Prepare datasets

Datasets are not included in this repository. Prepare x4 LR/HR image pairs using this layout:

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

The LR and HR images in each folder should have matching file names. See `docs/dataset_preparation.md` for details and historical path fallbacks.

### 5. Run FP32 test

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision fp32 \
  --output_dir results/test8k_fp32
```

Expected output:

- Test8K x4 PSNR approximately 32.64 dB.
- Average FLOPs approximately 145.52 M.

### 6. Run BF16 test

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --scale 4 \
  --precision bf16 \
  --output_dir results/test8k_bf16
```

Expected output:

- Test8K BF16 PSNR approximately 32.6238 dB.

BF16 evaluation requires CUDA autocast support.

### 7. Run INT8 test

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

Expected output:

- Test8K INT8 PSNR approximately 30.5352 dB.

`int8` denotes the repository's network-level fake-quantized INT8 evaluation path.

### 8. Run all precision modes

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

The tester writes:

```text
results/<dataset>_<precision>/metrics.json
results/<dataset>_<precision>/metrics.txt
results/<dataset>_<precision>/metrics.csv
results/<dataset>_<precision>/metrics.md
```

If you run `python tester.py` without arguments, it prompts for the checkpoint, dataset root, dataset name, precision, and output directory.

## Quick Profiling

Parameter count:

```bash
python tools/profile_classic_sr_params.py --output results/param_profile
```

Route-weighted FLOPs for the reported Test8K route distribution:

```bash
python tools/compute_classic_sr_flops.py --route 89561 154952 51899
```

Expected Test8K x4:

- Average FLOPs: 145.52 M.
- FLOPs reduction over ClassSR-FSRCNN: 43.26%.

## Expected Results

Test8K x4:

- PSNR: 32.64 dB.
- Average FLOPs: 145.52 M.
- FLOPs reduction over ClassSR-FSRCNN: 43.26%.

Quantization robustness:

| Dataset | FP32 | BF16 | INT8 |
|---|---:|---:|---:|
| Test2K | 25.6058 | 25.6025 | 24.9282 |
| Test4K | 26.9062 | 26.9012 | 26.0767 |
| Test8K | 32.6444 | 32.6238 | 30.5352 |

Small numerical differences may occur due to PyTorch/CUDA versions, image I/O libraries, and hardware environment. See `results/EXPECTED_RESULTS.md`.

## Advanced Evaluation Script

`tester.py` is the recommended public entry point. For advanced runs, call the underlying quantization script directly:

```bash
python tools/eval_classic_sr_quantization.py \
  --precision all \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --scale 4 \
  --datasets DIV2K_valid Test2K Test4K Test8K \
  --fixed_routes true \
  --calib_dataset DIV2K_valid \
  --calib_patches 1000 \
  --output_csv results/classic_sr_quantization_results.csv \
  --output_md results/classic_sr_quantization_table.md \
  --output_json results/classic_sr_quantization_results.json
```

Notes:

- `fp32` is the baseline inference path.
- `bf16` uses PyTorch CUDA autocast when supported.
- `int8_sim` uses fake quantize/dequantize simulation for network weights and activations; it does not depend on backend INT8 kernels.
- With `--fixed_routes true`, BF16 and INT8-sim reuse FP32 classifier-selected branches to isolate reconstruction-branch numerical error.

## Inference-only Release

This public repository is intended for algorithm-level inference, evaluation, profiling, and quantization robustness checks. Training entry points and training configs are intentionally not included in the public release.

## Documentation

- `docs/reproducibility.md`: evaluation protocol and precision modes.
- `docs/dataset_preparation.md`: dataset directory requirements.
- `docs/model_card.md`: model scope and intended use.
- `pretrained/README.md`: checkpoint release status.
- `results/EXPECTED_RESULTS.md`: compact expected result numbers.

## Citation

See `CITATION.cff`.

## License

This repository is released under the MIT License unless a dependency or imported upstream component states otherwise.
