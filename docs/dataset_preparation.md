# Dataset Preparation

Datasets are not included in this repository. Download or prepare the benchmark images separately, then place LR/HR x4 pairs under the repository root.

## Required Folders

The recommended root is `datasets/`:

```text
datasets/
  DIV2K/
    DIV2K_valid_HR/
      0801.png
      ...
    DIV2K_valid_LR_bicubic/X4/
      0801.png
      ...
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

## Naming Assumptions

- LR and HR images must have matching file names.
- For x4 SR, LR width and height should be one quarter of the HR width and height.
- Supported image extensions include PNG, JPG, JPEG, BMP, and PPM in the underlying tools.

## Historical Layout Also Supported

`tools/eval_classic_sr_quantization.py` and `tester.py` also search this historical layout:

```text
Test2K4K8K/
  test2k/HR/X4/
  test2k/LR/X4/
  test4k/HR/X4/
  test4k/LR/X4/
  test8k/HR/X4/
  test8k/LR/X4/
```

For DIV2K_valid, the tools also search:

```text
data/valid/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4/
```

## Debugging Missing Dataset Errors

If `tester.py` reports a missing dataset path, check:

1. The `--data_root` argument points to the intended dataset root. For the README layout, use `--data_root datasets`.
2. Both `HR/X4` and `LR/X4` folders exist for Test2K/Test4K/Test8K.
3. LR and HR file names match exactly.
4. The dataset name passed to `--dataset` is one of `DIV2K_valid`, `Test2K`, `Test4K`, or `Test8K`.

Example:

```bash
python tester.py \
  --checkpoint pretrained/classic_sr_version_a.pth \
  --data_root datasets \
  --dataset Test8K \
  --precision fp32 \
  --output_dir results/test8k_fp32
```
