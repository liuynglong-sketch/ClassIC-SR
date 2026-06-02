# Dataset Preparation

Datasets are not included in this repository. Download or prepare the benchmark images separately, then place LR/HR x4 pairs under the repository root.

## Expected Root

The default data root is the repository root. Evaluation scripts expect datasets under `./datasets` unless `--data_root` or YAML paths are changed.

## Recommended Layout

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

The LR and HR images in each pair should have matching file names. For x4 SR, LR width and height should be one quarter of the HR width and height.

## Historical Layout Also Supported

`tools/eval_classic_sr_quantization.py` also searches this historical layout:

```text
Test2K4K8K/
  test2k/HR/X4/
  test2k/LR/X4/
  test4k/HR/X4/
  test4k/LR/X4/
  test8k/HR/X4/
  test8k/LR/X4/
```

For DIV2K_valid, it also searches:

```text
data/valid/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4/
```

## Config Files

The default config uses the recommended layout:

```bash
python codes/eval_classic_sr.py -opt configs/test_classic_sr_x4.yml
```

If your dataset is stored elsewhere, edit `dataroot_GT` and `dataroot_LQ` in the YAML file or pass the appropriate `--data_root` to tool scripts.
