# Reproducibility Notes

Run all commands from the repository root.

1. Put x4 LR/HR image pairs under `datasets/` following `README.md`.
2. Put an approved ClassIC-SR checkpoint at `pretrained/classic_sr_version_a.pth`.
3. Run `python codes/eval_classic_sr.py -opt configs/test_classic_sr_x4.yml` for PSNR and route distribution.
4. Run `python tools/profile_classic_sr_params.py --output results/param_profile` for parameter counts.
5. Run `python tools/eval_classic_sr_quantization.py --precision all ...` for FP32/BF16/INT8-sim network evaluation.

The repository is intended for network-level reproduction only. Dataset files and trained checkpoints are external assets.
