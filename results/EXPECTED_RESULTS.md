# Expected Results

## Main Test8K x4 Result

- PSNR: 32.64 dB
- Average FLOPs: 145.52 M
- FLOPs reduction over ClassSR-FSRCNN: 43.26%

## Quantization Robustness

| Dataset | FP32 | BF16 | INT8 |
|---|---:|---:|---:|
| Test2K | 25.6058 | 25.6025 | 24.9282 |
| Test4K | 26.9062 | 26.9012 | 26.0767 |
| Test8K | 32.6444 | 32.6238 | 30.5352 |

Small numerical differences may occur due to PyTorch/CUDA versions, image I/O libraries, and hardware environment.
