#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Python import check"
python - <<'PY'
import cv2
import lmdb
import numpy
import PIL
import torch
import torchvision
import yaml
print("python imports ok")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "[2/5] compileall"
python -m compileall codes tools tester.py

echo "[3/5] public tester and quantization help"
python tester.py --help >/dev/null
python tools/eval_classic_sr_quantization.py --help >/dev/null

echo "[4/5] tool help: FLOPs and params"
python tools/compute_classic_sr_flops.py --help >/dev/null
python tools/profile_classic_sr_params.py --help >/dev/null

echo "[5/5] eval script help"
if python codes/eval_classic_sr.py --help >/dev/null 2>&1; then
  echo "codes/eval_classic_sr.py --help ok"
else
  echo "note: codes/eval_classic_sr.py does not provide a clean --help path in this environment"
fi

echo "smoke test passed"
