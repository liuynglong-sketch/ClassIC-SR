#!/usr/bin/env python3
"""Count parameters for ClassSR-FSRCNN and ClassIC-SR / Version A."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
if str(CODES) not in sys.path:
    sys.path.insert(0, str(CODES))

from models.archs import classSR_fsrcnn_arch as arch  # noqa: E402

MODELS = [
    ("ClassSR-FSRCNN", "classSR_3class_fsrcnn_net", arch.classSR_3class_fsrcnn_net),
    (
        "ClassIC-SR / Version A",
        "classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net",
        arch.classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net,
    ),
]


def nparams(module, trainable=None):
    if trainable is None:
        return sum(p.numel() for p in module.parameters())
    return sum(p.numel() for p in module.parameters() if p.requires_grad is trainable)


def count_prefix(model, prefix):
    return sum(p.numel() for name, p in model.named_parameters() if name.startswith(prefix))


def prelu_params(model):
    return sum(sum(p.numel() for p in m.parameters()) for m in model.modules() if isinstance(m, nn.PReLU))


def profile(label, key, ctor):
    model = ctor(in_nc=3, out_nc=3)
    easy = count_prefix(model, "net1.")
    medium = count_prefix(model, "net2.")
    hard = count_prefix(model, "net3.")
    return {
        "model": label,
        "model_key": key,
        "total_params": nparams(model),
        "classifier_params": count_prefix(model, "classifier."),
        "sr_module_params": easy + medium + hard,
        "easy_branch_params": easy,
        "medium_branch_params": medium,
        "hard_branch_params": hard,
        "prelu_params": prelu_params(model),
        "trainable_params": nparams(model, True),
        "non_trainable_params": nparams(model, False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/param_profile")
    args = parser.parse_args()
    out_prefix = ROOT / args.output
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [profile(*m) for m in MODELS]
    payload = {"classifier_identical": len({r["classifier_params"] for r in rows}) == 1, "rows": rows}
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
