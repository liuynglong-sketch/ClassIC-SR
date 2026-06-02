#!/usr/bin/env python3
"""User-facing ClassIC-SR evaluation entry point.

This wrapper keeps the public workflow simple while reusing the repository's
existing quantization/evaluation script.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - only used for clear CLI errors.
    torch = None

DATASETS = ["DIV2K_valid", "Test2K", "Test4K", "Test8K"]
PRECISIONS = ["fp32", "bf16", "int8", "all"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def prompt_value(label, default=None, choices=None):
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            value = default
        if choices and value not in choices:
            print(f"Please choose one of: {', '.join(choices)}")
            continue
        if value:
            return value


def parse_args():
    parser = argparse.ArgumentParser(description="Run ClassIC-SR evaluation with a simple public interface.")
    parser.add_argument("--checkpoint", help="Path to pretrained checkpoint, e.g. pretrained/classic_sr_version_a.pth")
    parser.add_argument("--data_root", help="Dataset root, usually datasets")
    parser.add_argument("--dataset", choices=DATASETS, help="Dataset to evaluate")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--precision", choices=PRECISIONS, help="Precision mode: fp32, bf16, int8, or all")
    parser.add_argument("--output_dir", help="Directory for metrics outputs")
    parser.add_argument("--fixed_routes", type=str2bool, default=True)
    parser.add_argument("--calib_dataset", choices=DATASETS, default="DIV2K_valid", help="Calibration dataset for INT8/all")
    parser.add_argument("--calib_patches", type=int, default=1000)
    parser.add_argument("--max_images", type=int, default=None, help="Debug only: limit images per dataset")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("Interactive ClassIC-SR tester")
        args.checkpoint = prompt_value("Enter checkpoint path", "pretrained/classic_sr_version_a.pth")
        args.data_root = prompt_value("Enter dataset root", "datasets")
        args.dataset = prompt_value("Select dataset", "Test8K", DATASETS)
        args.precision = prompt_value("Select precision", "fp32", PRECISIONS)
        args.output_dir = prompt_value("Enter output directory", f"results/{args.dataset.lower()}_{args.precision}")
        args.fixed_routes = True
    else:
        missing = [name for name in ["checkpoint", "data_root", "dataset", "precision", "output_dir"] if getattr(args, name) in (None, "")]
        if missing:
            parser.error("missing required arguments: " + ", ".join("--" + m for m in missing))
    return args


def load_resolver():
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    from tools.eval_classic_sr_quantization import resolve_dataset  # noqa: WPS433
    return resolve_dataset


def validate_inputs(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Please download the approved pretrained checkpoint "
            "and place it under pretrained/. See README.md and pretrained/README.md."
        )
    if checkpoint.stat().st_size < 1024:
        raise ValueError(
            f"Checkpoint file is too small and is likely an incomplete download: {checkpoint}. "
            "Please re-download it from the GitHub Release page."
        )
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}. Expected folders such as datasets/Test8K/HR/X4 and "
            "datasets/Test8K/LR/X4, or the documented historical layout. See docs/dataset_preparation.md."
        )
    if args.scale != 4:
        raise ValueError("ClassIC-SR public evaluation currently supports scale=4 only.")
    if args.precision == "bf16" and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError(
            "BF16 evaluation requires CUDA autocast support. Falling back is not enabled. "
            "Please use FP32 or run on a CUDA-enabled GPU."
        )
    if args.precision == "all" and (torch is None or not torch.cuda.is_available()):
        print("Warning: CUDA is unavailable; the BF16 part of --precision all will fail. Use --precision fp32 on CPU.")

    resolve_dataset = load_resolver()
    try:
        resolve_dataset(args.data_root, args.dataset)
    except Exception as exc:
        raise FileNotFoundError(
            "Dataset path not found. Expected one of the README layouts, for example:\n"
            f"  datasets/{args.dataset}/HR/X4 and datasets/{args.dataset}/LR/X4\n"
            "or historical layout:\n"
            f"  Test2K4K8K/{args.dataset.lower().replace('test', 'test')}/HR/X4 and "
            f"Test2K4K8K/{args.dataset.lower().replace('test', 'test')}/LR/X4\n"
            f"Detailed resolver error: {exc}"
        ) from exc

    if args.precision in {"int8", "all"}:
        try:
            resolve_dataset(args.data_root, args.calib_dataset)
        except Exception as exc:
            raise FileNotFoundError(
                f"INT8/all evaluation needs calibration dataset {args.calib_dataset}. "
                "Prepare DIV2K_valid or pass --calib_dataset to an available dataset. "
                f"Detailed resolver error: {exc}"
            ) from exc


def run_eval(args):
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision = "int8_sim" if args.precision == "int8" else args.precision
    cmd = [
        sys.executable,
        str(repo_root / "tools/eval_classic_sr_quantization.py"),
        "--precision", precision,
        "--checkpoint", str(Path(args.checkpoint)),
        "--data_root", str(Path(args.data_root)),
        "--scale", str(args.scale),
        "--datasets", args.dataset,
        "--fixed_routes", "true" if args.fixed_routes else "false",
        "--calib_dataset", args.calib_dataset,
        "--calib_patches", str(args.calib_patches),
        "--output_csv", str(output_dir / "metrics.csv"),
        "--output_md", str(output_dir / "metrics.md"),
        "--output_json", str(output_dir / "metrics_raw.json"),
    ]
    if args.max_images is not None:
        cmd.extend(["--max_images", str(args.max_images)])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return output_dir


def summarize(output_dir, dataset, requested_precision):
    raw_path = output_dir / "metrics_raw.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows", [])
    friendly_rows = []
    for row in rows:
        precision = "int8" if row["precision"] == "int8_sim" else row["precision"]
        friendly_rows.append({
            "dataset": row["dataset"],
            "precision": precision,
            "psnr_rgb": row["psnr_rgb"],
            "delta_psnr_rgb_vs_fp32": row.get("delta_psnr_rgb_vs_fp32", 0.0),
            "psnr_y": row["psnr_y"],
            "delta_psnr_y_vs_fp32": row.get("delta_psnr_y_vs_fp32", 0.0),
            "avg_flops_per_subregion_m": row["avg_flops_per_subregion_m"],
            "route_e_m_h": [row["simple_count"], row["medium_count"], row["hard_count"]],
            "route_ratio": [row["simple_ratio"], row["medium_ratio"], row["hard_ratio"]],
            "notes": row.get("notes", ""),
        })
    result = {
        "dataset": dataset,
        "requested_precision": requested_precision,
        "output_dir": str(output_dir),
        "results": friendly_rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Requested precision: {requested_precision}")
    for row in friendly_rows:
        lines.append("")
        lines.append(f"Precision: {row['precision']}")
        lines.append(f"PSNR: {row['psnr_rgb']:.6f} dB")
        lines.append(f"PSNR_Y: {row['psnr_y']:.6f} dB")
        lines.append(f"Delta RGB vs FP32: {row['delta_psnr_rgb_vs_fp32']:+.6f} dB")
        lines.append(f"Delta Y vs FP32: {row['delta_psnr_y_vs_fp32']:+.6f} dB")
        lines.append(f"Average FLOPs: {row['avg_flops_per_subregion_m']:.2f} M")
        lines.append("Route e/m/h: {}/{}/{}".format(*row["route_e_m_h"]))
        if row["notes"]:
            lines.append(f"Notes: {row['notes']}")
    lines.append("")
    lines.append(f"Output directory: {output_dir}")
    text = "\n".join(lines) + "\n"
    (output_dir / "metrics.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    args = parse_args()
    try:
        validate_inputs(args)
        output_dir = run_eval(args)
        summarize(output_dir, args.dataset, args.precision)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
