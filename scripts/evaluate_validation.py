"""Evaluate saved validation reconstructions with challenge-aligned metrics."""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full  # noqa: E402


ACC_RE = re.compile(r"_acc(4|8)_", re.IGNORECASE)


def acceleration_from_name(name):
    match = ACC_RE.search(name)
    if not match:
        raise ValueError(f"Could not determine acceleration from filename: {name}")
    return f"acc{match.group(1)}"


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def equal_acceleration_mean(values):
    return float(np.mean(values)) if len(values) == 2 and all(value is not None for value in values) else None


def render(value):
    return "N/A" if value is None else f"{value:.6f}"


def save_slice_csv(rows, path):
    fields = ["filename", "slice", "acceleration", "ssim_full", "ssim_bbox", "bbox_count"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_worst_slice_figure(rows, image_dir, recon_dir, out_path, worst_k):
    candidates = [row for row in rows if row["ssim_full"] != ""]
    candidates.sort(key=lambda row: float(row["ssim_full"]))
    selected = candidates[:worst_k]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        source_path = image_dir / row["filename"]
        recon_path = recon_dir / row["filename"]
        slice_index = int(row["slice"])
        with h5py.File(source_path, "r") as hf:
            target = hf["image_label"][slice_index]
            annotations = json.loads(hf.attrs.get("annotations", "{}"))
        with h5py.File(recon_path, "r") as hf:
            recon = hf["reconstruction"][slice_index]

        vmax = max(float(np.max(target)), 1e-12)
        error = np.abs(target - recon)
        axes[row_index, 0].imshow(target, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 1].imshow(recon, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 2].imshow(error, cmap="inferno")
        for box in annotations.get(str(slice_index), []):
            for axis in axes[row_index, :2]:
                axis.add_patch(
                    Rectangle(
                        (box["x"], box["y"]), box["width"], box["height"],
                        fill=False, edgecolor="red", linewidth=1,
                    )
                )
        axes[row_index, 0].set_title(f"Target: {row['filename']} slice {slice_index}")
        axes[row_index, 1].set_title(f"Recon: SSIM_full={float(row['ssim_full']):.4f}")
        axes[row_index, 2].set_title("Absolute error")
        for axis in axes[row_index]:
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Score saved validation reconstructions")
    parser.add_argument("--exp-name", default="varnet_c6_ch12_s4_ep80_lr3e4")
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--result-root", type=Path, default=Path("../result"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--worst-k", type=int, default=6)
    args = parser.parse_args()

    image_dir = args.data_root / "val" / "image"
    recon_dir = args.result_root / args.exp_name / "reconstructions_val"
    out_dir = args.result_root / args.exp_name / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    if not recon_dir.is_dir():
        raise FileNotFoundError(f"Validation reconstruction directory not found: {recon_dir}")

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ssim = SSIM().to(device=device)

    full_by_acc = defaultdict(list)
    bbox_by_acc = defaultdict(list)
    slice_rows = []
    recon_files = sorted(recon_dir.glob("*.h5"))
    if not recon_files:
        raise RuntimeError(f"No reconstruction H5 files found in {recon_dir}")

    with torch.no_grad():
        for recon_path in recon_files:
            source_path = image_dir / recon_path.name
            if not source_path.exists():
                raise FileNotFoundError(f"Target H5 not found for {recon_path.name}: {source_path}")
            acceleration = acceleration_from_name(recon_path.name)

            with h5py.File(source_path, "r") as hf:
                target_vol = hf["image_label"][:]
                maximum = hf.attrs["max"]
                annotations = json.loads(hf.attrs.get("annotations", "{}"))
            with h5py.File(recon_path, "r") as hf:
                recon_vol = hf["reconstruction"][:]

            if target_vol.shape != recon_vol.shape:
                raise ValueError(
                    f"Shape mismatch for {recon_path.name}: target={target_vol.shape}, recon={recon_vol.shape}"
                )

            for slice_index in range(target_vol.shape[0]):
                target_t = torch.from_numpy(target_vol[slice_index]).to(device=device)
                recon_t = torch.from_numpy(recon_vol[slice_index]).to(device=device)
                mask_t = torch.from_numpy(foreground_mask(target_vol[slice_index])).to(
                    device=device, dtype=torch.float32
                )

                full_value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
                if full_value is not None:
                    full_by_acc[acceleration].append(full_value)

                box_values = []
                for box in annotations.get(str(slice_index), []):
                    value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if value is not None:
                        bbox_by_acc[acceleration].append(value)
                        box_values.append(value)

                slice_rows.append(
                    {
                        "filename": recon_path.name,
                        "slice": slice_index,
                        "acceleration": acceleration,
                        "ssim_full": "" if full_value is None else full_value,
                        "ssim_bbox": "" if not box_values else float(np.mean(box_values)),
                        "bbox_count": len(box_values),
                    }
                )

    per_acc = {}
    for acceleration in ("acc4", "acc8"):
        per_acc[acceleration] = {
            "ssim_full": mean_or_none(full_by_acc[acceleration]),
            "ssim_bbox": mean_or_none(bbox_by_acc[acceleration]),
            "full_slice_count": len(full_by_acc[acceleration]),
            "bbox_count": len(bbox_by_acc[acceleration]),
        }

    full_scores = [per_acc[key]["ssim_full"] for key in ("acc4", "acc8")]
    bbox_scores = [per_acc[key]["ssim_bbox"] for key in ("acc4", "acc8")]
    summary = {
        "experiment": args.exp_name,
        "metric_aggregation": "equal-weight mean of acc4 and acc8",
        "ssim_full": equal_acceleration_mean(full_scores),
        "ssim_bbox": equal_acceleration_mean(bbox_scores),
        "per_acceleration": per_acc,
    }

    save_slice_csv(slice_rows, out_dir / "validation_slice_metrics.csv")
    with (out_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        f"# Validation Metrics: {args.exp_name}",
        "",
        "These metrics use the same foreground and bounding-box functions as the leaderboard evaluator.",
        "",
        "## Overall",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| SSIM_full | {render(summary['ssim_full'])} |",
        f"| SSIM_bbox | {render(summary['ssim_bbox'])} |",
        "",
        "## By acceleration",
        "",
        "| Acceleration | SSIM_full | SSIM_bbox | Full slices | Bounding boxes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for acceleration in ("acc4", "acc8"):
        values = per_acc[acceleration]
        lines.append(
            f"| {acceleration} | {render(values['ssim_full'])} | {render(values['ssim_bbox'])} "
            f"| {values['full_slice_count']} | {values['bbox_count']} |"
        )
    lines.extend(["", "Overall scores are equal-weight means of the acceleration-specific scores.", ""])
    (out_dir / "validation_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    save_worst_slice_figure(
        slice_rows, image_dir, recon_dir,
        out_dir / "validation_worst_slices.png", args.worst_k,
    )

    print(f"Validation SSIM_full: {render(summary['ssim_full'])}")
    print(f"Validation SSIM_bbox: {render(summary['ssim_bbox'])}")
    print(f"Validation analysis saved to: {out_dir}")


if __name__ == "__main__":
    main()
