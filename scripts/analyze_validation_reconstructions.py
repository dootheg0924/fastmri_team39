"""Read-only, model-centric diagnostics for saved validation reconstructions.

This script does not modify or replace the challenge evaluator. It imports the
same metric functions and recomputes slice- and box-level values only to expose
failure patterns that an aggregate score cannot show.
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


ACC_RE = re.compile(r"_acc(4|8)_", re.IGNORECASE)
ACCELERATIONS = ("acc4", "acc8")
POSITION_LABELS = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")


def acceleration_from_name(name):
    match = ACC_RE.search(name)
    if not match:
        raise ValueError(f"Could not determine acceleration from filename: {name}")
    return f"acc{match.group(1)}"


def mean_or_none(values):
    return float(statistics.fmean(values)) if values else None


def quantile_or_none(values, q):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": mean_or_none(values),
        "median": quantile_or_none(values, 0.5),
        "p05": quantile_or_none(values, 0.05),
        "p25": quantile_or_none(values, 0.25),
        "p75": quantile_or_none(values, 0.75),
        "p95": quantile_or_none(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "std": float(statistics.pstdev(values)) if values else None,
    }


def summarize_metric(rows, value_key):
    by_acceleration = {}
    for acceleration in ACCELERATIONS:
        values = [row[value_key] for row in rows if row["acceleration"] == acceleration]
        by_acceleration[acceleration] = describe(values)

    acceleration_means = [by_acceleration[acc]["mean"] for acc in ACCELERATIONS]
    if any(value is None for value in acceleration_means):
        official_score = None
    else:
        official_score = float(statistics.fmean(acceleration_means))
    return {
        "equal_acceleration_mean": official_score,
        "by_acceleration": by_acceleration,
        "pooled_distribution": describe([row[value_key] for row in rows]),
    }


def cluster_bootstrap_score(rows, value_key, samples, seed):
    """Estimate score uncertainty by resampling whole volumes within each acceleration."""
    grouped = {acc: defaultdict(list) for acc in ACCELERATIONS}
    for row in rows:
        grouped[row["acceleration"]][row["filename"]].append(float(row[value_key]))
    for acceleration in ACCELERATIONS:
        if not grouped[acceleration]:
            raise ValueError(f"No {value_key} values found for {acceleration}")

    rng = random.Random(seed)
    draws = {acc: [] for acc in ACCELERATIONS}
    draws["overall"] = []
    for _ in range(samples):
        acceleration_scores = []
        for acceleration in ACCELERATIONS:
            volumes = sorted(grouped[acceleration])
            sampled_values = []
            for _ in volumes:
                sampled_name = volumes[rng.randrange(len(volumes))]
                sampled_values.extend(grouped[acceleration][sampled_name])
            score = float(statistics.fmean(sampled_values))
            draws[acceleration].append(score)
            acceleration_scores.append(score)
        draws["overall"].append(float(statistics.fmean(acceleration_scores)))

    result = {"samples": samples, "seed": seed, "unit": "volume"}
    for key, values in draws.items():
        result[key] = {
            "bootstrap_mean": float(statistics.fmean(values)),
            "ci95_low": quantile_or_none(values, 0.025),
            "ci95_high": quantile_or_none(values, 0.975),
        }
    return result


def build_volume_rows(slice_rows, box_rows):
    slices_by_volume = defaultdict(list)
    boxes_by_volume = defaultdict(list)
    for row in slice_rows:
        slices_by_volume[row["filename"]].append(row)
    for row in box_rows:
        boxes_by_volume[row["filename"]].append(row)

    result = []
    for filename in sorted(slices_by_volume):
        slices = slices_by_volume[filename]
        boxes = boxes_by_volume.get(filename, [])
        bbox_score = mean_or_none([row["ssim_bbox"] for row in boxes])
        result.append(
            {
                "filename": filename,
                "acceleration": slices[0]["acceleration"],
                "slice_count": len(slices),
                "box_count": len(boxes),
                "ssim_full": mean_or_none([row["ssim_full"] for row in slices]),
                "ssim_bbox": "" if bbox_score is None else bbox_score,
            }
        )
    return result


def grouped_box_summary(box_rows, group_key):
    grouped = defaultdict(list)
    for row in box_rows:
        grouped[row[group_key]].append(float(row["ssim_bbox"]))
    result = []
    for group in sorted(grouped, key=str):
        stats = describe(grouped[group])
        result.append(
            {
                group_key: group,
                "box_count": stats["count"],
                "ssim_bbox": stats["mean"],
                "median": stats["median"],
                "p05": stats["p05"],
                "min": stats["min"],
            }
        )
    return result


def box_area_summary(box_rows):
    areas = [float(row["area"]) for row in box_rows]
    if not areas:
        return []
    q1 = quantile_or_none(areas, 0.25)
    q2 = quantile_or_none(areas, 0.5)
    q3 = quantile_or_none(areas, 0.75)
    buckets = (
        (f"Q1 <= {q1:.0f}", lambda value: value <= q1),
        (f"Q2 ({q1:.0f}, {q2:.0f}]", lambda value: q1 < value <= q2),
        (f"Q3 ({q2:.0f}, {q3:.0f}]", lambda value: q2 < value <= q3),
        (f"Q4 > {q3:.0f}", lambda value: value > q3),
    )
    result = []
    for label, predicate in buckets:
        values = [row["ssim_bbox"] for row in box_rows if predicate(float(row["area"]))]
        stats = describe(values)
        result.append(
            {
                "area_bucket": label,
                "box_count": stats["count"],
                "ssim_bbox": stats["mean"],
                "median": stats["median"],
                "p05": stats["p05"],
                "min": stats["min"],
            }
        )
    return result


def slice_position_summary(slice_rows):
    grouped = defaultdict(list)
    for row in slice_rows:
        index = min(int(float(row["slice_position"]) * 5), 4)
        label = POSITION_LABELS[index]
        grouped[(label, "all")].append(float(row["ssim_full"]))
        grouped[(label, row["acceleration"])].append(float(row["ssim_full"]))

    result = []
    for label in POSITION_LABELS:
        for acceleration in ("all",) + ACCELERATIONS:
            stats = describe(grouped[(label, acceleration)])
            result.append(
                {
                    "position_bucket": label,
                    "acceleration": acceleration,
                    "slice_count": stats["count"],
                    "ssim_full": stats["mean"],
                    "p05": stats["p05"],
                    "min": stats["min"],
                }
            )
    return result


def annotation_presence_summary(slice_rows):
    result = []
    for acceleration in ("all",) + ACCELERATIONS:
        selected = (
            slice_rows
            if acceleration == "all"
            else [row for row in slice_rows if row["acceleration"] == acceleration]
        )
        for presence in ("annotated", "unannotated"):
            values = [
                row["ssim_full"]
                for row in selected
                if (int(row["bbox_count"]) > 0) == (presence == "annotated")
            ]
            stats = describe(values)
            result.append(
                {
                    "acceleration": acceleration,
                    "annotation": presence,
                    "slice_count": stats["count"],
                    "ssim_full": stats["mean"],
                    "p05": stats["p05"],
                }
            )
    return result


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    denominator = x_scale * y_scale
    return float(numerator / denominator) if denominator else None


def full_bbox_correlations(slice_rows):
    result = {}
    for acceleration in ("all",) + ACCELERATIONS:
        selected = [row for row in slice_rows if row["ssim_bbox"] != ""]
        if acceleration != "all":
            selected = [row for row in selected if row["acceleration"] == acceleration]
        result[acceleration] = {
            "annotated_slice_count": len(selected),
            "pearson_r": pearson_correlation(
                [float(row["ssim_full"]) for row in selected],
                [float(row["ssim_bbox"]) for row in selected],
            ),
        }
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(value, digits=6):
    return "N/A" if value is None or value == "" else f"{float(value):.{digits}f}"


def _plot_dependencies():
    import h5py
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return h5py, np, plt


def plot_distributions(slice_rows, box_rows, position_rows, out_path):
    _, _, plt = _plot_dependencies()
    colors = {"acc4": "tab:blue", "acc8": "tab:orange"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    for acceleration in ACCELERATIONS:
        slices = [row for row in slice_rows if row["acceleration"] == acceleration]
        boxes = [row for row in box_rows if row["acceleration"] == acceleration]
        axes[0, 0].hist(
            [row["ssim_full"] for row in slices], bins=35, alpha=0.55,
            color=colors[acceleration], label=acceleration,
        )
        axes[0, 1].hist(
            [row["ssim_bbox"] for row in boxes], bins=30, alpha=0.55,
            color=colors[acceleration], label=acceleration,
        )

        position = [
            row for row in position_rows
            if row["acceleration"] == acceleration and row["slice_count"] > 0
        ]
        axes[1, 0].plot(
            range(len(position)), [row["ssim_full"] for row in position],
            marker="o", color=colors[acceleration], label=acceleration,
        )

        annotated = [
            row for row in slices
            if row["ssim_bbox"] != "" and row["acceleration"] == acceleration
        ]
        axes[1, 1].scatter(
            [row["ssim_full"] for row in annotated],
            [row["ssim_bbox"] for row in annotated],
            s=18, alpha=0.55, color=colors[acceleration], label=acceleration,
        )

    axes[0, 0].set_title("Slice SSIM_full distribution")
    axes[0, 1].set_title("Lesion-box SSIM_bbox distribution")
    axes[1, 0].set_title("SSIM_full by normalized slice position")
    axes[1, 0].set_xticks(range(len(POSITION_LABELS)), POSITION_LABELS)
    axes[1, 0].set_ylabel("mean SSIM_full")
    axes[1, 1].set_title("Annotated slices: full vs mean box SSIM")
    axes[1, 1].set_xlabel("SSIM_full")
    axes[1, 1].set_ylabel("mean SSIM_bbox")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def plot_volume_scores(volume_rows, out_path):
    _, np, plt = _plot_dependencies()
    colors = {"acc4": "tab:blue", "acc8": "tab:orange"}
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))

    full_rows = sorted(volume_rows, key=lambda row: row["ssim_full"])
    axes[0].bar(
        np.arange(len(full_rows)), [row["ssim_full"] for row in full_rows],
        color=[colors[row["acceleration"]] for row in full_rows],
    )
    axes[0].set_title("Per-volume SSIM_full (sorted)")
    axes[0].set_ylabel("SSIM_full")

    bbox_rows = [row for row in volume_rows if row["ssim_bbox"] != ""]
    bbox_rows.sort(key=lambda row: row["ssim_bbox"])
    axes[1].bar(
        np.arange(len(bbox_rows)), [row["ssim_bbox"] for row in bbox_rows],
        color=[colors[row["acceleration"]] for row in bbox_rows],
    )
    axes[1].set_title("Per-volume SSIM_bbox for annotated volumes (sorted)")
    axes[1].set_xlabel("volumes")
    axes[1].set_ylabel("SSIM_bbox")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def save_slice_extremes(slice_rows, image_dir, recon_dir, out_path, worst_k, best_k):
    h5py, np, plt = _plot_dependencies()
    from matplotlib.patches import Rectangle

    ordered = sorted(slice_rows, key=lambda row: row["ssim_full"])
    selected = ordered[:worst_k] + list(reversed(ordered[-best_k:]))
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        slice_index = int(row["slice"])
        with h5py.File(image_dir / row["filename"], "r") as file:
            target = file["image_label"][slice_index]
            annotations = json.loads(file.attrs.get("annotations", "{}"))
        with h5py.File(recon_dir / row["filename"], "r") as file:
            reconstruction = file["reconstruction"][slice_index]
        vmax = max(float(np.max(target)), 1e-12)
        error = np.abs(target - reconstruction)
        axes[row_index, 0].imshow(target, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 1].imshow(reconstruction, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 2].imshow(error, cmap="inferno")
        for box in annotations.get(str(slice_index), []):
            for axis in axes[row_index, :2]:
                axis.add_patch(
                    Rectangle(
                        (box["x"], box["y"]), box["width"], box["height"],
                        fill=False, edgecolor="red", linewidth=1,
                    )
                )
        axes[row_index, 0].set_title(
            f"Target: {row['filename']} slice {slice_index}\n{row['acceleration']}"
        )
        axes[row_index, 1].set_title(f"Reconstruction: SSIM_full={row['ssim_full']:.4f}")
        axes[row_index, 2].set_title("Absolute error")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def save_bbox_extremes(box_rows, image_dir, recon_dir, out_path, worst_k, best_k):
    h5py, np, plt = _plot_dependencies()
    ordered = sorted(box_rows, key=lambda row: row["ssim_bbox"])
    selected = ordered[:worst_k] + list(reversed(ordered[-best_k:]))
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        slice_index = int(row["slice"])
        with h5py.File(image_dir / row["filename"], "r") as file:
            target = file["image_label"][slice_index]
        with h5py.File(recon_dir / row["filename"], "r") as file:
            reconstruction = file["reconstruction"][slice_index]
        raw_x, raw_y = int(row["x"]), int(row["y"])
        x0, y0 = max(0, raw_x), max(0, raw_y)
        x1 = min(target.shape[1], raw_x + int(row["width"]))
        y1 = min(target.shape[0], raw_y + int(row["height"]))
        target_crop = target[y0:y1, x0:x1]
        recon_crop = reconstruction[y0:y1, x0:x1]
        error = np.abs(target_crop - recon_crop)
        vmax = max(float(np.max(target_crop)), 1e-12)
        axes[row_index, 0].imshow(target_crop, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 1].imshow(recon_crop, cmap="gray", vmin=0, vmax=vmax)
        axes[row_index, 2].imshow(error, cmap="inferno")
        axes[row_index, 0].set_title(
            f"Target box: {row['label'] or '<unlabeled>'}\n"
            f"{row['filename']} slice {slice_index}, area={row['area']}"
        )
        axes[row_index, 1].set_title(f"Reconstruction: SSIM_bbox={row['ssim_bbox']:.4f}")
        axes[row_index, 2].set_title("Absolute error")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def write_markdown(
    path, experiment, full_summary, bbox_summary, full_bootstrap, bbox_bootstrap,
    correlations, annotation_rows, position_rows, label_rows, area_rows,
    volume_rows, slice_rows, box_rows,
):
    full_score = full_summary["equal_acceleration_mean"]
    bbox_score = bbox_summary["equal_acceleration_mean"]
    lines = [
        f"# Validation Model Diagnostics: {experiment}",
        "",
        "This is a read-only diagnostic report. The challenge evaluator and metric code are not modified.",
        "Scores use the same foreground and bounding-box functions; confidence intervals resample whole volumes.",
        "",
        "## Overall",
        "",
        "| Metric | Score | Volume-bootstrap 95% CI |",
        "| --- | ---: | ---: |",
        f"| SSIM_full | {render(full_score)} | "
        f"[{render(full_bootstrap['overall']['ci95_low'])}, "
        f"{render(full_bootstrap['overall']['ci95_high'])}] |",
        f"| SSIM_bbox | {render(bbox_score)} | "
        f"[{render(bbox_bootstrap['overall']['ci95_low'])}, "
        f"{render(bbox_bootstrap['overall']['ci95_high'])}] |",
        "",
        "## By acceleration and lower tail",
        "",
        "| Metric | Acc | Units | Mean | Median | P05 | Minimum | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, summary, bootstrap in (
        ("SSIM_full", full_summary, full_bootstrap),
        ("SSIM_bbox", bbox_summary, bbox_bootstrap),
    ):
        for acceleration in ACCELERATIONS:
            stats = summary["by_acceleration"][acceleration]
            ci = bootstrap[acceleration]
            lines.append(
                f"| {metric} | {acceleration} | {stats['count']} | {render(stats['mean'])} "
                f"| {render(stats['median'])} | {render(stats['p05'])} | {render(stats['min'])} "
                f"| [{render(ci['ci95_low'])}, {render(ci['ci95_high'])}] |"
            )

    full_gap = (
        full_summary["by_acceleration"]["acc4"]["mean"]
        - full_summary["by_acceleration"]["acc8"]["mean"]
    )
    bbox_gap = (
        bbox_summary["by_acceleration"]["acc4"]["mean"]
        - bbox_summary["by_acceleration"]["acc8"]["mean"]
    )
    lines.extend(
        [
            "",
            "## Diagnostic gaps",
            "",
            f"- acc4 - acc8 SSIM_full gap: `{full_gap:+.6f}`",
            f"- acc4 - acc8 SSIM_bbox gap: `{bbox_gap:+.6f}`",
            f"- Full/bbox correlation on annotated slices: `{render(correlations['all']['pearson_r'])}` "
            f"(n={correlations['all']['annotated_slice_count']})",
            "",
            "A weak full/bbox correlation means global reconstruction quality is not a reliable proxy for lesion quality.",
            "",
            "## Annotated versus unannotated slices",
            "",
            "| Acc | Slice group | Slices | SSIM_full | P05 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in annotation_rows:
        lines.append(
            f"| {row['acceleration']} | {row['annotation']} | {row['slice_count']} "
            f"| {render(row['ssim_full'])} | {render(row['p05'])} |"
        )

    lines.extend(
        [
            "",
            "## Slice-position robustness",
            "",
            "| Relative position | Acc | Slices | SSIM_full | P05 | Minimum |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in position_rows:
        lines.append(
            f"| {row['position_bucket']} | {row['acceleration']} | {row['slice_count']} "
            f"| {render(row['ssim_full'])} | {render(row['p05'])} | {render(row['min'])} |"
        )

    lines.extend(
        [
            "",
            "## Bounding boxes by label",
            "",
            "| Label | Boxes | Mean | Median | P05 | Minimum |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in label_rows:
        lines.append(
            f"| {row['label']} | {row['box_count']} | {render(row['ssim_bbox'])} "
            f"| {render(row['median'])} | {render(row['p05'])} | {render(row['min'])} |"
        )

    lines.extend(
        [
            "",
            "## Bounding boxes by area",
            "",
            "| Area quartile | Boxes | Mean | Median | P05 | Minimum |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in area_rows:
        lines.append(
            f"| {row['area_bucket']} | {row['box_count']} | {render(row['ssim_bbox'])} "
            f"| {render(row['median'])} | {render(row['p05'])} | {render(row['min'])} |"
        )

    lines.extend(
        [
            "",
            "## Weakest volumes",
            "",
            "| File | Acc | Slices | Boxes | SSIM_full | SSIM_bbox |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(volume_rows, key=lambda item: item["ssim_full"])[:10]:
        lines.append(
            f"| {row['filename']} | {row['acceleration']} | {row['slice_count']} "
            f"| {row['box_count']} | {render(row['ssim_full'])} | {render(row['ssim_bbox'])} |"
        )

    lines.extend(
        [
            "",
            "## Weakest slices",
            "",
            "| File | Slice | Acc | Position | Boxes | SSIM_full |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(slice_rows, key=lambda item: item["ssim_full"])[:10]:
        lines.append(
            f"| {row['filename']} | {row['slice']} | {row['acceleration']} "
            f"| {render(row['slice_position'], 3)} | {row['bbox_count']} "
            f"| {render(row['ssim_full'])} |"
        )

    lines.extend(
        [
            "",
            "## Weakest lesion boxes",
            "",
            "| File | Slice | Acc | Label | Area | SSIM_bbox |",
            "| --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(box_rows, key=lambda item: item["ssim_bbox"])[:10]:
        lines.append(
            f"| {row['filename']} | {row['slice']} | {row['acceleration']} "
            f"| {row['label']} | {row['area']} | {render(row['ssim_bbox'])} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `validation_slice_diagnostics.csv`: one row per foreground-valid slice.",
            "- `validation_box_diagnostics.csv`: one row per valid annotation box.",
            "- `validation_volume_diagnostics.csv`: volume-level mean scores.",
            "- `validation_label_summary.csv`: lesion-label breakdown.",
            "- `validation_box_area_summary.csv`: lesion-size breakdown.",
            "- `validation_slice_position_summary.csv`: edge-to-center slice behavior.",
            "- `validation_score_distributions.png`: score distributions and relationships.",
            "- `validation_volume_scores.png`: sorted volume-level scores.",
            "- `validation_slice_extremes.png`: weakest and strongest slice examples.",
            "- `validation_bbox_extremes.png`: weakest and strongest lesion crops.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Read-only diagnostics for one validation reconstruction set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-name", default="varnet_c6_ch12_s4_ep80_lr3e4")
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--result-root", type=Path, default=Path("/root/result"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--worst-k", type=int, default=4)
    parser.add_argument("--best-k", type=int, default=4)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    import sys

    import h5py
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full

    image_dir = args.data_root / "val" / "image"
    recon_dir = args.result_root / args.exp_name / "reconstructions_val"
    out_dir = args.result_root / args.exp_name / "analysis" / "model_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    if not recon_dir.is_dir():
        raise FileNotFoundError(f"Validation reconstruction directory not found: {recon_dir}")

    recon_files = sorted(recon_dir.glob("*.h5"))
    if not recon_files:
        raise RuntimeError(f"No reconstruction H5 files found in {recon_dir}")
    missing_targets = [path.name for path in recon_files if not (image_dir / path.name).is_file()]
    if missing_targets:
        raise FileNotFoundError(f"Missing validation targets, first files: {missing_targets[:5]}")

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ssim = SSIM().to(device=device)
    slice_rows = []
    box_rows = []

    with torch.no_grad():
        for recon_path in recon_files:
            acceleration = acceleration_from_name(recon_path.name)
            with h5py.File(image_dir / recon_path.name, "r") as file:
                target_volume = file["image_label"][:]
                maximum = file.attrs["max"]
                annotations = json.loads(file.attrs.get("annotations", "{}"))
            with h5py.File(recon_path, "r") as file:
                recon_volume = file["reconstruction"][:]
            if target_volume.shape != recon_volume.shape:
                raise ValueError(
                    f"Shape mismatch for {recon_path.name}: "
                    f"target={target_volume.shape}, reconstruction={recon_volume.shape}"
                )

            slice_count = target_volume.shape[0]
            for slice_index in range(slice_count):
                target_t = torch.from_numpy(target_volume[slice_index]).to(device=device)
                recon_t = torch.from_numpy(recon_volume[slice_index]).to(device=device)
                mask_t = torch.from_numpy(foreground_mask(target_volume[slice_index])).to(
                    device=device, dtype=torch.float32
                )
                full_value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
                box_values = []
                for box_index, box in enumerate(annotations.get(str(slice_index), [])):
                    bbox_value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if bbox_value is None:
                        continue
                    width, height = int(box["width"]), int(box["height"])
                    box_values.append(bbox_value)
                    box_rows.append(
                        {
                            "filename": recon_path.name,
                            "slice": slice_index,
                            "slice_position": (slice_index + 0.5) / slice_count,
                            "acceleration": acceleration,
                            "box_index": box_index,
                            "label": str(box.get("label", "")),
                            "x": int(box["x"]),
                            "y": int(box["y"]),
                            "width": width,
                            "height": height,
                            "area": width * height,
                            "ssim_bbox": bbox_value,
                        }
                    )
                if full_value is not None:
                    slice_rows.append(
                        {
                            "filename": recon_path.name,
                            "slice": slice_index,
                            "slice_position": (slice_index + 0.5) / slice_count,
                            "acceleration": acceleration,
                            "ssim_full": full_value,
                            "ssim_bbox": "" if not box_values else mean_or_none(box_values),
                            "bbox_count": len(box_values),
                        }
                    )

    if not slice_rows:
        raise RuntimeError("No foreground-valid slices were scored")
    if not box_rows:
        raise RuntimeError("No valid annotation boxes were scored")

    volume_rows = build_volume_rows(slice_rows, box_rows)
    full_summary = summarize_metric(slice_rows, "ssim_full")
    bbox_summary = summarize_metric(box_rows, "ssim_bbox")
    full_bootstrap = cluster_bootstrap_score(
        slice_rows, "ssim_full", args.bootstrap_samples, args.seed
    )
    bbox_bootstrap = cluster_bootstrap_score(
        box_rows, "ssim_bbox", args.bootstrap_samples, args.seed + 1
    )
    correlations = full_bbox_correlations(slice_rows)
    annotation_rows = annotation_presence_summary(slice_rows)
    position_rows = slice_position_summary(slice_rows)
    label_rows = grouped_box_summary(box_rows, "label")
    area_rows = box_area_summary(box_rows)

    write_csv(
        out_dir / "validation_slice_diagnostics.csv", slice_rows,
        [
            "filename", "slice", "slice_position", "acceleration", "ssim_full",
            "ssim_bbox", "bbox_count",
        ],
    )
    write_csv(
        out_dir / "validation_box_diagnostics.csv", box_rows,
        [
            "filename", "slice", "slice_position", "acceleration", "box_index",
            "label", "x", "y", "width", "height", "area", "ssim_bbox",
        ],
    )
    write_csv(
        out_dir / "validation_volume_diagnostics.csv", volume_rows,
        ["filename", "acceleration", "slice_count", "box_count", "ssim_full", "ssim_bbox"],
    )
    write_csv(
        out_dir / "validation_label_summary.csv", label_rows,
        ["label", "box_count", "ssim_bbox", "median", "p05", "min"],
    )
    write_csv(
        out_dir / "validation_box_area_summary.csv", area_rows,
        ["area_bucket", "box_count", "ssim_bbox", "median", "p05", "min"],
    )
    write_csv(
        out_dir / "validation_slice_position_summary.csv", position_rows,
        ["position_bucket", "acceleration", "slice_count", "ssim_full", "p05", "min"],
    )
    write_csv(
        out_dir / "validation_annotation_summary.csv", annotation_rows,
        ["acceleration", "annotation", "slice_count", "ssim_full", "p05"],
    )

    summary = {
        "experiment": args.exp_name,
        "split": "validation",
        "read_only_analysis": True,
        "metric_aggregation": "equal-weight mean of acc4 and acc8",
        "ssim_full": full_summary,
        "ssim_bbox": bbox_summary,
        "bootstrap_ssim_full": full_bootstrap,
        "bootstrap_ssim_bbox": bbox_bootstrap,
        "full_bbox_correlations": correlations,
        "volume_count": len(volume_rows),
        "slice_count": len(slice_rows),
        "box_count": len(box_rows),
    }
    with (out_dir / "validation_model_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    write_markdown(
        out_dir / "validation_model_diagnostics.md", args.exp_name,
        full_summary, bbox_summary, full_bootstrap, bbox_bootstrap,
        correlations, annotation_rows, position_rows, label_rows, area_rows,
        volume_rows, slice_rows, box_rows,
    )
    plot_distributions(
        slice_rows, box_rows, position_rows,
        out_dir / "validation_score_distributions.png",
    )
    plot_volume_scores(volume_rows, out_dir / "validation_volume_scores.png")
    save_slice_extremes(
        slice_rows, image_dir, recon_dir, out_dir / "validation_slice_extremes.png",
        args.worst_k, args.best_k,
    )
    save_bbox_extremes(
        box_rows, image_dir, recon_dir, out_dir / "validation_bbox_extremes.png",
        args.worst_k, args.best_k,
    )

    print(f"Validation SSIM_full: {render(full_summary['equal_acceleration_mean'])}")
    print(f"Validation SSIM_bbox: {render(bbox_summary['equal_acceleration_mean'])}")
    print(
        f"Analyzed {len(volume_rows)} volumes, {len(slice_rows)} slices, "
        f"and {len(box_rows)} boxes"
    )
    print(f"Model diagnostics saved to: {out_dir}")


if __name__ == "__main__":
    main()
