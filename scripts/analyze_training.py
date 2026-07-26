"""Generate reproducible training and GPU-monitoring artifacts.

The script is safe to run while training is in progress. Epoch summaries are
read from the atomically-written training_history.csv; reported iteration
losses and GPU telemetry are parsed from append-only logs.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(
    r"Epoch\s*=\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*"
    r"Iter\s*=\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*"
    r"Loss\s*=\s*([0-9.eE+-]+)"
)


def read_epoch_history(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "train_time_sec": float(row["train_time_sec"]),
                    "val_time_sec": float(row["val_time_sec"]),
                    "learning_rate": float(row["learning_rate"]),
                    "is_best": int(row["is_best"]),
                }
            )
    return rows


def read_iteration_losses(path: Path):
    if not path.exists():
        return []
    rows_by_key = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = ITER_RE.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            iteration = int(match.group(3))
            total_iter = int(match.group(4))
            rows_by_key[(epoch, iteration)] = {
                "epoch": epoch,
                "iter": iteration,
                "total_iter": total_iter,
                "global_step": epoch * total_iter + iteration,
                "loss": float(match.group(5)),
            }
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def as_float(value):
    try:
        return float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def read_gpu_metrics(path: Path):
    if not path.exists():
        return []
    fields = [
        "timestamp", "index", "name", "utilization_gpu_pct",
        "memory_used_mib", "memory_total_mib", "temperature_c", "power_draw_w",
    ]
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            return rows
        for row in reader:
            if row.get("name") is None:
                continue
            parsed = {"timestamp": row["timestamp"].strip(), "name": row["name"].strip()}
            for field in fields[1:]:
                if field != "name":
                    parsed[field] = as_float(row[field])
            if parsed["utilization_gpu_pct"] is not None:
                rows.append(parsed)
    return rows


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(epoch_rows, iter_rows, out_dir: Path, smooth_window: int):
    if not epoch_rows and not iter_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if iter_rows:
        steps = np.array([row["global_step"] for row in iter_rows])
        losses = np.array([row["loss"] for row in iter_rows])
        axes[0].plot(steps, losses, linewidth=0.8, alpha=0.35, label="reported train loss")
        smooth = moving_average(losses, smooth_window)
        smooth_steps = steps[smooth_window - 1 :] if len(smooth) != len(losses) else steps
        axes[0].plot(smooth_steps, smooth, linewidth=2, label=f"moving average ({smooth_window})")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No iteration logs yet", ha="center", va="center")
    axes[0].set_title("Reported iteration loss")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("SSIM loss")
    axes[0].grid(True, alpha=0.3)

    if epoch_rows:
        epochs = [row["epoch"] for row in epoch_rows]
        axes[1].plot(epochs, [row["train_loss"] for row in epoch_rows], marker="o", label="train")
        axes[1].plot(epochs, [row["val_loss"] for row in epoch_rows], marker="o", label="validation")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No completed epochs yet", ha="center", va="center")
    axes[1].set_title("Epoch loss")
    axes[1].set_xlabel("epoch (zero-based)")
    axes[1].set_ylabel("mean SSIM loss")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_curves.png", dpi=160)
    plt.close(fig)


def plot_epoch_times(epoch_rows, out_dir: Path):
    if not epoch_rows:
        return
    epochs = np.array([row["epoch"] for row in epoch_rows])
    train_minutes = np.array([row["train_time_sec"] for row in epoch_rows]) / 60
    val_minutes = np.array([row["val_time_sec"] for row in epoch_rows]) / 60
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(epochs, train_minutes, label="train")
    ax.bar(epochs, val_minutes, bottom=train_minutes, label="validation")
    ax.set_xlabel("epoch (zero-based)")
    ax.set_ylabel("minutes")
    ax.set_title("Epoch duration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_times.png", dpi=160)
    plt.close(fig)


def plot_gpu_metrics(rows, out_dir: Path, sample_interval: int):
    if not rows:
        return
    minutes = np.arange(len(rows)) * sample_interval / 60
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(minutes, [row["utilization_gpu_pct"] for row in rows])
    axes[0].set_ylabel("GPU utilization (%)")
    axes[1].plot(minutes, [row["memory_used_mib"] for row in rows])
    axes[1].set_ylabel("memory used (MiB)")
    axes[2].plot(minutes, [row["temperature_c"] for row in rows])
    axes[2].set_ylabel("temperature (C)")
    axes[2].set_xlabel("sampled runtime (minutes)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("GPU telemetry")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_telemetry.png", dpi=160)
    plt.close(fig)


def build_summary(exp_name, epoch_rows, iter_rows, gpu_rows):
    summary = {
        "experiment": exp_name,
        "finished_epochs": len(epoch_rows),
        "reported_iteration_points": len(iter_rows),
        "gpu_samples": len(gpu_rows),
    }
    if epoch_rows:
        best = min(epoch_rows, key=lambda row: row["val_loss"])
        first = epoch_rows[0]
        last = epoch_rows[-1]
        total_seconds = sum(row["train_time_sec"] + row["val_time_sec"] for row in epoch_rows)
        summary.update(
            {
                "best_epoch_zero_based": best["epoch"],
                "best_epoch_one_based": best["epoch"] + 1,
                "best_val_loss": best["val_loss"],
                "last_epoch_zero_based": last["epoch"],
                "last_train_loss": last["train_loss"],
                "last_val_loss": last["val_loss"],
                "val_loss_improvement_pct": (
                    (first["val_loss"] - last["val_loss"]) / first["val_loss"] * 100
                    if first["val_loss"] != 0 else None
                ),
                "generalization_gap_last": last["val_loss"] - last["train_loss"],
                "total_recorded_time_hours": total_seconds / 3600,
                "mean_train_time_min": np.mean([row["train_time_sec"] for row in epoch_rows]) / 60,
                "mean_val_time_min": np.mean([row["val_time_sec"] for row in epoch_rows]) / 60,
            }
        )
    if gpu_rows:
        summary.update(
            {
                "gpu_name": gpu_rows[0]["name"],
                "gpu_utilization_mean_pct": float(np.mean([row["utilization_gpu_pct"] for row in gpu_rows])),
                "gpu_utilization_max_pct": float(np.max([row["utilization_gpu_pct"] for row in gpu_rows])),
                "gpu_memory_peak_mib": float(np.max([row["memory_used_mib"] for row in gpu_rows])),
                "gpu_temperature_max_c": float(np.max([row["temperature_c"] for row in gpu_rows])),
            }
        )
    return summary


def write_summary(summary, out_dir: Path):
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [f"# Training Summary: {summary['experiment']}", ""]
    labels = {
        "finished_epochs": "Finished epochs",
        "best_epoch_zero_based": "Best epoch (zero-based)",
        "best_epoch_one_based": "Best epoch (one-based)",
        "best_val_loss": "Best validation loss",
        "last_train_loss": "Last training loss",
        "last_val_loss": "Last validation loss",
        "val_loss_improvement_pct": "Validation loss improvement (%)",
        "generalization_gap_last": "Last generalization gap",
        "total_recorded_time_hours": "Recorded training time (hours)",
        "mean_train_time_min": "Mean training time per epoch (min)",
        "mean_val_time_min": "Mean validation time per epoch (min)",
        "gpu_name": "GPU",
        "gpu_utilization_mean_pct": "Mean GPU utilization (%)",
        "gpu_memory_peak_mib": "Peak GPU memory (MiB)",
        "gpu_temperature_max_c": "Maximum GPU temperature (C)",
    }
    lines.extend(["| Item | Value |", "| --- | ---: |"])
    for key, label in labels.items():
        if key in summary and summary[key] is not None:
            value = summary[key]
            rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"| {label} | {rendered} |")
    lines.extend(
        [
            "",
            "The epoch index emitted by the training code is zero-based. Validation loss is the mean of the per-volume SSIM losses.",
            "",
        ]
    )
    (out_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze an in-progress or completed training run")
    parser.add_argument("--exp-name", default="varnet_c6_ch12_s4_ep80_lr3e4")
    parser.add_argument("--result-root", type=Path, default=Path("../result"))
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--gpu-sample-interval", type=int, default=60)
    args = parser.parse_args()

    exp_dir = args.result_root / args.exp_name
    log_dir = args.result_root / "logs"
    out_dir = exp_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_rows = read_epoch_history(exp_dir / "training_history.csv")
    iter_rows = read_iteration_losses(log_dir / f"{args.exp_name}.log")
    gpu_rows = read_gpu_metrics(log_dir / f"{args.exp_name}_gpu.csv")

    if iter_rows:
        write_csv(
            out_dir / "iteration_loss.csv",
            iter_rows,
            ["epoch", "iter", "total_iter", "global_step", "loss"],
        )
    plot_losses(epoch_rows, iter_rows, out_dir, args.smooth_window)
    plot_epoch_times(epoch_rows, out_dir)
    plot_gpu_metrics(gpu_rows, out_dir, args.gpu_sample_interval)
    summary = build_summary(args.exp_name, epoch_rows, iter_rows, gpu_rows)
    write_summary(summary, out_dir)

    print(f"Completed epochs: {len(epoch_rows)}")
    print(f"Reported iteration losses: {len(iter_rows)}")
    print(f"GPU samples: {len(gpu_rows)}")
    print(f"Analysis saved to: {out_dir}")


if __name__ == "__main__":
    main()
