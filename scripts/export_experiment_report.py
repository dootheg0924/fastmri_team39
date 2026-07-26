"""Copy lightweight VESSL artifacts into the repository for versioning."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_present(source, destination, copied):
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination)


def main():
    parser = argparse.ArgumentParser(description="Export lightweight experiment artifacts")
    parser.add_argument("--exp-name", default="varnet_c6_ch12_s4_ep80_lr3e4")
    parser.add_argument("--result-root", type=Path, default=Path("../result"))
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument("--log-tail-lines", type=int, default=500)
    args = parser.parse_args()

    exp_dir = args.result_root / args.exp_name
    analysis_dir = exp_dir / "analysis"
    log_dir = args.result_root / "logs"
    output_dir = args.output_root / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for name in (
        "resolved_config.env", "run_metadata.json", "training_history.csv",
        "val_loss_log.npy", "python_environment.txt", "gpu_environment.txt",
        "git_state.txt", "TRAINING_COMPLETED",
    ):
        copy_if_present(exp_dir / name, output_dir / name, copied)

    if analysis_dir.is_dir():
        for source in sorted(analysis_dir.iterdir()):
            if source.is_file():
                copy_if_present(source, output_dir / "analysis" / source.name, copied)

    gpu_log = log_dir / f"{args.exp_name}_gpu.csv"
    copy_if_present(gpu_log, output_dir / "gpu_metrics.csv", copied)

    training_log = log_dir / f"{args.exp_name}.log"
    if training_log.is_file():
        with training_log.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail_path = output_dir / "training_log_tail.txt"
        tail_path.write_text("".join(lines[-args.log_tail_lines :]), encoding="utf-8")
        copied.append(tail_path)

    manifest = {
        "experiment": args.exp_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(exp_dir.resolve()),
        "files": [],
    }
    for path in sorted(copied):
        manifest["files"].append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Exported {len(copied)} artifacts to: {output_dir}")
    print("Check the exported files before committing; checkpoints and reconstruction H5 files are excluded.")


if __name__ == "__main__":
    main()
