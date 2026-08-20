"""Parse the unchanged recon_eval.py summary into a checksummed JSON record."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_final_candidate import sha256_file


PATTERNS = {
    "ssim_full": re.compile(r"Leaderboard SSIM_full\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "ssim_bbox": re.compile(r"Leaderboard SSIM_bbox\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "timing": re.compile(
        r"Leaderboard Recon Time\s*:\s*([0-9]+(?:\.[0-9]+)?)s\s*"
        r"\(([0-9]+(?:\.[0-9]+)?)\s*ms/slice\)"
    ),
}


def _last(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"Could not find {label} in recon_eval log")
    return matches[-1]


def parse_recon_eval_log(text: str) -> dict[str, float]:
    full = _last(PATTERNS["ssim_full"], text, "Leaderboard SSIM_full")
    bbox = _last(PATTERNS["ssim_bbox"], text, "Leaderboard SSIM_bbox")
    timing = _last(PATTERNS["timing"], text, "Leaderboard Recon Time")
    return {
        "ssim_full": float(full.group(1)),
        "ssim_bbox": float(bbox.group(1)),
        "recon_time_seconds": float(timing.group(1)),
        "milliseconds_per_slice": float(timing.group(2)),
    }


def write_eval_metadata(
    *, candidate_dir: Path, log_path: Path, output_path: Path
) -> dict:
    candidate_dir = candidate_dir.resolve()
    log_path = log_path.resolve()
    checkpoint_path = candidate_dir / "checkpoints" / "best_model.pt"
    manifest_path = candidate_dir / "candidate_manifest.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Candidate checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Evaluation log not found: {log_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_sha = manifest["final_checkpoint"]["sha256"]
    if checkpoint_sha != expected_sha:
        raise ValueError(
            "Candidate checkpoint hash no longer matches candidate_manifest.json: "
            f"{checkpoint_sha} != {expected_sha}"
        )

    scores = parse_recon_eval_log(log_path.read_text(encoding="utf-8", errors="replace"))
    payload = {
        "format_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": manifest["candidate_id"],
        "candidate_mode": manifest["mode"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "stored_epoch": manifest["final_checkpoint"]["stored_epoch"],
        },
        "harness": "recon_eval.py",
        "log_path": str(log_path),
        "scores": scores,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_eval_metadata(
        candidate_dir=args.candidate_dir,
        log_path=args.log,
        output_path=args.output,
    )
    scores = payload["scores"]
    print(f"Evaluation metadata : {args.output}")
    print(f"SSIM_full          : {scores['ssim_full']:.4f}")
    print(f"SSIM_bbox          : {scores['ssim_bbox']:.4f}")
    print(f"Inference          : {scores['milliseconds_per_slice']:.1f} ms/slice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
