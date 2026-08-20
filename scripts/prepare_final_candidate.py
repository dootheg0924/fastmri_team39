"""Create an immutable final-submission checkpoint candidate.

Two candidate modes are supported without changing the evaluation contract:

* ``single`` copies one complete training checkpoint byte-for-byte.
* ``average`` computes a deterministic weighted arithmetic mean of floating
  point/complex model tensors. Non-floating buffers must be identical and are
  copied from the newest source checkpoint.

The output layout is always::

    <output-root>/<candidate-id>/
      checkpoints/best_model.pt
      candidate_manifest.json

``utils.learning.test_part.load_model`` can therefore evaluate either mode
through the unchanged ``recon_eval.py`` harness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary: {path}")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise KeyError(f"Checkpoint has no model state_dict: {path}")
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint has no saved args required by test_part.py: {path}")
    if "epoch" not in checkpoint:
        raise KeyError(f"Checkpoint has no completed epoch: {path}")
    try:
        checkpoint["epoch"] = int(checkpoint["epoch"])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Checkpoint epoch is not an integer: {path}") from exc
    return checkpoint


def validate_source_epochs(
    checkpoints: Sequence[dict[str, Any]],
    paths: Sequence[Path],
    expected_epochs: Sequence[int],
) -> list[int]:
    if len(expected_epochs) != len(checkpoints):
        raise ValueError("--expected-epochs must contain one value per checkpoint")
    actual = [int(checkpoint["epoch"]) for checkpoint in checkpoints]
    for path, expected, observed in zip(paths, expected_epochs, actual):
        if expected != observed:
            raise ValueError(
                f"Expected completed epoch {expected}, found {observed}: {path}"
            )
    if len(set(actual)) != len(actual):
        raise ValueError(f"Source epochs must be distinct, found: {actual}")
    return actual


def normalized_weights(count: int, values: Sequence[float] | None) -> list[float]:
    weights = [1.0] * count if values is None else [float(value) for value in values]
    if len(weights) != count:
        raise ValueError("--weights must contain one value per checkpoint")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Checkpoint weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one checkpoint weight must be positive")
    return [weight / total for weight in weights]


def _validate_tensor_compatibility(
    name: str, tensors: Sequence[torch.Tensor]
) -> None:
    reference = tensors[-1]
    for index, tensor in enumerate(tensors[:-1]):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} at source {index} is not a tensor")
        if tensor.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {tensor.shape} != {reference.shape}"
            )
        if tensor.dtype != reference.dtype:
            raise ValueError(
                f"Dtype mismatch for {name}: {tensor.dtype} != {reference.dtype}"
            )
        if tensor.layout != reference.layout:
            raise ValueError(
                f"Layout mismatch for {name}: {tensor.layout} != {reference.layout}"
            )


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    if len(state_dicts) < 2:
        raise ValueError("Averaging requires at least two checkpoints")
    if len(weights) != len(state_dicts):
        raise ValueError("Averaging requires one normalized weight per checkpoint")

    reference_keys = list(state_dicts[-1].keys())
    reference_key_set = set(reference_keys)
    for index, state in enumerate(state_dicts[:-1]):
        if set(state.keys()) != reference_key_set:
            missing = sorted(reference_key_set - set(state.keys()))
            extra = sorted(set(state.keys()) - reference_key_set)
            raise ValueError(
                f"State-dict keys differ at source {index}: missing={missing}, extra={extra}"
            )

    averaged: dict[str, torch.Tensor] = {}
    for name in reference_keys:
        tensors = [state[name].detach().cpu() for state in state_dicts]
        _validate_tensor_compatibility(name, tensors)
        reference = tensors[-1]

        if reference.is_floating_point() or reference.is_complex():
            accumulation_dtype = (
                torch.complex128 if reference.is_complex() else torch.float64
            )
            accumulator = torch.zeros_like(reference, dtype=accumulation_dtype)
            for weight, tensor in zip(weights, tensors):
                accumulator.add_(tensor.to(dtype=accumulation_dtype), alpha=weight)
            averaged[name] = accumulator.to(dtype=reference.dtype)
        else:
            for index, tensor in enumerate(tensors[:-1]):
                if not torch.equal(tensor, reference):
                    raise ValueError(
                        f"Non-floating buffer {name!r} differs between source "
                        f"{index} and the newest checkpoint"
                    )
            averaged[name] = reference.clone()
    return averaged


def _source_records(paths: Sequence[Path], epochs: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "filename": path.name,
            "completed_epoch": epoch,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, epoch in zip(paths, epochs)
    ]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_candidate(
    *,
    mode: str,
    checkpoint_paths: Sequence[Path],
    expected_epochs: Sequence[int],
    output_root: Path,
    candidate_id: str,
    weights: Sequence[float] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError(
            "candidate_id must use only ASCII letters, digits, dot, dash, and underscore"
        )
    if mode not in {"single", "average"}:
        raise ValueError(f"Unsupported candidate mode: {mode}")
    if mode == "single" and len(checkpoint_paths) != 1:
        raise ValueError("single mode requires exactly one checkpoint")
    if mode == "average" and len(checkpoint_paths) < 2:
        raise ValueError("average mode requires at least two checkpoints")

    paths = [Path(path).resolve() for path in checkpoint_paths]
    checkpoints = []
    for path in paths:
        loaded = load_training_checkpoint(path)
        # Full training checkpoints may contain multi-gigabyte Adam states.
        # Candidate construction needs only inference state, saved args and the
        # completed epoch, so release optimizer/scheduler/RNG tensors before
        # loading the next source checkpoint.
        checkpoints.append(
            {
                "model": loaded["model"],
                "args": loaded["args"],
                "epoch": loaded["epoch"],
            }
        )
        del loaded
    epochs = validate_source_epochs(checkpoints, paths, expected_epochs)
    source_records = _source_records(paths, epochs)
    resolved_weights = (
        [1.0]
        if mode == "single"
        else normalized_weights(len(checkpoints), weights)
    )

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_root / candidate_id
    if candidate_dir.exists():
        raise FileExistsError(
            f"Candidate directory already exists; refusing to overwrite: {candidate_dir}"
        )

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{candidate_id}.tmp-", dir=output_root)
    )
    try:
        checkpoint_dir = temporary_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        output_checkpoint = checkpoint_dir / "best_model.pt"

        if mode == "single":
            shutil.copy2(paths[0], output_checkpoint)
        else:
            averaged_model = average_state_dicts(
                [checkpoint["model"] for checkpoint in checkpoints],
                resolved_weights,
            )
            reference = checkpoints[-1]
            output_payload = {
                "model": averaged_model,
                "args": copy.deepcopy(reference["args"]),
                "epoch": max(epochs),
                "submission_metadata": {
                    "candidate_id": candidate_id,
                    "mode": "checkpoint_weight_average",
                    "source_epochs": epochs,
                    "normalized_weights": resolved_weights,
                    "resume_supported": False,
                },
            }
            torch.save(output_payload, output_checkpoint)

        verified = load_training_checkpoint(output_checkpoint)
        if list(verified["model"].keys()) != list(checkpoints[-1]["model"].keys()):
            raise RuntimeError("Output checkpoint state_dict changed during serialization")

        manifest: dict[str, Any] = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate_id,
            "mode": mode,
            "source_checkpoints": source_records,
            "source_epochs": epochs,
            "normalized_weights": resolved_weights,
            "final_checkpoint": {
                "relative_path": "checkpoints/best_model.pt",
                "filename": output_checkpoint.name,
                "stored_epoch": int(verified["epoch"]),
                "bytes": output_checkpoint.stat().st_size,
                "sha256": sha256_file(output_checkpoint),
            },
            "evaluation_contract": {
                "loader": "utils.learning.test_part.load_model",
                "harness": "recon_eval.py",
            },
        }
        _atomic_write_json(temporary_dir / "candidate_manifest.json", manifest)
        os.replace(temporary_dir, candidate_dir)
        return candidate_dir, manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one immutable final-submission checkpoint candidate"
    )
    parser.add_argument("--mode", choices=["single", "average"], required=True)
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", required=True,
        help="Source training checkpoint(s), ordered oldest to newest",
    )
    parser.add_argument(
        "--expected-epochs", type=int, nargs="+", required=True,
        help="Completed epoch stored in each source checkpoint",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="Optional averaging weights; defaults to equal weights",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_dir, manifest = prepare_candidate(
        mode=args.mode,
        checkpoint_paths=args.checkpoints,
        expected_epochs=args.expected_epochs,
        output_root=args.output_root,
        candidate_id=args.candidate_id,
        weights=args.weights,
    )
    final = manifest["final_checkpoint"]
    print(f"Candidate prepared : {candidate_dir}")
    print(f"Mode               : {manifest['mode']}")
    print(f"Source epochs      : {manifest['source_epochs']}")
    print(f"Stored epoch       : {final['stored_epoch']}")
    print(f"Checkpoint SHA-256 : {final['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
