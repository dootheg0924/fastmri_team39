"""Bind one evaluated candidate to the final submission README and record."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
FINAL_CANDIDATE_ID = "epoch89"
FINAL_MODE = "single"
FINAL_SOURCE_EPOCHS = [89]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_source_path(record: dict) -> str:
    path = Path(record["path"])
    if path.parent.name == "checkpoints" and len(path.parents) >= 2:
        experiment = path.parent.parent.name
        return f'"${{RESULT_ROOT}}/{experiment}/checkpoints/{path.name}"'
    return f'"{path}"'


def _candidate_command(manifest: dict) -> str:
    sources = manifest["source_checkpoints"]
    lines = [
        "python scripts/prepare_final_candidate.py \\",
        f"  --mode {manifest['mode']} \\",
        "  --checkpoints \\",
    ]
    for index, source in enumerate(sources):
        suffix = " \\" if index < len(sources) - 1 else " \\"
        lines.append(f"    {_portable_source_path(source)}{suffix}")
    epochs = " ".join(str(value) for value in manifest["source_epochs"])
    lines.append(f"  --expected-epochs {epochs} \\")
    if manifest["mode"] == "average":
        weights = " ".join(
            format(float(value), ".12g") for value in manifest["normalized_weights"]
        )
        lines.append(f"  --weights {weights} \\")
    lines.extend(
        [
            '  --output-root "${RESULT_ROOT}/final_candidates" \\',
            f"  --candidate-id {manifest['candidate_id']}",
        ]
    )
    return "\n".join(lines)


def _source_table(manifest: dict) -> str:
    rows = ["| Epoch | 파일 | SHA-256 |", "|---:|---|---|"]
    for source in manifest["source_checkpoints"]:
        rows.append(
            f"| {source['completed_epoch']} | `{source['filename']}` | "
            f"`{source['sha256']}` |"
        )
    return "\n".join(rows)


def finalize(
    *,
    candidate_dir: Path,
    eval_metadata_path: Path,
    template_path: Path,
    output_readme: Path,
    output_selection: Path,
    team_name: str,
    team_members: str,
) -> tuple[Path, Path]:
    candidate_dir = candidate_dir.resolve()
    manifest = _read_json(candidate_dir / "candidate_manifest.json")
    evaluation = _read_json(eval_metadata_path)
    if manifest.get("candidate_id") != FINAL_CANDIDATE_ID:
        raise ValueError(
            f"Final submission is fixed to candidate {FINAL_CANDIDATE_ID!r}"
        )
    if manifest.get("mode") != FINAL_MODE:
        raise ValueError("Final submission requires the single epoch-89 checkpoint")
    if manifest.get("source_epochs") != FINAL_SOURCE_EPOCHS:
        raise ValueError("Final submission source epoch must be exactly [89]")
    if int(manifest["final_checkpoint"]["stored_epoch"]) != 89:
        raise ValueError("Final submission checkpoint must store completed epoch 89")
    if evaluation["candidate_id"] != manifest["candidate_id"]:
        raise ValueError("Evaluation metadata belongs to a different candidate")
    candidate_sha = manifest["final_checkpoint"]["sha256"]
    if evaluation["checkpoint"]["sha256"] != candidate_sha:
        raise ValueError("Evaluation checkpoint hash does not match candidate manifest")

    scores = evaluation["scores"]
    mode_description = "epoch 89 단일 checkpoint (byte-for-byte copy)"
    replacements = {
        "{{TEAM_NAME}}": team_name,
        "{{TEAM_MEMBERS}}": team_members,
        "{{FINAL_CANDIDATE_ID}}": manifest["candidate_id"],
        "{{FINAL_MODE}}": manifest["mode"],
        "{{FINAL_MODE_DESCRIPTION}}": mode_description,
        "{{FINAL_SOURCE_EPOCHS}}": ", ".join(
            str(value) for value in manifest["source_epochs"]
        ),
        "{{FINAL_STORED_EPOCH}}": str(manifest["final_checkpoint"]["stored_epoch"]),
        "{{FINAL_CHECKPOINT_FILENAME}}": manifest["final_checkpoint"]["filename"],
        "{{FINAL_CHECKPOINT_SHA256}}": candidate_sha,
        "{{FINAL_CHECKPOINT_BYTES}}": str(manifest["final_checkpoint"]["bytes"]),
        "{{FINAL_SSIM_FULL}}": f"{float(scores['ssim_full']):.4f}",
        "{{FINAL_SSIM_BBOX}}": f"{float(scores['ssim_bbox']):.4f}",
        "{{FINAL_RECON_SECONDS}}": f"{float(scores['recon_time_seconds']):.2f}",
        "{{FINAL_MS_PER_SLICE}}": f"{float(scores['milliseconds_per_slice']):.1f}",
        "{{SOURCE_CHECKPOINT_TABLE}}": _source_table(manifest),
        "{{FINAL_CANDIDATE_BUILD_COMMAND}}": _candidate_command(manifest),
    }

    rendered = template_path.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    unresolved = sorted(set(PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise ValueError(f"Unresolved README placeholders: {unresolved}")

    output_readme.parent.mkdir(parents=True, exist_ok=True)
    temporary_readme = output_readme.with_suffix(output_readme.suffix + ".tmp")
    temporary_readme.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary_readme, output_readme)

    selection = {
        "format_version": 1,
        "team_name": team_name,
        "team_members": team_members,
        "candidate_manifest": manifest,
        "official_evaluation": evaluation,
    }
    output_selection.parent.mkdir(parents=True, exist_ok=True)
    temporary_selection = output_selection.with_suffix(output_selection.suffix + ".tmp")
    temporary_selection.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_selection, output_selection)
    return output_readme, output_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument(
        "--template", type=Path,
        default=Path("submission/README.template.md"),
    )
    parser.add_argument(
        "--output-readme", type=Path,
        default=Path("submission/README.md"),
    )
    parser.add_argument(
        "--output-selection", type=Path,
        default=Path("submission/FINAL_SELECTION.json"),
    )
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--team-members", required=True)
    args = parser.parse_args()
    readme, selection = finalize(
        candidate_dir=args.candidate_dir,
        eval_metadata_path=args.eval_metadata,
        template_path=args.template,
        output_readme=args.output_readme,
        output_selection=args.output_selection,
        team_name=args.team_name,
        team_members=args.team_members,
    )
    print(f"Final README    : {readme}")
    print(f"Selection record: {selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
