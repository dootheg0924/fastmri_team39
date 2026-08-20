"""Fail-fast checks for code-only and selected-candidate submission states."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_CODE = [
    "recon_eval.py",
    "train.py",
    "requirements.txt",
    "requirements-vessl.lock.txt",
    "pytest.ini",
    "utils/learning/test_part.py",
    "scripts/run_final_staged_reproduction.sh",
    "scripts/prepare_final_candidate.py",
    "scripts/prepare_epoch89_final_candidate.sh",
    "scripts/run_final_eval.sh",
    "scripts/capture_submission_evidence.sh",
    "scripts/parse_recon_eval.py",
    "scripts/finalize_submission.py",
    "scripts/package_final_submission.sh",
    "experiments/007_fivarnet_mraugment/config.env",
    "experiments/008_fivarnet_cross_acc/config.env",
    "submission/README.template.md",
    "submission/EVIDENCE.md",
    "submission/MANIFEST.md",
    "submission/PRESENTATION_OUTLINE.md",
    "submission/VIDEO_SCRIPT.md",
    "submission/EMAIL.template.md",
    "submission/NEXT_STEPS.md",
]
FORBIDDEN_PREP_CALL_FRAGMENTS = (
    "ifft",
    "grappa",
    "coil_combine",
    "sensitivity",
    "sens_net",
)


def _git_clean(path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Function not found: {name}")


def _call_name(node: ast.Call) -> str:
    target = node.func
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def inference_contract_errors() -> list[str]:
    path = REPO_ROOT / "utils/learning/test_part.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prep = _function(tree, "prep_volume")
    recon = _function(tree, "recon_slice")
    errors: list[str] = []

    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "image_path"
        for node in ast.walk(prep)
    ):
        errors.append("prep_volume reads image_path; image fields cannot be inference input")

    prep_calls = [_call_name(node).lower() for node in ast.walk(prep) if isinstance(node, ast.Call)]
    for call in prep_calls:
        if any(fragment in call for fragment in FORBIDDEN_PREP_CALL_FRAGMENTS):
            errors.append(f"prep_volume contains reconstruction-like call: {call}")

    recon_calls = [_call_name(node) for node in ast.walk(recon) if isinstance(node, ast.Call)]
    if "model" not in recon_calls:
        errors.append("recon_slice does not contain the timed model call")
    return errors


def verify_candidate(candidate_dir: Path, eval_metadata: Path | None) -> tuple[list[str], dict]:
    from scripts.prepare_final_candidate import load_training_checkpoint, sha256_file

    errors: list[str] = []
    candidate_dir = candidate_dir.resolve()
    manifest_path = candidate_dir / "candidate_manifest.json"
    checkpoint_path = candidate_dir / "checkpoints" / "best_model.pt"
    details: dict = {"candidate_dir": str(candidate_dir)}
    if not manifest_path.is_file():
        return [f"Missing candidate manifest: {manifest_path}"], details
    if not checkpoint_path.is_file():
        return [f"Missing candidate checkpoint: {checkpoint_path}"], details
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = load_training_checkpoint(checkpoint_path)
    digest = sha256_file(checkpoint_path)
    if digest != manifest["final_checkpoint"]["sha256"]:
        errors.append("Candidate checkpoint SHA-256 does not match its manifest")
    if int(checkpoint["epoch"]) != int(manifest["final_checkpoint"]["stored_epoch"]):
        errors.append("Candidate stored epoch does not match its manifest")
    details.update(
        {
            "candidate_id": manifest["candidate_id"],
            "mode": manifest["mode"],
            "checkpoint_sha256": digest,
            "stored_epoch": int(checkpoint["epoch"]),
        }
    )
    if eval_metadata is not None:
        evaluation = json.loads(eval_metadata.read_text(encoding="utf-8"))
        if evaluation["candidate_id"] != manifest["candidate_id"]:
            errors.append("Evaluation metadata belongs to another candidate")
        if evaluation["checkpoint"]["sha256"] != digest:
            errors.append("Evaluation metadata was produced from another checkpoint hash")
        details["scores"] = evaluation.get("scores")
    return errors, details


def run_checks(
    *,
    candidate_dir: Path | None,
    eval_metadata: Path | None,
    final_readme: Path | None,
    require_vessl: bool,
) -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    missing = [path for path in REQUIRED_CODE if not (REPO_ROOT / path).is_file()]
    record("required_code", not missing, "missing=" + repr(missing))

    immutable = ["recon_eval.py", "utils/common/metrics.py"]
    changed = [path for path in immutable if not _git_clean(path)]
    record("official_files_unmodified", not changed, "changed=" + repr(changed))

    contract_errors = inference_contract_errors()
    record("inference_contract", not contract_errors, "; ".join(contract_errors) or "ok")

    template = (REPO_ROOT / "submission/README.template.md").read_text(encoding="utf-8")
    record("readme_template", "{{FINAL_CANDIDATE_ID}}" in template, "template present")

    candidate_details = None
    if candidate_dir is not None:
        errors, candidate_details = verify_candidate(candidate_dir, eval_metadata)
        record("candidate_integrity", not errors, "; ".join(errors) or "ok")
    else:
        record("candidate_integrity", True, "pending by request")

    if final_readme is not None:
        if not final_readme.is_file():
            record("final_readme", False, f"missing={final_readme}")
        else:
            text = final_readme.read_text(encoding="utf-8")
            unresolved = "{{" in text or "}}" in text
            record("final_readme", not unresolved, "unresolved placeholders" if unresolved else "ok")
    else:
        record("final_readme", True, "pending by request")

    if final_readme is not None and candidate_details is not None:
        fixed_selection = (
            candidate_details.get("candidate_id") == "epoch89"
            and candidate_details.get("mode") == "single"
            and candidate_details.get("stored_epoch") == 89
        )
        record(
            "fixed_final_selection",
            fixed_selection,
            "expected candidate=epoch89 mode=single stored_epoch=89",
        )
    else:
        record("fixed_final_selection", True, "checked during finalization")

    if require_vessl:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        environment_ok = (
            sys.version_info[:2] == (3, 10)
            and torch.__version__.startswith("2.3.1")
            and numpy.__version__ == "1.24.4"
            and torch.cuda.is_available()
            and "GTX 1080" in gpu_name
        )
        record(
            "vessl_environment",
            environment_ok,
            f"python={sys.version.split()[0]} torch={torch.__version__} "
            f"numpy={numpy.__version__} gpu={gpu_name}",
        )
    else:
        record("vessl_environment", True, "not required for code-only preflight")

    return {
        "format_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "candidate": candidate_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--eval-metadata", type=Path)
    parser.add_argument("--final-readme", type=Path)
    parser.add_argument("--require-vessl", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.eval_metadata is not None and args.candidate_dir is None:
        parser.error("--eval-metadata requires --candidate-dir")
    report = run_checks(
        candidate_dir=args.candidate_dir,
        eval_metadata=args.eval_metadata,
        final_readme=args.final_readme,
        require_vessl=args.require_vessl,
    )
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['name']}: {check['detail']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
