import json
from argparse import Namespace
from pathlib import Path

import torch

from scripts.finalize_submission import finalize
from scripts.parse_recon_eval import write_eval_metadata
from scripts.prepare_final_candidate import prepare_candidate


def test_finalize_binds_readme_to_candidate_and_score(tmp_path):
    source = tmp_path / "best_model.pt"
    torch.save(
        {"model": {"weight": torch.ones(1)}, "args": Namespace(), "epoch": 89},
        source,
    )
    candidate_dir, manifest = prepare_candidate(
        mode="single",
        checkpoint_paths=[source],
        expected_epochs=[89],
        output_root=tmp_path / "candidates",
        candidate_id="epoch89",
    )
    log = tmp_path / "eval.log"
    log.write_text(
        "Leaderboard SSIM_full : 0.9001\n"
        "Leaderboard SSIM_bbox : 0.8002\n"
        "Leaderboard Recon Time : 10.00s (20.3 ms/slice)\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    write_eval_metadata(
        candidate_dir=candidate_dir, log_path=log, output_path=eval_path
    )
    template = Path("submission/README.template.md")
    readme = tmp_path / "README.md"
    selection = tmp_path / "FINAL_SELECTION.json"
    finalize(
        candidate_dir=candidate_dir,
        eval_metadata_path=eval_path,
        template_path=template,
        output_readme=readme,
        output_selection=selection,
        team_name="Team 39",
        team_members="A, B",
    )
    rendered = readme.read_text(encoding="utf-8")
    assert "{{" not in rendered
    assert manifest["final_checkpoint"]["sha256"] in rendered
    assert "0.9001" in rendered
    assert "0.8002" in rendered
    assert json.loads(selection.read_text(encoding="utf-8"))["candidate_manifest"][
        "candidate_id"
    ] == "epoch89"
