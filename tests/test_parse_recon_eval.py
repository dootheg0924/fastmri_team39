import json
from argparse import Namespace

import torch

from scripts.parse_recon_eval import parse_recon_eval_log, write_eval_metadata
from scripts.prepare_final_candidate import prepare_candidate


LOG = """
Leaderboard SSIM_full : 0.9123
Leaderboard SSIM_bbox : 0.8456
Leaderboard Recon Time : 123.45s (67.8 ms/slice)
========== Details ==========
"""


def test_parse_recon_eval_summary():
    assert parse_recon_eval_log(LOG) == {
        "ssim_full": 0.9123,
        "ssim_bbox": 0.8456,
        "recon_time_seconds": 123.45,
        "milliseconds_per_slice": 67.8,
    }


def test_eval_metadata_is_bound_to_candidate_hash(tmp_path):
    source = tmp_path / "source.pt"
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
    log_path = tmp_path / "recon.log"
    log_path.write_text(LOG, encoding="utf-8")
    output_path = tmp_path / "eval_metadata.json"

    payload = write_eval_metadata(
        candidate_dir=candidate_dir,
        log_path=log_path,
        output_path=output_path,
    )
    assert payload["checkpoint"]["sha256"] == manifest["final_checkpoint"]["sha256"]
    assert payload["scores"]["ssim_bbox"] == 0.8456
    assert json.loads(output_path.read_text(encoding="utf-8"))["candidate_id"] == "epoch89"
