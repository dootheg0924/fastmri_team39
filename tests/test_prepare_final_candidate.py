import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from scripts.prepare_final_candidate import prepare_candidate, sha256_file


def _checkpoint(path: Path, epoch: int, value: float, counter: int = 7) -> Path:
    torch.save(
        {
            "model": {
                "weight": torch.tensor([value, value + 2], dtype=torch.float32),
                "complex": torch.tensor([complex(value, value + 1)]),
                "counter": torch.tensor(counter, dtype=torch.int64),
            },
            "args": Namespace(model_name="tiny", cascade=1),
            "epoch": epoch,
            "optimizer": {"not": "needed by inference"},
        },
        path,
    )
    return path


def test_single_candidate_is_an_exact_copy(tmp_path):
    source = _checkpoint(tmp_path / "epoch89.pt", 89, 3.0)
    candidate_dir, manifest = prepare_candidate(
        mode="single",
        checkpoint_paths=[source],
        expected_epochs=[89],
        output_root=tmp_path / "candidates",
        candidate_id="epoch89",
    )

    output = candidate_dir / "checkpoints" / "best_model.pt"
    assert sha256_file(output) == sha256_file(source)
    assert manifest["source_epochs"] == [89]
    assert manifest["final_checkpoint"]["stored_epoch"] == 89


def test_three_checkpoint_average_uses_equal_weights(tmp_path):
    sources = [
        _checkpoint(tmp_path / "epoch80.pt", 80, 1.0),
        _checkpoint(tmp_path / "epoch85.pt", 85, 4.0),
        _checkpoint(tmp_path / "epoch89.pt", 89, 7.0),
    ]
    candidate_dir, manifest = prepare_candidate(
        mode="average",
        checkpoint_paths=sources,
        expected_epochs=[80, 85, 89],
        output_root=tmp_path / "candidates",
        candidate_id="avg_80_85_89",
    )

    checkpoint = torch.load(
        candidate_dir / "checkpoints" / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(checkpoint["model"]["weight"], torch.tensor([4.0, 6.0]))
    assert torch.equal(
        checkpoint["model"]["complex"], torch.tensor([complex(4.0, 5.0)])
    )
    assert checkpoint["model"]["counter"].item() == 7
    assert checkpoint["epoch"] == 89
    assert "optimizer" not in checkpoint
    assert manifest["normalized_weights"] == pytest.approx([1 / 3] * 3)

    saved_manifest = json.loads(
        (candidate_dir / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["final_checkpoint"]["sha256"] == sha256_file(
        candidate_dir / "checkpoints" / "best_model.pt"
    )


def test_average_rejects_different_nonfloating_buffers(tmp_path):
    first = _checkpoint(tmp_path / "epoch80.pt", 80, 1.0, counter=7)
    second = _checkpoint(tmp_path / "epoch89.pt", 89, 7.0, counter=8)
    with pytest.raises(ValueError, match="Non-floating buffer"):
        prepare_candidate(
            mode="average",
            checkpoint_paths=[first, second],
            expected_epochs=[80, 89],
            output_root=tmp_path / "candidates",
            candidate_id="bad_average",
        )


def test_candidate_directory_is_never_overwritten(tmp_path):
    source = _checkpoint(tmp_path / "epoch89.pt", 89, 3.0)
    kwargs = dict(
        mode="single",
        checkpoint_paths=[source],
        expected_epochs=[89],
        output_root=tmp_path / "candidates",
        candidate_id="epoch89",
    )
    prepare_candidate(**kwargs)
    with pytest.raises(FileExistsError):
        prepare_candidate(**kwargs)


def test_expected_epoch_must_match_checkpoint(tmp_path):
    source = _checkpoint(tmp_path / "epoch89.pt", 89, 3.0)
    with pytest.raises(ValueError, match="Expected completed epoch 88"):
        prepare_candidate(
            mode="single",
            checkpoint_paths=[source],
            expected_epochs=[88],
            output_root=tmp_path / "candidates",
            candidate_id="wrong_epoch",
        )
