"""Checks for the staged-resume reproducibility guards.

The rule book judges a submission by whether the leaderboard score reproduces to
four decimals from the submitted README. A two-stage run (007 to epoch 50, then
008 with cross-acceleration re-masking) only reproduces if stage two provably
starts from the same stage-one weights, so the resume path fingerprints the
checkpoint and refuses an earlier one.
"""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# train_part imports utils.model.varnet, which does a bare `import fastmri`.
# train.py puts utils/model on the path at import time; do the same explicitly
# so this module does not depend on import ordering.
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'utils' / 'model'))

from utils.learning.train_part import checkpoint_digest, save_model  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)


def _save(tmp_path, epoch, checkpoint_epochs=(), interval=0):
    model = _Model()
    optimizer = torch.optim.Adam(model.parameters())
    args = SimpleNamespace(
        checkpoint_interval=interval,
        checkpoint_epochs=list(checkpoint_epochs),
        warm_start_metadata=None,
    )
    save_model(
        args, tmp_path, epoch, model, optimizer,
        best_val_loss=0.5, is_new_best=True, history=[],
    )
    return args


# --- checkpoint fingerprint ----------------------------------------------

def test_digest_matches_the_file_contents(tmp_path):
    path = tmp_path / 'blob.pt'
    path.write_bytes(b'fastmri' * 1000)
    expected = hashlib.sha256(b'fastmri' * 1000).hexdigest()
    assert checkpoint_digest(path) == expected


def test_digest_spans_chunk_boundaries(tmp_path):
    payload = bytes(range(256)) * 20_000            # ~5 MB, several chunks
    path = tmp_path / 'big.pt'
    path.write_bytes(payload)
    assert checkpoint_digest(path, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_digest_distinguishes_two_checkpoints(tmp_path):
    (tmp_path / 'a').write_bytes(b'epoch-50')
    (tmp_path / 'b').write_bytes(b'epoch-51')
    assert checkpoint_digest(tmp_path / 'a') != checkpoint_digest(tmp_path / 'b')


# --- standalone epoch snapshots -------------------------------------------

def test_requested_epochs_are_snapshotted(tmp_path):
    for epoch in (79, 80, 85, 90, 95, 96):
        _save(tmp_path, epoch, checkpoint_epochs=(80, 85, 90, 95))
    kept = sorted(p.name for p in tmp_path.glob('checkpoint_epoch_*.pt'))
    assert kept == [
        'checkpoint_epoch_0080.pt',
        'checkpoint_epoch_0085.pt',
        'checkpoint_epoch_0090.pt',
        'checkpoint_epoch_0095.pt',
    ]


def test_snapshots_do_not_replace_best_model(tmp_path):
    _save(tmp_path, 80, checkpoint_epochs=(80,))
    # A later epoch that is not a new best must leave best_model.pt alone while
    # still advancing model.pt -- the snapshot must not stand in for either.
    model, optimizer = _Model(), None
    optimizer = torch.optim.Adam(model.parameters())
    args = SimpleNamespace(
        checkpoint_interval=0, checkpoint_epochs=[85], warm_start_metadata=None
    )
    best_before = checkpoint_digest(tmp_path / 'best_model.pt')
    save_model(args, tmp_path, 85, model, optimizer, 0.9, False, [])
    assert checkpoint_digest(tmp_path / 'best_model.pt') == best_before
    assert (tmp_path / 'checkpoint_epoch_0085.pt').is_file()
    assert torch.load(
        tmp_path / 'checkpoint_epoch_0085.pt', map_location='cpu', weights_only=False
    )['epoch'] == 85


def test_snapshot_epochs_and_interval_compose(tmp_path):
    for epoch in range(1, 13):
        _save(tmp_path, epoch, checkpoint_epochs=(7,), interval=5)
    kept = sorted(int(p.stem.split('_')[-1]) for p in tmp_path.glob('checkpoint_epoch_*.pt'))
    assert kept == [5, 7, 10]


def test_no_snapshots_by_default(tmp_path):
    for epoch in (80, 85):
        _save(tmp_path, epoch)
    assert not list(tmp_path.glob('checkpoint_epoch_*.pt'))
    assert (tmp_path / 'model.pt').is_file()


def test_snapshot_content_equals_model_pt(tmp_path):
    _save(tmp_path, 90, checkpoint_epochs=(90,))
    assert (
        checkpoint_digest(tmp_path / 'checkpoint_epoch_0090.pt')
        == checkpoint_digest(tmp_path / 'model.pt')
    )


def test_no_temporary_files_are_left_behind(tmp_path):
    _save(tmp_path, 95, checkpoint_epochs=(95,))
    assert not list(tmp_path.glob('*.tmp'))
