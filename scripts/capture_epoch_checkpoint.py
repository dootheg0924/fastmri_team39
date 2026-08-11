"""Capture named epoch checkpoints from a running experiment, without touching it.

`experiments/007_fivarnet_mraugment` runs with `CHECKPOINT_INTERVAL=0`, so it
keeps only `model.pt` (rewritten every epoch) and `best_model.pt`. Staging a
second run from a specific epoch needs that epoch's weights preserved, and
turning on `--checkpoint-interval` would mean restarting and losing the epoch in
flight.

`save_model` writes `model.pt` through `os.replace`, so a reader always sees a
complete file. This script copies it whenever it changes, keeps the copy if its
stored epoch was requested, and reports the SHA-256 to pin in the README. It
only ever reads the training directory.

    python scripts/capture_epoch_checkpoint.py \
        -c ../result/<experiment>/checkpoints \
        -e 50 -o ../result/staged_checkpoints
"""

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

import torch


def sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--checkpoints-dir', type=Path, required=True,
                        help="The running experiment's checkpoints directory")
    parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True,
                        help='Completed epoch counts to preserve')
    parser.add_argument('-o', '--out-dir', type=Path, required=True,
                        help='Where the captured checkpoints are written')
    parser.add_argument('-i', '--poll-seconds', type=float, default=300.0,
                        help='How often to check model.pt for a new epoch')
    parser.add_argument('--once', action='store_true',
                        help='Check the current model.pt once and exit')
    return parser.parse_args()


def capture(source, out_dir, wanted, seen):
    """Copy `source` if its epoch is still wanted. Returns the captured epoch."""
    staging = out_dir / 'capture.tmp'
    # Copy first, then read the copy: the training process may replace model.pt
    # at any moment, and the copy is what we would keep anyway.
    shutil.copyfile(source, staging)
    try:
        epoch = int(torch.load(staging, map_location='cpu', weights_only=False)['epoch'])
    except Exception:
        staging.unlink(missing_ok=True)
        raise

    if epoch in seen or epoch not in wanted:
        staging.unlink(missing_ok=True)
        return epoch if epoch in seen else None

    destination = out_dir / f'checkpoint_epoch_{epoch:04d}.pt'
    os.replace(staging, destination)
    print(f'captured epoch {epoch} -> {destination}', flush=True)
    print(f'  sha256 {sha256(destination)}', flush=True)
    return epoch


def main():
    args = parse()
    source = args.checkpoints_dir / 'model.pt'
    if not source.is_file():
        raise SystemExit(f'No model.pt under {args.checkpoints_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.epochs)
    seen = set()
    for epoch in sorted(wanted):
        existing = args.out_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        if existing.is_file():
            seen.add(epoch)
            print(f'already captured epoch {epoch}: {existing}', flush=True)

    last_mtime = None
    while wanted - seen:
        mtime = source.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            captured = capture(source, args.out_dir, wanted, seen)
            if captured is not None:
                seen.add(captured)
            elif not args.once:
                print(f'model.pt updated, epoch not requested', flush=True)
        if args.once:
            break
        remaining = sorted(wanted - seen)
        if remaining:
            print(f'waiting for epochs {remaining}', flush=True)
            time.sleep(args.poll_seconds)

    missing = sorted(wanted - seen)
    if missing:
        print(f'still missing: {missing}')
        return 1
    print('all requested epochs captured')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
