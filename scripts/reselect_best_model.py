"""Re-select the best checkpoint by the *leaderboard* metric, retroactively.

Background
----------
Training picks ``best_model.pt`` by ``train_part.validate``: the mean skimage
SSIM over the whole 384x384 frame (``utils.common.utils.ssim_loss``). The
leaderboard instead scores

    final = full_weight * SSIM_full + bbox_weight * SSIM_bbox

where SSIM_full is the foreground-masked SSIM and SSIM_bbox is the SSIM inside
each fastMRI+ lesion box, each averaged over acc4 and acc8 (see
``recon_eval.py`` / ``leaderboard_eval.py``, both 0.5 / 0.5). The two criteria
disagree twice over: the selection metric ignores the bbox term *and* skips the
foreground mask. So the epoch training saved as "best" is not necessarily the
epoch that scores best on the board.

This script re-scores every saved checkpoint on the validation set with the
exact leaderboard functions (``metrics.ssim_full`` / ``metrics.ssim_bbox`` and
``test_part.recon_slice``, all unmodified) and reports which checkpoint the
leaderboard metric would have chosen. Nothing is retrained; only the choice of
which snapshot to submit changes.

It requires per-epoch snapshots to choose from. exp/003 ran with
``--checkpoint-interval 10``, so ``checkpoints/`` holds ``checkpoint_epoch_0010
.pt`` ... plus ``best_model.pt`` (old criterion) and ``model.pt`` (last epoch).
Snapshot resolution is therefore 10 epochs; a peak between snapshots is not
recoverable without those checkpoints.

Run it in the same environment that runs ``recon_eval.py`` (needs cv2 for
``metrics.foreground_mask`` and skimage for the old-criterion column).

Example
-------
    python scripts/reselect_best_model.py \
        -c ../result/fivarnet_f4i2_ch12_s4_att01_bw1_lr3e4/checkpoints \
        -v /root/Data/val -g 0

Add ``--write`` to promote the leaderboard-best snapshot to ``best_model.pt``
(the previous ``best_model.pt`` is backed up as ``best_model_by_valloss.pt``).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / 'utils' / 'model'
for p in (str(REPO_ROOT), str(MODEL_ROOT)):
    if p not in sys.path:
        sys.path.insert(1, p)

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full  # noqa: E402
from utils.common.utils import ssim_loss  # noqa: E402
from utils.learning.test_part import prep_volume, recon_slice  # noqa: E402
from utils.learning.train_part import build_model  # noqa: E402


def parse():
    parser = argparse.ArgumentParser(
        description='Retroactively re-select best_model.pt by the leaderboard metric.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--checkpoints-dir', type=Path, required=True,
                        help='Directory with checkpoint_epoch_*.pt / model.pt / best_model.pt')
    parser.add_argument('-v', '--data-path-val', type=Path, default=Path('/root/Data/val'),
                        help='Validation root containing image/ and kspace/ subdirectories')
    parser.add_argument('-g', '--gpu-num', type=int, default=0)
    parser.add_argument('--full-weight', type=float, default=0.5,
                        help='Weight of SSIM_full in the final score')
    parser.add_argument('--bbox-weight', type=float, default=0.5,
                        help='Weight of SSIM_bbox in the final score')
    parser.add_argument('--target-key', type=str, default='image_label')
    parser.add_argument('--max-key', type=str, default='max')
    parser.add_argument('--out-csv', type=Path, default=None,
                        help='Where to write the per-checkpoint table (default: <checkpoints-dir>/reselect_scores.csv)')
    parser.add_argument('--write', action='store_true',
                        help='Promote the leaderboard-best snapshot to best_model.pt (backs up the old one)')
    return parser.parse_args()


def acc_of(fname: str):
    """Classify a volume as 'acc4' / 'acc8' from its file name, with a k-space
    fallback. Returns the group label or None if it cannot be determined."""
    low = fname.lower()
    if 'acc4' in low:
        return 'acc4'
    if 'acc8' in low:
        return 'acc8'
    return None


def acc_from_mask(mask: np.ndarray):
    """Fallback acc detection from the sampling mask: the most common spacing of
    the outer (non-ACS) sampled lines. Mirrors the model's mask heuristic."""
    sampled = np.where(np.asarray(mask).reshape(-1) > 0)[0]
    if sampled.size < 2:
        return None
    gaps = np.diff(sampled)
    outer = gaps[gaps > 1]  # drop the contiguous ACS block (gap == 1)
    if outer.size == 0:
        return None
    spacing = int(np.bincount(outer).argmax())
    if spacing <= 6:
        return 'acc4'
    return 'acc8'


def discover_checkpoints(ckpt_dir: Path):
    """Ordered, de-duplicated checkpoint list: epoch snapshots first (by epoch),
    then model.pt (last) and best_model.pt (old criterion)."""
    items = []  # (label, path, epoch_or_None)
    for path in sorted(ckpt_dir.glob('checkpoint_epoch_*.pt')):
        stem = path.stem.rsplit('_', 1)[-1]
        epoch = int(stem) if stem.isdigit() else None
        items.append((path.name, path, epoch))
    for name in ('model.pt', 'best_model.pt'):
        path = ckpt_dir / name
        if path.exists():
            items.append((name, path, None))
    return items


def load_model_from(path: Path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt.get('args')).to(device=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    stored_epoch = ckpt.get('epoch')
    return model, stored_epoch


def collect_volumes(val_root: Path, target_key: str, max_key: str):
    """List validation volumes as (name, image_path, kspace_path, acc)."""
    image_dir, kspace_dir = val_root / 'image', val_root / 'kspace'
    if not image_dir.is_dir() or not kspace_dir.is_dir():
        raise FileNotFoundError(f'Expected {image_dir} and {kspace_dir} to exist.')
    vols = []
    for img_path in sorted(image_dir.iterdir()):
        ks_path = kspace_dir / img_path.name
        if not ks_path.exists():
            print(f'[warn] no kspace for {img_path.name}; skipping')
            continue
        acc = acc_of(img_path.name)
        if acc is None:
            with h5py.File(ks_path, 'r') as hf:
                acc = acc_from_mask(np.array(hf['mask']))
            if acc is None:
                print(f'[warn] cannot determine acc for {img_path.name}; skipping')
                continue
        vols.append((img_path.name, img_path, ks_path, acc))
    return vols


@torch.no_grad()
def score_checkpoint(model, ssim, device, volumes, target_key, max_key):
    """Return per-acc totals and the old skimage-SSIM validation loss."""
    tot = {'acc4': {'full': 0.0, 'full_n': 0, 'bbox': 0.0, 'bbox_n': 0},
           'acc8': {'full': 0.0, 'full_n': 0, 'bbox': 0.0, 'bbox_n': 0}}
    old_loss_sum, old_subjects = 0.0, 0

    for name, img_path, ks_path, acc in volumes:
        with h5py.File(img_path, 'r') as hf:
            target_vol = hf[target_key][:]
            maximum = hf.attrs[max_key]
            annotations = json.loads(hf.attrs.get('annotations', '{}'))

        ctx = prep_volume(img_path, ks_path, device)
        n = ctx['num_slices']
        recon_vol = [recon_slice(model, ctx, s) for s in range(n)]  # each (H, W) on device

        for s in range(n):
            recon_t = recon_vol[s]
            target_t = torch.from_numpy(target_vol[s]).to(device=device)
            mask_t = torch.from_numpy(foreground_mask(target_vol[s])).to(device=device).type(torch.float)

            value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
            if value is not None:
                tot[acc]['full'] += value
                tot[acc]['full_n'] += 1
            for box in annotations.get(str(s), []):
                value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                if value is not None:
                    tot[acc]['bbox'] += value
                    tot[acc]['bbox_n'] += 1

        # Old selection criterion: mean whole-frame skimage SSIM loss per subject.
        pred_vol = np.stack([r.cpu().numpy() for r in recon_vol])
        old_loss_sum += ssim_loss(target_vol, pred_vol)
        old_subjects += 1

    def mean(acc, key):
        n = tot[acc][f'{key}_n']
        return tot[acc][key] / n if n > 0 else 0.0

    full4, full8 = mean('acc4', 'full'), mean('acc8', 'full')
    bbox4, bbox8 = mean('acc4', 'bbox'), mean('acc8', 'bbox')
    old_val_loss = old_loss_sum / old_subjects if old_subjects > 0 else float('nan')
    return {
        'full4': full4, 'full8': full8, 'bbox4': bbox4, 'bbox8': bbox8,
        'SSIM_full': (full4 + full8) / 2, 'SSIM_bbox': (bbox4 + bbox8) / 2,
        'old_val_loss': old_val_loss,
    }


def main():
    args = parse()
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print('Device:', device)

    checkpoints = discover_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        raise SystemExit(f'No checkpoints found in {args.checkpoints_dir}. '
                         'Was the run trained with --checkpoint-interval > 0?')
    volumes = collect_volumes(args.data_path_val, args.target_key, args.max_key)
    n4 = sum(1 for v in volumes if v[3] == 'acc4')
    n8 = sum(1 for v in volumes if v[3] == 'acc8')
    print(f'Validation volumes: {len(volumes)} (acc4: {n4}, acc8: {n8})')
    print(f'Checkpoints to score: {len(checkpoints)}')
    print(f'Final score = {args.full_weight:g}*SSIM_full + {args.bbox_weight:g}*SSIM_bbox\n')

    ssim = SSIM().to(device=device)
    rows = []
    for label, path, epoch in checkpoints:
        model, stored_epoch = load_model_from(path, device)
        ep = epoch if epoch is not None else stored_epoch
        m = score_checkpoint(model, ssim, device, volumes, args.target_key, args.max_key)
        final = args.full_weight * m['SSIM_full'] + args.bbox_weight * m['SSIM_bbox']
        rows.append({'checkpoint': label, 'path': path, 'epoch': ep, 'final_score': final, **m})
        print(f"{label:26s} ep={str(ep):>4} | "
              f"full a4/a8 {m['full4']:.4f}/{m['full8']:.4f} | "
              f"bbox a4/a8 {m['bbox4']:.4f}/{m['bbox8']:.4f} | "
              f"FULL {m['SSIM_full']:.4f} BBOX {m['SSIM_bbox']:.4f} | "
              f"FINAL {final:.4f} | old_valloss {m['old_val_loss']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r['final_score'])
    old_best = min((r for r in rows if not np.isnan(r['old_val_loss'])),
                   key=lambda r: r['old_val_loss'], default=None)
    print('\n' + '=' * 60)
    print(f"Leaderboard-best  : {best['checkpoint']} (ep {best['epoch']}), "
          f"FINAL {best['final_score']:.4f} "
          f"[FULL {best['SSIM_full']:.4f} / BBOX {best['SSIM_bbox']:.4f}]")
    if old_best is not None:
        print(f"Old-criterion best: {old_best['checkpoint']} (ep {old_best['epoch']}), "
              f"FINAL {old_best['final_score']:.4f} "
              f"[FULL {old_best['SSIM_full']:.4f} / BBOX {old_best['SSIM_bbox']:.4f}]")
        if old_best['checkpoint'] == best['checkpoint']:
            print('-> Same checkpoint. Re-selection does not change the submission.')
        else:
            gain = best['final_score'] - old_best['final_score']
            print(f'-> Different. Re-selection gains {gain:+.4f} on the final metric '
                  'among the scored snapshots.')

    out_csv = args.out_csv or (args.checkpoints_dir / 'reselect_scores.csv')
    fields = ['checkpoint', 'epoch', 'full4', 'full8', 'bbox4', 'bbox8',
              'SSIM_full', 'SSIM_bbox', 'final_score', 'old_val_loss']
    import csv
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    print(f'\nPer-checkpoint table written to {out_csv}')

    if args.write:
        import shutil
        best_pt = args.checkpoints_dir / 'best_model.pt'
        if best_pt.exists():
            backup = args.checkpoints_dir / 'best_model_by_valloss.pt'
            if not backup.exists():
                shutil.copyfile(best_pt, backup)
                print(f'Backed up existing best_model.pt -> {backup.name}')
        shutil.copyfile(best['path'], best_pt)
        print(f"Promoted {best['checkpoint']} -> best_model.pt")
    else:
        print('\n(dry run) Re-run with --write to promote the leaderboard-best snapshot '
              'to best_model.pt.')


if __name__ == '__main__':
    main()
