import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / 'utils' / 'model'
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(1, str(MODEL_ROOT))
from utils.learning.train_part import train  # noqa: E402

from utils.common.utils import seed_fix  # noqa: E402


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    parser.add_argument('--result-root', type=Path, default='../result', help='Root directory for experiment outputs')
    parser.add_argument('--resume', action='store_true', help='Resume from <experiment>/checkpoints/model.pt when present')
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='Keep an epoch snapshot every N epochs; 0 disables snapshots')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--pin-memory', action='store_true', help='Use pinned DataLoader memory for CUDA transfers')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter

    parser.add_argument('--model-name', type=str, default='varnet', choices=['varnet', 'fivarnet'],
                        help='Model architecture. fivarnet = Feature-Image VarNet (exp/003)')
    parser.add_argument('--image-cascades', type=int, default=2,
                        help='[fivarnet] Number of image-space cascades after the feature cascades')
    parser.add_argument('--pools', type=int, default=4,
                        help='[fivarnet] Down/up-sampling layers of cascade U-Nets')
    parser.add_argument('--sens-pools', type=int, default=4,
                        help='[fivarnet] Down/up-sampling layers of the sensitivity U-Net')
    parser.add_argument('--attention-cascades', type=int, nargs='*', default=[0],
                        help='[fivarnet] Feature-cascade indices with aliasing attention (subset keeps 8GB VRAM)')
    parser.add_argument('--kspace-mult-factor', type=float, default=1e6,
                        help='[fivarnet] k-space scaling applied before / undone after the cascades')
    parser.add_argument('--no-grad-checkpoint', action='store_true',
                        help='[fivarnet] Disable gradient checkpointing (uses more VRAM)')
    parser.add_argument('--acc-film', action='store_true',
                        help='[fivarnet] Acceleration-conditioned FiLM (per-acc4/acc8 channel-wise '
                             'gamma/beta on each feature cascade). Identity at init, so enabling it '
                             'does not change the model until training moves the parameters')
    parser.add_argument('--bbox-loss-weight', type=float, default=1.0,
                        help='Weight of the annotation-box SSIM loss term; 0 = pure foreground SSIM loss')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    experiment_dir = args.result_root / args.net_name
    args.exp_dir = experiment_dir / 'checkpoints'
    args.val_dir = experiment_dir / 'reconstructions_val'
    args.val_loss_dir = experiment_dir

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
