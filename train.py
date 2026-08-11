import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / 'utils' / 'model'
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(1, str(MODEL_ROOT))
from utils.learning.train_part import train  # noqa: E402

from utils.common.utils import seed_fix  # noqa: E402


PAPER_FIVARNET_PRESET = 'fi-varnet-paper'
FINAL_FIVARNET_PRESET = 'fi-varnet-final'


def apply_training_preset(args):
    """Apply reproducible, named training configurations after CLI parsing.

    This repository's FI-VarNet profile uses the paper's size-matched 6
    feature-space + 6 image-space model, with checkpointing and batch 1 for
    GTX 1080 training. Keeping the values here makes the configuration atomic:
    callers cannot accidentally combine the architecture with a different
    optimizer, loss, or stopping rule.
    """
    if args.training_preset not in {
        PAPER_FIVARNET_PRESET,
        FINAL_FIVARNET_PRESET,
    }:
        if getattr(args, 'checkpoint_metric', None) is None:
            args.checkpoint_metric = 'challenge-final'
        return args

    overrides = {
        # Architecture (Giannakopoulos et al., Scientific Reports 2024).
        'model_name': 'fivarnet',
        'cascade': 6,
        'image_cascades': 6,
        'chans': 32,
        'sens_chans': 8,
        'pools': 4,
        'sens_pools': 4,
        'attention_cascades': list(range(6)),
        'kspace_mult_factor': 1e6,
        'feature_processor': 'paper-unet2d',
        # Checkpoint every sensitivity/feature/image cascade so the 93.8M
        # model can be smoke-tested and trained on an 8 GB GTX 1080.
        'no_grad_checkpoint': False,
        'acc_film': False,
        'split_attention_cascades': [],
        'balance_accelerations': False,
        # Optimization and loss.
        'batch_size': 1,
        # FI-VarNet NormStats and variable-length bbox annotations require a
        # physical batch of one. Four sequential microbatches reproduce the
        # reference four-GPU run's effective optimizer batch without raising
        # the single-GPU activation peak.
        'gradient_accumulation_steps': 4,
        'lr': 3e-4,
        'optimizer': 'adamw',
        'weight_decay': 0.0,
        # exp/003 metric-aligned objective:
        # (1 - foreground SSIM) + 0.5 * mean(1 - bbox SSIM).
        'loss_name': 'bbox-aware-ssim',
        'bbox_loss_weight': 0.5,
        'lr_scheduler': (
            'fi-varnet-epochs'
            if args.training_preset == FINAL_FIVARNET_PRESET
            else 'fi-varnet-paper'
        ),
        'max_steps': (
            None
            if args.training_preset == FINAL_FIVARNET_PRESET
            else 210_000
        ),
        'lr_warmup_steps': 7_500,
        'lr_cosine_start_step': 150_000,
        'lr_min_factor': 1e-8,
        'gradient_clip_val': 1.0,
        # Explicit PyTorch AdamW defaults used by the reference implementation.
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'adam_amsgrad': False,
        'seed': 42,
        # Reference runtime protocol. The train/validation split remains a
        # runtime data-protocol choice: the knee paper uses train only, while
        # the released brain leaderboard runner combines train + validation.
        'data_sampler_seed': 0,
        'num_workers': (
            2
            if args.training_preset == FINAL_FIVARNET_PRESET
            else 4
        ),
        'pin_memory': args.training_preset == FINAL_FIVARNET_PRESET,
        # The paper runner is non-deterministic, but the challenge final must
        # reproduce its submitted score from scratch.
        'deterministic': (
            args.training_preset == FINAL_FIVARNET_PRESET
        ),
        'float32_matmul_precision': 'high',
    }
    # The paper preset reproduces final-step selection. The final submission
    # preset keeps the latest complete fixed-horizon epoch.
    if getattr(args, 'checkpoint_metric', None) is None:
        overrides['checkpoint_metric'] = (
            'submission-latest'
            if args.training_preset == FINAL_FIVARNET_PRESET
            else 'paper-final'
        )

    for name, value in overrides.items():
        setattr(args, name, value)
    args.training_preset_overrides = overrides
    return args


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
    parser.add_argument(
        '--warm-start-checkpoint',
        type=Path,
        default=None,
        help='Initialize a new experiment from an older checkpoint with compatible state migration',
    )
    parser.add_argument(
        '--additional-epochs',
        type=int,
        default=None,
        help='When warm-starting, train for this many epochs after the source checkpoint epoch',
    )
    parser.add_argument(
        '--expected-resume-epoch',
        type=int,
        default=None,
        help='Fail unless the resumed model.pt has at least this many completed '
             'epochs. Guards a staged run against silently continuing from the '
             'wrong stage-one checkpoint',
    )
    parser.add_argument(
        '--expected-warm-start-epoch',
        type=int,
        default=None,
        help='Fail unless the warm-start checkpoint has this completed epoch count',
    )
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        nargs='*',
        default=[],
        help='Also keep a standalone checkpoint_epoch_XXXX.pt after each of these '
             'epochs. Snapshots are copies taken alongside model.pt and never '
             'replace best_model.pt, so a submission checkpoint is unaffected',
    )
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='Keep an epoch snapshot every N epochs; 0 disables snapshots')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--pin-memory', action='store_true', help='Use pinned DataLoader memory for CUDA transfers')
    parser.add_argument(
        '--combine-train-val',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use both train and validation splits as training data while still '
             'evaluating on validation (the released brain leaderboard protocol)',
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Batch-1 microbatches averaged before one optimizer update',
    )
    parser.add_argument(
        '--data-sampler-seed',
        type=int,
        default=None,
        help='Shuffle seed; the paper DDP DistributedSampler uses 0',
    )
    parser.add_argument(
        '--deterministic',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable cuDNN deterministic mode; disabled by the public FI-VarNet runner',
    )
    parser.add_argument(
        '--float32-matmul-precision',
        choices=['highest', 'high', 'medium'],
        default='highest',
        help='torch float32 matrix multiplication precision policy',
    )
    parser.add_argument(
        '--mraugment',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Apply paper-aligned MRAugment to full training k-space before masking',
    )
    parser.add_argument(
        '--mraugment-schedule',
        choices=['constant', 'ramp', 'exp'],
        default='exp',
        help='Epoch-level augmentation probability schedule',
    )
    parser.add_argument(
        '--mraugment-strength',
        type=float,
        default=0.55,
        help='Maximum base probability p_max (paper fastMRI default: 0.55)',
    )
    parser.add_argument(
        '--mraugment-exp-decay',
        type=float,
        default=5.0,
        help='Normalized exponential schedule coefficient c',
    )
    parser.add_argument(
        '--mraugment-delay-epochs',
        type=int,
        default=0,
        help='Initial epochs with zero augmentation',
    )
    parser.add_argument(
        '--mraugment-seed',
        type=int,
        default=42,
        help='Stable augmentation and regenerated-mask seed',
    )
    parser.add_argument(
        '--mraugment-min-bbox-size',
        type=int,
        default=7,
        help='Cancel sample geometry if an augmented box is smaller than this',
    )
    parser.add_argument(
        '--cross-acceleration',
        type=float,
        default=0.0,
        help='Probability that a volume-epoch is re-undersampled at a drawn '
             'acceleration instead of its stored one. The label is RSS of the '
             'complete k-space and does not depend on the mask, so this pools '
             'the R4 and R8 volumes into a single source set for both. '
             '0 disables it',
    )
    parser.add_argument(
        '--cross-acceleration-p8',
        type=float,
        default=0.5,
        help='P(R8) when --cross-acceleration draws an acceleration',
    )
    parser.add_argument(
        '--training-preset',
        choices=['legacy', PAPER_FIVARNET_PRESET, FINAL_FIVARNET_PRESET],
        default='legacy',
        help='Atomic FI-VarNet presets: paper reproduces the fixed-step run; '
             'final trains a fixed epoch horizon and keeps the latest epoch',
    )
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam',
                        help='Optimizer used for reconstruction training')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Optimizer weight decay')
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--adam-eps', type=float, default=1e-8)
    parser.add_argument('--adam-amsgrad', action='store_true')
    parser.add_argument(
        '--lr-scheduler',
        choices=['none', 'fi-varnet-paper', 'fi-varnet-epochs'],
        default='none',
        help='Per-optimizer-step learning-rate schedule',
    )
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Stop after this many optimizer updates, independent of epochs')
    parser.add_argument(
        '--max-training-epochs',
        type=int,
        default=None,
        help='Optional hard epoch cap even when --max-steps is active',
    )
    parser.add_argument(
        '--training-time-budget-hours',
        type=float,
        default=None,
        help='Resolve a safe epoch target from the first measured epoch time',
    )
    parser.add_argument(
        '--training-time-reserve-fraction',
        type=float,
        default=0.05,
        help='Fraction of the time budget reserved for startup/checkpoints/reconstruction',
    )
    parser.add_argument(
        '--training-time-probe-epochs',
        type=int,
        default=2,
        help='Completed epochs averaged before resolving the time budget',
    )
    parser.add_argument('--lr-warmup-steps', type=int, default=7500)
    parser.add_argument('--lr-cosine-start-step', type=int, default=150000)
    parser.add_argument('--lr-min-factor', type=float, default=1e-8,
                        help='Minimum LambdaLR multiplier in the paper scheduler')
    parser.add_argument('--gradient-clip-val', type=float, default=0.0,
                        help='Global gradient-norm clipping; 0 disables clipping')
    parser.add_argument(
        '--loss-name',
        choices=['bbox-aware-ssim', 'ssim'],
        default='bbox-aware-ssim',
        help='Training objective. ssim is the paper 1-SSIM objective',
    )
    parser.add_argument(
        '--checkpoint-metric',
        choices=[
            'challenge-final',
            'paper-ssim',
            'paper-final',
            'submission-latest',
        ],
        default=None,
        help='Checkpoint protocol. Defaults to challenge-final for legacy '
             'training; submission-latest skips validation and keeps the '
             'latest completed epoch submission-ready',
    )
    
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
    parser.add_argument(
        '--feature-processor',
        choices=['norm-unet', 'paper-unet2d'],
        default='norm-unet',
        help='[fivarnet] Feature-cascade U-Net; paper-unet2d is the archived '
             'fastMRI FI-VarNet implementation',
    )
    parser.add_argument('--no-grad-checkpoint', action='store_true',
                        help='[fivarnet] Disable gradient checkpointing (uses more VRAM)')
    parser.add_argument('--acc-film', action='store_true',
                        help='[fivarnet] Acceleration-conditioned FiLM (per-acc4/acc8 channel-wise '
                             'gamma/beta on each feature cascade). Identity at init, so enabling it '
                             'does not change the model until training moves the parameters')
    parser.add_argument(
        '--split-attention-cascades',
        type=int,
        nargs='*',
        default=[],
        help='[fivarnet] Attention cascade indices with copied acc4/acc8 experts',
    )
    parser.add_argument(
        '--balance-accelerations',
        action='store_true',
        help='Train on an exact alternating 50/50 acc4/acc8 slice stream',
    )
    parser.add_argument(
        '--acceleration-balance-mode',
        choices=['oversample', 'undersample'],
        default='oversample',
        help='Balance by repeating the minority group or dropping majority slices',
    )
    parser.add_argument('--bbox-loss-weight', type=float, default=1.0,
                        help='Weight of the annotation-box SSIM loss term; 0 = pure foreground SSIM loss')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return apply_training_preset(args)

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed, deterministic=args.deterministic)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    experiment_dir = args.result_root / args.net_name
    args.exp_dir = experiment_dir / 'checkpoints'
    args.val_dir = experiment_dir / 'reconstructions_val'
    args.val_loss_dir = experiment_dir

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
