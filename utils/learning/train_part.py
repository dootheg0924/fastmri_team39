import shutil
import copy
import csv
import gc
import json
import math
import numpy as np
import torch
import time
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone

from collections import defaultdict
from utils.data.load_data import (
    acceleration_from_filename,
    create_data_loaders,
    set_data_epoch,
)
from utils.data.mraugment import augmentation_probability
from utils.common.utils import save_reconstructions
from utils.common.bbox_loss import BboxAwareSSIMLoss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

HISTORY_FIELDS = [
    'epoch', 'train_loss', 'val_loss', 'paper_val_loss',
    'challenge_val_loss', 'ssim_full', 'ssim_bbox', 'final_score',
    'full_acc4', 'full_acc8', 'bbox_acc4', 'bbox_acc8',
    'train_time_sec', 'val_time_sec', 'learning_rate', 'global_step',
    'samples_seen', 'is_best',
]


def build_model(args):
    """Build a model from an args namespace.

    Also used by test_part.load_model with the args stored inside a
    checkpoint, so every hyperparameter falls back to a default via getattr:
    older checkpoints (e.g. exp/002) predate the fivarnet arguments.
    """
    model_name = getattr(args, 'model_name', 'varnet')
    if model_name == 'varnet':
        return VarNet(num_cascades=args.cascade,
                      chans=args.chans,
                      sens_chans=args.sens_chans)
    if model_name == 'fivarnet':
        from utils.model.fi_varnet import FIVarNet
        return FIVarNet(
            num_cascades=args.cascade,
            num_image_cascades=getattr(args, 'image_cascades', 2),
            sens_chans=args.sens_chans,
            sens_pools=getattr(args, 'sens_pools', 4),
            chans=args.chans,
            pools=getattr(args, 'pools', 4),
            attention_cascades=getattr(args, 'attention_cascades', None),
            kspace_mult_factor=getattr(args, 'kspace_mult_factor', 1e6),
            use_checkpoint=not getattr(args, 'no_grad_checkpoint', False),
            use_acc_film=getattr(args, 'acc_film', False),
            split_attention_cascades=getattr(args, 'split_attention_cascades', []),
            feature_processor=getattr(args, 'feature_processor', 'norm-unet'),
        )
    raise ValueError(f'Unknown model name: {model_name}')


def build_training_loss(args, device):
    """Build the selected training objective without changing validation scoring."""
    loss_name = getattr(args, 'loss_name', 'bbox-aware-ssim')
    if loss_name == 'ssim':
        return SSIMLoss().to(device=device)
    if loss_name == 'bbox-aware-ssim':
        return BboxAwareSSIMLoss(
            bbox_weight=getattr(args, 'bbox_loss_weight', 1.0)
        ).to(device=device)
    raise ValueError(f'Unknown loss name: {loss_name}')


def build_optimizer(args, model):
    """Construct Adam/AdamW with every effective hyperparameter explicit."""
    kwargs = {
        'lr': args.lr,
        'betas': (
            getattr(args, 'adam_beta1', 0.9),
            getattr(args, 'adam_beta2', 0.999),
        ),
        'eps': getattr(args, 'adam_eps', 1e-8),
        'weight_decay': getattr(args, 'weight_decay', 0.0),
        'amsgrad': getattr(args, 'adam_amsgrad', False),
    }
    optimizer_name = getattr(args, 'optimizer', 'adam')
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f'Unknown optimizer: {optimizer_name}')


def paper_lr_multiplier(
    step,
    *,
    base_lr,
    warmup_steps,
    cosine_start_step,
    max_steps,
    min_factor,
):
    """FI-VarNet warm-up/plateau/quarter-cosine schedule as a LR multiplier."""
    if base_lr <= 0:
        raise ValueError('base_lr must be positive.')
    if warmup_steps <= 0:
        raise ValueError('warmup_steps must be positive.')
    if not 0 < warmup_steps <= cosine_start_step < max_steps:
        raise ValueError(
            'Expected 0 < warmup_steps <= cosine_start_step < max_steps.'
        )
    if not 0 <= min_factor <= 1:
        raise ValueError('min_factor must be between 0 and 1.')

    step = max(0, min(int(step), int(max_steps)))
    if step < warmup_steps:
        return step / warmup_steps
    if step < cosine_start_step:
        return 1.0

    progress = (step - cosine_start_step) / (max_steps - cosine_start_step)
    return max(math.cos(progress * math.pi / 2), min_factor)


def build_lr_scheduler(args, optimizer):
    scheduler_name = getattr(args, 'lr_scheduler', 'none')
    if scheduler_name == 'none':
        return None
    if scheduler_name not in {'fi-varnet-paper', 'fi-varnet-epochs'}:
        raise ValueError(f'Unknown LR scheduler: {scheduler_name}')

    initial_schedule_steps = (
        getattr(args, 'lr_total_steps', None)
        if scheduler_name == 'fi-varnet-epochs'
        else getattr(args, 'max_steps', None)
    )
    if initial_schedule_steps is None:
        raise ValueError(
            f'{scheduler_name} requires its total optimizer-step horizon.'
        )

    def step_fn(step):
        schedule_steps = (
            getattr(args, 'lr_total_steps')
            if scheduler_name == 'fi-varnet-epochs'
            else initial_schedule_steps
        )
        return paper_lr_multiplier(
            step,
            base_lr=args.lr,
            warmup_steps=getattr(args, 'lr_warmup_steps', 7500),
            cosine_start_step=getattr(args, 'lr_cosine_start_step', 150000),
            max_steps=schedule_steps,
            min_factor=getattr(args, 'lr_min_factor', 1e-8),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, step_fn)


def configure_epoch_lr_schedule(args, loader_length):
    """Scale the FI LR phases to an epoch-defined final-training run."""
    if getattr(args, 'lr_scheduler', 'none') != 'fi-varnet-epochs':
        return None
    accumulation_steps = int(
        getattr(args, 'gradient_accumulation_steps', 1)
    )
    if accumulation_steps <= 0 or loader_length % accumulation_steps != 0:
        raise ValueError(
            'Epoch LR scheduling requires complete optimizer-step boundaries.'
        )
    steps_per_epoch = int(loader_length) // accumulation_steps
    total_steps = int(args.num_epochs) * steps_per_epoch
    if total_steps < 3:
        raise ValueError('Epoch LR scheduling requires at least three updates.')

    # Preserve the released FI schedule's relative phases while allowing the
    # final run length to be selected in epochs rather than a fixed step cap.
    args.lr_total_steps = total_steps
    args.lr_warmup_steps = max(1, round(total_steps * 7_500 / 210_000))
    args.lr_cosine_start_step = max(
        args.lr_warmup_steps,
        min(total_steps - 1, round(total_steps * 150_000 / 210_000)),
    )
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'warmup_steps': args.lr_warmup_steps,
        'cosine_start_step': args.lr_cosine_start_step,
    }


def resolve_time_budget_epochs(
    args,
    *,
    launch_start_epoch,
    measured_epoch_seconds,
    loader_length,
):
    """Resolve a submission run's target from measured real epoch time."""
    budget_hours = getattr(args, 'training_time_budget_hours', None)
    if budget_hours is None:
        return None
    reserve = float(
        getattr(args, 'training_time_reserve_fraction', 0.05)
    )
    if budget_hours <= 0:
        raise ValueError('--training-time-budget-hours must be positive.')
    if not 0 <= reserve < 1:
        raise ValueError(
            '--training-time-reserve-fraction must be in [0, 1).'
        )
    if measured_epoch_seconds <= 0:
        raise ValueError('Measured epoch time must be positive.')

    safe_seconds = float(budget_hours) * 3600.0 * (1.0 - reserve)
    affordable_epochs = max(1, int(safe_seconds // measured_epoch_seconds))
    requested_target = int(
        getattr(args, 'requested_num_epochs', args.num_epochs)
    )
    resolved_target = min(
        requested_target,
        int(launch_start_epoch) + affordable_epochs,
    )
    args.num_epochs = max(int(launch_start_epoch) + 1, resolved_target)
    schedule = configure_epoch_lr_schedule(args, loader_length)
    if getattr(args, 'mraugment', False):
        args.mraugment_total_epochs = int(args.num_epochs)
    return {
        'budget_hours': float(budget_hours),
        'reserve_fraction': reserve,
        'safe_training_seconds': safe_seconds,
        'measured_epoch_seconds': float(measured_epoch_seconds),
        'launch_start_epoch': int(launch_start_epoch),
        'affordable_epochs_this_launch': affordable_epochs,
        'requested_target_epoch': requested_target,
        'resolved_target_epoch': int(args.num_epochs),
        'lr_schedule': schedule,
    }


def train_epoch(
    args,
    epoch,
    model,
    data_loader,
    optimizer,
    scheduler,
    loss_type,
    device,
    global_step=0,
):
    model.train()
    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    set_data_epoch(data_loader.dataset, epoch)
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    completed_steps = 0
    completed_microbatches = 0
    samples_seen = 0
    max_steps = getattr(args, 'max_steps', None)
    accumulation_steps = int(
        getattr(args, 'gradient_accumulation_steps', 1)
    )
    if accumulation_steps <= 0:
        raise ValueError('--gradient-accumulation-steps must be positive.')
    if len_loader % accumulation_steps != 0:
        raise ValueError(
            f'Training loader has {len_loader} microbatches, which is not '
            f'divisible by gradient accumulation {accumulation_steps}. Use '
            'the padded paper sampler or change the accumulation setting.'
        )

    optimizer.zero_grad(set_to_none=True)
    microbatches_in_step = 0
    update_loss = 0.0
    for iter, data in enumerate(data_loader):
        if (
            microbatches_in_step == 0
            and max_steps is not None
            and global_step + completed_steps >= max_steps
        ):
            break
        mask, kspace, target, maximum, _, _, boxes = data
        mask = mask.to(device=device, non_blocking=True)
        kspace = kspace.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        maximum = maximum.to(device=device, non_blocking=True)
        # boxes stay on the CPU: only their integer coordinates are used for cropping.

        output = model(kspace, mask)
        if getattr(args, 'loss_name', 'bbox-aware-ssim') == 'ssim':
            loss = loss_type(output, target, maximum)
        else:
            loss = loss_type(output, target, maximum, boxes)
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
        update_loss += loss.item()
        completed_microbatches += 1
        microbatches_in_step += 1
        samples_seen += int(target.shape[0])

        if microbatches_in_step != accumulation_steps:
            continue

        clip_val = getattr(args, 'gradient_clip_val', 0.0)
        if clip_val and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        microbatches_in_step = 0

        current_step = global_step + completed_steps
        if current_step == 1 or current_step % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Step = [{current_step}/{max_steps or "-"}] '
                f'LR = {optimizer.param_groups[0]["lr"]:.4g} '
                f'Loss = {update_loss / accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
        update_loss = 0.0
    if microbatches_in_step != 0:
        raise RuntimeError(
            'Training stopped inside a gradient-accumulation window; no '
            'partial optimizer update is allowed.'
        )
    if completed_steps == 0:
        raise RuntimeError('Training epoch completed no optimizer steps.')
    total_loss = total_loss / completed_microbatches
    return (
        total_loss,
        time.perf_counter() - start_epoch,
        completed_steps,
        samples_seen,
    )


def _acc_bucket(fname):
    """Route a volume filename to its acceleration bucket (leaderboard uses two
    directories, acc4 / acc8; validation volumes carry the tag in the name)."""
    acceleration = acceleration_from_filename(fname)
    if acceleration is None:
        raise ValueError(
            f'Validation filename must contain one _acc4_ or _acc8_ tag: {fname}'
        )
    return f'acc{acceleration}'


def validate(args, model, val_metric, data_loader, device, paper_loss=None):
    """Score the validation set with the exact competition metric.

    Reproduces recon_eval.py / metrics.py aggregation without cv2: per-slice
    foreground SSIM and per-box bbox SSIM are averaged within each acceleration
    bucket, then final = 0.5 * (SSIM_full + SSIM_bbox) with
    SSIM_full = (full_acc4 + full_acc8) / 2 and likewise for bbox. The returned
    ``paper_val_loss`` is the public FI-VarNet implementation's plain,
    slice-weighted full-image 1-SSIM. ``val_loss`` selects either this value or
    ``1 - final`` according to ``args.checkpoint_metric``. ``paper-final``
    reports the same SSIM value but reserves model selection for the final
    210k-step checkpoint, matching the paper's knee protocol.
    """
    if paper_loss is None:
        paper_loss = SSIMLoss().to(device=device)
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    agg = {acc: {'full_total': 0.0, 'full_idx': 0, 'bbox_total': 0.0, 'bbox_idx': 0}
           for acc in ('acc4', 'acc8')}
    paper_loss_total = 0.0
    paper_loss_count = 0
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, fnames, slices, boxes = data
            kspace = kspace.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            maximum = maximum.to(device=device, non_blocking=True)
            output = model(kspace, mask)
            target_dev = target.to(device=device, non_blocking=True)
            batch_paper_loss = paper_loss(output, target_dev, maximum)
            paper_loss_total += float(batch_paper_loss.item()) * output.shape[0]
            paper_loss_count += output.shape[0]

            full_scores = val_metric.foreground_ssim_score(output, target_dev, maximum)
            box_scores = val_metric.bbox_ssim_scores(output, target_dev, maximum, boxes)

            for i in range(output.shape[0]):
                bucket = agg[_acc_bucket(fnames[i])]
                if full_scores[i] is not None:
                    bucket['full_total'] += full_scores[i]
                    bucket['full_idx'] += 1
                for score in box_scores[i]:
                    bucket['bbox_total'] += score
                    bucket['bbox_idx'] += 1

                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    def _mean(total, idx):
        return total / idx if idx > 0 else 0.0

    full4 = _mean(agg['acc4']['full_total'], agg['acc4']['full_idx'])
    full8 = _mean(agg['acc8']['full_total'], agg['acc8']['full_idx'])
    bbox4 = _mean(agg['acc4']['bbox_total'], agg['acc4']['bbox_idx'])
    bbox8 = _mean(agg['acc8']['bbox_total'], agg['acc8']['bbox_idx'])
    ssim_full = (full4 + full8) / 2
    ssim_bbox = (bbox4 + bbox8) / 2
    final_score = 0.5 * ssim_full + 0.5 * ssim_bbox
    challenge_val_loss = 1.0 - final_score
    paper_val_loss = (
        paper_loss_total / paper_loss_count
        if paper_loss_count > 0
        else float('inf')
    )
    checkpoint_metric = getattr(args, 'checkpoint_metric', 'challenge-final')
    if checkpoint_metric in {'paper-ssim', 'paper-final'}:
        selected_val_loss = paper_val_loss
    elif checkpoint_metric in {'challenge-final', 'submission-latest'}:
        selected_val_loss = challenge_val_loss
    else:
        raise ValueError(f'Unknown checkpoint metric: {checkpoint_metric}')

    result = {
        'val_loss': selected_val_loss,
        'paper_val_loss': paper_val_loss,
        'challenge_val_loss': challenge_val_loss,
        'checkpoint_metric': checkpoint_metric,
        'final_score': final_score,
        'ssim_full': ssim_full,
        'ssim_bbox': ssim_bbox,
        'full_acc4': full4,
        'full_acc8': full8,
        'bbox_acc4': bbox4,
        'bbox_acc8': bbox8,
        'num_subjects': len(reconstructions),
    }
    return result, reconstructions, targets, time.perf_counter() - start


def checkpoint_decision(
    args,
    val_loss,
    best_val_loss,
    global_step,
    completed_epochs=None,
):
    """Return updated best loss and whether to promote this checkpoint.

    The paper's knee experiment skipped validation, so its authoritative
    checkpoint is the final optimizer state. The released brain runner and
    legacy challenge training retain validation-based selection.
    """
    checkpoint_metric = getattr(args, 'checkpoint_metric', 'challenge-final')
    if checkpoint_metric == 'paper-final':
        max_steps = getattr(args, 'max_steps', None)
        if max_steps is None:
            raise ValueError('paper-final checkpointing requires --max-steps.')
        is_final = int(global_step) >= int(max_steps)
        if completed_epochs is not None:
            is_final = training_limit_reached(
                args, completed_epochs, global_step
            )
        return (float(val_loss) if is_final else float(best_val_loss), is_final)
    if checkpoint_metric == 'submission-latest':
        return float(val_loss), True

    is_new_best = float(val_loss) < float(best_val_loss)
    return min(float(best_val_loss), float(val_loss)), is_new_best


def training_limit_reached(args, completed_epochs, global_step):
    """Return whether either the configured step or explicit epoch cap is done."""
    max_steps = getattr(args, 'max_steps', None)
    max_training_epochs = getattr(args, 'max_training_epochs', None)
    step_limit = max_steps is not None and int(global_step) >= int(max_steps)
    epoch_limit = (
        max_training_epochs is not None
        and int(completed_epochs) >= int(max_training_epochs)
    )
    if max_steps is None and max_training_epochs is None:
        epoch_limit = int(completed_epochs) >= int(args.num_epochs)
    return step_limit or epoch_limit


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def save_training_history(history, out_path):
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    with tmp_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp_path, out_path)


def save_val_loss_log(history, out_path):
    values = np.array([[row['epoch'], row['val_loss']] for row in history], dtype=np.float64)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    with tmp_path.open('wb') as f:
        np.save(f, values)
    os.replace(tmp_path, out_path)


def save_model(
    args,
    exp_dir,
    epoch,
    model,
    optimizer,
    best_val_loss,
    is_new_best,
    history,
    scheduler=None,
    global_step=0,
    samples_seen=0,
):
    checkpoint = {
        'epoch': epoch,
        'global_step': int(global_step),
        'samples_seen': int(samples_seen),
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'optimizer_parameter_names': [
            name for name, _ in model.named_parameters()
        ],
        'best_val_loss': float(best_val_loss),
        'exp_dir': exp_dir,
        'history': history,
        'rng_state': capture_rng_state(),
        'warm_start_metadata': getattr(args, 'warm_start_metadata', None),
    }
    model_path = exp_dir / 'model.pt'
    tmp_path = exp_dir / 'model.pt.tmp'
    torch.save(checkpoint, f=tmp_path)
    os.replace(tmp_path, model_path)
    if is_new_best:
        _atomic_promote_checkpoint(
            model_path, exp_dir / 'best_model.pt'
        )
    if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
        snapshot_path = exp_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        snapshot_tmp = snapshot_path.with_suffix('.pt.tmp')
        shutil.copyfile(model_path, snapshot_tmp)
        os.replace(snapshot_tmp, snapshot_path)


def _atomic_promote_checkpoint(source_path, destination_path):
    """Atomically promote a checkpoint, preferring a zero-copy hard link."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    tmp_path = destination_path.with_suffix(destination_path.suffix + '.tmp')
    tmp_path.unlink(missing_ok=True)
    try:
        os.link(source_path, tmp_path)
    except OSError:
        shutil.copyfile(source_path, tmp_path)
    os.replace(tmp_path, destination_path)


def load_checkpoint(checkpoint_path, model, optimizer, device, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    scheduler_state = checkpoint.get('scheduler')
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError(
                'Checkpoint has no scheduler state but this run requests one; '
                'resume with the original training configuration.'
            )
        scheduler.load_state_dict(scheduler_state)
    restore_rng_state(checkpoint.get('rng_state'))
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    if torch.is_tensor(best_val_loss):
        best_val_loss = best_val_loss.item()
    return (
        checkpoint['epoch'],
        int(checkpoint.get('global_step', 0)),
        int(checkpoint.get('samples_seen', 0)),
        float(best_val_loss),
        checkpoint.get('history', []),
        checkpoint.get('warm_start_metadata'),
    )


WARM_START_MODEL_FIELDS = (
    'model_name',
    'cascade',
    'image_cascades',
    'chans',
    'sens_chans',
    'pools',
    'sens_pools',
    'attention_cascades',
    'feature_processor',
    'kspace_mult_factor',
    'acc_film',
    'bbox_loss_weight',
    'input_key',
    'target_key',
    'max_key',
)


def _namespace_value(namespace, name, default=None):
    if isinstance(namespace, dict):
        return namespace.get(name, default)
    return getattr(namespace, name, default)


def inherit_warm_start_args(args, source_args):
    """Use the checkpoint's actual model/loss shape, retaining new run controls."""
    inherited = {}
    for name in WARM_START_MODEL_FIELDS:
        source_value = _namespace_value(source_args, name, None)
        if source_value is None:
            continue
        source_value = copy.deepcopy(source_value)
        if getattr(args, name, None) != source_value:
            inherited[name] = {
                'requested': getattr(args, name, None),
                'checkpoint': source_value,
            }
        setattr(args, name, source_value)
    if getattr(args, 'model_name', None) != 'fivarnet':
        raise ValueError('Attention-split warm-start requires a FI-VarNet checkpoint.')
    return inherited


def _acc8_source_name(name, source_names):
    """Map a split acc8 expert parameter to its unsplit checkpoint parameter."""
    if name in source_names:
        return name
    token = '.attention_layer_acc8.'
    if token in name:
        candidate = name.replace(token, '.attention_layer.', 1)
        if candidate in source_names:
            return candidate
    return None


def _clone_optimizer_state(state):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.items()
    }


def migrate_optimizer_state_by_name(
    source_optimizer_state,
    source_parameter_names,
    target_model,
    target_optimizer,
):
    """Copy Adam state by parameter name and clone shared attention state to acc8.

    The epoch-40 checkpoint predates the extra acc8 module, so a normal
    ``optimizer.load_state_dict`` fails on the changed parameter-group length.
    This migration preserves every common parameter's moments and gives the
    copied expert an independent clone of the original attention moments.
    """
    if len(source_optimizer_state.get('param_groups', [])) != 1:
        raise ValueError('Warm-start currently requires one optimizer parameter group.')
    source_group = source_optimizer_state['param_groups'][0]
    source_ids = list(source_group['params'])
    if len(source_ids) != len(source_parameter_names):
        raise ValueError(
            'Checkpoint optimizer parameter count does not match the source model: '
            f'{len(source_ids)} optimizer entries vs {len(source_parameter_names)} names.'
        )
    source_id_by_name = dict(zip(source_parameter_names, source_ids))
    source_names = set(source_id_by_name)

    migrated = target_optimizer.state_dict()
    if len(migrated['param_groups']) != 1:
        raise ValueError('Target warm-start optimizer must have one parameter group.')
    target_group = migrated['param_groups'][0]
    target_ids = list(target_group['params'])
    target_names = [name for name, _ in target_model.named_parameters()]
    if len(target_ids) != len(target_names):
        raise ValueError('Target optimizer parameter order does not match target model.')

    copied_group = copy.deepcopy(source_group)
    copied_group['params'] = target_ids
    copied_group.pop('param_names', None)
    migrated['param_groups'] = [copied_group]
    migrated['state'] = {}

    target_parameter_by_name = dict(target_model.named_parameters())
    for target_name, target_id in zip(target_names, target_ids):
        source_name = _acc8_source_name(target_name, source_names)
        if source_name is None:
            raise ValueError(f'No warm-start optimizer source for {target_name}.')
        source_id = source_id_by_name[source_name]
        if source_id not in source_optimizer_state['state']:
            continue
        state = _clone_optimizer_state(source_optimizer_state['state'][source_id])
        parameter = target_parameter_by_name[target_name]
        for key, value in state.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.numel() > 1
                and value.shape != parameter.shape
            ):
                raise ValueError(
                    f'Optimizer state shape mismatch for {target_name}.{key}: '
                    f'{tuple(value.shape)} vs {tuple(parameter.shape)}'
                )
        migrated['state'][target_id] = state

    target_optimizer.load_state_dict(migrated)


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def load_attention_split_warm_start(checkpoint, model, optimizer, device):
    """Migrate an unsplit FI-VarNet checkpoint into a split-attention model."""
    source_args = checkpoint['args']
    source_split = _namespace_value(source_args, 'split_attention_cascades', [])
    if source_split:
        raise ValueError(
            'The source checkpoint already contains split attention; use --resume instead.'
        )
    split_cascades = getattr(model, 'cascades', None)
    requested_split = [
        i for i, block in enumerate(split_cascades or [])
        if getattr(block, 'attention_layer_acc8', None) is not None
    ]
    if not requested_split:
        raise ValueError('Warm-start target has no split attention cascades.')

    source_state = checkpoint['model']
    source_keys = set(source_state)
    migrated_state = {}
    for target_key in model.state_dict():
        source_key = _acc8_source_name(target_key, source_keys)
        if source_key is None:
            raise ValueError(f'No warm-start model source for {target_key}.')
        migrated_state[target_key] = source_state[source_key].detach().clone()
    model.load_state_dict(migrated_state, strict=True)

    source_parameter_names = checkpoint.get('optimizer_parameter_names')
    if source_parameter_names is None:
        # Reconstruct only long enough to recover the exact optimizer parameter
        # ordering used by the old checkpoint. No source weights need be loaded.
        source_model = build_model(source_args)
        source_parameter_names = [
            name for name, _ in source_model.named_parameters()
        ]
        del source_model
        gc.collect()
    migrate_optimizer_state_by_name(
        checkpoint['optimizer'],
        source_parameter_names,
        model,
        optimizer,
    )
    _move_optimizer_state(optimizer, device)

    # These asserts protect the function-preserving late split.
    for cascade_index in requested_split:
        block = model.cascades[cascade_index]
        acc4_state = block.attention_layer.state_dict()
        acc8_state = block.attention_layer_acc8.state_dict()
        if acc4_state.keys() != acc8_state.keys() or any(
            not torch.equal(acc4_state[key], acc8_state[key])
            for key in acc4_state
        ):
            raise RuntimeError(
                f'Cascade {cascade_index} acc4/acc8 attention copies differ at warm-start.'
            )

    restore_rng_state(checkpoint.get('rng_state'))
    source_epoch = int(checkpoint['epoch'])
    source_lr = float(optimizer.param_groups[0]['lr'])
    return source_epoch, source_lr, {
        'source_epoch': source_epoch,
        'source_best_val_loss': float(checkpoint.get('best_val_loss', float('nan'))),
        'split_attention_cascades': requested_split,
    }


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_metadata(args, device):
    metadata_path = args.val_loss_dir / 'run_metadata.json'
    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'experiment': str(args.net_name),
            'arguments': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            'launches': [],
        }
    launch = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version,
        'torch': torch.__version__,
        'git_commit': git_commit(),
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        'resume_requested': args.resume,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    metadata['launches'].append(launch)
    tmp_path = metadata_path.with_suffix('.json.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, metadata_path)

        
def train(args):
    torch.set_float32_matmul_precision(
        getattr(args, 'float32_matmul_precision', 'highest')
    )
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print('Training device:', device)

    checkpoint_path = args.exp_dir / 'model.pt'
    resume_existing = args.resume and checkpoint_path.exists()
    warm_start_path = getattr(args, 'warm_start_checkpoint', None)
    warm_start_checkpoint = None
    warm_start_rng = None
    warm_started = False

    if warm_start_path is not None and checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f'{checkpoint_path} already exists. Use --resume or choose a new experiment name.'
        )
    if warm_start_path is not None and not resume_existing:
        warm_start_path = Path(warm_start_path)
        if not warm_start_path.is_file():
            raise FileNotFoundError(f'Warm-start checkpoint not found: {warm_start_path}')
        warm_start_checkpoint = torch.load(
            warm_start_path, map_location='cpu', weights_only=False
        )
        source_epoch = int(warm_start_checkpoint['epoch'])
        expected_epoch = getattr(args, 'expected_warm_start_epoch', None)
        if expected_epoch is not None and source_epoch != expected_epoch:
            raise ValueError(
                f'Expected an epoch-{expected_epoch} warm-start checkpoint, '
                f'but {warm_start_path} stores epoch {source_epoch}.'
            )
        additional_epochs = getattr(args, 'additional_epochs', None)
        if additional_epochs is None or additional_epochs <= 0:
            raise ValueError(
                '--additional-epochs must be a positive integer when warm-starting.'
            )
        inherited = inherit_warm_start_args(args, warm_start_checkpoint['args'])
        args.num_epochs = source_epoch + additional_epochs
        args.warm_start_metadata = {
            'source_checkpoint': str(warm_start_path.resolve()),
            'source_epoch': source_epoch,
            'additional_epochs': additional_epochs,
            'inherited_argument_overrides': inherited,
        }
        print(
            f'Preparing warm-start from {warm_start_path} at epoch {source_epoch}; '
            f'target epoch: {args.num_epochs}.'
        )
        if inherited:
            print('Using checkpoint model/loss arguments instead of conflicting CLI values:')
            for name, values in inherited.items():
                print(
                    f'  {name}: requested={values["requested"]!r}, '
                    f'checkpoint={values["checkpoint"]!r}'
                )

    if (
        getattr(args, 'split_attention_cascades', [])
        or getattr(args, 'balance_accelerations', False)
    ) and args.batch_size != 1:
        raise ValueError(
            'Attention split and acceleration balancing require batch_size=1 '
            'because FI-VarNet infers one acceleration per batch.'
        )

    args.requested_num_epochs = int(args.num_epochs)
    budget_hours = getattr(args, 'training_time_budget_hours', None)
    reserve_fraction = float(
        getattr(args, 'training_time_reserve_fraction', 0.05)
    )
    probe_epochs = int(
        getattr(args, 'training_time_probe_epochs', 2)
    )
    if budget_hours is not None and budget_hours <= 0:
        raise ValueError('--training-time-budget-hours must be positive.')
    if not 0 <= reserve_fraction < 1:
        raise ValueError(
            '--training-time-reserve-fraction must be in [0, 1).'
        )
    if probe_epochs <= 0:
        raise ValueError('--training-time-probe-epochs must be positive.')

    train_paths = [args.data_path_train]
    if getattr(args, 'combine_train_val', False):
        train_paths.append(args.data_path_val)
        print(
            'Training on train + validation splits; validation is still '
            'evaluated separately.'
        )
    train_loader = create_data_loaders(
        data_path=train_paths,
        args=args,
        shuffle=True,
    )
    val_loader = None
    if getattr(args, 'checkpoint_metric', None) != 'submission-latest':
        val_loader = create_data_loaders(
            data_path=args.data_path_val, args=args
        )
    lr_schedule = configure_epoch_lr_schedule(args, len(train_loader))
    if lr_schedule is not None:
        print(
            'Final epoch LR schedule: '
            f'epochs={args.num_epochs}, '
            f'steps_per_epoch={lr_schedule["steps_per_epoch"]}, '
            f'total_steps={lr_schedule["total_steps"]}, '
            f'warmup_steps={lr_schedule["warmup_steps"]}, '
            f'cosine_start_step={lr_schedule["cosine_start_step"]}.'
        )
    if getattr(args, 'mraugment', False):
        accumulation_steps = int(
            getattr(args, 'gradient_accumulation_steps', 1)
        )
        if len(train_loader) % accumulation_steps != 0:
            raise ValueError(
                'MRAugment scheduling requires a complete optimizer-step '
                'boundary at every epoch.'
            )
        steps_per_epoch = len(train_loader) // accumulation_steps
        max_steps = getattr(args, 'max_steps', None)
        args.mraugment_total_epochs = (
            math.ceil(max_steps / steps_per_epoch)
            if max_steps is not None
            else int(args.num_epochs)
        )
        max_training_epochs = getattr(args, 'max_training_epochs', None)
        if max_training_epochs is not None:
            args.mraugment_total_epochs = min(
                args.mraugment_total_epochs, int(max_training_epochs)
            )
        if args.mraugment_total_epochs <= getattr(
            args, 'mraugment_delay_epochs', 0
        ):
            raise ValueError(
                'MRAugment total schedule epochs must exceed its delay.'
            )
        print(
            'MRAugment schedule: '
            f'{args.mraugment_schedule}, '
            f'p_max={args.mraugment_strength:g}, '
            f'decay={args.mraugment_exp_decay:g}, '
            f'total_epochs={args.mraugment_total_epochs}, '
            f'steps_per_epoch={steps_per_epoch}.'
        )

    model = build_model(args)
    model.to(device=device)

    loss_type = build_training_loss(args, device)
    validation_metric = BboxAwareSSIMLoss(
        bbox_weight=getattr(args, 'bbox_loss_weight', 1.0)
    ).to(device=device)
    paper_validation_loss = SSIMLoss().to(device=device)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0
    samples_seen = 0
    history = []

    if resume_existing:
        (
            start_epoch,
            global_step,
            samples_seen,
            best_val_loss,
            history,
            warm_start_metadata,
        ) = load_checkpoint(
            checkpoint_path, model, optimizer, device, scheduler=scheduler
        )
        if warm_start_metadata is not None:
            args.warm_start_metadata = warm_start_metadata
        print(f'Resumed from {checkpoint_path} at epoch {start_epoch}; '
              f'global step {global_step}; best val loss: {best_val_loss:.6f}')
    elif warm_start_checkpoint is not None:
        if scheduler is not None:
            raise ValueError(
                'Attention-split warm-start does not support a new LR scheduler. '
                'Use the source experiment optimizer configuration.'
            )
        warm_start_rng = copy.deepcopy(warm_start_checkpoint.get('rng_state'))
        start_epoch, source_lr, details = load_attention_split_warm_start(
            warm_start_checkpoint, model, optimizer, device
        )
        global_step = int(warm_start_checkpoint.get('global_step', 0))
        samples_seen = int(warm_start_checkpoint.get('samples_seen', 0))
        args.lr = source_lr
        args.warm_start_metadata.update(details)
        args.warm_start_metadata['optimizer_state_migrated'] = True
        warm_started = True
        print(
            f'Warm-started split attention at epoch {start_epoch}; '
            f'preserved Adam learning rate {source_lr:g}.'
        )
    elif args.resume:
        print(f'No checkpoint found at {checkpoint_path}; starting a new run.')

    write_run_metadata(args, device)
    if hasattr(train_loader.sampler, 'summary'):
        summary = train_loader.sampler.summary()
        sampling_path = args.val_loss_dir / 'acceleration_sampling.json'
        sampling_tmp = sampling_path.with_suffix('.json.tmp')
        with sampling_tmp.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(sampling_tmp, sampling_path)
        print(f'Training sampler: {summary}')

    if warm_started:
        print(
            f'Evaluating the function-preserving epoch-{start_epoch} '
            'warm-start baseline.'
        )
        baseline_result, baseline_recons, baseline_targets, baseline_time = validate(
            args,
            model,
            validation_metric,
            val_loader,
            device,
            paper_loss=paper_validation_loss,
        )
        if baseline_result['num_subjects'] == 0:
            raise RuntimeError('Validation loader produced no subjects at warm-start.')
        best_val_loss = float(baseline_result['val_loss'])
        baseline = {
            **args.warm_start_metadata,
            **{key: float(value) if isinstance(value, (float, np.floating)) else value
               for key, value in baseline_result.items()},
            'validation_time_sec': float(baseline_time),
        }
        baseline_path = args.val_loss_dir / 'warm_start_baseline.json'
        baseline_tmp = baseline_path.with_suffix('.json.tmp')
        with baseline_tmp.open('w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        os.replace(baseline_tmp, baseline_path)
        print(
            f'Warm-start baseline: final={baseline_result["final_score"]:.4f}, '
            f'full={baseline_result["ssim_full"]:.4f}, '
            f'bbox={baseline_result["ssim_bbox"]:.4f}.'
        )
        # Validation worker setup may consume the global RNG. Restore the exact
        # source state so the continuation starts from the epoch-40 RNG state.
        restore_rng_state(warm_start_rng)
        save_model(
            args, args.exp_dir, start_epoch, model, optimizer,
            best_val_loss, True, history,
            scheduler=scheduler, global_step=global_step,
            samples_seen=samples_seen,
        )
        del baseline_recons, baseline_targets, warm_start_checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    max_steps = getattr(args, 'max_steps', None)
    if max_steps is not None and max_steps <= 0:
        raise ValueError('--max-steps must be positive.')
    max_training_epochs = getattr(args, 'max_training_epochs', None)
    if max_training_epochs is not None and max_training_epochs <= 0:
        raise ValueError('--max-training-epochs must be positive.')
    if getattr(args, 'checkpoint_metric', None) == 'paper-final' and max_steps is None:
        raise ValueError('paper-final checkpointing requires --max-steps.')

    training_complete = training_limit_reached(
        args, start_epoch, global_step
    )
    completion_message = (
        f'Checkpoint already reached epoch {start_epoch}, step {global_step}; '
        f'limits: epochs={max_training_epochs or args.num_epochs}, '
        f'steps={max_steps or "-"}.'
    )
    if training_complete:
        if (
            getattr(args, 'checkpoint_metric', None)
            in {'paper-final', 'submission-latest'}
            and checkpoint_path.exists()
        ):
            # A process may have stopped after atomically replacing model.pt
            # but before copying it to the inference filename. Re-promote the
            # completed final checkpoint on resume as an idempotent repair.
            _atomic_promote_checkpoint(
                checkpoint_path, args.exp_dir / 'best_model.pt'
            )
        print(completion_message)
        return

    epoch = start_epoch
    launch_start_epoch = start_epoch
    launch_training_seconds = 0.0
    time_budget_resolved = False
    while not training_limit_reached(args, epoch, global_step):
        augmentation_p = 0.0
        if getattr(args, 'mraugment', False):
            augmentation_p = augmentation_probability(
                epoch,
                args.mraugment_total_epochs,
                schedule=args.mraugment_schedule,
                maximum=args.mraugment_strength,
                decay=args.mraugment_exp_decay,
                delay=args.mraugment_delay_epochs,
            )
        print(
            f'Epoch #{epoch:2d} ............... {args.net_name} '
            f'............... MRAugment p={augmentation_p:.6f}'
        )
        
        train_loss, train_time, completed_steps, epoch_samples_seen = train_epoch(
            args,
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_type,
            device,
            global_step=global_step,
        )
        global_step += completed_steps
        samples_seen += epoch_samples_seen
        launch_training_seconds += train_time
        completed_this_launch = epoch - launch_start_epoch + 1

        if (
            getattr(args, 'training_time_budget_hours', None) is not None
            and not time_budget_resolved
            and (
                completed_this_launch >= probe_epochs
                or launch_training_seconds
                + launch_training_seconds / completed_this_launch
                > float(args.training_time_budget_hours)
                * 3600.0
                * (1.0 - reserve_fraction)
            )
        ):
            average_epoch_seconds = (
                launch_training_seconds / completed_this_launch
            )
            budget_resolution = resolve_time_budget_epochs(
                args,
                launch_start_epoch=launch_start_epoch,
                measured_epoch_seconds=average_epoch_seconds,
                loader_length=len(train_loader),
            )
            time_budget_resolved = True
            budget_path = (
                args.val_loss_dir / 'resolved_training_time_budget.json'
            )
            budget_tmp = budget_path.with_suffix('.json.tmp')
            with budget_tmp.open('w', encoding='utf-8') as f:
                json.dump(
                    budget_resolution, f, indent=2, ensure_ascii=False
                )
            os.replace(budget_tmp, budget_path)
            print(
                'Time-budget epoch target resolved: '
                f'{budget_resolution["resolved_target_epoch"]} '
                f'({completed_this_launch}-epoch average '
                f'{average_epoch_seconds:.1f}s, '
                f'budget {budget_resolution["budget_hours"]:g}h, '
                f'reserve {budget_resolution["reserve_fraction"]:.0%}).'
            )
        elif (
            getattr(args, 'training_time_budget_hours', None) is not None
            and time_budget_resolved
        ):
            average_epoch_seconds = (
                launch_training_seconds / completed_this_launch
            )
            safe_seconds = (
                float(args.training_time_budget_hours)
                * 3600.0
                * (1.0 - reserve_fraction)
            )
            if (
                epoch + 1 < args.num_epochs
                and launch_training_seconds + average_epoch_seconds
                > safe_seconds
            ):
                args.num_epochs = epoch + 1
                configure_epoch_lr_schedule(args, len(train_loader))
                if getattr(args, 'mraugment', False):
                    args.mraugment_total_epochs = args.num_epochs
                print(
                    'Time-budget safety stop: the next epoch is unlikely to '
                    f'fit; finalizing completed epoch {epoch + 1}.'
                )

        paper_final_pending_row = False
        checkpoint_metric = getattr(args, 'checkpoint_metric', None)
        if checkpoint_metric in {'paper-final', 'submission-latest'}:
            # These protocols do not use validation for checkpoint selection.
            # Persist every epoch boundary first so an interruption cannot
            # lose the latest complete submission state.
            reached_final_step = training_limit_reached(
                args, epoch + 1, global_step
            )
            keep_as_submission = (
                checkpoint_metric == 'submission-latest'
                or reached_final_step
            )
            nan = float('nan')
            history.append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': nan,
                'paper_val_loss': nan,
                'challenge_val_loss': nan,
                'ssim_full': nan,
                'ssim_bbox': nan,
                'final_score': nan,
                'full_acc4': nan,
                'full_acc8': nan,
                'bbox_acc4': nan,
                'bbox_acc8': nan,
                'train_time_sec': float(train_time),
                'val_time_sec': 0.0,
                'learning_rate': float(optimizer.param_groups[0]['lr']),
                'global_step': int(global_step),
                'samples_seen': int(samples_seen),
                'is_best': int(keep_as_submission),
            })
            save_training_history(
                history, args.val_loss_dir / 'training_history.csv'
            )
            save_val_loss_log(history, args.val_loss_dir / 'val_loss_log.npy')
            save_model(
                args, args.exp_dir, epoch + 1, model, optimizer,
                best_val_loss, keep_as_submission, history,
                scheduler=scheduler, global_step=global_step,
                samples_seen=samples_seen,
            )
            if checkpoint_metric == 'submission-latest':
                print(
                    f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
                    f'TrainLoss = {train_loss:.4g} '
                    f'Step = [{global_step}] Samples = {samples_seen} '
                    'Validation = skipped (all labels used for training); '
                    'latest completed epoch promoted to best_model.pt '
                    f'TrainTime = {train_time:.4f}s',
                )
                epoch += 1
                continue
            if not reached_final_step:
                print(
                    f'Epoch = [{epoch:4d}] TrainLoss = {train_loss:.4g} '
                    f'Step = [{global_step}/{max_steps}] '
                    f'Samples = {samples_seen} '
                    'Validation = skipped '
                    '(paper knee final-checkpoint protocol) '
                    f'TrainTime = {train_time:.4f}s',
                )
                epoch += 1
                continue
            paper_final_pending_row = True
            print(
                f'Authoritative paper checkpoint saved at step {global_step}; '
                'running held-out reporting metrics.',
            )

        val_result, reconstructions, targets, val_time = validate(
            args,
            model,
            validation_metric,
            val_loader,
            device,
            paper_loss=paper_validation_loss,
        )
        if val_result['num_subjects'] == 0:
            raise RuntimeError('Validation loader produced no subjects.')
        val_loss = float(val_result['val_loss'])

        best_val_loss, is_new_best = checkpoint_decision(
            args,
            val_loss,
            best_val_loss,
            global_step,
            completed_epochs=epoch + 1,
        )

        completed_history_row = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': val_loss,
            'paper_val_loss': float(val_result['paper_val_loss']),
            'challenge_val_loss': float(val_result['challenge_val_loss']),
            'ssim_full': float(val_result['ssim_full']),
            'ssim_bbox': float(val_result['ssim_bbox']),
            'final_score': float(val_result['final_score']),
            'full_acc4': float(val_result['full_acc4']),
            'full_acc8': float(val_result['full_acc8']),
            'bbox_acc4': float(val_result['bbox_acc4']),
            'bbox_acc8': float(val_result['bbox_acc8']),
            'train_time_sec': float(train_time),
            'val_time_sec': float(val_time),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'global_step': int(global_step),
            'samples_seen': int(samples_seen),
            'is_best': int(is_new_best),
        }
        if paper_final_pending_row:
            history[-1] = completed_history_row
        else:
            history.append(completed_history_row)
        save_training_history(history, args.val_loss_dir / 'training_history.csv')
        save_val_loss_log(history, args.val_loss_dir / 'val_loss_log.npy')
        print(f"Training history saved to {args.val_loss_dir}")

        save_model(
            args, args.exp_dir, epoch + 1, model, optimizer,
            best_val_loss, is_new_best, history,
            scheduler=scheduler, global_step=global_step,
            samples_seen=samples_seen,
        )
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'Step = [{global_step}/{max_steps or "-"}] '
            f'Samples = {samples_seen} '
            f'ValLoss[{val_result["checkpoint_metric"]}] = {val_loss:.4g} '
            f'(paper={val_result["paper_val_loss"]:.4g} '
            f'challenge={val_result["challenge_val_loss"]:.4g} '
            f'final={val_result["final_score"]:.4f} '
            f'full={val_result["ssim_full"]:.4f} bbox={val_result["ssim_bbox"]:.4f}) '
            f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=None)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
        epoch += 1
