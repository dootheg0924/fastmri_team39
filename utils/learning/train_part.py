import shutil
import copy
import csv
import gc
import json
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
from utils.data.load_data import acceleration_from_filename, create_data_loaders
from utils.common.utils import save_reconstructions
from utils.common.bbox_loss import BboxAwareSSIMLoss
from utils.model.varnet import VarNet

import os

HISTORY_FIELDS = [
    'epoch', 'train_loss', 'val_loss', 'ssim_full', 'ssim_bbox', 'final_score',
    'full_acc4', 'full_acc8', 'bbox_acc4', 'bbox_acc8',
    'train_time_sec', 'val_time_sec', 'learning_rate', 'is_best',
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
        )
    raise ValueError(f'Unknown model name: {model_name}')


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, device):
    model.train()
    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, boxes = data
        mask = mask.to(device=device, non_blocking=True)
        kspace = kspace.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        maximum = maximum.to(device=device, non_blocking=True)
        # boxes stay on the CPU: only their integer coordinates are used for cropping.

        # With hard-routed experts, a non-selected expert must keep grad=None;
        # otherwise Adam momentum can update it on the wrong acceleration.
        optimizer.zero_grad(set_to_none=True)
        output = model(kspace, mask)
        loss = loss_type(output, target, maximum, boxes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def _acc_bucket(fname):
    """Route a volume filename to its acceleration bucket (leaderboard uses two
    directories, acc4 / acc8; validation volumes carry the tag in the name)."""
    acceleration = acceleration_from_filename(fname)
    if acceleration is None:
        raise ValueError(
            f'Validation filename must contain one _acc4_ or _acc8_ tag: {fname}'
        )
    return f'acc{acceleration}'


def validate(args, model, val_metric, data_loader, device):
    """Score the validation set with the exact competition metric.

    Reproduces recon_eval.py / metrics.py aggregation without cv2: per-slice
    foreground SSIM and per-box bbox SSIM are averaged within each acceleration
    bucket, then final = 0.5 * (SSIM_full + SSIM_bbox) with
    SSIM_full = (full_acc4 + full_acc8) / 2 and likewise for bbox. The returned
    ``val_loss`` is ``1 - final`` so that "lower is better" still holds.
    """
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    agg = {acc: {'full_total': 0.0, 'full_idx': 0, 'bbox_total': 0.0, 'bbox_idx': 0}
           for acc in ('acc4', 'acc8')}
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, fnames, slices, boxes = data
            kspace = kspace.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            maximum = maximum.to(device=device, non_blocking=True)
            output = model(kspace, mask)
            target_dev = target.to(device=device, non_blocking=True)

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

    result = {
        'val_loss': 1.0 - final_score,
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


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, history):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
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
        best_tmp = exp_dir / 'best_model.pt.tmp'
        shutil.copyfile(model_path, best_tmp)
        os.replace(best_tmp, exp_dir / 'best_model.pt')
    if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
        snapshot_path = exp_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        snapshot_tmp = snapshot_path.with_suffix('.pt.tmp')
        shutil.copyfile(model_path, snapshot_tmp)
        os.replace(snapshot_tmp, snapshot_path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    restore_rng_state(checkpoint.get('rng_state'))
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    if torch.is_tensor(best_val_loss):
        best_val_loss = best_val_loss.item()
    return (
        checkpoint['epoch'],
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

    model = build_model(args)
    model.to(device=device)

    loss_type = BboxAwareSSIMLoss(
        bbox_weight=getattr(args, 'bbox_loss_weight', 1.0)
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = float('inf')
    start_epoch = 0
    history = []

    if resume_existing:
        start_epoch, best_val_loss, history, warm_start_metadata = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        if warm_start_metadata is not None:
            args.warm_start_metadata = warm_start_metadata
        print(f'Resumed from {checkpoint_path} at epoch {start_epoch}; '
              f'best val loss: {best_val_loss:.6f}')
    elif warm_start_checkpoint is not None:
        warm_start_rng = copy.deepcopy(warm_start_checkpoint.get('rng_state'))
        start_epoch, source_lr, details = load_attention_split_warm_start(
            warm_start_checkpoint, model, optimizer, device
        )
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

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    if hasattr(train_loader.sampler, 'summary'):
        summary = train_loader.sampler.summary()
        sampling_path = args.val_loss_dir / 'acceleration_sampling.json'
        sampling_tmp = sampling_path.with_suffix('.json.tmp')
        with sampling_tmp.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(sampling_tmp, sampling_path)
        print(
            'Acceleration-balanced training sampler: '
            f'mode={summary["mode"]}, source acc4/acc8='
            f'{summary["source_acc4"]}/{summary["source_acc8"]}, '
            f'sampled per acceleration={summary["sampled_per_acceleration"]}, '
            f'steps per epoch={summary["samples_per_epoch"]}.'
        )

    if warm_started:
        print(
            f'Evaluating the function-preserving epoch-{start_epoch} '
            'warm-start baseline.'
        )
        baseline_result, baseline_recons, baseline_targets, baseline_time = validate(
            args, model, loss_type, val_loader, device
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
        )
        del baseline_recons, baseline_targets, warm_start_checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if start_epoch >= args.num_epochs:
        print(f'Checkpoint already reached epoch {start_epoch}; requested epochs: {args.num_epochs}.')
        return

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, loss_type, device
        )
        val_result, reconstructions, targets, val_time = validate(
            args, model, loss_type, val_loader, device
        )
        if val_result['num_subjects'] == 0:
            raise RuntimeError('Validation loader produced no subjects.')
        val_loss = float(val_result['val_loss'])

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': val_loss,
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
            'is_best': int(is_new_best),
        })
        save_training_history(history, args.val_loss_dir / 'training_history.csv')
        save_val_loss_log(history, args.val_loss_dir / 'val_loss_log.npy')
        print(f"Training history saved to {args.val_loss_dir}")

        save_model(
            args, args.exp_dir, epoch + 1, model, optimizer,
            best_val_loss, is_new_best, history,
        )
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} (final={val_result["final_score"]:.4f} '
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
