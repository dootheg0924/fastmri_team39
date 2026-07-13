import shutil
import csv
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
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

HISTORY_FIELDS = [
    'epoch', 'train_loss', 'val_loss', 'train_time_sec', 'val_time_sec',
    'learning_rate', 'is_best',
]


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.to(device=device, non_blocking=True)
        kspace = kspace.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        maximum = maximum.to(device=device, non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
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


def validate(args, model, data_loader, device):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
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
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


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
        'best_val_loss': float(best_val_loss),
        'exp_dir': exp_dir,
        'history': history,
        'rng_state': capture_rng_state(),
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
    return checkpoint['epoch'], float(best_val_loss), checkpoint.get('history', [])


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

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = float('inf')
    start_epoch = 0
    history = []

    checkpoint_path = args.exp_dir / 'model.pt'
    if args.resume and checkpoint_path.exists():
        start_epoch, best_val_loss, history = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        print(f'Resumed from {checkpoint_path} at epoch {start_epoch}; '
              f'best val loss: {best_val_loss:.6f}')
    elif args.resume:
        print(f'No checkpoint found at {checkpoint_path}; starting a new run.')

    write_run_metadata(args, device)

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    if start_epoch >= args.num_epochs:
        print(f'Checkpoint already reached epoch {start_epoch}; requested epochs: {args.num_epochs}.')
        return

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(
            args, epoch, model, train_loader, optimizer, loss_type, device
        )
        val_loss_sum, num_subjects, reconstructions, targets, inputs, val_time = validate(
            args, model, val_loader, device
        )
        if num_subjects == 0:
            raise RuntimeError('Validation loader produced no subjects.')
        val_loss = float(val_loss_sum / num_subjects)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': val_loss,
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
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
