"""Memory-conscious CPU-only reconstruction and leaderboard evaluation.

This script is intended to run beside a GPU training process on VESSL. It
hides every GPU before importing torch, limits CPU threads, and reads one
slice at a time so a complete image/k-space volume is never retained in RAM.
The SSIM functions are the same functions used by recon_eval.py. Timing is
intentionally omitted because CPU timing is not a valid submission metric.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    with open("/proc/self/oom_score_adj", "w", encoding="ascii") as oom_file:
        oom_file.write("1000")
except (OSError, PermissionError):
    pass

import h5py
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full
from utils.learning.test_part import recon_slice
from utils.learning.train_part import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only leaderboard reconstruction/evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Exact best_model.pt snapshot to evaluate",
    )
    parser.add_argument(
        "-p",
        "--path_data",
        type=Path,
        default=Path("/root/Data/leaderboard"),
        help="Directory containing acc4/ and acc8/",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="CPU threads; keep low while GPU training is active",
    )
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num_threads must be at least 1")
    return args


def apply_thread_limit(num_threads):
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def load_checkpoint_model(checkpoint_path, device):
    """Load the inference model, releasing optimizer tensors before build."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")

    saved_args = checkpoint.get("args")
    if saved_args is None:
        raise KeyError(
            "Checkpoint has no saved 'args'; the architecture cannot be rebuilt safely."
        )
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    state_dict = checkpoint.pop("model")
    del checkpoint
    gc.collect()

    model = build_model(saved_args).to(device=device)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict
    gc.collect()
    return model, checkpoint_epoch


def volume_paths(acc_dir):
    image_dir = acc_dir / "image"
    paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No image volumes found in {image_dir}")
    return paths


def run_acc(model, ssim, device, acc_dir):
    image_paths = volume_paths(acc_dir)
    kspace_dir = acc_dir / "kspace"

    full_total = 0.0
    full_count = 0
    bbox_total = 0.0
    bbox_count = 0
    slice_count = 0

    for image_path in tqdm(
        image_paths, desc=f"[{acc_dir.name}] volumes", unit="vol"
    ):
        kspace_path = kspace_dir / image_path.name
        if not kspace_path.is_file():
            raise FileNotFoundError(f"Missing matching k-space: {kspace_path}")

        with h5py.File(image_path, "r") as image_hf, h5py.File(
            kspace_path, "r"
        ) as kspace_hf:
            image_dataset = image_hf["image_label"]
            kspace_dataset = kspace_hf["kspace"]
            if len(image_dataset) != len(kspace_dataset):
                raise ValueError(
                    f"Slice count mismatch for {image_path.name}: "
                    f"image={len(image_dataset)}, kspace={len(kspace_dataset)}"
                )

            maximum = image_hf.attrs["max"]
            annotations = json.loads(image_hf.attrs.get("annotations", "{}"))
            mask = np.asarray(kspace_hf["mask"])

            with torch.inference_mode():
                for slice_index in range(len(image_dataset)):
                    context = {
                        "kspace": kspace_dataset[slice_index : slice_index + 1],
                        "mask": mask,
                        "device": device,
                        "num_slices": 1,
                    }
                    reconstruction = recon_slice(model, context, 0)
                    target_array = image_dataset[slice_index]
                    target = torch.from_numpy(target_array).to(device=device)
                    foreground = torch.from_numpy(
                        foreground_mask(target_array)
                    ).to(device=device, dtype=torch.float32)

                    value = ssim_full(
                        ssim, reconstruction, target, foreground, maximum
                    )
                    if value is not None:
                        full_total += value
                        full_count += 1

                    for box in annotations.get(str(slice_index), []):
                        value = ssim_bbox(
                            ssim, reconstruction, target, box, maximum
                        )
                        if value is not None:
                            bbox_total += value
                            bbox_count += 1

                    slice_count += 1
                    del context, reconstruction, target, foreground, target_array

        gc.collect()

    full_score = full_total / full_count if full_count else 0.0
    bbox_score = bbox_total / bbox_count if bbox_count else 0.0
    return full_score, bbox_score, slice_count


def validate_inputs(args):
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {args.checkpoint}")
    for acceleration in ("acc4", "acc8"):
        for subdirectory in ("image", "kspace"):
            path = args.path_data / acceleration / subdirectory
            if not path.is_dir():
                raise FileNotFoundError(f"Missing leaderboard directory: {path}")


def main():
    args = parse_args()
    apply_thread_limit(args.num_threads)
    validate_inputs(args)
    device = torch.device("cpu")

    print("=" * 60, flush=True)
    print("CPU-only leaderboard evaluation", flush=True)
    print(f"Checkpoint : {args.checkpoint.resolve()}", flush=True)
    print(f"Data       : {args.path_data.resolve()}", flush=True)
    print(f"Threads    : {args.num_threads}", flush=True)
    print("GPU        : hidden / unused", flush=True)
    print("Timing     : N/A (CPU-only mode)", flush=True)
    print("=" * 60, flush=True)

    model, checkpoint_epoch = load_checkpoint_model(args.checkpoint, device)
    print(f"Checkpoint epoch: {checkpoint_epoch}", flush=True)
    ssim = SSIM().to(device=device)

    full4, bbox4, slices4 = run_acc(
        model, ssim, device, args.path_data / "acc4"
    )
    full8, bbox8, slices8 = run_acc(
        model, ssim, device, args.path_data / "acc8"
    )

    print("", flush=True)
    print(
        "Leaderboard SSIM_full : {:.4f}".format((full4 + full8) / 2),
        flush=True,
    )
    print(
        "Leaderboard SSIM_bbox : {:.4f}".format((bbox4 + bbox8) / 2),
        flush=True,
    )
    print("Leaderboard Recon Time: N/A (CPU-only mode)", flush=True)
    print("=" * 10 + " Details " + "=" * 10, flush=True)
    print(
        "SSIM_full (acc4): {:.4f}   SSIM_full (acc8): {:.4f}".format(
            full4, full8
        ),
        flush=True,
    )
    print(
        "SSIM_bbox (acc4): {:.4f}   SSIM_bbox (acc8): {:.4f}".format(
            bbox4, bbox8
        ),
        flush=True,
    )
    print(f"Total slices: {slices4 + slices8} (acc4: {slices4}, acc8: {slices8})")


if __name__ == "__main__":
    main()
