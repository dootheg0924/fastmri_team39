"""Run one real training slice through forward and backward before a long job.

Reports peak VRAM (torch.cuda.max_memory_allocated) so an 8GB fit can be
verified before launching. Supports both the baseline VarNet and the exp/003
FIVarNet via --model-name (built through train_part.build_model, i.e. the
exact model construction used by train.py).
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "utils" / "model"
for path in (REPO_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.common.bbox_loss import BboxAwareSSIMLoss  # noqa: E402
from utils.data.load_data import SliceData  # noqa: E402
from utils.data.transforms import DataTransform  # noqa: E402
from utils.learning.train_part import (  # noqa: E402
    build_lr_scheduler,
    build_model,
    build_optimizer,
)


def _mask_acceleration(mask: np.ndarray) -> int:
    """Infer the outer Cartesian sampling stride used by FI-VarNet attention."""
    sampled = np.flatnonzero(np.asarray(mask).reshape(-1))
    if sampled.size >= 2:
        gaps = np.diff(sampled)
        outer = gaps[gaps > 1]
        if outer.size:
            values, counts = np.unique(outer, return_counts=True)
            stride = int(values[np.argmax(counts)])
            if 2 <= stride <= 16:
                return stride
    return 4


def _select_stress_slice(dataset):
    """Choose the volume most likely to have the largest activation footprint.

    Attention memory scales approximately with spatial pixels times the
    acceleration, while sensitivity estimation also grows with coil count.
    The first slice of the lexicographically largest candidate is sufficient
    because those dimensions are constant within a volume.
    """
    candidates = []
    seen = set()
    for dataset_index, (path, _) in enumerate(dataset.kspace_examples):
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        with h5py.File(path, "r") as hf:
            kspace = hf[dataset.input_key]
            shape = tuple(int(dim) for dim in kspace.shape)
            if len(shape) < 4:
                raise ValueError(
                    f"Expected slice/coil/height/width k-space at {path}, got {shape}"
                )
            coils, height, width = shape[-3:]
            acceleration = _mask_acceleration(hf["mask"][()])
            attention_pressure = height * width * acceleration
            input_elements = coils * height * width
        candidates.append(
            (
                attention_pressure,
                input_elements,
                dataset_index,
                path.name,
                coils,
                height,
                width,
                acceleration,
            )
        )
    if not candidates:
        raise RuntimeError("Training dataset has no k-space volumes")
    return max(candidates)


def main():
    parser = argparse.ArgumentParser(description="One-slice training smoke test")
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Dataset index to use; default scans metadata and selects the "
             "largest spatial/acceleration candidate",
    )
    parser.add_argument("--model-name", type=str, default="fivarnet",
                        choices=["varnet", "fivarnet"])
    parser.add_argument("--cascade", type=int, default=4)
    parser.add_argument("--image-cascades", type=int, default=2)
    parser.add_argument("--chans", type=int, default=12)
    parser.add_argument("--sens-chans", type=int, default=4)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens-pools", type=int, default=4)
    parser.add_argument("--attention-cascades", type=int, nargs="*", default=[0])
    parser.add_argument("--split-attention-cascades", type=int, nargs="*", default=[])
    parser.add_argument("--kspace-mult-factor", type=float, default=1e6)
    parser.add_argument(
        "--feature-processor",
        choices=["norm-unet", "paper-unet2d"],
        default="norm-unet",
    )
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--acc-film", action="store_true")
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--paper-training",
        action="store_true",
        help="Use paper AdamW/scheduler/clipping with the configured "
             "exp/003 bbox-aware SSIM objective",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA is required for the long-run smoke test")
    device = torch.device(f"cuda:{args.gpu_num}")
    torch.cuda.set_device(device)

    dataset = SliceData(
        root=args.data_root / "train",
        transform=DataTransform(False, "max"),
        input_key="kspace",
        target_key="image_label",
    )
    if len(dataset) == 0:
        raise RuntimeError("Training dataset has no slices")

    if args.slice_index is None:
        (
            _,
            _,
            selected_index,
            selected_file,
            selected_coils,
            selected_height,
            selected_width,
            selected_acceleration,
        ) = _select_stress_slice(dataset)
        print(
            "Auto-selected VRAM stress candidate: "
            f"{selected_file}, coils={selected_coils}, "
            f"matrix={selected_height}x{selected_width}, "
            f"acceleration={selected_acceleration}"
        )
    else:
        selected_index = args.slice_index

    mask, kspace, target, maximum, filename, slice_index, boxes = dataset[selected_index]
    mask = mask.unsqueeze(0).to(device=device)
    kspace = kspace.unsqueeze(0).to(device=device)
    target = target.unsqueeze(0).to(device=device)
    maximum = torch.as_tensor([maximum], device=device)

    model_args = SimpleNamespace(
        model_name=args.model_name,
        cascade=args.cascade,
        image_cascades=args.image_cascades,
        chans=args.chans,
        sens_chans=args.sens_chans,
        pools=args.pools,
        sens_pools=args.sens_pools,
        attention_cascades=args.attention_cascades,
        kspace_mult_factor=args.kspace_mult_factor,
        feature_processor=args.feature_processor,
        no_grad_checkpoint=args.no_grad_checkpoint,
        acc_film=args.acc_film,
        split_attention_cascades=args.split_attention_cascades,
    )
    model = build_model(model_args).to(device=device)
    if args.paper_training:
        torch.set_float32_matmul_precision("high")
        optimization_args = SimpleNamespace(
            lr=3e-4,
            optimizer="adamw",
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            adam_amsgrad=False,
            lr_scheduler="fi-varnet-paper",
            max_steps=210_000,
            lr_warmup_steps=7_500,
            lr_cosine_start_step=150_000,
            lr_min_factor=1e-8,
        )
        loss_fn = BboxAwareSSIMLoss(
            bbox_weight=args.bbox_loss_weight
        ).to(device=device)
        optimizer = build_optimizer(optimization_args, model)
        scheduler = build_lr_scheduler(optimization_args, optimizer)
    else:
        loss_fn = BboxAwareSSIMLoss(
            bbox_weight=args.bbox_loss_weight
        ).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler = None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(kspace, mask)
    loss = loss_fn(output, target, maximum, boxes)
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke test produced a non-finite output or loss")
    loss.backward()
    if args.split_attention_cascades:
        acceleration = model._infer_acceleration(mask)
        for cascade_index in args.split_attention_cascades:
            block = model.cascades[cascade_index]
            active = (
                block.attention_layer_acc8
                if acceleration >= 6
                else block.attention_layer
            )
            inactive = (
                block.attention_layer
                if acceleration >= 6
                else block.attention_layer_acc8
            )
            if not any(parameter.grad is not None for parameter in active.parameters()):
                raise RuntimeError(
                    f"Cascade {cascade_index} selected attention expert received no gradient."
                )
            if any(parameter.grad is not None for parameter in inactive.parameters()):
                raise RuntimeError(
                    f"Cascade {cascade_index} inactive attention expert received a gradient."
                )
    if args.paper_training:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    torch.cuda.synchronize(device)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    print(f"Smoke-test file: {filename}, slice: {slice_index}, boxes: {tuple(boxes.shape)}")
    print(f"Model: {args.model_name}, parameters: {parameter_count:,}")
    print(f"k-space shape: {tuple(kspace.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"target/output shape: {tuple(target.shape)} / {tuple(output.shape)}")
    print(
        f"loss: {loss.item():.6f} "
        f"((1-SSIM_full) + {args.bbox_loss_weight:g} * (1-SSIM_bbox))"
    )
    print(
        f"optimizer/schedule: {optimizer.__class__.__name__}, "
        f"lr={optimizer.param_groups[0]['lr']:.6g}"
    )
    print(
        f"GPU memory: allocated peak={peak_mib:.1f} MiB, "
        f"reserved peak={peak_reserved_mib:.1f} MiB, total={total_mib:.1f} MiB"
    )
    if total_mib <= 10 * 1024 and max(peak_mib, peak_reserved_mib) > 0.92 * total_mib:
        raise SystemExit(
            "[ERROR] Smoke test leaves less than 8% GPU-memory headroom. "
            "Do not start the long run; reducing sens_chans or chans would be "
            "required and would no longer be the paper 6+6 configuration."
        )
    print("Training smoke test: OK")


if __name__ == "__main__":
    main()
