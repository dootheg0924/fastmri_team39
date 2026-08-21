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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "utils" / "model"
for path in (REPO_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.common.bbox_loss import BboxAwareSSIMLoss  # noqa: E402
from utils.data.load_data import SliceData  # noqa: E402
from utils.data.transforms import DataTransform  # noqa: E402
from utils.learning.train_part import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="One-slice training smoke test")
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--slice-index", type=int, default=0, help="Dataset index to use")
    parser.add_argument("--model-name", type=str, default="fivarnet",
                        choices=["varnet", "fivarnet"])
    parser.add_argument("--cascade", type=int, default=4)
    parser.add_argument("--image-cascades", type=int, default=2)
    parser.add_argument("--chans", type=int, default=12)
    parser.add_argument("--sens-chans", type=int, default=4)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens-pools", type=int, default=4)
    parser.add_argument("--attention-cascades", type=int, nargs="*", default=[0])
    parser.add_argument("--kspace-mult-factor", type=float, default=1e6)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0)
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

    mask, kspace, target, maximum, filename, slice_index, boxes = dataset[args.slice_index]
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
        no_grad_checkpoint=args.no_grad_checkpoint,
    )
    model = build_model(model_args).to(device=device)
    loss_fn = BboxAwareSSIMLoss(bbox_weight=args.bbox_loss_weight).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    output = model(kspace, mask)
    loss = loss_fn(output, target, maximum, boxes)
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke test produced a non-finite output or loss")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"Smoke-test file: {filename}, slice: {slice_index}, boxes: {tuple(boxes.shape)}")
    print(f"Model: {args.model_name}, parameters: {parameter_count:,}")
    print(f"k-space shape: {tuple(kspace.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"target/output shape: {tuple(target.shape)} / {tuple(output.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(f"peak allocated GPU memory: {peak_mib:.1f} MiB")
    if peak_mib > 7300:
        print("[WARNING] peak VRAM is close to/over the 8GB budget "
              "(ladder: sens-chans down -> fewer attention-cascades -> chans down -> fewer cascades)")
    print("Training smoke test: OK")


if __name__ == "__main__":
    main()
