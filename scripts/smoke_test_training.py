"""Run one real training slice through forward and backward before a long job."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "utils" / "model"
for path in (REPO_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.common.loss_function import SSIMLoss  # noqa: E402
from utils.data.load_data import SliceData  # noqa: E402
from utils.data.transforms import DataTransform  # noqa: E402
from utils.model.varnet import VarNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="One-slice VarNet training smoke test")
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--cascade", type=int, default=6)
    parser.add_argument("--chans", type=int, default=12)
    parser.add_argument("--sens-chans", type=int, default=4)
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

    mask, kspace, target, maximum, filename, slice_index = dataset[0]
    mask = mask.unsqueeze(0).to(device=device)
    kspace = kspace.unsqueeze(0).to(device=device)
    target = target.unsqueeze(0).to(device=device)
    maximum = torch.as_tensor([maximum], device=device)

    model = VarNet(
        num_cascades=args.cascade,
        chans=args.chans,
        sens_chans=args.sens_chans,
    ).to(device=device)
    loss_fn = SSIMLoss().to(device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    output = model(kspace, mask)
    loss = loss_fn(output, target, maximum)
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke test produced a non-finite output or loss")
    loss.backward()
    torch.cuda.synchronize(device)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"Smoke-test file: {filename}, slice: {slice_index}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"k-space shape: {tuple(kspace.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"target/output shape: {tuple(target.shape)} / {tuple(output.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(f"peak allocated GPU memory: {peak_mib:.1f} MiB")
    print("Training smoke test: OK")


if __name__ == "__main__":
    main()
