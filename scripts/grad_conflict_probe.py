"""Re-run the acc4/acc8 gradient-conflict diagnostic with configurable repeats.

In-repo reimplementation of the acceleration-gradient audit that produced
`acceleration_gradient_epoch0040_*.md` (07-26): per feature cascade and per
component, the cosine between the acc4 and acc8 mean gradients of the training
objective, plus the g8/g4 norm ratio and within-acceleration directional
coherence. The 07-26 run used 3 repeats and was too noisy to decide on a
branch; this script defaults to 10 repeats for the convergence-time re-check.

Uses the same objective as training (foreground SSIM + bbox_weight * bbox
SSIM via BboxAwareSSIMLoss -- no cv2), restricted to annotated slices by
default. Shared modules (sens_net, encoder, decoder, image cascades) are
excluded, matching the original audit. Needs the training data; run where
`/root/Data/train` (or -t) is available, GPU strongly recommended:

    python scripts/grad_conflict_probe.py -c /root/result/<exp>/checkpoints/best_model.pt \
        -t /root/Data/train --repeats 10 -o grad_conflict_report.md
"""

import argparse
import io
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(REPO_ROOT / "utils" / "model"))

from utils.common.bbox_loss import BboxAwareSSIMLoss  # noqa: E402
from utils.data.load_data import SliceData  # noqa: E402
from utils.data.transforms import DataTransform  # noqa: E402
from utils.learning.train_part import build_model  # noqa: E402

SHARED_PREFIXES = ("encoder.", "decoder.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-t", "--data-path-train", type=Path, default=Path("/root/Data/train"))
    parser.add_argument("-g", "--gpu-num", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--slices-per-acc", type=int, default=8)
    parser.add_argument("--max-per-volume", type=int, default=1)
    parser.add_argument("--bbox-weight", type=float, default=None,
                        help="Objective bbox weight; default = checkpoint args.bbox_loss_weight")
    parser.add_argument("--include-unannotated", action="store_true",
                        help="Sample from all slices instead of annotated slices only")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write the markdown report here as well as stdout")
    return parser.parse_args()


def annotated_slices_per_file(image_dir):
    """fname -> set of slice indices that carry at least one lesion box."""
    annotated = {}
    for fname in sorted(Path(image_dir).iterdir()):
        slices = set()
        with h5py.File(fname, "r") as hf:
            raw = hf.attrs.get("annotations")
        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                parsed = {}
            slices = {int(k) for k, v in parsed.items() if v}
        annotated[fname.name] = slices
    return annotated


def acc_of(name):
    if "acc8" in name:
        return 8
    if "acc4" in name:
        return 4
    return None


def build_candidates(dataset, image_dir, annotated_only):
    """acc -> {volume name -> [dataset indices]} honoring the annotation filter."""
    annotated = annotated_slices_per_file(image_dir) if annotated_only else None
    candidates = {4: defaultdict(list), 8: defaultdict(list)}
    for idx, (fname, slice_ind) in enumerate(dataset.kspace_examples):
        acc = acc_of(fname.name)
        if acc is None:
            continue
        if annotated is not None and slice_ind not in annotated.get(fname.name, ()):
            continue
        candidates[acc][fname.name].append(idx)
    return candidates


def sample_indices(candidates, rng, slices_per_acc, max_per_volume):
    picked = []
    volumes = sorted(candidates)
    rng.shuffle(volumes)
    for volume in volumes:
        if len(picked) >= slices_per_acc:
            break
        take = min(max_per_volume, slices_per_acc - len(picked))
        pool = candidates[volume]
        chosen = rng.choice(len(pool), size=min(take, len(pool)), replace=False)
        picked.extend(pool[int(i)] for i in chosen)
    return picked


def component_groups(model):
    """(cascade index, component name) -> params, excluding shared encoder/decoder."""
    groups = OrderedDict()
    for i, block in enumerate(model.cascades):
        for name, param in block.named_parameters():
            if name.startswith(SHARED_PREFIXES):
                continue
            comp = name.split(".", 1)[0]
            groups.setdefault((i, comp), []).append(param)
    return groups


def flat_grad(params, device):
    parts = []
    for p in params:
        parts.append(torch.zeros(p.numel(), device=device) if p.grad is None
                     else p.grad.detach().reshape(-1).float())
    return torch.cat(parts)


def collate_one(sample, device):
    mask, kspace, target, maximum, _, _, boxes = sample
    return (
        mask.unsqueeze(0).to(device),
        kspace.unsqueeze(0).to(device),
        target.unsqueeze(0).to(device),
        torch.tensor([float(maximum)], device=device),
        boxes.unsqueeze(0),
    )


def cosine(a, b):
    na, nb = a.norm().item(), b.norm().item()
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a.double(), b.double()) / (na * nb))


def fmt_stats(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return "n/a", "n/a", f"0/0 ({len(values)})"
    mean = statistics.fmean(valid)
    std = statistics.stdev(valid) if len(valid) > 1 else 0.0
    negative = sum(1 for v in valid if v < 0)
    return (f"{mean:.4f} +/- {std:.4f}",
            f"[{min(valid):.4f}, {max(valid):.4f}]",
            f"{negative}/{len(valid)} ({len(values)})")


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint["args"]
    model = build_model(ckpt_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()  # keep gradient checkpointing active; no dropout/batchnorm in the model

    bbox_weight = args.bbox_weight
    if bbox_weight is None:
        bbox_weight = getattr(ckpt_args, "bbox_loss_weight", 0.5)
    loss_fn = BboxAwareSSIMLoss(bbox_weight=bbox_weight).to(device)

    dataset = SliceData(
        root=args.data_path_train,
        transform=DataTransform(False, getattr(ckpt_args, "max_key", "max")),
        input_key=getattr(ckpt_args, "input_key", "kspace"),
        target_key=getattr(ckpt_args, "target_key", "image_label"),
    )
    candidates = build_candidates(dataset, args.data_path_train / "image",
                                  annotated_only=not args.include_unannotated)
    for acc in (4, 8):
        n_vol = len(candidates[acc])
        n_slice = sum(len(v) for v in candidates[acc].values())
        print(f"acc{acc}: {n_slice} candidate slices in {n_vol} volumes")
        if n_vol < args.slices_per_acc:
            print(f"  [note] fewer volumes than --slices-per-acc; "
                  f"repeats will overlap more")

    groups = component_groups(model)
    cascade_ids = sorted({i for i, _ in groups})

    # metric[key] -> list over repeats; keys are (i, comp) and ("cascade", i)
    cos_hist = defaultdict(list)
    ratio_hist = defaultdict(list)
    coh_hist = {4: defaultdict(list), 8: defaultdict(list)}

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed * 1000 + repeat)
        sums = {4: {}, 8: {}}
        norm_sums = {4: defaultdict(float), 8: defaultdict(float)}
        for acc in (4, 8):
            indices = sample_indices(candidates[acc], rng,
                                     args.slices_per_acc, args.max_per_volume)
            for idx in indices:
                mask, kspace, target, maximum, boxes = collate_one(dataset[idx], device)
                model.zero_grad(set_to_none=True)
                loss = loss_fn(model(kspace, mask), target, maximum, boxes)
                loss.backward()
                sq_by_cascade = defaultdict(float)
                for key, params in groups.items():
                    g = flat_grad(params, device)
                    sums[acc][key] = sums[acc].get(key, 0.0) + g
                    g_norm = g.norm().item()
                    norm_sums[acc][key] += g_norm
                    sq_by_cascade[key[0]] += g_norm ** 2
                for i in cascade_ids:
                    norm_sums[acc][("cascade", i)] += sq_by_cascade[i] ** 0.5
            model.zero_grad(set_to_none=True)

        for i in cascade_ids:
            comp_keys = [k for k in groups if k[0] == i]
            vec = {acc: torch.cat([sums[acc][k] for k in comp_keys]) for acc in (4, 8)}
            cos_hist[("cascade", i)].append(cosine(vec[4], vec[8]))
            n4, n8 = vec[4].norm().item(), vec[8].norm().item()
            ratio_hist[("cascade", i)].append(n8 / n4 if n4 > 0 else None)
            for acc in (4, 8):
                total = norm_sums[acc][("cascade", i)]
                coh_hist[acc][("cascade", i)].append(
                    vec[acc].norm().item() / total if total > 0 else None)
            for key in comp_keys:
                cos_hist[key].append(cosine(sums[4][key], sums[8][key]))
                c4, c8 = sums[4][key].norm().item(), sums[8][key].norm().item()
                ratio_hist[key].append(c8 / c4 if c4 > 0 else None)
                for acc in (4, 8):
                    total = norm_sums[acc][key]
                    coh_hist[acc][key].append(
                        sums[acc][key].norm().item() / total if total > 0 else None)
        print(f"repeat {repeat + 1}/{args.repeats} done")

    def mean_or_na(values):
        valid = [v for v in values if v is not None]
        return f"{statistics.fmean(valid):.3f}" if valid else "n/a"

    param_count = {key: sum(p.numel() for p in params) for key, params in groups.items()}
    out = io.StringIO()
    out.write(f"# Acceleration Gradient Diagnostic (repeats={args.repeats})\n\n")
    out.write(f"- Checkpoint: `{args.checkpoint}` (epoch {checkpoint.get('epoch')})\n")
    out.write(f"- Data: `{args.data_path_train}`, "
              f"{'annotated slices only' if not args.include_unannotated else 'all slices'}, "
              f"objective foreground + {bbox_weight} * bbox\n")
    out.write(f"- Sampling: {args.repeats} repeats x {args.slices_per_acc} slices per "
              f"acceleration; at most {args.max_per_volume} per volume; seed {args.seed}\n")
    out.write("- Gradients: mean per acceleration first, cosine second\n")
    out.write("- Shared encoder/decoder, sens_net and image cascades: excluded\n\n")

    out.write("## Cascade summary\n\n")
    out.write("| Cascade | Parameters | Cosine mean +/- std | Range | Negative / valid (total) "
              "| Norm ratio (g8/g4) | Coherence acc4 / acc8 |\n")
    out.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for i in cascade_ids:
        key = ("cascade", i)
        n_params = sum(param_count[k] for k in groups if k[0] == i)
        mean_std, rng_txt, neg = fmt_stats(cos_hist[key])
        out.write(f"| F{i} | {n_params:,} | {mean_std} | {rng_txt} | {neg} "
                  f"| {mean_or_na(ratio_hist[key])} "
                  f"| {mean_or_na(coh_hist[4][key])} / {mean_or_na(coh_hist[8][key])} |\n")

    out.write("\n## Component summary\n\n")
    out.write("| Cascade | Component | Parameters | Cosine mean +/- std "
              "| Negative / valid (total) | Norm ratio (g8/g4) | Coherence acc4 / acc8 |\n")
    out.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for key in groups:
        i, comp = key
        mean_std, _, neg = fmt_stats(cos_hist[key])
        out.write(f"| F{i} | {comp} | {param_count[key]:,} | {mean_std} | {neg} "
                  f"| {mean_or_na(ratio_hist[key])} "
                  f"| {mean_or_na(coh_hist[4][key])} / {mean_or_na(coh_hist[8][key])} |\n")

    out.write("\n## Reading the result\n\n")
    out.write("- A stable negative cosine means acc4 and acc8 ask that parameter group to "
              "move in opposing directions on these samples.\n")
    out.write("- Coherence near zero means within-acceleration gradients cancel, making the "
              "mean-gradient cosine a weak basis for a branch decision.\n")
    out.write("- `dc_weight` is one scalar, so its cosine is necessarily near +1 or -1.\n")
    out.write("- Branch only on whole-cascade conflict that is stable across repeats; the "
              "07-26 3-repeat audit found none at epoch 40.\n")

    report = out.getvalue()
    print("\n" + report)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
