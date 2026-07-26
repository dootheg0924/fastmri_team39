"""Report how far each cascade's acc-FiLM parameters moved apart per acceleration.

Reads a checkpoint trained with --acc-film and prints, per feature cascade,
how far gamma/beta drifted from identity for each acceleration and how far
the acc4 and acc8 rows diverged from each other. Large divergence in a
cascade means the model wanted acceleration-specific processing there --
evidence for (and a location hint for) a hard acc4/acc8 branch.

CPU-only, no data or cv2 needed, so this runs on any machine:

    python scripts/film_divergence.py /path/to/best_model.pt [--top 5]
"""

import argparse
import re
import sys

import torch


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint.get("epoch")
    return checkpoint, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="Checkpoint file (model.pt / best_model.pt)")
    parser.add_argument("--top", type=int, default=5,
                        help="Channels with the largest |gamma4 - gamma8| to list per cascade")
    args = parser.parse_args()

    state, epoch = load_state_dict(args.checkpoint)
    film_keys = sorted(
        (int(m.group(1)), key)
        for key in state
        for m in [re.match(r"cascades\.(\d+)\.acc_film\.weight$", key)]
        if m
    )
    if not film_keys:
        sys.exit("No cascades.<i>.acc_film.weight keys found -- was the model "
                 "trained with --acc-film?")

    print(f"# FiLM acc4/acc8 divergence: {args.checkpoint}")
    if epoch is not None:
        print(f"- epoch: {epoch}")
    print()
    print("| Cascade | RMS(g4-1) | RMS(g8-1) | RMS(g4-g8) | max|g4-g8| | RMS(b4) | RMS(b8) | RMS(b4-b8) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    rms = lambda t: float(t.pow(2).mean().sqrt())
    per_cascade = []
    for idx, key in film_keys:
        w = state[key].float()  # (2, 2C): row 0 = acc4, row 1 = acc8
        chans = w.shape[1] // 2
        g4, b4 = 1.0 + w[0, :chans], w[0, chans:]
        g8, b8 = 1.0 + w[1, :chans], w[1, chans:]
        dg = g4 - g8
        per_cascade.append((idx, dg.abs(), rms(dg) + rms(b4 - b8)))
        print(f"| F{idx} | {rms(g4 - 1):.4f} | {rms(g8 - 1):.4f} | {rms(dg):.4f} "
              f"| {float(dg.abs().max()):.4f} | {rms(b4):.4f} | {rms(b8):.4f} "
              f"| {rms(b4 - b8):.4f} |")

    print()
    ranked = sorted(per_cascade, key=lambda item: item[2], reverse=True)
    order = ", ".join(f"F{idx} ({score:.4f})" for idx, _, score in ranked)
    print(f"Divergence ranking (RMS(g4-g8) + RMS(b4-b8)): {order}")
    print()
    for idx, dg_abs, _ in per_cascade:
        top = torch.topk(dg_abs, k=min(args.top, dg_abs.numel()))
        pairs = ", ".join(f"ch{int(i)}={float(v):.4f}" for v, i in zip(top.values, top.indices))
        print(f"- F{idx} top |g4-g8| channels: {pairs}")

    print()
    print("Reading the result: values near 0 everywhere mean the model found no "
          "use for acceleration-specific modulation (a hard branch is unlikely to "
          "help); divergence concentrated in the last cascades supports branching "
          "only those cascades.")


if __name__ == "__main__":
    main()
