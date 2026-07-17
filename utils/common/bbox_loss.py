"""Metric-aligned training loss: L = L_foreground_ssim + lambda * L_bbox_ssim.

Both terms mirror the leaderboard scoring in `utils/common/metrics.py`
(which must not be modified):
- L_foreground_ssim: differentiable SSIM map averaged only inside the
  foreground mask, exactly as `metrics.ssim_full`.
- L_bbox_ssim: SSIMLoss on each annotation box crop (boxes smaller than the
  SSIM window are skipped), exactly as `metrics.ssim_bbox`, averaged over
  boxes. Slices without annotations skip this term.

The foreground mask reimplements `metrics.foreground_mask` (cv2 morphology)
with max-pooling so it runs on the GPU without a cv2 dependency: binary
3x3 erode == 1 - maxpool(1 - x) and 3x3 dilate == maxpool(x), and cv2's
default border values (+inf for erode, -inf for dilate) match zero padding
on the inverted/plain mask. The mask depends only on the target, so no
gradients flow through it.

With bbox_weight=0 the loss is the pure foreground SSIM loss, which
isolates the effect of a model change from the loss change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common.loss_function import SSIMLoss


def _dilate(x: torch.Tensor, iterations: int) -> torch.Tensor:
    for _ in range(iterations):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


def _erode(x: torch.Tensor, iterations: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - x, iterations)


def foreground_mask(target: torch.Tensor, threshold: float = 2e-5) -> torch.Tensor:
    """Torch port of metrics.foreground_mask. target (B, H, W) -> mask (B, 1, H, W)."""
    with torch.no_grad():
        mask = (target > threshold).to(target.dtype).unsqueeze(1)
        mask = _erode(mask, 1)
        mask = _dilate(mask, 15)
        mask = _erode(mask, 14)
    return mask


class BboxAwareSSIMLoss(nn.Module):
    """loss(output, target, maximum, boxes) with output/target (B, H, W).

    boxes: (B, N, 4) or (N, 4) tensor of [x, y, width, height] in image
    coordinates (384x384), N may be 0. Assumes batch size 1 for the bbox term
    (the data loader collates one slice's boxes into (1, N, 4)).
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03,
                 bbox_weight: float = 1.0):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.bbox_weight = bbox_weight
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)
        self.bbox_ssim = SSIMLoss(win_size=win_size, k1=k1, k2=k2)

    def _ssim_map(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """Per-pixel SSIM map, same math as metrics.SSIM. X, Y: (B, 1, H, W)."""
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        return (A1 * A2) / (B1 * B2)

    def _foreground_loss(self, output, target, data_range):
        fg = foreground_mask(target)
        S = self._ssim_map(output.unsqueeze(1) * fg, target.unsqueeze(1) * fg, data_range)
        pad = self.win_size // 2
        fg_valid = fg[..., pad: fg.shape[-2] - pad, pad: fg.shape[-1] - pad]
        denom = fg_valid.sum()
        if denom > 0:
            return 1 - (S * fg_valid).sum() / denom
        return 1 - S.mean()

    def _bbox_loss_terms(self, output, target, maximum, boxes):
        if boxes is None:
            return []
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        height, width = target.shape[-2], target.shape[-1]
        terms = []
        for b in range(min(output.shape[0], boxes.shape[0])):
            for box in boxes[b]:
                x, y, w, h = (int(v) for v in box.tolist())
                x0, y0 = max(0, x), max(0, y)
                x1 = min(width, x + w)
                y1 = min(height, y + h)
                if (x1 - x0) < self.win_size or (y1 - y0) < self.win_size:
                    continue
                terms.append(self.bbox_ssim(
                    output[b: b + 1, y0:y1, x0:x1],
                    target[b: b + 1, y0:y1, x0:x1],
                    maximum[b: b + 1],
                ))
        return terms

    def forward(self, output, target, maximum, boxes=None):
        maximum = maximum.to(dtype=output.dtype)
        data_range = maximum[:, None, None, None]
        loss = self._foreground_loss(output, target, data_range)

        if self.bbox_weight != 0:
            terms = self._bbox_loss_terms(output, target, maximum, boxes)
            if terms:
                loss = loss + self.bbox_weight * torch.stack(terms).mean()
        return loss
