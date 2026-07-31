import json

import numpy as np
import torch

from utils.data.mraugment import (
    MRAugmentParameters,
    augment_multicoil_kspace,
    augmentation_probability,
    deterministic_rng,
    infer_acquired_column_bounds,
    infer_equispaced_mask,
    make_equispaced_mask,
    sample_parameters,
    transform_boxes,
)


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


def annotation_boxes(attrs, slice_idx):
    """Parse the `annotations` attribute (JSON, 384x384 image coordinates) of an
    image H5 and return this slice's lesion boxes as an (N, 4) float tensor of
    [x, y, width, height]. Returns (0, 4) when there is no annotation.
    """
    empty = torch.zeros((0, 4), dtype=torch.float32)
    if not isinstance(attrs, dict):
        return empty
    raw = attrs.get('annotations')
    if raw is None:
        return empty
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return empty
    entries = parsed.get(str(int(slice_idx)), [])
    boxes = [
        [float(e['x']), float(e['y']), float(e['width']), float(e['height'])]
        for e in entries
    ]
    if not boxes:
        return empty
    return torch.tensor(boxes, dtype=torch.float32)


class DataTransform:
    def __init__(self, isforward, max_key, args=None, is_train=False):
        self.isforward = isforward
        self.max_key = max_key
        self.args = args
        self.is_train = bool(is_train)
        self.epoch = 0
        self._mask_specifications = {}

    @property
    def mraugment_enabled(self):
        return (
            self.is_train
            and not self.isforward
            and bool(getattr(self.args, "mraugment", False))
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def augmentation_probability(self):
        if not self.mraugment_enabled:
            return 0.0
        total_epochs = getattr(self.args, "mraugment_total_epochs", None)
        if total_epochs is None:
            total_epochs = getattr(self.args, "num_epochs", 1)
        return augmentation_probability(
            self.epoch,
            total_epochs,
            schedule=getattr(self.args, "mraugment_schedule", "exp"),
            maximum=getattr(self.args, "mraugment_strength", 0.55),
            decay=getattr(self.args, "mraugment_exp_decay", 5.0),
            delay=getattr(self.args, "mraugment_delay_epochs", 0),
        )

    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            maximum = attrs[self.max_key]
            boxes = annotation_boxes(attrs, slice)
        else:
            target = -1
            maximum = -1
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        kspace_input = np.asarray(input, dtype=np.complex64)
        applied_mask = np.asarray(mask).astype(bool).reshape(-1)
        if self.mraugment_enabled:
            source_width = kspace_input.shape[-1]
            acquired_left, acquired_right = infer_acquired_column_bounds(
                kspace_input
            )
            specification = self._mask_specifications.get(fname)
            if specification is None:
                specification = infer_equispaced_mask(applied_mask)
                self._mask_specifications[fname] = specification
            probability = self.augmentation_probability()
            seed = getattr(
                self.args,
                "mraugment_seed",
                getattr(self.args, "seed", 42),
            )
            augmentation_rng = deterministic_rng(
                seed, self.epoch, fname, slice, purpose="augment"
            )
            parameters = sample_parameters(
                augmentation_rng, probability, kspace_input.shape[-2:]
            )

            transformed_boxes = transform_boxes(
                boxes.numpy(),
                parameters,
                kspace_input.shape[-2:],
                target_shape=np.asarray(target).shape[-2:],
                minimum_size=getattr(self.args, "mraugment_min_bbox_size", 7),
            )
            if transformed_boxes is None:
                # Bbox supervision cannot describe a lesion that was cropped
                # away or reduced below the SSIM window. Cancel geometry only;
                # acquisition-mask resampling remains active.
                parameters = MRAugmentParameters()
                transformed_boxes = boxes.numpy()

            if not parameters.is_identity:
                kspace_input, target = augment_multicoil_kspace(
                    kspace_input,
                    parameters,
                    output_shape=np.asarray(target).shape[-2:],
                )
            boxes = torch.from_numpy(
                np.asarray(transformed_boxes, dtype=np.float32).reshape(-1, 4)
            )

            # The source data contains complete k-space. Generate a valid mask
            # after augmentation, retaining its R and ACS fraction. The offset
            # varies by volume and epoch but is shared by all volume slices.
            mask_rng = deterministic_rng(
                seed, self.epoch, fname, purpose="mask"
            )
            offset = int(mask_rng.integers(0, specification.acceleration))
            applied_mask = make_equispaced_mask(
                kspace_input.shape[-1], specification, offset=offset
            )
            output_width = kspace_input.shape[-1]
            output_left = int(round(acquired_left / source_width * output_width))
            output_right = int(
                round(acquired_right / source_width * output_width)
            )
            applied_mask[:output_left] = False
            applied_mask[output_right:] = False

        if not self.isforward:
            target = to_tensor(np.asarray(target, dtype=np.float32))

        kspace = to_tensor(
            kspace_input * applied_mask.reshape(1, 1, -1)
        )
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(
            applied_mask.reshape(1, 1, kspace.shape[-2], 1)
        ).byte()
        return mask, kspace, target, maximum, fname, slice, boxes
