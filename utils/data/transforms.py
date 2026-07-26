import json

import numpy as np
import torch

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
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            boxes = annotation_boxes(attrs, slice)
        else:
            target = -1
            maximum = -1
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice, boxes
