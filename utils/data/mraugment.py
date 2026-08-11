"""Paper-aligned MRAugment for fully sampled multi-coil MRI data.

The implementation follows Fabian et al. (ICML 2021): transform every real
and imaginary coil image with identical parameters, Fourier transform the
result, and only then apply an acquisition mask.  The affine implementation
uses the paper's bicubic interpolation and separate isotropic/anisotropic
scaling rather than the simplified torchvision implementation in the current
public repository.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from skimage.transform import AffineTransform, warp


@dataclass(frozen=True)
class MaskSpecification:
    acceleration: int
    offset: int
    center_fraction: float
    num_low_frequencies: int


@dataclass(frozen=True)
class MRAugmentParameters:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rot90_k: int = 0
    translation_y: int = 0
    translation_x: int = 0
    rotation_degrees: float = 0.0
    isotropic_scale: float = 1.0
    anisotropic_scale_y: float = 1.0
    anisotropic_scale_x: float = 1.0
    shear_degrees: float = 0.0

    @property
    def has_affine(self):
        return (
            self.rotation_degrees != 0.0
            or self.isotropic_scale != 1.0
            or self.anisotropic_scale_y != 1.0
            or self.anisotropic_scale_x != 1.0
            or self.shear_degrees != 0.0
        )

    @property
    def is_identity(self):
        return not (
            self.horizontal_flip
            or self.vertical_flip
            or self.rot90_k
            or self.translation_y
            or self.translation_x
            or self.has_affine
        )


def augmentation_probability(
    epoch,
    total_epochs,
    *,
    schedule="exp",
    maximum=0.55,
    decay=5.0,
    delay=0,
):
    """Return the paper's epoch-level augmentation strength ``p(t)``."""
    epoch = float(epoch)
    total_epochs = float(total_epochs)
    delay = float(delay)
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("MRAugment maximum probability must be in [0, 1].")
    if total_epochs <= delay:
        raise ValueError("MRAugment total epochs must be greater than its delay.")
    if epoch < delay:
        return 0.0

    progress = max(0.0, min(1.0, (epoch - delay) / (total_epochs - delay)))
    if schedule == "constant":
        return float(maximum)
    if schedule == "ramp":
        return float(maximum * progress)
    if schedule == "exp":
        if decay <= 0:
            raise ValueError("MRAugment exponential decay must be positive.")
        return float(
            maximum
            * (1.0 - math.exp(-decay * progress))
            / (1.0 - math.exp(-decay))
        )
    raise ValueError(f"Unknown MRAugment schedule: {schedule}")


def deterministic_rng(seed, epoch, fname, slice_num=None, purpose="augment"):
    """Build a stable per-example RNG independent of worker scheduling."""
    parts = [str(int(seed)), str(int(epoch)), str(fname), str(purpose)]
    if slice_num is not None:
        parts.append(str(int(slice_num)))
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8)
    return np.random.default_rng(int.from_bytes(digest.digest(), "little"))


def _center_bounds(width, num_low_frequencies):
    start = (int(width) - int(num_low_frequencies) + 1) // 2
    return start, start + int(num_low_frequencies)


def make_equispaced_mask(width, specification, offset=None):
    """Create a 1-D equispaced mask with a centered fully sampled ACS region."""
    width = int(width)
    acceleration = int(specification.acceleration)
    if width <= 0 or acceleration <= 0:
        raise ValueError("Mask width and acceleration must be positive.")
    if offset is None:
        offset = specification.offset
    offset = int(offset) % acceleration
    num_low = max(
        1, min(width, int(round(specification.center_fraction * width)))
    )
    mask = np.zeros(width, dtype=bool)
    mask[offset::acceleration] = True
    start, end = _center_bounds(width, num_low)
    mask[start:end] = True
    return mask


def infer_acquired_column_bounds(kspace):
    """Infer zero-padded phase-encoding bounds from complete k-space."""
    kspace = np.asarray(kspace)
    if kspace.ndim < 2:
        raise ValueError("K-space must have at least two dimensions.")
    reduce_axes = tuple(range(kspace.ndim - 1))
    present = np.any(kspace != 0, axis=reduce_axes)
    indices = np.flatnonzero(present)
    if not len(indices):
        return 0, kspace.shape[-1]
    return int(indices[0]), int(indices[-1]) + 1


def infer_equispaced_mask(mask, accelerations=(4, 8)):
    """Recover acceleration, offset and exact ACS width from a stored mask.

    The challenge masks are the union of an equispaced pattern and one centered
    ACS interval.  Exact reconstruction is required; silently guessing a mask
    would break the data-consistency model.
    """
    observed = np.asarray(mask).astype(bool).reshape(-1)
    width = observed.size
    candidates = []
    for acceleration in accelerations:
        for offset in range(int(acceleration)):
            periodic = np.zeros(width, dtype=bool)
            periodic[offset::int(acceleration)] = True
            for num_low in range(1, width + 1):
                expected = periodic.copy()
                start, end = _center_bounds(width, num_low)
                expected[start:end] = True
                if np.array_equal(expected, observed):
                    candidates.append(
                        MaskSpecification(
                            acceleration=int(acceleration),
                            offset=offset,
                            center_fraction=num_low / width,
                            num_low_frequencies=num_low,
                        )
                    )
    if not candidates:
        raise ValueError(
            "Stored mask is not an exact centered-ACS equispaced R4/R8 mask."
        )
    # Touching periodic samples can make the visually contiguous center one
    # line wider. The smallest exact ACS interval is the generative one.
    return min(
        candidates,
        key=lambda item: (
            item.num_low_frequencies,
            item.acceleration,
            item.offset,
        ),
    )


def resample_acceleration(rng, probability, high_probability, low=4, high=8):
    """Draw a volume-epoch acceleration, or ``None`` to keep the stored one.

    Because the stored k-space is complete and ``image_label`` is its RSS, the
    label does not depend on which mask is applied. Any volume can therefore be
    re-undersampled at the other acceleration and remain a valid training pair,
    which pools both acceleration groups into one acc8 (and acc4) source set.
    """
    if probability <= 0.0:
        return None
    if rng.random() >= probability:
        return None
    return int(high) if rng.random() < high_probability else int(low)


def retarget_specification(specification, acceleration, width, catalog=None):
    """Re-express ``specification`` at another acceleration for a given width.

    The challenge ships a slightly different ACS width per acceleration (the
    sample volumes carry 29/368 at R4 and 31/372 at R8), so a re-targeted mask
    should adopt the destination acceleration's own centre fraction. ``catalog``
    supplies it when the dataset has been scanned; otherwise the source volume's
    centre fraction is kept, which differs by at most a couple of lines.

    ``offset`` is zeroed because the caller always redraws it.
    """
    acceleration = int(acceleration)
    if acceleration == int(specification.acceleration):
        return specification
    reference = (catalog or {}).get(acceleration)
    center_fraction = (
        specification.center_fraction if reference is None
        else reference.center_fraction
    )
    num_low = max(1, min(int(width), int(round(center_fraction * int(width)))))
    return MaskSpecification(
        acceleration=acceleration,
        offset=0,
        center_fraction=center_fraction,
        num_low_frequencies=num_low,
    )


def fft2c(image):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c(kspace):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def center_crop_or_pad(array, output_shape):
    """Center crop and/or zero pad the first two dimensions."""
    output_h, output_w = map(int, output_shape)
    result = np.zeros((output_h, output_w) + array.shape[2:], dtype=array.dtype)
    input_h, input_w = array.shape[:2]
    copy_h, copy_w = min(input_h, output_h), min(input_w, output_w)
    input_y = (input_h - copy_h) // 2
    input_x = (input_w - copy_w) // 2
    output_y = (output_h - copy_h) // 2
    output_x = (output_w - copy_w) // 2
    result[output_y:output_y + copy_h, output_x:output_x + copy_w] = (
        array[input_y:input_y + copy_h, input_x:input_x + copy_w]
    )
    return result


def center_crop_max(array, maximum_shape):
    """Center crop dimensions that exceed a maximum, without padding."""
    maximum_h, maximum_w = map(int, maximum_shape)
    height, width = array.shape[:2]
    output_h, output_w = min(height, maximum_h), min(width, maximum_w)
    start_y = (height - output_h) // 2
    start_x = (width - output_w) // 2
    return np.ascontiguousarray(
        array[start_y:start_y + output_h, start_x:start_x + output_w]
    )


def boxes_to_masks(boxes, shape):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    masks = np.zeros(tuple(shape) + (len(boxes),), dtype=np.uint8)
    height, width = map(int, shape)
    for index, (x, y, box_width, box_height) in enumerate(boxes):
        x0 = max(0, min(width, int(math.floor(x))))
        y0 = max(0, min(height, int(math.floor(y))))
        x1 = max(0, min(width, int(math.ceil(x + box_width))))
        y1 = max(0, min(height, int(math.ceil(y + box_height))))
        if x1 > x0 and y1 > y0:
            masks[y0:y1, x0:x1, index] = 1
    return masks


def masks_to_boxes(masks, minimum_size=7):
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[..., None]
    boxes = []
    for index in range(masks.shape[-1]):
        ys, xs = np.nonzero(masks[..., index] > 0)
        if not len(xs):
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        if x1 - x0 < minimum_size or y1 - y0 < minimum_size:
            return None
        boxes.append([x0, y0, x1 - x0, y1 - y0])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _translate(array, translation_y, translation_x, mode):
    height, width = array.shape[:2]
    translation_y = int(np.clip(translation_y, -height + 1, height - 1))
    translation_x = int(np.clip(translation_x, -width + 1, width - 1))
    top_pad = max(-translation_y, 0)
    bottom_pad = max(translation_y, 0)
    left_pad = max(translation_x, 0)
    right_pad = max(-translation_x, 0)
    pad = ((top_pad, bottom_pad), (left_pad, right_pad))
    pad += ((0, 0),) * (array.ndim - 2)
    padded = np.pad(array, pad, mode=mode)
    top = bottom_pad if translation_y >= 0 else 0
    left = 0 if translation_x >= 0 else right_pad
    return padded[top:top + height, left:left + width]


def _centered_affine(shape, parameters):
    height, width = map(int, shape)
    scale_y = parameters.isotropic_scale * parameters.anisotropic_scale_y
    scale_x = parameters.isotropic_scale * parameters.anisotropic_scale_x
    base = AffineTransform(
        scale=(scale_x, scale_y),
        rotation=np.deg2rad(parameters.rotation_degrees),
        shear=np.deg2rad(parameters.shear_degrees),
    ).params
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    to_origin = np.array(
        [[1.0, 0.0, -center[0]], [0.0, 1.0, -center[1]], [0.0, 0.0, 1.0]]
    )
    from_origin = np.array(
        [[1.0, 0.0, center[0]], [0.0, 1.0, center[1]], [0.0, 0.0, 1.0]]
    )
    return from_origin @ base @ to_origin


def _affine_padding(shape, parameters):
    height, width = map(int, shape)
    matrix = _centered_affine(shape, parameters)
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1],
         [width - 1, height - 1, 1]],
        dtype=np.float64,
    ).T
    transformed = matrix @ corners
    span_x = transformed[0].max() - transformed[0].min() + 1
    span_y = transformed[1].max() - transformed[1].min() + 1
    return (
        max(0, int(math.ceil((span_y - height) / 2))),
        max(0, int(math.ceil((span_x - width) / 2))),
    )


def apply_parameters(array, parameters, *, is_label=False):
    """Apply one sampled geometry to an ``H x W [x channels]`` array."""
    result = np.asarray(array)
    if parameters.horizontal_flip:
        result = np.flip(result, axis=1)
    if parameters.vertical_flip:
        result = np.flip(result, axis=0)
    if parameters.rot90_k:
        result = np.rot90(result, parameters.rot90_k, axes=(0, 1))
    if parameters.translation_y or parameters.translation_x:
        result = _translate(
            result,
            parameters.translation_y,
            parameters.translation_x,
            mode="constant" if is_label else "reflect",
        )
    if parameters.has_affine:
        original_shape = result.shape[:2]
        pad_y, pad_x = _affine_padding(original_shape, parameters)
        pad = ((pad_y, pad_y), (pad_x, pad_x))
        pad += ((0, 0),) * (result.ndim - 2)
        result = np.pad(
            result, pad, mode="constant" if is_label else "reflect"
        )
        inverse_map = np.linalg.inv(_centered_affine(result.shape[:2], parameters))
        result = warp(
            result,
            inverse_map=inverse_map,
            output_shape=result.shape[:2],
            order=0 if is_label else 3,
            mode="constant" if is_label else "reflect",
            cval=0.0,
            clip=False,
            preserve_range=True,
        )
        result = center_crop_or_pad(result, original_shape)
    return np.ascontiguousarray(result)


def sample_parameters(rng, probability, shape):
    """Sample the eight independent fastMRI transforms from the paper table."""
    probability = float(probability)

    def apply(weight):
        return bool(rng.random() < probability * weight)

    horizontal_flip = apply(0.5)
    vertical_flip = apply(0.5)
    rot90_k = int(rng.integers(0, 4)) if apply(0.5) else 0
    height, width = map(int, shape)
    if rot90_k % 2:
        height, width = width, height

    if apply(1.0):
        translation_y = int(rng.uniform(-0.125, 0.125) * height)
        translation_x = int(rng.uniform(-0.08, 0.08) * width)
    else:
        translation_y = translation_x = 0

    rotation = rng.uniform(-180.0, 180.0) if apply(0.5) else 0.0
    isotropic = rng.uniform(0.75, 1.25) if apply(0.5) else 1.0
    if apply(0.5):
        anisotropic_y = rng.uniform(0.75, 1.25)
        anisotropic_x = rng.uniform(0.75, 1.25)
    else:
        anisotropic_y = anisotropic_x = 1.0
    shear = rng.uniform(-12.5, 12.5) if apply(1.0) else 0.0

    return MRAugmentParameters(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rot90_k=rot90_k,
        translation_y=translation_y,
        translation_x=translation_x,
        rotation_degrees=float(rotation),
        isotropic_scale=float(isotropic),
        anisotropic_scale_y=float(anisotropic_y),
        anisotropic_scale_x=float(anisotropic_x),
        shear_degrees=float(shear),
    )


def augment_multicoil_kspace(kspace, parameters, output_shape=(384, 384)):
    """Apply sampled geometry and return full k-space plus its RSS target."""
    maximum_shape = np.asarray(kspace).shape[-2:]
    coil_images = ifft2c(np.asarray(kspace, dtype=np.complex64))
    channels = np.concatenate(
        [coil_images.real, coil_images.imag], axis=0
    ).transpose(1, 2, 0)
    channels = apply_parameters(channels, parameters, is_label=False)
    channels = center_crop_max(channels, maximum_shape)
    num_coils = kspace.shape[0]
    channel_first = channels.transpose(2, 0, 1)
    augmented_images = (
        channel_first[:num_coils] + 1j * channel_first[num_coils:]
    ).astype(np.complex64)
    target = np.sqrt(np.sum(np.abs(augmented_images) ** 2, axis=0))
    target = center_crop_or_pad(target, output_shape).astype(np.float32)
    return fft2c(augmented_images).astype(np.complex64), target


def transform_boxes(
    boxes,
    parameters,
    source_shape,
    *,
    target_shape=(384, 384),
    minimum_size=7,
):
    """Apply image geometry to challenge boxes, or return ``None`` if unsafe."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return boxes
    target_masks = boxes_to_masks(boxes, target_shape)
    source_masks = center_crop_or_pad(target_masks, source_shape)
    transformed = apply_parameters(source_masks, parameters, is_label=True)
    transformed = center_crop_max(transformed, source_shape)
    transformed = center_crop_or_pad(transformed, target_shape)
    return masks_to_boxes(transformed, minimum_size=minimum_size)
