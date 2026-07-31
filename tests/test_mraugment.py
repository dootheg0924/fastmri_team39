import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.data.mraugment import (
    MRAugmentParameters,
    MaskSpecification,
    augment_multicoil_kspace,
    augmentation_probability,
    deterministic_rng,
    infer_acquired_column_bounds,
    infer_equispaced_mask,
    make_equispaced_mask,
    sample_parameters,
    transform_boxes,
)
from utils.data.transforms import DataTransform


def test_paper_exponential_schedule_is_normalized():
    assert augmentation_probability(0, 300) == 0.0
    assert augmentation_probability(300, 300) == pytest.approx(0.55)
    expected = 0.55 * (1 - np.exp(-2.5)) / (1 - np.exp(-5))
    assert augmentation_probability(150, 300) == pytest.approx(expected)


def test_schedule_delay_and_alternatives():
    assert augmentation_probability(
        2, 10, schedule="ramp", maximum=0.6, delay=3
    ) == 0.0
    assert augmentation_probability(
        6.5, 10, schedule="ramp", maximum=0.6, delay=3
    ) == pytest.approx(0.3)
    assert augmentation_probability(
        3, 10, schedule="constant", maximum=0.6, delay=3
    ) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "width,acceleration,offset,num_low",
    [(368, 4, 0, 29), (372, 8, 2, 30)],
)
def test_mask_specification_is_recovered_exactly(
    width, acceleration, offset, num_low
):
    source = MaskSpecification(
        acceleration=acceleration,
        offset=offset,
        center_fraction=num_low / width,
        num_low_frequencies=num_low,
    )
    mask = make_equispaced_mask(width, source)
    recovered = infer_equispaced_mask(mask)
    assert recovered.acceleration == acceleration
    assert recovered.offset == offset
    assert recovered.num_low_frequencies == num_low
    assert np.array_equal(make_equispaced_mask(width, recovered), mask)


def test_rng_and_parameter_sampling_are_worker_independent():
    first = sample_parameters(
        deterministic_rng(42, 7, "volume.h5", 3), 0.55, (640, 368)
    )
    second = sample_parameters(
        deterministic_rng(42, 7, "volume.h5", 3), 0.55, (640, 368)
    )
    other_slice = sample_parameters(
        deterministic_rng(42, 7, "volume.h5", 4), 0.55, (640, 368)
    )
    assert first == second
    assert first != other_slice


def test_zero_padded_acquisition_bounds_are_inferred():
    kspace = np.zeros((2, 8, 12), dtype=np.complex64)
    kspace[..., 2:9] = 1
    assert infer_acquired_column_bounds(kspace) == (2, 9)


def test_boxes_follow_horizontal_flip():
    boxes = np.array([[2, 4, 5, 8]], dtype=np.float32)
    transformed = transform_boxes(
        boxes,
        MRAugmentParameters(horizontal_flip=True),
        (32, 32),
        target_shape=(32, 32),
        minimum_size=1,
    )
    np.testing.assert_array_equal(transformed, [[25, 4, 5, 8]])


def test_bbox_geometry_is_rejected_when_lesion_leaves_crop():
    boxes = np.array([[0, 0, 7, 7]], dtype=np.float32)
    transformed = transform_boxes(
        boxes,
        MRAugmentParameters(translation_y=-30, translation_x=30),
        (32, 32),
        target_shape=(32, 32),
        minimum_size=7,
    )
    assert transformed is None


def test_multicoil_identity_round_trip_and_rot90_resolution_cap():
    rng = np.random.default_rng(9)
    kspace = (
        rng.normal(size=(2, 40, 32))
        + 1j * rng.normal(size=(2, 40, 32))
    ).astype(np.complex64)
    recovered, target = augment_multicoil_kspace(
        kspace, MRAugmentParameters(), output_shape=(32, 32)
    )
    np.testing.assert_allclose(recovered, kspace, rtol=2e-6, atol=2e-6)
    assert target.shape == (32, 32)
    assert target.dtype == np.float32

    rotated, rotated_target = augment_multicoil_kspace(
        kspace, MRAugmentParameters(rot90_k=1), output_shape=(32, 32)
    )
    assert rotated.shape == (2, 32, 32)
    assert rotated_target.shape == (32, 32)


def _transform_args(enabled):
    return SimpleNamespace(
        mraugment=enabled,
        mraugment_total_epochs=10,
        mraugment_schedule="exp",
        mraugment_strength=0.55,
        mraugment_exp_decay=5.0,
        mraugment_delay_epochs=0,
        mraugment_seed=42,
        mraugment_min_bbox_size=7,
        num_epochs=10,
        seed=42,
    )


def test_validation_transform_never_augments_or_remasks():
    rng = np.random.default_rng(3)
    kspace = (
        rng.normal(size=(2, 32, 32))
        + 1j * rng.normal(size=(2, 32, 32))
    ).astype(np.complex64)
    specification = MaskSpecification(4, 1, 4 / 32, 4)
    mask = make_equispaced_mask(32, specification)
    target = rng.random((32, 32), dtype=np.float32)
    attrs = {"max": 1.0, "annotations": json.dumps({})}
    transform = DataTransform(
        False, "max", args=_transform_args(True), is_train=False
    )
    result_mask, result_kspace, result_target, *_ = transform(
        mask, kspace, target, attrs, "knee_acc4_1.h5", 0
    )
    np.testing.assert_array_equal(
        result_mask.numpy().reshape(-1), mask.astype(np.uint8)
    )
    expected = torch.from_numpy(kspace * mask.reshape(1, 1, -1))
    torch.testing.assert_close(
        torch.view_as_complex(result_kspace), expected
    )
    torch.testing.assert_close(result_target, torch.from_numpy(target))


def test_training_transform_is_deterministic_and_remasks_after_augmentation():
    rng = np.random.default_rng(4)
    kspace = (
        rng.normal(size=(2, 40, 32))
        + 1j * rng.normal(size=(2, 40, 32))
    ).astype(np.complex64)
    specification = MaskSpecification(4, 0, 4 / 32, 4)
    mask = make_equispaced_mask(32, specification)
    target = rng.random((32, 32), dtype=np.float32)
    attrs = {"max": 1.0, "annotations": json.dumps({})}
    transform = DataTransform(
        False, "max", args=_transform_args(True), is_train=True
    )
    transform.set_epoch(4)
    first = transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 2)
    second = transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 2)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    assert first[0].shape[-2] == first[1].shape[-2]
    expanded_mask = first[0].bool().expand_as(first[1])
    assert torch.count_nonzero(first[1][~expanded_mask]) == 0
