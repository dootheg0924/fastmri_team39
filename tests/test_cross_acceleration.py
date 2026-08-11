"""Checks for cross-acceleration re-masking (B-1).

The stored k-space is complete and ``image_label`` is its RSS, so the label does
not depend on which mask is applied. Re-undersampling an R4 volume at R8 is
therefore a valid acc8 training pair, which pools both acceleration groups into
one source set for each acceleration.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.data.mraugment import (
    MaskSpecification,
    infer_equispaced_mask,
    make_equispaced_mask,
    resample_acceleration,
    retarget_specification,
)
from utils.data.transforms import DataTransform


def _args(cross=0.0, p8=0.5, mraugment=True):
    return SimpleNamespace(
        mraugment=mraugment,
        mraugment_total_epochs=10,
        mraugment_schedule="exp",
        mraugment_strength=0.55,
        mraugment_exp_decay=5.0,
        mraugment_delay_epochs=0,
        mraugment_seed=42,
        mraugment_min_bbox_size=7,
        cross_acceleration=cross,
        cross_acceleration_p8=p8,
        num_epochs=10,
        seed=42,
    )


def _volume(width=64, coils=2, height=40, seed=5):
    """Synthetic complete k-space: every column carries energy, so the acquired
    bounds cover the full width and the generated mask is exactly recoverable."""
    rng = np.random.default_rng(seed)
    kspace = (
        rng.normal(size=(coils, height, width))
        + 1j * rng.normal(size=(coils, height, width))
    ).astype(np.complex64)
    specification = MaskSpecification(4, 0, 8 / width, 8)
    mask = make_equispaced_mask(width, specification)
    target = rng.random((width, width), dtype=np.float32)
    attrs = {"max": 1.0, "annotations": json.dumps({})}
    return kspace, mask, target, attrs


# --- pure helpers ---------------------------------------------------------

def test_disabled_probability_keeps_the_stored_acceleration():
    rng = np.random.default_rng(0)
    assert resample_acceleration(rng, 0.0, 0.5) is None


def test_drawn_acceleration_follows_the_requested_split():
    rng = np.random.default_rng(1)
    drawn = [resample_acceleration(rng, 1.0, 0.5) for _ in range(4000)]
    assert set(drawn) == {4, 8}
    assert np.mean(np.array(drawn) == 8) == pytest.approx(0.5, abs=0.03)

    rng = np.random.default_rng(2)
    assert {resample_acceleration(rng, 1.0, 1.0) for _ in range(200)} == {8}
    rng = np.random.default_rng(3)
    assert {resample_acceleration(rng, 1.0, 0.0) for _ in range(200)} == {4}


def test_partial_probability_keeps_some_volumes_on_their_stored_mask():
    rng = np.random.default_rng(4)
    drawn = [resample_acceleration(rng, 0.25, 0.5) for _ in range(4000)]
    assert np.mean([item is None for item in drawn]) == pytest.approx(0.75, abs=0.03)


def test_retarget_is_identity_at_the_same_acceleration():
    specification = MaskSpecification(8, 3, 31 / 372, 31)
    assert retarget_specification(specification, 8, 372) is specification


def test_retarget_adopts_the_catalogue_acs_width():
    source = MaskSpecification(4, 0, 29 / 368, 29)          # an acc4 volume
    catalog = {8: MaskSpecification(8, 0, 31 / 372, 31)}    # acc8 geometry
    retargeted = retarget_specification(source, 8, 372, catalog=catalog)
    assert retargeted.acceleration == 8
    assert retargeted.center_fraction == pytest.approx(31 / 372)
    assert retargeted.num_low_frequencies == 31
    # The generated mask really carries that ACS and an R8 outer stride.
    mask = make_equispaced_mask(372, retargeted, offset=2)
    assert infer_equispaced_mask(mask).acceleration == 8
    assert infer_equispaced_mask(mask).num_low_frequencies == 31


def test_retarget_without_a_catalogue_keeps_the_source_acs_fraction():
    source = MaskSpecification(4, 0, 29 / 368, 29)
    retargeted = retarget_specification(source, 8, 368)
    assert retargeted.acceleration == 8
    assert retargeted.center_fraction == pytest.approx(29 / 368)


# --- transform integration ------------------------------------------------

def test_disabled_by_default_reproduces_the_stored_pipeline():
    kspace, mask, target, attrs = _volume()
    reference = DataTransform(False, "max", args=_args(cross=0.0), is_train=True)
    enabled = DataTransform(False, "max", args=_args(cross=1.0, p8=0.0), is_train=True)
    for transform in (reference, enabled):
        transform.set_epoch(3)

    got = [
        transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 0)
        for transform in (reference, enabled)
    ]
    # p8=0 always draws R4, which is this volume's stored acceleration, so the
    # geometry, the offset draw and the mask must all be untouched. This is the
    # property that makes enabling B-1 safe on a resumed run.
    for a, b in zip(got[0][:3], got[1][:3]):
        torch.testing.assert_close(a, b)


def test_an_r4_volume_is_presented_at_r8():
    kspace, mask, target, attrs = _volume()
    transform = DataTransform(False, "max", args=_args(cross=1.0, p8=1.0), is_train=True)
    transform.set_epoch(1)
    result_mask, *_ = transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 0)
    generated = result_mask.numpy().reshape(-1).astype(bool)
    assert infer_equispaced_mask(generated).acceleration == 8
    # Genuinely more accelerated. Compared as a density because MRAugment's
    # 90-degree rotation can swap the k-space axes, changing the mask length.
    assert generated.mean() < mask.mean()


def test_the_catalogue_overrides_the_acs_of_a_retargeted_volume():
    kspace, mask, target, attrs = _volume(width=64)
    transform = DataTransform(False, "max", args=_args(cross=1.0, p8=1.0), is_train=True)
    transform.set_mask_catalog({8: MaskSpecification(8, 0, 12 / 64, 12)})
    transform.set_epoch(1)
    result_mask, *_ = transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 0)
    recovered = infer_equispaced_mask(result_mask.numpy().reshape(-1).astype(bool))
    assert recovered.acceleration == 8
    assert recovered.num_low_frequencies == 12


def test_every_slice_of_a_volume_shares_one_acceleration_per_epoch():
    """The draw excludes the slice index, mirroring the offset draw: one
    acquisition per volume and epoch, not a different one per slice.

    Only the acceleration is compared, not the mask array: MRAugment geometry is
    sampled per slice and a 90-degree rotation swaps the k-space axes, so the
    mask is legitimately regenerated at a different width for some slices.
    """
    kspace, mask, target, attrs = _volume()
    for epoch in (5, 7):
        transform = DataTransform(False, "max", args=_args(cross=1.0), is_train=True)
        transform.set_epoch(epoch)
        drawn = {
            infer_equispaced_mask(
                transform(mask, kspace, target, attrs, "knee_acc4_1.h5", index)[0]
                .numpy()
                .reshape(-1)
                .astype(bool)
            ).acceleration
            for index in range(8)
        }
        assert len(drawn) == 1


def test_the_draw_varies_across_epochs_and_volumes():
    kspace, mask, target, attrs = _volume()
    transform = DataTransform(False, "max", args=_args(cross=1.0), is_train=True)

    def acceleration(fname, epoch):
        transform.set_epoch(epoch)
        applied = transform(mask, kspace, target, attrs, fname, 0)[0]
        return infer_equispaced_mask(
            applied.numpy().reshape(-1).astype(bool)
        ).acceleration

    over_epochs = {acceleration("knee_acc4_1.h5", epoch) for epoch in range(24)}
    over_volumes = {acceleration(f"knee_acc4_{i}.h5", 0) for i in range(24)}
    assert over_epochs == {4, 8}
    assert over_volumes == {4, 8}


def test_the_draw_is_reproducible():
    kspace, mask, target, attrs = _volume()
    first, second = (
        DataTransform(False, "max", args=_args(cross=1.0), is_train=True)
        for _ in range(2)
    )
    for transform in (first, second):
        transform.set_epoch(9)
    a = first(mask, kspace, target, attrs, "knee_acc4_1.h5", 2)
    b = second(mask, kspace, target, attrs, "knee_acc4_1.h5", 2)
    for left, right in zip(a[:3], b[:3]):
        torch.testing.assert_close(left, right)


def test_validation_is_never_retargeted():
    kspace, mask, target, attrs = _volume()
    transform = DataTransform(False, "max", args=_args(cross=1.0, p8=1.0), is_train=False)
    transform.set_epoch(1)
    result_mask, *_ = transform(mask, kspace, target, attrs, "knee_acc4_1.h5", 0)
    np.testing.assert_array_equal(
        result_mask.numpy().reshape(-1), mask.astype(np.uint8)
    )


def test_works_without_mraugment():
    """B-1 only needs the mask regenerated, not the geometry augmented."""
    kspace, mask, target, attrs = _volume()
    transform = DataTransform(
        False, "max", args=_args(cross=1.0, p8=1.0, mraugment=False), is_train=True
    )
    transform.set_epoch(1)
    result_mask, _, result_target, *_ = transform(
        mask, kspace, target, attrs, "knee_acc4_1.h5", 0
    )
    assert infer_equispaced_mask(
        result_mask.numpy().reshape(-1).astype(bool)
    ).acceleration == 8
    # No MRAugment means no geometry, so the label must be bit-identical.
    torch.testing.assert_close(result_target, torch.from_numpy(target))


def test_zero_padded_columns_stay_unsampled_after_retargeting():
    kspace, mask, target, attrs = _volume(width=64)
    kspace[..., :6] = 0
    kspace[..., -5:] = 0
    transform = DataTransform(False, "max", args=_args(cross=1.0, p8=1.0), is_train=True)
    transform.set_epoch(1)
    result_mask, result_kspace, *_ = transform(
        mask, kspace, target, attrs, "knee_acc4_1.h5", 0
    )
    generated = result_mask.numpy().reshape(-1).astype(bool)
    assert not generated[:6].any()
    assert not generated[-5:].any()
    assert torch.count_nonzero(result_kspace[..., :6, :]) == 0
