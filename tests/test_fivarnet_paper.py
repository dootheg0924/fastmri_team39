import math
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.smoke_test_training import _select_stress_slice
from train import (
    FINAL_FIVARNET_PRESET,
    PAPER_FIVARNET_PRESET,
    apply_training_preset,
)
from utils.common.bbox_loss import BboxAwareSSIMLoss
from utils.data.load_data import PaddedRandomSampler
from utils.learning.train_part import (
    build_lr_scheduler,
    build_model,
    build_optimizer,
    build_training_loss,
    checkpoint_decision,
    configure_epoch_lr_schedule,
    load_checkpoint,
    paper_lr_multiplier,
    resolve_time_budget_epochs,
    save_model,
    training_limit_reached,
    train_epoch,
    validate,
)
from utils.model.fi_varnet import FeatureUnet2d


def paper_args():
    return apply_training_preset(
        Namespace(
            training_preset=PAPER_FIVARNET_PRESET,
            combine_train_val=False,
        )
    )


def test_gtx1080_preset_resolves_requested_values():
    args = paper_args()

    assert args.model_name == "fivarnet"
    assert args.cascade == 6
    assert args.image_cascades == 6
    assert args.chans == 32
    assert args.pools == 4
    assert args.sens_chans == 8
    assert args.sens_pools == 4
    assert args.attention_cascades == list(range(6))
    assert args.feature_processor == "paper-unet2d"
    assert args.kspace_mult_factor == 1e6
    assert args.no_grad_checkpoint is False
    assert args.acc_film is False
    assert args.split_attention_cascades == []

    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.optimizer == "adamw"
    assert args.lr == pytest.approx(3e-4)
    assert args.weight_decay == 0.0
    assert (args.adam_beta1, args.adam_beta2) == (0.9, 0.999)
    assert args.adam_eps == 1e-8
    assert args.adam_amsgrad is False
    assert args.loss_name == "bbox-aware-ssim"
    assert args.bbox_loss_weight == pytest.approx(0.5)
    assert args.checkpoint_metric == "paper-final"
    assert args.lr_scheduler == "fi-varnet-paper"
    assert args.max_steps == 210_000
    assert args.lr_warmup_steps == 7_500
    assert args.lr_cosine_start_step == 150_000
    assert args.lr_min_factor == 1e-8
    assert args.gradient_clip_val == 1.0
    assert args.seed == 42
    assert args.data_sampler_seed == 0
    assert args.deterministic is False
    assert args.float32_matmul_precision == "high"
    assert args.combine_train_val is False  # paper knee protocol

    loss = build_training_loss(args, torch.device("cpu"))
    assert isinstance(loss, BboxAwareSSIMLoss)
    assert loss.bbox_weight == pytest.approx(0.5)


def test_paper_preset_preserves_explicit_data_split_protocol():
    args = apply_training_preset(
        Namespace(
            training_preset=PAPER_FIVARNET_PRESET,
            combine_train_val=True,
            checkpoint_metric="paper-ssim",
        )
    )

    assert args.combine_train_val is True  # released brain leaderboard runner
    assert args.checkpoint_metric == "paper-ssim"


def test_final_preset_uses_epochs_and_latest_submission_checkpoint():
    args = apply_training_preset(
        Namespace(
            training_preset=FINAL_FIVARNET_PRESET,
            combine_train_val=False,
            checkpoint_metric=None,
            num_epochs=100,
        )
    )

    assert args.max_steps is None
    assert args.lr_scheduler == "fi-varnet-epochs"
    assert args.checkpoint_metric == "submission-latest"
    assert args.num_epochs == 100
    assert args.model_name == "fivarnet"
    assert args.cascade == args.image_cascades == 6
    assert args.loss_name == "bbox-aware-ssim"
    assert args.bbox_loss_weight == pytest.approx(0.5)


def test_epoch_lr_schedule_preserves_paper_phase_ratios():
    args = Namespace(
        lr_scheduler="fi-varnet-epochs",
        gradient_accumulation_steps=4,
        num_epochs=100,
    )
    resolved = configure_epoch_lr_schedule(args, loader_length=400)

    assert resolved["steps_per_epoch"] == 100
    assert resolved["total_steps"] == 10_000
    assert resolved["warmup_steps"] == round(10_000 * 7_500 / 210_000)
    assert resolved["cosine_start_step"] == round(
        10_000 * 150_000 / 210_000
    )
    assert args.lr_total_steps == 10_000


def test_time_budget_resolves_epoch_and_both_schedules():
    args = Namespace(
        lr_scheduler="fi-varnet-epochs",
        gradient_accumulation_steps=4,
        num_epochs=100,
        requested_num_epochs=100,
        training_time_budget_hours=10,
        training_time_reserve_fraction=0.15,
        mraugment=True,
    )
    resolved = resolve_time_budget_epochs(
        args,
        launch_start_epoch=0,
        measured_epoch_seconds=1800,
        loader_length=400,
    )

    assert resolved["affordable_epochs_this_launch"] == 17
    assert resolved["resolved_target_epoch"] == 17
    assert args.num_epochs == 17
    assert args.mraugment_total_epochs == 17
    assert args.lr_total_steps == 1700


def test_submission_latest_always_promotes_completed_epoch():
    args = Namespace(checkpoint_metric="submission-latest")
    best, promote = checkpoint_decision(
        args,
        val_loss=float("nan"),
        best_val_loss=float("inf"),
        global_step=123,
    )
    assert math.isnan(best)
    assert promote is True


def test_exp003_bbox_objective_uses_requested_half_weight(monkeypatch):
    loss = BboxAwareSSIMLoss(bbox_weight=0.5)
    output = torch.zeros(1, 8, 8)

    monkeypatch.setattr(
        loss,
        "_foreground_loss",
        lambda output, target, data_range: output.new_tensor(0.2),
    )
    monkeypatch.setattr(
        loss,
        "_bbox_loss_terms",
        lambda output, target, maximum, boxes: [
            output.new_tensor(0.4),
            output.new_tensor(0.6),
        ],
    )

    result = loss(
        output,
        torch.zeros_like(output),
        torch.ones(1),
        torch.tensor([[[0, 0, 8, 8]]]),
    )

    assert result.item() == pytest.approx(0.2 + 0.5 * ((0.4 + 0.6) / 2))


def test_smoke_test_selects_largest_attention_pressure_candidate(tmp_path):
    acc4_path = tmp_path / "sample_acc4_a.h5"
    acc8_path = tmp_path / "sample_acc8_b.h5"
    for path, coils, stride in ((acc4_path, 20, 4), (acc8_path, 8, 8)):
        mask = np.zeros(16, dtype=np.uint8)
        mask[::stride] = 1
        with h5py.File(path, "w") as hf:
            hf.create_dataset(
                "kspace",
                data=np.zeros((1, coils, 16, 16), dtype=np.complex64),
            )
            hf.create_dataset("mask", data=mask)

    dataset = Namespace(
        input_key="kspace",
        kspace_examples=[(acc4_path, 0), (acc8_path, 0)],
    )

    selected = _select_stress_slice(dataset)

    assert selected[2] == 1
    assert selected[3] == acc8_path.name
    assert selected[-1] == 8


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.0),
        (7_499, 7_499 / 7_500),
        (7_500, 1.0),
        (149_999, 1.0),
        (150_000, 1.0),
        (180_000, math.sqrt(0.5)),
        (210_000, 1e-8),
    ],
)
def test_paper_lr_multiplier_boundaries(step, expected):
    multiplier = paper_lr_multiplier(
        step,
        base_lr=3e-4,
        warmup_steps=7_500,
        cosine_start_step=150_000,
        max_steps=210_000,
        min_factor=1e-8,
    )
    assert multiplier == pytest.approx(expected)


def test_full_paper_model_structure_on_meta_device():
    args = paper_args()
    with torch.device("meta"):
        model = build_model(args)

    assert len(model.cascades) == 6
    assert len(model.image_cascades) == 6
    assert model.use_checkpoint is True
    assert all(block.attention_layer is not None for block in model.cascades)
    assert [
        index for index, block in enumerate(model.cascades) if block.use_image_conv
    ] == [0, 3]
    assert all(isinstance(block.feature_processor, FeatureUnet2d)
               for block in model.cascades)
    assert all(block.encoder is model.encoder for block in model.cascades)
    assert all(block.decoder is model.decoder for block in model.cascades)
    assert len({id(block.feature_processor) for block in model.cascades}) == 6

    # Filled after parity-checking this port against the author-contributed
    # fastMRI implementation. This is a zero-allocation meta-tensor count.
    assert sum(parameter.numel() for parameter in model.parameters()) == 93_805_980


class FourSampleDataset(Dataset):
    def __init__(self):
        self.inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.targets = torch.tensor([[0.5], [1.0], [1.5], [2.0]])

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (
            torch.ones(1),
            self.inputs[index],
            self.targets[index],
            torch.tensor(1.0),
            f"knee_acc4_{index}.h5",
            index,
            torch.empty(0, 4),
        )


class ToyReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, kspace, mask):
        return self.linear(kspace)


class ToyLoss(nn.Module):
    def forward(self, output, target, maximum):
        return torch.mean((output - target) ** 2)


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_four_microbatches_equal_one_global_batch_update():
    accumulated = ToyReconstructor()
    reference = ToyReconstructor()
    with torch.no_grad():
        accumulated.linear.weight.fill_(0.25)
        reference.load_state_dict(accumulated.state_dict())

    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    loader = DataLoader(FourSampleDataset(), batch_size=1, shuffle=False)
    args = Namespace(
        loss_name="ssim",
        gradient_accumulation_steps=4,
        gradient_clip_val=0.0,
        max_steps=1,
        report_interval=100,
        num_epochs=1,
    )

    _, _, completed_steps, samples_seen = train_epoch(
        args,
        epoch=0,
        model=accumulated,
        data_loader=loader,
        optimizer=accumulated_optimizer,
        scheduler=scheduler,
        loss_type=ToyLoss(),
        device=torch.device("cpu"),
        global_step=0,
    )

    batch_inputs = FourSampleDataset().inputs
    batch_targets = FourSampleDataset().targets
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = torch.mean(
        (reference(batch_inputs, torch.ones(4, 1)) - batch_targets) ** 2
    )
    reference_loss.backward()
    reference_optimizer.step()

    assert completed_steps == 1
    assert samples_seen == 4
    assert scheduler.steps == 1
    torch.testing.assert_close(
        accumulated.linear.weight.detach(),
        reference.linear.weight.detach(),
    )


def test_padded_sampler_is_deterministic_and_step_aligned():
    dataset = list(range(5))
    sampler = PaddedRandomSampler(dataset, multiple=4, seed=0)

    epoch_zero = list(iter(sampler))
    assert len(sampler) == 8
    assert len(epoch_zero) == 8
    assert set(range(5)).issubset(epoch_zero)
    assert epoch_zero == list(iter(sampler))

    sampler.set_epoch(1)
    assert list(iter(sampler)) != epoch_zero


class FixedValidationModel(nn.Module):
    def forward(self, kspace, mask):
        return kspace


class FixedChallengeMetric:
    def foreground_ssim_score(self, output, target, maximum):
        return [0.8] * output.shape[0]

    def bbox_ssim_scores(self, output, target, maximum, boxes):
        return [[0.6] for _ in range(output.shape[0])]


class FixedPaperLoss(nn.Module):
    def forward(self, output, target, maximum):
        return output.new_tensor(0.25)


def validation_batches():
    batches = []
    for acceleration in (4, 8):
        batches.append(
            (
                torch.ones(1, 1),
                torch.zeros(1, 8, 8),
                torch.zeros(1, 8, 8),
                torch.ones(1),
                [f"knee_acc{acceleration}_1.h5"],
                torch.tensor([0]),
                torch.empty(1, 0, 4),
            )
        )
    return batches


def test_paper_checkpoint_selection_keeps_challenge_metrics():
    args = Namespace(checkpoint_metric="paper-ssim")
    result, _, _, _ = validate(
        args,
        FixedValidationModel(),
        FixedChallengeMetric(),
        validation_batches(),
        torch.device("cpu"),
        paper_loss=FixedPaperLoss(),
    )

    assert result["paper_val_loss"] == pytest.approx(0.25)
    assert result["challenge_val_loss"] == pytest.approx(0.3)
    assert result["final_score"] == pytest.approx(0.7)
    assert result["val_loss"] == pytest.approx(0.25)


def test_knee_paper_checkpoint_selection_uses_only_final_step():
    args = Namespace(checkpoint_metric="paper-final", max_steps=210_000)

    best, promote = checkpoint_decision(
        args,
        val_loss=0.25,
        best_val_loss=float("inf"),
        global_step=200_000,
    )
    assert math.isinf(best)
    assert promote is False

    best, promote = checkpoint_decision(
        args,
        val_loss=0.30,
        best_val_loss=best,
        global_step=210_000,
    )
    assert best == pytest.approx(0.30)
    assert promote is True


def test_explicit_epoch_cap_finishes_step_based_training():
    args = Namespace(
        checkpoint_metric="paper-final",
        max_steps=210_000,
        max_training_epochs=100,
        num_epochs=100,
    )
    assert not training_limit_reached(args, 99, 120_000)
    assert training_limit_reached(args, 100, 120_000)
    assert training_limit_reached(args, 20, 210_000)

    best, promote = checkpoint_decision(
        args,
        val_loss=0.2,
        best_val_loss=float("inf"),
        global_step=120_000,
        completed_epochs=100,
    )
    assert best == pytest.approx(0.2)
    assert promote is True


def test_optimizer_scheduler_checkpoint_round_trip(tmp_path):
    args = Namespace(
        lr=3e-4,
        optimizer="adamw",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_amsgrad=False,
        lr_scheduler="fi-varnet-paper",
        max_steps=210_000,
        lr_warmup_steps=7_500,
        lr_cosine_start_step=150_000,
        lr_min_factor=1e-8,
        checkpoint_interval=0,
    )
    model = nn.Linear(1, 1)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()

    save_model(
        args,
        tmp_path,
        epoch=2,
        model=model,
        optimizer=optimizer,
        best_val_loss=0.2,
        is_new_best=True,
        history=[],
        scheduler=scheduler,
        global_step=3,
        samples_seen=12,
    )

    restored_model = nn.Linear(1, 1)
    restored_optimizer = build_optimizer(args, restored_model)
    restored_scheduler = build_lr_scheduler(args, restored_optimizer)
    (
        epoch,
        global_step,
        samples_seen,
        best_val_loss,
        history,
        warm_start,
    ) = load_checkpoint(
        tmp_path / "model.pt",
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        scheduler=restored_scheduler,
    )

    assert (epoch, global_step, samples_seen) == (2, 3, 12)
    assert best_val_loss == pytest.approx(0.2)
    assert history == []
    assert warm_start is None
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
