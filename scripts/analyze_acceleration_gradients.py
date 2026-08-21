"""Measure acc4/acc8 gradient alignment in each FI-VarNet feature cascade.

This is a read-only checkpoint diagnostic: it loads model weights, performs
forward/backward passes on balanced samples, and never creates an optimizer or
updates the model.

For each repeat and acceleration, the script first estimates the mean task
gradient

    g_R = (1 / N) * sum_i grad(loss_i),  R in {4, 8},

then reports cos(g_4, g_8), both gradient norms, and their ratio.  Feature
cascade rows contain only parameters owned by that cascade:

* feature_processor
* attention_layer
* dc_weight
* output_conv (when present)

The shared feature encoder/decoder are deliberately excluded.  They are
registered inside every feature block as shared module references, so blindly
using ``cascade.parameters()`` would double-count the same tensors and make the
per-cascade comparison misleading.

Keep the model in train mode: FI-VarNet activates gradient checkpointing only
in train mode, which is needed for the 8 GB experiment configuration.  The
current architecture has dropout=0 and InstanceNorm without running statistics,
so this does not introduce train-mode randomness or mutate normalization state.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "utils" / "model"
for import_path in (REPO_ROOT, MODEL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.common.bbox_loss import BboxAwareSSIMLoss  # noqa: E402
from utils.data.load_data import SliceData  # noqa: E402
from utils.data.transforms import DataTransform  # noqa: E402
from utils.learning.train_part import build_model  # noqa: E402


ACCELERATION_RE = re.compile(r"_acc(4|8)_", re.IGNORECASE)
ACCELERATIONS = (4, 8)
COMPONENT_ORDER = {
    "feature_processor": 0,
    "attention_layer": 1,
    "dc_weight": 2,
    "output_conv": 3,
    "other_owned": 4,
}
REPEAT_FIELDS = [
    "repeat",
    "scope",
    "cascade",
    "component",
    "parameter_count",
    "cosine",
    "grad_norm_acc4",
    "grad_norm_acc8",
    "rms_grad_acc4",
    "rms_grad_acc8",
    "norm_ratio_acc8_acc4",
    "directional_coherence_acc4",
    "directional_coherence_acc8",
]
SUMMARY_FIELDS = [
    "scope",
    "cascade",
    "component",
    "parameter_count",
    "total_repeats",
    "valid_repeats",
    "cosine_mean",
    "cosine_std",
    "cosine_min",
    "cosine_max",
    "negative_repeats",
    "grad_norm_acc4_mean",
    "grad_norm_acc8_mean",
    "rms_grad_acc4_mean",
    "rms_grad_acc8_mean",
    "norm_ratio_acc8_acc4_mean",
    "directional_coherence_acc4_mean",
    "directional_coherence_acc8_mean",
]


@dataclass(frozen=True)
class SampleRecord:
    dataset_index: int
    acceleration: int
    filename: str
    slice_index: int
    normalized_position: float
    position_bin: int


@dataclass
class GradientGroup:
    cascade: int
    component: str
    parameters: List[Tuple[str, torch.nn.Parameter]]

    @property
    def key(self) -> str:
        return f"cascade_{self.cascade}.{self.component}"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.parameters)


def acceleration_from_name(filename: str) -> int:
    match = ACCELERATION_RE.search(filename)
    if not match:
        raise ValueError(
            f"Could not determine acceleration from filename: {filename!r}. "
            "Expected a name containing '_acc4_' or '_acc8_'."
        )
    return int(match.group(1))


def decode_json_attribute(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def valid_annotated_slices(
    image_dir: Path,
    target_key: str,
    win_size: int,
) -> Dict[str, set]:
    """Return slice indices having at least one leaderboard-valid box."""
    result = {}
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        with h5py.File(image_path, "r") as hf:
            if target_key not in hf:
                continue
            height, width = hf[target_key].shape[-2:]
            annotations = decode_json_attribute(hf.attrs.get("annotations"))

        valid = set()
        for raw_slice, boxes in annotations.items():
            try:
                slice_index = int(raw_slice)
            except (TypeError, ValueError):
                continue
            if not isinstance(boxes, list):
                continue
            for box in boxes:
                try:
                    x = int(float(box["x"]))
                    y = int(float(box["y"]))
                    box_width = int(float(box["width"]))
                    box_height = int(float(box["height"]))
                except (KeyError, TypeError, ValueError):
                    continue
                x0, y0 = max(0, x), max(0, y)
                x1 = min(width, x + box_width)
                y1 = min(height, y + box_height)
                if (x1 - x0) >= win_size and (y1 - y0) >= win_size:
                    valid.add(slice_index)
                    break
        result[image_path.name] = valid
    return result


def build_candidates(
    dataset: SliceData,
    annotated_slices: Optional[Dict[str, set]] = None,
) -> Dict[int, List[SampleRecord]]:
    candidates = {acceleration: [] for acceleration in ACCELERATIONS}
    slices_per_volume = defaultdict(int)
    for kspace_path, _ in dataset.kspace_examples:
        slices_per_volume[kspace_path.name] += 1
    for dataset_index, (kspace_path, slice_index) in enumerate(dataset.kspace_examples):
        filename = kspace_path.name
        acceleration = acceleration_from_name(filename)
        if acceleration not in candidates:
            continue
        if annotated_slices is not None:
            if int(slice_index) not in annotated_slices.get(filename, set()):
                continue
        normalized_position = (
            (int(slice_index) + 0.5) / slices_per_volume[filename]
        )
        candidates[acceleration].append(
            SampleRecord(
                dataset_index=dataset_index,
                acceleration=acceleration,
                filename=filename,
                slice_index=int(slice_index),
                normalized_position=normalized_position,
                position_bin=min(3, int(normalized_position * 4)),
            )
        )
    return candidates


def sample_one_repeat(
    candidates: Sequence[SampleRecord],
    count: int,
    max_slices_per_volume: int,
    rng: random.Random,
    used_indices: set,
) -> List[SampleRecord]:
    """Uniformly cycle over volumes and select unused slices."""
    by_volume = defaultdict(list)
    for record in candidates:
        if record.dataset_index not in used_indices:
            by_volume[record.filename].append(record)

    volume_names = list(by_volume)
    rng.shuffle(volume_names)
    for records in by_volume.values():
        rng.shuffle(records)

    selected = []
    selected_per_volume = defaultdict(int)
    selected_per_position_bin = defaultdict(int)
    while len(selected) < count:
        progress = False
        rng.shuffle(volume_names)
        for filename in volume_names:
            if len(selected) >= count:
                break
            if selected_per_volume[filename] >= max_slices_per_volume:
                continue
            records = [
                record
                for record in by_volume[filename]
                if record.dataset_index not in used_indices
            ]
            if not records:
                continue
            least_used_bin_count = min(
                selected_per_position_bin[record.position_bin] for record in records
            )
            eligible = [
                record
                for record in records
                if selected_per_position_bin[record.position_bin] == least_used_bin_count
            ]
            record = rng.choice(eligible)
            selected.append(record)
            selected_per_volume[filename] += 1
            selected_per_position_bin[record.position_bin] += 1
            used_indices.add(record.dataset_index)
            progress = True
        if not progress:
            break
    return selected


def plan_samples(
    candidates: Dict[int, List[SampleRecord]],
    repeats: int,
    samples_per_acceleration: int,
    max_slices_per_volume: int,
    seed: int,
) -> List[Dict[int, List[SampleRecord]]]:
    """Create disjoint, acceleration-balanced, volume-balanced repeat sets."""
    plans_by_acceleration = {}
    max_attempts = 256

    for acceleration in ACCELERATIONS:
        acceleration_candidates = candidates[acceleration]
        unique_volumes = len(
            {record.filename for record in acceleration_candidates}
        )
        if len(acceleration_candidates) < repeats * samples_per_acceleration:
            raise RuntimeError(
                f"acc{acceleration} has only {len(acceleration_candidates)} candidate "
                f"slices, but {repeats * samples_per_acceleration} distinct slices "
                "are required."
            )
        per_repeat_capacity = sum(
            min(max_slices_per_volume, len(volume_records))
            for volume_records in _records_by_volume(
                acceleration_candidates
            ).values()
        )
        if per_repeat_capacity < samples_per_acceleration:
            raise RuntimeError(
                f"acc{acceleration} can supply at most {per_repeat_capacity} slices "
                "per repeat under the current volume cap, but "
                f"{samples_per_acceleration} were requested."
            )

        acceleration_plans = None
        for attempt in range(max_attempts):
            rng = random.Random(seed + acceleration * 100_003 + attempt * 1_000_003)
            used_indices = set()
            attempt_plans = []
            for _ in range(repeats):
                selected = sample_one_repeat(
                    candidates=acceleration_candidates,
                    count=samples_per_acceleration,
                    max_slices_per_volume=max_slices_per_volume,
                    rng=rng,
                    used_indices=used_indices,
                )
                if len(selected) != samples_per_acceleration:
                    break
                attempt_plans.append(selected)
            if len(attempt_plans) == repeats:
                acceleration_plans = attempt_plans
                break

        if acceleration_plans is None:
            raise RuntimeError(
                f"Could not construct {repeats} disjoint acc{acceleration} repeats "
                f"after {max_attempts} balanced assignment attempts. Candidates: "
                f"{len(acceleration_candidates)} slices from {unique_volumes} volumes. "
                "Lower --repeats or --samples-per-acc, or increase "
                "--max-slices-per-volume."
            )
        plans_by_acceleration[acceleration] = acceleration_plans

    return [
        {
            acceleration: plans_by_acceleration[acceleration][repeat_index]
            for acceleration in ACCELERATIONS
        }
        for repeat_index in range(repeats)
    ]


def _records_by_volume(
    records: Sequence[SampleRecord],
) -> Dict[str, List[SampleRecord]]:
    by_volume = defaultdict(list)
    for record in records:
        by_volume[record.filename].append(record)
    return by_volume


def classify_owned_parameter(local_name: str) -> Optional[str]:
    if local_name == "dc_weight":
        return "dc_weight"
    for component in ("feature_processor", "attention_layer", "output_conv"):
        if local_name.startswith(component + "."):
            return component
    if local_name.startswith(("encoder.", "decoder.")):
        return None
    return "other_owned"


def collect_gradient_groups(model) -> List[GradientGroup]:
    if not hasattr(model, "cascades"):
        raise TypeError(
            "The loaded model has no feature cascades; "
            "an FI-VarNet checkpoint is required."
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    groups = []
    seen_parameter_ids = set()
    for cascade_index, cascade in enumerate(model.cascades):
        parameters_by_component = defaultdict(list)
        for local_name, parameter in cascade.named_parameters():
            component = classify_owned_parameter(local_name)
            if component is None:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                raise RuntimeError(
                    f"Cascade-owned parameter unexpectedly appears more than once: "
                    f"cascades.{cascade_index}.{local_name}"
                )
            seen_parameter_ids.add(parameter_id)
            parameter.requires_grad_(True)
            global_name = f"cascades.{cascade_index}.{local_name}"
            parameters_by_component[component].append((global_name, parameter))

        for component in sorted(
            parameters_by_component,
            key=lambda value: COMPONENT_ORDER.get(value, 999),
        ):
            groups.append(
                GradientGroup(
                    cascade=cascade_index,
                    component=component,
                    parameters=parameters_by_component[component],
                )
            )

    if not groups:
        raise RuntimeError("No feature-cascade-owned parameters were found.")
    return groups


class DiagnosticLoss:
    def __init__(self, audit: str, bbox_weight: float, device: torch.device):
        self.audit = audit
        effective_weight = 0.0 if audit == "foreground" else bbox_weight
        self.loss = BboxAwareSSIMLoss(bbox_weight=effective_weight).to(device=device)

    @property
    def requires_annotations(self) -> bool:
        return self.audit in {"bbox", "training-annotated"}

    @property
    def description(self) -> str:
        if self.audit == "foreground":
            return "foreground SSIM loss only (bbox_weight=0)"
        if self.audit == "bbox":
            return "bounding-box SSIM loss only, on annotated slices"
        if self.audit == "training":
            return (
                "checkpoint training objective on balanced random slices "
                f"(foreground + {self.loss.bbox_weight:g} * bbox when annotated)"
            )
        return (
            "checkpoint training objective restricted to annotated slices "
            f"(foreground + {self.loss.bbox_weight:g} * bbox)"
        )

    def __call__(self, output, target, maximum, boxes):
        if self.audit != "bbox":
            return self.loss(output, target, maximum, boxes)

        maximum = maximum.to(dtype=output.dtype)
        terms = self.loss._bbox_loss_terms(output, target, maximum, boxes)
        if not terms:
            raise RuntimeError(
                "A bbox-only sample had no valid box loss terms. "
                "The candidate filter and loss disagree."
            )
        return torch.stack(terms).mean()


def initialize_gradient_sums(
    groups: Sequence[GradientGroup],
) -> Dict[str, torch.Tensor]:
    sums = {}
    for group in groups:
        for name, parameter in group.parameters:
            if name in sums:
                raise RuntimeError(f"Duplicate gradient parameter name: {name}")
            sums[name] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    return sums


def accumulate_selected_gradients(
    groups: Sequence[GradientGroup],
    sums: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    component_squared_norms = {}
    cascade_squared_norms = defaultdict(lambda: None)
    all_finite = torch.ones((), dtype=torch.bool, device=next(iter(sums.values())).device)
    for group in groups:
        squared_norm = None
        for name, parameter in group.parameters:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"Selected parameter received no gradient: {name}")
            all_finite.logical_and_(torch.isfinite(gradient).all())
            sums[name].add_(gradient.detach())
            term = torch.sum(gradient.detach() * gradient.detach())
            squared_norm = term if squared_norm is None else squared_norm + term
        component_squared_norms[group.key] = squared_norm
        cascade_key = f"cascade_{group.cascade}.all_owned"
        current = cascade_squared_norms[cascade_key]
        cascade_squared_norms[cascade_key] = (
            squared_norm if current is None else current + squared_norm
        )

    if not bool(all_finite):
        for group in groups:
            for name, parameter in group.parameters:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(
                        f"Selected parameter has a non-finite gradient: {name}"
                    )
        raise RuntimeError("A selected parameter has a non-finite gradient.")

    return {
        **{
            key: torch.sqrt(torch.clamp(value, min=0.0))
            for key, value in component_squared_norms.items()
        },
        **{
            key: torch.sqrt(torch.clamp(value, min=0.0))
            for key, value in cascade_squared_norms.items()
        },
    }


def prepare_sample(dataset: SliceData, record: SampleRecord, device: torch.device):
    mask, kspace, target, maximum, filename, slice_index, boxes = dataset[
        record.dataset_index
    ]
    if filename != record.filename or int(slice_index) != record.slice_index:
        raise RuntimeError(
            "Dataset metadata changed between sampling and loading: "
            f"planned={record.filename}:{record.slice_index}, "
            f"loaded={filename}:{slice_index}"
        )
    return (
        mask.unsqueeze(0).to(device=device, non_blocking=True),
        kspace.unsqueeze(0).to(device=device, non_blocking=True),
        target.unsqueeze(0).to(device=device, non_blocking=True),
        torch.as_tensor([maximum], device=device),
        boxes.unsqueeze(0),
    )


def mean_gradients_for_acceleration(
    model,
    dataset: SliceData,
    records: Sequence[SampleRecord],
    groups: Sequence[GradientGroup],
    loss_fn: DiagnosticLoss,
    device: torch.device,
    repeat_number: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[float]], List[dict]]:
    sums = initialize_gradient_sums(groups)
    sum_individual_norms = {
        group.key: torch.zeros((), device=device) for group in groups
    }
    for cascade in {group.cascade for group in groups}:
        sum_individual_norms[f"cascade_{cascade}.all_owned"] = torch.zeros(
            (), device=device
        )
    sample_rows = []

    for sample_number, record in enumerate(records, start=1):
        mask, kspace, target, maximum, boxes = prepare_sample(dataset, record, device)
        inferred_acceleration = int(model._infer_acceleration(mask))
        if inferred_acceleration != record.acceleration:
            raise RuntimeError(
                f"Acceleration mismatch for {record.filename} slice {record.slice_index}: "
                f"filename says acc{record.acceleration}, mask inference says "
                f"acc{inferred_acceleration}."
            )

        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = model(kspace, mask)
        loss = loss_fn(output, target, maximum, boxes)
        if not torch.isfinite(output).all():
            raise RuntimeError(
                f"Non-finite output for {record.filename} slice {record.slice_index}"
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for {record.filename} slice {record.slice_index}"
            )
        loss.backward()
        sample_gradient_norms = accumulate_selected_gradients(groups, sums)
        for key, norm in sample_gradient_norms.items():
            sum_individual_norms[key].add_(norm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        sample_rows.append(
            {
                "repeat": repeat_number,
                "acceleration": record.acceleration,
                "filename": record.filename,
                "slice": record.slice_index,
                "normalized_position": record.normalized_position,
                "position_bin": record.position_bin,
                "bbox_count": int(boxes.shape[1]),
                "loss": float(loss.detach().cpu()),
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"[repeat {repeat_number}] acc{record.acceleration} "
            f"{sample_number:02d}/{len(records):02d} "
            f"{record.filename} slice={record.slice_index} "
            f"boxes={boxes.shape[1]} loss={loss.item():.6f} time={elapsed:.2f}s",
            flush=True,
        )

        del mask, kspace, target, maximum, boxes, output, loss

    count = len(records)
    model.zero_grad(set_to_none=True)
    mean_gradients = {
        name: (gradient_sum / count).detach().cpu()
        for name, gradient_sum in sums.items()
    }
    directional_coherence = {}
    groups_by_cascade = defaultdict(list)
    for group in groups:
        groups_by_cascade[group.cascade].append(group)
        mean_norm = gradient_vector_norm(
            mean_gradients,
            (name for name, _ in group.parameters),
        )
        denominator = float(sum_individual_norms[group.key].detach().cpu())
        directional_coherence[group.key] = (
            min(1.0, max(0.0, mean_norm * count / denominator))
            if denominator > 0.0 else None
        )
    for cascade, cascade_groups in groups_by_cascade.items():
        key = f"cascade_{cascade}.all_owned"
        mean_norm = gradient_vector_norm(
            mean_gradients,
            (
                name
                for group in cascade_groups
                for name, _ in group.parameters
            ),
        )
        denominator = float(sum_individual_norms[key].detach().cpu())
        directional_coherence[key] = (
            min(1.0, max(0.0, mean_norm * count / denominator))
            if denominator > 0.0 else None
        )
    del sums
    del sum_individual_norms
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean_gradients, directional_coherence, sample_rows


def gradient_vector_norm(
    gradients: Dict[str, torch.Tensor],
    parameter_names: Iterable[str],
) -> float:
    squared_norm = 0.0
    for name in parameter_names:
        gradient = gradients[name].to(dtype=torch.float64)
        squared_norm += float(torch.sum(gradient * gradient))
    return math.sqrt(max(squared_norm, 0.0))


def gradient_pair_stats(
    acc4_gradients: Dict[str, torch.Tensor],
    acc8_gradients: Dict[str, torch.Tensor],
    parameter_names: Iterable[str],
) -> dict:
    dot = 0.0
    norm4_squared = 0.0
    norm8_squared = 0.0
    for name in parameter_names:
        gradient4 = acc4_gradients[name].to(dtype=torch.float64)
        gradient8 = acc8_gradients[name].to(dtype=torch.float64)
        dot += float(torch.sum(gradient4 * gradient8))
        norm4_squared += float(torch.sum(gradient4 * gradient4))
        norm8_squared += float(torch.sum(gradient8 * gradient8))

    norm4 = math.sqrt(max(norm4_squared, 0.0))
    norm8 = math.sqrt(max(norm8_squared, 0.0))
    denominator = norm4 * norm8
    cosine = (
        min(1.0, max(-1.0, dot / denominator))
        if denominator > 0.0 else None
    )
    ratio = norm8 / norm4 if norm4 > 0.0 else None
    return {
        "cosine": cosine,
        "grad_norm_acc4": norm4,
        "grad_norm_acc8": norm8,
        "norm_ratio_acc8_acc4": ratio,
    }


def repeat_rows(
    repeat_number: int,
    groups: Sequence[GradientGroup],
    acc4_gradients: Dict[str, torch.Tensor],
    acc8_gradients: Dict[str, torch.Tensor],
    acc4_coherence: Dict[str, Optional[float]],
    acc8_coherence: Dict[str, Optional[float]],
) -> List[dict]:
    rows = []
    groups_by_cascade = defaultdict(list)
    for group in groups:
        groups_by_cascade[group.cascade].append(group)
        stats = gradient_pair_stats(
            acc4_gradients,
            acc8_gradients,
            (name for name, _ in group.parameters),
        )
        parameter_count = group.parameter_count
        rows.append(
            {
                "repeat": repeat_number,
                "scope": "component",
                "cascade": group.cascade,
                "component": group.component,
                "parameter_count": parameter_count,
                **stats,
                "rms_grad_acc4": stats["grad_norm_acc4"] / math.sqrt(parameter_count),
                "rms_grad_acc8": stats["grad_norm_acc8"] / math.sqrt(parameter_count),
                "directional_coherence_acc4": acc4_coherence[group.key],
                "directional_coherence_acc8": acc8_coherence[group.key],
            }
        )

    for cascade, cascade_groups in sorted(groups_by_cascade.items()):
        parameter_names = [
            name
            for group in cascade_groups
            for name, _ in group.parameters
        ]
        stats = gradient_pair_stats(
            acc4_gradients,
            acc8_gradients,
            parameter_names,
        )
        parameter_count = sum(group.parameter_count for group in cascade_groups)
        coherence_key = f"cascade_{cascade}.all_owned"
        rows.append(
            {
                "repeat": repeat_number,
                "scope": "cascade",
                "cascade": cascade,
                "component": "all_owned",
                "parameter_count": parameter_count,
                **stats,
                "rms_grad_acc4": stats["grad_norm_acc4"] / math.sqrt(parameter_count),
                "rms_grad_acc8": stats["grad_norm_acc8"] / math.sqrt(parameter_count),
                "directional_coherence_acc4": acc4_coherence[coherence_key],
                "directional_coherence_acc8": acc8_coherence[coherence_key],
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["cascade"],
            0 if row["scope"] == "cascade" else 1,
            COMPONENT_ORDER.get(row["component"], 999),
        ),
    )


def optional_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def summarize_repeat_rows(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["cascade"], row["component"])].append(row)

    summaries = []
    for (scope, cascade, component), group_rows in grouped.items():
        cosine_values = [
            row["cosine"]
            for row in group_rows
            if row["cosine"] is not None and math.isfinite(row["cosine"])
        ]
        summaries.append(
            {
                "scope": scope,
                "cascade": cascade,
                "component": component,
                "parameter_count": group_rows[0]["parameter_count"],
                "total_repeats": len(group_rows),
                "valid_repeats": len(cosine_values),
                "cosine_mean": optional_mean(cosine_values),
                "cosine_std": (
                    statistics.stdev(cosine_values)
                    if len(cosine_values) >= 2 else None
                ),
                "cosine_min": min(cosine_values) if cosine_values else None,
                "cosine_max": max(cosine_values) if cosine_values else None,
                "negative_repeats": sum(value < 0 for value in cosine_values),
                "grad_norm_acc4_mean": optional_mean(
                    row["grad_norm_acc4"] for row in group_rows
                ),
                "grad_norm_acc8_mean": optional_mean(
                    row["grad_norm_acc8"] for row in group_rows
                ),
                "rms_grad_acc4_mean": optional_mean(
                    row["rms_grad_acc4"] for row in group_rows
                ),
                "rms_grad_acc8_mean": optional_mean(
                    row["rms_grad_acc8"] for row in group_rows
                ),
                "norm_ratio_acc8_acc4_mean": optional_mean(
                    row["norm_ratio_acc8_acc4"] for row in group_rows
                ),
                "directional_coherence_acc4_mean": optional_mean(
                    row["directional_coherence_acc4"] for row in group_rows
                ),
                "directional_coherence_acc8_mean": optional_mean(
                    row["directional_coherence_acc8"] for row in group_rows
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["cascade"],
            0 if row["scope"] == "cascade" else 1,
            COMPONENT_ORDER.get(row["component"], 999),
        ),
    )


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def render_number(value: Optional[float], digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    metadata: dict,
    summaries: Sequence[dict],
) -> None:
    lines = [
        f"# Acceleration Gradient Diagnostic: {metadata['experiment']}",
        "",
        f"- Checkpoint: `{metadata['checkpoint']}` (epoch {metadata['checkpoint_epoch']})",
        f"- Data: `{metadata['split']}`",
        f"- Audit: `{metadata['audit']}` -- {metadata['loss_description']}",
        (
            f"- Sampling: {metadata['repeats']} repeats x "
            f"{metadata['samples_per_acceleration_per_repeat']} slices per acceleration; "
            f"at most {metadata['max_slices_per_volume']} slices per volume"
        ),
        "- Gradients: mean per acceleration first, cosine second",
        "- Shared encoder/decoder and all image-space cascades: excluded",
        "",
        "## Cascade summary",
        "",
        "| Cascade | Parameters | Cosine mean +/- std | Range | Negative / valid (total) | "
        "Norm ratio (g8/g4) | Coherence acc4 / acc8 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    cascade_rows = [row for row in summaries if row["scope"] == "cascade"]
    for row in cascade_rows:
        mean = render_number(row["cosine_mean"], 4)
        std = render_number(row["cosine_std"], 4)
        minimum = render_number(row["cosine_min"], 4)
        maximum = render_number(row["cosine_max"], 4)
        lines.append(
            f"| F{row['cascade']} | {row['parameter_count']:,} | {mean} +/- {std} | "
            f"[{minimum}, {maximum}] | {row['negative_repeats']}/"
            f"{row['valid_repeats']} ({row['total_repeats']}) | "
            f"{render_number(row['norm_ratio_acc8_acc4_mean'], 3)} | "
            f"{render_number(row['directional_coherence_acc4_mean'], 3)} / "
            f"{render_number(row['directional_coherence_acc8_mean'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Component summary",
            "",
            "| Cascade | Component | Parameters | Cosine mean +/- std | "
            "Negative / valid (total) | Norm ratio (g8/g4) | "
            "Coherence acc4 / acc8 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    component_rows = [row for row in summaries if row["scope"] == "component"]
    for row in component_rows:
        lines.append(
            f"| F{row['cascade']} | {row['component']} | "
            f"{row['parameter_count']:,} | "
            f"{render_number(row['cosine_mean'], 4)} +/- "
            f"{render_number(row['cosine_std'], 4)} | "
            f"{row['negative_repeats']}/{row['valid_repeats']} "
            f"({row['total_repeats']}) | "
            f"{render_number(row['norm_ratio_acc8_acc4_mean'], 3)} | "
            f"{render_number(row['directional_coherence_acc4_mean'], 3)} / "
            f"{render_number(row['directional_coherence_acc8_mean'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "- A stable negative cosine means acc4 and acc8 ask that parameter group "
            "to move in opposing directions on these samples.",
            "- A positive cosine means the directions agree; a value near zero can "
            "still be noisy, so use the repeat range and a second audit before branching.",
            "- A large or small norm ratio signals task imbalance even when cosine is positive.",
            "- Directional coherence is `||sum_i g_i|| / sum_i ||g_i||`. Values near "
            "zero mean within-acceleration gradients cancel, making the mean-gradient "
            "cosine a weak basis for a branch decision.",
            "- `dc_weight` is one scalar, so its cosine is necessarily near +1 or -1; "
            "interpret it with its repeat stability and gradient norms.",
            "- Do not choose 3/3 versus 4/2 from one negative component. Prefer a boundary "
            "supported by whole-cascade conflict in multiple repeats and confirmed by the "
            "foreground and lesion-focused audits.",
            "",
        ]
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary_path, path)


def resolve_checkpoint(exp_dir: Path, checkpoint_argument: Path) -> Path:
    if checkpoint_argument.is_absolute():
        return checkpoint_argument
    if len(checkpoint_argument.parts) == 1:
        return exp_dir / "checkpoints" / checkpoint_argument
    return exp_dir / checkpoint_argument


def serialize_args_subset(checkpoint_args) -> dict:
    keys = (
        "model_name",
        "cascade",
        "image_cascades",
        "chans",
        "sens_chans",
        "pools",
        "sens_pools",
        "attention_cascades",
        "kspace_mult_factor",
        "bbox_loss_weight",
        "input_key",
        "target_key",
        "max_key",
    )
    return {
        key: getattr(checkpoint_args, key)
        for key in keys
        if hasattr(checkpoint_args, key)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure acc4/acc8 mean-gradient cosine by FI-VarNet feature cascade. "
            "The checkpoint is never modified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--result-root", type=Path, default=Path("/root/result"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model.pt"),
        help=(
            "Absolute checkpoint path, a filename under <experiment>/checkpoints, "
            "or a path relative to the experiment directory"
        ),
    )
    parser.add_argument(
        "--expected-epoch",
        type=int,
        default=None,
        help="Fail instead of silently analyzing the wrong checkpoint epoch",
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--audit",
        choices=["foreground", "bbox", "training", "training-annotated"],
        default="foreground",
        help=(
            "foreground: primary reconstruction audit; bbox: bbox-only lesion audit; "
            "training: exact training loss on random slices; training-annotated: exact "
            "training loss on annotated slices"
        ),
    )
    parser.add_argument(
        "--samples-per-acc",
        "--samples-per-acc-per-repeat",
        dest="samples_per_acceleration",
        type=int,
        default=8,
        help="Number of distinct slices for each acceleration in each repeat",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-slices-per-volume",
        type=int,
        default=1,
        help="Per-repeat cap to prevent one volume from dominating",
    )
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment>/analysis",
    )
    args = parser.parse_args()
    for name in ("samples_per_acceleration", "repeats", "max_slices_per_volume"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "[ERROR] CUDA is required for this FI-VarNet backward diagnostic. "
            "CPU execution would be impractically slow."
        )
    device = torch.device(f"cuda:{args.gpu_num}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_dir = args.result_root / args.exp_name
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_epoch = int(checkpoint["epoch"])
    if args.expected_epoch is not None and checkpoint_epoch != args.expected_epoch:
        raise RuntimeError(
            f"Expected epoch {args.expected_epoch}, but {checkpoint_path} contains "
            f"epoch {checkpoint_epoch}."
        )
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None:
        raise KeyError("Checkpoint has no stored 'args'; the FI-VarNet architecture is ambiguous.")
    if getattr(checkpoint_args, "model_name", None) != "fivarnet":
        raise TypeError(
            f"Expected a fivarnet checkpoint, got "
            f"{getattr(checkpoint_args, 'model_name', None)!r}."
        )

    data_path = args.data_root / args.split
    image_dir = data_path / "image"
    kspace_dir = data_path / "kspace"
    if not image_dir.is_dir() or not kspace_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image/ and kspace/ under data split: {data_path}"
        )

    input_key = getattr(checkpoint_args, "input_key", "kspace")
    target_key = getattr(checkpoint_args, "target_key", "image_label")
    max_key = getattr(checkpoint_args, "max_key", "max")
    dataset = SliceData(
        root=data_path,
        transform=DataTransform(False, max_key),
        input_key=input_key,
        target_key=target_key,
    )

    bbox_weight = float(getattr(checkpoint_args, "bbox_loss_weight", 1.0))
    loss_fn = DiagnosticLoss(args.audit, bbox_weight, device)
    annotations = None
    if loss_fn.requires_annotations:
        annotations = valid_annotated_slices(
            image_dir=image_dir,
            target_key=target_key,
            win_size=loss_fn.loss.win_size,
        )
    candidates = build_candidates(dataset, annotations)
    plans = plan_samples(
        candidates=candidates,
        repeats=args.repeats,
        samples_per_acceleration=args.samples_per_acceleration,
        max_slices_per_volume=args.max_slices_per_volume,
        seed=args.seed,
    )

    model = build_model(checkpoint_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device)
    model.train()
    groups = collect_gradient_groups(model)
    if not getattr(model, "use_checkpoint", False):
        print(
            "[WARNING] This checkpoint has gradient checkpointing disabled; "
            "the diagnostic may exceed 8 GB VRAM.",
            flush=True,
        )

    selected_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    group_parameter_count = sum(group.parameter_count for group in groups)
    if selected_parameter_count != group_parameter_count:
        raise RuntimeError(
            "Selected parameter accounting mismatch: "
            f"requires_grad={selected_parameter_count}, groups={group_parameter_count}"
        )

    output_dir = args.output_dir or exp_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_slug = args.audit.replace("-", "_")
    run_slug = (
        f"epoch{checkpoint_epoch:04d}_{args.split}_{audit_slug}"
        f"_s{args.seed}_r{args.repeats}_n{args.samples_per_acceleration}"
        f"_v{args.max_slices_per_volume}"
    )

    print("============================================================", flush=True)
    print(f"Experiment       : {args.exp_name}", flush=True)
    print(f"Checkpoint       : {checkpoint_path}", flush=True)
    print(f"Checkpoint epoch : {checkpoint_epoch}", flush=True)
    print(f"Split / audit    : {args.split} / {args.audit}", flush=True)
    print(f"Loss             : {loss_fn.description}", flush=True)
    print(
        f"Sampling         : {args.repeats} repeats x "
        f"{args.samples_per_acceleration} slices x acc4/acc8",
        flush=True,
    )
    print(f"Gradient params  : {selected_parameter_count:,}", flush=True)
    print(f"GPU              : {torch.cuda.get_device_name(device)}", flush=True)
    print("============================================================", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    all_repeat_rows = []
    all_sample_rows = []

    for repeat_index, repeat_plan in enumerate(plans, start=1):
        mean_gradients = {}
        directional_coherence = {}
        for acceleration in ACCELERATIONS:
            gradients, coherence, sample_rows = mean_gradients_for_acceleration(
                model=model,
                dataset=dataset,
                records=repeat_plan[acceleration],
                groups=groups,
                loss_fn=loss_fn,
                device=device,
                repeat_number=repeat_index,
            )
            mean_gradients[acceleration] = gradients
            directional_coherence[acceleration] = coherence
            all_sample_rows.extend(sample_rows)

        current_rows = repeat_rows(
            repeat_number=repeat_index,
            groups=groups,
            acc4_gradients=mean_gradients[4],
            acc8_gradients=mean_gradients[8],
            acc4_coherence=directional_coherence[4],
            acc8_coherence=directional_coherence[8],
        )
        all_repeat_rows.extend(current_rows)
        cascade_values = [
            f"F{row['cascade']}={render_number(row['cosine'], 4)}"
            for row in current_rows
            if row["scope"] == "cascade"
        ]
        print(
            f"[repeat {repeat_index}] cascade cosines: "
            + ", ".join(cascade_values),
            flush=True,
        )
        del mean_gradients
        del directional_coherence
        gc.collect()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    summaries = summarize_repeat_rows(all_repeat_rows)

    repeat_csv = output_dir / f"acceleration_gradient_{run_slug}_repeats.csv"
    summary_csv = output_dir / f"acceleration_gradient_{run_slug}_summary.csv"
    samples_csv = output_dir / f"acceleration_gradient_{run_slug}_samples.csv"
    json_path = output_dir / f"acceleration_gradient_{run_slug}.json"
    markdown_path = output_dir / f"acceleration_gradient_{run_slug}.md"

    checkpoint_best_val_loss = checkpoint.get("best_val_loss")
    if torch.is_tensor(checkpoint_best_val_loss):
        checkpoint_best_val_loss = checkpoint_best_val_loss.item()
    if checkpoint_best_val_loss is not None:
        checkpoint_best_val_loss = float(checkpoint_best_val_loss)
        if not math.isfinite(checkpoint_best_val_loss):
            checkpoint_best_val_loss = None

    metadata = {
        "experiment": args.exp_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_val_loss": checkpoint_best_val_loss,
        "checkpoint_model_args": serialize_args_subset(checkpoint_args),
        "split": args.split,
        "audit": args.audit,
        "loss_description": loss_fn.description,
        "bbox_weight_from_checkpoint": bbox_weight,
        "repeats": args.repeats,
        "samples_per_acceleration_per_repeat": args.samples_per_acceleration,
        "max_slices_per_volume": args.max_slices_per_volume,
        "seed": args.seed,
        "candidate_slices": {
            f"acc{acceleration}": len(candidates[acceleration])
            for acceleration in ACCELERATIONS
        },
        "candidate_volumes": {
            f"acc{acceleration}": len(
                {record.filename for record in candidates[acceleration]}
            )
            for acceleration in ACCELERATIONS
        },
        "normalized_slice_position_bins": 4,
        "selected_parameter_count": selected_parameter_count,
        "shared_encoder_decoder_excluded": True,
        "image_cascades_excluded": True,
        "elapsed_sec": elapsed,
        "peak_allocated_gpu_memory_mib": peak_memory_mib,
    }
    payload = {
        "metadata": metadata,
        "samples": all_sample_rows,
        "repeat_results": all_repeat_rows,
        "summary": summaries,
    }
    json_text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    write_csv(repeat_csv, all_repeat_rows, REPEAT_FIELDS)
    write_csv(summary_csv, summaries, SUMMARY_FIELDS)
    write_csv(
        samples_csv,
        all_sample_rows,
        [
            "repeat",
            "acceleration",
            "filename",
            "slice",
            "normalized_position",
            "position_bin",
            "bbox_count",
            "loss",
            "elapsed_sec",
        ],
    )
    json_temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary_path.write_text(json_text, encoding="utf-8")
    os.replace(json_temporary_path, json_path)
    write_markdown(markdown_path, metadata, summaries)

    print("============================================================", flush=True)
    for row in summaries:
        if row["scope"] == "cascade":
            print(
                f"F{row['cascade']}: cosine="
                f"{render_number(row['cosine_mean'], 4)} +/- "
                f"{render_number(row['cosine_std'], 4)}, "
                f"negative={row['negative_repeats']}/{row['valid_repeats']}, "
                f"||g8||/||g4||="
                f"{render_number(row['norm_ratio_acc8_acc4_mean'], 3)}",
                flush=True,
            )
    print(f"Elapsed          : {elapsed / 60:.1f} min", flush=True)
    print(f"Peak GPU memory  : {peak_memory_mib:.1f} MiB", flush=True)
    print(f"Report           : {markdown_path}", flush=True)
    print(f"Machine-readable : {json_path}", flush=True)
    print("Checkpoint was not modified.", flush=True)


if __name__ == "__main__":
    main()
