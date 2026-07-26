import unittest

import torch
from torch import nn

from scripts.analyze_acceleration_gradients import (
    GradientGroup,
    SampleRecord,
    collect_gradient_groups,
    gradient_pair_stats,
    plan_samples,
    repeat_rows,
    summarize_repeat_rows,
)


class FakeFeatureBlock(nn.Module):
    def __init__(self, shared_encoder, shared_decoder, with_attention, with_output):
        super().__init__()
        self.encoder = shared_encoder
        self.decoder = shared_decoder
        self.feature_processor = nn.Linear(2, 2, bias=False)
        self.attention_layer = nn.Linear(2, 2, bias=False) if with_attention else None
        self.dc_weight = nn.Parameter(torch.ones(1))
        if with_output:
            self.output_conv = nn.Linear(2, 2, bias=False)


class FakeFIVarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)
        self.cascades = nn.ModuleList(
            [
                FakeFeatureBlock(self.encoder, self.decoder, True, True),
                FakeFeatureBlock(self.encoder, self.decoder, False, False),
            ]
        )
        self.image_cascades = nn.ModuleList([nn.Linear(2, 2)])


class AccelerationGradientAnalysisTest(unittest.TestCase):
    def test_collect_groups_excludes_shared_and_image_parameters(self):
        model = FakeFIVarNet()
        groups = collect_gradient_groups(model)

        names = [
            parameter_name
            for group in groups
            for parameter_name, _ in group.parameters
        ]
        self.assertTrue(names)
        self.assertFalse(any(".encoder." in name for name in names))
        self.assertFalse(any(".decoder." in name for name in names))
        self.assertFalse(any("image_cascades" in name for name in names))
        self.assertEqual(
            {group.component for group in groups},
            {"feature_processor", "attention_layer", "dc_weight", "output_conv"},
        )

        selected_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        grouped_ids = {
            id(parameter)
            for group in groups
            for _, parameter in group.parameters
        }
        self.assertEqual(selected_ids, grouped_ids)

    def test_gradient_stats_concatenate_without_flattening(self):
        acc4 = {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([1.0]),
        }
        acc8 = {
            "a": torch.tensor([0.0, 1.0]),
            "b": torch.tensor([-1.0]),
        }
        stats = gradient_pair_stats(acc4, acc8, ["a", "b"])

        self.assertAlmostEqual(stats["cosine"], -0.5)
        self.assertAlmostEqual(stats["grad_norm_acc4"], 2 ** 0.5)
        self.assertAlmostEqual(stats["grad_norm_acc8"], 2 ** 0.5)
        self.assertAlmostEqual(stats["norm_ratio_acc8_acc4"], 1.0)

    def test_sampling_is_disjoint_balanced_and_volume_capped(self):
        candidates = {4: [], 8: []}
        dataset_index = 0
        for acceleration in (4, 8):
            for volume_index in range(4):
                for slice_index in range(4):
                    candidates[acceleration].append(
                        SampleRecord(
                            dataset_index=dataset_index,
                            acceleration=acceleration,
                            filename=f"case{volume_index}_acc{acceleration}_x.h5",
                            slice_index=slice_index,
                            normalized_position=(slice_index + 0.5) / 4,
                            position_bin=slice_index,
                        )
                    )
                    dataset_index += 1

        plans = plan_samples(
            candidates,
            repeats=2,
            samples_per_acceleration=4,
            max_slices_per_volume=1,
            seed=430,
        )

        for repeat_plan in plans:
            self.assertEqual(len(repeat_plan[4]), 4)
            self.assertEqual(len(repeat_plan[8]), 4)
            for acceleration in (4, 8):
                filenames = [record.filename for record in repeat_plan[acceleration]]
                self.assertEqual(len(filenames), len(set(filenames)))

        for acceleration in (4, 8):
            selected_indices = [
                record.dataset_index
                for repeat_plan in plans
                for record in repeat_plan[acceleration]
            ]
            self.assertEqual(len(selected_indices), len(set(selected_indices)))

    def test_sampling_retries_a_sparse_but_feasible_assignment(self):
        candidates = {4: [], 8: []}
        dataset_index = 0
        for acceleration in (4, 8):
            for filename, slice_count in (("A", 2), ("B", 1), ("C", 1)):
                for slice_index in range(slice_count):
                    candidates[acceleration].append(
                        SampleRecord(
                            dataset_index=dataset_index,
                            acceleration=acceleration,
                            filename=f"{filename}_acc{acceleration}_x.h5",
                            slice_index=slice_index,
                            normalized_position=(slice_index + 0.5) / slice_count,
                            position_bin=min(
                                3,
                                int((slice_index + 0.5) / slice_count * 4),
                            ),
                        )
                    )
                    dataset_index += 1

        plans = plan_samples(
            candidates,
            repeats=2,
            samples_per_acceleration=2,
            max_slices_per_volume=1,
            seed=7,
        )

        for acceleration in (4, 8):
            selected = [
                record
                for repeat_plan in plans
                for record in repeat_plan[acceleration]
            ]
            self.assertEqual(len(selected), 4)
            self.assertEqual(
                len({record.dataset_index for record in selected}),
                4,
            )
            self.assertEqual(
                sum(record.filename.startswith("A_") for record in selected),
                2,
            )

    def test_repeat_summary_counts_negative_signs(self):
        parameter = nn.Parameter(torch.zeros(2))
        group = GradientGroup(
            cascade=0,
            component="feature_processor",
            parameters=[("cascades.0.feature_processor.weight", parameter)],
        )
        rows = []
        for repeat_number, acc8 in enumerate(
            (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])),
            start=1,
        ):
            rows.extend(
                repeat_rows(
                    repeat_number=repeat_number,
                    groups=[group],
                    acc4_gradients={
                        "cascades.0.feature_processor.weight": torch.tensor(
                            [1.0, 0.0]
                        )
                    },
                    acc8_gradients={
                        "cascades.0.feature_processor.weight": acc8
                    },
                    acc4_coherence={
                        "cascade_0.feature_processor": 0.8,
                        "cascade_0.all_owned": 0.8,
                    },
                    acc8_coherence={
                        "cascade_0.feature_processor": 0.7,
                        "cascade_0.all_owned": 0.7,
                    },
                )
            )

        summaries = summarize_repeat_rows(rows)
        cascade_summary = next(row for row in summaries if row["scope"] == "cascade")
        self.assertEqual(cascade_summary["total_repeats"], 2)
        self.assertEqual(cascade_summary["negative_repeats"], 1)
        self.assertAlmostEqual(cascade_summary["cosine_mean"], 0.0)
        self.assertAlmostEqual(
            cascade_summary["directional_coherence_acc4_mean"], 0.8
        )


if __name__ == "__main__":
    unittest.main()
