import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from utils.data.load_data import (
    BalancedAccelerationSampler,
    acceleration_from_filename,
)


def fake_dataset(acc4_count, acc8_count):
    examples = [
        (Path(f"knee_acc4_{i}.h5"), 0) for i in range(acc4_count)
    ] + [
        (Path(f"knee_acc8_{i}.h5"), 0) for i in range(acc8_count)
    ]
    return SimpleNamespace(kspace_examples=examples)


class BalancedAccelerationSamplerTest(unittest.TestCase):
    def acceleration_sequence(self, dataset, indices):
        return [
            acceleration_from_filename(dataset.kspace_examples[index][0].name)
            for index in indices
        ]

    def test_oversample_is_exact_balanced_alternating_and_keeps_majority(self):
        dataset = fake_dataset(3, 5)
        sampler = BalancedAccelerationSampler(dataset, mode="oversample", seed=17)
        indices = list(sampler)
        accelerations = self.acceleration_sequence(dataset, indices)

        self.assertEqual(Counter(accelerations), {4: 5, 8: 5})
        self.assertEqual(accelerations, [4, 8] * 5)
        majority_indices = set(range(3, 8))
        self.assertEqual(set(indices) & majority_indices, majority_indices)
        self.assertEqual(len(indices), 10)

    def test_epoch_is_reproducible_and_alternating_start_flips(self):
        dataset = fake_dataset(4, 6)
        first = BalancedAccelerationSampler(dataset, seed=430)
        second = BalancedAccelerationSampler(dataset, seed=430)
        first.set_epoch(40)
        second.set_epoch(40)
        self.assertEqual(list(first), list(second))

        first.set_epoch(41)
        indices = list(first)
        self.assertEqual(self.acceleration_sequence(dataset, indices), [8, 4] * 6)

    def test_equal_groups_visit_every_slice_once(self):
        dataset = fake_dataset(4, 4)
        sampler = BalancedAccelerationSampler(dataset, seed=3)
        indices = list(sampler)
        self.assertEqual(len(indices), 8)
        self.assertEqual(set(indices), set(range(8)))

    def test_undersample_drops_only_majority(self):
        dataset = fake_dataset(2, 5)
        sampler = BalancedAccelerationSampler(dataset, mode="undersample", seed=8)
        indices = list(sampler)
        self.assertEqual(
            Counter(self.acceleration_sequence(dataset, indices)),
            {4: 2, 8: 2},
        )
        self.assertTrue({0, 1}.issubset(indices))
        self.assertEqual(len([index for index in indices if index >= 2]), 2)

    def test_unknown_or_missing_group_fails(self):
        unknown = SimpleNamespace(
            kspace_examples=[(Path("knee_unknown_0.h5"), 0)]
        )
        with self.assertRaisesRegex(ValueError, "explicit _acc4_ or _acc8_"):
            BalancedAccelerationSampler(unknown)

        with self.assertRaisesRegex(ValueError, "requires both acc4 and acc8"):
            BalancedAccelerationSampler(fake_dataset(2, 0))


if __name__ == "__main__":
    unittest.main()
