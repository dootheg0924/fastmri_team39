import unittest

from scripts.analyze_validation_reconstructions import (
    box_area_summary,
    build_volume_rows,
    cluster_bootstrap_score,
    pearson_correlation,
    slice_position_summary,
    summarize_metric,
)


class ValidationReconstructionAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.slice_rows = [
            {
                "filename": "knee_acc4_1.h5", "slice": 0, "slice_position": 0.1,
                "acceleration": "acc4", "ssim_full": 0.80, "ssim_bbox": 0.75,
                "bbox_count": 1,
            },
            {
                "filename": "knee_acc4_1.h5", "slice": 1, "slice_position": 0.9,
                "acceleration": "acc4", "ssim_full": 0.82, "ssim_bbox": "",
                "bbox_count": 0,
            },
            {
                "filename": "knee_acc8_1.h5", "slice": 0, "slice_position": 0.5,
                "acceleration": "acc8", "ssim_full": 0.70, "ssim_bbox": 0.65,
                "bbox_count": 1,
            },
        ]
        self.box_rows = [
            {
                "filename": "knee_acc4_1.h5", "slice": 0, "acceleration": "acc4",
                "label": "lesion_a", "area": 100, "ssim_bbox": 0.75,
            },
            {
                "filename": "knee_acc8_1.h5", "slice": 0, "acceleration": "acc8",
                "label": "lesion_b", "area": 400, "ssim_bbox": 0.65,
            },
        ]

    def test_summary_uses_equal_acceleration_weighting(self):
        summary = summarize_metric(self.slice_rows, "ssim_full")
        expected = (((0.80 + 0.82) / 2) + 0.70) / 2
        self.assertAlmostEqual(summary["equal_acceleration_mean"], expected)
        self.assertEqual(summary["by_acceleration"]["acc4"]["count"], 2)

    def test_volume_rows_keep_exact_box_units(self):
        rows = build_volume_rows(self.slice_rows, self.box_rows)
        acc4 = next(row for row in rows if row["acceleration"] == "acc4")
        self.assertEqual(acc4["slice_count"], 2)
        self.assertEqual(acc4["box_count"], 1)
        self.assertAlmostEqual(acc4["ssim_full"], 0.81)
        self.assertAlmostEqual(acc4["ssim_bbox"], 0.75)

    def test_bootstrap_is_deterministic_and_volume_clustered(self):
        first = cluster_bootstrap_score(self.slice_rows, "ssim_full", 50, 430)
        second = cluster_bootstrap_score(self.slice_rows, "ssim_full", 50, 430)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["acc4"]["bootstrap_mean"], 0.81)
        self.assertAlmostEqual(first["acc8"]["bootstrap_mean"], 0.70)
        self.assertAlmostEqual(first["overall"]["bootstrap_mean"], 0.755)

    def test_position_area_and_correlation_helpers(self):
        positions = slice_position_summary(self.slice_rows)
        edge = next(
            row for row in positions
            if row["position_bucket"] == "0-20%" and row["acceleration"] == "all"
        )
        self.assertEqual(edge["slice_count"], 1)
        self.assertAlmostEqual(edge["ssim_full"], 0.80)
        self.assertEqual(len(box_area_summary(self.box_rows)), 4)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()
