import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_training import (
    build_summary,
    plot_epoch_times,
    plot_gpu_metrics,
    plot_losses,
    read_epoch_history,
    read_gpu_metrics,
    read_iteration_losses,
    write_summary,
)


class AnalysisToolsTest(unittest.TestCase):
    def test_parses_training_and_gpu_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "training_history.csv"
            with history_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch", "train_loss", "val_loss", "train_time_sec",
                        "val_time_sec", "learning_rate", "is_best",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "epoch": 0, "train_loss": 0.2, "val_loss": 0.18,
                            "train_time_sec": 60, "val_time_sec": 10,
                            "learning_rate": 3e-4, "is_best": 1,
                        },
                        {
                            "epoch": 1, "train_loss": 0.15, "val_loss": 0.16,
                            "train_time_sec": 62, "val_time_sec": 11,
                            "learning_rate": 3e-4, "is_best": 1,
                        },
                    ]
                )

            log_path = root / "train.log"
            log_path.write_text(
                "Epoch = [  0/ 80] Iter = [   0/ 100] Loss = 0.25 Time = 1.0s\n"
                "Epoch = [  0/ 80] Iter = [  50/ 100] Loss = 0.20 Time = 1.0s\n"
                "Epoch = [  1/ 80] Iter = [   0/ 100] Loss = 0.17 Time = 1.0s\n",
                encoding="utf-8",
            )

            gpu_path = root / "gpu.csv"
            gpu_path.write_text(
                "timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,temperature_c,power_draw_w\n"
                "2026/07/13 12:00:00, 0, NVIDIA A100, 90, 10000, 40960, 60, 200\n"
                "2026/07/13 12:01:00, 0, NVIDIA A100, 100, 12000, 40960, 65, 220\n",
                encoding="utf-8",
            )

            epochs = read_epoch_history(history_path)
            iterations = read_iteration_losses(log_path)
            gpu = read_gpu_metrics(gpu_path)
            summary = build_summary("test", epochs, iterations, gpu)

            self.assertEqual(len(epochs), 2)
            self.assertEqual(iterations[-1]["global_step"], 100)
            self.assertEqual(len(gpu), 2)
            self.assertEqual(summary["best_epoch_zero_based"], 1)
            self.assertAlmostEqual(summary["best_val_loss"], 0.16)
            self.assertAlmostEqual(summary["gpu_memory_peak_mib"], 12000)

            plot_losses(epochs, iterations, root, smooth_window=2)
            plot_epoch_times(epochs, root)
            plot_gpu_metrics(gpu, root, sample_interval=60)
            write_summary(summary, root)
            for artifact in (
                "training_loss_curves.png", "epoch_times.png", "gpu_telemetry.png",
                "training_summary.json", "training_summary.md",
            ):
                self.assertTrue((root / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
