# Experiment 002: VarNet c6/ch12/s4 Long Run

## Objective

Increase reconstruction capacity over the one-cascade baseline while keeping the sensitivity-map network small enough for a practical VESSL run. This experiment changes only model capacity and the training duration; the data split, loss, seed, and evaluation functions remain aligned with the baseline.

The tracked experiment branch is `exp/002-varnet-c6-long`. Every launch also records the exact Git commit and runtime environment in `run_metadata.json`.

## Hypothesis and baseline

The baseline uses `cascade=1`, `chans=9`, and `sens_chans=4`. Its leaderboard result is:

| Model | SSIM_full | SSIM_bbox | Time |
| --- | ---: | ---: | ---: |
| Baseline `test_Varnet` | 0.8787 | 0.8650 | 87.8 ms/slice |

Six cascades and 12 regularizer channels should improve both SSIM metrics, with increased memory use and reconstruction time. The experiment is successful only if the accuracy gain justifies that time increase.

## Canonical configuration

The source of truth is [config.env](./config.env).

| Setting | Value |
| --- | ---: |
| Experiment name | `varnet_c6_ch12_s4_ep80_lr3e4` |
| Cascades | 6 |
| Cascade channels | 12 |
| Sensitivity channels | 4 |
| Epochs | 80 |
| Batch size | 1 |
| Learning rate | 3e-4 |
| Seed | 430 |
| Checkpoint snapshot interval | 10 epochs |
| GPU telemetry interval | 60 seconds |
| Preflight | One real-slice forward/backward smoke test |

The model has exactly 6,666,660 trainable parameters with the current implementation. Training uses the existing full-image SSIM loss; challenge-aligned foreground and bounding-box metrics are computed separately from the best validation reconstruction.

## VESSL quick start

Use a CUDA/PyTorch image and mount the challenge data at `/root/Data` with this structure:

```text
/root/Data/
  train/image/*.h5
  train/kspace/*.h5
  val/image/*.h5
  val/kspace/*.h5
  leaderboard/acc4/{image,kspace}/*.h5
  leaderboard/acc8/{image,kspace}/*.h5
```

If VESSL already cloned the repository, use this start command:

```bash
set -euo pipefail
cd /root/fastmri_team39
git fetch origin exp/002-varnet-c6-long
git switch --detach origin/exp/002-varnet-c6-long
INSTALL_DEPS=1 DATA_ROOT=/root/Data RESULT_ROOT=/root/result \
  bash scripts/vessl_entrypoint.sh
```

`INSTALL_DEPS=1` can be omitted when the image already satisfies `requirements.txt`. A detached checkout is intentional: it prevents an accidental branch change during a long run, while `run_metadata.json` records the exact commit.

For a different VESSL checkout path, no script edit is needed. The scripts resolve the repository from their own location. Dataset and result locations can be overridden with `DATA_ROOT` and `RESULT_ROOT`.

## Resume behavior

The launch command is idempotent for the same `EXP_NAME` and `RESULT_ROOT`.

- First launch: no `model.pt` exists, so training starts at epoch 0.
- Restart: `model.pt` is loaded with model, optimizer, best loss, history, and RNG states.
- Completed run: if the checkpoint already reached 80 epochs, training exits without overwriting it.

VESSL storage containing `/root/result` must be persistent across job restarts. Without a persistent result volume, checkpoint resume is impossible.

## Monitoring

```bash
tail -f /root/result/logs/varnet_c6_ch12_s4_ep80_lr3e4.log
tail -f /root/result/logs/varnet_c6_ch12_s4_ep80_lr3e4_gpu.csv
watch -n 1 nvidia-smi
```

Partial analysis can be regenerated while training is active:

```bash
cd /root/fastmri_team39
python scripts/analyze_training.py \
  --exp-name varnet_c6_ch12_s4_ep80_lr3e4 \
  --result-root /root/result
```

## Generated artifacts

```text
/root/result/varnet_c6_ch12_s4_ep80_lr3e4/
  checkpoints/
    model.pt                       # latest resumable state
    best_model.pt                  # lowest validation loss
    checkpoint_epoch_0010.pt       # periodic snapshots
    ...
  reconstructions_val/             # output of the current best model
  analysis/
    training_loss_curves.png
    epoch_times.png
    gpu_telemetry.png
    training_summary.{md,json}
    iteration_loss.csv
    validation_metrics.{md,json}
    validation_slice_metrics.csv
    validation_worst_slices.png
  training_history.csv             # normalized per-epoch losses
  val_loss_log.npy                 # normalized validation loss
  resolved_config.env              # effective shell configuration
  run_metadata.json                # args, environment, GPU, Git SHA, launches
  python_environment.txt           # pip freeze snapshot
  gpu_environment.txt              # detailed nvidia-smi snapshot
  git_state.txt                     # exact commit and dirty-state record
  TRAINING_COMPLETED               # written only after train.py succeeds
```

Global logs live in `/root/result/logs/`. The run script appends rather than truncates them, so restart history is preserved.

## Post-training leaderboard evaluation

Use the same architecture flags that created the checkpoint:

```bash
cd /root/fastmri_team39
python recon_eval.py \
  -n varnet_c6_ch12_s4_ep80_lr3e4 \
  -p /root/Data/leaderboard \
  --result-root /root/result \
  --cascade 6 \
  --chans 12 \
  --sens_chans 4 \
  | tee /root/result/varnet_c6_ch12_s4_ep80_lr3e4/recon_eval_output.txt
```

Record overall and per-acceleration SSIM, reconstruction time, GPU model, and the exact Git SHA in [RESULTS.md](./RESULTS.md).

## Export lightweight results back to Git

Checkpoints and H5 reconstructions should remain on artifact storage. Export only reproducibility metadata, metrics, plots, and log tails:

```bash
python scripts/export_experiment_report.py \
  --exp-name varnet_c6_ch12_s4_ep80_lr3e4 \
  --result-root /root/result \
  --output-root reports
```

Review `reports/varnet_c6_ch12_s4_ep80_lr3e4/` before committing it. The generated manifest contains sizes and SHA-256 hashes.

## Operational checklist

- Confirm `/root/result` is persistent.
- Confirm the smoke test reports finite loss and acceptable peak GPU memory.
- Confirm the metadata Git SHA matches the intended branch commit.
- Check for NaN or rising validation loss after the first few epochs.
- Compare train and validation loss for overfitting.
- Check peak GPU memory and temperature.
- Confirm `best_model.pt` and periodic snapshots are present.
- Run validation metric analysis before leaderboard evaluation.
- Run leaderboard evaluation with matching architecture flags.
- Export lightweight artifacts and complete `RESULTS.md`.
