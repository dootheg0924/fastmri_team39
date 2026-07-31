# Final submission: FI-VarNet 6+6 with MRAugment

This final training configuration adds the fastMRI MRAugment profile from Fabian et al.,
*Data augmentation for deep learning based accelerated MRI reconstruction
with limited data* (ICML 2021), to the final FI-VarNet. This is a submission
training profile, not a fixed-step reproduction experiment:

- train and validation labels are both used for training;
- validation passes are skipped;
- a separate two-epoch run measures throughput before final training;
- the final epoch count is fixed before the scratch model is initialized;
- `model.pt` is saved at every epoch boundary for resume;
- every completed epoch is atomically promoted to `best_model.pt`, so an
  interrupted run always leaves a submission-ready checkpoint.

The FI learning-rate shape is mapped once onto the fixed final horizon: 3.57%
warm-up, plateau until 71.43%, then quarter-cosine decay. MRAugment uses the
same fixed horizon. Neither schedule changes after epoch zero or on resume.

The calibration run has `T=2`: epoch 0 measures the no-geometry path and epoch
1 reaches `p~=0.51`, close to the final augmentation cost. Its checkpoint is
discarded. The resolver subtracts calibration wall time from 95% of the
480-hour allocation, uses the slower of the late-augmentation epoch and
calibration wall-time-per-epoch, and caps the recommendation at 100 epochs.
It writes `recommended_final_epochs.env` and a JSON audit record.

The final profile deliberately leaves `TRAINING_TIME_BUDGET_HOURS` empty.
This prevents timing jitter, a safety stop, or process restart from changing
the trained model. Final training starts from a new random initialization and
can resume only with the same fixed horizon.

## Paper-aligned augmentation

Each transform is sampled independently with probability `p(t) * weight`.

| Transform | Weight | Sampled parameter |
|---|---:|---|
| Horizontal flip | 0.5 | n/a |
| Vertical flip | 0.5 | n/a |
| 90-degree rotation | 0.5 | k in {0, 1, 2, 3} |
| Integer translation | 1.0 | y +/-12.5% height, x +/-8% width |
| Arbitrary rotation | 0.5 | [-180 deg, 180 deg] |
| Isotropic scaling | 0.5 | [0.75, 1.25] |
| Anisotropic scaling | 0.5 | independent axes in [0.75, 1.25] |
| Shear | 1.0 | [-12.5 deg, 12.5 deg] |

Affine transforms use bicubic interpolation (`order=3`). As in the paper's
multi-coil experiments, 2x pre-upsampling is disabled. The exact same geometry
is applied to every real/imaginary coil channel, followed by FFT and masking.

The base probability uses the paper's normalized exponential schedule:

`p(t) = 0.55 * (1 - exp(-5 * t/T)) / (1 - exp(-5))`.

For the final run, `T` is the fixed calibration recommendation (at most 100).

## Deliberate project-specific decisions

1. The MRAugment paper's fastMRI experiment used random R8 masks with 4% ACS.
   This challenge mixes equispaced R4/R8 and its supplied ACS differs by
   acceleration. We infer R and ACS exactly from each stored mask, retain both,
   and randomize only the equispaced offset per volume and epoch. Zero-padded
   acquisition bounds are inferred from full k-space and kept unsampled, as in
   fastMRI's `padding_left`/`padding_right` mask handling.
2. The current public MRAugment repository uses torchvision bilinear affine,
   one scale, and two shear axes. Its README notes that the paper results used
   bicubic interpolation, separate isotropic/anisotropic scaling, and one
   shear. This implementation follows the paper rather than that simplified
   code path.
3. Challenge bbox labels are not part of MRAugment. Boxes are transformed with
   the same geometry and enclosed axis-aligned afterward. If a labeled lesion
   leaves the final 384x384 field or becomes smaller than 7 pixels on either
   axis, geometry is canceled for that sample; remasking is still allowed.
4. Augmentation and mask RNGs are derived from seed, epoch, filename, and
   slice (mask offset excludes slice). This is reproducible across worker
   counts and resumed runs, unlike mutable per-worker RNG state.
5. The challenge explicitly allows participants to repartition train and
   validation. This final refit uses all 200 labeled volumes. The leaderboard
   data remains completely excluded from training.
6. There is no validation-based model selection after combining the labels.
   The latest completed epoch is the submission checkpoint. This is why the
   LR and MRAugment horizons are resolved to a duration expected to finish.

## Reproducible launch sequence

Run the calibration once from a fresh result directory:

```bash
bash scripts/run_fivarnet_mraugment_calibration.sh
```

Do not warm-start from its checkpoint. The final launcher automatically reads
`${RESULT_ROOT}/calibrate_fivarnet_f6i6_mraugment_all_data/recommended_final_epochs.env`.
Review the printed fixed epoch count, then start a new scratch experiment:

```bash
bash scripts/run_fivarnet_mraugment.sh
```

An explicit audited value takes precedence:

```bash
FINAL_NUM_EPOCHS=<calibrated integer> \
  bash scripts/run_fivarnet_mraugment.sh
```

Calibration and final training together may use up to 456 hours. The
remaining 24 hours are reserved for runtime variation, final checkpointing,
and leaderboard reconstruction. Before submission, copy the resolved
`FINAL_NUM_EPOCHS` into the submitted reproduction instructions.

The final preset enables deterministic PyTorch algorithms, deterministic
cuDNN, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `PYTHONHASHSEED=42`. The
calibration uses the same deterministic runtime so its throughput reflects
the final run.

The supplied runtime exposes two CPU cores and about 10 GiB of host memory.
The final loader therefore uses `NUM_WORKERS=2` and `PIN_MEMORY=1`: one worker
per CPU core, with pinned batches enabling the trainer's non-blocking GPU
copies. More workers are avoided because prior runs already held the GPU near
full utilization and host memory reached roughly 8 GiB.
