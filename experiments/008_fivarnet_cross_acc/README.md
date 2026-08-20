# Stage two: cross-acceleration re-masking from the epoch-50 checkpoint

This profile continues `experiments/007_fivarnet_mraugment` and changes exactly
one thing about training: which acceleration a volume is presented at.

The run is deliberately **staged**, not restarted. Stage one is the 007 profile
from epoch 0 to 50; stage two is this profile from epoch 50 to 100, resumed from
stage one's epoch-50 checkpoint. The supplied end-to-end runner reproduces both
stages from scratch without a timing-dependent manual stop.

## Rule book compliance

The challenge rule book (LISTatSNU/FastMRI_challenge issue #412) governs this:

- *"주어진 train/val 데이터를 가공(augmentation 등)하여 학습에 사용하는 것은 가능합니다.
  이때 데이터 전처리 코드 또한 제출 대상입니다."* — cross-acceleration re-masking is
  processing of the provided training data. `utils/data/transforms.py`,
  `utils/data/mraugment.py` and `utils/data/load_data.py` are part of the submission.
- *"VESSL 서버(GTX 1080)와 주어진 train/val 데이터, 그리고 직접 작성하신 코드만으로
  처음부터 끝까지 재현이 가능해야 합니다."* — no external data, no pretrained
  weights, no leaderboard data. Nothing here introduces any.
- *"README.md의 내용을 그대로 따라 모델을 훈련하였을 때 ... 소수점 네 자릿수에서
  동일하게 재현"* — see **Reproducibility** below.
- *"사용하시는 함수에 따라 기본으로 제공된 `seed_fix()`만으로는 완전한 재현성이
  보장되지 않을 수 있습니다."* — this profile keeps 007's `DETERMINISTIC=1`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED`, and every new random
  draw is hash-derived rather than drawn from a mutable global stream.
- The rule book also forbids using image-file fields as inference input. This
  profile changes training only; `recon_slice()` is untouched.

## What changes

007 regenerates each volume's mask after augmentation, "retaining its R and ACS
fraction", and randomizes only the equispaced offset. So an R8 training example
can only come from a volume that shipped an R8 mask.

That restriction is not required by the data. The stored `kspace` is complete
and `image_label` is its RSS — measured on the sample volumes, the label matches
`RSS(IFFT(kspace))` at Pearson r = 1.000000, and the columns the stored mask
drops still carry energy. **The label does not depend on the mask.** Any volume
can therefore be re-undersampled at R8 and stay a valid acc8 pair.

With `CROSS_ACCELERATION=1.0` and `CROSS_ACCELERATION_P8=0.5`, each volume-epoch
draws its acceleration instead of inheriting it. In expectation, the R8 half of
training is then sourced from *all* volumes rather than only the R8-tagged half,
roughly doubling the distinct anatomy behind R8 examples. The draw is
deterministic but Bernoulli per volume, so an individual epoch is not guaranteed
to contain an exact 50/50 split (`BALANCE_ACCELERATIONS=0`).

This is not a new technique: it is what the reference fastMRI `MaskFunc` does,
vendored in this repository at `utils/model/fastmri/data/subsample.py:65-67`,
where the acceleration and its paired centre fraction are drawn per example.
The challenge ships one fixed mask per volume, which removed that randomization.

The ACS width is not carried over from the source volume by assumption.
`load_data.scan_mask_specifications` reads every volume's 1-D mask once in the
parent process, and a re-targeted mask adopts the destination acceleration's own
ACS fraction. The scan reuses 007's `infer_equispaced_mask`, which refuses any
mask that is not an exact centered-ACS equispaced R4/R8 pattern, and resolves
ties in sorted order so it is a pure function of the data.

Scanned on the real training set (200 volumes, 2026-08-12), both accelerations
turn out to share one ACS geometry:

    Cross-acceleration mask catalogue -> R4: 100 volumes, ACS 29 lines (0.0788),
                                         R8: 100 volumes, ACS 29 lines (0.0788)

So re-targeting leaves the ACS essentially unchanged and only the outer stride
moves. An earlier reading of two sample volumes suggested R8 used a wider ACS
(31/372); that pair was not representative of the set. The full dataset scan
and the expected startup output below both record the corrected 29-line ACS.

## Reproducibility

Stage two reproduces only if it starts from the same stage-one weights, so:

- `--expected-resume-epoch 50` fails the launch if `model.pt` holds an earlier
  epoch. Resuming later than 50 after an interruption is allowed and logged.
- The SHA-256 of the resumed checkpoint is printed at startup and recorded in
  `run_metadata.json` under `arguments.resume_metadata`. It proves the lineage
  of the actual VESSL stage transition. It is not a requirement that a later
  reproduction produce a byte-identical checkpoint archive: serialized CLI
  metadata can change the file digest without changing learned state. The rule
  book criterion is the reproduced final score.
- The acceleration draw uses
  `deterministic_rng(seed, epoch, fname, purpose="cross-acceleration")`, a
  blake2b hash of those fields. It does not depend on worker count, worker
  scheduling, or how many samples preceded it, so a rerun with a different
  `NUM_WORKERS` still draws the same acceleration for every volume-epoch.
- That purpose string is distinct from 007's `"augment"` and `"mask"` streams,
  so enabling this leaves the geometry and offset draws of any volume-epoch
  untouched. `tests/test_cross_acceleration.py` asserts that a volume whose
  drawn acceleration equals its stored one produces a bit-identical sample.
- Every schedule-bearing value is held at the 007 figure. `FINAL_NUM_EPOCHS`
  stays 100, so the LR shape and the MRAugment horizon do not retune on resume.
- `CROSS_ACCELERATION=0` restores 007 behaviour exactly.

## Fresh end-to-end reproduction

Run this only with fresh stage-one and stage-two result directories:

```bash
FINAL_STAGE_STOP_EPOCH=89 bash scripts/run_final_staged_reproduction.sh
```

`scripts/run_final_staged_reproduction.sh` runs 007 with
`STAGE_STOP_EPOCH=50`, verifies that `model.pt` stores exactly 50 completed
epochs, archives and fingerprints it, copies it into a fresh 008 result
directory, and launches stage two through completed epoch 89.
`STAGE_STOP_EPOCH` is deliberately separate from `NUM_EPOCHS=100`: it ends each
stage without changing the 100-epoch LR or MRAugment schedule horizon.

The defaults assume `/root/Data` and `../result`. `DATA_ROOT`, `RESULT_ROOT`,
`STAGE1_EXP_NAME`, and `STAGE2_EXP_NAME` may be overridden for an isolated
reproduction. The runner refuses to reuse either experiment directory.

## Historical stage-one procedure and checkpoint

007 runs with `CHECKPOINT_INTERVAL=0`, so it keeps only `model.pt` (rewritten
every epoch) and `best_model.pt`. There is one GPU, so stage one has to be
stopped before stage two can start anyway — and `save_model` only writes at
epoch boundaries, so `model.pt` still holds epoch 50 for the whole of epoch 51.
The window is a full epoch, not an instant.

**The epoch printed in the log is one less than the epoch stored in the
checkpoint**: the loop prints its zero-based index and saves `epoch + 1`. Read
the checkpoint, not the log:

```bash
SRC=../result/final_fivarnet_submission_f6i6_mraugment_all_data/checkpoints
python -c "
import torch
print('completed epochs =', torch.load('$SRC/model.pt', map_location='cpu', weights_only=False)['epoch'])
"
```

The submitted run predates the automated stage boundary. When the command
printed 50, stage one was stopped and the weights were archived. The archive
copy is separate on purpose: stage two overwrites its own `model.pt` at its
first epoch, and this file is both the baseline for `leaderboard_eval.py` and
the provenance record.

```bash
mkdir -p ../result/staged_checkpoints
cp "$SRC/model.pt" ../result/staged_checkpoints/checkpoint_epoch_0050.pt
sha256sum ../result/staged_checkpoints/checkpoint_epoch_0050.pt
```

`scripts/capture_epoch_checkpoint.py` does the same thing unattended, for when
nobody will be watching for longer than an epoch, or to preserve several epochs
at once. It is a convenience, not a requirement.

### This run's staged checkpoint

Stage one ran from 2026-08-03 to 2026-08-12 on VESSL (GTX 1080) under
`experiments/007_fivarnet_mraugment`, and was stopped once `model.pt` reported
50 completed epochs. That checkpoint is the boundary between the two stages:

```
file    result/staged_checkpoints/checkpoint_epoch_0050.pt
epoch   50
sha256  4ae5f62c343f28e25809e0b0f2fd04154a34d8144fdff9546359096ac3cf43b6
```

The digest identifies the checkpoint used by the actual VESSL run. A fresh
reproduction must reach completed epoch 50 with the documented schedule and
then reproduce the final score; its serialized checkpoint digest need not be
identical. Stage two's startup log records the digest it actually resumed from.

Stage-one epochs took about 5h20m each at this point in training (measured at
3.52 s per micro-batch over 5,444 micro-batches). Earlier epochs were faster
because MRAugment's probability ramps from 0, so the run's overall average
understates the cost of later epochs.

## Stage two: launch

```bash
DST=../result/final_fivarnet_submission_f6i6_mraugment_cross_acc
mkdir -p "$DST/checkpoints"
cp ../result/staged_checkpoints/checkpoint_epoch_0050.pt "$DST/checkpoints/model.pt"

bash scripts/run_fivarnet_cross_acc.sh
```

A fresh result directory is deliberate: stage one's `best_model.pt` stays intact
as a submission fallback, so an unhelpful stage two costs nothing but time.

Check two lines in the startup log before letting it run:

```
Resumed from .../model.pt at epoch 50; global step ...
Resume checkpoint sha256: <64 hex chars>
Cross-acceleration mask catalogue -> R4: 100 volumes, ACS 29 lines (0.0788), R8: 100 volumes, ACS 29 lines (0.0788)
```

The catalogue is derived exclusively from the supplied train and validation
volumes. Leaderboard data and leaderboard-derived statistics are not used by
the training pipeline.

## VESSL provenance and evidence

The actual final run left the following evidence under `../result` on VESSL:

```text
final_fivarnet_submission_f6i6_mraugment_all_data/
  resolved_config.env
  git_state.txt
  python_environment.txt
  gpu_environment.txt
  run_metadata.json
  training_history.csv
final_fivarnet_submission_f6i6_mraugment_cross_acc/
  resolved_config.env
  git_state.txt
  python_environment.txt
  gpu_environment.txt
  run_metadata.json
  training_history.csv
logs/final_fivarnet_submission_f6i6_mraugment_all_data.log
logs/final_fivarnet_submission_f6i6_mraugment_cross_acc.log
stage2_launch.log
staged_checkpoints/checkpoint_epoch_0050.pt
```

The experiment launchers also record the exact Git commit, dirty-worktree
state, resolved arguments, Python packages, GPU environment, per-epoch history,
and the stage-two resume digest. These files must be archived with the final
submission evidence even though local `*.log` files are excluded from Git.

The submitted VESSL environment used Python 3.10.12 and PyTorch 2.3.1+cu121.
Direct dependencies are pinned in `requirements.txt`; the full captured VESSL
environment is retained in `requirements-vessl.lock.txt` and each experiment's
`python_environment.txt`.

## Checkpoints kept

| File | Written | Purpose |
|---|---|---|
| `model.pt` | every epoch | resume point |
| `best_model.pt` | every epoch (`submission-latest`) | latest complete epoch, submission-ready |
| `checkpoint_epoch_{0080,0085,0090,0095}.pt` | those epochs only | standalone late-epoch snapshots |

`CHECKPOINT_EPOCHS="80 85 90 95"` preserves diagnostic snapshots from the
actual run. They are not final candidates: the final submission is fixed to
the epoch-89 `best_model.pt`. The snapshots are copies taken alongside
`model.pt` and are never promoted, so `best_model.pt` is unaffected.

## Known limits

1. **This run cannot measure its own effect.** The final profile combines the
   train and validation labels and skips validation, so there is no held-out
   score. Judging the change means running `leaderboard_eval.py` on the epoch-50
   checkpoint and on a stage-two checkpoint, on a machine with `cv2`.
2. The change is a distribution shift and augmentation costs before it pays.
   `CROSS_ACCELERATION` accepts any probability in [0, 1]; 0.25 leaves three
   quarters of volume-epochs on their stored acceleration.
3. Keep `CROSS_ACCELERATION_P8` at 0.5. Raising it raises the share of R8
   samples, whose gradients were measured at 2.0-3.2x the R4 norm, which would
   tilt the effective objective further toward R8.
4. `RUN_SMOKE_TEST=1` runs `scripts/smoke_test_training.py` before training.
   It has not been exercised locally with the new flags — no GPU or data here —
   so watch the first launch through the smoke stage.
