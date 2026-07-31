# Experiment 006: FI-VarNet 6+6 for GTX 1080

Giannakopoulos et al., *Accelerated MRI reconstructions via variational
network and feature domain learning* (Scientific Reports, 2024)의 6+6
FI-VarNet을 GTX 1080과 이 저장소의 knee challenge loss에 맞춘 실험이다.

- 논문: https://www.nature.com/articles/s41598-024-59705-0
- 저자 기여 공식 코드(PR #340):
  https://github.com/facebookresearch/fastMRI/pull/340
- 재현 기준 고정 commit:
  https://github.com/facebookresearch/fastMRI/commit/b66159850dbdc2569d7683f6f86bcd5cc8534339
- exp/003 bbox-aware loss 기준:
  https://github.com/dootheg0924/fastmri_team39/blob/36c57e081c26cd1faa92b8b2015e852ccf755f2a/utils/common/bbox_loss.py

## Which paper model

논문에는 두 FI-VarNet 크기가 있다.

- 공정 비교용: feature 6 + image 6 cascades, 약 93.8M parameters
- leaderboard/clinical/knee 최종 모델: feature 12 + image 12 cascades,
  약 187M parameters

이 실험은 8 GB GTX 1080 실행을 위해 6+6 모델을 사용한다. 로컬 구현의
정확한 parameter count는 93,805,980이다.

## Resolved architecture

| Setting | Value |
| --- | --- |
| Feature / image cascades | 6 / 6 |
| Feature channels | 32 |
| Feature encoder / decoder | shared 5x5 conv, padding 2, bias |
| Feature U-Net | author code `Unet2d`, 32 base channels, 4 pools |
| Image U-Net | `NormUnet`, 32 base channels, 4 pools |
| Sensitivity U-Net | 8 base channels, 4 pools, ACS only |
| Attention | all 6 feature cascades |
| Attention Q/K/V | 1x1 conv + shared 3x3 dilated conv (dilation 2) |
| Positional encoding | sinusoidal, base 10000 |
| Extra feature residual blocks | cascades 0, 3 |
| U-Net conv / activation | 3x3, InstanceNorm, LeakyReLU 0.2 |
| Pool / upsample | average pool 2; transpose conv 2 |
| Dropout | 0 |
| Feature/image DC scalar | one per cascade, initialized to 1 |
| FFT | centered, orthonormal |
| k-space scale | 1e6 |
| Output | coil magnitude RSS |

Feature encoder/decoder와 sensitivity map만 공유한다. 각 cascade의
attention, U-Net, data-consistency scalar는 서로 독립이다.

## Optimization

| Setting | Value |
| --- | --- |
| Loss | `(1 - SSIM_full) + 0.5 * (1 - SSIM_bbox)` |
| SSIM | window 7, k1=0.01, k2=0.03, H5 `max` data range |
| Optimizer | AdamW |
| LR / weight decay | 3e-4 / 0 |
| Adam defaults | betas=(0.9, 0.999), eps=1e-8, amsgrad=False |
| Training length | 210,000 optimizer updates |
| Warm-up | 7,500 updates |
| Cosine start | update 150,000 |
| Gradient clipping | global norm 1.0 |
| Seed | 42 |
| Precision | FP32, `torch.set_float32_matmul_precision("high")` |
| Initialization | PyTorch defaults; no custom initialization |
| Augmentation / EMA | none / none |
| cuDNN deterministic | disabled, matching the public runner |
| Physical / effective batch | 1 / 4 |
| Gradient accumulation | 4 |
| Gradient checkpointing | enabled for sensitivity + all cascades |
| DataLoader workers / pin memory | 4 / false |
| Knee checkpoint | final weights at optimizer update 210,000 |

The released scheduler is reproduced literally:

```text
step < 150000:
    multiplier = min(step / 7500, 1)
step >= 150000:
    multiplier = max(
        cos(((step - 150000) / 60000) * pi/2),
        1e-8
    )
lr = 3e-4 * multiplier
```

The paper calls the plateau “140k steps,” while the released code starts
cosine decay at 150k (a 142.5k-step plateau after warm-up). The `1e-8` floor
in the code is a **multiplier**, so the terminal configured LR is `3e-12`;
this subtlety is kept for code-level parity.

`SSIM_full`은 384x384 전체를 그대로 평균한 plain SSIM이 아니라 exp/003과
대회 metric의 foreground-masked SSIM이다. bbox 항은 유효한 annotation
crop들의 `1-SSIM`을 동일 가중 평균하며, 한 변이 7 pixel보다 작거나
annotation이 없는 slice에서는 bbox 항을 생략한다.

Physical batch 1은 선택 사항이 아니다. 논문 자체도 GPU당 batch 1을
사용했고, FI-VarNet의 `NormStats`가 batch 1을 요구하며 annotation box 수도
slice마다 달라 기본 collate로 batch>1을 묶을 수 없다. 공개 학습은 4-GPU라
optimizer update마다 총 4 samples를 보므로, 단일 GTX 1080에서는 네 개의
batch-1 microbatch를 누적해 effective batch 4를 재현한다. Gradient
accumulation은 한 번에 유지하는 activation을 늘리지 않으므로 peak VRAM은
물리 batch 1과 같다.

The paper explicitly skipped validation for knee. Accordingly, intermediate
epochs save only the resumable `model.pt`. At update 210,000 the final state
is atomically saved and promoted to `best_model.pt` before a reporting-only
validation pass, so evaluation failure cannot discard the paper checkpoint.
Challenge foreground/bbox scores and plain SSIM are then written for
reporting, but neither can replace that final state.
`CHECKPOINT_METRIC=paper-ssim` remains available for the released brain
runner's validation-selected protocol.

## Data protocol and unavoidable challenge differences

The paper trains a separate model for each acceleration:

| R | ACS center fraction |
| --- | --- |
| 4 | 0.08 |
| 5 | 0.07 |
| 8 | 0.04 |

Its train transform generates an `equispaced_fraction` mask with a new random
offset, while validation uses a filename-seeded fixed mask. This challenge
ships already-masked k-space and its mask in HDF5, so those masks cannot be
regenerated without changing the supplied task.

The reference transform passes each acquisition's target crop and ACS count
to the model. This repository's competition contract instead fixes the target
to 384x384 and infers the ACS extent from the supplied mask, so the port keeps
that required local behavior.

The local submission also loads one checkpoint for both acc4 and acc8.
Therefore it infers the attention period from each HDF5 mask. The paper uses
a fixed R in each separately trained network. Architecture and optimizer
hyperparameters are the paper values, but this mixed-acceleration routing is
an explicit challenge-only extension; it is not a claim of bit-exact paper
weights.

For knee, the paper trains on the knee training split and uses validation as
test without validation-based checkpoint selection, so
`COMBINE_TRAIN_VAL=0` and `CHECKPOINT_METRIC=paper-final`. The public **brain
leaderboard** runner instead hard-codes `combine_train_val=True` and monitors
validation SSIM. To deliberately reproduce that protocol, set both
`COMBINE_TRAIN_VAL=1` and `CHECKPOINT_METRIC=paper-ssim`, accepting validation
reuse.

The public README says PyTorch 1.7, but the released code calls an API added in
PyTorch 1.12 (`set_float32_matmul_precision`). Its dependency versions are not
fully pinned, and its random train-mask generator is not seeded by seed 42.
Consequently the publication artifacts do not permit bit-for-bit reruns.

## Launch

```bash
bash scripts/run_fivarnet_paper.sh
```

Only runtime locations/devices normally need overriding:

```bash
DATA_ROOT=/root/Data \
RESULT_ROOT=/root/result \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_fivarnet_paper.sh
```

6+6 모델은 cascade checkpointing을 활성화한다. Launch script는 데이터셋
메타데이터에서 공간 크기×가속도(동률이면 coil 수)가 가장 큰 후보를
자동으로 골라 forward, backward, gradient clipping, AdamW step까지
수행한다. 따라서 optimizer state가 할당된 상태의 peak VRAM도 확인한다.
로컬 환경에는 GTX 1080이 없어 8 GB peak를 사전 실측할 수 없으므로 이
smoke test가 OOM을 내거나 여유 VRAM이 8% 미만이면 본 학습을 시작하지
않는다.

## Verification

```bash
python -m pytest tests/test_fivarnet_paper.py -q
```

The tests pin every preset value, the requested exp/003 loss weighting, LR
boundaries, 93,805,980-parameter 6+6 topology, checkpointing, generic
microbatch gradient equivalence, paper-final checkpointing, retained
challenge metrics, and optimizer/scheduler resume.
