# Experiment 005: epoch-40 F5 attention split

This experiment continues the trained exp003 checkpoint for exactly 40 more
epochs. It does not restart FI-VarNet and does not split the 2.46M-parameter F5
feature U-Net.

## Model change

- F0-F4 remain fully shared.
- F5 `feature_processor`, `dc_weight`, and `output_conv` remain shared.
- F5 attention is copied into an acc4 expert and an acc8 expert.
- The existing mask-stride acceleration inference selects one expert; there is
  no learned router.
- Decoder and all image-space cascades remain shared.

At migration, both attention experts receive the exact epoch-40 weights and
independent clones of the original Adam `step`, `exp_avg`, and `exp_avg_sq`.
The split model is therefore function-identical to the source model before its
first continuation update.

## Acceleration balance

Training uses an exact alternating acc4/acc8 sampler with batch size 1. In the
default `oversample` mode, every slice in the larger group is visited once and
the smaller group is reshuffled and cycled to the same count. Even absolute
epochs start with acc4 and odd epochs start with acc8. Sampling uses a private
`seed + absolute_epoch` generator, so resuming epoch 40-80 is deterministic.

Validation is not sampled or balanced. It retains every validation slice in
the original distribution, while `validate()` continues to give acc4 and acc8
equal weight in the competition score.

Any shared-control continuation used for comparison must use the same balanced
sampler and epoch range; otherwise sampling and attention splitting are
confounded.

If oversampling becomes a concern, set:

```bash
ACCELERATION_BALANCE_MODE=undersample
```

This drops majority slices instead of repeating minority slices and is mainly
an ablation; `oversample` is preferred because it retains all training data.

## Launch

Keep the source checkpoint outside the new result directory:

```bash
WARM_START_CHECKPOINT=/root/result/<exp003>/checkpoints/model.pt \
CONFIG_FILE=experiments/005_f5_attention_split/config.env \
bash scripts/run_fivarnet_bbox.sh
```

The launch fails unless the source checkpoint stores epoch 40. The checkpoint's
actual architecture and loss arguments take precedence over conflicting config
values. This prevents accidentally loading the f6/i6/ch18 checkpoint into the
repository's smaller f4/i2/ch12 default.

The first launch:

1. migrates weights and Adam state;
2. evaluates the function-preserving epoch-40 baseline with the competition
   metric;
3. writes `warm_start_baseline.json`;
4. saves a native split `model.pt`, `best_model.pt`, and epoch-40 snapshot;
5. trains absolute epochs 40 through 79.

After interruption, run the same command again. `--resume` prefers the native
split `model.pt` in this experiment over repeating the source migration.

## Imbalance interpretation

The sampler balances the number of acc4/acc8 slice updates. It intentionally
does not also balance annotation counts, lesion types, or volumes, because
changing all of those at once would confound the attention-split experiment.
The resolved config and `acceleration_sampling.json` record the source and
sampled slice counts. If annotation-box counts remain strongly imbalanced,
evaluate that separately before introducing a four-stratum sampler.
