# Final submission: FI-VarNet 6+6 with MRAugment

This final training configuration adds the fastMRI MRAugment profile from Fabian et al.,
*Data augmentation for deep learning based accelerated MRI reconstruction
with limited data* (ICML 2021), to the final FI-VarNet. This is a submission
training profile, not a fixed-step reproduction experiment:

- training runs for 100 complete epochs;
- `model.pt` is saved at every epoch boundary for resume;
- validation is run every epoch;
- `best_model.pt` tracks the best challenge final score.

The FI learning-rate shape is retained without a fixed step count. After the
loader size is known, the 100-epoch optimizer-step horizon is calculated and
the original relative phases are mapped onto it: 3.57% warm-up, plateau until
71.43%, then quarter-cosine decay.

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

For this final run, `T=100`.

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
5. The final model is selected on held-out validation, so train and validation
   are not combined. This avoids selecting a checkpoint on data used to update
   the model. A later train+validation refit would require a separately fixed
   epoch chosen before that refit.

Run with:

```bash
bash scripts/run_fivarnet_mraugment.sh
```
