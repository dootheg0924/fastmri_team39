# Experiment 007: FI-VarNet 6+6 with MRAugment

This experiment adds the fastMRI MRAugment profile from Fabian et al.,
*Data augmentation for deep learning based accelerated MRI reconstruction
with limited data* (ICML 2021), to experiment 006. The model, optimizer,
210,000 optimizer-update safety ceiling, and objective remain unchanged. The
run now has an explicit 100-epoch hard cap and can be resumed from the
epoch-boundary `model.pt` checkpoint.

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

For this experiment, `T=100`, unless the 210k optimizer-update safety ceiling
would finish earlier. This preserves the paper's epoch-normalized curve over
the actual configured run.

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
5. Training stops at the first of 100 epochs or 210k optimizer updates. The
   100-epoch cap makes intentional interruption/resume practical while the
   original FI safety ceiling prevents an unexpectedly large dataset from
   exceeding the paper-aligned update budget.

Run with:

```bash
bash scripts/run_fivarnet_mraugment.sh
```
