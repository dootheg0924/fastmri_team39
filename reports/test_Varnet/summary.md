# test_Varnet

## Validation result

- Best epoch: 4
- Best validation loss: 3.742340
- Last epoch: 4
- Last validation loss: 3.742340

The stored validation values above are sums across validation volumes from the original baseline training code, not normalized per-volume losses.

## Leaderboard result

- SSIM_full: 0.8787
- SSIM_bbox: 0.8650
- Reconstruction time: 194.33 s total (87.8 ms/slice)
- Detailed report: [leaderboard_results.md](leaderboard_results.md)

## VESSL paths

- Result directory: `/root/result/test_Varnet`
- Best model: `/root/result/test_Varnet/checkpoints/best_model.pt`
