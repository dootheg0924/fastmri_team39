# test_Varnet Leaderboard Results

## Overall

| Metric | Result |
| --- | ---: |
| SSIM_full | **0.8787** |
| SSIM_bbox | **0.8650** |
| Total reconstruction time | **194.33 s** |
| Mean reconstruction time | **87.8 ms/slice** |

## Results by acceleration

| Acceleration | SSIM_full | SSIM_bbox | Reconstruction time | Mean time |
| --- | ---: | ---: | ---: | ---: |
| acc4 | 0.9086 | 0.8953 | 100.49 s | 88.1 ms/slice |
| acc8 | 0.8488 | 0.8347 | 93.84 s | 87.5 ms/slice |

The overall SSIM scores are the equal-weight averages of the acc4 and acc8 scores. The total reconstruction time is the sum of the two acceleration-specific times.

## Raw evaluation output

```text
Leaderboard SSIM_full : 0.8787
Leaderboard SSIM_bbox : 0.8650
Leaderboard Recon Time : 194.33s (87.8 ms/slice)
========== Details ==========
SSIM_full (acc4): 0.9086   SSIM_full (acc8): 0.8488
SSIM_bbox (acc4): 0.8953   SSIM_bbox (acc8): 0.8347
Recon Time (acc4): 100.49s (88.1 ms/slice)   (acc8): 93.84s (87.5 ms/slice)
Recon Time (total): 194.33s
```
