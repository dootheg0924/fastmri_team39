# Submission manifest plan

## 고정된 code path

- 학습: `scripts/run_final_staged_reproduction.sh`
- 후보 생성: `scripts/prepare_final_candidate.py`
- 공식 평가: `scripts/run_final_eval.sh` → 변경 없는 `recon_eval.py`
- inference: `utils/learning/test_part.py`
- stage configs: `experiments/007_fivarnet_mraugment`, `experiments/008_fivarnet_cross_acc`

## 후보 두 개

| Candidate ID | Source | 생성 규칙 | 상태 |
|---|---|---|---|
| `epoch89` | epoch 89 | 원본 checkpoint byte-for-byte copy | checkpoint 도착 대기 |
| `avg_80_85_89` | epoch 80, 85, 89 | floating/complex state 1:1:1 arithmetic mean | checkpoint 도착 대기 |

두 후보는 동일한 `checkpoints/best_model.pt` layout과 `candidate_manifest.json`을 사용한다. 최종 선택 전에는 어느 것도 `submission/FINAL_SELECTION.json`으로 승격하지 않는다.

epoch 89 학습이 완전히 저장된 뒤 두 후보를 한 번에 생성한다.

```bash
RESULT_ROOT=/root/result bash scripts/prepare_both_final_candidates.sh
```

각 candidate directory는 덮어쓰지 않으므로 생성 이후 내용이 바뀌면 즉시 탐지할 수 있다.

## 최종 package 필수 항목

- `checkpoint/best_model.pt`
- `candidate_manifest.json`
- `FINAL_SELECTION.json`
- `code.tar.gz`
- `evidence/`
- PPT 파일
- 설명 영상 파일
- `SHA256SUMS`

정확한 외부 파일명과 이메일 양식은 운영진 이메일 첨부 발표자료를 따른다.
