# Submission manifest plan

## 고정된 code path

- 학습: `scripts/run_final_staged_reproduction.sh`
- 최종 후보 생성: `scripts/prepare_epoch89_final_candidate.sh`
- 공식 평가: `scripts/run_final_eval.sh` → 변경 없는 `recon_eval.py`
- inference: `utils/learning/test_part.py`
- stage configs: `experiments/007_fivarnet_mraugment`, `experiments/008_fivarnet_cross_acc`

## 확정된 최종 후보

| Candidate ID | Source | 생성 규칙 | 상태 |
|---|---|---|---|
| `epoch89` | epoch 89 | 원본 checkpoint byte-for-byte copy | **최종 선택 확정** |

80/85/89 weight average는 diagnostic에서 성능 하락을 보여 최종 제출에서
제외했다. prediction ensemble은 구현 기록만 별도 branch에 보존하고 main 및
최종 package에는 포함하지 않는다.

epoch 89 학습이 완전히 저장된 뒤 최종 후보 하나만 생성한다.

```bash
RESULT_ROOT=/root/result bash scripts/prepare_epoch89_final_candidate.sh
```

candidate directory는 덮어쓰지 않으므로 생성 이후 내용이 바뀌면 즉시
탐지할 수 있다. `scripts/finalize_submission.py`는 `epoch89`, `single`, source
epoch `[89]`, stored epoch `89`가 아니면 최종화를 거부한다.

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
