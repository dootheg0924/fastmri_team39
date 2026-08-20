# Final submission evidence index

## One-command server capture

After the submission-preparation commit is present on the VESSL workspace, run:

```bash
RESULT_ROOT=/root/result bash scripts/capture_submission_evidence.sh
```

This creates a new timestamped directory under
`/root/result/final_submission_evidence/`. It records the exact Git commit and
dirty patch, a source archive, Python/package/GPU/process state, available
experiment metadata and logs, checkpoint hashes, and a `SHA256SUMS` index. It
does not signal the training process or modify checkpoints.

최종 모델은 epoch 89 단일 checkpoint로 확정했다. 아래 자료를 원본 그대로
보존하고 복사본에는 SHA-256을 붙이며 원본 terminal/Codex 창은 닫지 않는다.

## final_training

- [ ] stage 1 전체 log
- [ ] stage 2 전체 log
- [ ] `training_history.csv`
- [ ] GPU telemetry CSV
- [ ] `TRAINING_COMPLETED` 또는 마지막 완료 epoch 시각
- [ ] VESSL workload/activity 화면

## stage_transition

- [ ] epoch-50 checkpoint
- [ ] epoch-50 SHA-256
- [ ] stage-2 startup의 resume epoch/global step/hash 출력

## reproducibility

- [ ] stage-1 초기 trajectory A/B log
- [ ] 동일 epoch-50 checkpoint에서 시작한 stage-2 초기 trajectory A/B log
- [ ] iteration/loss/LR/sample/augmentation 비교표
- [ ] 불일치가 있으면 원인과 최종 점수 네 자리 영향 보고서

## environment

- [ ] `resolved_config.env`
- [ ] `run_metadata.json`
- [ ] `python_environment.txt`
- [ ] `gpu_environment.txt`
- [ ] `git_state.txt`
- [ ] `git diff --binary HEAD`
- [ ] 학습 당시 source tree SHA-256 manifest

## official_eval

- [ ] weight average 성능 하락 diagnostic 결과(제출 후보 아님)
- [ ] `epoch89` candidate의 VESSL 공식 `recon_eval.py` log
- [ ] 최종 `eval_metadata_*.json`
- [ ] checkpoint 평가 전/후 SHA-256
- [ ] leaderboard 업로드 화면

## screenshots

- [ ] 실행 중 terminal 또는 Codex 창 + 화면 clock
- [ ] `tail -f` log + 화면 clock
- [ ] checkpoint 파일/시각/hash
- [ ] 최종 leaderboard 값
- [ ] cloud/VESSL 제출 경로
- [ ] 전송 완료 이메일

## epoch89 저장·평가 뒤에 할 수 있는 것

- epoch89 checkpoint SHA-256 확정
- 최종 공식 점수·시간 확정
- `scripts/finalize_submission.py` 실행
- PPT/영상에 최종 값 삽입
- 최종 package와 `SHA256SUMS` 생성
