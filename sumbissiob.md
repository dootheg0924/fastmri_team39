# Team 39 — 2026 FastMRI Final Submission 실행 계획

> 기준일: 2026-08-20 (KST)
> 최종 마감: **2026-08-20 23:59 KST**, 즉 8월 21일 00:00이 되기 전
> 목표: 최종 leaderboard 점수와 동일한 체크포인트를, 제출한 README만 따라 **VESSL GTX 1080에서 end-to-end 재현**할 수 있는 상태로 제출한다.

## 현재 확정된 범위

최종 선택만 보류하고 다음은 코드와 문서로 확정했다.

- 학습 schedule은 100 epoch 기준을 유지하되 stage 2를 **완료 epoch 89에서 안전하게 정지**한다.
- 후보 A `epoch89`: epoch 89 checkpoint를 byte-for-byte 복사한다.
- 후보 B `avg_80_85_89`: epoch 80/85/89의 floating/complex `state_dict`를 1:1:1 산술평균한다. 정수형 buffer가 다르면 생성 자체를 중단한다.
- 두 후보 모두 `<candidate>/checkpoints/best_model.pt`와 `candidate_manifest.json`을 가지며, 기존 candidate directory는 덮어쓰지 않는다.
- 두 후보 모두 동일한 변경 없는 `recon_eval.py`와 `utils/learning/test_part.py` 경로로 평가한다.
- 후보 생성, 공식 평가 log/score JSON, 최종 README 주입, VESSL preflight, package/SHA-256 절차를 자동화했다.
- `scripts/capture_submission_evidence.sh`로 실행 중인 학습을 건드리지 않고 commit/dirty diff, 환경·GPU·process, 로그·메타데이터, checkpoint hash를 timestamp snapshot으로 보존한다.
- 제출 PPT outline, 설명 영상 script, evidence index, 이메일 template를 준비했다.

현재 보류된 값은 아래뿐이다.

- 실제 epoch 80/85/89 checkpoint 파일과 각 SHA-256
- 두 후보의 VESSL 평가 결과
- 최종 candidate ID, checkpoint SHA-256, leaderboard 점수·시간
- 최종 선택값을 넣은 PPT/영상 파일과 운영진 양식의 정확한 이메일 외부 파일명

epoch 89가 저장되면 다음 명령부터 진행한다.

```bash
RESULT_ROOT=/root/result bash scripts/prepare_both_final_candidates.sh

FINAL_CANDIDATE_DIR=/root/result/final_candidates/epoch89 \
LEADERBOARD_PATH=/root/Data/leaderboard \
bash scripts/run_final_eval.sh

FINAL_CANDIDATE_DIR=/root/result/final_candidates/avg_80_85_89 \
LEADERBOARD_PATH=/root/Data/leaderboard \
bash scripts/run_final_eval.sh
```

최종 선택 뒤에는 해당 candidate의 `eval_metadata_*.json`을 사용해 `scripts/finalize_submission.py`를 실행하고 package를 만든다.

## 0. 먼저 고정할 원칙

- [ ] 최종 체크포인트는 실제로 VESSL GTX 1080에서 처음부터 학습된 결과물이어야 한다.
- [ ] 학습에는 제공된 `train`/`val`만 사용한다. 외부 데이터, 외부 pretrained weight, leaderboard 데이터·통계량을 학습/검증/파라미터 결정에 사용하지 않는다.
- [ ] inference 입력은 `kspace`와 mask 등 허용된 정보만 사용한다. `image/*.h5`의 label, annotation, GRAPPA image는 inference 입력으로 읽지 않는다.
- [ ] `recon_eval.py`는 수정하지 않는다. 영상 안내에 따라 이 파일이 호출하는 scoring 함수도 제출 직전에 공식본과 hash를 대조한다.
- [ ] `prep_volume()`에는 dtype/layout 변환, device 이동, `mask * kspace` 같은 순수 입력 전처리만 둔다. IFFT, sensitivity map, coil combine, GRAPPA, model forward 등 일부라도 reconstruction인 연산은 반드시 `recon_slice()` 안에 둔다.
- [ ] 최종 checkpoint 선택 규칙은 미리 고정한다. 현재 설계대로 **정해진 schedule에서 마감 전에 완료된 마지막 epoch (`submission-latest`)**를 사용하고, 여러 checkpoint의 public leaderboard 점수를 비교해 고르지 않는다.
- [ ] README대로 재학습·평가했을 때 `SSIM_full`, `SSIM_bbox`가 각각 소수점 넷째 자리까지 재현되어야 한다. 비트 단위 checkpoint 동일성이 목표는 아니지만, 실제 VESSL 학습을 입증할 로그와 환경 기록은 반드시 보존한다.

## 1. 영상에서 확인한 최종 제출물 5종

| 번호 | 제출물 | 완료 조건 |
|---|---|---|
| 1 | 최종 public leaderboard 점수 | 아래 2번 checkpoint를 VESSL의 공식 `recon_eval.py`로 평가한 `SSIM_full`, `SSIM_bbox`, 실제 `ms/slice`를 업로드한다. 2000 ms/slice를 넘어도 실제 값을 적는다. |
| 2 | 모델 checkpoint | 1번 점수를 낸 **바로 그 파일**을 제출하고 SHA-256을 남긴다. |
| 3 | 학습·평가 전체 코드 | checkpoint를 처음부터 만들 수 있는 train/data preprocessing 코드와 같은 점수를 내는 evaluation 코드를 모두 압축한다. `README.md`와 `requirements.txt`는 필수다. |
| 4 | 설명 영상 파일 + PPT | 실행법, 모델 개요, 주요 decision point, staged training과 checkpoint 선택 규칙을 설명한다. 수상작 공개 영상과 다른 제출용 설명 자료이므로 화려한 편집은 불필요하다. 영상은 수정 가능한 YouTube 링크가 아니라 **파일로 제출**한다. |
| 5 | VESSL 학습·재현성 증빙 | 원본 terminal/Codex 화면, timestamp 캡처, 전체 train log, `training_history.csv`, 초기 trajectory 반복 로그, 환경·GPU·git 기록, checkpoint hash를 제출한다. |

추가 일정/대상:

- leaderboard는 8월 13일 00:00부터 freeze 상태이지만 본인 점수는 확인 가능하며, 8월 21일부터 다시 공개된다.
- freeze 직전 15위 이내이거나 당시 15위 점수를 넘겼다면 최종 평가 대상으로 보고 제출한다.
- 순위와 무관하게 참여 인증서가 필요하면 최종 제출할 수 있다.
- 최종 평가는 public/private 데이터 **4:6**으로 반영되며 tie-breaker가 항상 포함된다. 2000 ms/slice 초과는 tie-breaker 0점이지만 별도 성능 penalty는 없다.
- 운영진의 evaluation/verification 기간은 8월 21–30일이고 수상 발표는 8월 31일 10:00 예정이다. 재학습이 10일 안에 끝나야 한다는 별도 제한은 없지만, 오래 걸리면 추가 검증 대상이 될 수 있으므로 예상 시간을 README에 명시한다.

## 2. 현재 저장소 진단

### 이미 준비된 것

- 최종 알고리즘은 `experiments/007_fivarnet_mraugment` → `experiments/008_fivarnet_cross_acc`의 2단계 FI-VarNet 흐름으로 정리되어 있다.
- `scripts/run_final_staged_reproduction.sh`가 fresh directory에서 007을 epoch 50까지 실행하고, epoch와 SHA-256을 확인한 뒤 008을 epoch 100까지 재개하도록 구성되어 있다.
- deterministic mode, `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`, hash 기반 MRAugment/cross-acceleration RNG, optimizer/scheduler/RNG checkpoint 복원이 구현되어 있다.
- `scripts/run_fivarnet_bbox.sh`가 `resolved_config.env`, `python_environment.txt`, `gpu_environment.txt`, `git_state.txt`, `run_metadata.json`, 전체 train log, GPU telemetry를 남긴다.
- `utils/learning/test_part.py`의 현재 `prep_volume()`은 k-space/mask 로딩만 하고, 모델 연산은 `recon_slice()`에서 수행하므로 #412/#419 원칙과 맞는다.
- `requirements.txt`는 핵심 패키지를 pin하고, `requirements-vessl.lock.txt`에는 실제 VESSL의 `torch==2.3.1+cu121`, CUDA/cuDNN 계열까지 기록되어 있다.

### 제출 전 반드시 닫아야 할 위험

1. **코드 상태가 아직 동결되지 않았다.** 현재 branch는 `exp/008-b1-cross-acc`, HEAD는 `c8ef5d6...`지만 9개 tracked 파일이 수정되어 있고 3개 파일이 untracked다. 특히 최종 orchestrator인 `scripts/run_final_staged_reproduction.sh`도 아직 untracked다.
2. launcher의 `git_state.txt`는 commit과 dirty 파일명만 남기고 실제 patch 내용은 보존하지 않는다. 진행 중인 VESSL 학습이 dirty tree에서 시작했다면, 지금 즉시 당시 코드 전체와 `git diff --binary HEAD`를 별도 보관해야 한다.
3. root `README.md`의 한국어가 mojibake 상태이고 제출 항목/실행법이 오래된 baseline 중심이다. 이 상태로는 조교가 코드를 수정하지 않고 실행하기 어렵다.
4. `scripts/vessl_entrypoint.sh`는 아직 final 007→008이 아니라 예전 `run_varnet_c6_long.sh`를 실행한다.
5. `recon_eval.sh`는 `test_Varnet`을 hard-code한다. final checkpoint용 공식 one-command evaluation entrypoint가 없다.
6. `run_recon_cpu_safe.sh`와 `recon_eval_cpuonly.py`는 보조 진단 도구일 뿐이다. 최종 점수와 inference time은 이 코드가 아니라 VESSL GPU의 변경 없는 `recon_eval.py`에서 얻어야 한다.
7. 현재 final profile은 train+val 전체를 학습에 사용하므로 validation log가 없다. 이는 허용되지만 제출 README에서 `training_history.csv`/train log를 validation log 대신 제출한다고 명시해야 한다.
8. 체크포인트, 코드 archive, 설명 영상, PPT, evidence 묶음 전체를 아우르는 manifest와 checksum 파일이 아직 없다.

## 3. 실행 우선순위 — 오늘 해야 할 순서

### P0-A. 진행 중인 VESSL 상태부터 보존한다

- [ ] 현재 학습 프로세스, epoch, 남은 예상 시간, GPU utilization, disk 여유를 확인한다. 학습을 중단하거나 재시작하지 않는다.
- [ ] VESSL terminal/Codex 화면과 화면에 보이는 clock을 한 프레임에 캡처한다. terminal을 이미 닫았다면 `tail -f`로 현재 log를 표시한 화면도 캡처한다.
- [ ] 현재 VESSL code tree를 읽기 전용 archive로 즉시 보존한다.
- [ ] 아래 자료를 별도 evidence 폴더에 복사한다.
  - `git rev-parse HEAD`, `git status --short`, `git diff --binary HEAD`
  - 실제 학습에 사용된 source/config/script 전체의 SHA-256 manifest
  - stage 1·2 train log와 GPU CSV
  - `resolved_config.env`, `run_metadata.json`, `git_state.txt`
  - `python_environment.txt`, `gpu_environment.txt`
  - `training_history.csv`, checkpoint 생성 시각, VESSL workload/activity 화면
- [ ] stage 1의 `checkpoint_epoch_0050.pt`와 기록된 hash `4ae5f62c...43b6`가 실제 VESSL 파일과 맞는지 다시 확인한다. 다르면 README의 기존 hash를 그대로 믿지 말고 실제 파일 기준으로 정정하고 경위를 남긴다.

### P0-B. final checkpoint를 한 개로 잠근다

- [ ] selection rule을 문서에 먼저 쓴다: `submission-latest`, 즉 leaderboard 성능과 무관하게 마감 전에 완료·저장된 마지막 epoch.
- [ ] 현재 학습이 마감 전에 epoch 100까지 끝나지 않으면 마지막으로 완전히 저장된 `best_model.pt`/`model.pt`의 epoch를 확인하고 그 epoch를 final로 선언한다.
- [ ] 학습 프로세스가 checkpoint를 교체하는 동안 파일을 직접 평가하지 않는다. 원자적 저장 완료 후 snapshot을 만들어 고정한다.
- [ ] snapshot 이름은 공백·한글 없이 예를 들어 `team39_fivarnet_final.pt`로 하고 SHA-256, byte size, stored epoch, source path, snapshot time을 manifest에 기록한다.
- [ ] checkpoint 내부 `args`, model state, optimizer/scheduler/RNG state를 CPU에서 load하여 손상 여부와 architecture를 확인한다.
- [ ] 이후 1번 leaderboard 점수, code README, 발표자료, 이메일에는 모두 이 checkpoint hash 하나만 사용한다.

### P0-C. 공식 VESSL 평가와 leaderboard 업로드

- [ ] `run_recon_cpu_safe.sh` 결과는 참고용으로만 보관하고 final 값으로 제출하지 않는다.
- [ ] final snapshot을 `../result/<FINAL_EXP_NAME>/checkpoints/best_model.pt`에 두어 현재 `test_part.py`의 load contract를 만족시킨다.
- [ ] 변경 없는 공식 `recon_eval.py`를 VESSL GTX 1080에서 실행한다.
- [ ] 전체 stdout/stderr, 시작/종료 시각, GPU/driver/runtime 정보, checkpoint SHA-256을 `evidence/official_eval/`에 저장한다.
- [ ] 출력된 `SSIM_full`, `SSIM_bbox`, **평균 ms/slice**를 leaderboard에 그대로 업로드한다. 2000 ms/slice가 넘어도 2000으로 바꾸지 않는다.
- [ ] 업로드 화면을 캡처하고 점수·시간·checkpoint hash를 `FINAL_SCORE.md`에 함께 기록한다.
- [ ] 최종 private 평가는 public/private 4:6이므로, 이 평가 뒤 public 점수를 보고 checkpoint나 알고리즘을 다시 선택하지 않는다.

### P0-D. 제출 코드가 조교 손에서 한 명령으로 도는 상태를 만든다

마감 직전에는 학습 로직/모델 구조를 다시 refactor하지 않는다. 실제 checkpoint를 만든 코드는 그대로 두고 **entrypoint와 문서만 명확하게 통합**한다.

- [x] `scripts/run_final_staged_reproduction.sh`를 final source snapshot에 포함하고 `FINAL_STAGE_STOP_EPOCH=89`를 지원한다. 007/008 config는 합치지 않고 실제 2단계 provenance를 보존한다.
- [x] `scripts/vessl_entrypoint.sh`가 `DATA_ROOT`, `RESULT_ROOT`, `FINAL_STAGE_STOP_EPOCH`를 받아 staged runner 하나만 호출하게 한다.
- [x] `scripts/run_final_eval.sh`가 candidate, `LEADERBOARD_PATH`, `GPU_NUM`을 받아 공식 `recon_eval.py`만 호출하고 log/score JSON을 저장한다. `recon_eval.py` 자체는 수정하지 않았다.
- [x] `scripts/vessl_entrypoint.sh`의 오래된 experiment 002 호출을 final staged runner 호출로 교체했다.
- [x] `scripts/verify_submission.py`가 다음을 fail-fast로 확인한다.
  - Python/Torch/NumPy/CUDA/cuDNN 버전
  - data directory 구조
  - checkpoint 존재, stored epoch, SHA-256
  - `recon_eval.py`/scoring 함수의 공식 hash
  - GPU가 GTX 1080이고 CUDA 사용 가능함
  - 핵심 모듈 import와 one-slice `recon_slice()` smoke test
- [x] `scripts/package_final_submission.sh`는 Linux에서 curated code archive를 만들고 checkpoint/PPT/video/evidence와 모든 파일의 SHA-256을 생성한다.
- [x] curated code archive는 `recon_eval_cpuonly.py`, `run_recon_cpu_safe.sh`, 과거 experiment와 불필요한 reconstruction/cache를 포함하지 않는다.

## 4. 제출용 README 재작성 목차

현재 root README를 다음 순서로 다시 쓴다. 조교가 별도 이메일 질문이나 소스 수정 없이 그대로 복사해 실행할 수 있는 수준이 완료 기준이다.

1. **제출 식별 정보**
   - 팀명/팀원, final checkpoint 파일명·SHA-256·stored epoch
   - leaderboard `SSIM_full`, `SSIM_bbox`, `ms/slice`
   - public score를 만든 VESSL 평가 log 경로
2. **한 줄 재현 명령**
   - 환경 설치
   - fresh end-to-end train
   - 공식 reconstruction + scoring
3. **정확한 환경**
   - Python 3.10.12
   - PyTorch 2.3.1+cu121
   - NumPy 1.24.4
   - CUDA runtime/cuDNN, GPU/driver 정보
   - `requirements.txt`와 실제 `pip freeze`의 역할 구분
4. **데이터 구조와 경로 인자**
   - `DATA_ROOT/train`, `DATA_ROOT/val`, `DATA_ROOT/leaderboard`
   - `RESULT_ROOT`; 절대 경로 hard-code 없이 env/CLI 예시 제공
5. **모델/학습 방법**
   - FI-VarNet 6 feature + 6 image cascades
   - batch 1, grad accumulation 4, bbox-aware SSIM, deterministic 설정
   - MRAugment와 cross-acceleration re-masking
   - train+val 결합, 외부 data/weight 없음
6. **staged 007→008 재현 절차**
   - stage 1: epoch 0–50
   - epoch-50 checkpoint 검증·복사·SHA-256
   - stage 2: epoch 50–target, optimizer/scheduler/RNG 전체 resume
   - 실제 checkpoint가 epoch 100 이전이면 문서와 command의 target도 실제와 일치시킨다.
7. **checkpoint 선택 규칙**
   - `submission-latest`; leaderboard를 model selection에 사용하지 않음
   - validation 미사용이므로 train loss log를 재현 증빙으로 제출함
8. **inference contract**
   - `test_part.py`에서 읽는 입력과 shape
   - `prep_volume()`과 `recon_slice()`의 경계
   - image/annotation/GRAPPA를 inference에 사용하지 않음
9. **예상 실행 시간과 저장공간**
   - 실제 VESSL epoch 시간으로 stage별/전체 예상 시간을 계산
   - verification 10일 제한은 없지만 장기 실행임을 명시
10. **재현성 및 알려진 변동**
    - 모든 seed/RNG 설정
    - early trajectory A/B 비교 결과
    - 완전 동일하지 않다면 원인, 관측 범위, 최종 점수 4자리 검증 결과
11. **파일 트리와 checkpoint routing 표**
    - 각 파일의 배치 위치, 적용 대상, 실행 순서를 표로 작성
12. **증빙 자료 index와 troubleshooting**
    - log/env/hash/screenshot 위치
    - OOM, resume, path, dependency 오류의 확인법만 기술하고 조교가 코드를 바꾸게 만들지 않는다.

## 5. 재현성 검증 계획

### 빠른 자동 검증

- [ ] `python -m compileall`로 제출 Python 파일 syntax 확인
- [ ] 모든 final shell script에 `bash -n`과 `shellcheck` 수행
- [ ] 현재 unit test 전체 실행; 특히 deterministic padding, MRAugment, cross-acceleration, staged resume, optimizer/scheduler/RNG round-trip을 통과시킨다.
- [ ] checkpoint를 CPU와 GTX 1080에서 각각 load하고 동일 architecture/state key가 구성되는지 확인한다.
- [ ] static audit로 `test_part.py`의 inference path가 `image_path`를 열거나 annotation/GRAPPA를 읽지 않는지 확인한다.

### 초기 loss trajectory 반복

- [ ] stage 1 fresh directory A/B에서 동일한 final config로 초기 epoch 또는 합의한 step 수를 각각 실행한다.
- [ ] stage 2는 같은 epoch-50 checkpoint의 복사본 A/B에서 한 epoch를 각각 재개한다.
- [ ] iteration별 loss, learning rate, sampled filename/slice, augmentation decision, cross-acceleration decision을 비교한다.
- [ ] 완전 일치하면 그 비교표와 log를 제출한다.
- [ ] 차이가 있으면 seed 고정 위치, CUDA/cuDNN/driver, DataLoader worker scheduling, resume 시 RNG 복원 여부를 분석한 `REPRODUCIBILITY_REPORT.md`를 작성한다. 차이를 숨기지 말고 최종 score 4자리 영향 검증을 함께 적는다.

### clean-room 제출물 검증

- [ ] VESSL의 새 빈 directory에서 최종 archive를 직접 풀어본다.
- [ ] `requirements.txt`만으로 환경을 구성하고 `verify_final.sh`를 통과시킨다.
- [ ] README의 copy-paste 명령만 사용해 one-slice smoke → checkpoint load → 공식 full `recon_eval.py` 순으로 실행한다.
- [ ] 새 환경의 최종 `SSIM_full`/`SSIM_bbox`가 제출값과 각각 소수점 넷째 자리까지 같은지 확인한다.
- [ ] Linux archive 안의 path가 `/` separator이고 역슬래시, `__MACOSX`, `.DS_Store`, 공백/한글 checkpoint명이 없는지 검사한다.

## 6. 권장 제출 묶음

실제 외부 파일명은 **운영진 이메일에 첨부된 발표자료의 양식**을 그대로 따른다. 영상 화면만 보고 보이지 않는 파일명을 추측하지 않는다.

```text
team39_final_submission/
├── MANIFEST.md
├── SHA256SUMS
├── checkpoint/
│   └── team39_fivarnet_final.pt
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── requirements-vessl.lock.txt
│   ├── train_final.sh
│   ├── eval_final.sh
│   ├── verify_final.sh
│   ├── train.py
│   ├── recon_eval.py
│   ├── experiments/007_fivarnet_mraugment/
│   ├── experiments/008_fivarnet_cross_acc/
│   ├── scripts/
│   └── utils/
├── evidence/
│   ├── final_training/
│   ├── stage_transition/
│   ├── repro_probe_A/
│   ├── repro_probe_B/
│   ├── official_eval/
│   └── screenshots/
├── presentation/
│   └── team39_final.pptx
└── video/
    └── team39_final.mp4
```

`MANIFEST.md`에는 최소한 아래를 기록한다.

- 파일별 역할, byte size, SHA-256
- final checkpoint ↔ official eval log ↔ leaderboard entry의 연결
- VESSL server/workspace 이름과 제출물 절대 경로
- Git commit + dirty patch hash 또는 code snapshot hash
- stage-1 checkpoint와 final checkpoint lineage
- README 재현 명령과 예상 소요시간

## 7. 설명 영상/PPT 구성

- [ ] 문제와 최종 점수 요약
- [ ] FI-VarNet 구조와 007→008 staged training 개요
- [ ] MRAugment/cross-acceleration을 선택한 근거와 외부 data/weight 미사용 확인
- [ ] checkpoint 선택 규칙과 public leaderboard 비사용 원칙
- [ ] 실제 `train_final.sh`, `eval_final.sh` 실행 화면
- [ ] `prep_volume()`/`recon_slice()` 규정 준수 설명
- [ ] 환경 pin, seed/RNG, stage resume SHA, 초기 trajectory A/B 결과
- [ ] 제출 checkpoint hash와 leaderboard score/time
- [ ] 알려진 한계와 재현에 걸리는 예상 시간

영상은 파일로 제출하며 PPT 원본도 함께 낸다. 화질보다 실행 정보가 읽히는지와 내용의 일치가 중요하다.

## 8. 전송 및 이메일 최종 점검

- [ ] 운영진이 보낸 이메일 첨부 발표자료에서 **정확한 수신 주소, 제목, 본문, 파일명 규칙**을 다시 복사한다.
- [ ] 양식의 `{...}`는 안내 placeholder이므로 중괄호를 남기지 않고 실제 값만 쓴다.
- [ ] 대용량 checkpoint/code/video는 cloud link와 VESSL workspace 양쪽에 두는 것이 가장 안전하다. 한쪽만 제출해도 된다는 안내는 있으나, 이중화한다.
- [ ] VESSL 제출 시 이메일 본문에 server/workspace 이름과 정확한 절대 경로를 적고 관련 없는 파일은 제출 경로 밖으로 정리한다.
- [ ] cloud link는 로그인/권한/만료 문제 없이 조교가 다운로드 가능한지 시크릿 창에서 확인한다.
- [ ] 이메일 본문에 각 링크/경로와 SHA-256을 적는다.
- [ ] 전송 후 보낸 메일, 첨부 목록, 링크 접근 화면, 제출 시각을 캡처한다.
- [ ] 최종 leaderboard 업로드, checkpoint hash, code archive hash, PPT/video hash가 서로 다른 버전을 가리키지 않는지 마지막으로 대조한다.

## 9. 완료 판정표

아래가 모두 `YES`일 때만 제출 완료로 본다.

| Gate | 질문 | YES/NO |
|---|---|---|
| G1 | final checkpoint가 실제 VESSL GTX 1080 end-to-end 학습 결과인가? | |
| G2 | final checkpoint SHA-256과 official eval에 사용한 파일 SHA-256이 같은가? | |
| G3 | 공식 `recon_eval.py`의 두 점수와 실제 ms/slice를 leaderboard에 올렸는가? | |
| G4 | inference가 image/annotation/GRAPPA를 입력으로 사용하지 않는가? | |
| G5 | 모든 reconstruction 연산이 `recon_slice()` 계측 구간 안에 있는가? | |
| G6 | README 한 명령으로 fresh training과 evaluation이 가능한가? | |
| G7 | Python/CUDA/PyTorch/NumPy와 모든 의존성이 정확히 기록되었는가? | |
| G8 | train log, 초기 trajectory A/B, env/GPU/git/hash/timestamp 증빙이 있는가? | |
| G9 | Linux clean extraction과 clean-room 실행을 통과했는가? | |
| G10 | checkpoint, code, PPT, video, evidence가 모두 manifest/checksum에 있는가? | |
| G11 | 이메일 양식·파일명은 첨부 발표자료를 그대로 따르고 placeholder를 제거했는가? | |
| G12 | 2026-08-20 23:59 KST 전에 전송과 leaderboard 업로드를 완료했는가? | |

## 10. 근거 자료

- [OVERVIEW - Final Submission [ESSENTIAL] — SNU FastMRI Challenge](https://www.youtube.com/watch?v=EfEVaIlTHog)
- [2026 Rule Book — issue #412](https://github.com/LISTatSNU/FastMRI_challenge/issues/412)
- [`prep_volume()` 허용/금지 경계 — issue #419](https://github.com/LISTatSNU/FastMRI_challenge/issues/419)
- [validation 미사용 시 train log 제출 — issue #340](https://github.com/LISTatSNU/FastMRI_challenge/issues/340)
- [재현 학습의 10일 제한 없음 — issue #442](https://github.com/LISTatSNU/FastMRI_challenge/issues/442)
- [GTX 1080 환경 차이와 4자리 재현 기준 — issue #455](https://github.com/LISTatSNU/FastMRI_challenge/issues/455)
- [정확한 파일명은 이메일 첨부 발표자료 기준 — issue #456](https://github.com/LISTatSNU/FastMRI_challenge/issues/456)
- [Codex 사용 시 timestamp/terminal 증빙 — issue #460](https://github.com/LISTatSNU/FastMRI_challenge/issues/460)
- [설명 영상은 수정 가능한 링크가 아닌 파일 제출 — issue #461](https://github.com/LISTatSNU/FastMRI_challenge/issues/461)

---

### 최종 의사결정

현재 시점에는 모델 로직을 더 크게 합치거나 재작성하는 것보다, **진행 중인 VESSL 실행과 코드 snapshot을 보존하고 → 마지막 완료 checkpoint를 잠그고 → 공식 평가·leaderboard 업로드 → 얇은 final wrapper/README/manifest/evidence를 완성**하는 순서가 가장 안전하다. 007/008 내부 구현은 그대로 보존하고, 제출자가 실행할 표면만 한 명령으로 통합한다.
