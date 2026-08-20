# 이메일 제출 템플릿

> 수신 주소, 제목, 외부 파일명은 운영진 이메일에 첨부된 발표자료를 그대로 복사한다. 영상에서 보이지 않은 값을 추측하지 않는다. 아래 중괄호는 전송 전에 모두 제거한다.

## 제목

`{운영진 첨부 발표자료의 정확한 제목 양식}`

## 본문

안녕하세요. `{팀명}`의 `{팀원}`입니다.

2026 SNU FastMRI Challenge 최종 제출물을 전달드립니다.

- Final candidate: `{candidate_id}`
- Checkpoint SHA-256: `{sha256}`
- SSIM_full: `{score}`
- SSIM_bbox: `{score}`
- Reconstruction time: `{ms/slice}` ms/slice
- Cloud download: `{권한과 만료를 확인한 링크}`
- VESSL workspace/server: `{정확한 이름}`
- VESSL absolute path: `{정확한 절대 경로}`
- Code archive SHA-256: `{sha256}`
- PPT SHA-256: `{sha256}`
- Video SHA-256: `{sha256}`

첨부/링크/서버 경로에는 checkpoint, 전체 train/evaluation code, README, requirements, PPT, 설명 영상 파일, 재현성 증빙, SHA256SUMS가 포함되어 있습니다.

감사합니다.

`{팀명 / 팀원 / 연락처}`

## 보내기 전

- [ ] 모든 `{...}` 제거
- [ ] 운영진 발표자료의 파일명 규칙 적용
- [ ] 시크릿 창에서 cloud link 다운로드 확인
- [ ] VESSL 경로 확인
- [ ] SHA-256 재검증
- [ ] 보낸 시각과 첨부 목록 캡처
