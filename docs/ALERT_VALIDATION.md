# 객체 필터와 거리 경고 기능 검증

2026-10-16 (Asia/Seoul), 기존 프로젝트를 수정한 뒤 실행한 기록입니다.
변경 전 결과는 [기존 검증 기록](VALIDATION.md)에 보존했습니다.

## 변경 파일

| 파일 | 변경 내용 |
| --- | --- |
| `config/settings.toml`, `app/settings.py` | 10종 ID·모델 이름과 경고 설정, TOML 읽기·유효성 검사 |
| `app/config.py`, `app/detector.py`, `app/pipeline.py` | 설정의 classes를 YOLO에 전달, 실제 model.names 대조, 진단에 목록 제공 |
| `app/alerts.py` | 클래스별 IoU 연결, 이동 중앙값, 중앙 겹침, 단계 확인·히스테리시스, 반복 간격 |
| `app/sound.py` | Windows 비동기 Beep, 단일 대기 슬롯, Danger 우선, 음소거·실패·종료 처리 |
| `app/main.py`, `app/visualization.py` | 표시 거리 제한, M 키, 상태 표시, 오래된 결과 취소, 통신 장애 재시도 |
| `Dockerfile`, `compose.yaml` | 설정 COPY와 읽기 전용 바인드 마운트. 소리는 호스트에서만 실행 |
| `tools/check_sound.py` | 모델·웹캠 없이 두 경고 패턴을 재생하는 수동 검사 |
| 새 테스트 6개 파일 | `test_detector`, `test_alerts`, `test_sound`, `test_settings`, `test_display_filter`, `test_alert_client` |
| `README.md`, 이 문서와 `alert_validation/` | 변경 설명, 재현 명령과 실제 실행 자료 |

YOLO·Metric Depth 구현과 가중치, 서버의 JPEG/JSON API 계약, 기존 가상환경·설치 절차를 유지했습니다. 새 패키지 설치는 필요하지 않습니다.

## 자동 테스트

실제 CUDA·웹캠 검사를 활성화한 전체 실행: **211 passed**, upstream DeprecationWarning 1건, **실패 없음**.
Ruff, Python compileall, pip check도 통과했습니다. 기본 테스트 실행에서는 실제 하드웨어 검사 2건을 건너뜁니다.

| 요구 항목 | 검증한 내용 |
| --- | --- |
| 지정 클래스만 탐지 | 정확한 10개 `classes` 전달, model.names 불일치 거절, 설정으로 다른 COCO 클래스 선택 |
| 3m 표시 제한 | 3.0m 포함, 3m 초과·None·0·NaN·Inf 제외, 화면 객체 번호 보존 |
| Warning / Danger | 2m·1m 경계, CENTER와 겹침 후보, 가장 가까운 유효 중앙 객체 우선 |
| 거리 안정화 | 최근 5개 거리 중앙값, 박스 순서 변경, 클래스·객체별 이력 분리, 오래된 이력 폐기 |
| 반복 제한 | Warning 3초·Danger 1.5초, 단계 재진입 후 반복 억제, Warning→Danger 상승 시 대기 우회 |
| 단계 안정화 | 확인 프레임, 0.15m 히스테리시스, 무효·오래된 결과는 Unknown |
| M 키 | 실제 SoundPlayer setter를 사용하는 가짜 GUI 입력으로 OFF/ON과 화면 상태 확인 |
| 비동기 사운드 | 재생 콜백을 막아도 play 호출이 반환됨, 단일 대기 항목, 우선순위·취소·종료 |
| 재생 실패 | 예외를 일으키는 재생기에서도 worker 생존, 오류 로그 제한 |
| 통신·깊이 실패 | 통신 실패 7회 후 복구, 기본 무제한 재시도, 이전 거리 재사용·소리 반복 없음 |
| 프레임 만료 | 다음 추론이 멈춘 동안에도 캡처 기준 timeout 이후 소리와 박스 폐기 |

단위·통합 경고 테스트는 가짜 거리·가상 시각·가짜 카메라·GUI·소리를 사용하므로 스피커나 웹캠이 없어도 실행할 수 있습니다.
`tests/test_hardware.py`만 실제 CUDA와 카메라를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:RUN_HARDWARE_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_HARDWARE_TESTS
.\.venv\Scripts\python.exe -m ruff check app server tools tests
.\.venv\Scripts\python.exe -m compileall -q app server tools
.\.venv\Scripts\python.exe -m pip check
```

## 실제 모델과 API

RTX 3080, Python 3.11.15, PyTorch 2.6.0+cu124 환경에서 두 모델의 `cuda:0` 배치를 다시 확인했습니다.
실제로 로드한 YOLO11n의 model.names는 다음 설정과 모두 일치했습니다.

```text
0 person, 1 bicycle, 13 bench, 16 dog, 24 backpack,
28 suitcase, 56 chair, 57 couch, 58 potted plant, 60 dining table
```

기존 공개 COCO 실내 이미지로 전체 GPU 추론을 반복했습니다.
변경 전 탐지되던 고양이·리모컨은 제외되고 허용된 **소파(class 57) 하나만 반환**됐습니다.
원본 640×480, YOLO 640 / Depth 518, 워밍업 3회 후 10회 평균은 **69.42ms**였습니다.
카메라·GUI·HTTP를 제외한 반복 정지 이미지 측정이며 웹캠 FPS나 성능 보장으로 해석하지 않습니다.
원시 자료는 [model_report.json](alert_validation/model_report.json)에 있습니다.

Windows에서 직접 실행한 localhost FastAPI 서버로 같은 이미지를 전달했습니다.
PyTorch가 없는 `.venv-client`에서 받은 소파 깊이는 약 **1.63m**였으며,
클라이언트 정책은 `Unknown → Warning → Warning`으로 바뀌었습니다.
소리는 이 검사에서는 껐고, 전송 결과와 진단은 [alert_api_report.json](alert_validation/alert_api_report.json)에 저장했습니다.
공개 이미지의 실제 거리를 모르므로 **이 결과는 거리 정확도 검증이 아닙니다**.
실제 이미지에서 Danger가 되는 물체를 물리적으로 배치하는 검증도 수행하지 않았습니다. Danger의 단계·소리 요청은 가상 데이터로 검증했습니다.

## 실제 웹캠과 Windows 소리

카메라 0, DirectShow, 640×480에서 다음 실행이 정상 종료됐습니다.

| 모드 | 성공 결과 | 최근 완료 FPS | 첫 결과 제외 평균 클라이언트 추론 |
| --- | ---: | ---: | ---: |
| 로컬, headless, mute | 30 | 13.75 | 70.59ms |
| localhost API, 경량 클라이언트 OpenCV 창, mute | 10 | 12.00 | 70.89ms |

두 웹캠 실행 장면에서는 허용한 클래스가 탐지되지 않았습니다. 이 수치는 영상 수집·빈 탐지 처리·화면·통신 실행 검증이며 실제 가까운 장애물의 경고 검증은 아닙니다.
짧은 별도 실행이므로 모드 성능을 비교하거나 목표 FPS로 일반화하지 않습니다.
원시 자료: [로컬 CSV](alert_validation/webcam_alert_local.csv), [원격 CSV](alert_validation/webcam_alert_remote.csv), [집계 JSON](alert_validation/benchmark_summary.json).

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --headless --mute --max-frames 30 --max-seconds 40 --benchmark-csv .\outputs\webcam_alert_local.csv
# 별도 PowerShell에서 서버 실행:
.\.venv\Scripts\python.exe -m uvicorn server.api:app --host 127.0.0.1 --port 8000 --workers 1
# 클라이언트 PowerShell:
.\.venv-client\Scripts\python.exe -m app.main --mode remote --mute --max-frames 10 --max-seconds 20 --benchmark-csv .\outputs\webcam_alert_remote.csv
```

실제 Windows에서 다음 명령으로 두 패턴 재생을 요청했고, 재생 오류 로그 없이 종료 코드 0을 확인했습니다.
물리적 스피커의 음량·가청성·두 패턴 구분을 사람이 청취해 확인한 것은 아닙니다.

```powershell
.\.venv-client\Scripts\python.exe -m tools.check_sound
```

Q/M 키의 물리적 키보드 조작은 가짜 GUI 키 입력 테스트로 대체했습니다. 직접 실행해 OpenCV 창에 포커스를 주고 Q/M을 확인할 수 있습니다.

## 남은 항목

- Docker 미설치: 이미지 빌드, 컨테이너 GPU 전달과 설정 마운트의 실제 실행은 미검증입니다. Compose YAML, GPU 예약·localhost 바인딩·설정 경로는 정적으로 검사했습니다.
- 실제 스피커 청취, 실내 장애물에 접근하며 수행하는 Warning/Danger 전환 검증은 남아 있습니다.
- 줄자 실측, 다른 조명·렌즈·가림·교차 물체에서의 거리 오차와 IoU 연결 안정성은 별도 검증이 필요합니다.
- 1m/2m와 색상·음 패턴은 프로토타입 설정이며 실제 보행 안전거리가 아닙니다. 필터는 미탐지 위험 요소까지 찾아주는 기능이 아닙니다.

실행에 사용한 설정 사본은 [settings.toml](alert_validation/settings.toml)입니다. 실제 설정은 프로젝트의 `config/settings.toml`을 편집합니다.
Docker 서버에는 같은 디렉터리가 읽기 전용 마운트되며 편집 후 서버를 재시작해야 탐지 목록이 적용됩니다.
