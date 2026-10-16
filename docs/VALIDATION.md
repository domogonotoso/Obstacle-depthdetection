# 실행 검증 기록

이 문서는 객체 10종 필터·경고 기능 추가 전의 기록입니다. 변경 후 결과는 [객체 필터·경고 기능 검증](ALERT_VALIDATION.md)을 확인하세요.

검증일: **2026-10-16 (Asia/Seoul)**. 이 작업 폴더에서 실제 명령을 실행한 결과입니다.
공식 모델의 기능·장치 실행·처리 시간을 확인했으며, 줄자로 잰 실제 거리의 정확도는 아직 검증하지 않았습니다.

## 실행 환경

| 항목 | 확인한 값 |
| --- | --- |
| OS | Windows 11, PowerShell 5.1 |
| Python | 프로젝트 전용 CPython 3.11.15 (`.venv`) |
| GPU | NVIDIA GeForce RTX 3080, 약 10GB |
| NVIDIA 드라이버 | 595.95, nvidia-smi 표시 CUDA 지원 수준 13.2 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| PyTorch CUDA 런타임 | 12.4 |
| 모델 | Ultralytics 8.3.221 / YOLO11n + Metric Depth Anything V2 Small Hypersim |
| 웹캠 | 카메라 인덱스 0, DirectShow, 실제 640×480 BGR |
| 입력 크기 | YOLO 640, Depth 518, FP32 |
| Docker | 실행 파일 없음. 설치·시스템 변경을 수행하지 않음 |

Python은 기존 `uv`로 프로젝트 내부 `.runtime/python`에 준비했습니다. 시스템 Python 등록이나 PATH 변경은 하지 않았습니다.
설치 패키지 기록은 [tested-environment.txt](validation/tested-environment.txt), 실제 CUDA 연산 결과는 [gpu.json](validation/gpu.json)에 있습니다.

## 테스트 결과

| 검증 | 결과 | 범위 |
| --- | --- | --- |
| 기본 pytest | **111 passed, 2 skipped** | 하드웨어를 쓰지 않는 결합·방향·오류·CSV·API·클라이언트·캐시 테스트 |
| 실제 하드웨어를 활성화한 전체 pytest | **113 passed** | 위 테스트 + 실제 CUDA 텐서 연산 + 실제 웹캠/두 모델 결합 |
| Ruff | **All checks passed** | app, server, tools, tests |
| compileall | 통과 | app, server, tools, 공식 third_party 소스 |
| pip check | **No broken requirements found** | 설치 의존성 검사 |
| YOLO 단독 실제 추론 | 통과 | 공개 실내 이미지의 고양이 2개·리모컨·소파 탐지 |
| Metric Depth 단독 실제 추론 | 통과 | 원본 480×640, NaN/Inf 없음, 값이 0~20m 범위 안 |
| 실제 GPU 결합 추론 | 통과 | 원본 좌표와 깊이맵 정렬, 객체별 중앙 ROI 거리·방향 출력 |
| 실제 웹캠 로컬 실행 | 통과 | 30개 결과, 처리 시간 CSV 기록 |
| 실제 웹캠 localhost API 실행 | 통과 | 30개 결과, JPEG 요청과 JSON 응답, 처리 시간 CSV 기록 |
| PyTorch 없는 호스트 클라이언트 | 통과 | 별도 `.venv-client`의 NumPy/OpenCV/Requests만으로 진단·실제 웹캠 GUI 5개 결과 처리 |
| 실제 OpenCV GUI 실행 | 통과 | 15개 결과 후 실행 제한에 따른 자동 종료, 예외 없음 |
| Q 키와 자원 해제 | 단위 테스트 통과 | 가짜 GUI Q 입력으로 종료·카메라/백엔드/창 정리 확인. 사람이 실제 키를 누르는 검증은 미수행 |
| Compose 구성 정적 검사 | 통과 | YAML 파싱, localhost 포트 바인딩과 NVIDIA GPU 예약 확인 |
| Docker 빌드·컨테이너 GPU | **미검증** | Docker 미설치. 아래 재현 절차 필요 |
| 줄자 기준 거리 정확도 | **미검증** | 실측 장비·동일 객체·거리별 비교 필요. 가상의 정확도 결과는 작성하지 않음 |

전체 테스트에는 Starlette TestClient의 AnyIO `BlockingPortal` 별칭과 관련한 upstream DeprecationWarning 1건이 있습니다. 테스트 실패는 없습니다.
API 테스트는 잘못된 JPEG, 크기 제한, 서버 오류, HTTP 429, health 응답성, 요청 취소 후 GPU 작업이 끝날 때까지 슬롯 유지 등을 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:RUN_HARDWARE_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_HARDWARE_TESTS
.\.venv\Scripts\python.exe -m ruff check app server tools tests
.\.venv\Scripts\python.exe -m compileall -q app server tools third_party
.\.venv\Scripts\python.exe -m pip check
```

## 정지 이미지 GPU 처리 시간

입력은 Hugging Face 공식 문서에서 제공하는 [공개 COCO 실내 이미지](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png)입니다.
원본 COCO 이미지 서버의 TLS 인증서 검증이 실패하여 HTTPS 검증을 끄지 않고 이 공식 문서 사본을 사용했습니다.
이미지 크기는 640×480이며, 각 모델을 3번 워밍업한 뒤 같은 이미지를 10번 처리했습니다. CUDA 연산 전후 동기화로 측정했습니다.

| 구성 | 평균 시간 | 중앙 시간 | 측정된 순차 처리량 |
| --- | ---: | ---: | ---: |
| YOLO11n 단독 | 23.93 ms | 23.83 ms | 41.80 FPS |
| Metric Hypersim Small 단독 | 53.00 ms | 52.77 ms | 18.87 FPS |
| YOLO + Depth + 결합 | 84.37 ms | 84.64 ms | 11.85 FPS |

이는 정지 이미지 반복 추론 수치이며 **웹캠 FPS가 아닙니다**. 캡처·HTTP·JPEG·GUI 시간은 제외합니다.
단독 깊이 출력의 최소/중앙/최대는 약 1.161 / 1.518 / 1.808m였으나, 이 이미지의 실측 거리를 모르므로 정확도 근거로 사용하지 않습니다.
원시 실행 시간과 결과는 [YOLO](validation/yolo_report.json), [Depth](validation/depth_report.json), [결합](validation/all_report.json)에 있습니다.
검토용 오버레이와 깊이맵은 로컬 `outputs/smoke/`에 저장했습니다. 웹캠 영상이나 사진은 저장하지 않았습니다.

```powershell
.\.venv\Scripts\python.exe -m tools.smoke_models --image .\outputs\fixtures\coco_indoor.png --component yolo --warmup 3 --runs 10 --output-dir .\outputs\smoke
.\.venv\Scripts\python.exe -m tools.smoke_models --image .\outputs\fixtures\coco_indoor.png --component depth --warmup 3 --runs 10 --output-dir .\outputs\smoke
.\.venv\Scripts\python.exe -m tools.smoke_models --image .\outputs\fixtures\coco_indoor.png --component all --warmup 3 --runs 10 --output-dir .\outputs\smoke
```

## 실제 웹캠 측정

웹캠은 요청·실제 해상도 모두 640×480이었습니다. 아래 결과는 짧은 실행의 관측값이며 장시간 안정성·목표 FPS를 보장하지 않습니다.
원격 모드의 서버는 **Windows에서 직접 실행한 FastAPI 서버**입니다. 컨테이너 성능으로 해석하면 안 됩니다.

| 모드 | 결과 수 | 최근 완료 FPS | 첫 추론 | 첫 결과를 제외한 평균 클라이언트 추론 | 첫 결과를 제외한 캡처→완료 지연 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 로컬, 화면 없음 | 30 | 11.67 | 1016 ms | 80.76 ms | 98.03 ms |
| localhost API, 화면 없음 | 30 | 11.67 | 797 ms | 82.45 ms | 101.83 ms |
| 로컬, OpenCV 창 | 15 | 12.11 | 953 ms | 76.93 ms | 93.79 ms |

각 행은 별도 실행이므로 미세한 차이로 모드 간 성능 우열을 판단하지 않습니다. 첫 결과에는 초기 추론 비용이 반영됩니다.
FPS는 완료된 결과 간 간격을 사용해 첫 추론 대기 시간과 모델 초기화 시간을 포함하지 않습니다.
캡처 시각은 `read()` 완료 후의 호스트 시각이므로 카메라·드라이버 내부 지연은 알 수 없습니다.
원격 클라이언트 추론 시간에는 JPEG 인코딩과 HTTP 왕복이 포함됩니다.

원시 자료: [로컬 CSV](validation/webcam_local.csv), [API CSV](validation/webcam_remote.csv), [GUI CSV](validation/webcam_gui.csv), [집계 JSON](validation/benchmark_summary.json).

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --headless --max-frames 30 --max-seconds 40 --benchmark-csv .\outputs\webcam_local.csv
# 별도 PowerShell에서 서버를 먼저 실행:
.\.venv\Scripts\python.exe -m uvicorn server.api:app --host 127.0.0.1 --port 8000 --workers 1
# 클라이언트 PowerShell:
.\.venv\Scripts\python.exe -m app.main --mode remote --headless --max-frames 30 --max-seconds 40 --benchmark-csv .\outputs\webcam_remote.csv
.\.venv\Scripts\python.exe -m app.main --mode local --max-frames 15 --max-seconds 20 --benchmark-csv .\outputs\webcam_gui.csv
```

CSV는 추가 기록 방식입니다. 같은 경로로 다시 측정하면 이전 결과와 합쳐지므로 새로운 측정에는 다른 파일명을 사용하는 편이 명확합니다.

## 남은 검증과 재현 방법

Docker Desktop·WSL2를 준비한 뒤 아래 명령을 실행해야 컨테이너 검증이 완료됩니다. 설치 과정의 시스템 변경은 README에 별도로 안내했습니다.
GPU 진단에서는 `cuda_available`, `cuda_tensor_test`가 true이고, 두 모델 장치가 `cuda:0`이어야 합니다.

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
.\.venv-client\Scripts\python.exe -m tools.diagnose --server http://127.0.0.1:8000 --require-cuda
.\.venv-client\Scripts\python.exe -m app.main --mode remote
docker compose down
```

줄자 비교는 README의 `--truth-m`, `--measurement-csv`, `--object-index`와 S 키를 사용합니다.
여러 거리·조명·물체·화면 위치에서 반복하고, 실제로 선택한 객체와 측정한 표면을 일치시켜야 합니다.
카메라 광축 깊이와 줄자로 잰 직선 거리는 화면 가장자리에서 특히 달라지므로 비교 방법을 구분하세요.
현재 숫자는 추정값이며, 색상 기준은 임의의 표시 설정입니다.
