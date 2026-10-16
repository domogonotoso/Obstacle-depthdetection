# 설치 및 실행 가이드

모든 명령은 프로젝트 최상위 폴더에서 실행하는 **Windows PowerShell** 기준입니다.
VS Code 터미널과 별도 PowerShell 창을 모두 사용할 수 있습니다. VS Code는 선택한 편집기이며 실행에 필수는 아닙니다.
처음에는 Docker 없이 로컬 모드로 시작할 수 있습니다. Conda도 필요하지 않습니다.

- [로컬 환경 설치](#local-install)
- [웹캠 실행과 조작](#local-run)
- [설정 변경](#configuration)
- [Docker 실행](#docker)
- [Docker 없이 API 실행](#local-api)
- [거리·성능 CSV 및 테스트](#validation)
- [문제 해결](#troubleshooting)

<a id="local-install"></a>

## 로컬 환경 설치

- Windows 11, 내장 또는 USB 웹캠
- [Python 3.11 64비트](https://www.python.org/downloads/windows/)와 Python Launcher
- CUDA를 지원하는 NVIDIA GPU와 Windows NVIDIA 드라이버. 검증 장비는 RTX 3080입니다.
- 최초 패키지·모델 다운로드를 위한 인터넷 연결

Python·드라이버 설치는 PC의 시스템 환경을 변경하며 관리자 권한 또는 재부팅이 필요할 수 있습니다.
아래 명령에서 폴더 경로는 실제 프로젝트 위치로 바꾸세요.

```powershell
Set-Location 'C:\경로\Obstacle_Detection'
Get-Location
py -3.11 --version
nvidia-smi
```

### 가상환경과 라이브러리

새 PC에서는 `.venv`를 새로 만듭니다. 다른 PC의 `.venv` 폴더를 복사해서 사용하지 마세요.
현재 PC에서 이미 설치를 마쳤다면 이 단계를 반복하지 않고 [웹캠 실행](#local-run)으로 이동합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tools.diagnose --require-cuda
```

진단의 `cuda_available`과 `cuda_tensor_test`가 모두 `true`인지 확인합니다.
기본 장치는 `cuda:0`이며 CUDA 실패 시 CPU로 자동 전환하지 않습니다.
CUDA 라이브러리는 위 PyTorch wheel에 포함되므로 이 실행을 위해 CUDA Toolkit을 별도로 설치할 필요는 없습니다.
`nvidia-smi`의 CUDA 표시와 PyTorch wheel의 CUDA 버전은 의미가 달라 숫자가 반드시 같을 필요는 없습니다.

가상환경을 활성화하는 명령은 사용하지 않습니다. 항상 `.venv\Scripts\python.exe`를 직접 실행하므로 PowerShell 실행 정책 변경도 필요하지 않습니다.

### 공식 모델과 캐시

첫 추론 초기화에서 다운로드하거나 다음 명령으로 미리 준비할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m tools.download_models --model-dir .\models
```

| 용도 | 모델 | 캐시 파일 |
| --- | --- | --- |
| 객체 탐지 | Ultralytics YOLO11n | `models/yolo11n.pt` |
| 실내 깊이 추정 | Metric Depth Anything V2 Small / Hypersim | `models/depth_anything_v2_metric_hypersim_vits.pth` |

가중치는 공식 Ultralytics GitHub 릴리스와 Depth Anything Hugging Face 저장소에서 받습니다.
[다운로드 구현](../app/model_assets.py)에 고정한 SHA256으로 새 다운로드와 기존 캐시를 검증합니다.
정상 캐시가 있으면 다시 다운로드하지 않습니다. 프로젝트를 옮길 때 `models/`의 가중치는 재사용할 수 있고 동일하게 해시 검증을 받습니다.
`models/`는 Git과 Docker 이미지에 포함하지 않으며, Docker에서는 호스트의 같은 폴더를 마운트합니다.
모델 구현·가중치의 출처와 라이선스는 [THIRD_PARTY.md](../THIRD_PARTY.md)를 확인하세요.

<a id="local-run"></a>

## 웹캠 실행과 조작

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --camera 0 --width 640 --height 480 --device cuda:0
```

모델 로딩 후 별도의 OpenCV 창이 열립니다. Windows에서 웹캠·GPU 추론·화면·경고음을 모두 처리합니다.
소리 없이 시작하려면 `--mute`를 추가합니다.

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --camera 0 --mute
```

**OpenCV 영상 창을 클릭한 뒤** 키를 누르세요.

| 키 | 동작 |
| --- | --- |
| `Q` | 종료, 카메라·추론·소리 자원 정리 |
| `M` | 경고음 ON/OFF 전환 |
| `S` | 거리 CSV 기록 옵션을 지정했을 때 현재 선택 객체의 측정값 저장 |

화면에는 객체 박스·이름·신뢰도·`LEFT/CENTER/RIGHT`·`Est. 1.50m`, 완료 결과 기준 FPS와 프레임 지연이 표시됩니다.
`Sound ON/OFF`, `Normal/Warning/Danger/Unknown`, 무효 깊이 개수도 확인할 수 있습니다.
신뢰도는 객체 인식의 신뢰도이며 거리 정확도의 신뢰도는 아닙니다.
유효한 추정 거리가 표시 한도 이내인 객체만 박스와 라벨을 그립니다.

### 실행 옵션

```powershell
.\.venv\Scripts\python.exe -m app.main --help
.\.venv\Scripts\python.exe -m app.main --mode local --diagnostics
```

| 옵션 | 기본값 / 용도 |
| --- | --- |
| `--camera`, `--camera-backend` | `0`, `auto`; 다른 카메라나 `dshow`, `msmf` 선택 |
| `--width`, `--height`, `--camera-fps` | `640`, `480`, `30`; 카메라 요청값이며 지원 여부는 장치에 따름 |
| `--yolo-size`, `--depth-size` | `640`, `518`; 로컬 모델 입력 크기 |
| `--confidence` | `0.35`; 로컬 YOLO 신뢰도 하한 |
| `--central-ratio` | `0.5`; 거리 샘플링 박스 중앙 가로·세로 비율 |
| `--min-valid-fraction` | `0.2`; 샘플 영역의 최소 유효 깊이 비율 |
| `--min-depth-m`, `--max-depth-m` | `0.05`, `20.0`; 유효 깊이 범위 |
| `--hide-zones` | 방향 구분선 숨김 |
| `--headless` | OpenCV 창 없이 실행; 키 조작 불가 |
| `--max-frames`, `--max-seconds` | `0`은 무제한; 자동 종료용 |
| `--request-timeout`, `--jpeg-quality` | 원격 읽기 timeout `30`초·연결 timeout 최대 `3`초, JPEG 품질 `90` |
| `--failure-limit` | `0`은 추론·통신 실패 후 계속 재시도 |

`--yolo-size`는 32의 배수(32~2048), `--depth-size`는 14의 배수(56~1036)여야 합니다.
낮은 입력 크기는 처리시간과 정확도에 영향을 줄 수 있으므로 실제 영상으로 비교하세요.

<a id="configuration"></a>

## 설정 변경

기본 파일은 [config/settings.toml](../config/settings.toml)입니다.
설정은 시작할 때 읽으므로 변경 후 프로그램을 재시작합니다. 다른 파일도 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --config .\config\settings.toml
```

탐지 대상은 아래 COCO ID와 실제 모델 이름을 사용합니다. 초기화 시 `model.names`와 대조하며 불일치하면 오류를 표시합니다.

```toml
[detection.classes]
"0" = "person"
"1" = "bicycle"
"13" = "bench"
"16" = "dog"
"24" = "backpack"
"28" = "suitcase"
"56" = "chair"
"57" = "couch"
"58" = "potted plant"
"60" = "dining table"

[alerts]
display_max_m = 3.0
warning_m = 2.0
danger_m = 1.0
warning_repeat_s = 3.0
danger_repeat_s = 1.5
sound_enabled = true
smoothing_window = 5
confirm_frames = 2
hysteresis_m = 0.15
center_overlap_ratio = 0.35
track_iou_threshold = 0.3
track_ttl_s = 1.0
result_timeout_s = 1.0
```

`display_max_m`는 표시 한도입니다. `warning_m` 이하에서는 Warning, `danger_m` 이하에서는 Danger 후보가 됩니다.
`warning_repeat_s`와 `danger_repeat_s`는 같은 단계의 소리 반복 최소 간격이며, 확정된 Warning→Danger 상승은 대기를 우회합니다.
`smoothing_window`는 객체별 이동 중앙값 이력 수, `confirm_frames`는 단계 변경 확인에 필요한 새 추론 결과 수입니다.
`hysteresis_m`는 단계가 낮아질 때 적용하는 거리 여유입니다.

객체 중심이 CENTER이거나, 중앙 1/3과 겹치는 가로 폭이 객체 폭의 `center_overlap_ratio` 이상이면 중앙 후보입니다.
같은 클래스의 박스를 `track_iou_threshold`로 임시 연결하며 `track_ttl_s` 이후 이력을 폐기합니다. 영구 추적 ID는 아닙니다.
`result_timeout_s`보다 오래된 결과나 실패한 결과로 경고를 유지하지 않습니다.
Warning은 660Hz 120ms 한 번, Danger는 1000Hz 140ms 두 번(중간 60ms)이며 Windows의 별도 소리 스레드에서 재생합니다.
이 거리·반복 설정은 프로토타입 기준이며 실제 보행 안전거리가 아닙니다.

<a id="docker"></a>

## Docker 실행

Docker는 선택 사항입니다. 컨테이너가 GPU 추론을 담당하고 Windows 클라이언트가 웹캠·화면·경고음을 처리합니다.
프로젝트의 Docker 빌드·컨테이너 GPU 전달·설정 마운트 실제 실행은 **미검증**입니다. 아래는 재현 및 확인 절차입니다.

### Windows 사전 준비

[Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/), WSL2 기반 Linux 컨테이너,
[Docker Desktop GPU 지원](https://docs.docker.com/desktop/features/gpu/)에 맞는 NVIDIA 드라이버가 필요합니다.
설치·WSL 활성화는 시스템 변경이며 관리자 권한 또는 재부팅이 필요할 수 있습니다.
WSL이 없다면 **관리자 PowerShell**에서 설치한 뒤 재부팅 안내를 따릅니다.

```powershell
wsl --install
```

이미 준비돼 있다면 일반 PowerShell에서 확인합니다. Docker Desktop은 실행 중이어야 합니다.

```powershell
wsl --status
wsl --list --verbose
docker version
docker compose version
```

Docker Desktop에서 WSL2 엔진과 필요한 WSL 통합을 활성화하고 Linux 컨테이너 모드를 사용합니다.

### 추론 서버와 GPU 진단

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
docker compose config
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail 100 inference
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/diagnostics
```

첫 시작에서는 모델 다운로드·초기화가 끝날 때까지 기다립니다.
`/diagnostics`의 `cuda_available`, `cuda_tensor_test`, `detector_device`, `depth_device`로 GPU 연산과 두 모델의 CUDA 배치를 확인합니다.
[compose.yaml](../compose.yaml)은 NVIDIA GPU를 예약하고 API를 호스트 `127.0.0.1:8000`에만 공개합니다.
`./models:/models`로 가중치를 보존하며 `docker compose down` 후에도 호스트 캐시는 유지됩니다.

### Windows 경량 클라이언트

PyTorch 없이 웹캠·GUI·소리·HTTP만 실행할 수 있습니다.

```powershell
py -3.11 -m venv .venv-client
.\.venv-client\Scripts\python.exe -m pip install --upgrade pip
.\.venv-client\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv-client\Scripts\python.exe -m tools.diagnose --server http://127.0.0.1:8000 --require-cuda
.\.venv-client\Scripts\python.exe -m app.main --mode remote --server http://127.0.0.1:8000 --camera 0 --width 640 --height 480
```

기존 `.venv`로 같은 `--mode remote` 명령을 실행해도 됩니다. 종료 키는 `Q`, 소리 전환은 `M`입니다.
소리는 컨테이너가 아닌 Windows 클라이언트에서 재생합니다.
호스트의 `config/`는 서버에 읽기 전용 마운트됩니다. Windows에서 설정을 편집한 뒤 서버와 클라이언트를 재시작하세요.

```powershell
docker compose restart inference
```

원격 모드의 `--yolo-size`, `--depth-size`, `--confidence`는 서버를 변경하지 않습니다.
서버 입력 크기는 Compose의 `YOLO_SIZE`, `DEPTH_SIZE`, `CONFIDENCE` 환경변수를 사용합니다. 변경 후에는 컨테이너를 다시 생성합니다.

```powershell
$env:DEPTH_SIZE = '392'
docker compose up -d --force-recreate inference
Remove-Item Env:DEPTH_SIZE
```

서버 종료:

```powershell
docker compose down
```

<a id="local-api"></a>

## Docker 없이 API 실행

원격 클라이언트 구조만 확인하려면 로컬 GPU 환경에서 FastAPI를 실행합니다. 이 방식의 성공이 Docker 검증을 의미하지는 않습니다.
첫 번째 PowerShell에서 다음을 실행하고 유지합니다. GPU 모델 인스턴스가 중복되지 않도록 worker는 하나를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-server.txt
.\.venv\Scripts\python.exe -m uvicorn server.api:app --host 127.0.0.1 --port 8000 --workers 1
```

다른 PowerShell에서 프로젝트 폴더로 이동한 뒤 위 경량 클라이언트 명령을 사용합니다.
클라이언트는 `Q`, 서버는 서버 터미널의 `Ctrl+C`로 종료합니다. 기본 서버 모델·설정 경로는 `models/`, `config/settings.toml`입니다.

<a id="validation"></a>

## 거리·성능 CSV 및 테스트

### 줄자 기준 거리 CSV

카메라와 객체의 기준점을 정하고 거리를 직접 측정합니다. 다음은 실측 1.5m 예시입니다.

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --truth-m 1.5 --measurement-csv .\outputs\distance_validation.csv --object-index 0
```

화면에서 `0*`로 표시한 객체를 확인하고 `S`를 누르면 실측값·추정값·절대/상대 오차·객체 정보가 기록됩니다.
`--object-index`는 현재 결과 목록의 번호이며 고정된 객체 ID가 아닙니다. 매번 대상 물체를 확인하세요.
기록되는 거리는 클라이언트에서 평활화한 추정값입니다. 표시 거리 밖·깊이 무효 객체를 숨겨도 남은 객체 번호를 다시 매기지 않습니다.
유효 깊이가 없으면 `invalid_depth`로 기록하며 거리·오차 칸을 비웁니다. 탐지가 없으면 행을 추가하지 않습니다.
`--truth-m`와 `--measurement-csv`는 함께 지정하며 키 입력이 필요하므로 `--headless`와 함께 사용할 수 없습니다.
거리·조명·물체 종류를 바꾸어 반복 측정하세요. 줄자 기준 거리 정확도는 아직 미검증입니다.

### 성능 CSV

```powershell
.\.venv\Scripts\python.exe -m app.main --mode local --headless --mute --max-frames 30 --max-seconds 40 --benchmark-csv .\outputs\benchmark.csv
```

`--benchmark-csv`는 완료 결과의 FPS·캡처부터 결과까지의 지연·추론 시간 등을 기록합니다.
거리 오차를 기록하는 `--measurement-csv`와 용도가 다릅니다. 워밍업과 영상 내용, 입력 크기를 명시해 비교하세요.

### 자동·수동 테스트

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app server tools tests
.\.venv\Scripts\python.exe -m tools.check_sound
```

기본 실행은 하드웨어 테스트 2건을 건너뜁니다. 실제 GPU·웹캠을 사용하는 검사:

```powershell
$env:RUN_HARDWARE_TESTS = '1'
$env:CAMERA_INDEX = '0'
.\.venv\Scripts\python.exe -m pytest -m hardware -v
Remove-Item Env:RUN_HARDWARE_TESTS
Remove-Item Env:CAMERA_INDEX
```

모델별 검사는 직접 준비한 이미지 경로로 실행합니다. `--component`는 `yolo`, `depth`, `all`을 지원합니다.

```powershell
.\.venv\Scripts\python.exe -m tools.smoke_models --image .\sample.jpg --component all --device cuda:0 --output-dir .\outputs\smoke
```

정지 이미지 반복 추론 시간은 웹캠 FPS와 다릅니다. 실제 기록과 남은 검증은 [경고 기능 검증](ALERT_VALIDATION.md), [초기 검증](VALIDATION.md)에 있습니다.
소리 검사에서는 사람이 음량·Warning/Danger 구분을 확인해야 합니다. OpenCV 창에서 물리적인 `M/Q` 조작도 직접 확인하세요.

<a id="troubleshooting"></a>

## 문제 해결

| 증상 | 확인 및 해결 |
| --- | --- |
| `py -3.11`을 찾지 못함 | Python 3.11 64비트와 Python Launcher 설치 확인 |
| CUDA 진단 실패 | `nvidia-smi`, cu124 PyTorch 설치, `.venv` 실행 경로 확인 후 `tools.diagnose --require-cuda` 실행 |
| 카메라가 열리지 않음 | Windows 설정의 카메라·데스크톱 앱 권한 확인, 카메라 앱·Teams 등 종료, `--camera 1` 또는 `--camera-backend msmf` 시도 |
| GUI가 열리지 않음 | Windows 클라이언트에 `opencv-python` 설치 확인. 컨테이너에서 GUI를 열지 않음 |
| 객체가 표시되지 않음 | 설정의 10종 대상인지, 유효 깊이와 3m 표시 조건을 만족하는지 확인 |
| 소리가 나지 않음 | `M`, `--mute`, `sound_enabled`, Windows 볼륨·출력 장치, 중앙·거리 조건 확인. `tools.check_sound` 실행 |
| 경고 전환이 늦음 | 이동 중앙값·연속 결과 확인·히스테리시스·추론 지연의 영향을 확인 |
| 지연 또는 VRAM 부족 | 다른 GPU 프로그램 종료 후 `--depth-size 392`, `--width 640 --height 480`으로 측정 비교 |
| 다운로드 실패 | GitHub/Hugging Face 접속, 디스크 공간, `models/` 쓰기 권한 확인 |
| SHA256 불일치 | 검증을 우회하지 말고 오류에 나온 캐시 파일만 제거한 뒤 공식 다운로드를 재시도 |
| API 연결·timeout 오류 | 서버 로그·`/health` 확인. 통신 실패 동안 오래된 거리 경고는 중단되고 기본값에서는 재시도 |
| Docker GPU 인식 실패 | Docker Desktop 실행, WSL2/Linux 컨테이너 설정, NVIDIA 드라이버, Compose 로그·`/diagnostics` 확인 |
| 설정 변경이 반영되지 않음 | 로컬은 프로그램 재시작, Docker는 서버·Windows 클라이언트 재시작. Compose 환경변수 변경은 재생성 |

기술적 한계와 프로젝트 소개는 [README](../README.md)를 참고하세요. 추정 깊이는 실측 거리 또는 충돌 위험을 보장하지 않으며, 탐지 목록 필터가 모델이 모르는 위험 요소를 추가로 찾아주지는 않습니다.
