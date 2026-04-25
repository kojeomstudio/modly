# Modly — 아키텍처 및 동작 원리 분석

> 본 문서는 Modly 프로젝트의 내부 구조, 사용 모델, 실행 파이프라인, Docker 적용 가능성에 대한 기술 검토 결과를 정리한 문서입니다.

---

## 1. 프로젝트 개요

**Modly**는 단일 이미지로부터 3D 메시(.glb)를 생성하는 로컬 데스크탑 애플리케이션입니다.

- **프론트엔드**: Electron + React + Three.js (WebGL 기반 3D 뷰어)
- **백엔드**: 로컬 FastAPI 서버 (`127.0.0.1:8765`)
- **추론 엔진**: HuggingFace 기반 이미지→3D 생성 모델 (확장 플러그인 시스템)
- **타깃 OS**: Windows / Linux (macOS는 추후 지원)

본 프로젝트는 LLM(대화형 언어 모델)을 사용하지 않습니다. "AI"는 모두 비전 기반 3D 생성 모델을 의미합니다.

---

## 2. 사용 모델

모델은 모두 외부 GitHub 확장(extension)으로 분리되어 있고, 매니페스트에 명시된 HuggingFace 레포에서 가중치를 자동 다운로드합니다.

| 모델 | HF Repo / 출처 | 카테고리 | 핵심 기법 |
|---|---|---|---|
| **TripoSR** (기본) | `stabilityai/TripoSR` | 단일 이미지 3D 복원 | LRM 계열 — Transformer로 triplane 잠재 표현 생성 → NeRF 디코더 → 메시 추출 |
| **Hunyuan3D 2 Mini / Mini Turbo / Mini Fast** | Tencent | 2단계 3D 생성 | ① shape diffusion (DiT) ② multi-view texture diffusion |
| **TripoSG** | VAST-AI | 단일 이미지 3D | Rectified Flow + score distillation |
| **Trellis2 GGUF** | Microsoft (양자화판) | 구조화 잠재 확산 | Structured Latent diffusion → Gaussians/RBF → 메시 |

공통 사항: PyTorch + CUDA 기반이며 GPU 추론을 전제로 합니다.

---

## 3. 시스템 아키텍처

### 3.1 컴포넌트 구성

```
┌─────────────────────────────────────────────┐
│            Electron Main Process            │
│  ┌──────────────────────────────────────┐   │
│  │  python-bridge.ts                    │   │
│  │   - spawn(uvicorn main:app)          │───┼──┐
│  │   - killProcessOnPort(8765)          │   │  │
│  │  python-setup.ts                     │   │  │
│  │   - venv 생성 / pip install          │   │  │
│  │  ipc-handlers.ts                     │   │  │
│  │   - nvidia-smi 호출                  │   │  │
│  │   - 확장 GitHub 다운로드             │   │  │
│  └──────────────────────────────────────┘   │  │
│             ▲                                │  │
│             │ IPC                            │  │
│  ┌──────────┴──────────┐                     │  │
│  │  Renderer (React)   │                     │  │
│  │   - Three.js viewer │                     │  │
│  └─────────────────────┘                     │  │
└─────────────────────────────────────────────┘  │
                                                 │
   ┌────────── stdout (HTTP) ────────────────────┘
   ▼
┌─────────────────────────────────────────────┐
│          FastAPI (127.0.0.1:8765)           │
│   /generate/from-image, /model/status, ...  │
│                                             │
│  ┌──────── GeneratorRegistry ──────────┐    │
│  │  EXTENSIONS_DIR 스캔 → 확장 로드     │    │
│  └──────────────────────────────────────┘    │
│                  │                          │
│        ┌─────────┴─────────┐                │
│        ▼                   ▼                │
│  ┌──────────┐        ┌──────────┐           │
│  │ Direct   │        │ Subproc  │           │
│  │ mode     │        │ mode     │           │
│  │(legacy)  │        │(per ext) │           │
│  └──────────┘        └────┬─────┘           │
└──────────────────────────│──────────────────┘
                           │ NDJSON over
                           │ stdin/stdout
                           ▼
        ┌─────────────────────────────────┐
        │  Extension Subprocess           │
        │  (각 확장의 격리된 .venv)        │
        │   runner.py → generator.py      │
        │   ├─ load(): GPU 적재            │
        │   ├─ generate(image, params)    │
        │   └─ unload(): VRAM 회수        │
        └─────────────────────────────────┘
```

### 3.2 디렉토리 구조 (요점)

```
modly/
├── electron/main/
│   ├── python-bridge.ts        # FastAPI 라이프사이클
│   ├── python-setup.ts         # venv·pip 부트스트랩
│   ├── process-runner.ts       # 일회성 파이썬 실행
│   └── ipc-handlers.ts         # GPU 감지·확장 다운로드 등
├── src/                        # React UI (Three.js)
├── api/                        # FastAPI 백엔드
│   ├── main.py                 # FastAPI 진입점
│   ├── runner.py               # 확장 subprocess 진입점 (NDJSON)
│   ├── routers/                # /generate, /model, /optimize, ...
│   ├── services/
│   │   ├── generator_registry.py
│   │   ├── extension_process.py
│   │   └── generators/
│   │       └── base.py         # BaseGenerator 추상 클래스
│   ├── texture_baker/
│   └── uv_unwrapper/
└── resources/                  # python-embed (Win/Linux 패키징용)
```

---

## 4. 확장 플러그인 시스템

### 4.1 확장 매니페스트

각 확장은 다음 두 파일을 포함합니다.

```
<EXTENSIONS_DIR>/<ext_id>/
├── manifest.json     # id, generator_class, hf_repo, pip_requirements, nodes, params
├── generator.py      # BaseGenerator 상속 클래스
└── .venv/            # (선택) 격리된 파이썬 환경 — 존재하면 subprocess 모드
```

`manifest.json`이 정의하는 핵심 필드:
- `id`, `generator_class`
- `hf_repo`, `hf_skip_prefixes`, `download_check`
- `pip_requirements` — 확장 venv에 설치되는 의존성
- `nodes` — UI에 노출되는 모델 변형(여러 변형 가능: mini/turbo/fast 등)
- 노드별 `params` — UI 자동 생성용 스키마

### 4.2 로딩 모드 결정 (`generator_registry.py`)

```python
has_venv         = _venv_python(ext_dir).exists()
has_build_vendor = (ext_dir / "build_vendor.py").exists()
vendor_built     = (ext_dir / "vendor").exists()
subprocess_mode  = has_venv or (has_build_vendor and not vendor_built)
```

- **Subprocess 모드** (권장): 확장 venv의 파이썬으로 `runner.py`를 띄워 격리 실행
- **Direct 모드** (레거시): venv 없는 확장을 백엔드 프로세스에 직접 import (의존성 충돌 위험)

### 4.3 IPC 프로토콜 (`runner.py`)

부모(FastAPI) ↔ 자식(확장 subprocess) 통신은 **NDJSON**(Newline-Delimited JSON):

- stdout: `{"type": "ready" | "progress" | "result" | "error", ...}`
- stdin: `{"cmd": "load" | "generate" | "unload" | "is_loaded" | "params_schema", ...}`
- stderr: 별도 파이프로 캡처되어 로깅에만 사용

환경변수 주입:
- `EXTENSION_DIR`, `MODELS_DIR`, `WORKSPACE_DIR`, `MODLY_API_DIR`
- `MODEL_DIR` (오버라이드), `HF_TOKEN` 등

---

## 5. 생성 파이프라인

### 5.1 엔드포인트 흐름

```
POST /generate/from-image (multipart)
    image, model_id, collection, remesh, enable_texture, texture_resolution, params(JSON)
        │
        ├─ 입력 검증 (콘텐츠 타입, remesh 옵션, 콜렉션 이름 sanitize)
        ├─ generator_registry.switch_model(model_id)
        ├─ job_id = uuid4()
        ├─ background_tasks.add_task(_run_generation, ...)
        └─ return { job_id }

GET /generate/status/{job_id}
    → JobStatus { status, progress, step, result_path }
```

### 5.2 `_run_generation` 단계

1. 모델 적재 — `BaseGenerator.load()` (필요 시 `_auto_download()`로 HF 가중치 다운로드)
2. 추론 — `generate(image_bytes, params, progress_cb, cancel_event) -> Path("*.glb")`
3. 후처리 (옵션):
   - **Remesh**: `pymeshlab` 기반 quad/triangle 재구성
   - **Texture baking**: `texture_baker/` 모듈
   - **UV unwrap**: `uv_unwrapper/` 모듈
4. 결과 메시를 `WORKSPACE_DIR/<collection>/`에 저장
5. `JobStatus.status = "completed"`

### 5.3 진행률 평활화 (`smooth_progress`)

추론 모델이 콜백을 자주 발행하지 못하므로, 별도 스레드에서 `start..end` 사이를 일정 간격으로 부드럽게 증가시키다가 실제 단계 완료 시 점프시킵니다. UI 사용자에게 "정지된 듯한" 인상을 막기 위함.

### 5.4 취소 메커니즘

- `_cancel_events: Dict[job_id, threading.Event]`
- `BaseGenerator._check_cancelled(cancel_event)`가 단계마다 `GenerationCancelled`를 발생시켜 graceful 중단

---

## 6. 모델 다운로드 전략

`BaseGenerator._auto_download()` (`api/services/generators/base.py`):

```python
from huggingface_hub import snapshot_download

ignore = list(self.hf_skip_prefixes) + [
    "*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes",
]
snapshot_download(
    repo_id=self.hf_repo,
    local_dir=str(self.model_dir),
    ignore_patterns=ignore,
)
```

- 각 모델 가중치는 `MODELS_DIR/<manifest_id>/`에 저장 (기본 `~/.modly/models/`)
- `download_check` 경로(매니페스트)로 다운로드 완료 검증
- HuggingFace 비공개 레포는 `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` 환경변수로 인증

---

## 7. 메모리/리소스 관리

`BaseGenerator.unload()`:

1. 내부 모델 객체 참조 해제 (`self._model = None`)
2. `gc.collect()`
3. `torch.cuda.empty_cache()` — VRAM 회수
4. Windows: `SetProcessWorkingSetSizeEx`로 OS에 워킹셋 회수 요청
5. 모델 스왑 시 자동 호출 — VRAM 절약 위해 동시 1개만 적재

---

## 8. Docker 적용 가능성

### 8.1 컴포넌트별 컨테이너화 적합도

| 컴포넌트 | 가능성 | 비고 |
|---|---|---|
| FastAPI 코어 (`api/`) | ✅ 가능 | 표준 uvicorn, requirements 단순 |
| 확장 추론 프로세스 | △ 조건부 | venv 동적 생성 + 영속 볼륨 + GPU 런타임 필수 |
| Electron UI | ❌ 비현실적 | WebGL/X11/macOS GUI 호환 문제 |
| GPU 추론 | OS 종속 | Linux + NVIDIA Container Toolkit. macOS Docker는 CUDA 미지원 |

### 8.2 주요 제약

1. **`python-bridge.ts`가 호스트에서 직접 프로세스 라이프사이클 관리** (`killProcessOnPort`, exit 핸들러). API를 컨테이너로 분리하면 이 로직은 우회/비활성 또는 "외부 API URL 모드" 패치 필요.
2. **확장 venv 경로가 호스트 파일시스템 가정**. `EXTENSIONS_DIR`, `MODELS_DIR`, `WORKSPACE_DIR`을 볼륨으로 일관되게 매핑.
3. **`pymeshlab`이 GL/X 의존**: `libgl1`, `libglu1-mesa`, `libxrender1` 등 시스템 패키지 설치 필요.
4. **Electron 빌드의 `extraResources`(python-embed)**: 컨테이너 환경에선 불필요·중복.

### 8.3 권장 구성

**A. API-only 컨테이너 (헤드리스 사용 권장)**
- 베이스: `nvidia/cuda:12.1-cudnn-runtime-ubuntu22.04`
- `api/`만 컨테이너화, REST 클라이언트로 직접 호출
- 볼륨: `~/.modly/{models,workspace,extensions}` 매핑
- env: `HF_TOKEN`, `MODELS_DIR`, `WORKSPACE_DIR`, `EXTENSIONS_DIR`

**B. 하이브리드 (Electron 호스트 + API 컨테이너)**
- Electron은 호스트 실행, 설정에서 `API_BASE_URL`을 컨테이너로 지정
- `python-bridge.ts`에 외부 API 모드 플래그 추가 필요 (start/kill 우회)

**C. 풀스택 컨테이너**
- 비추천. WebGL/X11 포워딩 비용 대비 이득 없음.

### 8.4 최소 Dockerfile 스케치 (참고)

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglu1-mesa libxrender1 libxext6 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/api
COPY api/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY api/ /app/api/

ENV MODELS_DIR=/data/models \
    WORKSPACE_DIR=/data/workspace \
    EXTENSIONS_DIR=/data/extensions

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
```

볼륨 매핑 예시:
```bash
docker run --gpus all \
  -p 8765:8765 \
  -v ~/.modly/models:/data/models \
  -v ~/.modly/workspace:/data/workspace \
  -v ~/.modly/extensions:/data/extensions \
  -e HF_TOKEN=$HF_TOKEN \
  modly-api
```

> 주의: 확장은 컨테이너 내부에서 `pip install`을 수행하므로 첫 실행 시 시간이 오래 걸리고, 컨테이너 재시작 시 사라지지 않도록 `EXTENSIONS_DIR`을 반드시 영속 볼륨으로 마운트해야 합니다.

---

## 9. 결론

- Modly의 핵심 가치는 **모델 추상화 + venv 격리 + HF 자동 다운로드 + 진행률/취소 표준화**한 어댑터 프레임워크에 있습니다.
- 추론 엔진은 LLM이 아닌 **이미지 컨디셔닝 기반 3D 확산/Transformer 모델**입니다.
- **API-only 컨테이너화는 명확히 가능**하지만, ① 확장 venv·모델 경로 볼륨화, ② NVIDIA Container Toolkit, ③ Electron 측 외부 API 모드 패치 — 세 가지가 전제됩니다.
- **풀 데스크탑 경험을 컨테이너에 담는 것은 권장하지 않습니다**. Electron은 호스트 실행 + API만 컨테이너 분리하는 하이브리드가 가장 현실적입니다.
