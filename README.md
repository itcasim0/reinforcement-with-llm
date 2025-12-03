# LLM기반 문서 교정 강화 학습
* 입력된 문서를 LLM기반으로 교정 시, 교정 방법에 대해 여러가지 action을 LLM기반으로 정의
* 교정은 한 번으로 끝나는 것이 아니라, state에 따라서 여러 번 수행을 할 수도 있음
* 이를 강화학습을 통해 입력된 문서에 따라 최적의 교정 방법을 찾고자 함

## 문제 제기
* 입력된 문서를 교정하는 시스템 프롬프트를 하나에 많이 넣을 경우 원하는 품질이 나오지 않음
  * 그래서, 여러가지 교정하는 방법을 두고 특정 상황에 따라 분기 처리하여 교정하도록 함
  * 하지만, 이 방법 또한 사람의 주관적인 판단으로 최적의 분기 처리를 하는 것이 쉽지 않음
* 따라서, 최적의 분기 처리로 문서 교정을 하기 위해 강화학습을 설계하고 수행하고자 함

## 강화학습 설계
* state = 문서 품질 평가 점수 (문법, 명확성, 간결성, 구조, 학술성, 유창성) + step + 이전 액션
* action_space = ("fix_grammar", "improve_clarity", "make_concise", "improve_structure", "make_academic")
* 알고리즘 = PPO (Proximal Policy Optimization)

## 요구사항
* Python >= 3.12
* uv (Python 패키지 매니저)
* OpenRouter API Key (LLM 사용을 위한 API 키)

## 설치 및 환경 설정

### 1. 가상환경 생성
```powershell
python -m venv .venv
```

### 2. 가상환경 활성화 (Windows PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### 3. 의존성 설치
```powershell
uv pip install -r requirements.txt
```
또는
```powershell
pip install -r requirements.txt
```
* uv sync를 할 경우, torch 관련하여 GPU버전 설치가 제대로 되지 않음 (이래저래 설정해봐도 잘 안됨...)

### 4. 환경 변수 설정
`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 API 키를 설정합니다.

```powershell
copy .env.example .env
```

`.env` 파일을 열어 다음과 같이 수정:
```
OPENROUTER_API_KEY=your_actual_api_key_here
```

## 사용 방법

### 1. 데이터 준비
논문 초록 데이터에 노이즈를 추가하여 학습용 데이터를 생성합니다.

```powershell
python src/add_noise.py
```

### 2. 오프라인 데이터셋 생성 (선택사항)
모든 가능한 액션 시퀀스에 대한 편집 결과를 미리 생성합니다.

```powershell
python src/generate_dataset.py
```

### 3. 강화학습 학습 및 평가
오프라인 데이터를 사용하여 PPO 알고리즘으로 정책을 학습합니다.

```powershell
python src/edit_document_offline.py
```

학습 중 다음이 자동으로 수행됩니다:
- 체크포인트 저장 (`logs/checkpoints/`)
- 학습 로그 기록
- 학습 완료 후 greedy policy 평가

### 4. 학습 결과 시각화

#### 학습 곡선 시각화
```powershell
python src/visualize_train.py
```

#### 액션 패턴 분석
```powershell
python src/visualize_action_patterns.py
```

#### 영향도 분석
```powershell
python src/visualize_impact.py
```

## 프로젝트 구조
```
reinforcement-with-llm/
├── src/
│   ├── environments/       # 강화학습 환경 (OfflineEditingEnv)
│   ├── methods/           # PPO 알고리즘 구현
│   ├── llm/              # LLM 인터페이스
│   ├── dataloader/       # 데이터 로더
│   ├── config/           # 설정 파일
│   └── utils/            # 유틸리티 함수
├── data/                 # 데이터 디렉토리
│   └── paper_data/       # 논문 데이터
├── logs/                 # 로그 및 체크포인트
└── references/           # 참고 논문
```

## 주요 파라미터 조정
`src/edit_document_offline.py` 파일에서 다음 파라미터를 조정할 수 있습니다:

- `NUM_EPISODES`: 학습 에피소드 수 (기본값: 1000)
- `TERMINAL_THRESHOLD`: 종료 임계값 (기본값: 9.5)
- `EDITOR_MODEL`: 사용할 LLM 모델 (기본값: "qwen/qwen3-8b")
- `COST_LAMBDA`: LLM 비용 패널티 가중치 (기본값: 1.0)
- `STEP_PENALTY`: 스텝당 패널티 (기본값: 0.1)

## 참고 자료
- 참고 논문: "Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning" (`references/2506.09033v2.pdf`)
- 체크포인트 사용법: `CHECKPOINT_USAGE.md`
