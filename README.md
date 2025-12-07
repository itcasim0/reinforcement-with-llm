# LLM기반 문서 교정 강화 학습

- 입력된 문서를 LLM기반으로 교정 시, 교정 방법에 대해 여러가지 action을 LLM기반으로 정의
- 교정은 한 번으로 끝나는 것이 아니라, state에 따라서 여러 번 수행을 할 수도 있음
- 이를 강화학습을 통해 입력된 문서에 따라 최적의 교정 방법을 찾고자 함

## 문제 제기

- 입력된 문서를 교정하는 시스템 프롬프트를 하나에 많이 넣을 경우 원하는 품질이 나오지 않음
  - 그래서, 여러가지 교정하는 방법을 두고 특정 상황에 따라 분기 처리하여 교정하도록 함
  - 하지만, 이 방법 또한 사람의 주관적인 판단으로 최적의 분기 처리를 하는 것이 쉽지 않음
- 또한, 분기 처리도 LLM을 강화학습 기반으로 파인튜닝을 할 수도 있으나 이는 시간과 자원의 비용이 상당히 커 쉽지 않음
- 따라서, 최적의 분기 처리로 문서 교정을 하기 위해 강화학습을 설계하고 수행하고자 함

## 요구사항

- Python >= 3.12
- uv (Python 패키지 매니저)
- OpenRouter API Key (LLM 사용을 위한 API 키)

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

- uv sync를 할 경우, torch 관련하여 GPU버전 설치가 제대로 되지 않음 (이래저래 설정해봐도 잘 안됨...)

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

### 0. Quick Start

#### 1) 오프라인 학습 및 평가

1.  [공유 드라이브](https://drive.google.com/drive/folders/17H69fxD9U-RU44bgt50dYbvhN6xCRK_U?usp=drive_link)에서 data/paper_data/offline 폴더를 프로젝트의 data 폴더 내에 똑같은 경로로 구성합니다.

2.  학습 및 평가를 진행하는 코드를 실행합니다.

```python
python scripts/train_eval/ppo_offline.py
```

3.  시각화 코드를 실행합니다.

```python
python scripts/visualizer/train_log_ppo_offline.py
```

#### 2) 온라인 학습 및 평가

- 온라인 학습은 API 키 설정 및 크레딧 충전이 되어있어야만 가능합니다.

1.  [공유 드라이브](https://drive.google.com/drive/folders/17H69fxD9U-RU44bgt50dYbvhN6xCRK_U?usp=drive_link)에서 data/paper_data/cache 폴더를 프로젝트의 data 폴더 내에 똑같은 경로로 구성합니다.

2.  학습 및 평가를 진행하는 코드를 실행합니다.

```python
# ppo, a2c, dqn 가능
python scripts/train_eval/ppo.py
```

3.  시각화 코드를 실행합니다.

```python
# ppo. a2c. dqn 가능
python scripts/visualizer/train_log_ppo.py
```

### 1. 데이터 준비

#### 1) 데이터 수집

- 크롤링을 통하여 데이터를 수집합니다.

```powershell
python scripts/dataset_builder/crawl.py
```

#### 2) 데이터 처리

- 크롤링한 데이터를 학습에 용이한 형태로 변환합니다.

```powershell
python scripts/dataset_builder/excel_to_json.py
```

#### 3) 데이터 재구성

- 학습을 위해 교정이 필요한 형태로 데이터를 재구성합니다.
  - 교정이 필요한 데이터셋이 존재하지 않으므로 다음의 코드를 활용하여 재구성합니다.
  - [공유 드라이브](https://drive.google.com/drive/folders/17H69fxD9U-RU44bgt50dYbvhN6xCRK_U?usp=drive_link)에서 data/paper_data/reconstruct에 학습에 사용한 재구성 데이터가 존재합니다.

```powershell
python scripts/dataset_builder/reconstruct.py
```

#### 4) 오프라인 데이터셋 생성

- 모든 가능한 액션 시퀀스에 대한 교정 결과를 미리 생성합니다.
  - 학습 시, LLM을 매번 호출하는 것은 시간과 비용이 많이 소모됩니다.
  - 사전에 미리 한번에 만들어두고, 재사용하는 것으로 시간과 비용을 절약합니다.

### 2. 학습/평가 및 시각화

#### 1) 오프라인 PPO, PPO, A2C, DQN 실행

```python
python scripts/train_eval/ppo_offline.py
python scripts/train_eval/ppo.py
python scripts/train_eval/a2c.py
python scripts/train_eval/dqn.py
```

#### 2) 오프라인 PPO, PPO, A2C, DQN 시각화

```python
python scripts/visualizer/train_log_ppo_offline.py
python scripts/visualizer/train_log_ppo.py
python scripts/visualizer/train_log_ppo.py
python scripts/visualizer/train_log_ppo.py
```

## 프로젝트 구조
```
reinforcement-with-llm/
├── src/
│   ├── config/                    # 설정 파일
│   │   └── paths.py              # 경로 설정
│   ├── dataloader/               # 데이터 로더
│   │   ├── cache_loader.py       # 캐시 데이터 로더
│   │   ├── offline_loader.py     # 오프라인 데이터 로더
│   │   └── reconstruct_loader.py # 재구성 데이터 로더
│   ├── dataset/                  # 데이터셋 처리
│   │   └── reconstructor.py      # 데이터 재구성기
│   ├── environments/             # 강화학습 환경
│   │   └── editing_env/          # 문서 교정 환경
│   │       ├── base_env.py       # 기본 환경
│   │       ├── offline_env.py    # 오프라인 환경
│   │       └── components/       # 환경 컴포넌트
│   │           ├── component.py  # 기본 컴포넌트
│   │           ├── editor.py     # 문서 편집기
│   │           └── eval/         # 평가 모듈
│   │               ├── evaluation_config.py
│   │               └── evaluator.py
│   ├── llm/                      # LLM 인터페이스
│   │   └── core.py               # LLM 핵심 기능
│   ├── methods/                  # 강화학습 알고리즘
│   │   ├── a2c/                  # A2C 알고리즘
│   │   │   ├── estimators.py
│   │   │   ├── policy.py
│   │   │   └── runner.py
│   │   ├── dqn/                  # DQN 알고리즘
│   │   │   ├── policy.py
│   │   │   ├── replay_buffer.py
│   │   │   └── runner.py
│   │   └── ppo/                  # PPO 알고리즘
│   │       ├── estimators.py
│   │       ├── policy.py
│   │       └── runner.py
│   └── utils/                    # 유틸리티 함수
│       ├── checkpoint_utils.py   # 체크포인트 관리
│       ├── logger_factory.py     # 로거 팩토리
│       ├── timer.py              # 타이머
│       └── util.py               # 기타 유틸리티
├── scripts/
│   ├── dataset_builder/          # 데이터셋 빌드
│   │   ├── crawl.py              # 데이터 크롤링
│   │   ├── excel_to_json.py      # 데이터 변환
│   │   ├── generate_offline.py   # 오프라인 데이터 생성
│   │   └── reconstruct.py        # 데이터 재구성
│   ├── test/                     # 테스트 스크립트
│   │   ├── evaluate_document.py
│   │   └── test_hybrid_editor.py
│   ├── train_eval/               # 학습 및 평가
│   │   ├── a2c.py                # A2C 학습
│   │   ├── dqn.py                # DQN 학습
│   │   ├── ppo_offline.py        # 오프라인 PPO 학습
│   │   └── ppo.py                # 온라인 PPO 학습
│   └── visualizer/               # 시각화
│       ├── action_patterns.py    # 액션 패턴 분석
│       ├── impact.py             # 영향도 분석
│       ├── train_log_a2c.py      # A2C 학습 로그
│       ├── train_log_dqn.py      # DQN 학습 로그
│       ├── train_log_ppo_offline.py  # 오프라인 PPO 로그
│       └── train_log_ppo.py      # 온라인 PPO 로그
├── data/                         # 데이터 디렉토리
│   └── paper_data/               # 논문 데이터
├── logs/                         # 로그 및 체크포인트
└── references/                   # 참고 논문
    └── 2506.09033v2.pdf          # Router-R1 논문
```
