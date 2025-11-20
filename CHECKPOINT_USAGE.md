# 체크포인트 기능 사용 가이드

## 개요
학습 중 중단되어도 이전 상태에서 재개할 수 있도록 체크포인트 기능이 구현되었습니다.

## 주요 기능

### 1. 자동 체크포인트 저장
- 학습 중 지정된 주기마다 자동으로 체크포인트를 저장합니다
- 학습 종료 시 최종 체크포인트도 자동 저장됩니다
- 체크포인트에는 다음 정보가 포함됩니다:
  - 현재 에피소드 번호
  - 정책 네트워크 가중치
  - 옵티마이저 상태
  - 하이퍼파라미터

### 2. 학습 재개
- 저장된 체크포인트에서 학습을 재개할 수 있습니다
- 에피소드 번호가 자동으로 이어집니다

## 사용 방법

### 기본 학습 (체크포인트 저장)
```bash
python src/edit_document.py
```
- 기본 설정으로 30 에피소드 학습
- 5 에피소드마다 체크포인트 저장 (`checkpoints/` 디렉토리)

### 커스텀 설정으로 학습
```bash
python src/edit_document.py --num-episodes 100 --checkpoint-interval 10 --checkpoint-dir my_checkpoints
```
- `--num-episodes`: 학습할 총 에피소드 수 (기본값: 30)
- `--checkpoint-interval`: 체크포인트 저장 주기 (기본값: 5)
- `--checkpoint-dir`: 체크포인트 저장 디렉토리 (기본값: checkpoints)

### 학습 재개
```bash
# 디렉토리에서 최신 체크포인트 자동 로드
python src/edit_document.py --resume checkpoints

# 특정 체크포인트 파일 지정
python src/edit_document.py --resume checkpoints/checkpoint_ep20.pt

# 재개하면서 추가 학습
python src/edit_document.py --resume checkpoints --num-episodes 50
```

## 체크포인트 파일 구조

```
checkpoints/
├── checkpoint_ep5.pt
├── checkpoint_ep10.pt
├── checkpoint_ep15.pt
├── checkpoint_ep20.pt
├── checkpoint_ep25.pt
├── checkpoint_ep30.pt
└── latest_checkpoint.txt  # 최신 체크포인트 경로 저장
```

## 예시 시나리오

### 시나리오 1: 학습 중단 후 재개
```bash
# 1. 초기 학습 시작 (30 에피소드)
python src/edit_document.py

# 2. 15 에피소드에서 중단됨 (Ctrl+C 또는 오류)

# 3. 학습 재개 (에피소드 15부터 계속)
python src/edit_document.py --resume checkpoints --num-episodes 15
# → 총 30 에피소드 완료
```

### 시나리오 2: 긴 학습을 여러 단계로 나누기
```bash
# 1단계: 50 에피소드 학습
python src/edit_document.py --num-episodes 50 --checkpoint-interval 10

# 2단계: 추가 50 에피소드 학습
python src/edit_document.py --resume checkpoints --num-episodes 50

# 3단계: 추가 100 에피소드 학습
python src/edit_document.py --resume checkpoints --num-episodes 100
# → 총 200 에피소드 완료
```

### 시나리오 3: 특정 체크포인트에서 재학습
```bash
# 에피소드 20 체크포인트에서 다시 시작
python src/edit_document.py --resume checkpoints/checkpoint_ep20.pt --num-episodes 30
# → 에피소드 20부터 50까지 학습
```

## 프로그래밍 방식 사용

코드에서 직접 체크포인트 기능을 사용할 수도 있습니다:

```python
from methods.ppo import PPORunner

# PPORunner 생성
runner = PPORunner(...)

# 체크포인트에서 로드
runner.load_checkpoint("checkpoints")

# 학습 (자동 체크포인트 저장)
runner.train(
    num_episodes=100,
    checkpoint_dir="checkpoints",
    checkpoint_interval=10
)

# 수동으로 체크포인트 저장
runner.save_checkpoint("checkpoints", episode=50)
```

## 주의사항

1. **디스크 공간**: 체크포인트 파일은 모델 크기에 따라 수 MB가 될 수 있습니다
2. **호환성**: 체크포인트는 동일한 모델 구조에서만 로드 가능합니다
3. **에피소드 번호**: 재개 시 에피소드 번호가 자동으로 이어지므로 로그 확인 시 참고하세요
