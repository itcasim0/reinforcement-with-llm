# Project Guideline

- Reference 논문은 "Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning" 입니다.
- 수집된 질문 데이터를 통해 강화학습을 하여 질문 별로 적절한 LLM을 선택할 수 있도록 합니다.

## CLI 사용 가이드

- os는 window입니다.
- CLI를 사용하게 될 때는 powershell이 기본 설정임으로 참고합니다.

## 코딩 가이드

- 내부의 모듈을 import할 때는 src를 사용하지 않도록 합니다.
  - 단, ./scripts/ 내의 코드에는 src를 붙여줍니다.
- 주석을 작성할 때는 순서를 메기지 않습니다. (순서는 언제든지 바뀔 수 있기 때문)
