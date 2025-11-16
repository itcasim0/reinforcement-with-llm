import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI


# ============================================
# 1. OpenRouter 클라이언트 생성
# ============================================

def get_openrouter_client() -> OpenAI:
    """
    OpenRouter용 OpenAI 클라이언트 생성.
    API 키는 환경변수 OPENROUTER_API_KEY에서 읽어온다.

    (예시 설정)
    - PowerShell:
        $env:OPENROUTER_API_KEY="sk-or-..."
    - Bash:
        export OPENROUTER_API_KEY="sk-or-..."
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENROUTER_API_KEY가 설정되어 있지 않습니다.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


# ============================================
# 2. 문서 및 LLM 래퍼 클래스 정의
# ============================================

@dataclass
class Document:
    """
    하나의 문서를 표현하는 데이터 클래스.
    실제 품질 점수는 LLM 평가기로부터 매번 계산한다.
    """
    text: str


class OpenRouterEditorLLM:
    """
    실제 OpenRouter (GPT-4o-mini 등)를 호출하는 편집 LLM 래퍼.

    editor.edit(text, action) -> (edited_text, cost)
    형태로 사용한다.
    """

    def __init__(self, client: OpenAI, model: str = "openai/gpt-4o-mini", base_cost: float = 0.02):
        self.client = client
        self.model = model
        # 보상에서 사용할 LLM 호출 패널티 (간단히 고정값으로 사용)
        self.base_cost = base_cost

    def _make_prompt(self, text: str, action: str) -> str:
        """
        액션에 따라 LLM에 전달할 지시문을 생성.
        실제 모델에게는 '수정된 글만 출력' + '언어 유지'를 강하게 요청한다.
        """
        if action == "fix_grammar":
            instruction = (
                "문법과 맞춤법 오류를 수정해줘. 내용은 바꾸지 말고, "
                "언어(한국어/영어)는 원문과 동일하게 유지해. 수정된 글만 출력해."
            )
        elif action == "improve_clarity":
            instruction = (
                "표현을 더 명확하고 이해하기 쉽게 고쳐줘. 불필요한 설명은 빼고, "
                "언어(한국어/영어)는 원문과 동일하게 유지해. 수정된 글만 출력해."
            )
        elif action == "make_concise":
            instruction = (
                "중복되거나 불필요한 문장을 줄이고, 간결하게 만들어줘. "
                "언어(한국어/영어)는 원문과 동일하게 유지해. 수정된 글만 출력해."
            )
        elif action == "improve_structure":
            instruction = (
                "문단과 문장 순서를 조정해서 논리 흐름이 자연스럽게 만들어줘. "
                "언어(한국어/영어)는 원문과 동일하게 유지해. 수정된 글만 출력해."
            )
        else:
            # 혹시 모르는 기타 액션용 fallback
            instruction = (
                f"다음 글을 자연스럽게 다듬어줘. (액션: {action}) "
                "언어(한국어/영어)는 원문과 동일하게 유지해. 수정된 글만 출력해."
            )

        prompt = f"""작업 지시: {instruction}

[원본 글]
{text}
"""
        return prompt

    def edit(self, text: str, action: str) -> Tuple[str, float]:
        """
        실제 LLM을 호출하여 문서를 편집.
        - text: 현재 문서 (string)
        - action: 편집 액션 이름
        - 반환: (편집된 텍스트, 패널티 계산에 쓸 cost 값)
        """
        prompt = self._make_prompt(text, action)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,  # 약간의 다양성은 허용
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국어/영어 글을 요청에 맞게 편집하는 글쓰기 보조 도우미입니다. "
                        "입력 글의 언어(한국어 또는 영어)는 반드시 그대로 유지해야 합니다. "
                        "반드시 수정된 글만 출력하고, 설명이나 메타 코멘트는 쓰지 마세요."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        edited_text = resp.choices[0].message.content.strip()

        # 여기서는 간단하게 base_cost를 고정 비용으로 사용
        cost = self.base_cost
        return edited_text, cost


class OpenRouterJudgeLLM:
    """
    실제 OpenRouter를 호출하는 평가 LLM 래퍼.

    judge.score(text) -> {"grammar": ..., "readability": ..., "coherence": ..., "overall": ...}
    형태로 동작한다.
    """

    def __init__(self, client: OpenAI, model: str = "openai/gpt-4.1"):
        self.client = client
        self.model = model

    def score(self, text: str) -> Dict[str, float]:
        """
        LLM에게 문서 품질을 0~5점으로 평가하게 하고, JSON 파싱 후 dict로 반환.
        파싱 실패 시에는 보수적으로 2.5 점대를 기본값으로 사용한다.
        """
        prompt = f"""
다음 글의 품질을 0~5 점수로 평가해줘.

각 항목:
- grammar: 문법 및 맞춤법 정확성
- readability: 읽기 쉬운 정도
- coherence: 논리적 연결성과 흐름
- overall: 전체적인 품질

반드시 아래 JSON 형식 그대로만 출력해.
설명 문장은 쓰지 말고, JSON만 출력해.

예시:
{{
  "grammar": 3.0,
  "readability": 2.5,
  "coherence": 3.0,
  "overall": 2.5
}}

평가할 글:
{text}
"""

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,  # 평가의 안정성을 위해 deterministic하게
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 글의 품질을 평가하는 심사위원입니다. "
                        "문법, 가독성, 논리적 일관성, 전체 품질을 0~5 점수로 평가합니다. "
                        "반드시 JSON 형식만 출력하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        raw = resp.choices[0].message.content.strip()

        # 혹시 모델이 실수로 앞뒤에 설명을 붙일 경우를 대비해 중괄호 부분만 추출
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_str = raw[start:end + 1]
            else:
                json_str = raw

            data = json.loads(json_str)

            # 각 항목을 float으로 안전하게 변환하고, 0~5 범위로 클램핑
            def safe(v: float) -> float:
                try:
                    x = float(v)
                except Exception:
                    x = 2.5
                return max(0.0, min(5.0, x))

            scores = {
                "grammar": safe(data.get("grammar", 2.5)),
                "readability": safe(data.get("readability", 2.5)),
                "coherence": safe(data.get("coherence", 2.5)),
                "overall": safe(data.get("overall", 2.5)),
            }
        except Exception as e:
            print("[경고] Judge LLM JSON 파싱 실패, 기본값 사용:", e)
            scores = {
                "grammar": 2.5,
                "readability": 2.5,
                 "coherence": 2.5,
                "overall": 2.5,
            }

        return scores


# ============================================
# 3. 강화학습 환경
# ============================================

class EditingEnv:
    """
    문서 자동 교정 강화학습 환경.

    - 상태: (grammar, readability, coherence, overall) 점수 (각 0~5 정수로 반올림)
    - 행동:
        0: fix_grammar
        1: improve_clarity
        2: make_concise
        3: improve_structure
        4: stop_editing
    """

    def __init__(
        self,
        documents: List[Document],
        editor: OpenRouterEditorLLM,
        judge: OpenRouterJudgeLLM,
        max_steps: int = 4,
        terminal_threshold: float = 3.0,
        cost_lambda: float = 1.0,
    ):
        self.documents = documents
        self.editor = editor
        self.judge = judge
        self.max_steps = max_steps
        self.terminal_threshold = terminal_threshold
        self.cost_lambda = cost_lambda

        self.actions = [
            "fix_grammar",
            "improve_clarity",
            "make_concise",
            "improve_structure",
            "stop_editing",
        ]
        self.num_actions = len(self.actions)

        # 현재 에피소드 상태
        self.current_text: str = ""
        self.current_scores: Dict[str, float] = {}
        self.current_step: int = 0

    def reset(self) -> Tuple[Tuple[int, int, int, int], str]:
        """
        에피소드 초기화.
        - 랜덤 문서를 하나 선택
        - 평가 LLM으로 초기 품질 점수 계산
        - 상태 (g, r, c, o)와 현재 텍스트 반환
        """
        self.current_step = 0
        base_doc = random.choice(self.documents)
        self.current_text = base_doc.text

        self.current_scores = self.judge.score(self.current_text)
        state = self._scores_to_state(self.current_scores)
        return state, self.current_text

    def _scores_to_state(self, scores: Dict[str, float]) -> Tuple[int, int, int, int]:
        """
        연속값(0~5 float)을 0~5 정수로 반올림하여 상태로 사용.
        """
        g = int(round(scores["grammar"]))
        r = int(round(scores["readability"]))
        c = int(round(scores["coherence"]))
        o = int(round(scores["overall"]))
        return (g, r, c, o)

    def step(self, action_index: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict]:
        """
        환경 한 step 진행.
        - action_index: 액션 인덱스
        - 반환: (다음 상태, 보상, done, info)
        """
        assert 0 <= action_index < self.num_actions
        action = self.actions[action_index]

        prev_scores = self.current_scores
        prev_overall = prev_scores["overall"]

        done = False
        info = {"action": action, "prev_scores": prev_scores}

        # stop_editing: 편집 없이 종료 보상만 부여
        if action == "stop_editing":
            final_reward = self._terminal_reward(prev_overall)
            reward = final_reward
            done = True
            info["reason"] = "stop_action"
            next_state = self._scores_to_state(prev_scores)
            info["new_scores"] = prev_scores
            return next_state, reward, done, info

        # 편집 LLM 호출
        edited_text, cost = self.editor.edit(self.current_text, action)
        self.current_text = edited_text
        self.current_step += 1

        # 평가 LLM 호출
        new_scores = self.judge.score(self.current_text)
        self.current_scores = new_scores
        new_overall = new_scores["overall"]

        # 품질 변화량
        delta_quality = new_overall - prev_overall

        # step 보상
        if delta_quality >= 0:
            step_reward = delta_quality
        else:
            step_reward = -0.2 * abs(delta_quality)

        # LLM 호출 비용 패널티
        step_reward -= self.cost_lambda * cost

        reward = step_reward

        # 최대 스텝 도달 시 자동 종료 + 종료 보상 추가
        if self.current_step >= self.max_steps:
            final_bonus = self._terminal_reward(new_overall)
            reward += final_bonus
            done = True
            info["reason"] = "max_steps"

        next_state = self._scores_to_state(new_scores)
        info["new_scores"] = new_scores
        info["delta_quality"] = delta_quality
        return next_state, reward, done, info

    def _terminal_reward(self, overall: float) -> float:
        """
        에피소드 종료 시 추가 보상:
        - overall >= threshold: +1.0
        - 그 외: -0.5
        """
        if overall >= self.terminal_threshold:
            return 1.0
        else:
            return -0.5


# ============================================
# 4. Q-learning 에이전트
# ============================================

class QLearningAgent:
    """
    간단한 탭형 Q-learning 에이전트.
    상태: (g, r, c, o) 4차원 정수 튜플
    액션: 0 ~ num_actions-1
    """

    def __init__(
        self,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.98,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Q 테이블: dict[(state, action)] = value
        self.q_table: Dict[Tuple[Tuple[int, int, int, int], int], float] = {}

    def get_q(self, state: Tuple[int, int, int, int], action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state: Tuple[int, int, int, int]) -> int:
        """
        ε-greedy 정책으로 액션 선택.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = [self.get_q(state, a) for a in range(self.num_actions)]
            max_q = max(q_values)
            candidates = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(candidates)

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        done: bool,
    ):
        """
        Q-learning 업데이트:
        Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]
        """
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in range(self.num_actions)]
            max_next_q = max(next_q_values)
            target = reward + self.gamma * max_next_q

        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """
        에피소드마다 탐색률 ε를 조금씩 감소.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ============================================
# 5. 더미 데이터셋
# ============================================

def create_dummy_documents() -> List[Document]:
    """
    간단한 더미 초안 문서들.
    실제 프로젝트에서는 HF 데이터셋, 보고서, 에세이 등으로 교체 가능.

    → 일부러 문법/구조가 많이 틀린 버전으로 설정해서
      편집 액션이 점수에 더 잘 반영되도록 함.
    """
    docs = [
        Document(
            text=(
                "this is rough drft of essay about reinforce learning, "
                "it have many grammar mistake and structure is very messy and hard to read."
            )
        ),
        Document(
            text=(
                "in this report we describe experiment setup but sentences are not well organized, "
                "also there is many redundancy and unclear pronouns and abrupt topic changes."
            )
        ),
        Document(
            text=(
                "these research note summarize main idea of project but tone is very casual and "
                "there is lot of repetition and some sentence are incomplete and confusing."
            )
        ),
    ]
    return docs


# ============================================
# 6. 학습 + 평가 루프
# ============================================

def train_and_evaluate(
    num_episodes: int = 5,
    max_steps: int = 3,
):
    """
    실제 OpenRouter LLM을 호출하면서 Q-learning을 돌려본다.
    처음에는 LLM 호출 비용을 고려해서 에피소드/스텝 수를 작게 잡는 것을 권장한다.
    """
    random.seed(42)

    client = get_openrouter_client()

    documents = create_dummy_documents()
    editor = OpenRouterEditorLLM(
        client=client,
        model="openai/gpt-4o-mini",
        base_cost=0.02,
    )
    judge = OpenRouterJudgeLLM(
        client=client,
        model="openai/gpt-4.1",
    )
    env = EditingEnv(
        documents=documents,
        editor=editor,
        judge=judge,
        max_steps=max_steps,
        terminal_threshold=3.0,
        cost_lambda=1.0,
    )

    agent = QLearningAgent(
        num_actions=env.num_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.4,
        min_epsilon=0.05,
        epsilon_decay=0.9,
    )

    print("=== 학습 시작 (실제 LLM 호출) ===")
    reward_history: List[float] = []

    for episode in range(1, num_episodes + 1):
        state, text = env.reset()
        episode_reward = 0.0

        print(f"\n[에피소드 {episode}] 초기 상태: {state} (점수: {env.current_scores})")

        for t in range(max_steps):
            action_idx = agent.select_action(state)
            action_name = env.actions[action_idx]

            print(f"  Step {t+1}: 선택된 액션 = {action_name}")

            next_state, reward, done, info = env.step(action_idx)

            print(f"    이전 점수: {info.get('prev_scores')}")
            print(f"    새로운 점수: {info.get('new_scores')}")
            print(f"    delta_quality: {info.get('delta_quality', 0.0)}")
            print(f"    보상: {reward:.3f}, 다음 상태: {next_state}")

            agent.update(state, action_idx, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if done:
                print(f"  에피소드 종료 (reason={info.get('reason', 'unknown')})")
                break

        agent.decay_epsilon()
        reward_history.append(episode_reward)
        print(f"[에피소드 {episode}] 총 보상: {episode_reward:.3f}, ε={agent.epsilon:.3f}")

    print("\n=== 학습 종료 ===")

    # ======================
    # 학습된 정책으로 평가 (greedy)
    # ======================
    print("\n=== 학습된 정책 평가 (ε=0, greedy) ===")
    agent.epsilon = 0.0

    num_eval_episodes = 1
    for i in range(num_eval_episodes):
        state, text = env.reset()
        print(f"\n[평가 에피소드 {i+1}] 초기 문서:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        print("초기 점수:", env.current_scores)

        actions_taken = []
        for t in range(max_steps):
            action_idx = agent.select_action(state)
            action_name = env.actions[action_idx]
            actions_taken.append(action_name)

            before_text = env.current_text

            next_state, reward, done, info = env.step(action_idx)
            state = next_state

            print(f"\n[Step {t+1}] 액션 = {action_name}, 보상 = {reward:.3f}")
            print("  이전 점수:", info.get("prev_scores"))
            print("  새로운 점수:", info.get("new_scores"))
            print("  delta_quality:", info.get("delta_quality", 0.0))
            print("[편집 전]")
            print(before_text)
            print("[편집 후]")
            print(env.current_text)

            if done:
                print(f"\n에피소드 종료 (reason={info.get('reason', 'unknown')}, step={t+1})")
                break

        print("\n최종 점수:", env.current_scores)
        print("선택된 액션 시퀀스:", " -> ".join(actions_taken))
        print("\n최종 편집된 문서:")
        print("-" * 60)
        print(env.current_text)
        print("-" * 60)


# ============================================
# 7. 엔트리 포인트
# ============================================

if __name__ == "__main__":
    # LLM 호출 비용이 있으므로 처음에는 작은 값으로 테스트
    train_and_evaluate(num_episodes=5, max_steps=3)
