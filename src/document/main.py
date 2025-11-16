import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from datasets import load_dataset

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

    def __init__(self, client: OpenAI, model: str = "openai/gpt-4o-mini", base_cost: float = 0.02, price_per_1k_tokens: float = 0.00015):
        self.client = client
        self.model = model
        # 보상에서 사용할 LLM 호출 패널티
        self.base_cost = base_cost
        self.price_per_1k_tokens = price_per_1k_tokens

    def _make_prompt(self, text: str, action: str) -> str:
        """
        액션에 따라 LLM에 전달할 지시문을 생성.
        각 액션의 역할을 최대한 분리해서, 서로 다른 타입의 수정이 나오도록 설계.
        반드시 '수정된 글만 출력'하도록 강하게 요청.
        """
        if action == "fix_grammar":
            # 문법/맞춤법만 고치기
            instruction = (
                "다음 글에서 문법, 맞춤법, 띄어쓰기, 시제 오류만 수정해줘. "
                "문장 구조나 표현 방식은 가능한 한 그대로 유지하고, "
                "오직 틀린 부분만 고쳐. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "improve_clarity":
            # 의미는 유지하면서 표현/구조를 더 명확하게
            instruction = (
                "다음 글의 의미와 정보는 그대로 유지하되, "
                "표현과 문장 구조를 더 명확하고 이해하기 쉽게 고쳐줘. "
                "가능하면 문장의 길이를 크게 줄이지 말고, "
                "독자가 쉽게 따라갈 수 있도록 재구성해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "make_concise":
            # 중복/군더더기 제거 + 간결화
            instruction = (
                "다음 글에서 중복되거나 불필요한 부분을 줄이고, "
                "핵심 내용만 남기도록 더 간결하게 만들어줘. "
                "의미가 손실되지 않도록 주의하면서, 군더더기 표현을 제거해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "improve_structure":
            # 문단/문장 순서 재배치, 논리 흐름 개선
            instruction = (
                "다음 글의 문장과 문단 순서를 조정해서, "
                "논리적인 흐름이 더 자연스럽게 느껴지도록 재구성해줘. "
                "필요하다면 문장을 나누거나 이어서, 전개가 매끄럽게 보이게 해. "
                "내용 자체를 추가로 발명하지 말고, 기존 내용을 재구성하는 데 집중해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        else:
            # 혹시 모를 기타 액션
            instruction = (
                f"다음 글을 자연스럽게 다듬어줘. (액션: {action}) "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )

        prompt = f"""작업 지시: {instruction}

[원본 글]
{text}
"""
        return prompt

    def edit(self, text: str, action: str) -> Tuple[str, Dict[str, float]]:
        """
        실제 LLM을 호출하여 문서를 편집.
        - text: 현재 문서 (string)
        - action: 편집 액션 이름
        - 반환: (편집된 텍스트, {"usd_cost": ..., "total_tokens": ...})
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

        # === 토큰 사용량/비용 계산 ===
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # openai 스타일: usage.total_tokens / prompt_tokens / completion_tokens
            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is None:
                # 혹시 필드명이 다르면 적당히 fallback
                total_tokens = (
                    getattr(usage, "prompt_tokens", 0)
                    + getattr(usage, "completion_tokens", 0)
                )
            usd_cost = (total_tokens / 1000.0) * self.price_per_1k_tokens
        else:
            # usage 정보가 없으면 기본값 사용
            total_tokens = None
            usd_cost = self.base_cost

        cost_info = {
            "usd_cost": float(usd_cost),
            "total_tokens": float(total_tokens) if total_tokens is not None else None,
        }
        return edited_text, cost_info


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
        LLM에게 문서 품질을 0~10점으로 평가하게 하고, JSON 파싱 후 dict로 반환.
        """
        prompt = f"""
    다음 글의 품질을 0~10 점수로 평가해줘.

    각 항목:
    - grammar: 문법 및 맞춤법 정확성
    - readability: 읽기 쉬운 정도
    - coherence: 논리적 연결성과 흐름
    - overall: 전체적인 품질

    반드시 아래 JSON 형식 그대로만 출력해.
    설명 문장은 쓰지 말고, JSON만 출력해.

    예시:
    {{
    "grammar": 7.0,
    "readability": 6.5,
    "coherence": 7.0,
    "overall": 6.5
    }}

    평가할 글:
    {text}
    """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 글의 품질을 평가하는 심사위원입니다. "
                            "문법, 가독성, 논리적 일관성, 전체 품질을 0~10 점수로 평가합니다. "
                            "반드시 JSON 형식만 출력하세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        raw = resp.choices[0].message.content.strip()

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_str = raw[start:end + 1]
            else:
                json_str = raw

            data = json.loads(json_str)

            def safe(v: float) -> float:
                try:
                    x = float(v)
                except Exception:
                    x = 5.0    # 10점 만점 기준, 중간값
                # ✅ 0~10으로 클램핑
                return max(0.0, min(10.0, x))

            scores = {
                "grammar": safe(data.get("grammar", 5.0)),
                "readability": safe(data.get("readability", 5.0)),
                "coherence": safe(data.get("coherence", 5.0)),
                "overall": safe(data.get("overall", 5.0)),
            }
        except Exception as e:
            print("[경고] Judge LLM JSON 파싱 실패, 기본값 사용:", e)
            scores = {
                "grammar": 5.0,
                "readability": 5.0,
                "coherence": 5.0,
                "overall": 5.0,
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

        # 같은 액션 반복 패널티를 위해 직전 액션 기억
        self.last_action = None

    def reset(self) -> Tuple[Tuple[int, int, int, int], str]:
        """
        에피소드 초기화.
        - 랜덤 문서를 하나 선택
        - 평가 LLM으로 초기 품질 점수 계산
        - 상태 (g, r, c, o)와 현재 텍스트 반환
        """
        self.current_step = 0
        self.last_action = None  # ★ 에피소드 시작할 때 직전 액션 초기화

        base_doc = random.choice(self.documents)
        self.current_text = base_doc.text

        self.current_scores = self.judge.score(self.current_text)
        state = self._scores_to_state(self.current_scores)
        return state, self.current_text

    def _scores_to_state(self, scores: Dict[str, float]) -> Tuple[float, float, float, float]:
        """
        점수를 그대로 float로 유지해서 상태로 사용.
        (예: 2.3, 2.7 같은 차이도 살려서 PPO에 전달)
        """
        return (
            scores["grammar"],
            scores["readability"],
            scores["coherence"],
            scores["overall"],
        )

    def step(self, action_index: int) -> Tuple[Tuple[float, float, float, float], float, bool, Dict]:
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
        info = {
            "action": action,
            "prev_scores": prev_scores,
        }

        # ================================
        # 0) stop_editing: 편집 없이 종료
        # ================================
        if action == "stop_editing":
            final_reward = self._terminal_reward(prev_scores)
            reward = final_reward
            done = True
            info["reason"] = "stop_action"
            info["new_scores"] = prev_scores
            next_state = self._scores_to_state(prev_scores)

            # stop에서는 반복 패널티 적용 X
            self.last_action = action
            return next_state, reward, done, info

        # ================================
        # 1) 편집 LLM 호출
        # ================================
        edited_text, cost_info = self.editor.edit(self.current_text, action)
        self.current_text = edited_text
        self.current_step += 1

        # LLM 실제 비용 (달러 기준)
        usd_cost = cost_info.get("usd_cost", self.editor.base_cost)
        total_tokens = cost_info.get("total_tokens", None)

        # ================================
        # 2) 평가 LLM 호출 후 점수 업데이트
        # ================================
        new_scores = self.judge.score(self.current_text)
        self.current_scores = new_scores
        new_overall = new_scores["overall"]

        # --- (1) 각 항목별 변화량 계산 ---
        dg = new_scores["grammar"]     - prev_scores["grammar"]
        dr = new_scores["readability"] - prev_scores["readability"]
        dc = new_scores["coherence"]   - prev_scores["coherence"]
        do = new_overall               - prev_overall

        # --- (2) 10점 스케일 → 변화량 살짝 줄이기 ---
        dg /= 2.0
        dr /= 2.0
        dc /= 2.0
        do /= 2.0

        # 4개 항목 평균 (equal weight)
        combined_delta = (dg + dr + dc + do) / 4.0

        # --- (3) step 보상 계산 ---
        if combined_delta >= 0:
            step_reward = combined_delta
        else:
            step_reward = -0.2 * abs(combined_delta)

        # LLM 호출 비용 패널티 (달러 기준)
        # 예: cost_lambda=10.0 이면 0.01달러 사용 시 -0.1 패널티
        step_reward -= self.cost_lambda * usd_cost

        reward = step_reward

        # ================================
        # 3) 같은 액션 반복 패널티
        # ================================
        repeat_penalty_applied = False
        if self.last_action is not None and self.last_action == action:
            # 같은 액션을 연달아 쓰면 약간 패널티 → 다른 액션 탐색 유도
            repeat_penalty = 0.3   # 0.1~0.3 사이에서 조정해봐도 좋음
            reward -= repeat_penalty
            repeat_penalty_applied = True

        # ================================
        # 4) 최대 스텝 도달 시 종료 보상 추가
        # ================================
        if self.current_step >= self.max_steps:
            final_bonus = self._terminal_reward(new_scores)
            reward += final_bonus
            done = True
            info["reason"] = "max_steps"

        # info 정리
        info["new_scores"] = new_scores
        info["combined_delta"] = combined_delta
        info["repeat_penalty"] = repeat_penalty_applied
        info["llm_cost_usd"] = usd_cost
        info["llm_total_tokens"] = total_tokens

        # 마지막에 직전 액션 업데이트
        self.last_action = action

        next_state = self._scores_to_state(new_scores)
        return next_state, reward, done, info

    def _terminal_reward(self, scores: Dict[str, float]) -> float:
        """
        에피소드 종료 시 추가 보상 (4개 점수 평균 기반):

        - scores: {"grammar", "readability", "coherence", "overall"} (0~10)
        - 평균 점수(avg_score)를 0~10 → -1 ~ +1로 스케일링

          예) avg=3.0 → (3 - 5) / 5 = -0.4
              avg=5.0 →  0.0
              avg=7.5 → +0.5
        """
        g = scores["grammar"]
        r = scores["readability"]
        c = scores["coherence"]
        o = scores["overall"]

        avg_score = (g + r + c + o) / 4.0  # 0~10
        # 0~10 → -1 ~ +1
        return (avg_score - 5.0) / 5.0


# ============================================
# 4. PPO 에이전트
# ============================================

class PPOAgent(nn.Module):
    """
    Q-learning 대신 사용할 PPO 에이전트.

    - 상태: (g, r, c, o) 4차원 정수 튜플 (0~5)
    - 액션: 0 ~ num_actions-1 (fix_grammar, improve_clarity, make_concise, improve_structure, stop_editing)
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.95,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        K_epochs: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs

        # ====== 정책 + 가치 네트워크 (공유 본체 + actor/critic 헤드) ======
        hidden_dim = 64
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # ====== Rollout 버퍼 ======
        self.states: List[Tuple[int, int, int, int]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def _to_tensor(self, states_list):
        """
        상태 리스트 [(g,r,c,o), ...] -> (batch, 4) 텐서
        0~10 범위를 0~1로 스케일링해서 넣음.
        """
        arr = torch.tensor(states_list, dtype=torch.float32)
        arr = arr / 10.0  # 10점 만점 기준 정규화
        return arr

    def forward(self, state_tensor: torch.Tensor):
        """
        state_tensor: (batch, state_dim)
        반환: logits (배치 x num_actions), values (배치,)
        """
        x = self.base(state_tensor)
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values

    # ====== 행동 선택 (학습 시: stochastic) ======
    def select_action(self, state: Tuple[int, int, int, int]):
        """
        학습 단계에서 사용하는 행동 선택:
        - 정책 분포에서 샘플링 (exploration 포함)
        - 함께 log_prob, value 를 반환해 buffer에 저장.
        """
        state_tensor = self._to_tensor([state])  # (1, state_dim)
        with torch.no_grad():
            logits, value = self.forward(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action_idx = int(action.item())
        log_prob_value = float(log_prob.item())
        value_scalar = float(value.item())
        return action_idx, log_prob_value, value_scalar

    # ====== 평가용 행동 선택 (deterministic, greedy) ======
    def act_greedy(self, state: Tuple[int, int, int, int]) -> int:
        """
        평가 단계에서 사용하는 행동 선택:
        - argmax(logits)로 결정적 행동 선택.
        """
        state_tensor = self._to_tensor([state])
        with torch.no_grad():
            logits, _ = self.forward(state_tensor)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())

    # ====== Rollout 버퍼에 transition 추가 ======
    def store_transition(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    # ====== PPO 업데이트 ======
    def update(self):
        """
        수집된 rollout(여러 step/에피소드)을 가지고 PPO 업데이트 수행.
        여기서는 간단하게:
        - Monte Carlo returns (역방향 누적)
        - Advantage = returns - values
        - Advantage 정규화
        - K_epochs 동안 클리핑된 surrogate objective로 업데이트
        """
        if len(self.states) == 0:
            return  # 비어 있으면 스킵

        # --- 텐서로 변환 ---
        states_tensor = self._to_tensor(self.states)           # (N, state_dim)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)   # (N,)
        old_log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32)  # (N,)
        values_tensor = torch.tensor(self.values, dtype=torch.float32)  # (N,)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)  # (N,)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32)    # (N,)

        # --- Returns 계산 (역순) ---
        returns = []
        G = 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # --- Advantage 계산 ---
        advantages = returns_tensor - values_tensor

        # 안정성을 위해 정규화 (샘플이 2개 이상일 때만)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        else:
            # 하나밖에 없으면 평균만 빼주고 그대로 사용
            advantages = advantages - advantages.mean()

        # --- PPO 클리핑 업데이트 ---
        for _ in range(self.K_epochs):
            logits, values = self.forward(states_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs_tensor)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values, returns_tensor)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # --- 버퍼 비우기 ---
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

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

def create_coedit_documents(
    split: str = "train",
    max_samples: int = 100,
    task_filter: List[str] = None,
) -> List[Document]:
    """
    Hugging Face의 grammarly/coedit에서 src를 가져와 Document 리스트로 만든다.
    - split: 'train' 또는 'validation'
    - max_samples: LLM 비용 때문에 너무 많이 쓰지 않도록 상한
    - task_filter: ['gec'] 처럼 특정 task만 고르고 싶을 때

    TODO : Fix grammar errors: prefix 제거
    """
    ds = load_dataset("grammarly/coedit", split=split)

    docs: List[Document] = []
    count = 0

    for row in ds:
        task = row.get("task", None)
        src = row.get("src", None)

        if src is None:
            continue

        # task 필터가 있으면 적용 (예: grammar 위주로 하고 싶으면 ['gec'])
        if task_filter is not None:
            if task not in task_filter:
                continue

        docs.append(Document(text=src))
        count += 1

        if count >= max_samples:
            break

    print(f"[INFO] Loaded {len(docs)} documents from grammarly/coedit ({split})")
    return docs

def create_pararev_documents(
    split: str = "train",
    subset: str = "pararev_annot_subset",
    max_docs: int = 50,
) -> List[Document]:
    """
    taln-ls2n/pararev 에서 parag_1만 뽑아서 Document 리스트로 만드는 함수.
    학습 비용 줄이려고 max_docs로 개수 제한.
    """
    ds = load_dataset("taln-ls2n/pararev", subset, split=split)
    
    docs = []
    for i, ex in enumerate(ds):
        if i >= max_docs:
            break
        text = ex.get("parag_1", "").strip()
        if not text:
            continue
        docs.append(Document(text=text))
    return docs

# ============================================
# 6. 학습 + 평가 루프
# ============================================

def train_and_evaluate(
    num_episodes: int = 5,
    max_steps: int = 3,
):
    """
    실제 OpenRouter LLM을 호출하면서 PPO를 돌려본다.
    """
    random.seed(42)
    torch.manual_seed(42)

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
        terminal_threshold=6.0,
        cost_lambda=1.0,
    )

    # ====== QLearningAgent 대신 PPOAgent 사용 ======
    agent = PPOAgent(
        state_dim=4,
        num_actions=env.num_actions,
        gamma=0.95,
        lr=3e-4,
        clip_epsilon=0.2,
        K_epochs=4,
    )

    print("=== 학습 시작 (실제 LLM 호출, PPO) ===")
    reward_history: List[float] = []

    for episode in range(1, num_episodes + 1):
        state, text = env.reset()
        episode_reward = 0.0

        print(f"\n[에피소드 {episode}] 초기 상태: {state} (점수: {env.current_scores})")

        for t in range(max_steps):
            # ----- 행동 선택 (stochastic policy) -----
            action_idx, log_prob, value = agent.select_action(state)
            action_name = env.actions[action_idx]
            print(f"  Step {t+1}: 선택된 액션 = {action_name}")

            # ----- 환경 step -----
            next_state, reward, done, info = env.step(action_idx)

            print(f"    이전 점수: {info.get('prev_scores')}")
            print(f"    새로운 점수: {info.get('new_scores')}")
            print(f"    delta_quality: {info.get('delta_quality', 0.0)}")
            print(f"    보상: {reward:.3f}, 다음 상태: {next_state}")

            # ----- transition을 PPO 버퍼에 저장 -----
            agent.store_transition(
                state=state,
                action=action_idx,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
            )

            episode_reward += reward
            state = next_state

            if done:
                print(f"  에피소드 종료 (reason={info.get('reason', 'unknown')})")
                break

        # ====== 에피소드 끝날 때 PPO 업데이트 ======
        agent.update()

        reward_history.append(episode_reward)
        print(f"[에피소드 {episode}] 총 보상: {episode_reward:.3f}")

    print("\n=== 학습 종료 ===")

    # ======================
    # 학습된 정책으로 평가 (greedy)
    # ======================
    print("\n=== 학습된 정책 평가 (greedy, PPO) ===")

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
            # 평가에서는 argmax(logits)로 greedy 행동 선택
            action_idx = agent.act_greedy(state)
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
