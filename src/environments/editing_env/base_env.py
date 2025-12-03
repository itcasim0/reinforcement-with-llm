from typing import Dict, Tuple
import random

from utils.logger_factory import log

from dataloader.reconstruct_loader import ReconstructDataLoader
from dataloader.offline_loader import OfflineDocumentLoader
from .components.component import DocumentJudge, DocumentScore, Action
from .components.editor import DocumentEditor


class EditingEnv:
    """
    문서 자동 교정 강화학습 환경

    - 상태: (grammar, readability, coherence, overall) 점수 (각 0~5 정수로 반올림)
    - 행동:
        0: fix_grammar
        1: improve_clarity
        2: make_concise
        3: improve_structure
        4: make_academic
        5: stop_editing

    Args:
        documents (List[Document]): 환경에서 사용할 문서 데이터 리스트
        max_steps (int): 에피소드 당 최대 허용 스텝 수
        terminal_threshold (float): 종료 판단을 위한 점수 임계값 (기본: 4.0)
        cost_lambda (float): LLM 비용(USD)에 대한 보상 페널티 가중치 (기본: 1.0, qwen 8b: 0.028$, gemma 3: 0.07$)
        repeat_penalty (float): 동일 액션을 반복해서 수행할 때 부여하는 페널티 (기본: 0.3)
    """

    def __init__(
        self,
        dataloader: ReconstructDataLoader | OfflineDocumentLoader,
        max_steps: 3,
        terminal_threshold: float = 9.5,
        cost_lambda: float = 1.0,
        repeat_penalty: float = 0.3,  # 같은 액션 반복 사용 시 패널티
        editor_model: str = "google/gemma-3-27b-it",
        base_cost: float = 0.02,
        step_penalty: float = 0.1,  # step 하나 당 패널티
    ):
        self.dataloader = dataloader
        self.documents, self.doc_idxes = self._load_data()
        self.max_steps = max_steps
        self.available_doc_idxes = self.doc_idxes.copy()
        self.editor = DocumentEditor(editor_model, base_cost)
        self.judge = DocumentJudge()
        self.terminal_threshold = terminal_threshold
        self.cost_lambda = cost_lambda
        self.repeat_penalty = repeat_penalty
        self.step_penalty = step_penalty

        self.actions = Action.as_list()
        self.num_actions = len(self.actions)

        # 현재 에피소드 상태
        self.current_text = ""
        self.current_score = DocumentScore()
        self.current_step: int = 0

        # 같은 액션 반복 패널티를 위해 이전에 사용한 액션 기록
        self.used_actions = set()

        # step마다 사용된 action을 기록하기 위함
        self.action_history = []

        # 강화 학습 환경 요약 출력
        self._env_summary()

    def _env_summary(self):
        log.info(
            f"""강화 학습 환경 요약:
학습에 사용할 데이터 수: {len(self.doc_idxes)}""")

    def _load_data(self):
        documents = self.dataloader.get_reconstructed_text(max_docs=5)
        return documents, [i for i in range(len(documents))]

    def reset(self):
        """
        에피소드 초기화
        - 랜덤 문서를 하나 선택 (비복원 추출 후 모두 추출하였다면 다시 리셋)
        - 평가 클래스를 통해 초기 문서 품질 점수 계산
        """
        self.current_step = 0
        self.used_actions = set()  # 에피소드 시작할 때 사용한 액션 기록 초기화
        self.action_history = []  # step마다 사용된 action을 기록하기 위함

        # 비복원 추출: 리스트가 비었으면 다시 채움
        if not self.available_doc_idxes:
            self.available_doc_idxes = self.doc_idxes.copy()

        # 무작위 인덱스 선택 후 pop
        picked = self.available_doc_idxes.pop(
            random.randrange(len(self.available_doc_idxes))
        )
        self.doc_index = picked
        base_doc = self.documents[picked]

        self.current_text = base_doc.text

        self.current_score = self.judge.score(self.current_text)
        state = self._scores_to_state(self.current_score)
        return state, self.current_text

    def _scores_to_state(
        self, scores: DocumentScore
    ) -> Tuple[float, float, float, float, float, float]:
        """
        점수를 그대로 float로 유지해서 상태로 사용.
        (예: 2.3, 2.7 같은 차이도 살려서 PPO에 전달)

        새로운 5가지 평가 기준 + overall:
        - structure: 구조적 완성도
        - length: 문장 길이 점수
        - academic_style: 학술적 형식 점수
        - information_density: 정보 밀도 점수
        - clarity: 명확성 점수
        - overall: 전체 점수
        """
        return (
            scores.structure,
            scores.length,
            scores.academic_style,
            scores.information_density,
            scores.clarity,
            scores.overall,
        )

    def _edit(self, action):
        return self.editor.edit(self.current_text, action)

    def step(
        self, action_index: int
    ) -> Tuple[Tuple[float, float, float, float], float, bool, Dict]:
        """
        환경 한 step 진행.
        - action_index: 액션 인덱스
        - 반환: (다음 상태, 보상, done, info)
        """
        assert 0 <= action_index < self.num_actions
        action = self.actions[action_index]
        self.action_history.append(action)

        prev_scores = self.current_score
        prev_overall = prev_scores.overall

        done = False
        info = {
            "action": action,
            "prev_scores": prev_scores,
        }

        # stop_editing: 편집 없이 종료
        if action == "stop_editing":
            final_reward = self._terminal_reward(prev_scores)
            reward = final_reward
            done = True
            info["reason"] = "stop_action"
            info["new_scores"] = prev_scores
            next_state = self._scores_to_state(prev_scores)

            # stop에서는 반복 패널티 적용 X
            self.used_actions.add(action)

            self.action_history.append(action)
            return next_state, reward, done, info

        # 문서 편집 클래스 호출
        edited_text, cost_info = self._edit(action)
        self.current_text = edited_text
        self.current_step += 1

        # LLM 실제 비용 (달러 기준)
        used_cost = cost_info.get("used_cost", self.editor.base_cost)
        total_tokens = cost_info.get("total_tokens", None)

        # 2) 문서 평가 호출 후 점수 업데이트
        new_scores = self.judge.score(self.current_text)
        self.current_score = new_scores
        new_overall = new_scores.overall

        # --- (1) 각 항목별 변화량 계산 (5가지 평가 기준) ---
        d_structure = new_scores.structure - prev_scores.structure
        d_length = new_scores.length - prev_scores.length
        d_academic = new_scores.academic_style - prev_scores.academic_style
        d_density = new_scores.information_density - prev_scores.information_density
        d_clarity = new_scores.clarity - prev_scores.clarity
        do = new_overall - prev_overall

        # --- (2) 10점 스케일 → 변화량 살짝 줄이기 ---
        d_structure /= 2.0
        d_length /= 2.0
        d_academic /= 2.0
        d_density /= 2.0
        d_clarity /= 2.0
        do /= 2.0

        # 6개 항목 평균 (equal weight)
        combined_delta = (
            d_structure + d_length + d_academic + d_density + d_clarity + do
        ) / 6.0

        # --- (3) step 보상 계산 ---
        if combined_delta >= 0:
            step_reward = combined_delta
        else:
            step_reward = -0.2 * abs(combined_delta)

        # LLM 호출 비용 패널티 (달러 기준)
        # 예: cost_lambda=10.0 이면 0.01달러 사용 시 -0.1 패널티
        step_reward -= self.cost_lambda * used_cost

        reward = step_reward

        # step별 비용 증가
        step_cost_multiplier = self.current_step * self.step_penalty
        reward -= step_cost_multiplier

        # 같은 액션 반복 패널티 (에피소드 내 중복 사용 시 패널티)
        repeat_penalty_applied = False
        if action in self.used_actions:
            # 이미 사용한 액션을 다시 쓰면 패널티
            repeat_penalty = self.repeat_penalty
            reward -= repeat_penalty
            repeat_penalty_applied = True

        # terminal_threshold 도달 시 조기 종료 (액션 실행 후 overall 점수 기준)
        if new_overall >= self.terminal_threshold:
            final_bonus = self._terminal_reward(new_scores)
            reward += final_bonus
            done = True
            info["reason"] = "threshold_reached"

        # 최대 스텝 도달 시 종료 보상 추가
        if self.current_step >= self.max_steps:
            final_bonus = self._terminal_reward(new_scores)
            reward += final_bonus
            done = True
            info["reason"] = "max_steps"

        # info 정리
        info["new_scores"] = new_scores
        info["combined_delta"] = combined_delta
        info["repeat_penalty"] = repeat_penalty_applied
        info["llm_cost_usd"] = used_cost
        info["llm_total_tokens"] = total_tokens

        # 마지막에 사용한 액션 업데이트
        self.used_actions.add(action)

        next_state = self._scores_to_state(new_scores)
        return next_state, reward, done, info

    def _terminal_reward(self, scores: DocumentScore) -> float:
        """
        에피소드 종료 시 추가 보상 (평가 지표 점수 평균 기반):

        - scores: 5가지 평가 기준 + overall (0~10)
        - 평균 점수(avg_score)를 0~10 → -1 ~ +1로 스케일링
        - 기본 값이 5인데, 평균이 5보다 이하일 경우 올바른 선택을 하지 못하였기에 마이너스
        - 평균이 5보다 클경우에는 비교적 올바른 선택을 했기 때문에 플러스
        - 만점은 1, 최하는 -1을 최종 reward로 추가 제공하기 위함

          예) avg=3.0 → (3 - 5) / 5 = -0.4
              avg=5.0 →  0.0
              avg=7.5 → +0.5
        """
        s = scores.structure
        l = scores.length
        a = scores.academic_style
        d = scores.information_density
        c = scores.clarity

        avg_score = (s + l + a + d + c) / 5.0  # 0~10
        # 0~10 → -1 ~ +1
        return (avg_score - 5.0) / 5.0
