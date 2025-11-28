from typing import List, Dict, Tuple
import random

from typing import override

from utils.logger_factory import log

from .components.component import Document, DocumentJudge, DocumentScore, Action
from .components.editor import DocumentEditor, OfflineSingleDocEditor
from .eval.strict_evaluator import StrictEvaluator


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
        cost_lambda (float): LLM 비용(USD)에 대한 보상 페널티 가중치 (기본: 1.0)
        repeat_penalty (float): 동일 액션을 반복해서 수행할 때 부여하는 페널티 (기본: 0.3)
    """

    def __init__(
        self,
        documents: List[Document],
        max_steps: 3,
        # TODO: terminal_threshold 적용될 수 있도록 코드 수정하기
        terminal_threshold: float = 9.5,
        cost_lambda: float = 1.0,
        repeat_penalty: float = 0.3,  # 같은 액션 반복 사용 시 패널티
        editor_model: str = "google/gemma-3-27b-it",
    ):
        self.documents = documents
        self.max_steps = max_steps
        self.available_documents = list(self.documents)
        self.editor = DocumentEditor(editor_model)
        self.judge = DocumentJudge()
        self.terminal_threshold = terminal_threshold
        self.cost_lambda = cost_lambda
        self.repeat_penalty = repeat_penalty

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

    def reset(self) -> Tuple[Tuple[int, int, int, int], str]:
        """
        에피소드 초기화
        - 랜덤 문서를 하나 선택 (비복원 추출 후 모두 추출하였다면 다시 리셋)
        - 평가 클래스를 통해 초기 문서 품질 점수 계산
        - 상태 (g, r, c, o)와 현재 텍스트 반환
        """
        self.current_step = 0
        self.used_actions = set()  # 에피소드 시작할 때 사용한 액션 기록 초기화
        self.action_history = []  # step마다 사용된 action을 기록하기 위함

        # 비복원 추출: 리스트가 비었으면 다시 채움
        if not self.available_documents:
            self.available_documents = list(self.documents)

        # 무작위 인덱스 선택 후 pop
        rand_idx = random.randrange(len(self.available_documents))
        base_doc = self.available_documents.pop(rand_idx)

        self.current_text = base_doc.text

        self.current_score = self.judge.score(self.current_text)
        state = self._scores_to_state(self.current_score)
        return state, self.current_text

    def _scores_to_state(
        self, scores: DocumentScore
    ) -> Tuple[float, float, float, float]:
        """
        점수를 그대로 float로 유지해서 상태로 사용.
        (예: 2.3, 2.7 같은 차이도 살려서 PPO에 전달)
        """
        return (
            scores.grammar,
            scores.readability,
            scores.coherence,
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
        usd_cost = cost_info.get("usd_cost", self.editor.base_cost)
        total_tokens = cost_info.get("total_tokens", None)

        # 2) 문서 평가 호출 후 점수 업데이트
        new_scores = self.judge.score(self.current_text)
        self.current_score = new_scores
        new_overall = new_scores.overall

        # --- (1) 각 항목별 변화량 계산 ---
        dg = new_scores.grammar - prev_scores.grammar
        dr = new_scores.readability - prev_scores.readability
        dc = new_scores.coherence - prev_scores.coherence
        do = new_overall - prev_overall

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

        # 같은 액션 반복 패널티 (에피소드 내 중복 사용 시 패널티)
        repeat_penalty_applied = False
        if action in self.used_actions:
            # 이미 사용한 액션을 다시 쓰면 패널티
            repeat_penalty = self.repeat_penalty
            reward -= repeat_penalty
            repeat_penalty_applied = True

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
        info["llm_cost_usd"] = usd_cost
        info["llm_total_tokens"] = total_tokens

        # 마지막에 사용한 액션 업데이트
        self.used_actions.add(action)

        next_state = self._scores_to_state(new_scores)
        return next_state, reward, done, info

    def _terminal_reward(self, scores: DocumentScore) -> float:
        """
        에피소드 종료 시 추가 보상 (4개 점수 평균 기반):

        - scores: "grammar", "readability", "coherence", "overall" (0~10)
        - 평균 점수(avg_score)를 0~10 → -1 ~ +1로 스케일링

          예) avg=3.0 → (3 - 5) / 5 = -0.4
              avg=5.0 →  0.0
              avg=7.5 → +0.5
        """
        g = scores.grammar
        r = scores.readability
        c = scores.coherence
        o = scores.overall

        avg_score = (g + r + c + o) / 4.0  # 0~10
        # 0~10 → -1 ~ +1
        return (avg_score - 5.0) / 5.0


# class OfflineEditingEnv(EditingEnv):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.editor = OfflineSingleDocEditor()

#     @override
#     def _edit(self, _):
#         return self.editor.edit(self.action_history)


class OfflineEditingEnv(EditingEnv):
    """
    오프라인 환경 - offline_ppo.py와 동일하게 동작

    기존 틀 유지:
    - __init__에서 *args, **kwargs 받기
    - super().__init__ 호출
    - _edit override
    - _load_data 메서드
    """

    def __init__(self, *args, **kwargs):
        # kwargs에서 오프라인 전용 파라미터 추출
        self.jsonl_path = kwargs.pop("jsonl_path", None)
        self.use_single_sequence = kwargs.pop("use_single_sequence", True)
        self.use_llm_judge = kwargs.pop("use_llm_judge", False)
        self.use_offline_reward = kwargs.pop("use_offline_reward", True)

        # jsonl_path가 없으면 에러
        if self.jsonl_path is None:
            raise ValueError("OfflineEditingEnv requires 'jsonl_path' parameter")

        # 더미 documents로 부모 클래스 초기화
        if "documents" not in kwargs or not kwargs.get("documents"):
            kwargs["documents"] = []

        super().__init__(*args, **kwargs)

        # OfflineSingleDocEditor로 교체
        self.editor = OfflineSingleDocEditor(self.jsonl_path)

        # 오프라인 데이터 로드
        self.all_sequences = []
        self.action_index = {}
        self._load_data()

        self.judge = StrictEvaluator()

        # 오버피팅 모드 설정
        self.fixed_sequence_idx = 0
        self.base_text = ""

    def _load_data(self):
        """JSONL 데이터 로드 및 인덱싱 (offline_ppo.py와 동일)"""
        import json

        self.all_sequences = []
        self.action_index = {}

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.all_sequences.append(record)
                    actions_tuple = tuple(record["actions"])
                    self.action_index[actions_tuple] = record

        log.info(f"[데이터 로드] 총 {len(self.all_sequences)}개 시퀀스")

    @override
    def reset(self) -> Tuple[Tuple[float, float, float, float], str]:
        """
        에피소드 초기화 (offline_ppo.py 방식)
        - use_single_sequence=True: 항상 첫 번째 시퀀스
        - use_single_sequence=False: 랜덤 선택
        """
        self.current_step = 0
        self.action_history = []
        self.used_actions = set()

        # 오버피팅 모드
        if self.use_single_sequence:
            seq = self.all_sequences[self.fixed_sequence_idx]
        else:
            seq = self.all_sequences[random.randint(0, len(self.all_sequences) - 1)]

        self.base_text = seq["base_text"]
        self.current_text = self.base_text

        # 초기 점수 계산
        self.current_score = self.judge.evaluate(self.current_text)
        state = self._scores_to_state(self.current_score)

        return state, self.current_text

    @override
    def _edit(self, _):
        """
        오프라인 편집 (OfflineSingleDocEditor 사용)
        action_history를 전달하여 사전 계산된 결과 조회
        """
        return self.editor.edit(self.action_history)

    @override
    def step(
        self, action_index: int
    ) -> Tuple[Tuple[float, float, float, float], float, bool, Dict]:
        """
        환경 step 실행 (offline_ppo.py 방식)
        """
        assert 0 <= action_index < self.num_actions
        action = self.actions[action_index]

        prev_scores = self.current_score
        done = False
        info = {
            "action": action,
            "prev_scores": prev_scores,
        }

        # stop_editing: 종료
        if action == "stop_editing":
            if self.use_offline_reward:
                # offline_ppo.py 스타일 stop 보상
                current_quality = (prev_scores.overall - 5.0) / 5.0

                if prev_scores.overall >= 7.0:
                    stop_bonus = 2.0
                elif prev_scores.overall >= 6.5:
                    stop_bonus = 1.0
                elif prev_scores.overall >= 6.0:
                    stop_bonus = 0.3
                elif prev_scores.overall >= 5.5:
                    stop_bonus = 0.0
                else:
                    stop_bonus = -1.0

                reward = current_quality + stop_bonus
                info["stop_bonus"] = stop_bonus
            else:
                # 기존 방식
                reward = self._terminal_reward(prev_scores)

            done = True
            info["reason"] = "stop_action"
            info["new_scores"] = prev_scores
            next_state = self._scores_to_state(prev_scores)
            self.used_actions.add(action)

            # action_history에 추가
            self.action_history.append(action)

            return next_state, reward, done, info

        # 액션 추가 (stop이 아닌 경우)
        self.action_history.append(action)
        self.current_step += 1

        # 오프라인 편집 수행
        edited_text, cost_info = self._edit(action)
        self.current_text = edited_text

        # LLM 비용
        usd_cost = cost_info.get("usd_cost", 0.02)
        total_tokens = cost_info.get("total_tokens", 2000)

        # 새로운 점수 계산
        new_scores = self.judge.score(self.current_text)
        self.current_score = new_scores

        # 보상 계산
        if self.use_offline_reward:
            # offline_ppo.py 스타일 보상
            score_delta = new_scores.overall - prev_scores.overall

            if score_delta > 0:
                reward = score_delta * 3.0  # 긍정적 변화 강화
            else:
                reward = score_delta * 1.0

            # step별 비용 증가
            step_cost_multiplier = 1.0 + (self.current_step * 0.15)
            reward -= self.cost_lambda * usd_cost * step_cost_multiplier

            info["score_delta"] = score_delta
        else:
            # 기존 방식 보상
            dg = new_scores.grammar - prev_scores.grammar
            dr = new_scores.readability - prev_scores.readability
            dc = new_scores.coherence - prev_scores.coherence
            do = new_scores.overall - prev_scores.overall

            dg /= 2.0
            dr /= 2.0
            dc /= 2.0
            do /= 2.0

            combined_delta = (dg + dr + dc + do) / 4.0

            if combined_delta >= 0:
                reward = combined_delta
            else:
                reward = -0.2 * abs(combined_delta)

            reward -= self.cost_lambda * usd_cost
            info["combined_delta"] = combined_delta

        # 반복 패널티
        repeat_penalty_applied = False
        if action in self.used_actions:
            reward -= self.repeat_penalty
            repeat_penalty_applied = True

        self.used_actions.add(action)

        # 최대 스텝 도달시 종료 보상
        if self.current_step >= self.max_steps:
            final_bonus = self._terminal_reward(new_scores)
            reward += final_bonus
            done = True
            info["reason"] = "max_steps"

        # info 정리
        info["new_scores"] = new_scores
        info["repeat_penalty"] = repeat_penalty_applied
        info["llm_cost_usd"] = usd_cost
        info["llm_total_tokens"] = total_tokens

        next_state = self._scores_to_state(new_scores)
        return next_state, reward, done, info
