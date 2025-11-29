from typing import Dict, Tuple, override

# internal
from .base_env import EditingEnv
from .components.editor import OfflineSingleDocEditor
from .components.data import DocOfflineData
from .components.component import Document


class OfflineEditingEnv(EditingEnv):

    def __init__(
        self,
        dataloader: DocOfflineData = DocOfflineData(),
        jsonl_path=None,
        use_single_sequence=True,
        fixed_sequence_idx=0,
        *args,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader, *args, **kwargs)

        # jsonl_path가 없으면 에러
        if jsonl_path is None:
            raise ValueError("OfflineEditingEnv requires 'jsonl_path' parameter")

        if use_single_sequence:
            self.documents = [self.documents[fixed_sequence_idx]]

        self.jsonl_path = jsonl_path

        self.use_single_sequence = use_single_sequence
        # 오프라인 데이터 로드
        self.all_sequences = dataloader.sequences

        # 오버피팅 모드 설정
        self.fixed_sequence_idx = fixed_sequence_idx

        # OfflineSingleDocEditor로 교체
        self.editor = OfflineSingleDocEditor(self.jsonl_path)

        self.base_text = ""

    @override
    def _load_data(self):
        doc_idxes = [i for i in range(self.dataloader.total_docs)]

        documents = []
        for i in doc_idxes:
            documents.append(Document(self.dataloader.get_by_index(i)["base_text"]))

        return documents, doc_idxes

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

        # stop_editing: 종료
        if action == "stop_editing":
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
            score_delta = new_scores.overall - prev_overall

            if score_delta > 0:
                reward = score_delta * 3.0  # 긍정적 변화 강화
            else:
                reward = score_delta * 1.0

            # step별 비용 증가
            step_cost_multiplier = 1.0 + (self.current_step * 0.15)
            reward -= self.cost_lambda * usd_cost * step_cost_multiplier

            info["score_delta"] = score_delta
        else:
            # 기존 방식 보상 (5가지 평가 기준)
            d_structure = new_scores.structure - prev_scores.structure
            d_length = new_scores.length - prev_scores.length
            d_academic = new_scores.academic_style - prev_scores.academic_style
            d_density = new_scores.information_density - prev_scores.information_density
            d_clarity = new_scores.clarity - prev_scores.clarity
            do = new_scores.overall - prev_overall

            d_structure /= 2.0
            d_length /= 2.0
            d_academic /= 2.0
            d_density /= 2.0
            d_clarity /= 2.0
            do /= 2.0

            combined_delta = (
                d_structure + d_length + d_academic + d_density + d_clarity + do
            ) / 6.0

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
