import json
from typing import List, Dict, Tuple

# internal
from config.paths import DATA_DIR
from utils.logger_factory import log


class SingleDocOfflineData:
    def __init__(self, jsonl_path=DATA_DIR / "paper_data" / "sequences_20251128_014521_tmp.jsonl"):
        """
        JSONL 파일을 로드하고 인덱스를 생성합니다.

        Args:
            jsonl_path: JSONL 파일 경로
        """

        self.jsonl_path = jsonl_path

        self.sequences, self.action_index = self._load_data()

    def _load_data(self):
        # JSONL 파일 로드
        sequences = []
        action_index = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sequences.append(record)

                    # actions로 인덱싱 (튜플로 변환하여 hashable하게 만듦)
                    actions_tuple = tuple(record["actions"])
                    action_index[actions_tuple] = record

        log.info(f"총 {len(sequences)}개의 시퀀스 로드 완료")

        return sequences, action_index

    def get_sequence_by_actions(self, actions: List[str] | Tuple[str]) -> Dict:
        """
        주어진 actions를 key로 하여 해당하는 sequence id 리스트를 반환합니다.

        Args:
            actions: action 리스트 (예: ["fix_grammar", "improve_clarity"])

        Returns:
            해당 actions를 가진 sequence들의 id 리스트
        """
        actions_tuple = tuple(actions)
        return self.action_index.get(actions_tuple, {})
