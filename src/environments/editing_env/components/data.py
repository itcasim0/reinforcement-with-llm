import json
from typing import List, Dict, Tuple
from pathlib import Path

# internal
from config.paths import DATA_DIR
from utils.logger_factory import log


# TODO: 단일 jsonl과 dir을 전달받아서 둘다 처리할 수 있도록하고 인터페이스 적절히 수정
class DocOfflineData:
    def __init__(self, jsonl_path=DATA_DIR / "paper_data" / "offline"):
        """
        JSONL 파일을 로드하고 인덱스를 생성합니다.

        Args:
            jsonl_path: JSONL 파일 경로
        """

        self.jsonl_path = Path(jsonl_path)
        if self.jsonl_path.is_dir():
            self.jsonl_paths = sorted(list(self.jsonl_path.glob("*.jsonl")))
        else:
            self.jsonl_paths = [self.jsonl_path]

        # 문서 개수
        self.total_docs = len(self.jsonl_paths)

        # TODO: 추후 제거
        self.sequences, self.action_index = self._load_data()

    # TODO: 추후 제거
    def _load_data(self):

        # JSONL 파일 로드
        sequences = []
        action_index = {}
        with open(self.jsonl_paths[0], "r", encoding="utf-8") as f:
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

    def get_by_index(self, index: int) -> Dict:
        """
        주어진 index에 해당하는 jsonl 파일을 로드하여 특정 형식으로 반환합니다.

        Args:
            index: jsonl_paths 리스트의 인덱스

        Returns:
            {
                "base_text": 첫 번째 sequence의 base_text,
                "action_sequence": action_index 데이터
            }
        """
        if index < 0 or index >= self.total_docs:
            raise IndexError(
                f"Index {index}는 범위를 벗어났습니다. (0 ~ {self.total_docs - 1})"
            )

        target_jsonl = self.jsonl_paths[index]

        # JSONL 파일 로드
        sequences = []
        action_index = {}

        with open(target_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    sequences.append(record)

                    # actions로 인덱싱 (튜플로 변환하여 hashable하게 만듦)
                    actions_tuple = tuple(record["actions"])
                    action_index[actions_tuple] = record

        # 첫 번째 sequence의 base_text 추출
        base_text = sequences[0]["base_text"] if sequences else ""

        return {"base_text": base_text, "action_sequence": action_index}
