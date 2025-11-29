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
