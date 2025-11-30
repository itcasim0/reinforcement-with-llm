from typing import override

# internal
from .base_env import EditingEnv
from .components.editor import OfflineDocumentEditor
from .components.component import Document

from dataloader.offline_loader import OfflineDocumentLoader

from utils.logger_factory import log


class OfflineEditingEnv(EditingEnv):

    def __init__(
        self,
        dataloader: OfflineDocumentLoader = OfflineDocumentLoader(),
        use_single_sequence=True,
        fixed_sequence_idx=0,
        *args,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader, *args, **kwargs)

        if use_single_sequence:
            log.info("단일 문서 학습 환경 초기화 중...")
            self.documents = [self.documents[fixed_sequence_idx]]
            self.doc_idxes = [fixed_sequence_idx]

        # OfflineDocumentEditor 교체
        self.editor = OfflineDocumentEditor(dataloader)

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
        오프라인 편집 (OfflineDocumentEditor 사용)
        action_history를 전달하여 사전 계산된 결과 조회
        """
        return self.editor.edit(self.doc_index, self.action_history)
