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
        # OfflineDocumentLoader의 속성명에 맞게 docs_count, doc_ids 속성 추가
        if not hasattr(dataloader, 'docs_count'):
            dataloader.docs_count = dataloader.total_docs
        if not hasattr(dataloader, 'doc_ids'):
            dataloader.doc_ids = [i for i in range(dataloader.total_docs)]
        # OfflineDocumentLoader에는 dict_to_document 메서드가 없으므로 추가
        if not hasattr(dataloader, 'dict_to_document'):
            def dict_to_document(data, text_type=None):
                # OfflineDocumentLoader는 {"base_text": ..., "action_sequences": ...} 형태 반환
                if isinstance(data, dict) and "base_text" in data:
                    return Document(data["base_text"])
                return Document("")
            dataloader.dict_to_document = dict_to_document
        # OfflineDocumentLoader에는 load_by_index 대신 get_by_index를 사용
        if not hasattr(dataloader, 'load_by_index'):
            dataloader.load_by_index = dataloader.get_by_index
        
        super().__init__(dataloader=dataloader, *args, **kwargs)

        if use_single_sequence:
            log.info("단일 문서 학습 환경 초기화 중...")
            # documents 리스트를 미리 로드하지 않으므로 doc_idxes만 필터링
            self.doc_idxes = [fixed_sequence_idx]
            self.available_doc_idxes = self.doc_idxes.copy()

        # OfflineDocumentEditor 교체
        self.editor = OfflineDocumentEditor(dataloader)
        
        # doc_index 속성 초기화 (offline_env의 _edit에서 사용)
        self.doc_index = 0

    @override
    def _edit(self):
        """
        오프라인 편집 (OfflineDocumentEditor 사용)
        action_history를 전달하여 사전 계산된 결과 조회
        """
        return self.editor.edit(self.doc_index, self.action_history)
