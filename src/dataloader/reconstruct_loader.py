import json
from typing import List, Dict
from pathlib import Path

# internal
from config.paths import DATA_DIR

from environments.editing_env.components.component import Document

from utils.logger_factory import log
from utils.util import load_json


class DomesticReconstructDataLoader:
    def __init__(self, json_path=DATA_DIR / "reconstruct"):
        """
        국내 논문 초록을 재가공한 데이터를 로드하는 클래스
        """

        self.json_path = Path(json_path)
        if self.json_path.is_dir():
            self.json_paths = sorted(list(self.json_path.glob("*.json")))
        else:
            self.json_paths = [self.json_path]

        # doc_id를 키로 하는 path dictionary 생성
        self.json_path_dict = {}
        for p in self.json_paths:
            self.json_path_dict[p.stem] = p

        # doc_id list
        self.doc_ids = list(self.json_path_dict.keys())

        # 문서 개수
        self.docs_count = len(self.json_paths)

    def load_by_index(self, index: int) -> Dict:
        """
        index를 입력받아서 해당 index에 위치한 재구성 문서를 로드

        Args:
            index: 로드할 문서의 인덱스 (0부터 시작)

        Returns:
            Dict: 로드된 JSON 문서 내용

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        if index < 0 or index >= self.docs_count:
            raise IndexError(f"Index {index} out of range [0, {self.docs_count})")

        doc_path = self.json_paths[index]
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"데이터 로드 시 오류가 발생하였습니다.: {e}")
            return {}

    def load_by_doc_id(self, doc_id: str) -> Dict:
        """
        doc_id를 입력받아서 해당하는 재구성 문서를 로드

        Args:
            doc_id: 로드할 문서의 ID (파일명에서 확장자를 제외한 부분)

        Returns:
            Dict: 로드된 JSON 문서 내용

        Raises:
            KeyError: doc_id가 존재하지 않는 경우
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        if doc_id not in self.json_path_dict:
            raise KeyError(f"doc_id '{doc_id}' not found in reconstruct data")

        doc_path = self.json_path_dict[doc_id]
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log.warning("데이터 로드 시 오류가 발생하였습니다.", exc_info=True)
            return {}

    def dict_to_document(
        self, data: Dict, text_type: str = "abstract_noised"
    ) -> Document:
        """
        재구성 데이터 dict을 Document 클래스로 변환

        Args:
            data: 재구성 데이터 dict
            text_type: 사용할 텍스트 필드명
                - "abstract_original": 원본 초록
                - "abstract_reconstructed": 재구성된 초록 (구어체)
                - "abstract_noised": 노이즈가 추가된 초록 (기본값)

        Returns:
            Document: text 필드에 선택한 텍스트가 담긴 Document 객체
                     (해당 key가 없으면 text=None으로 설정하고 warning 발생)
        """
        text = data.get(text_type)

        if text is None:
            log.warning(
                f"'{text_type}' 키가 데이터에 존재하지 않습니다. Document.text를 None으로 설정합니다."
            )

        return Document(text=text)


class NoiseDataLoader:
    def __init__(self):
        """
        맞춤법 및 띄어쓰기 오류 데이터
        """

        # 국내 논문 초록 데이터 파일은 현재 하나이니까 명시적으로 선언
        self.data_path = (
            DATA_DIR
            / "paper_data"
            / "noise"
            / "paper_abstract_with_noise_20251125_002418.json"
        )

    def get_noise_text(self, max_docs=float("inf")) -> List[Document]:

        noise_texts = []

        try:
            data = load_json(self.data_path)

            results: List[dict] = data.get("results", [])
            for result in results:
                abstract_text = result.get("abstract_noise", "")
                if abstract_text:
                    noise_texts.append(Document(text=abstract_text))
                    if len(noise_texts) > max_docs:
                        break

        except Exception as e:
            log.info(f"[WARN] Error reading {self.data_path}: {e}")

        return noise_texts
