import json
from typing import Dict
from pathlib import Path

# internal
from config.paths import DATA_DIR


class CacheDocumentLoader:
    def __init__(self, cache_dir=DATA_DIR / "paper_data" / "cache"):
        """
        강화학습 진행하면서 재사용이 가능한 문서를 로드, 저장 하는 클래스

        캐시 저장 파일 명 규칙은 {doc_id}.json 형태로 저장되어야 함
        """

        # 캐시 문서 파일 로드
        self.cache_dir = Path(cache_dir)
        if self.cache_dir.is_dir():
            self.cache_paths = sorted(list(self.cache_dir.glob("*.json")))
        elif self.cache_dir.is_file():
            self.cache_paths = [self.cache_dir]
        else:
            self.cache_paths = []

        # doc_id를 키로 하는 path dictionary 생성
        self.cache_dict = {}
        for p in self.cache_paths:
            self.cache_dict[p.stem] = p

        # 문서 개수
        self.total_docs = len(self.cache_paths)

    def load_by_index(self, index: int) -> Dict:
        """
        index를 입력받아서 해당 index에 위치한 캐시 문서를 로드

        Args:
            index: 로드할 문서의 인덱스 (0부터 시작)

        Returns:
            Dict: 로드된 JSON 문서 내용

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        if index < 0 or index >= self.total_docs:
            raise IndexError(f"Index {index} out of range [0, {self.total_docs})")

        cache_path = self.cache_paths[index]
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def load_by_doc_id(self, doc_id: str) -> Dict:
        """
        doc_id를 입력받아서 해당하는 캐시 문서를 로드

        Args:
            doc_id: 로드할 문서의 ID (파일명에서 확장자를 제외한 부분)

        Returns:
            Dict: 로드된 JSON 문서 내용 (tuple 키는 자동으로 복원됨)

        Raises:
            KeyError: doc_id가 존재하지 않는 경우
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        if doc_id not in self.cache_dict:
            raise KeyError(f"doc_id '{doc_id}' not found in cache")

        cache_path = self.cache_dict[doc_id]
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None

        # JSON에서 로드한 문자열 키를 tuple로 변환
        converted_data = {}
        for key, value in data.items():
            try:
                # 문자열을 tuple로 변환 (예: "('fix_grammar',)" -> ('fix_grammar',))
                tuple_key = eval(key)
                if isinstance(tuple_key, tuple):
                    converted_data[tuple_key] = value
                else:
                    converted_data[key] = value
            except:
                # 변환 실패 시 원래 키 사용
                converted_data[key] = value

        return converted_data

    def save(self, data: Dict, doc_id: str) -> None:
        """
        dict와 doc_id를 전달받아서 JSON 파일로 저장

        Args:
            data: 저장할 문서 데이터 (dict 형태, tuple 키는 자동으로 문자열로 변환됨)
            doc_id: 저장할 문서의 ID (파일명이 됨)

        Returns:
            None

        Raises:
            OSError: 파일 저장 중 오류가 발생한 경우
        """
        # 캐시 디렉토리가 없으면 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 파일 경로 생성
        cache_path = self.cache_dir / f"{doc_id}.json"

        # tuple 키를 문자열로 변환
        converted_data = {}
        for key, value in data.items():
            if isinstance(key, tuple):
                # tuple을 문자열로 변환 (예: ('fix_grammar',) -> "('fix_grammar',)")
                str_key = str(key)
                converted_data[str_key] = value
            else:
                converted_data[key] = value

        # JSON 파일로 저장
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

        # 새로 저장된 파일을 cache_paths와 cache_dict에 추가
        if cache_path not in self.cache_paths:
            self.cache_paths.append(cache_path)
            self.cache_paths.sort()
            self.total_docs = len(self.cache_paths)

        self.cache_dict[doc_id] = cache_path
