from typing import Dict, Tuple
from dataclasses import dataclass, fields
import json

# internal
from llm.core import client
from environments.editing_env.eval.evaluator import AbstractQualityEvaluator


@dataclass
class Document:
    """
    문서 데이터 클래스
    """

    text: str


@dataclass
class DocumentScore:
    """
    문서 점수 클래스

    각 점수는 0~10의 값을 가짐
    """

    grammar: float = 5.0
    readability: float = 5.0
    coherence: float = 5.0
    overall: float = 5.0


@dataclass
class Action:
    fix_grammar: str = "fix_grammar"
    improve_clarity: str = "improve_clarity"
    make_concise: str = "make_concise"
    improve_structure: str = "improve_structure"
    make_academic: str = "make_academic"
    stop_editing: str = "stop_editing"

    @classmethod
    def as_list(cls):
        return [getattr(cls, f.name) for f in fields(cls)]


@dataclass
class AbstractScore:
    """
    Abstract 점수 클래스 (주석 처리용)
    각 점수는 0~10의 값을 가짐

    NOTE: score_abstract()를 활성화할 때 사용
    """

    structure: float = 5.0  # 구조적 완성도
    length: float = 5.0  # 길이 적절성
    academic: float = 5.0  # 학술적 스타일
    density: float = 5.0  # 정보 밀도
    clarity: float = 5.0  # 명확성
    coherence: float = 5.0  # 일관성
    overall: float = 5.0  # 전체 점수
    grade: str = "C (Acceptable)"  # 등급

class DocumentJudge:
    """입력된 문서의 품질을 평가하는 클래스"""

    def __init__(self):

        self.abstract_evaluator = AbstractQualityEvaluator(language="ko")

    # ================================================================
    # 아이디어 낸 평가 방법 (주석 처리 필요시 활성화)
    # ================================================================

    def score(self, document) -> DocumentScore:
        """
        Abstract 전용 평가 (500개 논문 분석 기반)

        AbstractQualityEvaluator의 항목을 DocumentScore로 자연스럽게 매핑:
        - grammar <- structure (구조적 완성도)
        - readability <- clarity (명확성)
        - coherence <- coherence (일관성)
        - overall <- overall (전체 점수)

        Args:
            document: Document 객체, 문자열(str), 또는 Document.text

        Returns:
            DocumentScore: 0~10 스케일 (env.py와 호환)
        """
        # 입력 타입에 따라 텍스트 추출
        if isinstance(document, str):
            text = document
        elif isinstance(document, Document):
            text = document.text
        elif hasattr(document, "text"):
            text = document.text
        else:
            raise ValueError(f"Unsupported type: {type(document)}")

        # AbstractQualityEvaluator 호출
        result = self.abstract_evaluator.evaluate_abstract(text)

        # 0~1 스케일을 0~10으로 변환 후 매핑
        grammar = result["structure"]["structure_completeness"] * 10.0
        readability = result["clarity"]["clarity_score"] * 10.0
        coherence = result["coherence"]["coherence_score"] * 10.0
        overall = result["overall_score"] * 10.0

        return DocumentScore(
            grammar=round(grammar, 1),
            readability=round(readability, 1),
            coherence=round(coherence, 1),
            overall=round(overall, 1),
        )
