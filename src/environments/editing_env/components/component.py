from dataclasses import dataclass, fields

# internal
from environments.editing_env.components.eval.evaluator import AbstractQualityEvaluator


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

    각 점수는 0~10의 값을 가지며, 기본 값은 중간인 5.0을 부여
    """

    structure: float = 5.0  # 구조적 완성도
    length: float = 5.0  # 문장 길이 점수
    academic_style: float = 5.0  # 학술적 형식 점수
    information_density: float = 5.0  # 정보 밀도 점수
    clarity: float = 5.0  # 명확성 점수
    overall: float = 5.0  # 전체 점수


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


class DocumentEvaluator:
    """입력된 문서의 품질을 평가하는 클래스"""

    def __init__(self):

        self.abstract_evaluator = AbstractQualityEvaluator(language="ko")

    def score(self, document) -> DocumentScore:
        """
        Abstract 전용 평가 (500개 논문 분석 기반)

        AbstractQualityEvaluator의 5가지 평가 기준을 그대로 사용:
        - structure: 구조적 완성도
        - length: 문장 길이 점수
        - academic_style: 학술적 형식 점수
        - information_density: 정보 밀도 점수
        - clarity: 명확성 점수
        - overall: 전체 점수

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

        # 0~1점 스케일을 0~10점으로 변환
        structure = result["structure"]["structure_completeness"] * 10.0
        length = result["length"]["overall_length_score"] * 10.0
        academic_style = result["academic_style"]["academic_style_score"] * 10.0
        information_density = (
            result["information_density"]["information_density_score"] * 10.0
        )
        clarity = result["clarity"]["clarity_score"] * 10.0
        overall = result["overall_score"] * 10.0

        return DocumentScore(
            structure=round(structure, 1),
            length=round(length, 1),
            academic_style=round(academic_style, 1),
            information_density=round(information_density, 1),
            clarity=round(clarity, 1),
            overall=round(overall, 1),
        )
