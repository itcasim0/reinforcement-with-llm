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


class DocumentEditor:
    """
    입력된 문서와 행동을 토대로 편집하는 클래스

    NOTE: 한정된 자원과 보안 등을 고려하여 오픈소스 소형 LM을 활용하는 시나리오
    """

    def __init__(
        self,
        model: str = "google/gemma-3-27b-it",
        base_cost: float = 0.02,
        price_per_1k_tokens: float = 0.00015,
    ):
        self.model = model
        # 보상에서 사용할 LLM 호출 패널티
        self.base_cost = base_cost
        self.price_per_1k_tokens = price_per_1k_tokens

    def _system_prompt(self):
        return """당신은 한국어 글을 요청에 맞게 편집하는 글쓰기 보조 도우미입니다.
입력 글의 언어는 반드시 그대로 유지해야 합니다.
반드시 수정된 글만 출력하고, 설명이나 메타 코멘트는 쓰지 마세요."""

    def _user_prompt(self, text: str, action: str) -> str:
        """
        액션에 따라 LLM에 전달할 지시문을 생성.
        각 액션의 역할을 최대한 분리해서, 서로 다른 타입의 수정이 나오도록 설계.
        반드시 '수정된 글만 출력'하도록 강하게 요청.
        """
        if action == "fix_grammar":
            # 문법/맞춤법만 고치기
            instruction = (
                "다음 글에서 문법, 맞춤법, 띄어쓰기, 시제 오류만 수정해줘. "
                "문장 구조나 표현 방식은 가능한 한 그대로 유지하고, "
                "오직 틀린 부분만 고쳐. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "improve_clarity":
            # 의미는 유지하면서 표현/구조를 더 명확하게
            instruction = (
                "다음 글의 의미와 정보는 그대로 유지하되, "
                "표현과 문장 구조를 더 명확하고 이해하기 쉽게 고쳐줘. "
                "가능하면 문장의 길이를 크게 줄이지 말고, "
                "독자가 쉽게 따라갈 수 있도록 재구성해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "make_concise":
            # 중복/군더더기 제거 + 간결화
            instruction = (
                "다음 글에서 중복되거나 불필요한 부분을 줄이고, "
                "핵심 내용만 남기도록 더 간결하게 만들어줘. "
                "의미가 손실되지 않도록 주의하면서, 군더더기 표현을 제거해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "improve_structure":
            # 문단/문장 순서 재배치, 논리 흐름 개선
            instruction = (
                "다음 글의 문장과 문단 순서를 조정해서, "
                "논리적인 흐름이 더 자연스럽게 느껴지도록 재구성해줘. "
                "필요하다면 문장을 나누거나 이어서, 전개가 매끄럽게 보이게 해. "
                "내용 자체를 추가로 발명하지 말고, 기존 내용을 재구성하는 데 집중해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        else:
            # 혹시 모를 기타 액션
            instruction = (
                f"다음 글을 자연스럽게 다듬어줘. (액션: {action}) "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )

        prompt = f"""작업 지시: {instruction}

[원본 글]
{text}
"""
        return prompt

    def edit(self, text: str, action: str) -> Tuple[str, Dict[str, float]]:
        """
        실제 LLM을 호출하여 문서를 편집.
        - text: 현재 문서 (string)
        - action: 편집 액션 이름
        - 반환: (편집된 텍스트, {"usd_cost": ..., "total_tokens": ...})
        """
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.3,  # 약간의 다양성은 허용
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {"role": "user", "content": self._user_prompt(text, action)},
            ],
        )

        edited_text = response.choices[0].message.content.strip()

        # === 토큰 사용량/비용 계산 ===
        usage = getattr(response, "usage", None)
        if usage is not None:
            # openai 스타일: usage.total_tokens / prompt_tokens / completion_tokens
            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is None:
                # 혹시 필드명이 다르면 적당히 fallback
                total_tokens = getattr(usage, "prompt_tokens", 0) + getattr(
                    usage, "completion_tokens", 0
                )
            usd_cost = (total_tokens / 1000.0) * self.price_per_1k_tokens
        else:
            # usage 정보가 없으면 기본값 사용
            total_tokens = None
            usd_cost = self.base_cost

        cost_info = {
            "usd_cost": float(usd_cost),
            "total_tokens": float(total_tokens) if total_tokens is not None else None,
        }
        return edited_text, cost_info

    def general_edit(self, text: str) -> str:
        system_prompt = """당신은 한국어 글을 요청에 맞게 편집하는 글쓰기 보조 도우미입니다.
입력 글의 언어는 반드시 그대로 유지해야 합니다.
반드시 수정된 글만 출력하고, 설명이나 메타 코멘트는 쓰지 마세요."""

        instruction = """
글에서 문법, 맞춤법, 띄어쓰기, 시제 오류만 수정해줘.
문장 구조나 표현 방식은 가능한 한 그대로 유지하고,
오직 틀린 부분만 고쳐.
수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마.

글의 의미와 정보는 그대로 유지하되,
표현과 문장 구조를 더 명확하고 이해하기 쉽게 고쳐줘.
가능하면 문장의 길이를 크게 줄이지 말고,
독자가 쉽게 따라갈 수 있도록 재구성해. 
수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마.

중복되거나 불필요한 부분을 줄이고,
핵심 내용만 남기도록 더 간결하게 만들어줘.
의미가 손실되지 않도록 주의하면서, 군더더기 표현을 제거해.
수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마.

문장과 문단 순서를 조정해서,
논리적인 흐름이 더 자연스럽게 느껴지도록 재구성해줘.
필요하다면 문장을 나누거나 이어서, 전개가 매끄럽게 보이게 해.
내용 자체를 추가로 발명하지 말고, 기존 내용을 재구성하는 데 집중해.
수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마.
"""
        user_prompt = f"""작업 지시: {instruction}

[원본 글]
{text}
"""
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.3,  # 약간의 다양성은 허용
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()


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
