from typing import Dict, Tuple, List

# internal
from llm.core import client
from utils.logger_factory import log

from .data import SingleDocOfflineData


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


class OfflineSingleDocEditor:
    def __init__(self, base_cost: float = 0.02):
        self.data = SingleDocOfflineData()
        self.base_cost = base_cost
        self.base_token = 2000

    def edit(self, actions: List[str] | Tuple[str]) -> Tuple[str, Dict[str, float]]:
        sequence = self.data.get_sequence_by_actions(actions)
        steps = sequence.get("steps", [])

        if not steps:
            log.warning("Offline docs 데이터에 'steps' key가 없습니다.")
            steps = [{}]

        response: dict = steps[-1]

        edited_text = response.get("output_text", "")
        if not edited_text:
            log.warning("교정된 텍스트가 없습니다.")
            log.debug(response)

        cost_info = response.get("cost_info", {})
        if not cost_info:
            log.warning(
                f"비용 정보가 없어, 기본 값인 usd_cost={self.base_cost}, total_tokens={self.base_token}으로 설정합니다."
            )
            log.debug(response)
            cost_info = {"usd_cost": self.base_cost, "total_tokens": self.base_token}

        return edited_text, cost_info


