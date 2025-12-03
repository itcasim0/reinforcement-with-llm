from typing import Dict, Tuple, List, override

# external
from openai import OpenAI

# internal
from llm.core import client
from utils.logger_factory import log

from dataloader.offline_loader import OfflineDocumentLoader
from dataloader.cache_loader import CacheDocumentLoader

client: OpenAI = client


class DocumentEditor:
    """
    입력된 문서와 행동을 토대로 편집하는 클래스

    NOTE: 한정된 자원과 보안 등을 고려하여 오픈소스 소형 LM을 활용하는 시나리오

    NOTE: 일단 모델에 대한 정보는 수동으로 기입

    base_cost는 정보가 없을 경우 기본으로 사용하는 값
    price_per_1m_tokens는 실제 input 가격 (현재는 input만 추후 output 도 고려?)

    """

    def __init__(
        self,
        model: str = "google/gemma-3-27b-it",
        base_cost: float = 0.02,
        price_per_1m_tokens: float = 0.028,
    ):
        self.model = model
        # 보상에서 사용할 LLM 호출 패널티
        self.base_cost = base_cost
        self.price_per_1m_tokens = price_per_1m_tokens

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
                "정량적 표현에 대해서는 적절히 잘 표현해줘. "
                "다음의 수식언은 불필요하니까 제거해줘. "
                'FILLER_WORDS = ["매우", "아주", "상당히", "조금", "약간", "다소", "꽤"], FILLERS = ["매우 ", "아주 ","상당히 ","다소 ","꽤 ","어느 정도","기본적으로","일반적으로 말해서","말하자면","이를테면","다양한 ","여러 가지 ",]'
                "다음과 같은 모호한 표현도 사용하지 않도록 해줘. "
                '{"VAGUE_TERMS": ["여러", "다양한", "몇몇", "일부", "종종", "때때로", "가끔", "많은"], "VAGUE_PATTERNS": ["일지도 모르는", "일지도 모를", "있을지도 모르는", "아닐까", "않을까", "일 것이다", "좀 ", "약간 ", "조금 ", "가상의", "어떤 ", "그런 ", "같은 것", "라는 것", "라고 하는", "등등", "기타 등등"]}'
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "make_concise":
            # 중복/군더더기 제거 + 간결화
            instruction = (
                "다음 글에서 중복되거나 불필요한 부분을 줄이고, "
                "핵심 내용만 남기도록 더 간결하게 만들어줘. "
                "의미가 손실되지 않도록 주의하면서, 군더더기 표현을 제거해. "
                "글의 길이는 다음의 조건을 참고하여 적절히 구성해줘. "
                "OPTIMAL_WORD_COUNT_MIN = 200, OPTIMAL_WORD_COUNT_MAX = 500, OPTIMAL_WORDS_PER_SENTENCE_MIN = 20, OPTIMAL_WORDS_PER_SENTENCE_MAX = 40, TARGET_WORDS_PER_SENTENCE = 30"
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "improve_structure":
            # 문단/문장 순서 재배치, 논리 흐름 개선
            instruction = (
                "다음 글의 문장과 문단 순서를 조정해서, "
                "논리적인 흐름이 더 자연스럽게 느껴지도록 재구성해줘. "
                "필요하다면 문장을 나누거나 이어서, 전개가 매끄럽게 보이게 해. "
                "내용 자체를 추가로 발명하지 말고, 기존 내용을 재구성하는 데 집중해. "
                "입력된 내용을 토대로 배경, 목적, 방법, 결과, 결론의 흐름으로 구성하면 돼. "
                "적절한 내용이 없다고 임의로 생성하거나 하면 안돼."
                "다음의 내용을 참고해서 구조에 대한 표현을 할 수 있어"
                '{"background":["배경","맥락","동기","문제","과제","기존","현재"],"objective":["목적","목표","연구","조사","탐구","분석","규명","밝히"],"method":["방법","접근","기법","알고리즘","프레임워크","모델","제안","제시","개발","설계"],"result":["결과","발견","입증","보여","달성","성능","실험","분석","확인"],"conclusion":["결론","시사점","제안","기여","의의","중요","기대","향후"]}'
                "문장 연결 시에는 '그러나', '하지만', '또한', '더욱이', '따라서', '그러므로', "
                "'그럼에도', '결과적으로', '특히', '구체적으로', '즉', '반면', '한편', '나아가', "
                "'이에', '이를 통해'와 같은 연결어를 자연스럽게 활용해. "
                "수정된 글만 출력하고, 설명이나 코멘트는 쓰지 마."
            )
        elif action == "make_academic":
            # 학술적/논문 초록 스타일로 변환
            instruction = (
                "다음 글을 한국어 학술 논문 초록 스타일에 가깝게 다듬어줘. "
                "내용과 정보는 바꾸지 말고, 표현과 어투를 더 형식적이고 객관적으로 바꿔. "
                "필요하다면 '본 연구', '본 논문', '저자는', '필자는', '우리는' 과 같은 "
                "연구 주체 표현을 적절히 사용할 수 있어. "
                "문장 연결 시에는 '그러나', '하지만', '또한', '더욱이', '따라서', '그러므로', "
                "'그럼에도', '결과적으로', '특히', '구체적으로', '즉', '반면', '한편', '나아가', "
                "'이에', '이를 통해'와 같은 학술적 연결어를 자연스럽게 활용해. "
                "한국어 피동/수동 표현(예: '~되었다', '~되어', '~된다', '~되는', '~되고', "
                "'~됨으로써', '~이루어졌다', '~이루어진', '~수행되었다', '~수행된')을 "
                "남용하지 않는 선에서 사용해, 보다 학술적인 톤을 만들어줘. "
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
        - 반환: (편집된 텍스트, {"used_cost": ..., "total_tokens": ...})
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
            used_cost = (total_tokens / 1000000.0) * self.price_per_1m_tokens
        else:
            # usage 정보가 없으면 기본값 사용
            total_tokens = None
            used_cost = self.base_cost

        cost_info = {
            "used_cost": float(used_cost),
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

한국어 학술 논문 초록 스타일에 가깝게 다듬어줘.
내용과 정보는 바꾸지 말고, 표현과 어투를 더 형식적이고 객관적으로 바꿔.
필요하다면 '본 연구', '본 논문', '저자는', '필자는', '우리는' 과 같은
연구 주체 표현을 적절히 사용할 수 있어.
문장 연결 시에는 '그러나', '하지만', '또한', '더욱이', '따라서', '그러므로',
'그럼에도', '결과적으로', '특히', '구체적으로', '즉', '반면', '한편', '나아가',
'이에', '이를 통해'와 같은 학술적 연결어를 자연스럽게 활용해.
한국어 피동/수동 표현(예: '~되었다', '~되어', '~된다', '~되는', '~되고',
'~됨으로써', '~이루어졌다', '~이루어진', '~수행되었다', '~수행된')을
남용하지 않는 선에서 사용해, 보다 학술적인 톤을 만들어줘.
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


class OfflineDocumentEditor(DocumentEditor):
    def __init__(
        self,
        dataloader: OfflineDocumentLoader = OfflineDocumentLoader(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader

    @override
    def edit(
        self, doc_index, actions: List[str] | Tuple[str]
    ) -> Tuple[str, Dict[str, float]]:
        offline_doc = self.dataloader.get_by_index(doc_index)
        action_sequences: Dict = offline_doc.get("action_sequences")
        sequence: Dict = action_sequences.get(tuple(actions))
        steps = sequence.get("steps", [])
        if not steps:
            log.warning("Offline docs 데이터에 'steps' key가 없습니다.")
            steps = [{}]

        # sequence dict내에 마지막 step의 결과와 비용 정보만 출력하면 됨.
        response: dict = steps[-1]

        # step 이후 교정된 텍스트
        edited_text = response.get("output_text", "")
        if not edited_text:
            log.warning("교정된 텍스트가 없습니다.")

        # step 이후 비용 정보
        cost_info = response.get("cost_info", {})
        if not cost_info:
            log.warning(
                f"비용 정보가 없어, 기본 값인 used_cost={self.base_cost}, total_tokens=None으로 설정합니다."
            )
            cost_info = {"used_cost": self.base_cost, "total_tokens": None}

        return edited_text, cost_info


# TODO: 추후 default editor로 하나만 사용하도록 교체할 것
class HybridDocumentEditor(DocumentEditor):
    """
    캐시된 오프라인 데이터를 우선 사용하고, 없으면 LLM을 호출하는 하이브리드 에디터

    doc_id와 actions를 받아서:
    1. 먼저 캐시에서 해당 doc_id와 actions에 해당하는 편집 결과를 찾음
    2. 캐시에 없으면 LLM을 호출하여 실시간으로 편집
    """

    def __init__(
        self,
        dataloader: CacheDocumentLoader = CacheDocumentLoader(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader

    @override
    def edit(
        self, doc_id: int, doc: str, actions: List[str] | Tuple[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        doc_id, doc, actions를 받아서 편집 수행

        Args:
            doc_id: 문서 ID (캐시 검색용)
            doc: 편집할 문서 텍스트 (LLM 호출용)
            actions: 편집 액션 리스트 또는 튜플

        Returns:
            (편집된 텍스트, 비용 정보)
        """
        # actions의 마지막 항목 추출
        actions_list = list(actions) if isinstance(actions, tuple) else actions
        if not actions_list:
            raise ValueError("actions가 비어있습니다.")

        last_action = actions_list[-1]

        # 캐시에서 doc_sequences 로드 시도
        try:
            doc_sequences = self.dataloader.load_by_doc_id(doc_id)
        except Exception:
            doc_sequences = {}

        try:
            sequence: Dict = doc_sequences.get(tuple(actions))
            if sequence is not None:
                edited_text = sequence.get("output_text", "")
                if not edited_text:
                    log.warning("교정된 텍스트가 없습니다. LLM 호출로 대체합니다.")
                    raise ValueError("캐시에 output_text가 없음")

                cost_info = sequence.get("cost_info", {})
                if not cost_info:
                    log.warning(
                        f"비용 정보가 없어, 기본 값인 used_cost={self.base_cost}, total_tokens=None으로 설정합니다."
                    )
                    cost_info = {"used_cost": self.base_cost, "total_tokens": None}

                return edited_text, cost_info
        except Exception as e:
            log.info(f"캐시 로더에서 알 수 없는 오류 발생: {e}")

        # 캐시에 없으면 LLM 호출하여 실시간 편집

        # 부모 클래스의 edit 메서드 호출 (doc과 마지막 action 사용)
        edited_text, cost_info = super().edit(doc, last_action)

        # LLM 호출 결과를 캐시에 저장
        try:
            # actions를 키로 하여 편집 결과 추가
            actions_key = tuple(actions)
            doc_sequences[actions_key] = {
                "output_text": edited_text,
                "cost_info": cost_info,
            }

            # 캐시에 저장
            self.dataloader.save(doc_sequences, doc_id)

        except Exception as e:
            log.warning(f"캐시 저장 중 오류 발생: {e}. 편집 결과는 반환됩니다.")

        return edited_text, cost_info
