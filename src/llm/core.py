import os

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class CandidateLLM:
    """
    후보 LLM으로, 질문에 대해 응답을 호출하고, 이를 계산하여 강화학습에 활용.

    model: 모델명
    desc: 해당 모델에 대한 간단한 설명
    price_per_1m: {"input": x, "output": y}
    """

    def __init__(self, model, description, price_per_1m: dict):
        self.model = model
        self.description = description
        self.price_per_1m = price_per_1m

    def _calc_cost(self, usage: dict):
        """
        usage: API가 반환하는 토큰 사용량.
        """
        input_tokens = int(
            getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0)
        )
        output_tokens = int(
            getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0)
        )

        # 모델별 입력/출력 단가를 각각 곱해 비용 산출
        in_cost = (input_tokens / 1000000.0) * float(
            self.price_per_1m.get("input_price")
        )
        out_cost = (output_tokens / 1000000.0) * float(
            self.price_per_1m.get("output_price")
        )

        # TODO: return 값에 대한 고민이 필요.
        # TODO: 우선, 출력 토큰 수랑, 비용만 고려
        return output_tokens, (in_cost + out_cost)

    def answer(self, question):
        """
        질문에 따라서, LLM이 응답 후 결과를 계산하여 강화학습에 활용.

        - info: 모델의 주요 답변 텍스트(간략화)
        - out_toks: 출력 토큰 수
        - cost: 이번 호출 비용
        - ok: 간단한 성공 휴리스틱 (빈 응답이 아니면 True)
        """

        system_prompt = "당신은 사용자 질문에 명확하고 분명하게 응답하세요."

        try:
            # OpenAI 호환 Chat API 호출
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )

            # --- 응답 파싱 ---
            info = ""
            choice = response.choices[0]

            # 표준 chat.completions 구조
            if hasattr(choice, "message") and choice.message:
                if isinstance(choice.message, dict):
                    info = choice.message.get("content", "")
                else:
                    info = getattr(choice.message, "content", "") or ""

            # 일부 모델은 text 필드로 올 수도 있음
            if not info and hasattr(choice, "text"):
                info = getattr(choice, "text", "")

            # 최후 fallback
            if not info:
                info = str(choice)

        except Exception as e:
            info = f"[Error: {e}]"
            response = None

        # --- usage 파싱 ---
        usage = getattr(response, "usage", {}) if response else {}

        """
        * usage 예시
        CompletionUsage(
            completion_tokens=11,
            prompt_tokens=26,
            total_tokens=37,
            completion_tokens_details=CompletionTokensDetails(
                accepted_prediction_tokens=None,
                audio_tokens=None,
                reasoning_tokens=0,
                rejected_prediction_tokens=None,
                image_tokens=0,
            ),
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None,
                cached_tokens=0
            )
        )
        """

        # 객체/딕셔너리 양쪽 지원
        if not isinstance(usage, dict):
            usage = {
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        out_tokens, cost = self._calc_cost(usage)

        # TODO: LLM의 응답을 어떻게 수식화할 지 고민 필요.
        # TODO: 아마 ok는 없어지고, 해당 부분은 reward 계산하는 부분에서 결정해야할지도?
        ok = len(info.strip()) > 0

        info_msg = info if len(info) <= 200 else info[:200] + "..."
        return info_msg, out_tokens, cost, ok
