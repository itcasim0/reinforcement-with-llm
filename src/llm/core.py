import os
import random

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_APK_KEY"),
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

        """TODO: Responses API return을 어떤식으로 하는 지 확인 필요.
        Responses API 기준으로 아래와 유사한 필드를 가정:
          usage = {
            "input_tokens": ...,
            "output_tokens": ...,
            "total_tokens": ...
          }"""

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
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        # TODO: 아래에 부분에서 default값 설정 부분 다 제거하고, 확실하게 수정할 것
        # TODO: 쓸대없는 LLM API 호출은 비용만 발생할 뿐.
        try:
            # 가능한 필드 케이스에 폭넓게 대응.
            first_item = (
                response.output[0]
                if hasattr(response, "output")
                else response.choices[0]
            )
            if hasattr(first_item, "content") and first_item.content:
                info_text = getattr(first_item.content[0], "text", None)
                info = getattr(info_text, "value", "") if info_text else ""
            elif hasattr(first_item, "message"):
                info = first_item.message.get("content", "")
            else:
                # 최후 fallback
                info = str(first_item)
        except Exception:
            info = ""

        usage = getattr(response, "usage", {})

        # 객체/딕셔너리 양쪽 지원
        if not isinstance(usage, dict):
            usage = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        out_tokens, cost = self._calc_cost(usage)

        # TODO: LLM의 응답을 어떻게 수식화할 지 고민 필요.
        ok = len(info.strip()) > 0

        info_msg = info if len(info) <= 200 else info[:200] + "..."
        return info_msg, out_tokens, cost, ok
