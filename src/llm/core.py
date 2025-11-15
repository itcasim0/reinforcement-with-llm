import os

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

from utils.logger_factory import log

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
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])

        # 모델별 입력/출력 단가를 각각 곱해 비용 산출
        # 1M 토큰 당 비용은 거의 0의 비용이므로, 1M으로 나누지 않고 계산하도록 함.
        in_cost = (input_tokens) * float(self.price_per_1m["input_price"])
        out_cost = (output_tokens) * float(self.price_per_1m["output_price"])

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

        # OpenAI 호환 Chat API 호출
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        # --- 응답 파싱 ---
        content = response.choices[0].message.content

        # --- usage 파싱 ---
        usage = response.usage
        usage = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        out_tokens, cost = self._calc_cost(usage)

        # TODO: LLM의 응답을 어떻게 수식화할 지 고민 필요.
        # TODO: 아마 ok는 없어지고, 해당 부분은 reward 계산하는 부분에서 결정해야할지도?
        ok = len(content.strip()) > 0

        content = content if len(content) <= 200 else content[:200] + "..."
        return content, out_tokens, cost, ok
