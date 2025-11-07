from typing import List, Dict
import random
from dataclasses import dataclass

from llm.core import CandidateLLM

from utils.logger_factory import log


@dataclass
class Data:
    """
    사용자 질의에 대한 학습 데이터를 담고 있는 데이터 클래스

    Attributes:
        question (str): 입력 질문 (예: "프랑스의 수도는?")
        gt (str): 정답 (ground truth, 예: "파리")
        category (str): 문제 유형/카테고리 (예: "geo", "math", "code")
    """

    question: str
    gt: str
    category: str


@dataclass
class Context:
    """
    Environment내에서 변하는 상태에 대해 담는 데이터 클래스

    Attributes:
        logs (list): step에 따른 history성 log를 담는 리스트
        current_step (int): 현재 몇 번째 step인지
        category (str): 문제 유형/카테고리 (예: "geo", "math", "code")
    """

    logs: List[Dict[str, str]]
    current_step: int
    category: str


# ========== Environment ==========
class RouterEnv:
    """
    Router-r1의 논문을 시작으로, 구성된 environment임.

    본 논문의 특징 중에 하나가 여러 번 LLM을 호출하여 최적의 LLM을 선택하는 과정이 있음.
        단, 이 과정이 멀티 턴(대화 형식)은 아님. TODO: 이 과정을 멀티 턴 형식으로 풀어본다면 어떨지?

    Args:
        llms(list): 후보 LLM들
        max_step(int): 여러 번 호출 시 최대 몇 번까지 할지.
        alpha(float): LLM 비용에 대한 알파 값, (1-알파)는 정답에 대한 reward
    """

    def __init__(self, llms: List[CandidateLLM], max_step=4, alpha=0.6):
        self.llms = llms
        self.max_step = max_step
        self.alpha = alpha

    def reset(self, data: Data):
        self.question, self.gt, self.category = (data.question, data.gt, data.category)
        self.current_step = 0
        self.context = Context([], 0, self.category)
        self.calls = []
        self.answered = None
        return self._state()

    def step(self, action):

        # action: ("THINK") or ("STOP", response) or ("ROUTE", i)
        done = False
        reward = 0.0

        # action에 따른 상태 update.
        if action[0] == "THINK":
            self.context.logs.append(("<think>", "considering..."))

        elif action[0] == "ROUTE":
            # LLM 선택을 위한 index.
            i = action[1]
            llm: CandidateLLM = self.llms[i]

            # LLM 호출 후 reward 계산에 사용될 요소들.
            response, out_tokens, cost, ok = llm.answer(self.question)

            # Update context logs.
            self.context.logs.append(("<model>", f"{llm.model}"))
            self.context.logs.append(("<response>", response))

            # LLM 호출에 대한 결과 저장.
            self.calls.append((i, out_tokens, cost, ok))

        elif action[0] == "STOP":
            response = action[1]
            self.context.logs.append(("<answer>", response))
            done = True
            reward = self._final_reward(response)

        # step 추가.
        self.current_step += 1

        # environment context에서 현재 step update.
        self.context.current_step = self.current_step

        # 현재 step이 설정한 최대 step보다 같거나 크고, 아직 완료되지 않았을 경우.
        if self.current_step >= self.max_step and not done:
            # force stop with empty response
            # TODO: 응답을 비어있게 만드는 게 맞을 지,
            # TODO: 마지막 step의 response를 가져와서 활용하게끔 하는 게 맞을 지?
            
            # 빈 답변 강제 종료??
            # responses = [msg for tag, msg in self.context.logs if tag == "<response>"]
            # if responses:
            #     final_response = responses[-1]
            #     self.context.logs.append(("<answer>", final_response))
            #     reward = self._final_reward(final_response) - 0.1  # 약간의 패널티
            # else:
            #     self.context.logs.append(("<answer>", ""))
            #     reward = self._final_reward("") - 0.1  # 약간의 패널티

            # 마지막 응답을 사용하게끔 하는 게 더 나을 듯? # reward는 정답이면 1 아니면 0 오답이면 -1??
            # 
            
            
            self.context.logs.append(("<answer>", ""))
            done = True
            reward = self._final_reward("")
        return self._state(), reward, done

    def _state(self):
        # 비용의 누적
        total_cost = sum(cost for _, _, cost, _ in self.calls)

        # 토큰의 누적
        total_tokens = sum(tokens for _, tokens, _, _ in self.calls)
        return {
            "current_step": self.current_step,
            "max_step": self.max_step,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "n_calls": len(self.calls),  # LLM 호출을 몇 번 했는 지
        }

    def _format_reward(self):
        """
        reward 계산 시 제일 처음으로 수행하는 부분.
        하나의 episode가 끝난 후, state의 포맷이 설계한 형태로 나왔는 지 확인.
        포맷이 설계한 형태이면 0, 아니면 -1의 reward를 부여.

        """
        tags = [tag for tag, _ in self.context.logs]

        # <think>가 하나라도 있으면, True
        has_think = any(tag == "<think>" for tag in tags)

        # <answer>만 추출
        answers = [tag for tag in tags if tag == "<answer>"]

        # naive pairing check for model->response
        model_cnt = sum(1 for tag in tags if tag == "<model>")
        response_cnt = sum(1 for tag in tags if tag == "<response>")

        # TODO: <think>가 2개가 있을 수가 있나..? 1개인지만 확인하면 되는 거 아님?
        ok = has_think and len(answers) == 1 and model_cnt == response_cnt
        return 0.0 if ok else -1.0

    def _outcome_reward(self, response: str):
        # TODO: 정답에 대한 보상 체계를 좀 더 현실적인 부분으로 고민 필요.
        # TODO: 정답이 맞고 틀림도 LLM한테 시키면 어떨까 !?

        # response가 정답을 포함하면 맞춘 걸로 인정
        return 1.0 if self.gt.strip().lower() in response.strip().lower() else 0.0

    def _cost_reward(self):
        # TODO: 비용에 대한 최대 계산은 실제로 몇 번 돌려보고 재정의 필요할 듯.
        # invert & normalize cost into [0,1] on a rough scale
        raw = sum(c for _, _, c, _ in self.calls)
        # assume 0~200 arbitrary
        raw = max(0.0, min(raw, 200.0))
        return 1.0 - (raw / 200.0)

    def _final_reward(self, response):

        # 설계한 포맷인지 아닌 지에 따른 보상.
        Rf = self._format_reward()
        if Rf < 0:
            return Rf  # outcome/cost 무효화. (계층형 보상)

        # LLM의 응답한 결과에 따른 보상.
        Ro = self._outcome_reward(response)

        Rc = self._cost_reward()
        return Rf + (1 - self.alpha) * Ro + self.alpha * Rc


# ========== policy =========
# TODO: 이 부분이 배운 강화학습 알고리즘을 응용할 곳.
# TODO: Random, 벨만, 마코프, dynamic ,bandit, Q-Learning, etc...

class RandomPolicy:
    def __init__(self, n_models):
        self.n_models = n_models

    def act(self, state, context: Context):
        """
        정의된 action 수행

        action 종류
        1. THINK: 처음 step이라면, 아무일도 일어나지 않음.
        2. ROUTE: 랜덤으로 LLM을 선택함.
        3. STOP: 1 episode 종료.
        """

        # 아주 단순한 정책: t==0이면 THINK, 그 다음 ROUTE 하나, 마지막에 STOP
        if state["current_step"] == 0:
            return ("THINK",)
        if state["current_step"] < state["max_step"] - 1:
            i = random.randrange(self.n_models)
            return ("ROUTE", i)
        else:
            # 마지막 <response>를 최종으로 제출
            response = ""
            responses = [msg for tag, msg in context.logs if tag == "<response>"]
            if responses:
                response = responses[-1]
            return ("STOP", response)


def build_llms() -> List[CandidateLLM]:
    return [
        CandidateLLM(
            "google/gemini-2.0-flash-001",
            "25년 2월 출시 가성비 값 모델",
            {"input_price": 0.10, "output_price": 0.40},
        ),
        CandidateLLM(
            "google/gemini-2.5-flash-lite",
            "25년 7월 출시인데, 2.0보다 더 적게 쓰이는 모델",
            {"input_price": 0.10, "output_price": 0.40},
        ),
        CandidateLLM(
            "google/gemma-3-12b-it",
            "오픈소스 모델 하나 정도는 껴있어야 재밌지",
            {"input_price": 0.04, "output_price": 0.14},
        ),
    ]


def load_data():
    return [
        Data("프랑스의 수도는?", "파리", "geo"),
        Data("2018과 2021 중 늦은 해?", "2021", "date"),
        Data("1+1=?", "2", "math"),
        Data("파이썬에서 1+1은 몇 인지 코드 작성해줘.", "2", "code"),
    ]


def router_r1():
    env = RouterEnv(build_llms(), max_step=4, alpha=0.6)
    policy = RandomPolicy(n_models=len(env.llms))

    # data 1개 당, 1 episode 임.
    for ep, data in enumerate(load_data(), 1):

        # state 초기화.
        state = env.reset(data)

        # done이 나올 때 까지 실시. (즉, 1 episode 종료 sign)
        done = False
        while not done:
            action = policy.act(state, env.context)
            state, reward, done = env.step(action)

        log.info(
            f"[EP{ep}] R={reward:.3f} calls={state['n_calls']} cost={state['total_cost']:.1f}"
        )

        # 1 episode의 history를 출력.
        for tag, msg in env.context.logs:
            log.info(f"{tag}: {msg.strip()}")
        log.info(f"[EP{ep} 종료]\n\n")


if __name__ == "__main__":
    router_r1()
