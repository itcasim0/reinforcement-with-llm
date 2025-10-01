import random

from llm.core import CandidateLLM
from utils.logger_factory import log

# ========== Environment ==========
class RouterEnv:
    def __init__(self, llms, T_max=4, alpha=0.6):
        self.llms = llms
        self.T_max = T_max
        self.alpha = alpha

    def reset(self, sample):
        self.q, self.gt, self.need = sample
        self.t = 0
        self.ctx = {"log": [], "t": 0, "need": self.need}
        self.calls = []
        self.answered = None
        return self._state()

    def step(self, action):
        # action: ("THINK") or ("STOP", guess) or ("ROUTE", i)
        done = False
        reward = 0.0
        if action[0] == "THINK":
            self.ctx["log"].append(("<think>", "considering..."))
        elif action[0] == "ROUTE":
            i = action[1]
            info, out_toks, cost, ok = self.llms[i].answer(self.q)
            self.ctx["log"].append(("<search>", f"{self.llms[i].model}: subq"))
            self.ctx["log"].append(("<info>", info))
            self.calls.append((i, out_toks, cost, ok))
        elif action[0] == "STOP":
            guess = action[1]
            self.ctx["log"].append(("<answer>", guess))
            done = True
            reward = self._final_reward(guess)
        self.t += 1
        self.ctx["t"] = self.t
        if self.t >= self.T_max and not done:
            # force stop with empty guess
            self.ctx["log"].append(("<answer>", ""))
            done = True
            reward = self._final_reward("")
        return self._state(), reward, done

    # ---- helpers ----
    def _state(self):
        tot_cost = sum(c for _, _, c, _ in self.calls)
        tot_tokens = sum(t for _, t, _, _ in self.calls)
        return {
            "t": self.t,
            "T_max": self.T_max,
            "tot_cost": tot_cost,
            "tot_tokens": tot_tokens,
            "n_calls": len(self.calls),
        }

    def _format_reward(self):
        tags = [tag for tag, _ in self.ctx["log"]]
        has_think = any(t == "<think>" for t, _ in self.ctx["log"])
        answers = [t for t, _ in self.ctx["log"] if t == "<answer>"]
        # naive pairing check for search->info
        s_cnt = sum(1 for t, _ in self.ctx["log"] if t == "<search>")
        i_cnt = sum(1 for t, _ in self.ctx["log"] if t == "<info>")
        ok = has_think and len(answers) == 1 and s_cnt == i_cnt
        return 0.0 if ok else -1.0

    def _outcome_reward(self, guess):
        # guess가 정답을 포함하면 맞춘 걸로 인정
        return 1.0 if self.gt.strip().lower() in guess.strip().lower() else 0.0

    def _cost_reward(self):
        # invert & normalize cost into [0,1] on a rough scale
        raw = sum(c for _, _, c, _ in self.calls)
        # assume 0~200 arbitrary
        raw = max(0.0, min(raw, 200.0))
        return 1.0 - (raw / 200.0)

    def _final_reward(self, guess):
        Rf = self._format_reward()
        if Rf < 0:
            return Rf  # outcome/cost 무효화 (계층형 보상)
        Ro = self._outcome_reward(guess)
        Rc = self._cost_reward()
        return Rf + (1 - self.alpha) * Ro + self.alpha * Rc


# ========== policy =========
# TODO: 이 부분이 배운 강화학습 알고리즘을 응용할 곳.
# TODO: Random, 벨만, 마코프, dynamic ,bandit, Q-Learning, etc...
class RandomPolicy:
    def __init__(self, n_models):
        self.n_models = n_models

    def act(self, s, ctx=None):
        # 아주 단순한 정책: t==0이면 THINK, 그 다음 ROUTE 하나, 마지막에 STOP
        if s["t"] == 0:
            return ("THINK",)
        if s["t"] < s["T_max"] - 1:
            i = random.randrange(self.n_models)
            return ("ROUTE", i)
        else:
            # 마지막 <info>를 guess로 제출
            guess = ""
            if ctx:
                infos = [msg for tag, msg in ctx["log"] if tag == "<info>"]
                if infos:
                    guess = infos[-1]
            return ("STOP", guess)


# ---------- 4) Example Run ----------
def build_pool():
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

def samples():
    # (질문, 정답, 필요스킬)
    return [
        ("프랑스의 수도는?", "파리", "geo"),
        ("2018과 2021 중 늦은 해?", "2021", "date"),
        ("에펠탑은 어느 도시에 있는가, 그리고 그 도시는 어느 나라의 수도인가?", "프랑스", "multi-hop"),
    ]

def router_r1():
    env = RouterEnv(build_pool(), T_max=4, alpha=0.6)
    pi = RandomPolicy(n_models=len(env.llms))
    for ep, sample in enumerate(samples(), 1):
        s = env.reset(sample)
        done = False
        while not done:
            a = pi.act(s, env.ctx)
            s, r, done = env.step(a)
        log.info(f"[EP{ep}] R={r:.3f} calls={s['n_calls']} cost={s['tot_cost']:.1f}")

        # 출력
        for tag, msg in env.ctx["log"]:
            print(f"{tag}: {msg.strip()}")
        print("")

if __name__ == "__main__":
    router_r1()
