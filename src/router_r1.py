import random, math
from collections import namedtuple

# ---------- 1) Candidate LLM ----------
class CandidateLLM:
    def __init__(self, name, price_per_tok, skill_tag):
        self.name = name
        self.price = price_per_tok
        self.skill = skill_tag  # e.g., "date", "geo", "multi-hop"

    def answer(self, question, ctx):
        # toy: 성공확률은 (스킬이 맞을수록↑, 라운드가 진행될수록↑)
        t = ctx.get("t", 0)
        need = ctx.get("need", "geo")
        base_p = 0.25
        if need == self.skill:
            base_p += 0.35
        base_p += min(0.2, 0.05 * t)
        ok = random.random() < base_p
        out_tokens = random.randint(30, 120)
        if ok:
            info = f"Found a likely answer for {need}."
        else:
            info = "Not sure; partial hints only."
        cost = self.price * out_tokens
        return info, out_tokens, cost, ok

# ---------- 2) Env ----------
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
        done = False; reward = 0.0
        if action[0] == "THINK":
            self.ctx["log"].append(("<think>", "considering..."))
        elif action[0] == "ROUTE":
            i = action[1]
            info, out_toks, cost, ok = self.llms[i].answer(self.q, self.ctx)
            self.ctx["log"].append(("<search>", f"{self.llms[i].name}: subq"))
            self.ctx["log"].append(("<info>", info))
            self.calls.append((i, out_toks, cost, ok))
        elif action[0] == "STOP":
            guess = action[1]
            self.ctx["log"].append(("<answer>", guess))
            done = True
            reward = self._final_reward(guess)
        self.t += 1; self.ctx["t"] = self.t
        if self.t >= self.T_max and not done:
            # force stop with empty guess
            self.ctx["log"].append(("<answer>", ""))
            done = True
            reward = self._final_reward("")
        return self._state(), reward, done

    # ---- helpers ----
    def _state(self):
        tot_cost = sum(c for _,_,c,_ in self.calls)
        tot_tokens = sum(t for _,t,_,_ in self.calls)
        return {
            "t": self.t, "T_max": self.T_max,
            "tot_cost": tot_cost, "tot_tokens": tot_tokens,
            "n_calls": len(self.calls),
        }

    def _format_reward(self):
        tags = [tag for tag,_ in self.ctx["log"]]
        has_think = any(t=="<think>" for t,_ in self.ctx["log"])
        answers = [t for t,_ in self.ctx["log"] if t=="<answer>"]
        # naive pairing check for search->info
        s_cnt = sum(1 for t,_ in self.ctx["log"] if t=="<search>")
        i_cnt = sum(1 for t,_ in self.ctx["log"] if t=="<info>")
        ok = has_think and len(answers)==1 and s_cnt==i_cnt
        return 0.0 if ok else -1.0

    def _outcome_reward(self, guess):
        return 1.0 if guess.strip().lower()==self.gt.strip().lower() else 0.0

    def _cost_reward(self):
        # invert & normalize cost into [0,1] on a rough scale
        raw = sum(c for _,_,c,_ in self.calls)
        # assume 0~200 arbitrary
        raw = max(0.0, min(raw, 200.0))
        return 1.0 - (raw / 200.0)

    def _final_reward(self, guess):
        Rf = self._format_reward()
        if Rf < 0:
            return Rf  # outcome/cost 무효화 (계층형 보상)
        Ro = self._outcome_reward(guess)
        Rc = self._cost_reward()
        return Rf + (1 - self.alpha)*Ro + self.alpha*Rc

# ---------- 3) Toy Policy (랜덤/룰베이스 시작점) ----------
class RandomPolicy:
    def __init__(self, n_models):
        self.n_models = n_models
    def act(self, s):
        # 아주 단순한 정책: t==0이면 THINK, 그 다음 ROUTE 하나, 마지막에 STOP
        if s["t"]==0:
            return ("THINK",)
        if s["t"]<s["T_max"]-1:
            i = random.randrange(self.n_models)
            return ("ROUTE", i)
        else:
            # 데모: 정답을 모른다고 가정하고 "guess"를 빈 문자열로
            return ("STOP", "")

# ---------- 4) Example Run ----------
def build_pool():
    return [
        CandidateLLM("Small-Geo", 0.5, "geo"),
        CandidateLLM("Small-Date", 0.5, "date"),
        CandidateLLM("Large-Strong", 1.5, "multi-hop"),
    ]

def samples():
    # (질문, 정답, 필요스킬)
    return [
        ("프랑스의 수도는?", "파리", "geo"),
        ("2018과 2021 중 늦은 해?", "2021", "date"),
        ("A와 B 사실을 합쳐 결론?", "정답", "multi-hop"),
    ]

if __name__ == "__main__":
    env = RouterEnv(build_pool(), T_max=4, alpha=0.6)
    pi = RandomPolicy(n_models=len(env.llms))
    for ep, sample in enumerate(samples(), 1):
        s = env.reset(sample)
        done = False
        while not done:
            a = pi.act(s)
            s, r, done = env.step(a)
        print(f"[EP{ep}] R={r:.3f} calls={s['n_calls']} cost={s['tot_cost']:.1f}")
