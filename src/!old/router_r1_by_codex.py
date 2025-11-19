from __future__ import annotations

from dataclasses import dataclass
import os
import random
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelDescriptor:
    """Simple descriptor used by the router for generalization.

    Only includes lightweight signals: pricing, latency, and example performance.
    """

    name: str
    price_per_1k_tokens: float
    latency_ms: int
    example_score: float


class ActionType(Enum):
    THINK = auto()
    ROUTE = auto()  # invoke a specific model
    STOP = auto()


@dataclass
class Action:
    """Action selected by the policy.

    - THINK: internal deliberation, no external call
    - ROUTE: call a candidate model (by name)
    - STOP: finish and produce a final answer
    """

    kind: ActionType
    target_model: Optional[str] = None


@dataclass
class StepResult:
    state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class RewardFunction:
    """Rule-based reward with format, final, and cost components."""

    def format_reward(self, trace: List[Dict[str, Any]]) -> float:
        raise NotImplementedError

    def final_outcome_reward(self, final_answer: str, reference: Optional[str]) -> float:
        raise NotImplementedError

    def cost_reward(self, calls: List[Dict[str, Any]]) -> float:
        raise NotImplementedError

    def total(self, state: Dict[str, Any]) -> float:
        raise NotImplementedError


class Aggregator:
    """Aggregates intermediate model responses into the evolving context."""

    def update_context(self, context: str, response: str, model: ModelDescriptor) -> str:
        raise NotImplementedError

    def finalize(self, context: str) -> str:
        raise NotImplementedError


class RouterPolicy:
    """Policy that interleaves THINK and ROUTE actions across multiple rounds."""

    def __init__(self, models: List[ModelDescriptor]):
        self.models = {m.name: m for m in models}

    def act(self, state: Dict[str, Any]) -> Action:
        raise NotImplementedError

    def update(self, trajectories: List[Dict[str, Any]]) -> None:
        raise NotImplementedError


class MultiLLMRoutingEnv:
    """Sequential decision process for multi-round routing and aggregation.

    State contains:
      - context: str (prompt + internal trace)
      - trace: List[events] (think/route steps)
      - calls: List[call metadata]
      - reference: Optional[str] for supervised evaluation (if available)
    """

    def __init__(
        self,
        models: List[ModelDescriptor],
        aggregator: Aggregator,
        reward_fn: RewardFunction,
        max_rounds: int = 4,
        llm_caller: Optional["LLMCaller"] = None,
    ) -> None:
        self.models = {m.name: m for m in models}
        self.aggregator = aggregator
        self.reward_fn = reward_fn
        self.max_rounds = max_rounds
        self.llm_caller = llm_caller
        self._state: Dict[str, Any] = {}

    def reset(self, prompt: str, reference: Optional[str] = None) -> Dict[str, Any]:
        self._state = {
            "context": prompt,
            "trace": [],
            "calls": [],
            "round": 0,
            "reference": reference,
            "final_answer": None,
        }
        return self._state

    def step(self, action: Action) -> StepResult:
        if self._state.get("final_answer") is not None:
            return StepResult(state=self._state, reward=0.0, done=True, info={})

        self._state["round"] += 1
        done = False

        if action.kind == ActionType.THINK:
            thought = self._internal_think(self._state["context"])  # placeholder
            self._state["trace"].append({"type": "think", "content": thought})
            self._state["context"] += f"\n[THINK] {thought}"

        elif action.kind == ActionType.ROUTE:
            if not action.target_model or action.target_model not in self.models:
                raise ValueError("Unknown or missing target_model in ROUTE action")
            model = self.models[action.target_model]
            response, cost = self._call_model(model, self._state["context"])  # placeholder
            self._state["trace"].append(
                {"type": "route", "model": model.name, "response": response}
            )
            self._state["calls"].append(
                {"model": model.name, "cost": cost, "latency_ms": model.latency_ms}
            )
            self._state["context"] = self.aggregator.update_context(
                self._state["context"], response, model
            )

        elif action.kind == ActionType.STOP:
            final_answer = self.aggregator.finalize(self._state["context"])  # placeholder
            self._state["final_answer"] = final_answer
            done = True

        if self._state["round"] >= self.max_rounds and not done:
            self._state["final_answer"] = self.aggregator.finalize(self._state["context"])  # type: ignore
            done = True

        reward = self.reward_fn.total(self._state)
        return StepResult(state=self._state, reward=reward, done=done, info={})

    def _internal_think(self, context: str) -> str:
        raise NotImplementedError

    def _call_model(self, model: ModelDescriptor, context: str) -> Tuple[str, float]:
        """Invoke a candidate LLM and return (response, cost)."""
        raise NotImplementedError


class Trainer:
    """Minimal training loop placeholder for RL fine-tuning of the router policy."""

    def __init__(self, env: MultiLLMRoutingEnv, policy: RouterPolicy):
        self.env = env
        self.policy = policy

    def rollout(self, prompt: str, reference: Optional[str] = None) -> Dict[str, Any]:
        state = self.env.reset(prompt, reference)
        trajectory: List[Dict[str, Any]] = []

        done = False
        while not done:
            action = self.policy.act(state)
            result = self.env.step(action)
            trajectory.append(
                {"state": state, "action": action, "reward": result.reward}
            )
            state = result.state
            done = result.done

        return {"trajectory": trajectory, "final_state": state}

    def train(self, prompts: List[str], references: Optional[List[str]] = None, epochs: int = 1) -> None:
        for _ in range(epochs):
            trajectories: List[Dict[str, Any]] = []
            for i, prompt in enumerate(prompts):
                ref = references[i] if references else None
                traj = self.rollout(prompt, ref)
                trajectories.append(traj)
            self.policy.update(trajectories)


def example_usage() -> None:
    """Sketch of how to wire components together.

    Replace NotImplemented parts with your logic or API calls.
    """

    class SimpleAggregator(Aggregator):
        def update_context(self, context: str, response: str, model: ModelDescriptor) -> str:
            return context + f"\n[{model.name}] {response}"

        def finalize(self, context: str) -> str:
            return context.splitlines()[-1] if context else ""

    class SimpleReward(RewardFunction):
        def format_reward(self, trace: List[Dict[str, Any]]) -> float:
            has_think = any(t.get("type") == "think" for t in trace)
            has_route = any(t.get("type") == "route" for t in trace)
            return 0.1 * float(has_think) + 0.1 * float(has_route)

        def final_outcome_reward(self, final_answer: str, reference: Optional[str]) -> float:
            if not final_answer or not reference:
                return 0.0

            a = final_answer.strip().lower()
            b = reference.strip().lower()
            if a == b or b in a:
                return 1.0
            sim = SequenceMatcher(None, a, b).ratio()
            if sim >= 0.9:
                return 1.0
            if sim >= 0.6:
                return 0.5
            return 0.0

        def cost_reward(self, calls: List[Dict[str, Any]]) -> float:
            total_cost = sum(float(c.get("cost", 0.0)) for c in calls)
            # Penalize total cost linearly (encourages cheaper routing)
            return -1.0 * total_cost

        def total(self, state: Dict[str, Any]) -> float:
            trace = state.get("trace", [])
            calls = state.get("calls", [])
            final_answer = state.get("final_answer") or ""
            reference = state.get("reference")
            round_idx = int(state.get("round", 0))

            r_format = self.format_reward(trace)
            r_cost = self.cost_reward(calls)
            r_final = self.final_outcome_reward(final_answer, reference)
            r_step_penalty = -0.01 * round_idx
            return r_format + r_cost + r_final + r_step_penalty

    class HeuristicPolicy(RouterPolicy):
        def act(self, state: Dict[str, Any]) -> Action:
            if state.get("round", 0) == 0:
                return Action(ActionType.THINK)
            if state.get("round", 0) == 1:
                # choose the cheapest model as a placeholder
                cheapest = min(self.models.values(), key=lambda m: m.price_per_1k_tokens)
                return Action(ActionType.ROUTE, target_model=cheapest.name)
            return Action(ActionType.STOP)

        def update(self, trajectories: List[Dict[str, Any]]) -> None:
            return None

    class EpsilonGreedyPolicy(RouterPolicy):
        """Simple bandit-style policy:
        - First step THINK
        - If no ROUTE yet, choose model by epsilon-greedy on value estimates
        - Then STOP
        Value is updated per-trajectory using final reward as credit for each routed model.
        """

        def __init__(self, models: List[ModelDescriptor], epsilon: float = 0.2):
            super().__init__(models)
            self.epsilon = epsilon
            self.values: Dict[str, float] = {m.name: 0.0 for m in models}
            self.counts: Dict[str, int] = {m.name: 0 for m in models}

        def act(self, state: Dict[str, Any]) -> Action:
            round_idx = int(state.get("round", 0))
            trace: List[Dict[str, Any]] = state.get("trace", [])
            has_routed = any(ev.get("type") == "route" for ev in trace)

            if round_idx == 0:
                return Action(ActionType.THINK)
            if not has_routed:
                if random.random() < self.epsilon:
                    choice = random.choice(list(self.models.values()))
                else:
                    choice = max(self.models.values(), key=lambda m: self.values.get(m.name, 0.0))
                return Action(ActionType.ROUTE, target_model=choice.name)
            return Action(ActionType.STOP)

        def update(self, trajectories: List[Dict[str, Any]]) -> None:
            for traj in trajectories:
                final_state: Dict[str, Any] = traj.get("final_state", {})
                trace: List[Dict[str, Any]] = final_state.get("trace", [])
                # Use last recorded reward if available; else recompute naive
                if traj.get("trajectory"):
                    final_reward = traj["trajectory"][-1].get("reward", 0.0)
                else:
                    final_reward = 0.0
                routed_models = [ev.get("model") for ev in trace if ev.get("type") == "route"]
                for name in routed_models:
                    if name in self.values:
                        self.counts[name] += 1
                        n = self.counts[name]
                        # incremental average update
                        self.values[name] += (final_reward - self.values[name]) / float(n)

    class MultiRoundPolicy(RouterPolicy):
        """Epsilon-greedy over multiple route rounds (k routes max).

        Flow: THINK (round 0) -> ROUTE up to k models (no repeats) -> STOP.
        Values are updated with final reward credited equally to routed models.
        """

        def __init__(self, models: List[ModelDescriptor], epsilon: float = 0.2, max_routes: int = 2):
            super().__init__(models)
            self.epsilon = epsilon
            self.max_routes = max_routes
            self.values: Dict[str, float] = {m.name: 0.0 for m in models}
            self.counts: Dict[str, int] = {m.name: 0 for m in models}

        def act(self, state: Dict[str, Any]) -> Action:
            round_idx = int(state.get("round", 0))
            trace: List[Dict[str, Any]] = state.get("trace", [])
            routed = [ev.get("model") for ev in trace if ev.get("type") == "route"]

            if round_idx == 0:
                return Action(ActionType.THINK)

            if len(routed) < self.max_routes:
                # pick from models not yet routed this episode
                candidates = [m for m in self.models.values() if m.name not in routed]
                if not candidates:
                    return Action(ActionType.STOP)
                if random.random() < self.epsilon:
                    choice = random.choice(candidates)
                else:
                    choice = max(candidates, key=lambda m: self.values.get(m.name, 0.0))
                return Action(ActionType.ROUTE, target_model=choice.name)

            return Action(ActionType.STOP)

        def update(self, trajectories: List[Dict[str, Any]]) -> None:
            for traj in trajectories:
                final_state: Dict[str, Any] = traj.get("final_state", {})
                trace: List[Dict[str, Any]] = final_state.get("trace", [])
                if traj.get("trajectory"):
                    final_reward = traj["trajectory"][-1].get("reward", 0.0)
                else:
                    final_reward = 0.0
                routed_models = [ev.get("model") for ev in trace if ev.get("type") == "route"]
                if not routed_models:
                    continue
                credit = final_reward  # equal credit per model (incremental average still applied)
                for name in routed_models:
                    if name in self.values:
                        self.counts[name] += 1
                        n = self.counts[name]
                        self.values[name] += (credit - self.values[name]) / float(n)

    class LLMCaller:
        """Abstract LLM caller returning (text, tokens_used)."""

        def generate(self, model: ModelDescriptor, prompt: str) -> Tuple[str, int]:
            raise NotImplementedError

    class OpenAIChatCaller(LLMCaller):
        def __init__(self, api_key: str, model_map: Optional[Dict[str, str]] = None):
            self.api_key = api_key
            self.model_map = model_map or {}

        def generate(self, model: ModelDescriptor, prompt: str) -> Tuple[str, int]:
            try:
                # Prefer the new SDK if available
                from openai import OpenAI  # type: ignore

                client = OpenAI(api_key=self.api_key)
                model_id = self.model_map.get(model.name, self.model_map.get("default", "gpt-4o-mini"))
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                tokens = 0
                if usage is not None:
                    tokens = int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "completion_tokens", 0))
                return text, tokens
            except Exception as e:
                return f"[openai error: {e}]", 0

    class DummyEnv(MultiLLMRoutingEnv):
        def _internal_think(self, context: str) -> str:
            return "considering which model to call"

        def _call_model(self, model: ModelDescriptor, context: str) -> Tuple[str, float]:
            # Prefer calling a configured LLM caller if present
            if self.llm_caller is not None:
                text, tokens = self.llm_caller.generate(model, context)
                # Cost scaled by tokens if available
                cost = model.price_per_1k_tokens * (tokens / 1000.0 if tokens else 0.5)
                return text or "", cost
            # Offline fallback with a tiny hard-coded rule
            lower = context.lower()
            if "great gatsby" in lower:
                response = f"F. Scott Fitzgerald (via {model.name})"
            elif "capital of france" in lower or "france capital" in lower:
                response = f"Paris (via {model.name})"
            elif "mona lisa" in lower:
                response = f"Leonardo da Vinci (via {model.name})"
            else:
                response = f"response from {model.name}"
            cost = model.price_per_1k_tokens * 0.5
            return response, cost

    models = [
        ModelDescriptor(name="fast-small", price_per_1k_tokens=0.1, latency_ms=200, example_score=0.6),
        ModelDescriptor(name="accurate-large", price_per_1k_tokens=1.2, latency_ms=800, example_score=0.9),
    ]

    # Optional: wire OpenAI if API key exists
    llm_caller = None
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # map our descriptors to real model IDs; override via env if desired
        model_map = {
            "fast-small": os.getenv("OPENAI_MODEL_SMALL", "gpt-4o-mini"),
            "accurate-large": os.getenv("OPENAI_MODEL_LARGE", "gpt-4o"),
            "default": os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o-mini"),
        }
        llm_caller = OpenAIChatCaller(api_key=api_key, model_map=model_map)

    env = DummyEnv(models=models, aggregator=SimpleAggregator(), reward_fn=SimpleReward(), llm_caller=llm_caller)
    # Switch to multi-round epsilon-greedy policy that can learn model values
    policy = MultiRoundPolicy(models, epsilon=0.2, max_routes=2)
    trainer = Trainer(env, policy)

    # Small demo dataset
    prompts = [
        "Who wrote The Great Gatsby?",
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
    ]
    refs = [
        "F. Scott Fitzgerald",
        "Paris",
        "Leonardo da Vinci",
    ]

    # Train for a few epochs
    epochs = 5
    trainer.train(prompts, references=refs, epochs=epochs)

    # Report learned values
    print("=== Router-R1 Demo (Multi-Round) ===")
    print("Learned model values (higher is better):")
    if hasattr(policy, "values"):
        for name, val in policy.values.items():
            print(f"  {name}: {val:.4f}")

    # Rollout each prompt and show details
    for prompt, reference in zip(prompts, refs):
        out = trainer.rollout(prompt, reference)
        final_state = out["final_state"]
        final_answer = final_state.get("final_answer")
        total_cost = sum(c.get("cost", 0.0) for c in final_state.get("calls", []))
        total_reward = SimpleReward().total(final_state)

        print("\n---")
        print(f"Prompt: {prompt}")
        print(f"Final Answer: {final_answer}")
        print(f"Total Cost (arb): {total_cost:.4f}")
        print(f"Total Reward: {total_reward:.4f}")
        print("Trace:")
        for i, ev in enumerate(final_state.get("trace", []), 1):
            if ev.get("type") == "think":
                print(f"  {i}. THINK -> {ev.get('content')}")
            elif ev.get("type") == "route":
                print(f"  {i}. ROUTE[{ev.get('model')}] -> {ev.get('response')}")


if __name__ == "__main__":
    example_usage()
