"""
단일 문서 오버피팅 실험용 오프라인 RL (수정 버전)

주요 수정사항:
1. 항상 동일한 시퀀스(첫 번째)만 사용 → 완벽한 오버피팅 유도
2. action_history를 step별로 누적하여 중간 결과 추적
3. 평가 점수 범위를 더 넓게 조정 (저품질 문서 = 낮은 점수)
4. 보상 함수 단순화 및 명확화
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
import hashlib
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# 데이터 클래스
# ============================================================
@dataclass
class DocumentScore:
    """문서 점수 (0~10)"""
    grammar: float = 5.0
    readability: float = 5.0
    coherence: float = 5.0
    overall: float = 5.0


# ============================================================
# 깐깐한 평가기 (점수 범위 확대)
# ============================================================
class StrictEvaluator:
    """
    저품질 초록의 특징을 정확히 잡아내는 평가기
    
    수정: 기본 점수를 낮춰서 저품질 문서 = 낮은 점수가 되도록 함
    """
    
    def __init__(self):
        # === 감점 패턴들 ===
        
        # 1. 모호한/불필요한 표현 (심각도 높음)
        self.vague_patterns = [
            "일지도 모르는", "일지도 모를", "있을지도 모르는",
            "아닐까", "않을까", "일 것이다",
            "좀 ", "약간 ", "조금 ",
            "가상의", "어떤 ", "그런 ",
            "같은 것", "라는 것", "라고 하는",
            "등등", "기타 등등",
        ]
        
        # 2. 어색한 어미 (심각도 높음)
        self.awkward_endings = [
            "해보아 했다", "해보아야 했다",
            "인 것이다", "인 것이었다", 
            "라고 한다", "다고 한다",
            "했던 것이다", "였던 것이다",
            "하는 바이다", "되는 바이다",
            "것이라고", "것이었다고",
        ]
        
        # 3. 구어체/비학술적 표현 (심각도 중간)
        self.colloquial = [
            "뭐랄까", "글쎄", "아무튼",
            "그러니까", "어쩌면", "사실",
            "솔직히", "당연히", "물론",
            "엄청", "굉장히", "되게",
            "진짜", "정말로", "완전",
            "이런저런", "요즘", "얼마 전",
        ]
        
        # 4. 불필요한 수식/군더더기 (심각도 중간)
        self.fillers = [
            "매우 ", "아주 ", "상당히 ",
            "다소 ", "꽤 ", "어느 정도",
            "기본적으로", "일반적으로 말해서",
            "말하자면", "이를테면",
            "다양한 ", "여러 가지 ",
        ]
        
        # 5. 반복/중복 패턴
        self.redundant = [
            "즉, 다시 말해", "다시 말해서",
            "요약하자면, 결론적으로",
        ]
        
        # === 가점 패턴들 (학술적 표현) ===
        self.academic_positive = [
            "본 연구", "본 논문",
            "분석하였다", "검증하였다", "확인하였다",
            "제안한다", "제시한다",
            "따라서", "그러나", "한편",
            "결과적으로", "구체적으로",
        ]
        
        # 구조 키워드
        self.structure_keywords = {
            "background": ["배경", "기존", "현재", "문제점", "필요성"],
            "objective": ["목적", "목표", "규명", "분석", "검증"],
            "method": ["방법", "기법", "모델", "제안", "활용", "적용"],
            "result": ["결과", "입증", "확인", "나타났다", "보였다"],
            "conclusion": ["결론", "의의", "기여", "향후", "시사점"],
        }
    
    def evaluate(self, text: str) -> DocumentScore:
        """종합 평가"""
        grammar = self._eval_grammar(text)
        readability = self._eval_readability(text)
        coherence = self._eval_coherence(text)
        overall = (grammar + readability + coherence) / 3.0
        
        return DocumentScore(
            grammar=round(grammar, 2),
            readability=round(readability, 2),
            coherence=round(coherence, 2),
            overall=round(overall, 2),
        )
    
    def _eval_grammar(self, text: str) -> float:
        """
        문법/표현 품질 (0~10)
        - 수정: 기본 점수를 5.0으로 낮춤 (저품질 문서 = 낮은 점수)
        """
        score = 5.0  # 기본 점수 낮춤 (기존 8.0 → 5.0)
        
        # 어색한 어미 감점 (각 -1.0으로 증가)
        for pattern in self.awkward_endings:
            count = text.count(pattern)
            score -= count * 1.0  # 기존 0.8 → 1.0
        
        # 모호한 표현 감점 (각 -0.7로 증가)
        for pattern in self.vague_patterns:
            count = text.count(pattern)
            score -= count * 0.7  # 기존 0.5 → 0.7
        
        # 학술적 표현 가점 (각 +0.3, 최대 +3.0)
        bonus = 0
        for pattern in self.academic_positive:
            if pattern in text:
                bonus += 0.3
        score += min(3.0, bonus)
        
        return max(0, min(10, score))
    
    def _eval_readability(self, text: str) -> float:
        """
        가독성 (0~10)
        - 수정: 기본 점수 낮춤, 감점 강화
        """
        score = 5.0  # 기본 점수 낮춤 (기존 8.0 → 5.0)
        
        # 구어체 감점 (각 -0.8로 증가)
        for pattern in self.colloquial:
            count = text.count(pattern)
            score -= count * 0.8  # 기존 0.6 → 0.8
        
        # 불필요한 수식어 감점 (각 -0.5로 증가)
        for pattern in self.fillers:
            count = text.count(pattern)
            score -= count * 0.5  # 기존 0.3 → 0.5
        
        # 문장 길이 평가
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        if sentences:
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            if avg_length > 100:
                score -= 2.0  # 기존 1.5 → 2.0
            elif avg_length > 80:
                score -= 1.2  # 기존 0.8 → 1.2
            elif avg_length < 20:
                score -= 0.8  # 기존 0.5 → 0.8
        
        # 가점 추가 (짧고 명확한 문장)
        if sentences and 30 <= sum(len(s) for s in sentences) / len(sentences) <= 60:
            score += 1.0
        
        return max(0, min(10, score))
    
    def _eval_coherence(self, text: str) -> float:
        """
        논리적 일관성/구조 (0~10)
        - 수정: 기본 점수를 더 낮춤
        """
        score = 3.0  # 기본 점수 낮춤 (기존 5.0 → 3.0)
        
        # 구조 키워드 가점
        sections_found = 0
        for section, keywords in self.structure_keywords.items():
            for kw in keywords:
                if kw in text:
                    sections_found += 1
                    break
        
        score += sections_found * 1.0  # 기존 0.8 → 1.0
        
        # 연결어 가점
        connectives = ["그러나", "따라서", "한편", "또한", "이에", "결과적으로"]
        conn_count = sum(1 for c in connectives if c in text)
        score += min(1.5, conn_count * 0.4)  # 기존 (1.0, 0.3) → (1.5, 0.4)
        
        # 중복 패턴 감점
        for pattern in self.redundant:
            if pattern in text:
                score -= 0.8  # 기존 0.5 → 0.8
        
        return max(0, min(10, score))
    
    def detailed_report(self, text: str) -> dict:
        """상세 분석 리포트"""
        issues = {
            "vague": [],
            "awkward": [],
            "colloquial": [],
            "fillers": [],
        }
        
        for p in self.vague_patterns:
            if p in text:
                issues["vague"].append(p)
        for p in self.awkward_endings:
            if p in text:
                issues["awkward"].append(p)
        for p in self.colloquial:
            if p in text:
                issues["colloquial"].append(p)
        for p in self.fillers:
            if p in text:
                issues["fillers"].append(p)
        
        return issues


# ============================================================
# 오프라인 환경 (단일 시퀀스 오버피팅용)
# ============================================================
class OfflineEditingEnv:
    """
    수정된 오프라인 환경:
    1. 항상 첫 번째 시퀀스만 사용
    2. action_history를 누적하며 step별 결과 추적
    3. 보상 함수 단순화
    """
    
    def __init__(
        self,
        jsonl_path: str,
        max_steps: int = 6,
        cost_lambda: float = 1.0,
        repeat_penalty: float = 0.3,
        use_single_sequence: bool = True,  # 오버피팅용 플래그
    ):
        self.jsonl_path = jsonl_path
        self.max_steps = max_steps
        self.cost_lambda = cost_lambda
        self.repeat_penalty = repeat_penalty
        self.use_single_sequence = use_single_sequence
        
        # 데이터 로드
        self.all_sequences = []
        self.action_index = {}
        self._load_data()
        
        # 평가기
        self.evaluator = StrictEvaluator()
        
        # 액션 정의
        self.actions = ["fix_grammar", "improve_clarity", "make_concise", 
                       "improve_structure", "stop_editing"]
        self.num_actions = len(self.actions)
        
        # 현재 에피소드 상태
        self.current_text = ""
        self.current_score = DocumentScore()
        self.current_step = 0
        self.action_history = []
        self.used_actions = set()
        self.base_text = ""
        
        # 오버피팅용: 고정 시퀀스 인덱스
        self.fixed_sequence_idx = 0
    
    def _load_data(self):
        """JSONL 데이터 로드 및 인덱싱"""
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.all_sequences.append(record)
                    
                    # action sequence로 인덱싱
                    actions_tuple = tuple(record["actions"])
                    self.action_index[actions_tuple] = record
        
        log.info(f"[데이터 로드] 총 {len(self.all_sequences)}개 시퀀스")
    
    def reset(self) -> Tuple[Tuple[float, float, float, float], str]:
        """
        에피소드 초기화
        - 오버피팅 모드: 항상 첫 번째 시퀀스 사용
        """
        self.current_step = 0
        self.action_history = []
        self.used_actions = set()
        
        # 오버피팅 모드: 항상 동일한 시퀀스
        if self.use_single_sequence:
            seq = self.all_sequences[self.fixed_sequence_idx]
        else:
            # 일반 모드: 랜덤 선택
            seq = self.all_sequences[torch.randint(0, len(self.all_sequences), (1,)).item()]
        
        self.base_text = seq["base_text"]
        self.current_text = self.base_text
        
        # 초기 점수 계산
        self.current_score = self.evaluator.evaluate(self.current_text)
        state = self._scores_to_state(self.current_score)
        
        return state, self.current_text
    
    def _scores_to_state(self, scores: DocumentScore) -> Tuple[float, float, float, float]:
        """점수를 상태로 변환"""
        return (scores.grammar, scores.readability, scores.coherence, scores.overall)
    
    def step(self, action_index: int) -> Tuple[Tuple[float, float, float, float], float, bool, Dict]:
        """
        환경 step 실행
        - action_history를 누적하며 중간 결과 찾기
        """
        assert 0 <= action_index < self.num_actions
        action = self.actions[action_index]
        
        prev_scores = self.current_score
        done = False
        info = {
            "action": action,
            "prev_scores": prev_scores,
        }
        
        # stop_editing: 종료
        if action == "stop_editing":
            # 현재 점수에 따라 stop 보상 결정
            current_quality = (prev_scores.overall - 5.0) / 5.0
            
            # 점수가 충분히 좋으면 stop에 큰 보너스 (불필요한 편집 방지)
            if prev_scores.overall >= 7.0:
                stop_bonus = 2.0  # 정말 잘 만들어진 경우만
            elif prev_scores.overall >= 6.5:
                stop_bonus = 1.0
            elif prev_scores.overall >= 6.0:
                stop_bonus = 0.3  # 보너스 감소
            elif prev_scores.overall >= 5.5:
                stop_bonus = 0.0  # 보너스 없음 (아직 개선 여지 있음)
            else:
                stop_bonus = -1.0  # 패널티
            
            reward = current_quality + stop_bonus
            done = True
            info["reason"] = "stop_action"
            info["stop_bonus"] = stop_bonus
            info["new_scores"] = prev_scores
            next_state = self._scores_to_state(prev_scores)
            return next_state, reward, done, info
        
        # 액션 추가
        self.action_history.append(action)
        self.current_step += 1
        
        # 현재 action_history에 해당하는 결과 찾기
        actions_tuple = tuple(self.action_history)
        
        if actions_tuple in self.action_index:
            # 데이터에 있는 경우: 해당 결과 사용
            result = self.action_index[actions_tuple]
            steps = result.get("steps", [])
            
            if steps:
                last_step = steps[-1]
                self.current_text = last_step.get("output_text", self.current_text)
                cost_info = last_step.get("cost_info", {"usd_cost": 0.02, "total_tokens": 2000})
            else:
                # steps가 없으면 final_text 사용
                self.current_text = result.get("final_text", self.current_text)
                cost_info = {"usd_cost": 0.02, "total_tokens": 2000}
        else:
            # 데이터에 없는 경우: 텍스트 유지 + 높은 비용
            log.info(f"[경고] action sequence {actions_tuple}를 찾을 수 없음")
            cost_info = {"usd_cost": 0.05, "total_tokens": 5000}
        
        # 새로운 점수 계산
        new_scores = self.evaluator.evaluate(self.current_text)
        self.current_score = new_scores
        
        # === 보상 계산 (단순화) ===
        # 1) overall 점수 변화
        score_delta = new_scores.overall - prev_scores.overall
        
        # 2) 기본 보상: 점수 향상시 +, 감소시 -
        if score_delta > 0:
            reward = score_delta * 3.0  # 긍정적 변화 강화
        else:
            reward = score_delta * 1.0  # 부정적 변화는 그대로
        
        # 3) LLM 비용 패널티 (step이 진행될수록 비용 증가)
        usd_cost = cost_info.get("usd_cost", 0.02)
        step_cost_multiplier = 1.0 + (self.current_step * 0.15)  # step마다 15% 증가
        reward -= self.cost_lambda * usd_cost * step_cost_multiplier
        
        # 4) 반복 패널티
        if action in self.used_actions:
            reward -= self.repeat_penalty
            info["repeat_penalty"] = True
        else:
            info["repeat_penalty"] = False
        
        self.used_actions.add(action)
        
        # 5) 최대 스텝 도달시 종료 보상
        if self.current_step >= self.max_steps:
            final_bonus = self._terminal_reward(new_scores)
            reward += final_bonus
            done = True
            info["reason"] = "max_steps"
        
        info["new_scores"] = new_scores
        info["score_delta"] = score_delta
        info["llm_cost_usd"] = usd_cost
        
        next_state = self._scores_to_state(new_scores)
        return next_state, reward, done, info
    
    def _terminal_reward(self, scores: DocumentScore) -> float:
        """
        종료 보상: overall 점수 기반 (0~10 → -1~+1)
        """
        # 0~10 → -1 ~ +1로 스케일링
        return (scores.overall - 5.0) / 5.0


# ============================================================
# PPO 정책 네트워크
# ============================================================
class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 네트워크
    - 입력: [grammar, readability, coherence, overall, step, last_action_onehot]
    - 출력: (action_logits, value)
    """
    def __init__(self, state_dim=4, num_actions=5, hidden_size=128):
        super().__init__()
        # step(1) + last_action(num_actions) 추가
        input_dim = state_dim + 1 + num_actions
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


# ============================================================
# PPO Runner
# ============================================================
class PPORunner:
    """PPO 학습 러너"""
    
    def __init__(
        self,
        env: OfflineEditingEnv,
        lr: float = 3e-4,
        gamma: float = 0.95,
        clip_eps: float = 0.2,
        K_epochs: int = 4,
        entropy_coef: float = 0.02,
        hidden_size: int = 128,
    ):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.max_steps = 6
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 정책 네트워크
        self.policy = ActorCriticNetwork(
            state_dim=4,
            num_actions=env.num_actions,
            hidden_size=hidden_size,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.reward_history = []
    
    def _build_obs(self, state, step, last_action):
        """상태 + step + last_action을 concatenate"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        step_tensor = torch.tensor([step / self.max_steps], dtype=torch.float32, device=self.device)
        
        # last_action을 one-hot으로
        action_onehot = torch.zeros(self.env.num_actions, dtype=torch.float32, device=self.device)
        if last_action >= 0:
            action_onehot[last_action] = 1.0
        
        obs = torch.cat([state_tensor, step_tensor, action_onehot])
        return obs
    
    def _collect_trajectory(self):
        """하나의 에피소드 수집"""
        state, _ = self.env.reset()
        
        obs_list = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        infos = []
        
        last_action = -1
        
        for t in range(self.max_steps):
            obs = self._build_obs(state, t, last_action)
            
            with torch.no_grad():
                logits, value = self.policy(obs.unsqueeze(0))
                dist = Categorical(logits=logits)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
            
            action_idx = action_idx.item()
            next_state, reward, done, info = self.env.step(action_idx)
            
            obs_list.append(obs)
            actions.append(action_idx)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            infos.append(info)
            
            last_action = action_idx
            state = next_state
            if done:
                break
        
        return {
            "obs": obs_list, "actions": actions, "log_probs": log_probs,
            "rewards": rewards, "dones": dones, "values": values, "infos": infos,
        }
    
    def _compute_returns(self, rewards, dones):
        """Discounted returns 계산"""
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def _ppo_update(self, traj):
        """PPO 업데이트"""
        obs = torch.stack(traj["obs"])
        actions = torch.tensor(traj["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(traj["log_probs"], dtype=torch.float32, device=self.device)
        returns = self._compute_returns(traj["rewards"], traj["dones"])
        values = torch.tensor(traj["values"], dtype=torch.float32, device=self.device)
        advantages = returns - values
        
        # Advantage normalization
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.K_epochs):
            logits, new_values = self.policy(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns)
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        return policy_loss.item(), entropy.item()
    
    def train(self, num_episodes=500, log_interval=50):
        """PPO 학습"""
        log.info(f"\n{'='*60}")
        log.info(f"PPO 학습 시작 ({num_episodes} 에피소드)")
        log.info(f"액션 공간: {self.env.actions}")
        log.info(f"오버피팅 모드: {self.env.use_single_sequence}")
        log.info(f"{'='*60}")
        
        for ep in range(1, num_episodes + 1):
            traj = self._collect_trajectory()
            ep_return = sum(traj["rewards"])
            self.reward_history.append(ep_return)
            
            policy_loss, entropy = self._ppo_update(traj)
            
            if ep % log_interval == 0:
                avg = sum(self.reward_history[-log_interval:]) / log_interval
                actions = [self.env.actions[a] for a in traj["actions"]]
                
                if traj["infos"]:
                    first_score = traj["infos"][0]["prev_scores"]
                    last_info = traj["infos"][-1]
                    last_score = last_info["new_scores"]
                    log.info(f"[Ep {ep:4d}] return={ep_return:+.3f}, avg={avg:+.3f}, entropy={entropy:.3f}")
                    log.info(f"         actions={actions}")
                    log.info(f"         점수: {first_score.overall:.2f} → {last_score.overall:.2f}")
        
        return self.reward_history
    
    def evaluate_greedy(self, num_eval=5, verbose=True):
        """Greedy 정책 평가"""
        log.info(f"\n{'='*60}")
        log.info("Greedy 평가")
        log.info(f"{'='*60}")
        
        results = []
        for i in range(num_eval):
            state, _ = self.env.reset()
            initial_score = self.env.current_score
            last_action = -1
            actions_taken = []
            total_reward = 0
            
            for t in range(self.max_steps):
                obs = self._build_obs(state, t, last_action)
                with torch.no_grad():
                    logits, _ = self.policy(obs.unsqueeze(0))
                    action = torch.argmax(logits).item()
                
                actions_taken.append(self.env.actions[action])
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                last_action = action
                if done:
                    break
            
            final_score = self.env.current_score
            results.append({
                "return": total_reward,
                "actions": actions_taken,
                "initial": initial_score,
                "final": final_score,
            })
            
            if verbose:
                log.info(f"[Eval {i+1}] return={total_reward:+.3f}")
                log.info(f"         actions={actions_taken}")
                log.info(f"         점수: {initial_score.overall:.2f} → {final_score.overall:.2f}")
                log.info(f"         (g:{initial_score.grammar:.1f}→{final_score.grammar:.1f}, "
                      f"r:{initial_score.readability:.1f}→{final_score.readability:.1f}, "
                      f"c:{initial_score.coherence:.1f}→{final_score.coherence:.1f})")
        
        return results
    
    def show_policy(self):
        """학습된 정책의 액션 확률 분포 확인"""
        state, _ = self.env.reset()
        
        log.info(f"\n현재 문서 점수:")
        log.info(f"  grammar:     {state[0]:.2f}")
        log.info(f"  readability: {state[1]:.2f}")
        log.info(f"  coherence:   {state[2]:.2f}")
        log.info(f"  overall:     {state[3]:.2f}")
        
        obs = self._build_obs(state, 0, -1)
        with torch.no_grad():
            logits, value = self.policy(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        log.info(f"\nStep 1 액션 확률 (Value: {value.item():.3f}):")
        for action, prob in zip(self.env.actions, probs):
            bar = "█" * int(prob * 25)
            log.info(f"  {action:20s}: {prob:.3f} {bar}")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 데이터 파일 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, "first_doc_all_sequences_prefix_reuse_with_noise.jsonl")
    
    if not os.path.exists(jsonl_path):
        log.info(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {jsonl_path}")
        exit(1)
    
    # === 먼저 평가기 테스트 ===
    log.info("="*60)
    log.info("깐깐한 평가기 테스트")
    log.info("="*60)
    
    evaluator = StrictEvaluator()
    
    # 데이터 로드해서 base vs final 비교
    with open(jsonl_path, "r", encoding="utf-8") as f:
        seq = json.loads(f.readline())
    
    base_text = seq["base_text"]
    final_text = seq["final_text"]
    
    base_score = evaluator.evaluate(base_text)
    final_score = evaluator.evaluate(final_text)
    
    log.info(f"\n[저품질 초록 점수]")
    log.info(f"  grammar:     {base_score.grammar:.2f}")
    log.info(f"  readability: {base_score.readability:.2f}")
    log.info(f"  coherence:   {base_score.coherence:.2f}")
    log.info(f"  overall:     {base_score.overall:.2f}")
    
    log.info(f"\n[교정된 초록 점수]")
    log.info(f"  grammar:     {final_score.grammar:.2f}")
    log.info(f"  readability: {final_score.readability:.2f}")
    log.info(f"  coherence:   {final_score.coherence:.2f}")
    log.info(f"  overall:     {final_score.overall:.2f}")
    
    log.info(f"\n점수 차이: {base_score.overall:.2f} → {final_score.overall:.2f} "
          f"(+{final_score.overall - base_score.overall:.2f})")
    
    # 문제점 분석
    issues = evaluator.detailed_report(base_text)
    log.info(f"\n[저품질 초록의 문제점]")
    log.info(f"  모호한 표현: {issues['vague'][:5]}...")
    log.info(f"  어색한 어미: {issues['awkward'][:5]}...")
    log.info(f"  구어체: {issues['colloquial'][:5]}...")
    
    # === 환경 및 학습 ===
    log.info("\n" + "="*60)
    log.info("RL 학습 시작")
    log.info("="*60)
    
    env = OfflineEditingEnv(
        jsonl_path, 
        max_steps=3,
        cost_lambda=0.5,  # 비용 패널티 감소 (학습 용이)
        repeat_penalty=0.2,  # 반복 패널티 감소
        use_single_sequence=True,  # 오버피팅 모드 ON
    )
    
    runner = PPORunner(
        env=env,
        lr=3e-4,
        gamma=0.95,
        clip_eps=0.2,
        K_epochs=4,
        entropy_coef=0.01,  # entropy 감소 (exploitation 강화)
        hidden_size=128,
    )
    
    log.info("\n[학습 전] 정책:")
    runner.show_policy()
    
    # 학습
    rewards = runner.train(num_episodes=1000, log_interval=100)
    
    log.info("\n[학습 후] 정책:")
    runner.show_policy()
    
    # 평가
    runner.evaluate_greedy(num_eval=10)  # 평가 횟수 증가
    
    # 결과 요약
    log.info(f"\n{'='*60}")
    log.info("학습 결과 요약")
    log.info(f"{'='*60}")
    log.info(f"  초기 100ep 평균: {sum(rewards[:100])/100:+.3f}")
    log.info(f"  마지막 100ep 평균: {sum(rewards[-100:])/100:+.3f}")
    log.info(f"  개선도: {sum(rewards[-100:])/100 - sum(rewards[:100])/100:+.3f}")