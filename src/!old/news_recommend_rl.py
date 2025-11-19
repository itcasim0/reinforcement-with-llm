"""
MIND 데이터셋 기반 뉴스 추천 강화학습 시스템 (개선 버전)

개선 사항:
1. 실제 LLM 통합 (CandidateLLM 사용)
2. Q-Learning 정책 추가
3. Ground Truth 기반 보상 개선
4. 데이터 경로 자동 검증
"""

import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json

# TODO: 실제 경로에 맞게 import 수정 필요
from llm.core import CandidateLLM


# ========== Enums ==========
class TimeSlot(Enum):
    """시간대 구분"""
    MORNING = "morning"
    LUNCH = "lunch"
    EVENING = "evening"
    NIGHT = "night"


class SummaryLength(Enum):
    """요약 길이"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


# ========== Data Classes ==========
@dataclass
class NewsItem:
    """개별 뉴스 아이템"""
    news_id: str
    category: str
    title: str
    abstract: str


@dataclass
class UserAction:
    """사용자의 실제 반응 데이터"""
    clicked: bool = False
    read_time: float = 0.0
    shared: bool = False
    liked: bool = False


@dataclass
class Episode:
    """한 사용자의 뉴스 소비 세션"""
    user_id: str
    history: List[str]
    candidates: List[NewsItem]
    timestamp: str
    ground_truth_clicks: List[str]
    
    def get_time_slot(self) -> TimeSlot:
        """타임스탬프를 시간대로 변환"""
        try:
            dt = datetime.strptime(self.timestamp, "%m/%d/%Y %I:%M:%S %p")
            hour = dt.hour
            
            if 6 <= hour < 12:
                return TimeSlot.MORNING
            elif 12 <= hour < 14:
                return TimeSlot.LUNCH
            elif 14 <= hour < 20:
                return TimeSlot.EVENING
            else:
                return TimeSlot.NIGHT
        except:
            return TimeSlot.MORNING


@dataclass
class MDPState:
    """강화학습 State 정의

    Returns:
        user_history: 사용자가 과거에 본 뉴스 카테고리
        current_time: 현재 시간대 (아/점/저/밤)
        click_rate: 현재까지의 클릭률
        candidate_categories: 추천 가능한 뉴스들의 카테고리 리스트
        current_step: 현재 스텝, 현재 몇 번째 추천인가
        max_step: 최대 스텝
        read_completion: 사용자가 뉴스를 끝까지 읽을 확률
        remain_budget: 남은 일일 예산 (LLM 요약때문에 필요)
        last_summary_length: 마지막으로 추천한 뉴스 요약 길이
        total_cost: 지금까지 사용한 총 비용
    """
    user_history: List[str]
    current_time: TimeSlot
    click_rate: float
    candidate_categories: List[str]
    current_step: int
    max_step: int
    read_completion: float = 0.7
    remain_budget: float = 10.0 # '$'
    last_summary_length: Optional[SummaryLength] = None
    total_cost: float = 0.0 
    
    # def to_tuple(self) -> tuple:
    #     """최소 state"""
        
    #     # 사용자 관심사와 후보 뉴스 매칭 개수만
    #     user_interests = set(self.user_history)
    #     matches = sum(1 for cat in self.candidate_categories 
    #                 if cat in user_interests)
    #     match_level = min(matches, 2)  # 0, 1, 2+
        
    #     # 시간대
    #     time = self.current_time.value
        
    #     return (
    #         match_level,  # 0, 1, 2+ (3가지)
    #         time         # 4가지
    #     )
    def to_tuple(self) -> tuple:
        """Q-Learning을 위한 해시 가능한 state 표현 (개선)"""
        # 사용자 히스토리 (최대 3개 카테고리)
        # history_str = ','.join(sorted(set(self.user_history[-3:]))) if self.user_history else "none"
        
        # 후보 뉴스 중 관심사 매칭 개수
        user_interests = set(self.user_history)
        interest_matches = sum(1 for cat in self.candidate_categories if cat in user_interests)
        interest_level = min(interest_matches, 3)  # 0, 1, 2, 3, 사용자 관심사 매칭도
        
        # 클릭률 (3단계로 간소화)
        if self.click_rate < 0.3:
            click_level = 0  # 낮음
        elif self.click_rate < 0.7:
            click_level = 1  # 중간
        else:
            click_level = 2  # 높음
        
        # 현재 스텝 추가
        step = self.current_step
        
        return (
            # history_str,                  
            self.current_time.value,        
            interest_level,                
            click_level,
            step
        )

def build_llms() -> List:
    """LLM 후보 생성"""

    return [
        CandidateLLM("google/gemini-2.0-flash-001", "가성비", {"input_price": 0.10, "output_price": 0.40}),
        CandidateLLM("anthropic/claude-sonnet-4", "고품질", {"input_price": 3.0, "output_price": 15.0}),
        CandidateLLM("google/gemma-3-12b-it", "오픈소스", {"input_price": 0.04, "output_price": 0.14}),
    ]
        


# ========== MIND Data Loader ==========
class MINDDataLoader:
    """MIND 데이터셋 로더"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.news_df = None
        self.behaviors_df = None
    
    def load_data(self):
        """news.tsv와 behaviors.tsv 로드"""
        print(f"📂 Loading MIND data from {self.data_dir}...")
        
        news_path = os.path.join(self.data_dir, "news.tsv")
        if os.path.exists(news_path):
            self.news_df = pd.read_csv(
                news_path,
                sep='\t',
                header=None,
                names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                       'url', 'title_entities', 'abstract_entities']
            )
            print(f"Loaded {len(self.news_df)} news articles")
            print(f"   Categories: {list(self.news_df['category'].value_counts().head(10).items())}")
        else:
            raise FileNotFoundError(f"news.tsv not found in {self.data_dir}")
        
        behaviors_path = os.path.join(self.data_dir, "behaviors.tsv")
        if os.path.exists(behaviors_path):
            self.behaviors_df = pd.read_csv(
                behaviors_path,
                sep='\t',
                header=None,
                names=['impression_id', 'user_id', 'time', 'history', 'impressions']
            )
            print(f"Loaded {len(self.behaviors_df)} behavior logs")
            print(f"   Unique users: {self.behaviors_df['user_id'].nunique()}")
        else:
            raise FileNotFoundError(f"behaviors.tsv not found in {self.data_dir}")
    
    def create_episodes(self, num_episodes: int = 100, min_candidates: int = 3) -> List[Episode]:
        """MIND 데이터로부터 Episode 생성"""
        if self.news_df is None or self.behaviors_df is None:
            raise ValueError("데이터를 먼저 로드하세요 (load_data() 호출)")
        
        print(f"\nCreating {num_episodes} episodes from MIND data...")
        
        episodes = []
        sampled_behaviors = self.behaviors_df.sample(min(num_episodes * 2, len(self.behaviors_df)))
        
        for idx, row in sampled_behaviors.iterrows():
            try:
                user_id = row['user_id']
                timestamp = row['time']
                
                # History 파싱
                history = []
                if pd.notna(row['history']):
                    history_ids = row['history'].split()
                    for news_id in history_ids[-10:]:
                        news_info = self.news_df[self.news_df['news_id'] == news_id]
                        if not news_info.empty:
                            history.append(news_info.iloc[0]['category'])
                
                # Impressions 파싱
                if pd.isna(row['impressions']):
                    continue
                
                impressions = row['impressions'].split()
                candidates = []
                ground_truth_clicks = []
                
                for impression in impressions:
                    parts = impression.rsplit('-', 1)
                    if len(parts) != 2:
                        continue
                    
                    news_id, clicked = parts
                    news_info = self.news_df[self.news_df['news_id'] == news_id]
                    if not news_info.empty:
                        news_row = news_info.iloc[0]
                        candidates.append(NewsItem(
                            news_id=news_id,
                            category=news_row['category'],
                            title=news_row['title'] if pd.notna(news_row['title']) else "No title",
                            abstract=news_row['abstract'] if pd.notna(news_row['abstract']) else "No abstract"
                        ))
                        
                        if clicked == '1':
                            ground_truth_clicks.append(news_id)
                
                if len(candidates) >= min_candidates:
                    if len(candidates) > 8:
                        clicked_news = [c for c in candidates if c.news_id in ground_truth_clicks]
                        not_clicked_news = [c for c in candidates if c.news_id not in ground_truth_clicks]
                        num_not_clicked = min(8 - len(clicked_news), len(not_clicked_news))
                        if num_not_clicked > 0:
                            candidates = clicked_news + random.sample(not_clicked_news, num_not_clicked)
                        else:
                            candidates = clicked_news[:8]
                        random.shuffle(candidates)
                    
                    episode = Episode(
                        user_id=user_id,
                        history=history[-5:] if history else [],
                        candidates=candidates,
                        timestamp=timestamp,
                        ground_truth_clicks=ground_truth_clicks
                    )
                    episodes.append(episode)
                    
                    if len(episodes) >= num_episodes:
                        break
                        
            except Exception as e:
                continue
        
        print(f"Created {len(episodes)} episodes")
        return episodes


# ========== Environment ==========
class NewsRecommendationEnv:
    """뉴스 추천 환경"""
    
    def __init__(self, max_step: int = 5, daily_budget: float = 1.0, alpha: float = 0.05,
                 llms: List = None):
        self.max_step = max_step
        self.daily_budget = daily_budget
        self.alpha = alpha
        self.llm = self._build_single_llm()
        
        # 요약 길이별 비용 (평균값 사용)
        self.summary_costs = {
            SummaryLength.SHORT: 0.03,
            SummaryLength.MEDIUM: 0.08,
            SummaryLength.LONG: 0.15
        }
    
    def _build_single_llm(self):
        """단일 LLM 생성 - Mock 버전"""
        
        class MockLLM:
            def __init__(self):
                self.model = "mock-gemini-flash"
            
            def answer(self, question):
                # 요약 길이에 따라 비용 다르게
                if "짧게" in question:
                    out_tokens = 30
                    cost = 0.005  # 매우 작은 비용
                elif "상세하게" in question:
                    out_tokens = 100
                    cost = 0.02
                else:
                    out_tokens = 60
                    cost = 0.01
                
                summary = f"[Mock] 뉴스 요약"
                return summary, out_tokens, cost, True
        
        return MockLLM()
    # def _build_single_llm(self):
    #     """단일 LLM 생성 (gemini-flash 사용)"""
    #     from llm.core import CandidateLLM
        
    #     return CandidateLLM(
    #         "google/gemini-2.0-flash-001",
    #         "가성비 모델",
    #         {"input_price": 0.10, "output_price": 0.40}
    #     )
    
    def reset(self, episode: Episode) -> MDPState:
        """환경 초기화"""
        self.episode = episode
        self.current_step = 0
        self.total_cost = 0.0
        self.action_history = []
        self.user_actions = []
        self.click_history = []
        
        return self._get_state()
    
    def _get_state(self) -> MDPState:
        """현재 state 반환"""
        click_rate = sum(self.click_history) / len(self.click_history) if self.click_history else 0.5
        
        last_summary_length = None
        if self.action_history:
            last_action = self.action_history[-1]
            if last_action[0] == "RECOMMEND" and len(last_action) > 2:
                last_summary_length = last_action[2]
        
        return MDPState(
            user_history=self.episode.history,
            current_time=self.episode.get_time_slot(),
            click_rate=click_rate,
            read_completion=0.7,
            candidate_categories=[n.category for n in self.episode.candidates],
            last_summary_length=last_summary_length,
            total_cost=self.total_cost,
            remain_budget=self.daily_budget - self.total_cost,
            current_step=self.current_step,
            max_step=self.max_step
        )
    
    def step(self, action: Tuple) -> Tuple[MDPState, float, bool]:
        """행동 수행 및 보상 계산"""
        reward = 0.0
        done = False
        
        if action[0] == "STOP":
            done = True
            reward = self._calculate_final_reward()
            self.action_history.append(action)
            
        elif action[0] == "RECOMMEND":
            news_idx = action[1]
            summary_length = action[2] if len(action) > 2 else SummaryLength.MEDIUM
            
            if news_idx >= len(self.episode.candidates):
                reward = -1.0
                done = True
            else:
                news = self.episode.candidates[news_idx]
                
                # 고정된 LLM 사용 (선택 불필요)
                llm = self.llm
                
                # 요약 길이에 따른 프롬프트 생성
                length_instruction = {
                    SummaryLength.SHORT: "2-3문장으로 짧게",
                    SummaryLength.MEDIUM: "5-6문장으로",
                    SummaryLength.LONG: "상세하게"
                }
                
                question = f"다음 뉴스를 {length_instruction[summary_length]} 요약해주세요.\n\n제목: {news.title}\n내용: {news.abstract}"
                
                # LLM API 호출
                summary, out_tokens, cost, ok = llm.answer(question)
                
                self.total_cost += cost
                
                # 사용자 반응 시뮬레이션
                user_action = self._simulate_user_action(news, summary_length)
                
                self.user_actions.append(user_action)
                self.click_history.append(1.0 if user_action.clicked else 0.0)
                self.action_history.append((action[0], news_idx, summary_length))
                
                click_reward = 1.0 if user_action.clicked else -0.3
                cost_penalty = -self.alpha * cost
                gt_bonus = 0.5 if news.news_id in self.episode.ground_truth_clicks else 0.0
        
                reward = click_reward + cost_penalty + gt_bonus
                # # 즉시 보상: 비용 패널티
                # cost_penalty = -self.alpha * (cost / self.daily_budget)
                # reward = cost_penalty
                
                self.current_step += 1
        
        else:
            reward = -1.0
            done = True
        
        # 최대 스텝 도달
        if self.current_step >= self.max_step and not done:
            done = True
            reward += self._calculate_final_reward()
        
        return self._get_state(), reward, done
    
    def _simulate_user_action(self, news: NewsItem, summary_length: SummaryLength) -> UserAction:
        """
        뉴스 추천에 대한 사용자 반응 시뮬레이션
        """
        is_ground_truth = news.news_id in self.episode.ground_truth_clicks
        
        if is_ground_truth:
            # GT에 있으면 높은 확률로 클릭
            clicked = random.random() < 0.9
            read_time = random.uniform(30, 60)
            shared = random.random() < 0.3
        else:
            # 사용자 관심사 매칭 확인
            user_interests = set(self.episode.history)
            interest_match = news.category in user_interests
            
            click_prob = 0.3 if interest_match else 0.1
            clicked = random.random() < click_prob
            read_time = random.uniform(10, 25) if clicked else 0.0
            shared = random.random() < 0.05 if clicked else False
        
        return UserAction(
            clicked=clicked,
            read_time=read_time,
            shared=shared,
            liked=random.random() < 0.1
        )
    
    def _calculate_outcome_score(self, user_action: UserAction, summary_length: SummaryLength) -> float:
        """사용자 반응 점수 계산"""
        click_score = 1.0 if user_action.clicked else 0.0
        
        expected_time = {
            SummaryLength.SHORT: 20,
            SummaryLength.MEDIUM: 35,
            SummaryLength.LONG: 50
        }[summary_length]
        
        read_score = min(user_action.read_time / expected_time, 1.0)
        engagement = 1.0 if user_action.shared else 0.0
        
        outcome = 0.5 * click_score + 0.3 * read_score + 0.2 * engagement
        return outcome
    
    def _calculate_final_reward(self) -> float:
        """종료 시 최종 보상 (비용 제거)"""
        if not self.user_actions:
            return -1.0
        
        # 클릭률 계산
        clicks = sum(1 for ua in self.user_actions if ua.clicked)
        click_rate = clicks / len(self.user_actions)
        
        # 읽기 시간 계산
        avg_read_time = sum(ua.read_time for ua in self.user_actions) / len(self.user_actions)
        read_score = min(avg_read_time / 30.0, 1.0)  # 30초 기준
        
        # 참여도 (공유, 좋아요)
        engagement = sum(1 for ua in self.user_actions if ua.shared or ua.liked)
        engagement_score = engagement / len(self.user_actions)
        
        # 최종 보상: 클릭 50% + 읽기 30% + 참여 20%
        final_reward = 0.5 * click_rate + 0.3 * read_score + 0.2 * engagement_score
        
        return final_reward

# ========== Q-Learning Policy ==========
class QLearningPolicy:
    """Q-Learning 정책
    
    sample:
    Q-table = {
        ('morning', 3, 1, 0): {
            'REC_0_short': 0.88,   # 이 상황에서 뉴스[0] 짧게 추천 → 점수 0.88
            'REC_1_medium': 1.32,  # 이 상황에서 뉴스[1] 중간 추천 → 점수 1.32 최고
            'REC_2_long': 0.65,
            'STOP': -0.87,         # 이 상황에서 중단 → 점수 -0.87 (나쁨)
        },
        ('night', 2, 0, 1): {
            'REC_0_short': 0.45,
            ...
        }
    }
    """
    
    def __init__(self, learning_rate: float = 0.2, discount_factor: float = 0.9, 
                 epsilon: float = 0.8, epsilon_decay: float = 0.96):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        
        # Q-table: {state: {action: Q-value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 학습 통계
        self.training_episodes = 0
    
    def get_action_space(self, state: MDPState) -> List[Tuple]:
        """ 에이전트가 가능한 행동 액션 리스트 (뉴스 선택 + 요약 길이)"""
        actions = [("STOP",)]
        
        # 각 후보 뉴스 × 요약 길이 조합
        for idx in range(len(state.candidate_categories)):
            for length in SummaryLength:  # 요약 길이 추가
                actions.append(("RECOMMEND", idx, length))
        
        return actions
        
    def act(self, state: MDPState, training: bool = True) -> Tuple:
        """액션 선택 (epsilon-greedy)"""
        action_space = self.get_action_space(state)
        state_key = state.to_tuple()
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # 탐험: 랜덤 액션
            return random.choice(action_space)
        else:
            # 이용: Q-value 최대화
            action_values = {
                self._action_to_key(action): self.q_table[state_key][self._action_to_key(action)]
                for action in action_space
            }
            
            if not action_values:
                return random.choice(action_space)
            
            #모든 Q-value가 0이면 (처음 보는 state) STOP 제외하고 랜덤 선택
            max_q_value = max(action_values.values())
            
            if max_q_value == 0.0:
                # STOP 제외한 RECOMMEND 액션들만
                non_stop_actions = [a for a in action_space if a[0] != "STOP"]
                if non_stop_actions:
                    return random.choice(non_stop_actions)
            
            best_action_key = max(action_values, key=action_values.get)
            return self._key_to_action(best_action_key)
    
    def update(self, state: MDPState, action: Tuple, reward: float, 
               next_state: MDPState, done: bool):
        """Q-value 업데이트"""
        state_key = state.to_tuple()
        action_key = self._action_to_key(action)
        
        current_q = self.q_table[state_key][action_key]
        
        if done:
            target_q = reward
        else:
            next_state_key = next_state.to_tuple()
            next_actions = self.get_action_space(next_state)
            max_next_q = max(
                [self.q_table[next_state_key][self._action_to_key(a)] for a in next_actions],
                default=0.0
            )
            target_q = reward + self.gamma * max_next_q
        
        # Q-value 업데이트
        self.q_table[state_key][action_key] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Epsilon 감소"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _action_to_key(self, action: Tuple) -> str:
        """액션을 해시 가능한 키로 변환"""
        if action[0] == "STOP":
            return "STOP"
        elif action[0] == "RECOMMEND":
            idx = action[1]
            length = action[2].value if len(action) > 2 else "medium"
            return f"REC_{idx}_{length}"
        return str(action)
    
    def _key_to_action(self, key: str) -> Tuple:
        """키를 액션으로 변환"""
        if key == "STOP":
            return ("STOP",)
        elif key.startswith("REC_"):
            parts = key.split("_")
            idx = int(parts[1])
            length_str = parts[2]
            length = SummaryLength(length_str)
            return ("RECOMMEND", idx, length)
        return ("STOP",)
    
    def save_model(self, filepath: str):
        """Q-table 저장"""
        # defaultdict를 일반 dict로 변환하면서 튜플 키를 문자열로 변환
        q_dict = {}
        for state_key, actions in self.q_table.items():
            # 튜플 키를 문자열로 변환
            state_str = str(state_key) if isinstance(state_key, tuple) else state_key
            q_dict[state_str] = dict(actions)
        
        model_data = {
            "q_table": q_dict,
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Q-table saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Q-table 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in model_data["q_table"].items():
            # 문자열을 다시 튜플로 변환 (eval 사용)
            try:
                state_key = eval(state_str)
            except:
                state_key = state_str
                
            for action_key, q_value in actions.items():
                self.q_table[state_key][action_key] = q_value
        
        self.epsilon = model_data.get("epsilon", self.epsilon)
        self.training_episodes = model_data.get("training_episodes", 0)
        
        print(f"📂 Q-table loaded from {filepath}")

# ========== Baseline Policies ==========
class GreedyPolicy:
    """탐욕 정책 - 사용자 히스토리 기반"""
    
    def __init__(self):
        self.time_preferred_length = {
            TimeSlot.MORNING: SummaryLength.SHORT,
            TimeSlot.LUNCH: SummaryLength.MEDIUM,
            TimeSlot.EVENING: SummaryLength.LONG,
            TimeSlot.NIGHT: SummaryLength.MEDIUM
        }
    
    def act(self, state: MDPState) -> Tuple:
        if state.current_step >= state.max_step - 1:
            return ("STOP",)
        
        # 사용자 관심사와 가장 매칭되는 뉴스 선택
        user_interests = set(state.user_history)
        selected_idx = None
        
        for idx, category in enumerate(state.candidate_categories):
            if category in user_interests:
                selected_idx = idx
                break
        
        if selected_idx is None and state.candidate_categories:
            selected_idx = 0
        
        if selected_idx is not None:
            # 시간대에 따른 요약 길이 선택
            preferred_length = self.time_preferred_length[state.current_time]
            
            return ("RECOMMEND", selected_idx, preferred_length)
        
        return ("STOP",)

# ========== Training & Evaluation ==========
def train_q_learning(env: NewsRecommendationEnv, episodes: List[Episode], 
                     num_epochs: int = 30, save_path: str = None):
    """Q-Learning 정책 학습"""
    policy = QLearningPolicy(learning_rate=0.2, discount_factor=0.9, epsilon=0.8)
    
    print(f"\n{'='*80}")
    print(f"Q-Learning 학습 시작 (Epochs: {num_epochs}, Episodes: {len(episodes)})")
    print("=" * 80)
    
    all_rewards = []
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        
        for ep_idx, episode in enumerate(episodes, 1):
            state = env.reset(episode)
            episode_reward = 0.0
            done = False
            
            trajectory = []  # (state, action, reward) 기록
            
            while not done:
                action = policy.act(state, training=True)
                next_state, reward, done = env.step(action)
                
                # Q-value 업데이트
                policy.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                trajectory.append((state, action, reward))
                state = next_state
            
            epoch_rewards.append(episode_reward)
            policy.training_episodes += 1
            
            if ep_idx % 20 == 0:
                avg_reward = sum(epoch_rewards[-20:]) / len(epoch_rewards[-20:])
                print(f"  [Epoch {epoch+1}] Ep {ep_idx}/{len(episodes)} | "
                      f"Avg Reward (last 20): {avg_reward:.3f} | "
                      f"Epsilon: {policy.epsilon:.3f}")
        
        policy.decay_epsilon()
        all_rewards.extend(epoch_rewards)
        
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"\n  Epoch {epoch+1} 완료 | Avg Reward: {avg_epoch_reward:.3f}\n")
    
    if save_path:
        policy.save_model(save_path)
    
    print("\n 학습된 Q-table 통계:")
    print(f"   총 state 수: {len(policy.q_table)}")
    print(f"   평균 방문 횟수: {1000 / len(policy.q_table):.1f}회")

    # 샘플 Q-value 출력
    sample_states = list(policy.q_table.keys())[:5]
    for state in sample_states:
        print(f"\n   State: {state}")
        actions = policy.q_table[state]
        top_3 = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]
        for action, q in top_3:
            print(f"     {action}: Q={q:.4f}")
    
    return policy


def evaluate_policy(env: NewsRecommendationEnv, episodes: List[Episode], 
                    policy, policy_name: str):
    """정책 평가"""
    print(f"\n{'='*80}")
    print(f"정책 평가: {policy_name}")
    print("=" * 80)
    
    total_rewards = []
    total_costs = []
    total_clicks = []
    total_accuracy = []
    
    for ep_idx, episode in enumerate(episodes, 1):
        state = env.reset(episode)
        episode_reward = 0.0
        done = False
        
        # 첫 에피소드 디버깅
        if ep_idx == 1:
            print(f"\n첫 번째 에피소드 디버깅:")
            print(f"   초기 state - step: {state.current_step}, max_step: {state.max_step}")
            print(f"   candidates: {len(state.candidate_categories)}개")
        
        step_count = 0
        while not done:
            # Q-Learning일 경우 training=False로 설정
            if isinstance(policy, QLearningPolicy):
                action = policy.act(state, training=False)
            else:
                action = policy.act(state)
            
            # 첫 에피소드의 첫 액션 확인
            if ep_idx == 1 and step_count == 0:
                print(f"   첫 액션: {action}")
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1
            
            # 무한루프 방지
            if step_count > 10:
                print(f" 스텝이 10회 초과, 강제 종료")
                break
        
        # 첫 에피소드 결과
        if ep_idx == 1:
            print(f"   총 스텝: {step_count}")
            print(f"   추천 수: {len(env.user_actions)}")
            print(f"   에피소드 보상: {episode_reward:.3f}")
        
        # 통계 수집
        clicks = sum(1 for ua in env.user_actions if ua.clicked)
        total_recs = len(env.user_actions)
        
        recommended_ids = [episode.candidates[ah[1]].news_id for ah in env.action_history if ah[0] == "RECOMMEND"]
        correct_recs = sum(1 for news_id in recommended_ids if news_id in episode.ground_truth_clicks)
        accuracy = correct_recs / len(episode.ground_truth_clicks) if episode.ground_truth_clicks else 0.0
        
        total_rewards.append(episode_reward)
        total_costs.append(env.total_cost)
        if total_recs > 0:
            total_clicks.append(clicks / total_recs)
        else:
            total_clicks.append(0.0)
        total_accuracy.append(accuracy)
    
    # 결과 출력
    print(f"\n[{policy_name} 정책 종합 결과]")
    print(f"  평균 보상: {sum(total_rewards)/len(total_rewards):.3f}")
    print(f"  평균 비용: ${sum(total_costs)/len(total_costs):.2f}")
    if total_clicks:
        print(f"  평균 클릭률: {sum(total_clicks)/len(total_clicks)*100:.1f}%")
    else:
        print(f"  평균 클릭률: 0.0% (추천 없음)")
    print(f"  평균 GT Accuracy: {sum(total_accuracy)/len(total_accuracy)*100:.1f}%")
    print("=" * 80)
    
    return {
        "avg_reward": sum(total_rewards)/len(total_rewards),
        "avg_cost": sum(total_costs)/len(total_costs),
        "avg_click_rate": sum(total_clicks)/len(total_clicks) if total_clicks else 0.0,
        "avg_accuracy": sum(total_accuracy)/len(total_accuracy)
    }

# ========== Q-Learning 의사결정 시각화 ==========
def visualize_q_learning_decision(policy: QLearningPolicy, env: NewsRecommendationEnv, 
                                   episode: Episode, verbose: bool = True):
    """
    Q-Learning 정책의 의사결정 과정을 상세히 출력
    
    Args:
        policy: 학습된 Q-Learning 정책
        env: 환경
        episode: 테스트할 에피소드
        verbose: 상세 출력 여부
    """
    print("\n" + "="*80)
    print(" Q-Learning 의사결정 과정 분석")
    print("="*80)
    
    # 에피소드 정보
    print(f"\n 에피소드 정보:")
    print(f"   사용자 ID: {episode.user_id}")
    print(f"   시간: {episode.timestamp} ({episode.get_time_slot().value})")
    print(f"   사용자 히스토리: {episode.history}")
    print(f"   Ground Truth 클릭: {episode.ground_truth_clicks}")
    
    print(f"\n 후보 뉴스 목록:")
    for idx, news in enumerate(episode.candidates):
        gt_mark = "⭐" if news.news_id in episode.ground_truth_clicks else ""
        print(f"   [{idx}] {news.category:15s} | {news.title[:50]}... {gt_mark}")
    
    # 환경 초기화
    state = env.reset(episode)
    done = False
    step_num = 0
    
    while not done and step_num < 10:
        step_num += 1
        print(f"\n{'─'*80}")
        print(f" Step {step_num}")
        print(f"{'─'*80}")
        
        # 현재 state 정보
        print(f"\n 현재 State:")
        print(f"   user_history: {state.user_history}")
        print(f"   current_time: {state.current_time.value}")
        print(f"   click_rate: {state.click_rate:.2f}")
        print(f"   current_step: {state.current_step}/{state.max_step}")
        print(f"   total_cost: ${state.total_cost:.2f}")
        
        # State를 Q-table key로 변환
        state_key = state.to_tuple()
        print(f"\n State Key (Q-table 검색용):")
        print(f"   {state_key}")
        
        # 가능한 액션들과 Q-value 확인
        action_space = policy.get_action_space(state)
        print(f"\n 가능한 액션과 Q-value:")
        
        action_q_values = []
        for action in action_space[:10]:  # 처음 10개만 출력
            action_key = policy._action_to_key(action)
            q_value = policy.q_table[state_key][action_key]
            action_q_values.append((action, q_value))
            
            if action[0] == "STOP":
                print(f"   STOP                              Q={q_value:.4f}")
            else:
                news_idx = action[1]
                length = action[2].value
                category = state.candidate_categories[news_idx]
                print(f"   REC[{news_idx}] {category:12s} {length:6s}  Q={q_value:.4f}")
        
        if len(action_space) > 10:
            print(f"   ... (총 {len(action_space)}개 액션 중 10개만 표시)")
        
        # 최고 Q-value 찾기
        max_q = max([q for _, q in action_q_values])
        best_actions = [a for a, q in action_q_values if q == max_q]
        
        print(f"\n 최고 Q-value: {max_q:.4f}")
        print(f"   후보 액션 수: {len(best_actions)}개")
        
        # Q-Learning 정책으로 액션 선택
        selected_action = policy.act(state, training=False)
        
        print(f"\n 선택된 액션:")
        if selected_action[0] == "STOP":
            print(f"   STOP")
        else:
            news_idx = selected_action[1]
            length = selected_action[2].value
            news = episode.candidates[news_idx]
            gt_mark = "(GT 클릭!)" if news.news_id in episode.ground_truth_clicks else ""
            
            print(f"   뉴스 인덱스: {news_idx}")
            print(f"   카테고리: {news.category}")
            print(f"   요약 길이: {length}")
            print(f"   제목: {news.title[:60]}...")
            print(f"   {gt_mark}")
            
            # 선택 이유 분석
            print(f"\n선택 이유:")
            if max_q == 0.0:
                print(f"     처음 보는 state (Q-value 모두 0)")
                print(f"   → STOP 제외하고 랜덤 선택")
            elif max_q > 0:
                print(f"    학습된 경험 활용 (양수 Q-value)")
                print(f"   → 과거에 좋은 결과를 낸 액션")
            else:
                print(f"     학습된 경험상 좋지 않음 (음수 Q-value)")
                print(f"   → 그나마 덜 나쁜 선택")
            
            # 사용자 선호도와 매칭 확인
            user_interests = set(state.user_history)
            if news.category in user_interests:
                print(f"    사용자 관심사와 일치! (history에 '{news.category}' 있음)")
            else:
                print(f"     사용자 관심사와 불일치 (history: {user_interests})")
        
        # 환경에서 액션 실행
        next_state, reward, done = env.step(selected_action)
        
        print(f"\n 결과:")
        print(f"   즉시 보상: {reward:.4f}")
        
        if selected_action[0] == "RECOMMEND":
            user_action = env.user_actions[-1]
            print(f"   사용자 반응:")
            print(f"     - 클릭: {' Yes' if user_action.clicked else ' No'}")
            print(f"     - 읽기 시간: {user_action.read_time:.1f}초")
            print(f"     - 공유: {' Yes' if user_action.shared else ' No'}")
        
        if done:
            print(f"\n 에피소드 종료")
            print(f"   최종 보상: {reward:.4f}")
            print(f"   총 비용: ${env.total_cost:.2f}")
            print(f"   총 추천 수: {len(env.user_actions)}")
            
            if env.user_actions:
                clicks = sum(1 for ua in env.user_actions if ua.clicked)
                click_rate = clicks / len(env.user_actions)
                print(f"   클릭률: {click_rate*100:.1f}%")
        
        state = next_state
    
    print("\n" + "="*80)


def demo_q_learning_decisions(q_policy: QLearningPolicy, env: NewsRecommendationEnv, 
                               test_episodes: List[Episode], num_demos: int = 3):
    """
    학습된 Q-Learning의 의사결정을 여러 에피소드에서 시연
    
    Args:
        q_policy: 학습된 Q-Learning 정책
        env: 환경
        test_episodes: 테스트 에피소드들
        num_demos: 시연할 에피소드 수
    """
    print("\n" + "="*80)
    print(" Q-Learning 의사결정 시연")
    print("="*80)
    
    for i in range(min(num_demos, len(test_episodes))):
        episode = test_episodes[i]
        visualize_q_learning_decision(q_policy, env, episode, verbose=True)
        
        if i < num_demos - 1:
            input("\n 다음 에피소드를 보려면 Enter를 누르세요...")

# ========== Main ==========
def main():
    """메인 실행 함수"""
    
    print("="*80)
    print("MIND 데이터셋 기반 뉴스 추천 강화학습 (LLM 고정 버전)")
    print("="*80)
    
    DATA_DIR = "src/news_data"
    
    if not os.path.exists(DATA_DIR):
        print(f"\n 데이터 디렉토리가 없습니다: {DATA_DIR}")
        return
    
    # 데이터 로드
    loader = MINDDataLoader(DATA_DIR)
    
    try:
        loader.load_data()
    except FileNotFoundError as e:
        print(f"\n {e}")
        return
    
    # 에피소드 생성
    train_episodes = loader.create_episodes(num_episodes=500, min_candidates=4)
    test_episodes = loader.create_episodes(num_episodes=50, min_candidates=4)

    if not train_episodes or not test_episodes:
        print("\n 에피소드를 생성할 수 없습니다.")
        return
    
    #  환경 초기화 (LLM 파라미터 제거)
    env = NewsRecommendationEnv(
        max_step=4,
        daily_budget=10.0,  # 예산 증가
        alpha=0.05            # 비용 패널티 감소
    )
    
    print(f"\n💡 사용 LLM: {env.llm.model}")
    print(f"   예산: ${env.daily_budget}, Alpha: {env.alpha}")
    
    # Q-Learning 학습
    q_policy = train_q_learning(env, train_episodes, num_epochs=30, save_path="q_table.json")
    
    #  의사결정 과정 시연 추가
    print("\n" + "="*80)
    print(" 학습된 Q-Learning의 의사결정 과정을 확인하시겠습니까?")
    print("="*80)
    
    demo_q_learning_decisions(q_policy, env, test_episodes, num_demos=3)

    # 베이스라인 정책
    greedy_policy = GreedyPolicy()
    
    # 평가
    results = {}
    results["Q-Learning"] = evaluate_policy(env, test_episodes, q_policy, "Q-Learning")
    results["Greedy"] = evaluate_policy(env, test_episodes, greedy_policy, "Greedy")
    
    # 비교
    print(f"\n{'='*80}")
    print("정책 비교")
    print("=" * 80)
    for policy_name, metrics in results.items():
        print(f"\n{policy_name}:")
        for metric_name, value in metrics.items():
            if "cost" in metric_name:
                print(f"  {metric_name}: ${value:.3f}")
            elif "rate" in metric_name or "accuracy" in metric_name:
                print(f"  {metric_name}: {value*100:.1f}%")
            else:
                print(f"  {metric_name}: {value:.3f}")

if __name__ == "__main__":
    main()