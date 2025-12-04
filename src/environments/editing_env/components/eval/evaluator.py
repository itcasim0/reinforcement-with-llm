import re
import numpy as np
from typing import Dict
from collections import Counter

# internal
from utils.logger_factory import log

from .evaluation_config import KoreanEvaluationConfig


class AbstractQualityEvaluator:
    """
    논문 초록의 품질을 객관적으로 평가하는 클래스 (영어/한국어 지원)

    Parameters:
        language (str): 'en' (영어, 기본값) 또는 'ko' (한국어)

    Usage:
        # 영어 논문 평가
        evaluator = AbstractQualityEvaluator(language='en')

        # 한국어 논문 평가
        evaluator = AbstractQualityEvaluator(language='ko')
    """

    def __init__(self, language="en"):
        """
        Args:
            language: 'en' (영어) 또는 'ko' (한국어)
        """
        self.language = language.lower()

        if self.language == "en":
            self._init_english_keywords()
        elif self.language == "ko":
            self._init_korean_keywords()
        else:
            raise ValueError(
                f"지원하지 않는 언어: {language}. 'en' 또는 'ko'를 사용하세요."
            )

    def _init_english_keywords(self):
        """영어 키워드 초기화"""
        config = KoreanEvaluationConfig

        self.structure_keywords = config.STRUCTURE_KEYWORDS
        self.academic_connectives = config.ACADEMIC_CONNECTIVES
        self.first_person = config.FIRST_PERSON
        self.passive_indicators = config.PASSIVE_INDICATORS
        self.filler_words = config.FILLER_WORDS
        self.vague_terms = config.VAGUE_TERMS

        # 길이 기준
        self.optimal_word_count_min = config.OPTIMAL_WORD_COUNT_MIN
        self.optimal_word_count_max = config.OPTIMAL_WORD_COUNT_MAX
        self.optimal_words_per_sentence_min = config.OPTIMAL_WORDS_PER_SENTENCE_MIN
        self.optimal_words_per_sentence_max = config.OPTIMAL_WORDS_PER_SENTENCE_MAX
        self.target_words_per_sentence = config.TARGET_WORDS_PER_SENTENCE

    def _init_korean_keywords(self):
        """한국어 키워드 초기화"""
        config = KoreanEvaluationConfig

        self.structure_keywords = config.STRUCTURE_KEYWORDS
        self.academic_connectives = config.ACADEMIC_CONNECTIVES
        self.first_person = config.FIRST_PERSON
        self.passive_indicators = config.PASSIVE_INDICATORS
        self.filler_words = config.FILLER_WORDS
        self.vague_terms = config.VAGUE_TERMS

        # 길이 기준
        self.optimal_word_count_min = config.OPTIMAL_WORD_COUNT_MIN
        self.optimal_word_count_max = config.OPTIMAL_WORD_COUNT_MAX
        self.optimal_words_per_sentence_min = config.OPTIMAL_WORDS_PER_SENTENCE_MIN
        self.optimal_words_per_sentence_max = config.OPTIMAL_WORDS_PER_SENTENCE_MAX
        self.target_words_per_sentence = config.TARGET_WORDS_PER_SENTENCE

    def evaluate_structure_completeness(self, abstract: str) -> Dict[str, float]:
        """구조적 완성도 평가"""
        if self.language == "en":
            abstract_lower = abstract.lower()
        else:
            abstract_lower = abstract  # 한국어는 대소문자 구분 없음

        scores = {}

        for section, keywords in self.structure_keywords.items():
            keyword_count = sum(1 for kw in keywords if kw in abstract_lower)

            scores[section] = min(1.0, keyword_count)

        structure_completeness = np.mean(list(scores.values()))

        return {
            "section_scores": scores,
            "structure_completeness": structure_completeness,
            "missing_sections": [k for k, v in scores.items() if v < 0.2],
        }

    def evaluate_length(self, abstract: str) -> Dict[str, float]:
        """길이 평가"""
        words = abstract.split()
        word_count = len(words)

        # 문장 분리 (한국어/영어 공통)
        sentences = [s.strip() for s in re.split(r"[.!?]+", abstract) if s.strip()]
        sentence_count = len(sentences)

        # 길이 점수
        if self.optimal_word_count_min <= word_count <= self.optimal_word_count_max:
            length_score = 1.0
        elif word_count < self.optimal_word_count_min:
            length_score = word_count / self.optimal_word_count_min
        else:
            length_score = max(
                0,
                1.0
                - (word_count - self.optimal_word_count_max)
                / self.optimal_word_count_max,
            )

        # 문장 길이 점수
        avg_words_per_sentence = word_count / max(1, sentence_count)

        if (
            self.optimal_words_per_sentence_min
            <= avg_words_per_sentence
            <= self.optimal_words_per_sentence_max
        ):
            sentence_length_score = 1.0
        else:
            sentence_length_score = max(
                0,
                1.0
                - abs(avg_words_per_sentence - self.target_words_per_sentence)
                / self.target_words_per_sentence,
            )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "length_score": length_score,
            "sentence_length_score": sentence_length_score,
            "overall_length_score": (length_score + sentence_length_score) / 2,
        }

    def evaluate_academic_style(self, abstract: str) -> Dict[str, float]:
        """학술적 표현 평가"""
        if self.language == "en":
            abstract_lower = abstract.lower()
        else:
            abstract_lower = abstract

        # 연결어
        connective_count = sum(
            1 for conn in self.academic_connectives if conn in abstract_lower
        )
        connective_score = min(1.0, connective_count)

        # 1인칭 사용
        first_person_count = sum(abstract_lower.count(fp) for fp in self.first_person)

        # 한국어는 1인칭 사용이 더 자연스러울 수 있음
        if self.language == "ko":
            # 한국어는 1인칭을 가점
            first_person_penalty = min(1.0, first_person_count)
        else:
            first_person_penalty = max(0, 1.0 - first_person_count * 0.2)

        # 수동태/피동
        passive_count = sum(abstract_lower.count(pi) for pi in self.passive_indicators)
        passive_score = min(1.0, passive_count)

        academic_score = (connective_score + first_person_penalty + passive_score) / 3

        return {
            "connective_count": connective_count,
            "connective_score": connective_score,
            "first_person_count": first_person_count,
            "first_person_penalty": first_person_penalty,
            "passive_score": passive_score,
            "academic_style_score": academic_score,
        }

    def evaluate_information_density(self, abstract: str) -> Dict[str, float]:
        """정보 밀도 평가"""

        # TODO: 시간되면, tokenizer 써가지고 토큰이나 형태소단위로 분리해보는 것도 나쁘지 않을 듯
        words = abstract.split()
        word_count = len(words)

        if word_count == 0:
            return {"information_density_score": 0.0}

        # 숫자/통계
        numbers = re.findall(r"\d+\.?\d*%?", abstract)
        has_numbers = len(numbers) > 0
        number_score = 1.0 if has_numbers else 0.5

        # 불필요한 수식어
        filler_count = 0
        for fw in self.filler_words:
            if self.language == "en":
                filler_count += abstract.lower().count(f" {fw} ")
            else:
                filler_count += abstract.count(fw)

        filler_penalty = max(0, 1.0 - filler_count * 0.1)

        density_score = (number_score + filler_penalty) / 2

        return {
            "number_count": len(numbers),
            "number_score": number_score,
            "filler_count": filler_count,
            "filler_penalty": filler_penalty,
            "information_density_score": density_score,
        }

    def evaluate_clarity(self, abstract: str) -> Dict[str, float]:
        """명확성 평가"""
        if self.language == "en":
            abstract_lower = abstract.lower()
        else:
            abstract_lower = abstract

        # 모호한 표현
        vague_count = 0
        for vt in self.vague_terms:
            if self.language == "en":
                vague_count += abstract_lower.count(f" {vt} ")
            else:
                vague_count += abstract_lower.count(vt)

        vague_penalty = max(0, 1.0 - vague_count * 0.1)

        # 반복 개념
        sentences = [s.strip() for s in re.split(r"[.!?]+", abstract) if s.strip()]

        all_words = []
        for sent in sentences:
            if self.language == "en":
                words = [w.lower() for w in re.findall(r"\b\w+\b", sent) if len(w) > 4]
            else:
                # 한국어: 형태소 단위가 아닌 어절 단위로 (간단한 처리)
                words = [w for w in sent.split() if len(w) > 1]
            all_words.extend(words)

        word_freq = Counter(all_words)
        repeated_concepts = [w for w, c in word_freq.items() if c >= 2]

        concept_consistency = max(0, 1.0 - len(repeated_concepts) * 0.1)

        clarity_score = (vague_penalty + concept_consistency) / 2

        return {
            "vague_count": vague_count,
            "vague_penalty": vague_penalty,
            "repeated_concepts": len(repeated_concepts),
            "concept_consistency": concept_consistency,
            "clarity_score": clarity_score,
        }

    def evaluate_abstract(self, abstract: str) -> Dict:
        """종합 평가"""
        structure = self.evaluate_structure_completeness(abstract)
        length = self.evaluate_length(abstract)
        academic = self.evaluate_academic_style(abstract)
        density = self.evaluate_information_density(abstract)
        clarity = self.evaluate_clarity(abstract)

        weights = {
            "structure": 0.25,
            "length": 0.15,
            "academic": 0.15,
            "density": 0.20,
            "clarity": 0.20,
        }

        overall_score = (
            structure["structure_completeness"] * weights["structure"]
            + length["overall_length_score"] * weights["length"]
            + academic["academic_style_score"] * weights["academic"]
            + density["information_density_score"] * weights["density"]
            + clarity["clarity_score"] * weights["clarity"]
        )

        return {
            "overall_score": overall_score,
            "structure": structure,
            "length": length,
            "academic_style": academic,
            "information_density": density,
            "clarity": clarity,
            "grade": self._score_to_grade(overall_score),
            "language": self.language,
        }

    def _score_to_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
