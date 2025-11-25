"""
1. 길이 점수 (가중치 15%)

단어 수를 세서 점수 부여
149212 단어: 0.81.0점 (최적)
175 단어: 1.0점 (완벽)
범위 벗어날수록 감점 (최저 0.3점)

2. 구조 점수 (가중치 30%)

키워드 존재 여부로 체크
배경: "existing", "current", "however" 등 → 있으면 +0.171점
방법: "we propose", "we present", "our approach" 등 → 있으면 +0.246점
결과: "achieve", "outperform", "demonstrate" 등 → 있으면 +0.268점
3가지 모두 있으면 보너스 +0.2점
최대 1.0점

3. 언어 점수 (가중치 20%)

4가지 요소를 각각 체크 (각 0.25점씩)
숫자 포함: "95%", "10x" 등 → +0.167점
비교 표현: "better than", "outperform" 등 → +0.097점
1인칭: "we", "our" → +0.227점
쉼표 개수: 11개 근처(±3개) → +0.25점, 멀수록 감점

4. 내용 점수 (가중치 20%)

4가지 키워드 체크 (각 0.25점씩)
평가/실험: "evaluation", "experiment" → +0.157점
데이터셋: "ImageNet", "COCO", "dataset" → +0.102점
성능: "performance", "accuracy" → +0.142점
모델: "model", "architecture" → +0.186점

5. 문장 점수 (가중치 15%)

평균 단어 수/문장 계산
1525 단어/문장: 0.81.0점 (최적)
20 단어/문장: 1.0점 (완벽)
너무 짧거나 길면 감점
"""

import re
import numpy as np
from typing import Dict
from collections import Counter


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
        self.structure_keywords = {
            "background": [
                "background",
                "context",
                "motivation",
                "problem",
                "challenge",
            ],
            "objective": [
                "aim",
                "goal",
                "objective",
                "purpose",
                "investigate",
                "study",
                "explore",
            ],
            "method": [
                "method",
                "approach",
                "technique",
                "algorithm",
                "framework",
                "model",
                "propose",
            ],
            "result": [
                "result",
                "finding",
                "demonstrate",
                "show",
                "achieve",
                "performance",
            ],
            "conclusion": [
                "conclude",
                "implication",
                "suggest",
                "contribution",
                "significant",
            ],
        }

        self.academic_connectives = [
            "however",
            "moreover",
            "furthermore",
            "therefore",
            "thus",
            "consequently",
            "nevertheless",
            "additionally",
            "specifically",
        ]

        self.first_person = ["i ", "we ", "our ", "my ", "me "]
        self.passive_indicators = ["is ", "are ", "was ", "were ", "been ", "being "]
        self.filler_words = ["very", "really", "quite", "somewhat", "rather"]
        self.vague_terms = [
            "various",
            "several",
            "many",
            "some",
            "few",
            "often",
            "sometimes",
        ]

        # 영어 단어 길이 기준
        self.optimal_word_count_min = 150
        self.optimal_word_count_max = 300
        self.optimal_words_per_sentence_min = 15
        self.optimal_words_per_sentence_max = 25
        self.target_words_per_sentence = 20

    def _init_korean_keywords(self):
        """한국어 키워드 초기화"""
        self.structure_keywords = {
            "background": [
                "배경",
                "맥락",
                "동기",
                "문제",
                "과제",
                "기존",
                "현재",
                "기존의",
                "현재의",
            ],
            "objective": [
                "목적",
                "목표",
                "목적은",
                "목표는",
                "연구",
                "조사",
                "탐구",
                "분석",
                "규명",
                "밝히",
            ],
            "method": [
                "방법",
                "접근",
                "기법",
                "알고리즘",
                "프레임워크",
                "모델",
                "제안",
                "제시",
                "개발",
                "설계",
            ],
            "result": [
                "결과",
                "발견",
                "입증",
                "보여",
                "달성",
                "성능",
                "실험",
                "분석",
                "확인",
            ],
            "conclusion": [
                "결론",
                "시사점",
                "제안",
                "기여",
                "의의",
                "중요",
                "기대",
                "향후",
            ],
        }

        self.academic_connectives = [
            "그러나",
            "하지만",
            "또한",
            "더욱이",
            "따라서",
            "그러므로",
            "그럼에도",
            "결과적으로",
            "특히",
            "구체적으로",
            "즉",
            "반면",
            "한편",
            "나아가",
            "이에",
            "이를 통해",
        ]

        # 한국어는 1인칭 표현이 다름
        self.first_person = ["우리는", "본 연구", "저자는", "필자는", "본 논문"]

        # 한국어 수동태/피동 표현
        self.passive_indicators = [
            "되었다",
            "되어",
            "된다",
            "되는",
            "되고",
            "됨으로써",
            "이루어졌다",
            "이루어진",
            "수행되었다",
            "수행된",
        ]

        # 한국어 불필요한 수식어
        self.filler_words = ["매우", "아주", "상당히", "조금", "약간", "다소", "꽤"]

        # 한국어 모호한 표현
        self.vague_terms = [
            "여러",
            "다양한",
            "몇몇",
            "일부",
            "종종",
            "때때로",
            "가끔",
            "많은",
        ]

        # 한국어는 어절 기준으로 평가 (영어보다 짧음)
        # 한국어 300어절 ≈ 영어 200단어
        self.optimal_word_count_min = 200  # 어절
        self.optimal_word_count_max = 500  # 어절
        self.optimal_words_per_sentence_min = 20
        self.optimal_words_per_sentence_max = 40
        self.target_words_per_sentence = 30

    def evaluate_structure_completeness(self, abstract: str) -> Dict[str, float]:
        """구조적 완성도 평가"""
        if self.language == "en":
            abstract_lower = abstract.lower()
        else:
            abstract_lower = abstract  # 한국어는 대소문자 구분 없음

        scores = {}

        for section, keywords in self.structure_keywords.items():
            keyword_count = sum(1 for kw in keywords if kw in abstract_lower)
            scores[section] = min(1.0, keyword_count / len(keywords))

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
        connective_score = min(1.0, connective_count / 3)

        # 1인칭 사용
        first_person_count = sum(abstract_lower.count(fp) for fp in self.first_person)

        # 한국어는 1인칭 사용이 더 자연스러울 수 있음
        if self.language == "ko":
            # 한국어는 1인칭을 덜 패널티
            first_person_penalty = max(0, 1.0 - first_person_count * 0.1)
        else:
            first_person_penalty = max(0, 1.0 - first_person_count * 0.2)

        # 수동태/피동
        passive_count = sum(abstract_lower.count(pi) for pi in self.passive_indicators)
        passive_score = min(1.0, passive_count / 5)

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
        words = abstract.split()
        word_count = len(words)

        if word_count == 0:
            return {"information_density_score": 0.0}

        # 숫자/통계
        numbers = re.findall(r"\d+\.?\d*%?", abstract)
        has_numbers = len(numbers) > 0
        number_score = 1.0 if has_numbers else 0.5

        # 전문 용어 (영어는 대문자, 한국어는 한자어/외래어)
        if self.language == "en":
            technical_terms = re.findall(r"\b[A-Z][a-z]+(?:-[A-Za-z]+)*\b", abstract)
        else:
            # 한국어: 괄호 안 영어, 대문자로 시작하는 영어 단어
            technical_terms = re.findall(r"\([A-Za-z\s]+\)|[A-Z][a-z]+", abstract)

        technical_density = len(technical_terms) / word_count
        technical_score = min(1.0, technical_density * 10)

        # 불필요한 수식어
        filler_count = 0
        for fw in self.filler_words:
            if self.language == "en":
                filler_count += abstract.lower().count(f" {fw} ")
            else:
                filler_count += abstract.count(fw)

        filler_penalty = max(0, 1.0 - filler_count * 0.3)

        density_score = (number_score + technical_score + filler_penalty) / 3

        return {
            "has_numbers": has_numbers,
            "number_count": len(numbers),
            "number_score": number_score,
            "technical_term_count": len(technical_terms),
            "technical_score": technical_score,
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

        vague_penalty = max(0, 1.0 - vague_count * 0.15)

        # 약어 설명
        abbreviations_with_explanation = re.findall(
            r"\b([A-Z]{2,})\s*\([^)]+\)", abstract
        )
        abbreviations_without = re.findall(r"\b[A-Z]{2,}\b", abstract)

        if len(abbreviations_without) > 0:
            abbrev_explanation_score = len(abbreviations_with_explanation) / len(
                set(abbreviations_without)
            )
        else:
            abbrev_explanation_score = 1.0

        clarity_score = (vague_penalty + abbrev_explanation_score) / 2

        return {
            "vague_count": vague_count,
            "vague_penalty": vague_penalty,
            "abbreviations_explained": len(abbreviations_with_explanation),
            "abbreviations_total": len(set(abbreviations_without)),
            "abbrev_explanation_score": abbrev_explanation_score,
            "clarity_score": clarity_score,
        }

    def evaluate_coherence(self, abstract: str) -> Dict[str, float]:
        """일관성 평가"""
        sentences = [s.strip() for s in re.split(r"[.!?]+", abstract) if s.strip()]

        if len(sentences) < 2:
            return {"coherence_score": 0.5}

        # 연결어 사용
        transition_count = 0
        for sent in sentences[1:]:
            if self.language == "en":
                sent_lower = sent.lower()
            else:
                sent_lower = sent

            if any(conn in sent_lower for conn in self.academic_connectives):
                transition_count += 1

        transition_score = min(1.0, transition_count / (len(sentences) - 1))

        # 반복 개념
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
        concept_consistency = min(1.0, len(repeated_concepts) / 5)

        coherence_score = (transition_score + concept_consistency) / 2

        return {
            "transition_count": transition_count,
            "transition_score": transition_score,
            "repeated_concepts": len(repeated_concepts),
            "concept_consistency": concept_consistency,
            "coherence_score": coherence_score,
        }

    def evaluate_abstract(self, abstract: str) -> Dict:
        """종합 평가"""
        structure = self.evaluate_structure_completeness(abstract)
        length = self.evaluate_length(abstract)
        academic = self.evaluate_academic_style(abstract)
        density = self.evaluate_information_density(abstract)
        clarity = self.evaluate_clarity(abstract)
        coherence = self.evaluate_coherence(abstract)

        weights = {
            "structure": 0.25,
            "length": 0.10,
            "academic": 0.15,
            "density": 0.20,
            "clarity": 0.15,
            "coherence": 0.15,
        }

        overall_score = (
            structure["structure_completeness"] * weights["structure"]
            + length["overall_length_score"] * weights["length"]
            + academic["academic_style_score"] * weights["academic"]
            + density["information_density_score"] * weights["density"]
            + clarity["clarity_score"] * weights["clarity"]
            + coherence["coherence_score"] * weights["coherence"]
        )

        return {
            "overall_score": overall_score,
            "structure": structure,
            "length": length,
            "academic_style": academic,
            "information_density": density,
            "clarity": clarity,
            "coherence": coherence,
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


if __name__ == "__main__":
    print("=" * 80)
    print("영어 논문 초록 평가 테스트")
    print("=" * 80)

    evaluator_en = AbstractQualityEvaluator(language="en")

    sample_abstract_en = """
    Deep learning has revolutionized computer vision, but training large models requires 
    substantial computational resources. This study proposes an efficient training framework 
    that reduces GPU memory usage by 40% while maintaining model accuracy. Our approach 
    combines gradient checkpointing with mixed-precision training, achieving 1.5x speedup 
    compared to baseline methods. Experiments on ImageNet demonstrate that our method reaches 
    76.3% top-1 accuracy, comparable to standard training. These findings suggest that 
    efficient training techniques can significantly reduce costs without sacrificing performance.
    """

    results_en = evaluator_en.evaluate_abstract(sample_abstract_en)

    print(f"\nOverall Score: {results_en['overall_score']:.3f} - {results_en['grade']}")
    print(f"Language: {results_en['language']}")
    print(f"\nDetailed Scores:")
    print(f"  Structure: {results_en['structure']['structure_completeness']:.3f}")
    print(
        f"  Length: {results_en['length']['overall_length_score']:.3f} ({results_en['length']['word_count']} words)"
    )
    print(
        f"  Academic Style: {results_en['academic_style']['academic_style_score']:.3f}"
    )
    print(
        f"  Information: {results_en['information_density']['information_density_score']:.3f}"
    )
    print(f"  Clarity: {results_en['clarity']['clarity_score']:.3f}")
    print(f"  Coherence: {results_en['coherence']['coherence_score']:.3f}")

    print("\n" + "=" * 80)
    print("한국어 논문 초록 평가 테스트")
    print("=" * 80)

    evaluator_ko = AbstractQualityEvaluator(language="ko")

    sample_abstract_ko = """
    딥러닝은 컴퓨터 비전 분야를 혁신적으로 변화시켰으나, 대규모 모델 학습에는 상당한 
    계산 자원이 필요하다. 본 연구는 모델 정확도를 유지하면서 GPU 메모리 사용량을 40% 
    감소시키는 효율적인 학습 프레임워크를 제안한다. 우리의 접근법은 그래디언트 체크포인팅과 
    혼합 정밀도 학습을 결합하여 기존 방법 대비 1.5배의 속도 향상을 달성한다. ImageNet 
    데이터셋을 활용한 실험 결과, 제안된 방법이 표준 학습과 유사한 76.3%의 top-1 정확도를 
    달성함을 입증하였다. 이러한 결과는 효율적인 학습 기법이 성능 저하 없이 비용을 크게 
    절감할 수 있음을 시사한다.
    """

    results_ko = evaluator_ko.evaluate_abstract(sample_abstract_ko)

    print(f"\n전체 점수: {results_ko['overall_score']:.3f} - {results_ko['grade']}")
    print(f"언어: {results_ko['language']}")
    print(f"\n세부 점수:")
    print(f"  구조: {results_ko['structure']['structure_completeness']:.3f}")
    print(
        f"  길이: {results_ko['length']['overall_length_score']:.3f} ({results_ko['length']['word_count']} 어절)"
    )
    print(f"  학술 스타일: {results_ko['academic_style']['academic_style_score']:.3f}")
    print(
        f"  정보 밀도: {results_ko['information_density']['information_density_score']:.3f}"
    )
    print(f"  명확성: {results_ko['clarity']['clarity_score']:.3f}")
    print(f"  일관성: {results_ko['coherence']['coherence_score']:.3f}")
