import re

# internal
from environments.editing_env.components.component import DocumentScore

from .strict_evaluation_config import StrictEvaluationConfig


class StrictEvaluator:
    """저품질 초록의 특징을 정확히 잡아내는 평가기"""

    def __init__(self):
        # 설정 파일에서 패턴 가져오기
        config = StrictEvaluationConfig

        self.vague_patterns = config.VAGUE_PATTERNS
        self.awkward_endings = config.AWKWARD_ENDINGS
        self.colloquial = config.COLLOQUIAL
        self.fillers = config.FILLERS
        self.redundant = config.REDUNDANT
        self.academic_positive = config.ACADEMIC_POSITIVE
        self.structure_keywords = config.STRUCTURE_KEYWORDS
        self.connectives = config.CONNECTIVES

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

    def score(self, text: str) -> DocumentScore:
        """
        PPORunner와 호환되도록 evaluate()를 score()로 래핑
        """
        return self.evaluate(text)

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
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
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
        conn_count = sum(1 for c in self.connectives if c in text)
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
