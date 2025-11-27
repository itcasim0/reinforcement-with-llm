import re

# internal
from environments.editing_env.components.component import DocumentScore


class StrictEvaluator:
    """저품질 초록의 특징을 정확히 잡아내는 평가기"""

    def __init__(self):
        # === 감점 패턴들 ===

        # 1. 모호한/불필요한 표현 (심각도 높음)
        self.vague_patterns = [
            "일지도 모르는",
            "일지도 모를",
            "있을지도 모르는",
            "아닐까",
            "않을까",
            "일 것이다",
            "좀 ",
            "약간 ",
            "조금 ",
            "가상의",
            "어떤 ",
            "그런 ",
            "같은 것",
            "라는 것",
            "라고 하는",
            "등등",
            "기타 등등",
        ]

        # 2. 어색한 어미 (심각도 높음)
        self.awkward_endings = [
            "해보아 했다",
            "해보아야 했다",
            "인 것이다",
            "인 것이었다",
            "라고 한다",
            "다고 한다",
            "했던 것이다",
            "였던 것이다",
            "하는 바이다",
            "되는 바이다",
            "것이라고",
            "것이었다고",
        ]

        # 3. 구어체/비학술적 표현 (심각도 중간)
        self.colloquial = [
            "뭐랄까",
            "글쎄",
            "아무튼",
            "그러니까",
            "어쩌면",
            "사실",
            "솔직히",
            "당연히",
            "물론",
            "엄청",
            "굉장히",
            "되게",
            "진짜",
            "정말로",
            "완전",
            "이런저런",
            "요즘",
            "얼마 전",
        ]

        # 4. 불필요한 수식/군더더기 (심각도 중간)
        self.fillers = [
            "매우 ",
            "아주 ",
            "상당히 ",
            "다소 ",
            "꽤 ",
            "어느 정도",
            "기본적으로",
            "일반적으로 말해서",
            "말하자면",
            "이를테면",
            "다양한 ",
            "여러 가지 ",
        ]

        # 5. 반복/중복 패턴
        self.redundant = [
            "즉, 다시 말해",
            "다시 말해서",
            "요약하자면, 결론적으로",
        ]

        # === 가점 패턴들 (학술적 표현) ===
        self.academic_positive = [
            "본 연구",
            "본 논문",
            "분석하였다",
            "검증하였다",
            "확인하였다",
            "제안한다",
            "제시한다",
            "따라서",
            "그러나",
            "한편",
            "결과적으로",
            "구체적으로",
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
