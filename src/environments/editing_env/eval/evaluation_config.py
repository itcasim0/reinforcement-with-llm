"""
논문 초록 품질 평가를 위한 키워드 및 상수 설정
영어(en)와 한국어(ko) 평가 기준을 정의
"""


class EnglishEvaluationConfig:
    """영어 논문 초록 평가 설정"""

    # 구조 키워드
    STRUCTURE_KEYWORDS = {
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

    # 학술적 연결어
    ACADEMIC_CONNECTIVES = [
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

    # 1인칭 표현
    FIRST_PERSON = ["i ", "we ", "our ", "my ", "me "]

    # 수동태 표현
    PASSIVE_INDICATORS = ["is ", "are ", "was ", "were ", "been ", "being "]

    # 불필요한 수식어
    FILLER_WORDS = ["very", "really", "quite", "somewhat", "rather"]

    # 모호한 표현
    VAGUE_TERMS = [
        "various",
        "several",
        "many",
        "some",
        "few",
        "often",
        "sometimes",
    ]

    # 길이 기준 (단어 수)
    OPTIMAL_WORD_COUNT_MIN = 150
    OPTIMAL_WORD_COUNT_MAX = 300
    OPTIMAL_WORDS_PER_SENTENCE_MIN = 15
    OPTIMAL_WORDS_PER_SENTENCE_MAX = 25
    TARGET_WORDS_PER_SENTENCE = 20


class KoreanEvaluationConfig:
    """한국어 논문 초록 평가 설정"""

    # 구조 키워드
    STRUCTURE_KEYWORDS = {
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

    # 학술적 연결어
    ACADEMIC_CONNECTIVES = [
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

    # 1인칭 표현
    FIRST_PERSON = ["우리는", "본 연구", "저자는", "필자는", "본 논문"]

    # 수동태/피동 표현
    PASSIVE_INDICATORS = [
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

    # 불필요한 수식어
    FILLER_WORDS = ["매우", "아주", "상당히", "조금", "약간", "다소", "꽤"]

    # 모호한 표현
    VAGUE_TERMS = [
        "여러",
        "다양한",
        "몇몇",
        "일부",
        "종종",
        "때때로",
        "가끔",
        "많은",
    ]

    # 길이 기준 (어절 수)
    # 한국어 300어절 ≈ 영어 200단어
    OPTIMAL_WORD_COUNT_MIN = 200
    OPTIMAL_WORD_COUNT_MAX = 500
    OPTIMAL_WORDS_PER_SENTENCE_MIN = 20
    OPTIMAL_WORDS_PER_SENTENCE_MAX = 40
    TARGET_WORDS_PER_SENTENCE = 30


def get_evaluation_config(language: str):
    """언어에 맞는 평가 설정 반환

    Args:
        language: 'en' (영어) 또는 'ko' (한국어)

    Returns:
        EnglishEvaluationConfig 또는 KoreanEvaluationConfig 클래스

    Raises:
        ValueError: 지원하지 않는 언어인 경우
    """
    language = language.lower()

    if language == "en":
        return EnglishEvaluationConfig
    elif language == "ko":
        return KoreanEvaluationConfig
    else:
        raise ValueError(f"지원하지 않는 언어: {language}. 'en' 또는 'ko'를 사용하세요.")
