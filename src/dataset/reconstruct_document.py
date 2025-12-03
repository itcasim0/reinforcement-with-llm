import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import re
import random

# internal
from llm.core import client
from utils.logger_factory import log

VAGUE_PATTERNS = [
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

AWKWARD_ENDINGS = [
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

COLLOQUIAL = [
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

FILLERS = [
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


class TextReconstructorLLM:
    """LLM을 사용하여 텍스트를 재구성하는 클래스"""

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        """
        Args:
            model_name: 사용할 LLM 모델명
        """
        self.model_name = model_name

    def reconstruct_text(self, original_text: str) -> str:
        """
        원본 텍스트를 LLM으로 재구성

        Args:
            original_text: 재구성할 원본 텍스트

        Returns:
            재구성된 텍스트 (문자열)
        """

        prompt = """
        너는 한국어 논문 초록을 의도적으로 저품질로 변형하는 편집기이다.
        아래의 원문 초록을, 주어진 규칙을 엄격하게 지키면서
        가능한 한 품질이 낮은 형태로 재작성하라.

        [1] 길이 및 문장 구조 규칙
        - 전체 길이는 약 60~100 어절로 맞춘다.
        - 전체 문장 수는 6~10문장 정도로 구성한다.
        - 각 문장은 5~15 어절 사이의 짧은 길이로 유지하고,
        문장당 평균 어절 수가 8~12 정도가 되도록 한다.
        - 문장들은 서로 느슨하게 연결되며, 논리적 흐름이나 구조적 전개는 의도적으로 흐릿하게 만든다.

        [2] 반드시 포함해야 하는 표현들 (저품질 강화)
        아래 표현들을 초록 전체에서 최소 20회 이상 등장시키며 적극 사용한다.

        - 구어체/비학술적 표현
        "뭐랄까", "글쎄", "아무튼", "그러니까", "어쩌면", "사실", "솔직히",
        "당연히", "물론", "엄청", "굉장히", "되게", "진짜", "정말로",
        "완전", "이런저런", "요즘", "얼마 전"

        - 불필요한 수식어
        "매우", "아주", "상당히", "조금", "약간", "다소", "꽤",
        "어느 정도", "기본적으로", "일반적으로 말해서", "말하자면",
        "이를테면", "다양한 ", "여러 가지 "

        - 모호한 표현
        "여러", "다양한", "몇몇", "일부", "종종", "때때로", "가끔", "많은",
        "일지도 모르는", "일지도 모를", "있을지도 모르는",
        "아닐까", "않을까", "일 것이다",
        "좀 ", "약간 ", "조금 ",
        "가상의", "어떤 ", "그런 ", "같은 것", "라는 것", "라고 하는",
        "등등", "기타 등등"

        → 의미 명확성을 떨어뜨리기 위해 “여러 가지”, “어떤 느낌”, “그냥 그런 것” 같은 표현을 능동적으로 사용한다.

        [3] 절대 사용하지 말아야 하는 표현들 (학술/구조/수동 스타일 방지)

        아래 단어들은 어떤 형태로든 사용하지 말 것.
        단, 문맥상 꼭 필요한 경우에는 의미가 크게 겹치지 않는
        일상적이고 애매한 표현으로 변환하여 사용한다.

        (1) 구조/학술 키워드 금지
        - 배경, 맥락, 동기, 문제, 과제, 기존, 현재
        - 목적, 목표, 연구, 조사, 탐구, 분석, 규명, 밝히
        - 방법, 접근, 기법, 알고리즘, 프레임워크, 모델, 제안, 제시, 개발, 설계
        - 결과, 발견, 입증, 보여, 달성, 성능, 실험, 분석, 확인
        - 결론, 시사점, 제안, 기여, 의의, 중요, 기대, 향후

        → 필요하면 "이런저런 이야기", "어떤 부분", "흐름", "느낌", "과정" 등
        일상적이고 모호한 표현으로 자율적으로 대체.

        (2) 학술적 연결어 금지
        - 그러나, 하지만, 또한, 더욱이, 따라서, 그러므로, 그럼에도,
        결과적으로, 특히, 구체적으로, 즉, 반면, 한편, 나아가, 이에, 이를 통해

        → 필요하면 "그리고", "그래서인지", "아무튼", "이런저런 분위기로" 등
        비학술적 연결로 자율 대체.

        (3) 학술적 문장 표현 금지
        - 본 연구, 본 논문, 분석하였다, 검증하였다, 확인하였다,
        제안한다, 제시한다, 결과적으로, 구체적으로

        → 필요하면 "이 글 비슷한 것", "살펴보는 듯한 말투" 등
        부드럽고 흐릿한 표현으로 대체.

        (4) 1인칭 금지
        - 우리는, 본 연구, 저자는, 필자는, 본 논문

        → 필요하면 "사람들은", "글을 쓰는 입장에서는" 등으로 변경.

        (5) 수동·피동 표현 금지
        - 되었다, 되어, 된다, 되는, 되고, 됨으로써,
        이루어졌다, 이루어진, 수행되었다, 수행된

        → 필요하면 "그렇게 되는 느낌", "그렇게 흘러가는 것처럼" 등
        덜 공식적이고 덜 구조적인 표현으로 자율 변환.

        (6) 출력 전 최종 자체 점검
        생성된 문장을 출력하기 전에 위의 금지 단어가 0개인지 스스로 확인하고,
        발견되면 자연스러운 다른 단어로 자동 재작성할 것.

        [4] 정보 밀도 낮추기 규칙
        - 데이터셋, 알고리즘, 모델 이름 등 구체적인 고유명사는 모두 피하고,
        "어떤 데이터", "어떤 방식", "이런저런 자료"처럼 두루뭉술하게 표현할 것.

        [5] 명확성 낮추기 규칙
        - 인과관계나 구체적인 절차를 설명하지 말고,
        "이런저런 시도", "대체로 그렇게 되는 느낌"처럼 애매하게 적을 것.
        - 같은 개념을 반복해서 명확하게 설명하지 말고,
        문장마다 다른 말을 하는 것처럼 보이지만, 실제로는 아무 정보도 주지 않는 문장을 만들 것.
        - "무엇을 했는지, 왜 했는지, 무엇이 새롭고 중요한지"가 드러나지 않도록 할 것.

        [6] 최종 출력 형식
        - 아래에 주어지는 원문 초록의 주제는 대략적으로만 유지하되,
        위의 규칙을 모두 지키면서 품질이 낮은 한국어 초록 한 개만 출력한다.
        - 설명, 메타 코멘트, 리스트 등은 출력하지 말고,
        저품질 초록 본문만 결과로 출력한다.

        [원문 초록]
        {INPUT_ABSTRACT}
        """

        content = f"""
        [원본 텍스트] {original_text}

        위 [원본 텍스트]를 위의 [강제 조건]과 [재구성 사항]을 모두 반영해서,
        평가 점수가 낮게 나오는 극단적으로 품질이 낮은 한국어 텍스트로 재작성해줘.
        """

        try:
            # LLM 호출
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
            )

            # --- 응답 파싱 ---
            reconstructed = response.choices[0].message.content
            return reconstructed

        except Exception as e:
            log.error(f"텍스트 재구성 중 오류 발생: {e}")
            return ""


def load_paper_data(json_path: Path) -> List[Dict[str, Any]]:
    """
    논문 JSON 파일 로드 (새 데이터 구조용)

    Args:
        json_path: JSON 파일 경로

    Returns:
        논문 데이터 리스트 (data["papers"])
    """
    log.info(f"Loading data from: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "papers" in data:
        papers = data.get("papers", [])
        log.info(f"Loaded {len(papers)} papers")
        return papers

    log.warning("Unexpected data structure: 'papers' key not found")
    return []


def add_noise(text: str, noise_ratio: float = 0.15) -> str:
    """
    텍스트에 노이즈 추가
    1. VAGUE_PATTERNS: 문장 중간 삽입
    2. AWKWARD_ENDINGS: 문장 끝부분 추가
    3. COLLOQUIAL: 문장 앞 삽입
    4. FILLERS: 단어 앞 삽입

    Args:
        text: 원본 텍스트
        noise_ratio: 노이즈 비율 (0~1)

    Returns:
        노이즈가 추가된 텍스트
    """
    if not text or not text.strip():
        return text

    # 문장 단위로 분리
    parts = re.split(r"([.!?])\s*", text)

    processed_sentences = []
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break

        sentence = parts[i].strip()
        if not sentence:
            continue

        punctuation = parts[i + 1] if i + 1 < len(parts) else "."

        # COLLOQUIAL 문장 앞 삽입
        if random.random() < noise_ratio * 0.7:
            colloquial = random.choice(COLLOQUIAL)
            sentence = f"{colloquial} {sentence}"

        # 문장을 단어로 분리
        words = sentence.split()

        if len(words) > 0:
            # VAGUE_PATTERNS 문장 중간 삽입
            if len(words) > 2 and random.random() < noise_ratio:
                vague = random.choice(VAGUE_PATTERNS)
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, vague)

            # FILLERS 단어 앞 삽입 (여러 개 가능)
            num_fillers = int(len(words) * noise_ratio * 0.5)
            if num_fillers > 0:
                filler_indices = random.sample(
                    range(len(words)), min(num_fillers, len(words))
                )
                for idx in sorted(filler_indices, reverse=True):
                    words.insert(idx, random.choice(FILLERS))

            sentence = " ".join(words)

        # AWKWARD_ENDINGS 문장 끝에 추가
        if random.random() < noise_ratio * 0.6:
            awkward = random.choice(AWKWARD_ENDINGS)
            sentence = sentence + " " + awkward

        # 문장과 처리된 문자들 병합.
        processed_sentences.append(sentence + punctuation)

    return " ".join(processed_sentences)


def adaptive_add_noise(text: str, pattern_score: float) -> tuple:
    """
    패턴 점수에 따라 noise 강도를 조절하여 적용

    Args:
        text: 원본 텍스트
        pattern_score: 패턴 포함 점수 (0.0 ~ 1.0)

    Returns:
        (noised_text, noise_applied, noise_ratio_used)
    """
    # 패턴 점수에 따라 noise 결정
    if pattern_score < 0.3:
        # 패턴 부족은 강하게 처리할거임
        noise_ratio = 0.3
        noise_applied = True
    elif pattern_score < 0.6:
        # 적당히 부족하면 덜 강하게처리할거임
        noise_ratio = 0.15
        noise_applied = True
    else:
        # 패턴이 많으면 노이즈 안넣음
        return text, False, 0.0

    noised_text = add_noise(text, noise_ratio)
    return noised_text, noise_applied, noise_ratio


def check_pattern_coverage(text: str) -> dict:
    """
    텍스트에 각 패턴이 얼마나 포함됐는지 체크

    Args:
        text: 분석할 텍스트

    Returns:
        {
            'vague_count': int,
            'awkward_count': int,
            'colloquial_count': int,
            'fillers_count': int,
            'total_score': float  # 0.0 ~ 1.0
        }
    """
    if not text:
        return {
            "vague_count": 0,
            "awkward_count": 0,
            "colloquial_count": 0,
            "fillers_count": 0,
            "total_score": 0.0,
        }

    # 각 패턴 카운트
    vague_count = sum(1 for pattern in VAGUE_PATTERNS if pattern in text)
    awkward_count = sum(1 for pattern in AWKWARD_ENDINGS if pattern in text)
    colloquial_count = sum(1 for pattern in COLLOQUIAL if pattern in text)
    fillers_count = sum(1 for pattern in FILLERS if pattern in text)

    # 총 패턴 개수
    total_patterns = vague_count + awkward_count + colloquial_count + fillers_count

    # 문장 수 추정 (구두점 기준)
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))

    # 점수 계산: 문장당 패턴 비율 (0.0 ~ 1.0 정규화)
    # 문장당 2개 이상이면 1.0으로 간주
    patterns_per_sentence = total_patterns / sentence_count
    total_score = min(1.0, patterns_per_sentence / 2.0)

    return {
        "vague_count": vague_count,
        "awkward_count": awkward_count,
        "colloquial_count": colloquial_count,
        "fillers_count": fillers_count,
        "total_score": total_score,
    }


def reconstruct_paper(
    doc_path: Path,
    output_path: Path = None,
    max_docs: int = None,
    model_name: str = "openai/gpt-4o-mini",
):
    """
    논문 데이터를 처리하여 텍스트 재구성 (개별 저장 버전)
    """
    for p in doc_path:

        log.info(f"Input: {p}")

        papers = load_paper_data(p)
        if not papers:
            log.error("No papers to process")
            return

        if max_docs:
            papers = papers[:max_docs]

        log.info(f"Processing {len(papers)} papers")

        reconstructor = TextReconstructorLLM(model_name=model_name)

        # output JSON 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"{p.stem}_{timestamp}.json"

        # 파일이 없다면 빈 리스트 형태로 초기화
        if not output_file.exists():
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"results": []}, f, ensure_ascii=False, indent=2)

        for idx, paper in enumerate(papers):
            title = paper.get("title", "N/A")
            log.info(f"[{idx+1}/{len(papers)}] Processing: {title}")

            abstract = (paper.get("abstract") or "").strip()
            if not abstract:
                log.warning(f"No abstract found for {title}")
                continue

            # LLM 재구성
            reconstructed = reconstructor.reconstruct_text(abstract)

            # 패턴 체크
            pattern_coverage = check_pattern_coverage(reconstructed)
            pattern_score = pattern_coverage["total_score"]

            # 점수에 따라 노이즈 적용
            noised, noise_applied, noise_ratio = adaptive_add_noise(
                reconstructed, pattern_score
            )

            paper_result = {
                "doc_id": paper.get("doc_id"),
                "title": title,
                "author": paper.get("author"),
                "journal": paper.get("journal"),
                "abstract_original": abstract,
                "abstract_reconstructed": reconstructed,
                "abstract_noised": noised,
                "metadata": {
                    "pattern_score": pattern_score,
                    "noise_applied": noise_applied,
                },
            }

            # 기존 파일 내용 읽기
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # append
            data["results"].append(paper_result)

            # 다시 저장
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(f"All results saved to: {output_file}")
