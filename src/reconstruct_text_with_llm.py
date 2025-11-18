"""
LLM을 사용하여 논문 요약 데이터의 original_text를 재구성하는 스크립트

데이터 구조:
- JSON 파일에는 논문 정보가 포함되어 있으며
- 각 논문에는 summary_entire 필드가 있고
- summary_entire에는 orginal_text와 summary_text가 포함됨
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from llm.core import client
from utils.logger_factory import log


class TextReconstructorLLM:
    """LLM을 사용하여 텍스트를 재구성하는 클래스"""

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        """
        Args:
            model_name: 사용할 LLM 모델명
        """
        self.model_name = model_name

    def reconstruct_text(self, original_text: str) -> Dict[str, Any]:
        """
        원본 텍스트를 LLM으로 재구성

        Args:
            original_text: 재구성할 원본 텍스트

        Returns:
            재구성 결과 딕셔너리 (reconstructed_text, tokens, cost)
        """
        # 프롬프트 구성
        prompt = f"""[원본 텍스트]를 [재구성 사항]을 참고하여 적절히 바꿔줘.

[재구성 사항]
1. 문법, 맞춤법, 띄어쓰기, 시제 오류가 있도록 해.
2. 표현과 문장 구조를 더 모호하고 애매하게 하여 독자가 쉽게 따라가지 못하도록 해.
3. 중복되거나 불필요한 부분을 추가하고 장황하게 구성해.
4. 문장과 문단 순서를 조정해서 논리가 없는 흐름으로 부자연스럽게 해줘.
"""
        content = f"""[원본 텍스트]
{original_text}

위 내용을 [재구성 사항]을 참고하여 텍스트에 오류가 많도록 재구성해줘."""

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
            content = response.choices[0].message.content
            return content

        except Exception as e:
            log.error(f"텍스트 재구성 중 오류 발생: {e}")
            return ""


def load_paper_data(json_path: str) -> List[Dict[str, Any]]:
    """
    논문 JSON 파일 로드

    Args:
        json_path: JSON 파일 경로

    Returns:
        논문 데이터 리스트
    """
    log.info(f"Loading data from: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data: List[dict] = json.load(f)

    # 데이터 구조: [{"totalcount": N, "data": [...]}]
    if isinstance(data, list) and len(data) > 0:
        papers = data[0].get("data", [])
        log.info(f"Loaded {len(papers)} papers")
        return papers

    log.warning("Unexpected data structure")
    return []


def reconstruct_paper(
    json_path: Path,
    output_path: Path = None,
    max_papers: int = None,
    model_name: str = "openai/gpt-4o-mini",
):
    """
    논문 데이터를 처리하여 텍스트 재구성

    Args:
        json_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (None이면 자동 생성)
        max_papers: 하나의 JSON내에 논문 개수 제한
        model_name: 사용할 LLM 모델명
    """

    # 데이터 로드
    papers = load_paper_data(json_path)

    if not papers:
        log.error("No papers to process")
        return

    # 처리할 아이템 수 제한
    if max_papers:
        papers = papers[:max_papers]

    log.info(f"Processing {len(papers)} papers")

    # llm기반 reconstructor 초기화
    reconstructor = TextReconstructorLLM(model_name=model_name)

    # 결과 저장 리스트
    results = []

    # 각 논문 처리
    for idx, paper in enumerate(papers):
        log.info(
            f"Processing paper {idx + 1}/{len(papers)}: {paper.get('title', 'N/A')}"
        )

        summary_entries = paper.get("summary_entire", [])

        if not summary_entries:
            log.warning(f"No summary_entire found for paper {idx + 1}")
            continue

        # 각 summary_entire 항목 처리
        reconstructed_summaries = []
        for entry in summary_entries:
            # 'orginal_text' 주의! (typo in source data)
            original_text = entry.get("orginal_text", "")

            if not original_text:
                log.warning(f"Empty original text in paper {idx + 1}")
                continue

            # 텍스트 재구성
            result = reconstructor.reconstruct_text(original_text)

            reconstructed_summaries.append(
                {
                    "original_text": original_text,
                    "reconstructed_text": result,
                }
            )

        # 논문 결과 저장
        paper_result = {
            "doc_id": paper.get("doc_id"),
            "title": paper.get("title"),
            "date": paper.get("date"),
            "reconstructed_summaries": reconstructed_summaries,
        }
        results.append(paper_result)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{json_path.stem}_{timestamp}.json"
    if output_path is None:
        output_path = Path("./")

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log.info(f"Results saved to: {output_path}")


def main():
    """메인 함수"""

    # 논문 데이터 경로
    paper_data_dir = Path(r"D:\SMC\projects\reinforcement-with-llm\data\paper_data")
    paper_dir = paper_data_dir / "paper"
    paper_paths = sorted(list(paper_dir.glob("*.json")))

    # 재구성한 데이터 저장 경로
    output_path = paper_data_dir / "reconstruct"
    output_path.mkdir(exist_ok=True, parents=True)

    # 사용할 모델
    model_name = "openai/gpt-4o-mini"

    # 데이터 별로 로드하여 재구성 시작
    log.info("Starting text reconstruction with LLM")
    log.info(f"Model: {model_name}")
    for p in paper_paths:
        log.info(f"Input: {p}")

        # 처리 실행
        reconstruct_paper(
            json_path=p, output_path=output_path, max_papers=1, model_name=model_name
        )
        break

    log.info("Text reconstruction completed!")


if __name__ == "__main__":
    main()
