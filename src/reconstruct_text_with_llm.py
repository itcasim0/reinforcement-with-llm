"""
LLM을 사용하여 논문 요약 데이터의 abstract를 재구성하는 스크립트

데이터 구조 (변경 후):
- JSON 파일은 하나의 dict
- "metadata": 메타 정보 (total_papers, source_files, conversion_date, years 등)
- "papers": 논문 리스트
  - 각 논문: {title, author, abstract, journal, source_file, ...}
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

    def reconstruct_text(self, original_text: str) -> str:
        """
        원본 텍스트를 LLM으로 재구성

        Args:
            original_text: 재구성할 원본 텍스트

        Returns:
            재구성된 텍스트 (문자열)
        """
        # 프롬프트 구성
        prompt = """[원본 텍스트]를 [재구성 사항]을 참고하여 적절히 바꿔줘.

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

    # 기대 구조:
    # {
    #   "metadata": {...},
    #   "papers": [ {...}, {...}, ... ]
    # }
    if isinstance(data, dict) and "papers" in data:
        papers = data.get("papers", [])
        log.info(f"Loaded {len(papers)} papers")
        return papers

    log.warning("Unexpected data structure: 'papers' key not found")
    return []


def reconstruct_paper(
    json_path: Path,
    output_path: Path = None,
    max_papers: int = None,
    model_name: str = "openai/gpt-4o-mini",
):
    """
    논문 데이터를 처리하여 텍스트 재구성 (개별 저장 버전)
    """

    papers = load_paper_data(json_path)
    if not papers:
        log.error("No papers to process")
        return

    if max_papers:
        papers = papers[:max_papers]

    log.info(f"Processing {len(papers)} papers")

    reconstructor = TextReconstructorLLM(model_name=model_name)

    # output JSON 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{json_path.stem}_{timestamp}.json"

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

        paper_result = {
            "title": title,
            "author": paper.get("author"),
            "journal": paper.get("journal"),
            "abstract_original": abstract,
            "abstract_reconstructed": reconstructed,
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

def main():
    """메인 함수"""

    # 논문 데이터 경로 (기존과 동일하게 paper 디렉터리 아래에 json 이 있다고 가정)
    paper_data_dir = Path("data/paper_data")
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
            json_path=p,
            output_path=output_path,
            max_papers=50,  # 필요에 따라 조정
            model_name=model_name,
        )

    log.info("Text reconstruction completed!")


if __name__ == "__main__":
    main()
