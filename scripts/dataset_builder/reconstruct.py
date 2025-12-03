"""
원본 문서를 교정이 필요한 형태로 재구성하는 실행파일
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(1, str(root_dir / "src"))

# internal
from src.dataset.reconstructor import reconstruct_paper
from src.utils.logger_factory import log


def main():

    # 기본 data가 있는 폴더 경로 설정
    DATA_DIR = Path("./data")

    # 폴더에 따라 원본 데이터(.json 형식)가 있는 폴더 경로로 설정
    ORG_DOC_DIR = DATA_DIR / "paper_data" / "paper_abstract"
    doc_paths = sorted(list(ORG_DOC_DIR.glob("*.json")))

    # 재구성한 데이터 저장 경로
    OUTPUT_PATH = DATA_DIR / "paper_data" / "reconstruct"
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # 사용할 모델
    MODEL_NAME = "openai/gpt-4o-mini"

    # 재구성할 문서 수
    MAX_DOCS = 50

    # 데이터 별로 로드하여 재구성 시작
    log.info("Starting text reconstruction with LLM")
    log.info(f"Model: {MODEL_NAME}")

    # 처리 실행
    reconstruct_paper(
        doc_path=doc_paths,
        output_path=OUTPUT_PATH,
        model_name=MODEL_NAME,
        max_docs=MAX_DOCS,
    )

    log.info("Document reconstruction completed!")


if __name__ == "__main__":
    main()
