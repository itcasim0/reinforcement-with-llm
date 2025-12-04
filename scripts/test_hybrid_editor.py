"""
HybridDocumentEditor의 edit 함수를 테스트하는 스크립트

각 action을 하나씩 실행하여 결과를 확인할 수 있습니다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(1, str(root_dir / "src"))

# internal
from src.dataloader.reconstruct_loader import DomesticReconstructDataLoader
from src.environments.editing_env.components.editor import HybridDocumentEditor
from src.config.paths import DATA_DIR
from src.utils.logger_factory import log


def print_separator(title="", char="=", length=80):
    """구분선 출력"""
    if title:
        print(f"\n{char * length}")
        print(f"  {title}")
        print(f"{char * length}\n")
    else:
        print(f"{char * length}\n")


def test_single_action(editor: HybridDocumentEditor, doc_id, doc_text, action):
    """단일 action 테스트"""
    print_separator(f"Action: {action}", char="-")

    # action 실행 시간 측정
    start_time = time.time()
    edited_text, cost_info = editor.edit(
        doc_id=doc_id,
        doc=doc_text,
        actions=[action],
        use_cache=False,
        save_to_cache=False,
    )
    elapsed_time = time.time() - start_time

    # 결과 출력
    print(f"[편집된 텍스트]")
    print(edited_text)
    print()

    print(f"[비용 정보]")
    print(f"  - 사용된 비용: ${cost_info.get('used_cost', 0):.6f}")
    print(f"  - 총 토큰 수: {cost_info.get('total_tokens', 'N/A')}")
    print(f"  - 실행 시간: {elapsed_time:.2f}초")
    print()

    log.info(f"Action '{action}' 완료 - 실행 시간: {elapsed_time:.2f}초")

    return edited_text, cost_info, elapsed_time


def test_multiple_actions(editor, doc_id, doc_text, actions):
    """여러 action을 순차적으로 적용 테스트"""
    print_separator(f"Multiple Actions: {' -> '.join(actions)}", char="-")

    # actions 실행 시간 측정
    start_time = time.time()
    edited_text, cost_info = editor.edit(doc_id=doc_id, doc=doc_text, actions=actions)
    elapsed_time = time.time() - start_time

    # 결과 출력
    print(f"[편집된 텍스트]")
    print(edited_text)
    print()

    print(f"[비용 정보]")
    print(f"  - 사용된 비용: ${cost_info.get('used_cost', 0):.6f}")
    print(f"  - 총 토큰 수: {cost_info.get('total_tokens', 'N/A')}")
    print(f"  - 실행 시간: {elapsed_time:.2f}초")
    print()

    log.info(
        f"Multiple actions {' -> '.join(actions)} 완료 - 실행 시간: {elapsed_time:.2f}초"
    )

    return edited_text, cost_info, elapsed_time


def main():
    # 데이터 로더 초기화
    log.info("데이터 로더 초기화")
    # edit_document.py와 동일한 경로 사용
    input_data_path = DATA_DIR / "paper_data" / "reconstruct"
    dataloader = DomesticReconstructDataLoader(json_path=input_data_path)

    # 사용 가능한 문서 확인
    print(f"사용 가능한 문서 수: {dataloader.docs_count}")
    print(f"문서 ID 목록: {dataloader.doc_ids[:5]}...")  # 처음 5개만 출력
    print()

    # 테스트할 문서 선택 (첫 번째 문서 사용)
    test_index = 0
    doc_data = dataloader.load_by_index(test_index)

    # doc_id는 파일명에서 가져오기
    doc_id = dataloader.doc_ids[test_index]

    # 테스트할 텍스트 선택 (abstract_noised 사용)
    doc_text = doc_data.get("abstract_noised", "")

    if not doc_text:
        log.error("문서 텍스트를 로드할 수 없습니다.")
        return

    print_separator("원본 문서")
    print(f"[문서 ID]: {doc_id}")
    print(f"[원본 텍스트]")
    print(doc_text)
    print()

    # HybridDocumentEditor 초기화
    log.info("HybridDocumentEditor 초기화")
    # editor = HybridDocumentEditor(model="google/gemma-3n-e4b-it")
    editor = HybridDocumentEditor(model="google/gemma-3n-e4b-it")

    # 사용 가능한 actions
    available_actions = [
        "fix_grammar",  # 문법/맞춤법 수정
        "improve_clarity",  # 명확성 향상
        "make_concise",  # 간결화
        "improve_structure",  # 구조 개선
        "make_academic",  # 학술적 스타일
    ]

    print_separator("단일 Action 테스트")

    # 각 action을 하나씩 테스트
    results = {}
    total_time = 0
    for action in available_actions:
        edited_text, cost_info, elapsed_time = test_single_action(
            editor=editor, doc_id=doc_id, doc_text=doc_text, action=action
        )
        results[action] = {
            "text": edited_text,
            "cost_info": cost_info,
            "elapsed_time": elapsed_time,
        }
        total_time += elapsed_time

    # 여러 action을 조합한 테스트 (선택적)
    print_separator("Multiple Actions 테스트")

    # 예: 문법 수정 -> 명확성 향상 -> 학술적 스타일
    test_actions_sequence = ["fix_grammar", "improve_clarity", "make_academic"]
    edited_text, cost_info, multi_elapsed_time = test_multiple_actions(
        editor=editor, doc_id=doc_id, doc_text=doc_text, actions=test_actions_sequence
    )

    # 전체 비용 요약
    print_separator("비용 요약")
    total_cost = sum(r["cost_info"].get("used_cost", 0) for r in results.values())
    total_tokens = sum(
        r["cost_info"].get("total_tokens", 0) or 0 for r in results.values()
    )

    print(f"단일 액션 테스트:")
    print(f"  - 총 비용: ${total_cost:.6f}")
    print(f"  - 총 토큰: {total_tokens}")
    print(f"  - 총 실행 시간: {total_time:.2f}초")
    print()
    print(f"다중 액션 테스트:")
    print(f"  - 비용: ${cost_info.get('used_cost', 0):.6f}")
    print(f"  - 토큰: {cost_info.get('total_tokens', 'N/A')}")
    print(f"  - 실행 시간: {multi_elapsed_time:.2f}초")
    print()

    log.info("테스트 완료")


if __name__ == "__main__":
    main()
