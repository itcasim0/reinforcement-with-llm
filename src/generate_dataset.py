"""
첫 번째 재구성 논문 초록에 대해
4가지 액션 조합(길이 1~3, 총 84개)에 대한
모든 LLM 편집 결과를 JSONL로 저장하는 스크립트.

특징:
- 시퀀스 ['A'] 를 먼저 만들고, ['A', 'B'] 는 ['A']의 결과를 이어 받아 한 번만 더 LLM 호출.
- 이미 만든 시퀀스(actions 튜플 기준)는 다시 LLM 호출하지 않고 재사용.
- output JSONL가 이미 있으면 읽어와서, 없는 시퀀스만 이어서 생성(중간 재시작 가능).
"""

import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

from utils.logger_factory import log

from config.paths import DATA_DIR
from dataloader.reconstruct_loader import DomesticReconstructDataLoader
from environments.editing_env.components.component import Document
from environments.editing_env.components.editor import DocumentEditor

# 사용할 액션 4개 (순서 그대로 product에 사용)
ACTIONS = [
    "fix_grammar",
    "improve_clarity",
    "make_concise",
    "improve_structure",
    "make_academic"
]

def generate_action_sequences(max_len: int = 3):
    """
    길이 1 ~ max_len 까지의 모든 액션 시퀀스를 생성.
    - ACTIONS = 5개, max_len = 3 이면 5^1 + 5^2 + 5^3 = 155개
    """
    for length in range(1, max_len + 1):
        for seq in itertools.product(ACTIONS, repeat=length):
            yield seq  # 튜플(str, str, ...)


def load_existing_sequences(
    output_path: Path,
) -> Tuple[Dict[Tuple[str, ...], Dict[str, Any]], int]:
    """
    이미 존재하는 JSONL 파일이 있다면 모두 읽어서
    - key: tuple(actions)  예) ('fix_grammar', 'make_concise')
    - value: record(dict)  (sequence_id, actions, base_text, final_text, steps, total_cost_usd)
    형태의 캐시로 만들어준다.

    또한, 가장 큰 sequence_id를 찾아서 반환한다. (이어붙이기용)
    """
    seq_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    max_seq_id = 0

    if not output_path.exists():
        return seq_cache, 0

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.info("[WARN] JSON decode error on existing line, skip.")
                continue

            actions = rec.get("actions")
            if not actions:
                continue
            key = tuple(actions)
            seq_cache[key] = rec

            sid = rec.get("sequence_id", 0)
            if isinstance(sid, int) and sid > max_seq_id:
                max_seq_id = sid

    log.info(f"[INFO] Loaded {len(seq_cache)} existing sequences from {output_path}")
    return seq_cache, max_seq_id


def build_all_sequences_for_first_doc(output_path: Path):
    """
    DomesticReconstructDataLoader에서 첫 번째 문서를 가져와서,
    모든 액션 시퀀스(길이 1~3, 총 155개)에 대해 편집을 수행하고 JSONL로 저장.

    - 각 시퀀스는 base_text에서 시작해서 시퀀스의 액션을 순서대로 적용한 결과.
    - 이미 계산된 prefix 시퀀스는 다시 LLM 호출하지 않고 재사용해서,
      새 액션(마지막 액션)만 추가로 호출.
    - output_path가 이미 존재하면 그 안의 시퀀스를 읽어와서
      아직 없는 시퀀스만 추가로 생성 (중간 재시작 가능).
    """
    # 1) 데이터 로더에서 첫 번째 문서 하나만 가져오기
    loader = DomesticReconstructDataLoader()
    docs: List[Document] = loader.get_reconstructed_text(max_docs=1)

    if not docs:
        log.info("[ERROR] No documents loaded from DomesticReconstructDataLoader.")
        return

    base_doc: Document = docs[0]
    base_text: str = base_doc.text

    log.info("[INFO] Base document loaded.")
    log.info(base_text[:200] + ("..." if len(base_text) > 200 else ""))

    # 2) 에디터 초기화
    editor = DocumentEditor(
        model="qwen/qwen3-8b",
        base_cost=0.02,
        price_per_1k_tokens=0.000028,
    )

    # 3) 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4) 기존 JSONL 파일 읽어서 seq_cache 구성 (있다면)
    seq_cache, max_seq_id = load_existing_sequences(output_path)
    next_sequence_id = max_seq_id + 1

    # 5) 새로 생성되는 시퀀스는 파일에 append 모드로 바로바로 기록
    with output_path.open("a", encoding="utf-8") as wf:

        def get_or_build_sequence(actions: Tuple[str, ...]) -> Dict[str, Any]:
            """
            주어진 액션 튜플에 대한 시퀀스 record를 반환.
            - 이미 seq_cache에 있다면 그대로 반환 (LLM 호출 없음)
            - 없으면 prefix를 먼저(재귀적으로) 만들고, 마지막 액션만 LLM으로 호출해서 새 record 생성
            - 새 record는 JSONL에도 한 줄 append
            """
            nonlocal next_sequence_id

            # 이미 만들어진 시퀀스면 그대로 사용
            if actions in seq_cache:
                return seq_cache[actions]

            # prefix 시퀀스를 먼저 확보
            if len(actions) == 1:
                # 길이 1이면 prefix 없음 → base_text에서 바로 시작
                prefix_record = None
                input_text = base_text
                steps_before: List[Dict[str, Any]] = []
                total_cost_before = 0.0
            else:
                prefix = actions[:-1]
                prefix_record = get_or_build_sequence(prefix)  # 재귀적으로 prefix 생성/획득
                input_text = prefix_record["final_text"]
                steps_before = prefix_record["steps"]
                total_cost_before = float(prefix_record["total_cost_usd"])

            last_action = actions[-1]

            log.info(f"[INFO] seq {next_sequence_id} (new) / actions = {list(actions)}")

            # LLM 한 번 호출: (input_text, last_action)
            try:
                edited_text, cost_info = editor.edit(input_text, last_action)
            except Exception as e:
                log.info(
                    f"[WARN] edit failed at seq={next_sequence_id}, "
                    f"actions={actions}, last_action={last_action}: {e}"
                )
                # 실패하면 이 시퀀스는 생성하지 않고 예외 전파/무시 선택 가능
                # 여기서는 None 반환 대신 예외를 그대로 올리거나,
                # 필요하다면 return prefix_record 등으로 우회 가능
                raise

            used_cost = float(cost_info.get("used_cost") or 0.0)

            # prefix의 step 뒤에 새 step 하나 추가
            new_step_idx = len(steps_before) + 1
            new_step = {
                "step": new_step_idx,
                "action": last_action,
                "input_text": input_text,
                "output_text": edited_text,
                "cost_info": cost_info,
            }

            steps = steps_before + [new_step]
            total_cost = total_cost_before + used_cost

            record: Dict[str, Any] = {
                "sequence_id": next_sequence_id,
                "actions": list(actions),
                "base_text": base_text,
                "final_text": edited_text,
                "steps": steps,
                "total_cost_usd": total_cost,
            }

            # 캐시에 저장
            seq_cache[actions] = record

            # JSONL에 한 줄 기록
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()

            next_sequence_id += 1
            return record

        # 6) 길이 1~3 모든 시퀀스에 대해 get_or_build_sequence 호출
        for actions in generate_action_sequences(max_len=3):
            actions = tuple(actions)
            # 이미 파일에 있는 시퀀스면 이 호출에서 아무것도 안 하고 넘어감
            try:
                get_or_build_sequence(actions)
            except Exception as e:
                log.info(f"[ERROR] Failed to build sequence {actions}: {e}")
                # 필요하면 여기서 continue로 다음 시퀀스 진행
                continue

    log.info(f"[DONE] saved all sequences dataset to {output_path}")


if __name__ == "__main__":
    output_file = DATA_DIR / "editing" / f"sequences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl"
    build_all_sequences_for_first_doc(output_file)
