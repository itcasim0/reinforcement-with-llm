import json
from dataclasses import asdict

# internal
from environments.editing_env.eval.evaluator import AbstractQualityEvaluator
from dataloader.offline_loader import OfflineDocumentLoader

from utils.logger_factory import log


def _evaluate_base(text):
    """기본 평가기 - dict 반환"""
    evaluator = AbstractQualityEvaluator("ko")
    score = evaluator.evaluate_abstract(text)
    return score

def _format_score_output(score_dict, indent=2):
    return json.dumps(score_dict, ensure_ascii=False, indent=indent)

def get_socre(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    noised_list = [item["abstract_reconstructed"] for item in data["results"]]

    for i, base_text in enumerate(noised_list):
        before_base_score = _evaluate_base(base_text)
        print((before_base_score["overall_score"]))
        
def main():

    SEQUENCE_IDX = 70

    data = OfflineDocumentLoader()
    sequences = data.sequences
    base_text = sequences[SEQUENCE_IDX - 1]["base_text"]

    log.info(f"초기 텍스트:\n{base_text}")
    final_text = sequences[SEQUENCE_IDX - 1]["final_text"]

    log.info(f"최종 텍스트:\n{final_text}")

    # 기본 평가기
    log.info("\n[기본 평가기 - AbstractQualityEvaluator]")
    log.info("-" * 80)

    before_base_score = _evaluate_base(base_text)
    log.info("교정 전:")
    log.info("\n" + _format_score_output(before_base_score))

    after_base_score = _evaluate_base(final_text)
    log.info("교정 후:")
    log.info("\n" + _format_score_output(after_base_score))

if __name__ == "__main__":
    # json_path = "data/paper_data/reconstruct/paper_abstract_20251130_021815.json"
    # get_socre(json_path)

    main()
