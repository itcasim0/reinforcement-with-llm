import json
from dataclasses import asdict

# internal
from environments.editing_env.env import StrictEvaluator

from environments.editing_env.eval.evaluator import AbstractQualityEvaluator
from environments.editing_env.components.data import SingleDocOfflineData

from utils.logger_factory import log


def _evaluate_base(text):
    """기본 평가기 - dict 반환"""
    evaluator = AbstractQualityEvaluator("ko")
    score = evaluator.evaluate_abstract(text)
    return score


def _evaluate_strict(text):
    """엄격한 평가기 - dict 반환"""
    evaluator = StrictEvaluator()
    score = evaluator.evaluate(text)
    # DocumentScore 객체를 dict으로 변환 (편하게 보기 위함)
    if hasattr(score, "__dataclass_fields__"):
        return asdict(score)
    return score


def _format_score_output(score_dict, indent=2):
    return json.dumps(score_dict, ensure_ascii=False, indent=indent)


def main():

    SEQUENCE_IDX = 70

    data = SingleDocOfflineData()
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

    # 엄격 평가기
    log.info("[엄격 평가기 - StrictEvaluator]")
    before_strict_score = _evaluate_strict(base_text)
    log.info("교정 전:")
    log.info("\n" + _format_score_output(before_strict_score))

    after_strict_score = _evaluate_strict(final_text)
    log.info("교정 후:")
    log.info("\n" + _format_score_output(after_strict_score))


if __name__ == "__main__":
    main()
