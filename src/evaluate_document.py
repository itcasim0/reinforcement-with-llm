from environments.editing_env.env import StrictEvaluator
from environments.editing_env.eval.evaluator import AbstractQualityEvaluator
from environments.editing_env.components.data import SingleDocOfflineData

from utils.logger_factory import log


def _evaluate_base(text):

    # 기본 평가기
    evaluator = AbstractQualityEvaluator("ko")
    score = evaluator.evaluate_abstract(text)
    return score


def _evaluate_strict(text):

    # 엄격한 평가기
    evaluator = StrictEvaluator()
    score = evaluator.evaluate(text)
    return score


def main():
    data = SingleDocOfflineData()
    sequences = data.sequences

    base_text = sequences[-1]["base_text"]
    final_text = sequences[-1]["final_text"]

    # 기존
    before_base_score = _evaluate_base(base_text)
    log.info(f"기존 평가기 [교정 전]: \n{before_base_score}")

    after_base_score = _evaluate_base(final_text)
    log.info(f"기존 평가기 [교정 후]: \n{after_base_score}")

    # 엄격
    before_strict_score = _evaluate_strict(base_text)
    log.info(f"엄격 평가기 [교정 전]: \n{before_strict_score}")

    after_strict_score = _evaluate_strict(final_text)
    log.info(f"엄격 평가기 [교정 후]: \n{after_strict_score}")


if __name__ == "__main__":
    main()
