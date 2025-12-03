import json
import random

# internal
from src.utils.logger_factory import log

# ---------------------------
# 1) 노이즈 규칙들 정의 (로그 포함)
# ---------------------------

def corrupt_spacing(text: str):
    """띄어쓰기 일부 없애거나 이상하게 만들기"""
    original = text
    words = text.split()
    if len(words) < 2:
        return None  # 변화 없음

    if random.random() < 0.5:
        i = random.randrange(len(words) - 1)
        words[i] = words[i] + words[i + 1]
        del words[i + 1]
        new = " ".join(words)
    else:
        i = random.randrange(len(words))
        w = words[i]
        if len(w) > 2:
            pos = random.randrange(1, len(w) - 1)
            words[i] = w[:pos] + " " + w[pos:]
        new = " ".join(words)

    if new != original:
        return ("corrupt_spacing", original, new)
    return None


def corrupt_josa(text: str):
    """조사 틀리게"""
    original = text
    pairs = [
        ("을", "를"),
        ("를", "을"),
        ("은", "는"),
        ("는", "은"),
        ("이", "가"),
        ("가", "이"),
        ("에", "에서"),
        ("에서", "에"),
    ]
    candidates = [s for (s, _) in pairs if s in text]
    if not candidates:
        return None

    src = random.choice(candidates)
    tgt = [t for (s, t) in pairs if s == src][0]
    new = text.replace(src, tgt, 1)

    if new != original:
        return ("corrupt_josa", original, new)
    return None


def corrupt_typo(text: str):
    """맞춤법 노이즈"""
    original = text
    typo_map = {
        "돼": "되",
        "된": "됀",
        "고": "구",
        "있": "잇",
        "했": "햇",
        "도": "두",
        "렇게": "러케",
        "떻게": "떡케",
        "몇": "몃",
        "되": "대",
        "많": "만",
        "로": "루",
        "왜": "외",
        "조": "죠",
        "여": "요",
        "렇": "럿",
        "좋": "조",
        "곧바로": "곳바루",
        "심": "씸",
        "고 있": "구 잇",
        "겠": "겟",
        "딱": "딲",
        "좀": "쫌",
        "웬": "왠",
        "데": "대",
        "곧": "곳",
    }

    candidates = [k for k in typo_map if k in text]
    if not candidates:
        return None

    key = random.choice(candidates)
    new = text.replace(key, typo_map[key], 1)

    if new != original:
        return ("corrupt_typo", original, new)
    return None

def corrupt_punctuation(text: str):
    """문장부호 변환"""
    original = text

    if "," in text and random.random() < 0.5:
        new = text.replace(",", "", 1)
    elif text.endswith("."):
        new = text[:-1] + "..."
    else:
        return None

    if new != original:
        return ("corrupt_punctuation", original, new)
    return None


RULES = [
    corrupt_spacing,
    corrupt_josa,
    corrupt_typo,
    corrupt_punctuation,
]


# ---------------------------
# 2) 여러 룰을 섞어서 노이즈 추가 (로그 출력)
# ---------------------------


def make_noisy(text: str, n_errors: int = 4) -> str:
    noisy = text
    for _ in range(n_errors):
        rule = random.choice(RULES)
        result = rule(noisy)

        if result:
            name, before, after = result
            log.info(f"[{name}] '{before}'  →  '{after}'")
            noisy = after

    return noisy


# ---------------------------
# 3) JSON 처리
# ---------------------------


def add_noise_to_json(input_path: str, output_path: str, n_errors: int = 4):

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("results", []):
        src = item.get("abstract_reconstructed")
        log.info("\n--- NEW ABSTRACT ---")
        log.info(f"[ORIGINAL] {src}")

        item["abstract_noised"] = src
        item["abstract_noised"] = make_noisy(src, n_errors=n_errors)

        log.info(f"[FINAL NOISE] {item["abstract_noised"]}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------
# 4) 실행 예시
# ---------------------------

if __name__ == "__main__":
    add_noise_to_json(
        input_path="data/paper_data/reconstruct/paper_abstract_20251130_021815.json",
        output_path="data/paper_data/noise/paper_abstract_with_noise_20251130_021815.json",
        n_errors=random.randint(0, 50),
    )
