"""종목 점수: 지표별 0~100점을 가중 평균한다. 값이 없는 지표는 제외하고 남은 가중치로 환산한다."""
from config import SCORE_MIN_COVERAGE, SCORE_RULES


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_stock(s: dict) -> dict:
    parts = {}
    for key, (_, _, lo, hi) in SCORE_RULES.items():
        v = s.get(key)
        if isinstance(v, (int, float)):
            parts[key] = _clip((v - lo) / (hi - lo)) * 100

    avail = sum(SCORE_RULES[k][1] for k in parts)
    coverage = avail / 100
    total = None
    if parts and coverage >= SCORE_MIN_COVERAGE:
        total = sum(SCORE_RULES[k][1] * p for k, p in parts.items()) / avail

    groups = {}
    for g in dict.fromkeys(v[0] for v in SCORE_RULES.values()):
        ks = [k for k in parts if SCORE_RULES[k][0] == g]
        w = sum(SCORE_RULES[k][1] for k in ks)
        groups[g] = sum(SCORE_RULES[k][1] * parts[k] for k in ks) / w if w else None
    return {"total": total, "coverage": coverage, "groups": groups, "parts": parts}
