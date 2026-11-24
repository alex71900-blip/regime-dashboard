"""전략 B2(D 바구니 50% + M2 바구니 50%)의 이번 주 목표 배분을 계산한다.

입력은 QQQ·TLT·GLD 수정종가(일간)이며, 데이터가 부족하면 None을 반환해 빌드를 깨뜨리지 않는다.
"""
import numpy as np
import pandas as pd

from config import (
    ALLOC_CAPITAL,
    ALLOC_D_BEAR_MAX,
    ALLOC_D_MAX,
    ALLOC_D_TARGET_VOL,
    ALLOC_M2_MAX,
    ALLOC_M2_TARGET_VOL,
)

M2_ASSETS = ["QQQ", "TLT", "GLD"]
MA_DAYS = 200
D_VOL_DAYS = 20
M2_VOL_DAYS = 60


def _vol(rets: pd.Series) -> float:
    """일간 수익률의 연율화 변동성."""
    return float(rets.std() * np.sqrt(252))


def _sleeve_d(qqq: pd.Series, sleeve: float) -> dict:
    """D 바구니: QQQ 변동성 목표 + 200일선 아래면 노출 제한."""
    rets = qqq.pct_change().dropna()
    v20 = _vol(rets.iloc[-D_VOL_DAYS:])
    ma = float(qqq.iloc[-MA_DAYS:].mean())
    last = float(qqq.iloc[-1])
    trend_up = last > ma
    exposure = ALLOC_D_MAX if v20 <= 0 else min(ALLOC_D_MAX, ALLOC_D_TARGET_VOL / v20)
    if not trend_up:
        exposure = min(exposure, ALLOC_D_BEAR_MAX)
    qqq_usd = sleeve * min(exposure, 1.0)
    qld_usd = sleeve * max(exposure - 1.0, 0.0) / 2  # QLD는 2배라 금액의 절반이면 추가 노출이 된다
    return {
        "state": "상승 추세 (200일선 위)" if trend_up else "하락 추세 (200일선 아래, 노출 제한)",
        "exposure": exposure,
        "v20": v20,
        "trend_up": trend_up,
        "vs_ma": last / ma - 1,
        "usd": {"QQQ": qqq_usd, "QLD": qld_usd, "TLT": 0.0, "GLD": 0.0},
        "sleeve": sleeve,
    }


def _sleeve_m2(px: pd.DataFrame, sleeve: float) -> dict:
    """M2 바구니: 200일선 위 자산만 역변동성 비중, 포트폴리오 변동성을 목표에 맞춘다."""
    rets = px[M2_ASSETS].pct_change().dropna().iloc[-M2_VOL_DAYS:]
    assets = {}
    for t in M2_ASSETS:
        s = px[t]
        ma = float(s.iloc[-MA_DAYS:].mean())
        assets[t] = {
            "vs_ma": float(s.iloc[-1]) / ma - 1,
            "eligible": bool(s.iloc[-1] > ma),
            "v60": _vol(rets[t]),
            "weight": 0.0,
        }
    elig = [t for t in M2_ASSETS if assets[t]["eligible"] and assets[t]["v60"] > 0]
    exposure = 0.0
    if elig:
        raw = np.array([1.0 / assets[t]["v60"] for t in elig])
        cov = rets[elig].cov().values * 252
        pvol = float(np.sqrt(raw @ cov @ raw))
        if pvol > 0:
            w = raw * (ALLOC_M2_TARGET_VOL / pvol)
            if w.sum() > ALLOC_M2_MAX:
                w = w * (ALLOC_M2_MAX / w.sum())
            for t, wi in zip(elig, w):
                assets[t]["weight"] = float(wi)
            exposure = float(w.sum())
    usd = {"QQQ": 0.0, "QLD": 0.0, "TLT": 0.0, "GLD": 0.0}
    for t in M2_ASSETS:
        usd[t] = sleeve * assets[t]["weight"]
    return {
        "state": "모든 자산이 200일선 아래 (전액 현금)" if exposure == 0 else f"{len(elig)}개 자산 투자",
        "exposure": exposure,
        "assets": assets,
        "leverage_needed": exposure > 1.0,
        "usd": usd,
        "sleeve": sleeve,
    }


def compute(px: pd.DataFrame, capital: float = ALLOC_CAPITAL):
    """이번 주 B2 목표 배분을 dict로 반환한다. 데이터가 없거나 짧으면 None."""
    try:
        if px is None or any(t not in px.columns for t in M2_ASSETS):
            return None
        px = px[M2_ASSETS].dropna()
        if len(px) < MA_DAYS:
            return None
        sleeve = capital / 2
        d = _sleeve_d(px["QQQ"], sleeve)
        m2 = _sleeve_m2(px, sleeve)
        dollars = {t: d["usd"][t] + m2["usd"][t] for t in ("QQQ", "QLD", "TLT", "GLD")}
        dollars["현금"] = capital - sum(dollars.values())  # 음수면 차입
        q = px["QQQ"]
        ma200 = float(q.iloc[-MA_DAYS:].mean())
        last = float(q.iloc[-1])
        return {
            "as_of": px.index[-1].strftime("%Y-%m-%d"),
            "capital": capital,
            "d": d,
            "m2": m2,
            "dollars": dollars,
            "weights": {t: v / capital for t, v in dollars.items()},
            "qqq_ma200": ma200,
            "qqq_last": last,
            "qqq_move_to_cross": ma200 / last - 1,
            "formula": f"D 노출 = min({ALLOC_D_MAX:g}, {ALLOC_D_TARGET_VOL:.0%} ÷ V20)",
        }
    except Exception as exc:  # 배분 계산 실패가 대시보드 빌드를 막지 않게 한다
        print(f"배분 계산 실패: {exc}")
        return None
