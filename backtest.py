"""국면 전략 백테스트.

매월 말: 그 시점에 알 수 있던 신호(월간 지표는 발표 지연 반영)로 국면을 판정하고,
새 국면이 CONFIRM_MONTHS개월 연속 나오면 전환을 확정 → 확정된 국면의 추천 비중을 다음 달 동안 보유.
거래비용: 비중을 바꾼 만큼(편도) 0.1% 차감. 배당 포함 수정주가 사용.
비교: SPY 100%, SPY 60% + IEF 40% (월 리밸런싱, 비용 없음).
"""
import math

import pandas as pd

from config import CONFIRM_MONTHS, REGIMES

LAG_DAYS = 45
COST = 0.001  # 편도 거래비용 (거래한 비중 대비)


def confirm(raw, n):
    """원시 국면 열(시간순)에서, 새 국면이 n번 연속 나올 때만 전환을 확정한 열을 만든다.
    각 시점의 결과는 그 시점까지의 값만 사용한다(미래 정보 없음)."""
    out, state, cand, run = [], None, None, 0
    for k in raw:
        if state is None:
            state = k
        elif k == state:
            cand, run = None, 0
        else:
            run = run + 1 if k == cand else 1
            cand = k
            if run >= n:
                state, cand, run = k, None, 0
        out.append(state)
    return out


def run(macro, px, evaluate, confirm_months=CONFIRM_MONTHS):
    me = px.resample("ME").last()
    # 모든 ETF 가격과 200일선 계산이 가능한 시점부터 시작
    first = px["DBC"].dropna().index[0] + pd.DateOffset(months=13)
    dates = [d for d in me.index if d >= first]

    raw = []
    for d in dates[:-1]:
        d_trade = px.index[px.index <= d][-1]  # 월말 라벨이 아닌 실제 마지막 거래일
        raw.append(evaluate(d_trade, macro, px, lag_days=LAG_DAYS)["key"])
    keys = confirm(raw, confirm_months)

    strat, spy, mix = [], [], []
    prev_w = None
    for i, key in enumerate(keys):
        d, nxt = dates[i], dates[i + 1]
        w = {t: x / 100 for t, x in REGIMES[key]["weights"].items()}
        r = lambda t: me.at[nxt, t] / me.at[d, t] - 1
        gross = sum(x * r(t) for t, x in w.items())
        if prev_w is None:
            turnover = 1.0
        else:
            names = set(w) | set(prev_w)
            turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in names) / 2
        strat.append(gross - COST * turnover)
        prev_w = w
        spy.append(r("SPY"))
        mix.append(0.6 * r("SPY") + 0.4 * r("IEF"))

    idx = pd.DatetimeIndex(dates[1:])
    df = pd.DataFrame({"전략": strat, "SPY": spy, "60/40": mix}, index=idx)
    return df, pd.Series(keys, index=idx), pd.Series(raw, index=idx)


def stats(rets: pd.Series):
    eq = (1 + rets).cumprod()
    years = len(rets) / 12
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * math.sqrt(12)
    dd = (eq / eq.cummax() - 1).min()
    return {
        "final": float(eq.iloc[-1] * 10000),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(rets.mean() * 12 / vol) if vol else 0.0,  # 무위험수익률 0 가정
        "mdd": float(dd),
    }


def annual(df: pd.DataFrame):
    return ((1 + df).groupby(df.index.year).prod() - 1)


def summarize(df: pd.DataFrame, keys: pd.Series, raw: pd.Series):
    eq = (1 + df).cumprod() * 10000
    return {
        "start": df.index[0].strftime("%Y-%m"),
        "end": df.index[-1].strftime("%Y-%m"),
        "months": len(df),
        "stats": {c: stats(df[c]) for c in df.columns},
        "annual": annual(df),
        "equity": eq,
        "regime_months": {k: int((keys == k).sum()) for k in REGIMES},
        "switches": int((keys != keys.shift()).sum() - 1),
        "raw_switches": int((raw != raw.shift()).sum() - 1),
        "confirm": CONFIRM_MONTHS,
    }
