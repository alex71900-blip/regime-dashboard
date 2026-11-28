"""S&P500 + 미국채 2자산 국면 전략 백테스트와 '수익 50% 출금' 변형.

규칙은 결과를 보기 전에 고정했다.
- 국면별 SPY 비중: 골디락스 90, 리플레이션 70, 스태그플레이션 50, 둔화 30 (나머지 국채)
- R1 국채=IEF, R2 TLT, R3 SHY / R4 국면 무시, SPY>200일선이면 90/10 아니면 30/70 (IEF)
- 비교: SPY 100%, 60/40(SPY/IEF)
- 월말 판정(경제지표 발표 지연 45일), 다음 달 보유, 비중 변경분 편도 0.1% 비용
출금 변형(1만 달러 시작):
- P: 리밸런싱 주기(월/분기/연) 동안 늘어난 금액이 양수면 그 50%를 출금 (손실은 보전 안 함)
- H: 고점 기준. 직전 출금 후 잔고(초기 1만)보다 늘어난 부분의 50%만 출금
실행: python research/two_asset.py  (결과는 표준출력, Actions가 two_asset_result.md로 저장)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import backtest
import build

SPY_W = {"goldilocks": 0.9, "reflation": 0.7, "stagflation": 0.5, "slowdown": 0.3}
COST = 0.001


def monthly_returns(macro, px):
    """월별 전략 수익률 표와 국면 열."""
    me = px.resample("ME").last()
    first = px["DBC"].dropna().index[0] + pd.DateOffset(months=13)
    dates = [d for d in me.index if d >= first]
    keys, trend = [], []
    for d in dates[:-1]:
        dt = px.index[px.index <= d][-1]
        keys.append(build.evaluate(dt, macro, px, lag_days=backtest.LAG_DAYS)["key"])
        up, _ = build.above_200dma(px["SPY"], dt)
        trend.append(bool(up))
    r = lambda t, i: me.at[dates[i + 1], t] / me.at[dates[i], t] - 1
    rows, prev = [], {}
    for i, k in enumerate(keys):
        w = {
            "R1 국면 SPY/IEF": (SPY_W[k], "IEF"),
            "R2 국면 SPY/TLT": (SPY_W[k], "TLT"),
            "R3 국면 SPY/SHY": (SPY_W[k], "SHY"),
            "R4 추세 SPY/IEF": (0.9 if trend[i] else 0.3, "IEF"),
            "60/40 고정": (0.6, "IEF"),
            "SPY 100%": (1.0, "IEF"),
        }
        row = {}
        for name, (ws, bond) in w.items():
            g = ws * r("SPY", i) + (1 - ws) * r(bond, i)
            turn = abs(ws - prev.get(name, ws)) if name in prev else 1.0
            if name in ("60/40 고정", "SPY 100%"):
                turn = 0.0 if name in prev else 1.0
            row[name] = g - COST * turn
            prev[name] = ws
        rows.append(row)
    idx = pd.DatetimeIndex(dates[1:])
    df = pd.DataFrame(rows, index=idx)
    nxt = {t: [r(t, i) for i in range(len(keys))] for t in ("SPY", "IEF", "TLT")}
    return df, pd.Series(keys, index=idx), pd.DataFrame(nxt, index=idx)


def stats(x: pd.Series):
    eq = (1 + x).cumprod()
    yrs = len(x) / 12
    return eq.iloc[-1] ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min(), x.std() * np.sqrt(12)


def mdd_in(x, a, b):
    s = x.loc[a:b]
    if len(s) == 0:
        return float("nan")
    eq = (1 + s).cumprod()
    return (eq / eq.cummax() - 1).min()


def withdraw(x: pd.Series, every: int, mode: str, start=10000.0, share=0.5):
    """출금 시뮬레이션. every: 몇 개월마다 리밸런싱/출금할지."""
    v, base, out, flows = start, start, 0.0, []
    for i, ri in enumerate(x.values, 1):
        v *= 1 + ri
        if i % every == 0:
            gain = v - base
            w = share * gain if gain > 0 else 0.0
            v -= w
            out += w
            flows.append(w)
            if mode == "P":
                base = v                    # 다음 기간의 기준 = 출금 후 잔고 (손실도 새 기준)
            else:
                base = max(base, v) if w == 0 else v  # 고점 기준: 손실 구간에서는 기준 유지
    yrs = len(x) / 12
    return {"잔고": v, "누적출금": out, "합계": v + out, "연평균출금": out / yrs, "출금횟수": sum(f > 0 for f in flows)}


def roll1y(x: pd.Series):
    eq = (1 + x).cumprod().values
    v = np.array([eq[i + 12] / eq[i] for i in range(len(eq) - 12)]) * 10000
    return np.median(v), np.quantile(v, 0.1), (v < 10000).mean()


def main():
    macro, px = build.load()
    df, keys, nxt = monthly_returns(macro, px)
    p = print
    p(f"# S&P500 + 국채 국면 전략 ({df.index[0]:%Y-%m} ~ {df.index[-1]:%Y-%m}, {len(df)}개월)\n")
    p("국면 판정에 원자재 ETF(DBC, 2006~)와 5년 기대인플레(2003~)가 필요해 2007년부터입니다. 2000~2002는 포함되지 않습니다.\n")
    p("## 1. 수익률 (출금 없음)\n")
    p("| 전략 | 전체 연평균 | 최대 낙폭 | 변동성 | 2007~2012 | 2013~ | 2008 낙폭 | 2020 낙폭 | 2022 낙폭 | 1년 중앙 | 1년 하위10% | 1년 손실 확률 |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in df.columns:
        x = df[c]
        cg, md, vol = stats(x)
        a, _, _ = stats(x.loc[:"2012-12-31"])
        b, _, _ = stats(x.loc["2013-01-01":])
        med, p10, loss = roll1y(x)
        p(f"| {c} | {cg:.1%} | {md:.1%} | {vol:.1%} | {a:.1%} | {b:.1%} | {mdd_in(x,'2007-10','2009-06'):.1%} | "
          f"{mdd_in(x,'2020-01','2020-06'):.1%} | {mdd_in(x,'2022-01','2022-12'):.1%} | ${med:,.0f} | ${p10:,.0f} | {loss:.0%} |")
    p("\n## 2. 국면별 체류 비율과 다음 달 평균 수익\n")
    p("| 국면 | 비율 | SPY | IEF | TLT |\n|---|---|---|---|---|")
    for k in SPY_W:
        m = keys == k
        p(f"| {k} | {m.mean():.0%} | {nxt['SPY'][m].mean():+.2%} | {nxt['IEF'][m].mean():+.2%} | {nxt['TLT'][m].mean():+.2%} |")
    p(f"| 전체 | 100% | {nxt['SPY'].mean():+.2%} | {nxt['IEF'].mean():+.2%} | {nxt['TLT'].mean():+.2%} |")
    p("\n## 3. 수익 50% 출금 (1만 달러 시작)\n")
    p("P = 기간 동안 늘어난 금액의 50% 출금(손실은 보전 안 함), H = 고점 기준(이전 출금 후 잔고를 넘은 부분만 50% 출금)\n")
    p("| 전략 | 방식 | 주기 | 최종 잔고 | 누적 출금 | 합계 | 연평균 출금 | 출금 없을 때 최종 |")
    p("|---|---|---|---|---|---|---|---|")
    for c in df.columns:
        x = df[c]
        full = 10000 * (1 + x).prod()
        for mode in ("P", "H"):
            for every, lab in ((1, "월"), (3, "분기"), (12, "연")):
                w = withdraw(x, every, mode)
                p(f"| {c} | {mode} | {lab} | ${w['잔고']:,.0f} | ${w['누적출금']:,.0f} | ${w['합계']:,.0f} | ${w['연평균출금']:,.0f} | ${full:,.0f} |")
    last = keys.index[-1]
    p(f"\n## 4. 현재\n\n마지막 판정 국면: {keys.iloc[-1]} (판정 월 {last:%Y-%m}) → R1 비중 SPY {SPY_W[keys.iloc[-1]]:.0%} / IEF {1-SPY_W[keys.iloc[-1]]:.0%}")
    dt = px.index[-1]
    cur = build.evaluate(dt, macro, px)["key"]
    p(f"\n오늘({dt:%Y-%m-%d}) 데이터 기준 국면: {cur} → R1 비중 SPY {SPY_W[cur]:.0%} / IEF {1-SPY_W[cur]:.0%}")


if __name__ == "__main__":
    main()
