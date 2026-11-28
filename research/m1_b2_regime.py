"""M1과 B2를 경기 국면에 따라 전환하는 전략 백테스트. 규칙은 결과를 보기 전에 고정했다.

일간 전략(주 1회=5거래일마다 전일 종가 정보로 목표 비중 결정, 비중 변경분 편도 0.05%, 1배 초과 노출은
2배 ETF 비용 모델 = 초과분에 (단기금리 + 연 1%) 부담, 현금은 단기금리 수익):
- D : QQQ 노출 = min(2, 30%/20일 변동성), QQQ<200일선이면 최대 0.5
- M1: QQQ·TLT·GLD 위험균형(1/60일 변동성), 포트폴리오 목표 변동성 15%, 총 노출 상한 2.0(2배 ETF로 구현 가능한 최대)
- M2: M1 + 자기 200일선 아래 자산 제외
- B2: D 50% + M2 50%, 월말 50:50 재조정
전환 전략(월말 판정, 경제지표 발표 지연 45일, 다음 달 적용, 전환 시 왕복 0.1%):
- S1 국면: 성장 상승(골디락스·리플레이션) -> B2, 성장 하락(스태그플레이션·둔화) -> M1
- S2 추세: SPY>200일선 -> B2, 아래 -> M1
- S3 고정: B2 50% + M1 50%, 월말 재조정
기간: 국면 판정이 가능한 2007-04 이후 (그 이전은 전략 단독 성과만 참고로 표시)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import backtest
import build

UNIT = 0.0005
FEE = 0.01 / 252


def load_all():
    macro, px = build.load()
    rate = build.fred("DTB3")
    return macro, px, rate


def strategies(px, rate):
    p = px[["QQQ", "TLT", "GLD", "SPY"]].dropna()
    idx = p.index
    R = p.pct_change().fillna(0.0)
    cash = (rate / 100 / 252).reindex(idx).ffill().fillna(0.0)
    ma200 = p.rolling(200).mean()
    v20 = R["QQQ"].rolling(20).std() * np.sqrt(252)
    A = ["QQQ", "TLT", "GLD"]
    v60 = R[A].rolling(60).std() * np.sqrt(252)

    def rp(i, trend):
        elig = [a for a in A if not trend or p[a].iloc[i] > ma200[a].iloc[i]]
        w = pd.Series(0.0, index=A)
        if not elig:
            return w
        raw = 1 / v60[elig].iloc[i]
        cov = R[elig].iloc[i - 59 : i + 1].cov().values * 252
        pv = float(np.sqrt(raw.values @ cov @ raw.values))
        x = raw * (0.15 / pv)
        if x.sum() > 2.0:
            x = x * (2.0 / x.sum())
        w[elig] = x
        return w

    def dw(i):
        e = min(2.0, 0.30 / v20.iloc[i])
        if p["QQQ"].iloc[i] <= ma200["QQQ"].iloc[i]:
            e = min(e, 0.5)
        return pd.Series({"QQQ": e, "TLT": 0.0, "GLD": 0.0})

    start = 260
    out = {k: np.zeros(len(idx)) for k in ("D", "M1", "M2")}
    for k, fn in (("D", dw), ("M1", lambda i: rp(i, False)), ("M2", lambda i: rp(i, True))):
        w = pd.Series(0.0, index=A)
        for i in range(start, len(idx)):
            if (i - start) % 5 == 0:
                nw = fn(i - 1)
                cost = UNIT * float((nw - w).abs().sum())
                w = nw
            else:
                cost = 0.0
            tot = float(w.sum())
            r = float((w * R[A].iloc[i]).sum()) + (1 - tot) * cash.iloc[i] if tot <= 1 else \
                float((w * R[A].iloc[i]).sum()) - (tot - 1) * (cash.iloc[i] + FEE)
            out[k][i] = r - cost
    df = pd.DataFrame(out, index=idx).iloc[start:]
    return df, p


def blend_monthly(a, b, wa_by_month, idx, switch_cost=0.001):
    """월별 a 비중(wa_by_month: 월말 날짜 -> 0~1)에 따라 섞는다. 월말에 재조정."""
    me = pd.Series(idx.to_period("M")).ne(pd.Series(idx.to_period("M")).shift(-1)).values
    out = np.zeros(len(idx))
    wa_cur = None
    va = vb = 0.0
    months = pd.Series(wa_by_month)
    for i, d in enumerate(idx):
        if wa_cur is None:
            prev = months[months.index < d]
            if len(prev) == 0:
                continue
            wa_cur = float(prev.iloc[-1]); va, vb = wa_cur, 1 - wa_cur
        pa = va + vb
        va *= 1 + a.iloc[i]; vb *= 1 + b.iloc[i]
        tot = va + vb
        r = tot / pa - 1
        if me[i] and d in months.index:
            nw = float(months[d])
            cost = switch_cost * abs(va / tot - nw)
            tot *= 1 - cost
            r = tot / pa - 1
            va, vb = tot * nw, tot * (1 - nw)
        out[i] = r
    return pd.Series(out, index=idx)


def stats(x):
    eq = (1 + x).cumprod()
    yrs = len(x) / 252
    return eq.iloc[-1] ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min()


def mdd(x, a, b):
    s = x.loc[a:b]
    eq = (1 + s).cumprod()
    return (eq / eq.cummax() - 1).min()


def roll1y(x):
    eq = (1 + x).cumprod().values
    v = np.array([eq[i + 252] / eq[i] for i in range(0, len(eq) - 252, 5)]) * 10000
    return np.median(v), np.quantile(v, 0.1), (v >= 12000).mean(), (v < 10000).mean()


def main():
    macro, px, rate = load_all()
    df, p = strategies(px, rate)
    idx = df.index
    # B2 = D 50% + M2 50%, 월말 재조정
    me_dates = [d for d in idx[pd.Series(idx.to_period("M")).ne(pd.Series(idx.to_period("M")).shift(-1)).values]]
    half = {d: 0.5 for d in me_dates}
    df["B2"] = blend_monthly(df["D"], df["M2"], half, idx, switch_cost=0.001)

    # 월말 국면·추세 판정
    first = px["DBC"].dropna().index[0] + pd.DateOffset(months=13)
    reg, trd = {}, {}
    for d in me_dates:
        if d < first:
            continue
        reg[d] = build.evaluate(d, macro, px, lag_days=backtest.LAG_DAYS)["key"]
        up, _ = build.above_200dma(px["SPY"], d)
        trd[d] = bool(up)
    s1 = {d: 1.0 if k in ("goldilocks", "reflation") else 0.0 for d, k in reg.items()}
    s2 = {d: 1.0 if u else 0.0 for d, u in trd.items()}
    s3 = {d: 0.5 for d in reg}
    df["S1 국면 전환"] = blend_monthly(df["B2"], df["M1"], s1, idx)
    df["S2 추세 전환"] = blend_monthly(df["B2"], df["M1"], s2, idx)
    df["S3 50:50 고정"] = blend_monthly(df["B2"], df["M1"], s3, idx)
    df["SPY 보유"] = p["SPY"].pct_change().reindex(idx).fillna(0.0)

    start = min(reg)
    print(f"# M1·B2 국면 전환 ({start:%Y-%m} ~ {idx[-1]:%Y-%m-%d})\n")
    print("데이터: Yahoo 수정종가(QQQ·TLT·GLD·SPY), FRED 단기금리(DTB3). 국면 판정이 가능한 2007년 이후만 비교합니다(2000~2002 제외).\n")
    print("| 전략 | 연평균 | 최대 낙폭 | 2007~2012 | 2013~ | 2008 낙폭 | 2020 낙폭 | 2022 낙폭 | 1년 중앙 | 1년 하위10% | 1년 +20% 이상 | 1년 손실 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ["S1 국면 전환", "S2 추세 전환", "S3 50:50 고정", "B2", "M1", "M2", "D", "SPY 보유"]:
        x = df[c].loc[start:]
        cg, md = stats(x)
        a, _ = stats(x.loc[:"2012-12-31"]); b, _ = stats(x.loc["2013-01-01":])
        med, p10, p20, loss = roll1y(x)
        print(f"| {c} | {cg:.1%} | {md:.1%} | {a:.1%} | {b:.1%} | {mdd(x,'2007-10','2009-06'):.1%} | {mdd(x,'2020-01','2020-06'):.1%} | "
              f"{mdd(x,'2022-01','2022-12'):.1%} | ${med:,.0f} | ${p10:,.0f} | {p20:.0%} | {loss:.0%} |")
    print("\n## 국면별 다음 달 평균 수익 (B2 vs M1)\n")
    print("| 국면 | 개월 | B2 | M1 | B2-M1 |\n|---|---|---|---|---|")
    mr = (1 + df[["B2", "M1"]]).resample("ME").prod() - 1
    ks = pd.Series(reg)
    for k in ("goldilocks", "reflation", "stagflation", "slowdown"):
        ds = [d for d in ks.index[ks == k]]
        nxt = [mr.index[mr.index > d][0] for d in ds if (mr.index > d).any()]
        if not nxt:
            continue
        b2, m1 = mr.loc[nxt, "B2"].mean(), mr.loc[nxt, "M1"].mean()
        print(f"| {k} | {len(nxt)} | {b2:+.2%} | {m1:+.2%} | {b2-m1:+.2%} |")
    print("\n## 참고: 단독 전략 1999~ 전체 (국면 없이)\n")
    for c in ["B2", "M1", "D", "M2", "SPY 보유"]:
        cg, md = stats(df[c]); a, _ = stats(df[c].loc[:"2012-12-31"]); b, _ = stats(df[c].loc["2013-01-01":])
        print(f"- {c}: {df.index[0]:%Y-%m}~ 연평균 {cg:.1%} / 최대 낙폭 {md:.1%} (~2012 {a:.1%}, 2013~ {b:.1%})")
    last = max(reg)
    print(f"\n현재: 마지막 월말({last:%Y-%m-%d}) 국면 {reg[last]} → S1은 {'B2' if s1[last] else 'M1'}, SPY 200일선 {'위' if trd[last] else '아래'} → S2는 {'B2' if s2[last] else 'M1'}")


if __name__ == "__main__":
    main()
