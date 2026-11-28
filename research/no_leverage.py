"""레버리지 ETF 없이 B2를 구현하는 변형 비교. 규칙은 결과를 보기 전에 고정했다.

공통: 주 1회(5거래일) 전일 종가 정보로 목표 비중, 비중 변경분 편도 0.05%, 현금은 단기금리(DTB3).
D 바구니: 노출 = min(상한, 30%/20일 변동성), 자산<200일선이면 최대 0.5
M2 바구니: 200일선 위 자산만 1/60일 변동성 비중, 목표 변동성 15%, 총 노출 상한
B2 = D 50% + M2 50%, 월말 재조정
- V0 기준: QQQ / QQQ·TLT·GLD, 상한 2.0 (1배 초과분은 2배 ETF 비용 = 단기금리 + 연 1%)
- V1 레버리지 없음: 같은 자산, 상한 1.0
- V2 V1 + 장기채를 EDV(초장기 국채)로
- V3 V2 + D 바구니 자산을 SMH(반도체)로
기간: EDV·SMH 공통 상장 이후(2008~)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

import build

UNIT, FEE = 0.0005, 0.01 / 252


def load():
    t = ["QQQ", "TLT", "GLD", "EDV", "SMH", "SPY"]
    px = yf.download(t, period="max", auto_adjust=True, progress=False)["Close"]
    return px.dropna(), build.fred("DTB3")


def sim(px, cash, weights_fn, start=260):
    R = px.pct_change().fillna(0.0)
    cols = list(px.columns)
    w = pd.Series(0.0, index=cols)
    out = np.zeros(len(px))
    for i in range(start, len(px)):
        cost = 0.0
        if (i - start) % 5 == 0:
            nw = weights_fn(i - 1).reindex(cols).fillna(0.0)
            cost = UNIT * float((nw - w).abs().sum())
            w = nw
        tot = float(w.sum())
        g = float((w * R.iloc[i]).sum())
        out[i] = g + ((1 - tot) * cash.iloc[i] if tot <= 1 else -(tot - 1) * (cash.iloc[i] + FEE)) - cost
    return pd.Series(out, index=px.index).iloc[start:]


def make(px, cash, d_asset, m2_assets, cap):
    R = px.pct_change().fillna(0.0)
    ma = px.rolling(200).mean()
    v20 = R.rolling(20).std() * np.sqrt(252)
    v60 = R.rolling(60).std() * np.sqrt(252)

    def dw(i):
        e = min(cap, 0.30 / v20[d_asset].iloc[i])
        if px[d_asset].iloc[i] <= ma[d_asset].iloc[i]:
            e = min(e, 0.5)
        return pd.Series({d_asset: e})

    def m2w(i):
        el = [a for a in m2_assets if px[a].iloc[i] > ma[a].iloc[i]]
        if not el:
            return pd.Series(dtype=float)
        raw = 1 / v60[el].iloc[i]
        cov = R[el].iloc[i - 59 : i + 1].cov().values * 252
        x = raw * (0.15 / float(np.sqrt(raw.values @ cov @ raw.values)))
        if x.sum() > cap:
            x *= cap / x.sum()
        return x

    d = sim(px, cash, dw)
    m = sim(px, cash, m2w)
    # 월말 50:50 재조정
    me = pd.Series(d.index.to_period("M")).ne(pd.Series(d.index.to_period("M")).shift(-1)).values
    va = vb = 0.5
    out = np.zeros(len(d))
    for i in range(len(d)):
        p0 = va + vb
        va *= 1 + d.iloc[i]; vb *= 1 + m.iloc[i]
        tot = va + vb
        if me[i]:
            tot *= 1 - 0.001 * abs(va / tot - 0.5)
            va = vb = tot / 2
        out[i] = tot / p0 - 1
    return pd.Series(out, index=d.index)


def stats(x):
    eq = (1 + x).cumprod()
    return eq.iloc[-1] ** (252 / len(x)) - 1, (eq / eq.cummax() - 1).min()


def mdd(x, a, b):
    eq = (1 + x.loc[a:b]).cumprod()
    return (eq / eq.cummax() - 1).min()


def main():
    px, rate = load()
    cash = (rate / 100 / 252).reindex(px.index).ffill().fillna(0.0)
    V = {
        "V0 기준(2배 ETF, 상한 2.0)": make(px, cash, "QQQ", ["QQQ", "TLT", "GLD"], 2.0),
        "V1 레버리지 없음(상한 1.0)": make(px, cash, "QQQ", ["QQQ", "TLT", "GLD"], 1.0),
        "V2 V1 + TLT 대신 EDV": make(px, cash, "QQQ", ["QQQ", "EDV", "GLD"], 1.0),
        "V3 V2 + D 바구니 SMH": make(px, cash, "SMH", ["QQQ", "EDV", "GLD"], 1.0),
    }
    V["QQQ 보유"] = px["QQQ"].pct_change().reindex(V["V0 기준(2배 ETF, 상한 2.0)"].index).fillna(0.0)
    V["SPY 보유"] = px["SPY"].pct_change().reindex(V["V0 기준(2배 ETF, 상한 2.0)"].index).fillna(0.0)
    s = V["SPY 보유"].index[0]
    print(f"# 레버리지 ETF 없는 B2 변형 ({s:%Y-%m} ~ {px.index[-1]:%Y-%m-%d})\n")
    print("| 전략 | 연평균 | 최대 낙폭 | 2008~2012 | 2013~ | 2008 낙폭 | 2020 낙폭 | 2022 낙폭 | 통과(2013~ ≥ 20%) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, x in V.items():
        cg, md = stats(x); a, _ = stats(x.loc[:"2012-12-31"]); b, _ = stats(x.loc["2013-01-01":])
        print(f"| {k} | {cg:.1%} | {md:.1%} | {a:.1%} | {b:.1%} | {mdd(x,'2008-01','2009-06'):.1%} | {mdd(x,'2020-01','2020-06'):.1%} | {mdd(x,'2022-01','2022-12'):.1%} | {'예' if b >= 0.2 and md > -0.5 else '아니오'} |")


if __name__ == "__main__":
    main()
