"""현재 국면의 추천 섹터 ETF에서 상위 보유종목 후보와 기본 지표를 가져온다.

참고용 후보 목록이며 매매 신호가 아니다. 실패하면 빈 목록을 돌려줘 대시보드 빌드가 깨지지 않게 한다.
"""
import math

import pandas as pd
import yfinance as yf

TOP_N = 5


def _top_holdings(etf: str):
    h = yf.Ticker(etf).funds_data.top_holdings
    if h is None or len(h) == 0:
        return []
    return [(sym, str(row["Name"]), float(row["Holding Percent"])) for sym, row in h.head(TOP_N).iterrows()]


def _metrics(s: pd.Series):
    s = s.dropna()
    if len(s) < 253:
        return None

    def r(n):
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1)

    ma200 = s.iloc[-200:].mean()
    return {
        "price": float(s.iloc[-1]),
        "m3": r(63),
        "m12": r(252),
        "vol": float(s.pct_change().iloc[-252:].std() * math.sqrt(252)),
        "ma200": float(s.iloc[-1] / ma200 - 1),
        "from_high": float(s.iloc[-1] / s.iloc[-252:].max() - 1),
    }


def candidates(weights: dict, assets: dict):
    """weights: {ETF: 비중}. 섹터 ETF(XL*)만 대상. 반환: [{etf, name, weight, stocks:[...]}]"""
    try:
        sectors = [t for t, _ in sorted(weights.items(), key=lambda kv: -kv[1]) if t.startswith("XL")]
        held = {}
        for etf in sectors:
            try:
                rows = _top_holdings(etf)
            except Exception:
                rows = []
            if rows:
                held[etf] = rows
        symbols = sorted({sym for rows in held.values() for sym, _, _ in rows})
        if not symbols:
            return []
        px = yf.download(symbols, period="2y", auto_adjust=True, progress=False)["Close"]
        out = []
        for etf, rows in held.items():
            stocks = []
            for sym, name, pct in rows:
                if sym not in px.columns:
                    continue
                m = _metrics(px[sym])
                if m:
                    stocks.append({"symbol": sym, "name": name, "etf_weight": pct, **m})
            if stocks:
                out.append({"etf": etf, "name": assets[etf][0], "weight": weights[etf], "stocks": stocks})
        return out
    except Exception:
        return []
