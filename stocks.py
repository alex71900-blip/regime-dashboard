"""현재 국면의 추천 섹터 ETF에서 상위 보유종목 후보와 기본 지표를 가져온다.

참고용 후보 목록이며 매매 신호가 아니다. 개별 데이터가 없으면 해당 칸만 비우고,
전체 실패 시 빈 목록을 돌려줘 대시보드 빌드가 깨지지 않게 한다.
"""
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from scoring import score_stock

TOP_N = 10  # 섹터당 가져올 종목 수(ETF 상위 보유종목은 최대 10개). 화면에서는 5개만 먼저 보여준다.


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

    return {
        "price": float(s.iloc[-1]),
        "m3": r(63),
        "m12": r(252),
        "vol": float(s.pct_change().iloc[-252:].std() * math.sqrt(252)),
        "ma200": float(s.iloc[-1] / s.iloc[-200:].mean() - 1),
        "from_high": float(s.iloc[-1] / s.iloc[-252:].max() - 1),
    }


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _fundamentals(sym: str):
    """재무·추정치 지표와 다음 실적 발표일. 못 가져오는 값은 None."""
    out = {k: None for k in ("fwd_pe", "earnings", "roe", "op_margin", "fcf_margin", "debt_eq", "eps_rev90")}
    t = yf.Ticker(sym)
    try:
        info = t.info or {}
        pe = _num(info.get("forwardPE"))
        out["fwd_pe"] = pe if pe and 0 < pe < 1000 else None
        out["roe"] = _num(info.get("returnOnEquity"))
        out["op_margin"] = _num(info.get("operatingMargins"))
        out["debt_eq"] = _num(info.get("debtToEquity"))
        fcf, rev = _num(info.get("freeCashflow")), _num(info.get("totalRevenue"))
        if fcf is not None and rev and rev > 0:
            out["fcf_margin"] = fcf / rev
    except Exception:
        pass
    try:
        tr = t.eps_trend
        cur, old = _num(tr.loc["0y", "current"]), _num(tr.loc["0y", "90daysAgo"])
        if cur is not None and old and old > 0:
            out["eps_rev90"] = cur / old - 1
    except Exception:
        pass
    try:
        ed = (t.calendar or {}).get("Earnings Date") or []
        today = date.today()
        future = sorted(d for d in ed if isinstance(d, (date, datetime)) and (d.date() if isinstance(d, datetime) else d) >= today)
        if future:
            d = future[0]
            out["earnings"] = (d.date() if isinstance(d, datetime) else d).isoformat()
    except Exception:
        pass
    return out


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
        with ThreadPoolExecutor(max_workers=6) as ex:
            funda = dict(zip(symbols, ex.map(_fundamentals, symbols)))
        out = []
        for etf, rows in held.items():
            stocks = []
            for sym, name, pct in rows:
                if sym not in px.columns:
                    continue
                m = _metrics(px[sym])
                if m:
                    row = {"symbol": sym, "name": name, "etf_weight": pct, **m, **funda.get(sym, {})}
                    row["score"] = score_stock(row)
                    stocks.append(row)
            if stocks:
                out.append({"etf": etf, "name": assets[etf][0], "weight": weights[etf], "stocks": stocks})
        return out
    except Exception:
        return []
