"""경기 국면을 판정하고 추천 포트폴리오 대시보드(docs/index.html)를 생성한다.

실행: python build.py
데이터: FRED(키 불필요 CSV) + Yahoo Finance(yfinance)
"""
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import backtest
from config import ASSETS, REGIMES

OUT = Path(__file__).parent / "docs"
HISTORY_MONTHS = 24


# ---------- 데이터 ----------
def fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def load():
    macro = {s: fred(s) for s in ["INDPRO", "UNRATE", "CPIAUCSL", "T5YIE", "T10Y2Y"]}
    tickers = sorted(set(ASSETS) | {"SPY", "DBC"})
    px = yf.download(tickers, period="max", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    return macro, px


# ---------- 판정 ----------
def asof(s: pd.Series, d, months_back=0):
    d = d - pd.DateOffset(months=months_back)
    s = s.loc[:d]
    return s.iloc[-1] if len(s) else float("nan")


def above_200dma(s: pd.Series, d):
    s = s.loc[:d].dropna()
    if len(s) < 200:
        return None, None
    return bool(s.iloc[-1] > s.iloc[-200:].mean()), float(s.iloc[-1] / s.iloc[-200:].mean() - 1)


def evaluate(d, macro, px, lag_days=0):
    """날짜 d 시점에 알 수 있는 데이터로 성장/물가 신호를 계산.

    lag_days: 월간 경제지표(산업생산·실업률·CPI)의 발표 지연. 백테스트에서는 45일을 써서
    아직 발표되지 않은 수치를 미리 쓰지 않도록 한다. 시장 가격·기대인플레는 지연 없음.
    """
    dm = d - pd.Timedelta(days=lag_days)
    cpi = macro["CPIAUCSL"]
    yoy_now = asof(cpi, dm) / asof(cpi, dm, 12) - 1
    yoy_3m = asof(cpi, dm, 3) / asof(cpi, dm, 15) - 1
    spy_up, spy_gap = above_200dma(px["SPY"], d)
    dbc_up, dbc_gap = above_200dma(px["DBC"], d)

    ind6 = asof(macro["INDPRO"], dm) / asof(macro["INDPRO"], dm, 6) - 1
    un6 = asof(macro["UNRATE"], dm) - asof(macro["UNRATE"], dm, 6)
    be_now, be_3m = asof(macro["T5YIE"], d), asof(macro["T5YIE"], d, 3)

    growth = [
        ("산업생산 6개월 변화", f"{ind6*100:+.2f}%", ind6 > 0),
        ("실업률 6개월 변화", f"{un6:+.1f}%p", un6 < 0.3),
        ("S&P500 vs 200일선", f"{(spy_gap or 0)*100:+.1f}%", bool(spy_up)),
    ]
    inflation = [
        ("CPI 상승률(YoY) 3개월 변화", f"{yoy_now*100:.2f}% ({(yoy_now-yoy_3m)*100:+.2f}%p)", yoy_now > yoy_3m),
        ("5년 기대인플레 3개월 변화", f"{be_now:.2f}% ({be_now-be_3m:+.2f}%p)", be_now > be_3m),
        ("원자재(DBC) vs 200일선", f"{(dbc_gap or 0)*100:+.1f}%", bool(dbc_up)),
    ]
    g_up = sum(x[2] for x in growth) >= 2
    i_up = sum(x[2] for x in inflation) >= 2
    key = {
        (True, False): "goldilocks",
        (True, True): "reflation",
        (False, True): "stagflation",
        (False, False): "slowdown",
    }[(g_up, i_up)]
    return {"key": key, "growth_up": g_up, "inflation_up": i_up, "growth": growth, "inflation": inflation}


def perf(px: pd.DataFrame):
    out = {}
    for t in ASSETS:
        s = px[t].dropna()

        def r(n):
            return float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else None

        out[t] = {"price": float(s.iloc[-1]), "m1": r(21), "m3": r(63), "m12": r(252)}
    return out


# ---------- 렌더 ----------
def pct(v):
    if v is None:
        return '<span class="mut">-</span>'
    cls = "up" if v >= 0 else "down"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span class="{cls}">{arrow} {abs(v)*100:.1f}%</span>'


CSS = """
:root{color-scheme:dark;--canvas:#0b0e11;--card:#1e2329;--elev:#2b3139;--hair:#2b3139;--body:#eaecef;--white:#fff;
--muted:#707a8a;--muted2:#929aa5;--primary:#FCD535;--primary-act:#f0b90b;--ink:#181a20;--up:#0ecb81;--down:#f6465d;--soft:#fafafa}
*{box-sizing:border-box}
body{margin:0;background:var(--canvas);color:var(--body);
font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","Apple SD Gothic Neo","Malgun Gothic",Roboto,sans-serif}
.num{font-family:"IBM Plex Sans","JetBrains Mono",ui-monospace,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
nav{height:64px;border-bottom:1px solid var(--hair);display:flex;align-items:center;padding:0 24px}
nav .t{font-weight:700;font-size:16px;color:var(--primary);letter-spacing:.2px}
nav .u{margin-left:auto;color:var(--muted);font-size:12px}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
.hero{padding:48px 0 32px}
.eyebrow{color:var(--muted);font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:.8px}
h1{margin:8px 0 8px;font-size:36px;line-height:1.1;font-weight:700;letter-spacing:-.5px;color:var(--primary)}
.desc{color:var(--body);font-size:16px;max-width:560px}
.axes{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap}
.axis{background:var(--card);border-radius:8px;padding:12px 16px;min-width:150px}
.axis small{display:block;color:var(--muted);font-size:12px;font-weight:500}
.axis b{font-size:20px;font-weight:700;color:var(--white)}
.alert{display:inline-block;margin-top:16px;padding:6px 12px;border-radius:6px;background:var(--elev);color:var(--primary);font-weight:600}
.grid{display:grid;gap:24px;grid-template-columns:1fr;padding-bottom:48px}
.card{background:var(--card);border-radius:12px;padding:24px;margin-bottom:24px}
.grid > div > .card:last-child{margin-bottom:0}
h2{margin:0 0 16px;font-size:20px;line-height:1.35;font-weight:600;color:var(--white)}
h3{margin:20px 0 4px;font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
h3:first-of-type{margin-top:0}
.cls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.cls span{background:var(--elev);border-radius:6px;padding:4px 10px;font-size:12px;color:var(--body)}
.cls b{color:var(--white);margin-left:6px}
.row{padding:10px 0;border-bottom:1px solid var(--hair)}
.row:last-child{border-bottom:0}
.line{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.tk{font-weight:600;color:var(--white)}.nm{color:var(--muted2);font-size:13px;margin-left:6px}
.w{font-weight:700;font-size:16px;color:var(--white)}
.track{height:6px;background:var(--elev);border-radius:3px;margin-top:8px}
.fill{height:6px;background:var(--primary);border-radius:3px}
table{width:100%;border-collapse:collapse}
th{color:var(--muted);font-size:12px;font-weight:500;text-align:right;padding:0 0 8px}th:first-child{text-align:left}
td{padding:10px 0;border-bottom:1px solid var(--hair);font-size:14px}tr:last-child td{border-bottom:0}
td.r{text-align:right;white-space:nowrap;font-weight:500}
.up{color:var(--up)}.down{color:var(--down)}.mut{color:var(--muted)}
.sig{font-size:12px;font-weight:600}.sig.ok{color:var(--white)}.sig.no{color:var(--muted)}
.h{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--hair)}
.h:last-child{border-bottom:0}.h .d{color:var(--muted2)}
.tag{padding:2px 10px;border-radius:9999px;background:var(--elev);font-size:12px;font-weight:500;color:var(--body)}
.tag.cur{background:var(--primary);color:var(--ink);font-weight:600}
.note{color:var(--muted);font-size:12px;margin:12px 0 0}
.sub{color:var(--muted2);font-size:12px;margin:-8px 0 12px}
.legend{display:flex;gap:16px;margin-bottom:8px;font-size:12px;color:var(--body)}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted2);font-size:13px;padding:8px 0}
details table{margin-top:4px}
footer{background:var(--soft);color:var(--ink);padding:48px 0}
footer p{margin:0;font-size:13px;line-height:1.7;max-width:760px}
footer b{display:block;margin-bottom:8px;font-size:14px}
@media(min-width:1024px){h1{font-size:64px;letter-spacing:-1px}.grid{grid-template-columns:7fr 5fr;align-items:start}.hero{padding:64px 0 40px}}
@media(max-width:480px){.card{padding:16px}nav{padding:0 16px}.axis{flex:1;min-width:0}}
"""


LINES = [("전략", "#FCD535", 2.5), ("SPY", "#eaecef", 1.5), ("60/40", "#707a8a", 1.5)]


def chart_svg(eq: pd.DataFrame) -> str:
    """로그 스케일 자산 곡선(1만 달러 시작)."""
    import math

    W, H, L, B, T, R = 640, 260, 52, 24, 10, 8
    lo = math.log(min(eq.min().min(), 10000) * 0.95)
    hi = math.log(eq.max().max() * 1.05)
    n = len(eq)
    x = lambda i: L + (W - L - R) * i / (n - 1)
    y = lambda v: T + (H - T - B) * (1 - (math.log(v) - lo) / (hi - lo))

    grid = ""
    v = 10000
    while v < math.exp(hi):
        if math.log(v) > lo:
            grid += (f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#2b3139"/>'
                     f'<text x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end" fill="#707a8a" font-size="11">${v/1000:,.0f}k</text>')
        v *= 2
    ticks = ""
    for i, d in enumerate(eq.index):
        if d.month == 12 and d.year % 4 == 0:
            ticks += f'<text x="{x(i):.1f}" y="{H-6}" text-anchor="middle" fill="#707a8a" font-size="11">{d.year}</text>'
    paths = ""
    for name, color, w in LINES:
        pts = " ".join(f"{x(i):.1f},{y(val):.1f}" for i, val in enumerate(eq[name]))
        paths += f'<polyline fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" points="{pts}"/>'
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="백테스트 자산 곡선">{grid}{ticks}{paths}</svg>'


def backtest_card(bt) -> str:
    st = bt["stats"]
    legend = "".join(
        f'<span><i style="background:{c}"></i>{n}</span>' for n, c, _ in LINES
    )
    rows = "".join(
        f'<tr><td>{lbl}</td>' + "".join(f'<td class="r num">{fmt(st[n][k])}</td>' for n, _, _ in LINES) + "</tr>"
        for lbl, k, fmt in [
            ("최종 자산(1만 달러 시작)", "final", lambda v: f"${v:,.0f}"),
            ("연평균 수익률", "cagr", lambda v: f"{v*100:.1f}%"),
            ("연 변동성", "vol", lambda v: f"{v*100:.1f}%"),
            ("샤프(무위험 0)", "sharpe", lambda v: f"{v:.2f}"),
            ("최대 낙폭", "mdd", lambda v: f"{v*100:.1f}%"),
        ]
    )
    ann = bt["annual"].sort_index(ascending=False)
    ann_rows = "".join(
        f'<tr><td class="num">{y}</td>' + "".join(f'<td class="r num">{pct(r[n])}</td>' for n, _, _ in LINES) + "</tr>"
        for y, r in ann.iterrows()
    )
    rm = bt["regime_months"]
    mix = " · ".join(f'{REGIMES[k]["name"]} {v}개월' for k, v in rm.items())
    return f"""<div class="card"><h2>백테스트</h2>
<div class="sub num">{bt["start"]} ~ {bt["end"]} · 월 1회 리밸런싱 · 거래비용 0.1% 반영</div>
<div class="legend">{legend}</div>{chart_svg(bt["equity"])}
<table style="margin-top:16px"><tr><th></th>{"".join(f"<th>{n}</th>" for n, _, _ in LINES)}</tr>{rows}</table>
<details><summary>연도별 수익률</summary><table><tr><th></th>{"".join(f"<th>{n}</th>" for n, _, _ in LINES)}</tr>{ann_rows}</table></details>
<p class="note">국면 구성: {mix} · 국면 전환 {bt["switches"]}회</p>
<p class="note">월간 경제지표는 45일 지연해 반영했지만 수정 후 확정치를 쓰고, 추천 비중은 과거 경험에 기반해 정한 값이라 실제 미래 성과를 보장하지 않습니다. 과거 성과는 미래를 보장하지 않으며, 이 기간(2007~)에는 금리 인하와 장기 상승장이 포함돼 있습니다.</p></div>"""


def render(now, cur, history, p, bt):
    reg = REGIMES[cur["key"]]
    e = html.escape
    arrow = lambda up: "▲ 상승" if up else "▼ 하락"

    # 자산군 합계
    totals = {}
    for t, w in reg["weights"].items():
        totals[ASSETS[t][1]] = totals.get(ASSETS[t][1], 0) + w
    cls = "".join(
        f'<span>{e(k)}<b class="num">{v}%</b></span>' for k, v in sorted(totals.items(), key=lambda kv: -kv[1])
    )

    weights = sorted(reg["weights"].items(), key=lambda kv: -kv[1])
    top = weights[0][1]
    bars = "".join(
        f'<div class="row"><div class="line"><span><span class="tk">{t}</span><span class="nm">{e(ASSETS[t][0])}</span></span>'
        f'<span class="w num">{w}%</span></div>'
        f'<div class="track"><div class="fill" style="width:{w/top*100:.0f}%"></div></div></div>'
        for t, w in weights
    )

    def sig_rows(rows):
        return "".join(
            f'<tr><td>{e(n)}</td><td class="r num">{e(v)}</td>'
            f'<td class="r"><span class="sig {"ok" if ok else "no"}">{"● 긍정" if ok else "○ 부정"}</span></td></tr>'
            for n, v, ok in rows
        )

    perf_rows = "".join(
        f'<tr><td><span class="tk">{t}</span><span class="nm">{e(ASSETS[t][0])}</span></td>'
        f'<td class="r num">${p[t]["price"]:,.2f}</td>'
        f'<td class="r num">{pct(p[t]["m1"])}</td><td class="r num">{pct(p[t]["m3"])}</td><td class="r num">{pct(p[t]["m12"])}</td></tr>'
        for t in ASSETS
    )
    hist = "".join(
        f'<div class="h"><span class="d num">{d}</span><span class="tag{" cur" if i == 0 else ""}">{e(REGIMES[k]["name"])}</span></div>'
        for i, (d, k) in enumerate(history)
    )
    changed = ""
    if len(history) > 1 and history[1][1] != cur["key"]:
        changed = f'<div class="alert">국면 전환 · {e(REGIMES[history[1][1]]["name"])} → {e(reg["name"])}</div>'

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0b0e11">
<title>경기 국면 · 추천 포트폴리오</title>
<style>{CSS}</style></head><body>
<nav><span class="t">REGIME PORTFOLIO</span><span class="u num">갱신 {now} UTC</span></nav>

<div class="wrap">
<section class="hero">
<div class="eyebrow">현재 경기 국면</div>
<h1>{e(reg["name"])}</h1>
<div class="desc">{e(reg["desc"])}</div>
{changed}
<div class="axes">
<div class="axis"><small>성장</small><b>{arrow(cur["growth_up"])}</b></div>
<div class="axis"><small>물가</small><b>{arrow(cur["inflation_up"])}</b></div>
</div>
</section>

<div class="grid">
<div>
<div class="card"><h2>추천 포트폴리오</h2><div class="cls">{cls}</div>{bars}</div>
{backtest_card(bt)}
<div class="card"><h2>자산별 수익률</h2>
<table><tr><th></th><th>현재가</th><th>1개월</th><th>3개월</th><th>12개월</th></tr>{perf_rows}</table></div>
</div>
<div>
<div class="card"><h2>판정 근거</h2>
<h3>성장</h3><table>{sig_rows(cur["growth"])}</table>
<h3>물가</h3><table>{sig_rows(cur["inflation"])}</table>
<p class="note">각 축에서 3개 신호 중 2개 이상이 긍정이면 상승으로 판정합니다.</p></div>
<div class="card"><h2>최근 {len(history)}개월 국면</h2>{hist}</div>
</div>
</div>
</div>

<footer><div class="wrap"><b>유의사항</b>
<p>이 페이지는 단순 규칙 기반 참고 자료이며 투자 권유나 수익 보장이 아닙니다. 경제지표는 발표 지연이 있고, 과거 국면 추이는 당시 발표 전 수치가 아닌 현재 확정 수치로 계산한 것이라 실제 대응과 다를 수 있습니다. 최종 투자 판단과 책임은 본인에게 있습니다.</p></div></footer>
</body></html>"""


def main():
    macro, px = load()
    last = px.index[-1]
    cur = evaluate(last, macro, px)

    months = pd.date_range(end=last, periods=HISTORY_MONTHS, freq="ME")
    history = []
    for d in months:
        d = px.index[px.index <= d][-1]
        history.append((d.strftime("%Y-%m"), evaluate(d, macro, px)["key"]))
    history = history[::-1]  # 최신이 앞

    df, keys, raw = backtest.run(macro, px, evaluate)
    bt = backtest.summarize(df, keys, raw)

    OUT.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    (OUT / "index.html").write_text(render(now, cur, history, perf(px), bt), encoding="utf-8")
    (OUT / "data.json").write_text(
        json.dumps({"updated": now, "regime": cur["key"], "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"현재 국면: {REGIMES[cur['key']]['name']} (성장 {'↑' if cur['growth_up'] else '↓'}, 물가 {'↑' if cur['inflation_up'] else '↓'})")
    for n, v, ok in cur["growth"] + cur["inflation"]:
        print(f"  {'+' if ok else '-'} {n}: {v}")


if __name__ == "__main__":
    main()
