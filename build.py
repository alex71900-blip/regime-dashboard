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
import stocks
from config import ASSETS, REGIMES, SCORE_LABELS, SCORE_RULES, SCORE_THRESHOLD

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
--muted:#707a8a;--muted2:#929aa5;--primary:#FCD535;--ink:#181a20;--up:#0ecb81;--down:#f6465d;--soft:#fafafa}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--canvas);color:var(--body);
font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","Apple SD Gothic Neo","Malgun Gothic",Roboto,sans-serif}
.num{font-family:"IBM Plex Sans","JetBrains Mono",ui-monospace,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
nav{height:56px;display:flex;align-items:center;gap:12px;padding:0 16px}
nav .t{font-weight:700;font-size:15px;color:var(--primary);letter-spacing:.2px}
nav .reg{margin-left:auto}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
.tabs{position:sticky;top:0;z-index:5;background:var(--canvas);border-bottom:1px solid var(--hair);display:flex;justify-content:center;gap:4px}
.tab{background:none;border:0;border-bottom:2px solid transparent;color:var(--muted2);font:inherit;font-weight:600;padding:14px 22px;cursor:pointer}
.tab.on{color:var(--white);border-bottom-color:var(--primary)}
.js .pane{display:none}.js .pane.on{display:block}
.pane{padding-bottom:48px}
.hero{padding:32px 0 24px}
.eyebrow{color:var(--muted);font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:.8px}
h1{margin:8px 0 8px;font-size:36px;line-height:1.1;font-weight:700;letter-spacing:-.5px;color:var(--primary)}
.desc{color:var(--body);font-size:16px;max-width:560px}
.axes{display:flex;gap:12px;margin-top:20px}
.axis{background:var(--card);border-radius:8px;padding:12px 16px;min-width:150px;flex:1;max-width:220px}
.axis small{display:block;color:var(--muted);font-size:12px;font-weight:500}
.axis b{font-size:20px;font-weight:700;color:var(--white)}
.alert{display:inline-block;margin-top:16px;padding:6px 12px;border-radius:6px;background:var(--elev);color:var(--primary);font-weight:600}
.grid{display:grid;gap:24px;grid-template-columns:1fr}
.card{background:var(--card);border-radius:12px;padding:24px;margin-bottom:24px}
.grid > div > .card:last-child,.pane > .card:last-of-type{margin-bottom:0}
.pagehead{padding:24px 0 16px}
.pagehead h2{margin:0}
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
.tag{padding:2px 10px;border-radius:9999px;background:var(--elev);font-size:12px;font-weight:500;color:var(--body);white-space:nowrap}
.tag.cur{background:var(--primary);color:var(--ink);font-weight:600}
.note{color:var(--muted);font-size:12px;margin:12px 0 0}
.sub{color:var(--muted2);font-size:12px;margin:-8px 0 12px}
.legend{display:flex;gap:16px;margin-bottom:8px;font-size:12px;color:var(--body)}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted2);font-size:13px;padding:8px 0}
details table{margin-top:4px}
.sechead{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}
.sechead h2{margin:0}
.stock{padding:12px 0;border-bottom:1px solid var(--hair)}
.stock:last-child{border-bottom:0}
.sh{display:flex;align-items:baseline;gap:6px}
.sh .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin:0}
.sh .sw{color:var(--muted2);font-size:12px}
.sm{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 6px;margin-top:8px}
.sm small{display:block;color:var(--muted);font-size:11px}
.sm b{font-weight:600;font-size:13px}
.score{margin-left:auto;min-width:34px;text-align:center;padding:1px 8px;border-radius:9999px;background:var(--elev);color:var(--muted2);font-weight:700;font-size:13px}
.score.hi{background:var(--primary);color:var(--ink)}
.sh .sw{margin-left:0}
.chip{background:var(--card);border:1px solid var(--hair);color:var(--body);font:inherit;font-size:13px;font-weight:600;padding:6px 14px;border-radius:9999px;cursor:pointer}
.chip.on{background:var(--primary);border-color:var(--primary);color:var(--ink)}
.only-hi .stock:not(.hi){display:none}
.why{margin-top:4px}.why summary{font-size:12px;padding:6px 0 0}.why .sm{margin-top:6px}
footer{background:var(--soft);color:var(--ink);padding:40px 0}
footer p{margin:0;font-size:13px;line-height:1.7;max-width:760px}
footer b{display:block;margin-bottom:8px;font-size:14px}
footer .upd{margin-top:12px;color:#707a8a;font-size:12px}
@media(min-width:1024px){h1{font-size:64px;letter-spacing:-1px}.hero{padding:48px 0 32px}
.grid.two{grid-template-columns:7fr 5fr;align-items:start}.grid.two>div>.card{margin-bottom:24px}
.stocks{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}.stocks .card{margin:0}}
@media(max-width:767px){
body{padding-bottom:64px}
.tabs{position:fixed;top:auto;bottom:0;left:0;right:0;border-bottom:0;border-top:1px solid var(--hair);padding-bottom:env(safe-area-inset-bottom)}
.tab{flex:1;padding:14px 4px;border-bottom:0;border-top:2px solid transparent;font-size:13px}
.tab.on{border-top-color:var(--primary)}
.stocks .card{margin-bottom:16px}}
@media(max-width:480px){.card{padding:16px}.axis{min-width:0}}
"""

JS = """
const tabs=[...document.querySelectorAll('[data-tab]')],panes=[...document.querySelectorAll('.pane')];
function show(id){if(!panes.some(p=>p.id===id))id='now';
panes.forEach(p=>p.classList.toggle('on',p.id===id));tabs.forEach(t=>t.classList.toggle('on',t.dataset.tab===id));}
tabs.forEach(t=>t.addEventListener('click',()=>{location.hash=t.dataset.tab;window.scrollTo(0,0)}));
window.addEventListener('hashchange',()=>show(location.hash.slice(1)));
show(location.hash.slice(1));
const oh=document.getElementById('onlyhi');
if(oh){oh.addEventListener('click',()=>{const on=document.getElementById('stocks').classList.toggle('only-hi');oh.classList.toggle('on',on);
document.querySelectorAll('#stocks .stocks > .card').forEach(c=>{c.style.display=(on&&!c.querySelector('.stock.hi'))?'none':'';
c.querySelectorAll('details:not(.why)').forEach(d=>{if(on&&d.querySelector('.stock.hi'))d.open=true})})})}
"""

LINES = [("전략", "#FCD535", 2.5), ("SPY", "#eaecef", 1.5), ("60/40", "#707a8a", 1.5)]
VISIBLE_STOCKS = 5


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
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c, _ in LINES)
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
    return f"""<div class="card">
<div class="sub num">{bt["start"]} ~ {bt["end"]} · 월 1회 리밸런싱 · 거래비용 0.1% 반영</div>
<div class="legend">{legend}</div>{chart_svg(bt["equity"])}
<table style="margin-top:16px"><tr><th></th>{"".join(f"<th>{n}</th>" for n, _, _ in LINES)}</tr>{rows}</table>
<details><summary>연도별 수익률</summary><table><tr><th></th>{"".join(f"<th>{n}</th>" for n, _, _ in LINES)}</tr>{ann_rows}</table></details>
<p class="note">국면 구성: {mix} · 국면 전환 {bt["switches"]}회</p>
<p class="note">월간 경제지표는 45일 지연해 반영했지만 수정 후 확정치를 쓰고, 추천 비중은 과거 경험에 기반해 정한 값이라 실제 미래 성과를 보장하지 않습니다. 과거 성과는 미래를 보장하지 않으며, 이 기간(2007~)에는 금리 인하와 장기 상승장이 포함돼 있습니다.</p></div>"""


def _cell(label, value):
    return f'<div><small>{label}</small><b class="num">{value}</b></div>'


def stock_item(s) -> str:
    e = html.escape
    sc = s["score"]
    total = sc["total"]
    hi = total is not None and total >= SCORE_THRESHOLD
    badge = (f'<span class="score num{" hi" if hi else ""}">{total:.0f}</span>' if total is not None
             else '<span class="score num na">-</span>')
    groups = "".join(_cell(g, "-" if v is None else f"{v:.0f}") for g, v in sc["groups"].items())
    why = (f'<details class="why"><summary>점수 상세 · 데이터 {sc["coverage"]*100:.0f}%</summary>'
           f'<div class="sm">{groups}</div></details>')
    fwd = f'{s["fwd_pe"]:.1f}배' if s.get("fwd_pe") else "-"
    ed = "-"
    if s.get("earnings"):
        y, m, d = s["earnings"].split("-")
        ed = f"{int(m)}/{int(d)}"
    return (
        f'<div class="stock{" hi" if hi else ""}"><div class="sh"><span class="tk">{e(s["symbol"])}</span>'
        f'<span class="nm">{e(s["name"])}</span><span class="sw num">ETF 내 {s["etf_weight"]*100:.1f}%</span>{badge}</div>'
        '<div class="sm">'
        + _cell("3개월", pct(s["m3"])) + _cell("12개월", pct(s["m12"]))
        + _cell("변동성", f'{s["vol"]*100:.0f}%') + _cell("200일선", pct(s["ma200"]))
        + _cell("52주 고점", pct(s["from_high"])) + _cell("선행 PER", fwd) + _cell("다음 실적", ed)
        + "</div>" + why + "</div>"
    )


def _rule_fmt(key, v):
    if key == "debt_eq":
        return f"{v:.0f}%"
    if key == "fwd_pe":
        return f"{v:.0f}배"
    return f"{v*100:.0f}%"


def criteria_card() -> str:
    rows = "".join(
        f'<tr><td>{html.escape(SCORE_LABELS[k])}<div class="nm" style="margin:0">{g}</div></td><td class="r num">{w}</td>'
        f'<td class="r num">{_rule_fmt(k, lo)}</td><td class="r num">{_rule_fmt(k, hi)}</td></tr>'
        for k, (g, w, lo, hi) in SCORE_RULES.items()
    )
    return f"""<details class="card"><summary style="margin:0;color:var(--white);font-weight:600;font-size:15px">점수 기준 보기</summary>
<table style="margin-top:12px"><tr><th>지표</th><th>가중치</th><th>0점</th><th>100점</th></tr>{rows}</table>
<p class="note">지표마다 0점·100점 기준 사이를 직선으로 환산한 뒤 가중 평균합니다. 값이 없는 지표는 제외하고 남은 가중치로 환산하며, 가중치 기준 60% 미만만 있으면 점수를 내지 않습니다. {SCORE_THRESHOLD}점 이상이면 노란색으로 강조합니다.</p>
<p class="note">기준값·가중치·강조 점수는 과거 경험으로 정한 것이며, 과거 시점의 재무 데이터를 무료로 복원할 수 없어 수익률로 검증하지 못했습니다. 높은 점수가 높은 수익을 뜻하지 않으며, 점수는 후보를 좁히기 위한 참고용입니다. 수치는 Yahoo Finance 기준이고 오류나 누락이 있을 수 있습니다.</p></details>"""


def stocks_pane(groups) -> str:
    if not groups:
        return '<div class="card"><p class="note" style="margin:0">종목 데이터를 가져오지 못했습니다. 다음 갱신 때 다시 시도합니다.</p></div>'
    cards = ""
    for g in groups:
        first = "".join(stock_item(s) for s in g["stocks"][:VISIBLE_STOCKS])
        rest = g["stocks"][VISIBLE_STOCKS:]
        more = (f'<details><summary>나머지 {len(rest)}개 보기</summary>{"".join(stock_item(s) for s in rest)}</details>'
                if rest else "")
        cards += (
            f'<div class="card"><div class="sechead"><h2>{g["etf"]} {html.escape(g["name"])}</h2>'
            f'<span class="tag">추천 비중 {g["weight"]}%</span></div>{first}{more}</div>'
        )
    scored = [s for g in groups for s in g["stocks"] if s["score"]["total"] is not None]
    n_hi = sum(1 for s in scored if s["score"]["total"] >= SCORE_THRESHOLD)
    return f"""<div class="pagehead"><h2>관심 종목 후보</h2>
<div class="sub" style="margin:4px 0 12px">현재 국면의 추천 섹터 ETF에서 비중이 큰 종목 · 점수 {SCORE_THRESHOLD}점 이상 {n_hi}개 / 점수 산출 {len(scored)}개</div>
<button class="chip" id="onlyhi" type="button">{SCORE_THRESHOLD}점 이상만 보기</button></div>
<div class="stocks">{cards}</div><div style="margin-top:16px">{criteria_card()}</div>
<p class="note">참고용 후보 목록이며 매수 추천이 아닙니다. 시가총액 비중이 큰 종목일 뿐 저평가나 상승을 뜻하지 않습니다. 개별 종목은 ETF보다 변동이 크고 한 종목에 집중하면 크게 손실날 수 있으며, 이 종목 목록은 백테스트하지 않았습니다. 변동성은 최근 1년 연환산 값, 200일선·52주 고점은 현재가와의 차이, 선행 PER은 예상 이익 기준이며 데이터에 따라 비어 있을 수 있습니다.</p>"""


def render(now, cur, history, p, bt, sk=None):
    reg = REGIMES[cur["key"]]
    e = html.escape
    arrow = lambda up: "▲ 상승" if up else "▼ 하락"

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
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b0e11">
<title>경기 국면 · 추천 포트폴리오</title>
<script>document.documentElement.classList.add('js')</script>
<style>{CSS}</style></head><body>
<nav><span class="t">REGIME PORTFOLIO</span><span class="tag cur reg">{e(reg["name"])}</span></nav>
<div class="tabs" role="tablist">
<button class="tab on" data-tab="now">현재</button><button class="tab" data-tab="stocks">종목</button>
<button class="tab" data-tab="backtest">백테스트</button><button class="tab" data-tab="market">시장</button></div>

<div class="wrap">
<section class="pane on" id="now">
<div class="hero">
<div class="eyebrow">현재 경기 국면</div>
<h1>{e(reg["name"])}</h1>
<div class="desc">{e(reg["desc"])}</div>
{changed}
<div class="axes">
<div class="axis"><small>성장</small><b>{arrow(cur["growth_up"])}</b></div>
<div class="axis"><small>물가</small><b>{arrow(cur["inflation_up"])}</b></div>
</div></div>
<div class="grid two">
<div><div class="card"><h2>추천 포트폴리오</h2><div class="cls">{cls}</div>{bars}</div></div>
<div><div class="card"><h2>판정 근거</h2>
<h3>성장</h3><table>{sig_rows(cur["growth"])}</table>
<h3>물가</h3><table>{sig_rows(cur["inflation"])}</table>
<p class="note">각 축에서 3개 신호 중 2개 이상이 긍정이면 상승으로 판정합니다.</p></div></div>
</div></section>

<section class="pane" id="stocks">{stocks_pane(sk)}</section>

<section class="pane" id="backtest">
<div class="pagehead"><h2>백테스트</h2></div>{backtest_card(bt)}</section>

<section class="pane" id="market">
<div class="pagehead"><h2>시장</h2></div>
<div class="grid two">
<div><div class="card"><h2>자산별 수익률</h2>
<table><tr><th></th><th>현재가</th><th>1개월</th><th>3개월</th><th>12개월</th></tr>{perf_rows}</table></div></div>
<div><div class="card"><h2>최근 {len(history)}개월 국면</h2>{hist}</div></div>
</div></section>
</div>

<footer><div class="wrap"><b>유의사항</b>
<p>이 페이지는 단순 규칙 기반 참고 자료이며 투자 권유나 수익 보장이 아닙니다. 경제지표는 발표 지연이 있고, 과거 국면 추이는 당시 발표 전 수치가 아닌 현재 확정 수치로 계산한 것이라 실제 대응과 다를 수 있습니다. 최종 투자 판단과 책임은 본인에게 있습니다.</p>
<div class="upd num">갱신 {now} UTC</div></div></footer>
<script>{JS}</script>
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

    sk = stocks.candidates(REGIMES[cur["key"]]["weights"], ASSETS)

    OUT.mkdir(exist_ok=True)
    now =datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    (OUT / "index.html").write_text(render(now, cur, history, perf(px), bt, sk), encoding="utf-8")
    (OUT / "data.json").write_text(
        json.dumps({"updated": now, "regime": cur["key"], "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"현재 국면: {REGIMES[cur['key']]['name']} (성장 {'↑' if cur['growth_up'] else '↓'}, 물가 {'↑' if cur['inflation_up'] else '↓'})")
    for n, v, ok in cur["growth"] + cur["inflation"]:
        print(f"  {'+' if ok else '-'} {n}: {v}")


if __name__ == "__main__":
    main()
