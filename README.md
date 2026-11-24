# 경기 국면 포트폴리오 대시보드

경기 국면(성장·물가)을 판정해 미국 ETF 추천 포트폴리오와 백테스트 결과를 보여주는 정적 페이지입니다.

- `python build.py` → `docs/index.html` 생성 (FRED + Yahoo Finance 공개 데이터, API 키 불필요)
- 국면별 비중: `config.py` / 백테스트: `backtest.py`
- GitHub Actions(`.github/workflows/update.yml`)가 평일마다 자동 갱신

참고용 자료이며 투자 권유나 수익 보장이 아닙니다.
