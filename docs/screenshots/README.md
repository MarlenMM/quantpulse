# Screenshot & demo GIF provenance

These images were captured locally against a throwaway scratch database
seeded with **synthetic price series for 15 well-known tickers**, run
through the *real* scoring/forecasting/backtest pipeline (`analysis.scoring`,
`analysis.forecasting`, `analysis.backtest`) — not hand-typed numbers. That
keeps every rating, chart, and metric internally consistent (a ticker whose
synthetic price trends up scores and forecasts accordingly), without needing
real API keys or a real historical backfill just to take a picture of the UI.

Never generated against `quantpulse.db` or `quantpulse_demo.db` — always a
separate scratch file, so this has no effect on your own local data or the
live-deployed demo database (see the root README's "Live Demo & Deployment"
section).

To refresh these images: seed a scratch SQLite DB the same way
`scripts/seed_initial_data.py` does for real data (tickers → price history →
`scoring.build_composite` → `forecasting.generate_forecasts` →
`backtest.sharpe_ratio`/`cagr`/bootstrap-CI on a synthetic basket), point
`DATABASE_URL` at it, launch `streamlit run app/Home.py`, and screenshot each
page. `demo.gif` is the four PNGs (`dashboard.png`, `screener.png`,
`stock_detail.png`, `backtest.png`) cross-faded together with Pillow.
