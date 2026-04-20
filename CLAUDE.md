# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A quantitative signal discovery and backtesting pipeline for Indian stock markets (NSE). Two components: a Signal Scanner that ranks features by predictive power using quintile sorting, and a Quick Backtest Engine that validates promising signals.

## Commands

```bash
cd strategy_factory

# Install dependencies
pip install yfinance pandas numpy scipy

# Signal scanning
python run_scan.py --test              # Quick test with 5 stocks
python run_scan.py                     # Full Nifty 200 scan (5-day forward)
python run_scan.py --fwd-days 10       # Different forward period
python run_scan.py --all-periods       # Scan all forward periods (1,3,5,10,20d)
python run_scan.py --detail rsi_3      # Detailed quintile breakdown for one feature
python run_scan.py --save              # Save results to results/scans/

# Backtesting
python run_backtest.py --signal rsi_3 --condition quintile_1
python run_backtest.py --signal rsi_3 --condition quintile_1 --hold 1,3,5,10,20
python run_backtest.py --signal overnight_gap --condition below_threshold --threshold -0.5
python run_backtest.py --signal rsi_3 --condition quintile_1 --test  # 5-stock test
```

## Architecture

The data flows linearly: **Download → Compute Features → Z-score Normalize → Quintile Sort → Measure Forward Returns → Classify Signal Strength → Backtest Winners**.

`config.py` is the single source of truth for universe (Nifty 200 ticker list), date ranges, thresholds, and backtest parameters. Both `run_scan.py` and `run_backtest.py` read from it.

`signals/features.py` contains pure functions that take OHLCV Series/DataFrames and return feature Series. `compute_all_features()` is the registry — to add a new feature, write a function and add one line there.

`signals/scanner.py` handles data download (with pickle caching in `data/cache/`), panel construction (stacking all stocks into one DataFrame with ticker/date/features/forward-returns), quintile analysis per feature, and t-stat significance testing.

`backtest/engine.py` simulates trades: enters on next day's open after signal, checks stop-loss against intraday lows daily, exits at close after hold period. `backtest/metrics.py` computes Sharpe, MDD, PF, WR, CAGR.

## Hard Rules (must be respected in all changes)

1. **Z-score normalize** all cross-stock signals by rolling 252-day history (except calendar/binary features)
2. **Minimum 30 trades/year** — below this, results are noise
3. **Always check last 12 months separately** — dead recent edge kills a strategy regardless of historical performance
4. **Long only** — no short signals
5. **5-day hold default** — test others too, but 5d is primary
6. **-8% stop-loss** on every position
7. **Don't combine weak signals** with strong ones — each must pass quintile test independently
8. **Check signal family overlap** — dead families: F&O Open Interest, SIP calendar flows, FII flows as filter
9. **Feasibility before code** — count qualifying universe first; if <30 trades/year, don't build it
10. **Never trust hardcoded financial data** — validate samples manually

## Key Design Decisions

- yfinance tickers use `.NS` suffix for NSE; index tickers use `^` prefix (e.g., `^NSEI`)
- Data cached as pickle files; cache invalidated after 1 day
- Nifty 200 list is hardcoded current constituents — has survivorship bias (acceptable for exploration, must fix for production)
- Backtest enters on next-day open (realistic: signal computed at close), not same-day
- Stop-loss checked against intraday lows, not just close prices
- Transaction cost of 0.1% per round trip deducted from each trade

## Data Freshness

- **Refresh the Nifty 200 constituent list periodically** (quarterly at minimum). The list in `config.py` is a point-in-time snapshot. Fetch the latest from `https://archives.nseindia.com/content/indices/ind_nifty200list.csv` and update `NIFTY_200` in `config.py`.
- **Use `--force-download`** flag to re-download OHLCV data when stale (cache auto-expires after 1 day, but force if needed mid-day).
- **Delete `data/cache/*.pkl`** if switching universes or after major index rebalancing events.
