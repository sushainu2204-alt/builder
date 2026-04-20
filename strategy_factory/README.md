# Strategy Exploration & Backtesting Factory

A pipeline to quickly scan for predictive signals and backtest them on Indian stock markets (NSE).

## Setup

```bash
pip install yfinance pandas numpy scipy
```

## Quick Start

### 1. Test with 5 stocks first
```bash
python run_scan.py --test
```

### 2. Full Nifty 200 scan
```bash
python run_scan.py
```

### 3. Scan with different forward periods
```bash
python run_scan.py --fwd-days 10
python run_scan.py --all-periods
```

### 4. Backtest a promising signal
```bash
python run_backtest.py --signal rsi_3 --condition quintile_1
python run_backtest.py --signal rsi_3 --condition quintile_1 --hold 1,3,5,10,20
```

### 5. Backtest with threshold
```bash
python run_backtest.py --signal overnight_gap --condition below_threshold --threshold -0.5
```

### 6. Save scan results
```bash
python run_scan.py --save
```

## Weekly Workflow

| Day       | Action |
|-----------|--------|
| Monday    | `python run_scan.py --save` — rank features |
| Tuesday   | Pick top 2-3 → `python run_backtest.py` for each |
| Wednesday | Review, kill bad ideas |
| Thursday  | Deeper analysis on survivors |
| Friday    | Log learnings, update dead signal list |

## Adding New Features

Edit `signals/features.py`:
1. Write a function that takes OHLCV columns and returns a Series
2. Register it in `compute_all_features()` — add one line

## Rules

1. All cross-stock signals are z-score normalized
2. Minimum 30 trades/year for validity
3. Last 12 months checked separately
4. Long only
5. 5-day hold is default
6. -8% stop-loss on every position
7. Don't combine weak signals with strong ones
8. Check for signal family overlap
9. Feasibility before code (min 30 trades/year)
10. Never trust hardcoded financial data — validate manually

## Known Limitations

- Nifty 200 list is current constituents (survivorship bias)
- yfinance data quality varies; some tickers may fail
- No dividend adjustments beyond yfinance's auto_adjust
- Simplified equal-weight position sizing
