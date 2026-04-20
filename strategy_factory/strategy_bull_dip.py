"""
BULL STRATEGY 4: Buy the Dip in Bull Market

When: Nifty > 20d SMA AND breadth > 60%
Signal: Buy bottom decile of 5-day returns (biggest dippers in a bull market)
Hold: 10 trading days
Positions: 3
Stop: -8%
Universe: Nifty 200
"""
import numpy as np
import pandas as pd
from signals.scanner import zscore_normalize
from quality_filters import (
    check_news_spike, check_liquidity, check_price_floor,
    check_dip_depth, check_crash, check_panic_days, check_sma_distance,
    merge_flag,
)

STRATEGY_NAME = "Bull Dip Buy + Breadth>60% (N200)"
HOLD_DAYS = 10
MAX_POSITIONS = 3
STOP_PCT = 0.08


def check_active(regime):
    return not regime["is_bear"] and regime["breadth"] > 60


def compute_signals(stock_data):
    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            vol = df["Volume"]
            if len(close) < 100:
                continue

            ret5d = close.pct_change(5)
            ret5d_z = zscore_normalize(ret5d, lookback=252)

            if pd.isna(ret5d_z.iloc[-1]):
                continue

            # --- Dip-specific filters ---
            # We're buying weakness in a bull market. Key risks:
            # 1. News-driven dip (earnings miss, fraud) - won't mean-revert
            # 2. Dip too deep (>15% in 5d) - structural break, not a dip
            # 3. Already in downtrend (below 60d SMA) - not a dip, it's a trend
            # 4. Panic volume dumping - smart money exiting
            flag, reasons = "CLEAN", []

            for result in [
                check_news_spike(close, vol, lookback=5, vol_mult=6, ret_thresh=0.06),  # tightest — dip buys are most vulnerable to news
                check_liquidity(close, vol, min_turnover_cr=2.0),     # N200 large caps
                check_price_floor(close, min_price=30),
                check_dip_depth(close, days=5, max_dip_pct=-12),      # >12% dip in bull = broken
                check_crash(close, lookback=10, skip_pct=-8, warn_pct=-5),
                check_panic_days(close, vol, lookback=10, vol_mult=3, ret_thresh=-0.04),  # tightest panic filter — any institutional selling is bad for dip buys
                check_sma_distance(close, period=60, skip_pct=-8, warn_pct=-4),  # tighter — dip buy needs stock still in uptrend
            ]:
                flag, reason = merge_flag(flag, result)
                if reason:
                    reasons.append(reason)

            prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
            high20 = close.rolling(20).max()
            dd20 = (close.iloc[-1] / high20.iloc[-1] - 1) * 100

            signals.append({
                "ticker": ticker,
                "close": round(close.iloc[-1], 2),
                "signal_value": round(ret5d_z.iloc[-1], 2),
                "ret5d": round(ret5d.iloc[-1] * 100, 1),
                "ret20d": round(close.pct_change(20).iloc[-1] * 100, 1) if len(close) >= 20 else 0,
                "dd20": round(dd20, 1),
                "gap_pct": round((close.iloc[-1] / prev_close - 1) * 100, 1),
                "flag": flag,
                "flag_reasons": "; ".join(reasons) if reasons else "No red flags",
            })
        except Exception:
            pass

    # Sort ascending — lowest z-score = biggest dippers
    return pd.DataFrame(signals).dropna(subset=["signal_value"]).sort_values("signal_value")


def get_picks(signals_df, held_tickers=set(), max_new=None):
    if max_new is None:
        max_new = MAX_POSITIONS
    cutoff = max(len(signals_df) // 10, 3)
    candidates = signals_df.head(cutoff).copy()
    candidates = candidates[~candidates["ticker"].isin(held_tickers)]
    candidates = candidates[candidates["gap_pct"] > -5]
    skipped = candidates[candidates["flag"] == "SKIP"]
    candidates = candidates[candidates["flag"] != "SKIP"]
    return candidates.head(max_new), skipped
