"""
BULL STRATEGY: Breakout + Accumulation (Nifty 500)

When: Nifty > 20d SMA (bull regime)
Signal: Stocks at 20d high with accumulation (rising price + rising volume)
Hold: 15 trading days
Positions: 5 (1.2L each at 6L capital)
Stop: -8%
Universe: Nifty 500
"""
import numpy as np
import pandas as pd
from signals.scanner import zscore_normalize
from quality_filters import (
    check_news_spike, check_liquidity, check_price_floor,
    check_over_extension, check_gap_up_blowoff,
    check_crash, check_panic_days, check_sma_distance,
    check_sma_overshoot, merge_flag,
)

STRATEGY_NAME = "Bull Breakout + Accumulation (N500)"
HOLD_DAYS = 15
MAX_POSITIONS = 5
STOP_PCT = 0.08


def check_active(regime):
    """Returns True if this strategy should trade."""
    return not regime["is_bear"]  # Nifty > 20d SMA


def compute_signals(stock_data):
    """Compute breakout + accumulation signals. Returns sorted DataFrame."""
    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            if len(close) < 100:
                continue

            # At 20d high?
            high_20d = close.rolling(20).max()
            at_20d_high = close.iloc[-1] >= high_20d.iloc[-1] * 0.99

            # Accumulation: rising price + rising volume
            ret_5d = close.pct_change(5)
            vol_5d = vol.rolling(5).mean()
            vol_20d = vol.rolling(20).mean()
            vol_trend = vol_5d / vol_20d
            accumulation = ret_5d * vol_trend
            acc_z = zscore_normalize(accumulation, lookback=252)

            if pd.isna(acc_z.iloc[-1]):
                continue

            # Only consider stocks with accumulation > 0.5 z-score
            if acc_z.iloc[-1] <= 0.5:
                continue

            signal_val = (1.0 if at_20d_high else 0.0) + acc_z.iloc[-1] * 0.1

            # --- Breakout-specific filters ---
            # Breakout stocks are at highs, so we allow extension but cap blow-offs.
            # N500 has small/mid-caps so liquidity floor matters more.
            flag, reasons = "CLEAN", []

            for result in [
                check_news_spike(close, vol, lookback=5, vol_mult=10, ret_thresh=0.10),
                check_liquidity(close, vol, min_turnover_cr=1.0),
                check_price_floor(close, min_price=20),
                check_over_extension(close, days=5, warn_pct=20, skip_pct=30),   # breakouts can run, but >30% in 5d is chasing
                check_gap_up_blowoff(close, max_gap_pct=8),
                check_sma_overshoot(close, period=20, skip_pct=25, warn_pct=18),  # looser for breakouts — they're at highs
                check_crash(close, lookback=10, skip_pct=-8, warn_pct=-5),
                check_panic_days(close, vol, lookback=10, vol_mult=4, ret_thresh=-0.05),
                check_sma_distance(close, period=60, skip_pct=-10, warn_pct=-5),
            ]:
                flag, reason = merge_flag(flag, result)
                if reason:
                    reasons.append(reason)

            prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
            gap_pct = (close.iloc[-1] / prev_close - 1) * 100
            ret_20d = close.pct_change(20).iloc[-1] * 100 if len(close) >= 20 else 0

            signals.append({
                "ticker": ticker,
                "close": round(close.iloc[-1], 2),
                "signal_value": round(signal_val, 3),
                "at_20d_high": at_20d_high,
                "accum_z": round(acc_z.iloc[-1], 2),
                "ret5d": round(close.pct_change(5).iloc[-1] * 100, 1),
                "ret20d": round(ret_20d, 1),
                "gap_pct": round(gap_pct, 1),
                "flag": flag,
                "flag_reasons": "; ".join(reasons) if reasons else "No red flags",
            })
        except Exception:
            pass

    df = pd.DataFrame(signals)
    if len(df) == 0:
        return df
    df = df.dropna(subset=["signal_value"]).sort_values("signal_value", ascending=False)
    return df


def get_picks(signals_df, held_tickers=set(), max_new=None):
    """Filter and return top picks."""
    if max_new is None:
        max_new = MAX_POSITIONS

    cutoff = max(len(signals_df) // 10, 3)
    candidates = signals_df.head(cutoff).copy()
    candidates = candidates[~candidates["ticker"].isin(held_tickers)]
    candidates = candidates[candidates["gap_pct"] > -5]
    skipped = candidates[candidates["flag"] == "SKIP"]
    candidates = candidates[candidates["flag"] != "SKIP"]

    picks = candidates.head(max_new)
    return picks, skipped
