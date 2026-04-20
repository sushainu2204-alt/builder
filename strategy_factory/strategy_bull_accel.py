"""
BULL STRATEGY 2: Price Acceleration + Strong Breadth

When: Nifty > 20d SMA AND breadth > 60%
Signal: Stocks with accelerating momentum (20d return outpacing 60d trend)
Hold: 15 trading days
Positions: 3
Stop: -8%
Universe: Nifty 200
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

STRATEGY_NAME = "Bull Acceleration + Breadth>60% (N200)"
HOLD_DAYS = 15
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

            ret_20d = close.pct_change(20)
            ret_60d = close.pct_change(60)
            acceleration = ret_20d - (ret_60d / 3)
            acc_z = zscore_normalize(acceleration, lookback=252)

            if pd.isna(acc_z.iloc[-1]):
                continue

            # --- Acceleration-specific filters ---
            # Accelerating stocks are gaining momentum, but could be post-spike.
            # Tighter over-extension since we're buying into momentum, not breakout.
            flag, reasons = "CLEAN", []

            for result in [
                check_news_spike(close, vol, lookback=5, vol_mult=8, ret_thresh=0.08),  # tighter — acceleration can be news-driven
                check_liquidity(close, vol, min_turnover_cr=2.0),     # N200 = large caps, higher bar
                check_price_floor(close, min_price=30),               # N200 shouldn't have penny stocks but safety net
                check_over_extension(close, days=5, warn_pct=15, skip_pct=25),   # tighter — acceleration ≠ already-extended
                check_over_extension(close, days=20, warn_pct=40, skip_pct=60),  # also check 20d extension
                check_gap_up_blowoff(close, max_gap_pct=6),           # tighter gap filter
                check_sma_overshoot(close, period=20, skip_pct=20, warn_pct=15),  # don't chase stocks too far above SMA
                check_crash(close, lookback=10, skip_pct=-8, warn_pct=-5),
                check_panic_days(close, vol, lookback=10, vol_mult=4, ret_thresh=-0.05),
                check_sma_distance(close, period=60, skip_pct=-10, warn_pct=-5),
            ]:
                flag, reason = merge_flag(flag, result)
                if reason:
                    reasons.append(reason)

            prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
            signals.append({
                "ticker": ticker,
                "close": round(close.iloc[-1], 2),
                "signal_value": round(acc_z.iloc[-1], 2),
                "ret20d": round(ret_20d.iloc[-1] * 100, 1),
                "ret60d": round(ret_60d.iloc[-1] * 100, 1),
                "ret5d": round(close.pct_change(5).iloc[-1] * 100, 1),
                "gap_pct": round((close.iloc[-1] / prev_close - 1) * 100, 1),
                "flag": flag,
                "flag_reasons": "; ".join(reasons) if reasons else "No red flags",
            })
        except Exception:
            pass

    return pd.DataFrame(signals).dropna(subset=["signal_value"]).sort_values("signal_value", ascending=False)


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
