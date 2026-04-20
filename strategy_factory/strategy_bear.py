"""
BEAR STRATEGY: RSI(3) Oversold + Market Breadth < 40%

When: Nifty < 20d SMA AND breadth < 40%
Signal: Buy bottom decile RSI(3) z-score stocks
Hold: 5 trading days
Positions: 5 (1.2L each at 6L capital)
Stop: -8%
"""
import numpy as np
import pandas as pd
from signals.scanner import zscore_normalize
from quality_filters import (
    check_news_spike, check_liquidity, check_price_floor,
    check_crash, check_panic_days, check_dip_depth,
    merge_flag,
)

STRATEGY_NAME = "Bear Dip (RSI3 + Breadth<40%)"
HOLD_DAYS = 5
MAX_POSITIONS = 5
STOP_PCT = 0.08


def check_active(regime):
    """Returns True if this strategy should trade."""
    return regime["is_bear"] and regime["breadth"] < 40


def compute_signals(stock_data):
    """Compute RSI(3) z-score signals. Returns sorted DataFrame."""
    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            if len(close) < 100:
                continue

            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/3, min_periods=3).mean()
            avg_loss = loss.ewm(alpha=1/3, min_periods=3).mean()
            rs = avg_gain / avg_loss
            rsi3 = 100 - (100 / (1 + rs))
            rsi3_z = zscore_normalize(rsi3, lookback=252)

            if pd.isna(rsi3_z.iloc[-1]):
                continue

            # --- Bear-specific filters ---
            # In bear dip, we EXPECT stocks to be beaten down.
            # But we still avoid: news spikes, illiquid, penny stocks,
            # panic dumping (institutional exit), and dips too deep to recover in 5d.
            vol = df["Volume"]
            flag, reasons = "CLEAN", []

            for result in [
                check_news_spike(close, vol, lookback=5, vol_mult=10, ret_thresh=0.10),
                check_liquidity(close, vol, min_turnover_cr=0.5),     # lower bar — bear markets have low volume
                check_price_floor(close, min_price=20),
                check_crash(close, lookback=10, skip_pct=-12, warn_pct=-8),  # higher tolerance — bear dips ARE crashes
                check_panic_days(close, vol, lookback=10, vol_mult=5, ret_thresh=-0.05),  # stricter vol mult — real panic only
                check_dip_depth(close, days=5, max_dip_pct=-20),      # if down >20% in 5d, it's broken
            ]:
                flag, reason = merge_flag(flag, result)
                if reason:
                    reasons.append(reason)

            # NOTE: no SMA distance check — bear dip stocks are expected to be below SMA

            prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
            gap_pct = (close.iloc[-1] / prev_close - 1) * 100

            signals.append({
                "ticker": ticker,
                "close": round(close.iloc[-1], 2),
                "signal_value": round(rsi3_z.iloc[-1], 2),
                "rsi3": round(rsi3.iloc[-1], 1),
                "ret5d": round(close.pct_change(5).iloc[-1] * 100, 1),
                "gap_pct": round(gap_pct, 1),
                "flag": flag,
                "flag_reasons": "; ".join(reasons) if reasons else "No red flags",
            })
        except Exception:
            pass

    df = pd.DataFrame(signals).dropna(subset=["signal_value"]).sort_values("signal_value")
    return df


def get_picks(signals_df, held_tickers=set(), max_new=None):
    """Filter and return top picks."""
    if max_new is None:
        max_new = MAX_POSITIONS

    # Bottom decile
    cutoff = max(len(signals_df) // 10, 3)
    candidates = signals_df.head(cutoff).copy()

    # Remove held
    candidates = candidates[~candidates["ticker"].isin(held_tickers)]

    # Gap filter
    candidates = candidates[candidates["gap_pct"] > -5]

    # Remove SKIP
    skipped = candidates[candidates["flag"] == "SKIP"]
    candidates = candidates[candidates["flag"] != "SKIP"]

    picks = candidates.head(max_new)
    return picks, skipped
