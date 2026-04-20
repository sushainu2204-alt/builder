"""
BULL STRATEGY 3: Relative Strength + Volume Breakout

When: Nifty > 20d SMA
Signal: Stocks with highest relative strength vs Nifty + volume breakout
Hold: 15 trading days
Positions: 3
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

STRATEGY_NAME = "Bull Rel Strength + Vol Breakout (N500)"
HOLD_DAYS = 15
MAX_POSITIONS = 3
STOP_PCT = 0.08


def check_active(regime):
    return not regime["is_bear"]


def compute_signals(stock_data, index_data=None):
    nifty_close = index_data["Close"] if index_data is not None else None

    signals = []
    for ticker, df in stock_data.items():
        try:
            close = df["Close"]
            vol = df["Volume"]
            if len(close) < 100:
                continue

            # Relative strength vs Nifty
            stock_ret_20 = close.pct_change(20)
            if nifty_close is not None:
                nifty_ret_20 = nifty_close.pct_change(20).reindex(close.index, method="ffill")
                rel_str = stock_ret_20 - nifty_ret_20
            else:
                rel_str = stock_ret_20
            rs_z = zscore_normalize(rel_str, lookback=252)

            # Volume breakout
            vol_60d = vol.rolling(60).mean()
            vol_break = vol / vol_60d
            vb_z = zscore_normalize(vol_break, lookback=252)

            if pd.isna(rs_z.iloc[-1]) or pd.isna(vb_z.iloc[-1]):
                continue

            # Require volume breakout > 0 (above average)
            if vb_z.iloc[-1] <= 0:
                continue

            # Combined score
            signal_val = rs_z.iloc[-1] + vb_z.iloc[-1] * 0.3

            # --- RelStr + VolBreak specific filters ---
            # High rel-strength stocks with volume can be news-driven (AFCONS situation).
            # Volume breakout already selects for unusual activity — need to distinguish
            # organic accumulation from news-spike volume.
            flag, reasons = "CLEAN", []

            for result in [
                check_news_spike(close, vol, lookback=5, vol_mult=8, ret_thresh=0.08),  # tighter — vol breakout strategy picks high-vol stocks
                check_liquidity(close, vol, min_turnover_cr=1.0),
                check_price_floor(close, min_price=20),
                check_over_extension(close, days=5, warn_pct=20, skip_pct=30),
                check_over_extension(close, days=20, warn_pct=35, skip_pct=50),  # 20d check since we use 20d returns
                check_gap_up_blowoff(close, max_gap_pct=7),
                check_sma_overshoot(close, period=20, skip_pct=20, warn_pct=15),
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
                "signal_value": round(signal_val, 2),
                "rel_str_z": round(rs_z.iloc[-1], 2),
                "vol_break_z": round(vb_z.iloc[-1], 2),
                "ret5d": round(close.pct_change(5).iloc[-1] * 100, 1),
                "ret20d": round(stock_ret_20.iloc[-1] * 100, 1),
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
