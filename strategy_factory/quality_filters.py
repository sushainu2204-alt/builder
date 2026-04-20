"""
Shared quality filter functions for all strategies.

Each function takes stock data and returns (flag, reason) or None.
Strategies call individual filters with their own thresholds.
"""


def check_news_spike(close, vol, lookback=5, vol_mult=10, ret_thresh=0.10):
    """News-driven spike: vol > Nx avg AND |return| > X% in any recent day."""
    daily_ret = close.pct_change()
    vol_avg_20 = vol.rolling(20).mean()
    for i in range(-lookback, 0):
        if i >= -len(close):
            v_ratio = vol.iloc[i] / vol_avg_20.iloc[i] if vol_avg_20.iloc[i] > 0 else 0
            ret = abs(daily_ret.iloc[i])
            if v_ratio > vol_mult and ret > ret_thresh:
                day_ret_pct = daily_ret.iloc[i] * 100
                return "SKIP", f"News spike: {day_ret_pct:+.1f}% on {v_ratio:.0f}x volume ({lookback}d)"
    return None


def check_liquidity(close, vol, min_turnover_cr=1.0):
    """Minimum liquidity: 20d avg turnover must exceed threshold."""
    turnover = close * vol
    avg_turnover = turnover.rolling(20).mean().iloc[-1]
    min_val = min_turnover_cr * 1e7  # 1 Cr = 10M
    if avg_turnover < min_val:
        return "SKIP", f"Illiquid: avg turnover {avg_turnover/1e7:.1f}Cr < {min_turnover_cr}Cr"
    return None


def check_price_floor(close, min_price=20):
    """Penny stock filter."""
    if close.iloc[-1] < min_price:
        return "SKIP", f"Price {close.iloc[-1]:.1f} below {min_price} floor"
    return None


def check_over_extension(close, days=5, warn_pct=25, skip_pct=35):
    """Stock that already ran too much in N days."""
    if len(close) < days:
        return None
    ret = (close.iloc[-1] / close.iloc[-days] - 1) * 100
    if ret > skip_pct:
        return "SKIP", f"Over-extended: +{ret:.1f}% in {days}d (>{skip_pct}%)"
    if ret > warn_pct:
        return "CAUTION", f"Extended: +{ret:.1f}% in {days}d (>{warn_pct}%)"
    return None


def check_gap_up_blowoff(close, max_gap_pct=8):
    """Latest day gap-up suggests blow-off top."""
    if len(close) < 2:
        return None
    gap = (close.iloc[-1] / close.iloc[-2] - 1) * 100
    if gap > max_gap_pct:
        return "SKIP", f"Gap-up blow-off: +{gap:.1f}% (>{max_gap_pct}%)"
    return None


def check_dip_depth(close, days=5, max_dip_pct=-15):
    """Too deep a dip is likely broken, not mean-reverting."""
    if len(close) < days:
        return None
    ret = (close.iloc[-1] / close.iloc[-days] - 1) * 100
    if ret < max_dip_pct:
        return "SKIP", f"Dip too deep: {ret:.1f}% in {days}d (broken, not dipping)"
    return None


def check_crash(close, lookback=10, skip_pct=-8, warn_pct=-5):
    """Single-day crash in recent window."""
    daily_ret = close.pct_change()
    worst_1d = daily_ret.tail(lookback).min() * 100
    if worst_1d < skip_pct:
        return "SKIP", f"Crashed {worst_1d:.1f}% in a day ({lookback}d)"
    if worst_1d < warn_pct:
        return "CAUTION", f"Dropped {worst_1d:.1f}% in a day ({lookback}d)"
    return None


def check_panic_days(close, vol, lookback=10, vol_mult=4, ret_thresh=-0.05):
    """Days with huge volume + big drop = institutional dumping."""
    daily_ret = close.pct_change()
    vol_avg = vol.rolling(20).mean()
    panic = sum(1 for i in range(-lookback, 0) if i >= -len(close)
                and daily_ret.iloc[i] < ret_thresh
                and vol.iloc[i] > vol_mult * vol_avg.iloc[i])
    if panic > 0:
        return "SKIP", f"{panic} panic day(s): >{vol_mult}x vol + >{abs(ret_thresh)*100:.0f}% drop"
    return None


def check_sma_distance(close, period=60, skip_pct=-10, warn_pct=-5):
    """Distance from SMA — deep below means structural weakness."""
    sma = close.rolling(period).mean().iloc[-1]
    if sma <= 0:
        return None
    dist = (close.iloc[-1] / sma - 1) * 100
    if dist < skip_pct:
        return "SKIP", f"Deep below {period}d SMA ({dist:+.1f}%)"
    if dist < warn_pct:
        return "CAUTION", f"Below {period}d SMA ({dist:+.1f}%)"
    return None


def check_sma_overshoot(close, period=20, skip_pct=20, warn_pct=15):
    """Too far ABOVE SMA — chasing an extended move."""
    sma = close.rolling(period).mean().iloc[-1]
    if sma <= 0:
        return None
    dist = (close.iloc[-1] / sma - 1) * 100
    if dist > skip_pct:
        return "SKIP", f"Over {period}d SMA by +{dist:.1f}% (>{skip_pct}%)"
    if dist > warn_pct:
        return "CAUTION", f"Extended above {period}d SMA (+{dist:.1f}%)"
    return None


def merge_flag(current_flag, result):
    """Merge a filter result into running flag. Returns (new_flag, reason_or_None)."""
    if result is None:
        return current_flag, None
    r_flag, r_reason = result
    if r_flag == "SKIP":
        return "SKIP", r_reason
    if r_flag == "CAUTION" and current_flag != "SKIP":
        return "CAUTION", r_reason
    return current_flag, r_reason
