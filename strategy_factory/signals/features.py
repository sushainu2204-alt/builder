"""
Feature computation functions.

Each function takes a DataFrame with OHLCV columns and returns a Series.
Easy to add new features: just write a function and register it in FEATURE_REGISTRY.
"""
import numpy as np
import pandas as pd


def _safe_divide(a, b):
    """Division that returns NaN instead of inf."""
    result = a / b
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# ── Price-based ──────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def n_day_return(close: pd.Series, n: int) -> pd.Series:
    """N-day percentage return."""
    return close.pct_change(n)


def distance_from_52w_high(close: pd.Series) -> pd.Series:
    """Close / rolling 252-day max."""
    rolling_max = close.rolling(252, min_periods=126).max()
    return _safe_divide(close, rolling_max)


def distance_from_52w_low(close: pd.Series) -> pd.Series:
    """Close / rolling 252-day min."""
    rolling_min = close.rolling(252, min_periods=126).min()
    return _safe_divide(close, rolling_min)


def bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
    """(close - SMA) / (2 * std). Position within Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return _safe_divide(close - sma, 2 * std)


# ── Volume-based ─────────────────────────────────────────────────────────

def volume_ratio(volume: pd.Series) -> pd.Series:
    """Today's volume / 20-day average volume."""
    avg_vol = volume.rolling(20).mean()
    return _safe_divide(volume, avg_vol)


def volume_spike(volume: pd.Series) -> pd.Series:
    """Binary flag: volume > 2x 20-day average."""
    avg_vol = volume.rolling(20).mean()
    return (volume > 2 * avg_vol).astype(float)


def volume_ratio_5d_vs_20d(volume: pd.Series) -> pd.Series:
    """5-day average volume / 20-day average volume."""
    avg_5 = volume.rolling(5).mean()
    avg_20 = volume.rolling(20).mean()
    return _safe_divide(avg_5, avg_20)


# ── Volatility ───────────────────────────────────────────────────────────

def realized_volatility_20d(close: pd.Series) -> pd.Series:
    """20-day realized volatility (annualized)."""
    returns = close.pct_change()
    return returns.rolling(20).std() * np.sqrt(252)


def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR(14) / close."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return _safe_divide(atr, close)


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(high - low) / close."""
    return _safe_divide(high - low, close)


# ── Gap/Open ─────────────────────────────────────────────────────────────

def overnight_gap(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """(open / previous close) - 1."""
    prev_close = close.shift(1)
    return _safe_divide(open_price, prev_close) - 1


def gap_direction(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """1 for positive gap, 0 for negative gap."""
    prev_close = close.shift(1)
    return (open_price > prev_close).astype(float)


# ── Calendar ─────────────────────────────────────────────────────────────

def day_of_week(dates: pd.DatetimeIndex) -> pd.Series:
    """Day of week: 0 (Monday) to 4 (Friday)."""
    return pd.Series(dates.dayofweek, index=dates, dtype=float)


def month_of_year(dates: pd.DatetimeIndex) -> pd.Series:
    """Month: 1-12."""
    return pd.Series(dates.month, index=dates, dtype=float)


def day_of_month_bucket(dates: pd.DatetimeIndex) -> pd.Series:
    """Day of month bucket: 1 (days 1-10), 2 (11-20), 3 (21-31)."""
    days = dates.day
    buckets = np.where(days <= 10, 1, np.where(days <= 20, 2, 3))
    return pd.Series(buckets, index=dates, dtype=float)


# ── Cross-sectional ──────────────────────────────────────────────────────

def relative_strength_vs_index(stock_close: pd.Series, index_close: pd.Series, period: int = 20) -> pd.Series:
    """Stock's N-day return minus index's N-day return."""
    stock_ret = stock_close.pct_change(period)
    index_ret = index_close.pct_change(period)
    # Align index to stock dates
    index_ret_aligned = index_ret.reindex(stock_close.index, method="ffill")
    return stock_ret - index_ret_aligned


# ── Feature Registry ─────────────────────────────────────────────────────

def compute_all_features(df: pd.DataFrame, index_close: pd.Series = None) -> pd.DataFrame:
    """
    Compute all features for a single stock's OHLCV DataFrame.

    df must have columns: Open, High, Low, Close, Volume
    Returns DataFrame with one column per feature.
    """
    features = {}
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    vol = df["Volume"]
    dates = df.index

    # Price-based
    features["rsi_3"] = rsi(close, 3)
    features["rsi_14"] = rsi(close, 14)
    for n in [1, 5, 10, 20, 60, 120, 252]:
        features[f"return_{n}d"] = n_day_return(close, n)
    features["dist_52w_high"] = distance_from_52w_high(close)
    features["dist_52w_low"] = distance_from_52w_low(close)
    features["bollinger_pos"] = bollinger_position(close)

    # Volume-based
    features["volume_ratio"] = volume_ratio(vol)
    features["volume_spike"] = volume_spike(vol)
    features["vol_5d_vs_20d"] = volume_ratio_5d_vs_20d(vol)

    # Volatility
    features["realized_vol_20d"] = realized_volatility_20d(close)
    features["norm_atr_14"] = normalized_atr(high, low, close)
    features["intraday_range"] = intraday_range(high, low, close)

    # Gap/Open
    features["overnight_gap"] = overnight_gap(opn, close)
    features["gap_direction"] = gap_direction(opn, close)

    # Calendar
    features["day_of_week"] = day_of_week(dates)
    features["month"] = month_of_year(dates)
    features["dom_bucket"] = day_of_month_bucket(dates)

    # Cross-sectional
    if index_close is not None:
        features["rel_strength_20d"] = relative_strength_vs_index(close, index_close, 20)

    return pd.DataFrame(features, index=df.index)
