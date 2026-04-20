"""
Signal Scanner — the "quintile sorter".

Downloads data, computes features, sorts into quintiles,
measures forward returns, and ranks signals by predictive power.
"""
import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CACHE_DIR, START_DATE, END_DATE, NIFTY50_TICKER,
    NUM_QUINTILES, FORWARD_RETURN_DAYS, DEFAULT_FORWARD_DAYS,
    MIN_TSTAT_STRONG, MIN_TSTAT_MODERATE,
)
from signals.features import compute_all_features

warnings.filterwarnings("ignore")


def get_cache_path(ticker: str) -> str:
    """Return path for cached ticker data."""
    safe_name = ticker.replace("^", "_IDX_").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.pkl")


def download_stock_data(ticker: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker. Caches to disk.
    Returns empty DataFrame on failure.
    """
    cache_path = get_cache_path(ticker)

    if not force and os.path.exists(cache_path):
        try:
            cached = pickle.load(open(cache_path, "rb"))
            # Re-download if cache is older than 1 day
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if cache_age < timedelta(days=1):
                return cached
        except Exception:
            pass

    try:
        # Add .NS suffix for NSE stocks (skip for index tickers starting with ^)
        yf_ticker = ticker if ticker.startswith("^") else f"{ticker}.NS"
        df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pickle.dump(df, open(cache_path, "wb"))
        return df
    except Exception as e:
        print(f"  [WARN] Failed to download {ticker}: {e}")
        return pd.DataFrame()


def download_universe(tickers: list, start: str, end: str, force: bool = False) -> dict:
    """Download data for all tickers. Returns dict of ticker -> DataFrame."""
    data = {}
    total = len(tickers)
    failed = []

    print(f"\nDownloading data for {total} stocks ({start} to {end})...")
    for i, ticker in enumerate(tickers, 1):
        pct = i / total * 100
        print(f"  [{i}/{total}] ({pct:.0f}%) {ticker}", end="")
        df = download_stock_data(ticker, start, end, force=force)
        if df.empty or len(df) < 60:
            print(f" — SKIPPED (insufficient data)")
            failed.append(ticker)
        else:
            print(f" — {len(df)} days")
            data[ticker] = df

    print(f"\nDownloaded: {len(data)} stocks | Failed/Skipped: {len(failed)}")
    if failed:
        print(f"  Failed tickers: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")

    return data


def compute_forward_returns(close: pd.Series, days_list: list) -> pd.DataFrame:
    """Compute forward returns for multiple holding periods."""
    fwd = {}
    for d in days_list:
        fwd[f"fwd_{d}d"] = close.pct_change(d).shift(-d)
    return pd.DataFrame(fwd, index=close.index)


def zscore_normalize(series: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Z-score normalize a series using its own rolling history.
    Rule #1: Normalize by the stock's own historical distribution.
    """
    rolling_mean = series.rolling(lookback, min_periods=60).mean()
    rolling_std = series.rolling(lookback, min_periods=60).std()
    result = (series - rolling_mean) / rolling_std
    return result.replace([np.inf, -np.inf], np.nan)


def build_panel(stock_data: dict, index_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a panel DataFrame with features, forward returns, and metadata
    for all stocks combined.

    Returns DataFrame with columns:
        ticker, date, feature_1, feature_2, ..., fwd_1d, fwd_3d, ...
    """
    index_close = index_data["Close"] if index_data is not None and not index_data.empty else None

    all_rows = []
    total = len(stock_data)

    print(f"\nComputing features for {total} stocks...")
    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] Processing features...")

        try:
            # Compute features
            features = compute_all_features(df, index_close)

            # Z-score normalize all non-calendar features (Rule #1)
            calendar_cols = ["day_of_week", "month", "dom_bucket", "gap_direction", "volume_spike"]
            for col in features.columns:
                if col not in calendar_cols:
                    features[col] = zscore_normalize(features[col])

            # Compute forward returns
            fwd_rets = compute_forward_returns(df["Close"], FORWARD_RETURN_DAYS)

            # Combine
            combined = pd.concat([features, fwd_rets], axis=1)
            combined["ticker"] = ticker
            combined["date"] = combined.index
            all_rows.append(combined)
        except Exception as e:
            print(f"  [WARN] Error processing {ticker}: {e}")

    if not all_rows:
        return pd.DataFrame()

    panel = pd.concat(all_rows, ignore_index=True)
    print(f"  Panel shape: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    return panel


def quintile_analysis(panel: pd.DataFrame, feature_col: str, fwd_col: str) -> dict:
    """
    Sort stocks into quintiles by feature value each day,
    compute average forward returns per quintile.

    Returns dict with quintile stats, spread, t-stat, etc.
    """
    # Drop rows with NaN in either feature or forward return
    subset = panel[[feature_col, fwd_col, "date"]].dropna()
    if len(subset) < 100:
        return None

    # Assign quintiles per day
    def assign_quintile(group):
        try:
            group["quintile"] = pd.qcut(group[feature_col], NUM_QUINTILES, labels=False, duplicates="drop") + 1
        except ValueError:
            group["quintile"] = np.nan
        return group

    subset = subset.groupby("date", group_keys=False).apply(assign_quintile)
    subset = subset.dropna(subset=["quintile"])

    if len(subset) < 100:
        return None

    # Average forward return per quintile
    quintile_returns = subset.groupby("quintile")[fwd_col].mean() * 100  # Convert to %
    quintile_counts = subset.groupby("quintile")[fwd_col].count()

    q1_returns = subset[subset["quintile"] == 1][fwd_col].values
    q5_returns = subset[subset["quintile"] == NUM_QUINTILES][fwd_col].values

    if len(q1_returns) < 30 or len(q5_returns) < 30:
        return None

    spread = (quintile_returns.iloc[0] - quintile_returns.iloc[-1])

    # T-test between Q1 and Q5
    t_stat, p_value = stats.ttest_ind(q1_returns, q5_returns, equal_var=False)

    # Check monotonicity (how well returns decrease/increase across quintiles)
    qr_values = quintile_returns.values
    diffs = np.diff(qr_values)
    if spread > 0:
        # Expect decreasing: Q1 > Q2 > ... > Q5
        monotonic_score = np.sum(diffs < 0) / len(diffs)
    else:
        # Expect increasing: Q1 < Q2 < ... < Q5
        monotonic_score = np.sum(diffs > 0) / len(diffs)

    # Determine direction label
    if spread > 0:
        direction = "LOW->HIGH" if qr_values[0] > qr_values[-1] else "HIGH->LOW"
    else:
        direction = "HIGH->LOW" if qr_values[0] < qr_values[-1] else "LOW->HIGH"

    return {
        "feature": feature_col,
        "fwd_period": fwd_col,
        "quintile_returns": quintile_returns,
        "quintile_counts": quintile_counts,
        "spread": spread,
        "abs_spread": abs(spread),
        "t_stat": abs(t_stat),
        "p_value": p_value,
        "monotonic_score": monotonic_score,
        "direction": direction,
        "n_observations": len(subset),
    }


def classify_signal(result: dict) -> str:
    """Classify signal strength."""
    if result is None:
        return "INSUFFICIENT_DATA"
    t = result["t_stat"]
    mono = result["monotonic_score"]
    if t >= MIN_TSTAT_STRONG and mono >= 0.5:
        return "STRONG"
    elif t >= MIN_TSTAT_MODERATE:
        return "MODERATE"
    else:
        return "WEAK"


def print_quintile_result(result: dict, classification: str):
    """Pretty-print a single quintile analysis result."""
    if result is None:
        return

    fwd_days = result["fwd_period"].replace("fwd_", "").replace("d", "")
    print(f"\n  Feature: {result['feature']}")
    print(f"  Forward returns by quintile ({fwd_days}d):")

    qr = result["quintile_returns"]
    qc = result["quintile_counts"]
    for q in sorted(qr.index):
        label = ""
        if q == 1:
            label = " (lowest)"
        elif q == NUM_QUINTILES:
            label = " (highest)"
        count = int(qc.get(q, 0))
        print(f"    Q{int(q)}{label:>10}: {qr[q]:>+7.2f}%  (n={count:,})")

    print(f"\n    Spread (Q1-Q5): {result['spread']:>+.2f}%")
    print(f"    T-stat: {result['t_stat']:.2f}")
    print(f"    Monotonicity: {result['monotonic_score']:.0%}")
    print(f"    Direction: {result['direction']}")

    marker = ""
    if classification == "STRONG":
        marker = " << CANDIDATE"
    elif classification == "MODERATE":
        marker = " ~ worth investigating"
    print(f"    Signal quality: {classification}{marker}")


def run_scan(panel: pd.DataFrame, fwd_days: int = None) -> list:
    """
    Run quintile analysis on all features.
    Returns sorted list of results.
    """
    if fwd_days is None:
        fwd_days = DEFAULT_FORWARD_DAYS

    fwd_col = f"fwd_{fwd_days}d"

    # Get all feature columns (exclude forward returns, ticker, date)
    exclude_cols = {"ticker", "date"} | {f"fwd_{d}d" for d in FORWARD_RETURN_DAYS}
    feature_cols = [c for c in panel.columns if c not in exclude_cols]

    results = []
    print(f"\n{'='*60}")
    print(f"SIGNAL SCAN — {fwd_days}-day forward returns")
    print(f"{'='*60}")
    print(f"Testing {len(feature_cols)} features...")

    for feat in feature_cols:
        result = quintile_analysis(panel, feat, fwd_col)
        if result is not None:
            classification = classify_signal(result)
            result["classification"] = classification
            results.append(result)

    # Sort by absolute spread (strongest signals first)
    results.sort(key=lambda x: x["t_stat"], reverse=True)

    return results


def print_scan_summary(results: list, fwd_days: int = None):
    """Print the final ranked summary of all signals."""
    if fwd_days is None:
        fwd_days = DEFAULT_FORWARD_DAYS

    print(f"\n{'='*70}")
    print(f"  SIGNAL SCAN RESULTS — {fwd_days}-day forward returns")
    print(f"  Sorted by t-stat (strongest first)")
    print(f"{'='*70}")

    strong = [r for r in results if r["classification"] == "STRONG"]
    moderate = [r for r in results if r["classification"] == "MODERATE"]
    weak = [r for r in results if r["classification"] == "WEAK"]

    if strong:
        print(f"\n  STRONG SIGNALS (t-stat > {MIN_TSTAT_STRONG}, monotonic):")
        for i, r in enumerate(strong, 1):
            print(f"    {i:>2}. {r['feature']:<22} | Spread: {r['spread']:>+6.2f}% | "
                  f"t-stat: {r['t_stat']:.2f} | {r['direction']}")

    if moderate:
        print(f"\n  MODERATE SIGNALS (t-stat {MIN_TSTAT_MODERATE}-{MIN_TSTAT_STRONG}):")
        for i, r in enumerate(moderate, 1):
            print(f"    {i:>2}. {r['feature']:<22} | Spread: {r['spread']:>+6.2f}% | "
                  f"t-stat: {r['t_stat']:.2f} | {r['direction']}")

    if weak:
        print(f"\n  WEAK/NO SIGNAL (t-stat < {MIN_TSTAT_MODERATE}):")
        for i, r in enumerate(weak, 1):
            print(f"    {i:>2}. {r['feature']:<22} | Spread: {r['spread']:>+6.2f}% | "
                  f"t-stat: {r['t_stat']:.2f} | NO SIGNAL")

    print(f"\n  Total features tested: {len(results)}")
    print(f"  Strong: {len(strong)} | Moderate: {len(moderate)} | Weak: {len(weak)}")
    print(f"{'='*70}")
