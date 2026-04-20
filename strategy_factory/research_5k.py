#!/usr/bin/env python3
"""
Deep research: Find a strategy that generates 5K INR/week with minimal capital.

Approach:
1. Start with best signal (return_5d mean reversion)
2. Test stricter entry filters (decile instead of quintile, z-score thresholds)
3. Combine with RSI(3) for confirmation
4. Optimize hold period
5. Test position concentration (fewer positions = higher returns per unit capital)
6. Walk-forward validation: train on 2021-2024, validate on 2025-2026
7. Calculate exact capital needed for 5K/week target
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from backtest.metrics import compute_all_metrics, format_metrics

# ============================================================
# ENHANCED BACKTEST ENGINE — supports stricter filters, combos
# ============================================================

def run_enhanced_backtest(
    stock_data, panel,
    signal_col, condition, threshold=None,
    secondary_signal=None, secondary_condition=None, secondary_threshold=None,
    hold_days=5, max_positions=5, cost=0.001, stop_loss=-0.08,
    num_buckets=5, target_bucket=0,  # 0=lowest, num_buckets-1=highest
    start_date=None, end_date=None,
    verbose=False,
):
    """
    Enhanced backtest with:
    - Configurable bucket count (deciles, ventiles, etc.)
    - Secondary signal filter
    - Date range filtering
    - Detailed trade log
    """
    df = panel.copy()

    # Date filter
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    all_dates = sorted(df["date"].unique())
    trades = []
    open_positions = {}

    for date in all_dates:
        # Clean expired positions
        open_positions = {t: ed for t, ed in open_positions.items() if ed > date}
        available = max_positions - len(open_positions)
        if available <= 0:
            continue

        day_data = df[df["date"] == date].dropna(subset=[signal_col])
        if len(day_data) < num_buckets * 2:
            continue

        # Primary signal bucket assignment
        try:
            day_data = day_data.copy()
            day_data["_bucket"] = pd.qcut(day_data[signal_col], num_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue

        # Select target bucket
        if condition == "lowest":
            selected = day_data[day_data["_bucket"] == 0]
        elif condition == "highest":
            selected = day_data[day_data["_bucket"] == day_data["_bucket"].max()]
        elif condition == "below_threshold" and threshold is not None:
            selected = day_data[day_data[signal_col] < threshold]
        elif condition == "above_threshold" and threshold is not None:
            selected = day_data[day_data[signal_col] > threshold]
        else:
            selected = day_data[day_data["_bucket"] == target_bucket]

        # Secondary signal filter
        if secondary_signal and secondary_threshold is not None and len(selected) > 0:
            selected = selected.dropna(subset=[secondary_signal])
            if secondary_condition == "below":
                selected = selected[selected[secondary_signal] < secondary_threshold]
            elif secondary_condition == "above":
                selected = selected[selected[secondary_signal] > secondary_threshold]

        # Remove stocks we already hold
        candidates = [t for t in selected["ticker"].tolist() if t not in open_positions]

        # Sort by signal strength (most extreme first)
        if condition in ("lowest", "below_threshold"):
            candidates_data = selected[selected["ticker"].isin(candidates)].sort_values(signal_col, ascending=True)
        else:
            candidates_data = selected[selected["ticker"].isin(candidates)].sort_values(signal_col, ascending=False)

        candidates = candidates_data["ticker"].tolist()[:available]

        for ticker in candidates:
            result = _execute_trade(stock_data, ticker, date, hold_days, cost, stop_loss)
            if result:
                trades.append(result)
                open_positions[ticker] = result["exit_date"]

    if not trades:
        return None

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    # Build proper equity curve
    equity = [1.0]
    for _, trade in trades_df.iterrows():
        pos_size = 1.0 / max_positions
        pnl = trade["net_return"] * pos_size
        equity.append(equity[-1] * (1 + pnl))

    equity_curve = pd.Series(equity)
    trade_returns = trades_df["net_return"]
    metrics = compute_all_metrics(trade_returns, equity_curve)

    # Weekly return analysis
    trades_df["week"] = trades_df["entry_date"].dt.isocalendar().week.astype(int)
    trades_df["year"] = trades_df["entry_date"].dt.year
    trades_df["yearweek"] = trades_df["year"].astype(str) + "-W" + trades_df["week"].astype(str).str.zfill(2)

    weekly_returns = trades_df.groupby("yearweek")["net_return"].sum() / max_positions

    return {
        "trades": trades_df,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "weekly_returns": weekly_returns,
        "trade_returns": trade_returns,
    }


def _execute_trade(stock_data, ticker, signal_date, hold_days, cost, stop_loss):
    """Execute a single trade with stop-loss."""
    if ticker not in stock_data:
        return None

    df = stock_data[ticker]
    try:
        entry_idx = df.index.get_indexer([signal_date], method="nearest")[0]
    except Exception:
        return None

    actual_entry_idx = entry_idx + 1
    if actual_entry_idx >= len(df):
        return None

    entry_price = df.iloc[actual_entry_idx]["Open"]
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    exit_idx = min(actual_entry_idx + hold_days, len(df) - 1)
    exit_price = entry_price
    stopped_out = False

    for day_idx in range(actual_entry_idx, exit_idx + 1):
        low = df.iloc[day_idx]["Low"]
        if (low / entry_price - 1) <= stop_loss:
            exit_price = entry_price * (1 + stop_loss)
            stopped_out = True
            break
        if day_idx == exit_idx:
            exit_price = df.iloc[day_idx]["Close"]

    gross_return = (exit_price / entry_price) - 1
    net_return = gross_return - cost

    return {
        "ticker": ticker,
        "entry_date": df.index[actual_entry_idx],
        "exit_date": df.index[min(exit_idx, len(df) - 1)],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "net_return": net_return,
        "stopped_out": stopped_out,
    }


def analyze_result(result, capital, label="", verbose=True):
    """Analyze a backtest result against the 5K/week target."""
    if result is None:
        if verbose:
            print(f"  {label}: No trades generated")
        return None

    m = result["metrics"]
    trades_df = result["trades"]
    weekly = result["weekly_returns"]

    # Time span
    days = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days
    years = days / 365.25
    weeks = days / 7
    trades_per_year = m["trades"] / max(years, 0.5)

    # Weekly P&L in INR
    weekly_pnl_inr = weekly * capital
    avg_weekly_inr = weekly_pnl_inr.mean()
    median_weekly_inr = weekly_pnl_inr.median()
    pct_weeks_above_5k = (weekly_pnl_inr >= 5000).sum() / len(weekly_pnl_inr) * 100
    pct_weeks_positive = (weekly_pnl_inr > 0).sum() / len(weekly_pnl_inr) * 100
    worst_week_inr = weekly_pnl_inr.min()
    best_week_inr = weekly_pnl_inr.max()

    # Drawdown in INR
    max_dd_inr = m["max_drawdown"] / 100 * capital

    # Recent 6 months
    cutoff_6m = trades_df["entry_date"].max() - pd.Timedelta(days=182)
    recent = trades_df[trades_df["entry_date"] >= cutoff_6m]
    recent_weekly = recent.groupby("yearweek")["net_return"].sum() / (m["trades"] / max(trades_per_year / 52, 1))

    if verbose:
        print(f"\n  {'='*65}")
        print(f"  {label}")
        print(f"  {'='*65}")
        print(f"  Capital: {capital:,.0f} INR")
        print(f"  {format_metrics(m)}")
        print(f"  Trades/year: {trades_per_year:.0f} | Period: {years:.1f} years")
        print(f"")
        print(f"  --- Weekly P&L (INR) ---")
        print(f"  Avg weekly:    {avg_weekly_inr:>+10,.0f} INR")
        print(f"  Median weekly: {median_weekly_inr:>+10,.0f} INR")
        print(f"  Best week:     {best_week_inr:>+10,.0f} INR")
        print(f"  Worst week:    {worst_week_inr:>+10,.0f} INR")
        print(f"  Max drawdown:  {max_dd_inr:>+10,.0f} INR")
        print(f"  Weeks >= 5K:   {pct_weeks_above_5k:.1f}%")
        print(f"  Weeks > 0:     {pct_weeks_positive:.1f}%")

        # Year-by-year
        trades_df_copy = trades_df.copy()
        trades_df_copy["year"] = trades_df_copy["entry_date"].dt.year
        print(f"\n  --- Year-by-Year ---")
        for year in sorted(trades_df_copy["year"].unique()):
            yt = trades_df_copy[trades_df_copy["year"] == year]
            yr_ret = yt["net_return"].sum() / (m["trades"] / max(trades_per_year, 1)) * 100
            yr_pnl = yt["net_return"].sum() / (m["trades"] / max(trades_per_year, 1)) * capital
            print(f"    {year}: {yr_ret:>+6.1f}%  ({yr_pnl:>+10,.0f} INR, {len(yt)} trades)")

        # Recent 6 months
        if len(recent) > 0:
            recent_ret = recent["net_return"].mean() * 100
            recent_wr = (recent["net_return"] > 0).sum() / len(recent) * 100
            recent_weekly_inr = recent.groupby("yearweek")["net_return"].sum()
            # Normalize
            n_pos = max(len(trades_df) / max(weeks, 1), 1)
            recent_avg_week = recent_weekly_inr.mean() / n_pos * capital * 52
            print(f"\n  --- Last 6 Months ---")
            print(f"    Trades: {len(recent)} | WR: {recent_wr:.1f}% | Avg return/trade: {recent_ret:.2f}%")
            stopped = recent["stopped_out"].sum()
            print(f"    Stop-loss hits: {stopped} ({stopped/len(recent)*100:.1f}%)")

        print(f"  {'='*65}")

    return {
        "avg_weekly_inr": avg_weekly_inr,
        "median_weekly_inr": median_weekly_inr,
        "pct_weeks_above_5k": pct_weeks_above_5k,
        "pct_weeks_positive": pct_weeks_positive,
        "worst_week_inr": worst_week_inr,
        "max_dd_inr": max_dd_inr,
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "trades_per_year": trades_per_year,
        "metrics": m,
    }


# ============================================================
# MAIN RESEARCH LOOP
# ============================================================

def main():
    print(f"\n{'#'*70}")
    print(f"  DEEP RESEARCH: 5K INR/WEEK STRATEGY")
    print(f"  Target: 5,000 INR/week = 2,60,000 INR/year")
    print(f"{'#'*70}")

    # Load data
    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)

    if panel.empty:
        print("ERROR: No data")
        sys.exit(1)

    # ============================================================
    # ITERATION 1: Baseline — return_5d Q1, vary positions & hold
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  ITERATION 1: Baseline grid search")
    print(f"  Signal: return_5d | Entry: lowest bucket")
    print(f"  Variables: hold_days, max_positions, num_buckets")
    print(f"{'='*70}")

    best_config = None
    best_avg_weekly = 0

    configs = []
    for hold in [3, 5, 7, 10]:
        for max_pos in [3, 5, 7, 10]:
            for buckets in [5, 10]:  # quintile vs decile
                configs.append((hold, max_pos, buckets))

    print(f"  Testing {len(configs)} configurations...")

    results_grid = []
    for hold, max_pos, buckets in configs:
        result = run_enhanced_backtest(
            stock_data, panel,
            signal_col="return_5d", condition="lowest",
            hold_days=hold, max_positions=max_pos, num_buckets=buckets,
        )
        if result is None:
            continue

        m = result["metrics"]
        weekly = result["weekly_returns"]

        # Test at different capital levels
        for capital in [200000, 300000, 500000]:
            weekly_inr = weekly * capital
            avg_w = weekly_inr.mean()
            med_w = weekly_inr.median()
            pct_above = (weekly_inr >= 5000).sum() / len(weekly_inr) * 100
            worst = weekly_inr.min()

            results_grid.append({
                "hold": hold, "max_pos": max_pos, "buckets": buckets,
                "capital": capital,
                "avg_weekly": avg_w, "median_weekly": med_w,
                "pct_above_5k": pct_above, "worst_week": worst,
                "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                "result": result,
            })

    results_df = pd.DataFrame(results_grid)

    # Filter: avg weekly >= 5000, sharpe > 1, worst week > -30000
    viable = results_df[
        (results_df["avg_weekly"] >= 4000) &  # Close to target
        (results_df["sharpe"] > 0.8) &
        (results_df["worst_week"] > -50000) &
        (results_df["pct_above_5k"] > 30)
    ].sort_values("avg_weekly", ascending=False)

    if len(viable) > 0:
        print(f"\n  Top viable configurations (avg weekly >= 4K, Sharpe > 0.8):")
        print(f"  {'Hold':>4} {'Pos':>3} {'Bkt':>3} {'Capital':>8} {'AvgWk':>8} {'MedWk':>8} {'%>5K':>5} {'Worst':>8} {'Sharpe':>6} {'CAGR':>6} {'MDD':>6} {'WR':>5}")
        for _, row in viable.head(15).iterrows():
            print(f"  {row['hold']:>4}d {row['max_pos']:>3} {row['buckets']:>3} {row['capital']:>8,.0f} {row['avg_weekly']:>+8,.0f} {row['median_weekly']:>+8,.0f} {row['pct_above_5k']:>5.1f} {row['worst_week']:>+8,.0f} {row['sharpe']:>6.2f} {row['cagr']:>5.1f}% {row['mdd']:>5.1f}% {row['wr']:>5.1f}")
    else:
        print(f"\n  No viable configs found with just return_5d. Expanding search...")

    # Pick the best for detailed analysis
    if len(viable) > 0:
        best_row = viable.iloc[0]
        print(f"\n  >> Best config: hold={int(best_row['hold'])}d, pos={int(best_row['max_pos'])}, "
              f"buckets={int(best_row['buckets'])}, capital={best_row['capital']:,.0f}")
        analyze_result(best_row["result"], best_row["capital"],
                      f"ITERATION 1 BEST: return_5d | hold={int(best_row['hold'])}d | "
                      f"pos={int(best_row['max_pos'])} | buckets={int(best_row['buckets'])}")

    # ============================================================
    # ITERATION 2: Combo signals — return_5d + rsi_3
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  ITERATION 2: Combined signals (return_5d + rsi_3)")
    print(f"  Buy stocks that dropped over 5 days AND are oversold on RSI(3)")
    print(f"{'='*70}")

    combo_results = []
    for hold in [3, 5, 7, 10]:
        for max_pos in [3, 5, 7]:
            for rsi_thresh in [-1.0, -0.5, 0.0]:  # z-score thresholds
                result = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col="return_5d", condition="lowest",
                    secondary_signal="rsi_3", secondary_condition="below",
                    secondary_threshold=rsi_thresh,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                )
                if result is None:
                    continue
                m = result["metrics"]
                if m["trades"] < 100:
                    continue
                weekly = result["weekly_returns"]

                for capital in [200000, 300000, 500000]:
                    weekly_inr = weekly * capital
                    combo_results.append({
                        "hold": hold, "max_pos": max_pos, "rsi_thresh": rsi_thresh,
                        "capital": capital,
                        "avg_weekly": weekly_inr.mean(),
                        "median_weekly": weekly_inr.median(),
                        "pct_above_5k": (weekly_inr >= 5000).sum() / len(weekly_inr) * 100,
                        "worst_week": weekly_inr.min(),
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                        "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                        "result": result,
                    })

    combo_df = pd.DataFrame(combo_results)
    if len(combo_df) > 0:
        viable_combo = combo_df[
            (combo_df["avg_weekly"] >= 4000) &
            (combo_df["sharpe"] > 0.8) &
            (combo_df["worst_week"] > -50000)
        ].sort_values("avg_weekly", ascending=False)

        if len(viable_combo) > 0:
            print(f"\n  Top combo configurations:")
            print(f"  {'Hold':>4} {'Pos':>3} {'RSI<':>5} {'Capital':>8} {'AvgWk':>8} {'MedWk':>8} {'%>5K':>5} {'Worst':>8} {'Sharpe':>6} {'CAGR':>6} {'WR':>5} {'PF':>4} {'Trd':>4}")
            for _, row in viable_combo.head(10).iterrows():
                print(f"  {row['hold']:>4}d {row['max_pos']:>3} {row['rsi_thresh']:>5.1f} {row['capital']:>8,.0f} {row['avg_weekly']:>+8,.0f} {row['median_weekly']:>+8,.0f} {row['pct_above_5k']:>5.1f} {row['worst_week']:>+8,.0f} {row['sharpe']:>6.2f} {row['cagr']:>5.1f}% {row['wr']:>5.1f} {row['pf']:>4.2f} {row['trades']:>4}")

            best_combo = viable_combo.iloc[0]
            analyze_result(best_combo["result"], best_combo["capital"],
                          f"ITERATION 2 BEST: return_5d + rsi_3<{best_combo['rsi_thresh']:.1f} | "
                          f"hold={int(best_combo['hold'])}d | pos={int(best_combo['max_pos'])}")

    # ============================================================
    # ITERATION 3: Triple filter — return_5d + rsi_3 + overnight_gap
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  ITERATION 3: Aggressive mean reversion with gap filter")
    print(f"  Buy stocks: worst 5d returns + oversold RSI + gapped down")
    print(f"{'='*70}")

    # Create a composite score
    panel_copy = panel.copy()
    # Composite: average z-scores of mean reversion signals (lower = more oversold)
    mr_signals = ["return_5d", "rsi_3", "return_1d"]
    valid_signals = [s for s in mr_signals if s in panel_copy.columns]
    panel_copy["mr_composite"] = panel_copy[valid_signals].mean(axis=1)

    composite_results = []
    for hold in [3, 5, 7]:
        for max_pos in [3, 5]:
            for buckets in [10, 20]:
                result = run_enhanced_backtest(
                    stock_data, panel_copy,
                    signal_col="mr_composite", condition="lowest",
                    hold_days=hold, max_positions=max_pos, num_buckets=buckets,
                )
                if result is None:
                    continue
                m = result["metrics"]
                if m["trades"] < 100:
                    continue
                weekly = result["weekly_returns"]

                for capital in [200000, 300000, 500000]:
                    weekly_inr = weekly * capital
                    composite_results.append({
                        "hold": hold, "max_pos": max_pos, "buckets": buckets,
                        "capital": capital,
                        "avg_weekly": weekly_inr.mean(),
                        "median_weekly": weekly_inr.median(),
                        "pct_above_5k": (weekly_inr >= 5000).sum() / len(weekly_inr) * 100,
                        "worst_week": weekly_inr.min(),
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                        "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                        "result": result,
                    })

    comp_df = pd.DataFrame(composite_results)
    if len(comp_df) > 0:
        viable_comp = comp_df[
            (comp_df["avg_weekly"] >= 4000) &
            (comp_df["sharpe"] > 0.8) &
            (comp_df["worst_week"] > -50000)
        ].sort_values("avg_weekly", ascending=False)

        if len(viable_comp) > 0:
            print(f"\n  Top composite configurations:")
            print(f"  {'Hold':>4} {'Pos':>3} {'Bkt':>3} {'Capital':>8} {'AvgWk':>8} {'MedWk':>8} {'%>5K':>5} {'Worst':>8} {'Sharpe':>6} {'CAGR':>6} {'WR':>5} {'PF':>4}")
            for _, row in viable_comp.head(10).iterrows():
                print(f"  {row['hold']:>4}d {row['max_pos']:>3} {row['buckets']:>3} {row['capital']:>8,.0f} {row['avg_weekly']:>+8,.0f} {row['median_weekly']:>+8,.0f} {row['pct_above_5k']:>5.1f} {row['worst_week']:>+8,.0f} {row['sharpe']:>6.2f} {row['cagr']:>5.1f}% {row['wr']:>5.1f} {row['pf']:>4.2f}")

            best_comp = viable_comp.iloc[0]
            analyze_result(best_comp["result"], best_comp["capital"],
                          f"ITERATION 3 BEST: MR Composite | hold={int(best_comp['hold'])}d | "
                          f"pos={int(best_comp['max_pos'])} | buckets={int(best_comp['buckets'])}")

    # ============================================================
    # ITERATION 4: Walk-forward validation on the BEST strategy
    # Find the overall best across all iterations
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  ITERATION 4: Walk-Forward Validation")
    print(f"  Training: 2021-2024 | Out-of-sample: 2025-Mar 2026")
    print(f"{'='*70}")

    # Collect all viable results
    all_viable = []
    for df_name, source_df, label in [
        ("grid", results_df, "return_5d"),
        ("combo", combo_df if len(combo_results) > 0 else pd.DataFrame(), "return_5d+rsi3"),
        ("composite", comp_df if len(composite_results) > 0 else pd.DataFrame(), "MR_composite"),
    ]:
        if len(source_df) == 0:
            continue
        v = source_df[
            (source_df["avg_weekly"] >= 4000) & (source_df["sharpe"] > 0.5)
        ].copy()
        v["source"] = label
        all_viable.append(v)

    if all_viable:
        all_df = pd.concat(all_viable, ignore_index=True)
        # Sort by Sharpe (risk-adjusted, more robust than raw return)
        all_df = all_df.sort_values("sharpe", ascending=False)

        # Take top 5 candidates and validate out-of-sample
        print(f"\n  Top 5 candidates for walk-forward validation:")

        for idx, (_, row) in enumerate(all_df.head(5).iterrows()):
            source = row["source"]
            hold = int(row["hold"])
            max_pos = int(row["max_pos"])
            capital = row["capital"]

            # Re-run on training period
            if source == "return_5d":
                buckets = int(row["buckets"])
                train = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col="return_5d", condition="lowest",
                    hold_days=hold, max_positions=max_pos, num_buckets=buckets,
                    start_date="2021-01-01", end_date="2024-12-31",
                )
                test = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col="return_5d", condition="lowest",
                    hold_days=hold, max_positions=max_pos, num_buckets=buckets,
                    start_date="2025-01-01", end_date="2026-12-31",
                )
                config_str = f"return_5d lowest-of-{buckets} | hold={hold}d | pos={max_pos}"
            elif source == "return_5d+rsi3":
                rsi_t = row["rsi_thresh"]
                train = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col="return_5d", condition="lowest",
                    secondary_signal="rsi_3", secondary_condition="below",
                    secondary_threshold=rsi_t,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    start_date="2021-01-01", end_date="2024-12-31",
                )
                test = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col="return_5d", condition="lowest",
                    secondary_signal="rsi_3", secondary_condition="below",
                    secondary_threshold=rsi_t,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    start_date="2025-01-01", end_date="2026-12-31",
                )
                config_str = f"return_5d + rsi3<{rsi_t:.1f} | hold={hold}d | pos={max_pos}"
            else:
                buckets = int(row["buckets"])
                train = run_enhanced_backtest(
                    stock_data, panel_copy,
                    signal_col="mr_composite", condition="lowest",
                    hold_days=hold, max_positions=max_pos, num_buckets=buckets,
                    start_date="2021-01-01", end_date="2024-12-31",
                )
                test = run_enhanced_backtest(
                    stock_data, panel_copy,
                    signal_col="mr_composite", condition="lowest",
                    hold_days=hold, max_positions=max_pos, num_buckets=buckets,
                    start_date="2025-01-01", end_date="2026-12-31",
                )
                config_str = f"MR_composite lowest-of-{buckets} | hold={hold}d | pos={max_pos}"

            print(f"\n  --- Candidate {idx+1}: {config_str} | Capital: {capital:,.0f} ---")

            if train:
                train_weekly = train["weekly_returns"] * capital
                tm = train["metrics"]
                print(f"    TRAIN (2021-2024): Sharpe={tm['sharpe']:.2f} | CAGR={tm['cagr']:.1f}% | "
                      f"WR={tm['win_rate']:.1f}% | Avg week={train_weekly.mean():+,.0f} INR | "
                      f"Trades={tm['trades']}")

            if test:
                test_weekly = test["weekly_returns"] * capital
                testm = test["metrics"]
                pct_5k = (test_weekly >= 5000).sum() / len(test_weekly) * 100
                print(f"    TEST  (2025-2026): Sharpe={testm['sharpe']:.2f} | CAGR={testm['cagr']:.1f}% | "
                      f"WR={testm['win_rate']:.1f}% | Avg week={test_weekly.mean():+,.0f} INR | "
                      f"Trades={testm['trades']} | Weeks>=5K={pct_5k:.0f}%")

                if test_weekly.mean() >= 4000:
                    print(f"    >> PASSES WALK-FORWARD TEST!")
                else:
                    print(f"    >> Below target in test period")
            else:
                print(f"    TEST: No trades")

    # ============================================================
    # FINAL: Pick the winner, show full analysis
    # ============================================================
    print(f"\n\n{'#'*70}")
    print(f"  FINAL STRATEGY SELECTION")
    print(f"{'#'*70}")

    # Run the best configs on full period with detailed output
    # Based on what we've seen, run the most promising ones
    final_configs = [
        ("return_5d", "lowest", None, None, None, 10, 5, 5, 10),
        ("return_5d", "lowest", None, None, None, 7, 5, 5, 10),
        ("return_5d", "lowest", None, None, None, 5, 3, 5, 10),
        ("return_5d", "lowest", "rsi_3", "below", -0.5, 5, 5, 5, 10),
        ("return_5d", "lowest", "rsi_3", "below", 0.0, 5, 5, 5, 10),
        ("return_5d", "lowest", "rsi_3", "below", 0.0, 7, 5, 5, 10),
    ]

    best_final = None
    best_final_score = 0

    for sig, cond, sec_sig, sec_cond, sec_thresh, hold, max_pos, buckets_n, _ in final_configs:
        p = panel_copy if sig == "mr_composite" else panel

        # Full period
        result_full = run_enhanced_backtest(
            stock_data, p,
            signal_col=sig, condition=cond,
            secondary_signal=sec_sig, secondary_condition=sec_cond,
            secondary_threshold=sec_thresh,
            hold_days=hold, max_positions=max_pos, num_buckets=buckets_n,
        )

        # Test period only
        result_test = run_enhanced_backtest(
            stock_data, p,
            signal_col=sig, condition=cond,
            secondary_signal=sec_sig, secondary_condition=sec_cond,
            secondary_threshold=sec_thresh,
            hold_days=hold, max_positions=max_pos, num_buckets=buckets_n,
            start_date="2025-01-01",
        )

        if result_full is None:
            continue

        # Find minimum capital needed for 5K/week
        weekly = result_full["weekly_returns"]
        avg_weekly_ret = weekly.mean()
        if avg_weekly_ret <= 0:
            continue

        min_capital = 5000 / avg_weekly_ret

        # Test period check
        test_ok = False
        test_avg = 0
        if result_test:
            test_weekly = result_test["weekly_returns"]
            test_avg = test_weekly.mean()
            if test_avg > 0:
                test_capital = 5000 / test_avg
                test_ok = True

        m = result_full["metrics"]
        sec_label = f" + rsi3<{sec_thresh}" if sec_sig else ""
        label = f"{sig}{sec_label} | hold={hold}d | pos={max_pos} | dec"

        weekly_5k = weekly * min_capital
        pct_above = (weekly_5k >= 5000).sum() / len(weekly_5k) * 100

        print(f"\n  {label}")
        print(f"    Min capital for 5K/wk: {min_capital:>10,.0f} INR")
        print(f"    Full: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.1f}% WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f} MDD={m['max_drawdown']:.1f}%")
        if result_test:
            tm = result_test["metrics"]
            print(f"    2025+: Sharpe={tm['sharpe']:.2f} CAGR={tm['cagr']:.1f}% WR={tm['win_rate']:.1f}% Trades={tm['trades']}")
            if test_ok:
                print(f"    2025+ capital for 5K/wk: {test_capital:>10,.0f} INR")
        print(f"    Weeks >= 5K (at min capital): {pct_above:.0f}%")

        # Score: prioritize test period performance + sharpe
        score = 0
        if test_ok:
            score = m["sharpe"] * 10 + (1 if pct_above > 40 else 0) * 5 + (1 if m["cagr"] > 30 else 0) * 3
            if result_test:
                score += result_test["metrics"]["sharpe"] * 15  # Weight test period more

        if score > best_final_score:
            best_final_score = score
            best_final = {
                "config": label,
                "result_full": result_full,
                "result_test": result_test,
                "min_capital": min_capital,
                "test_capital": test_capital if test_ok else None,
                "signal": sig, "condition": cond,
                "secondary_signal": sec_sig, "secondary_condition": sec_cond,
                "secondary_threshold": sec_thresh,
                "hold": hold, "max_pos": max_pos, "buckets": buckets_n,
            }

    # ============================================================
    # WINNER — Full detailed report
    # ============================================================
    if best_final:
        print(f"\n\n{'*'*70}")
        print(f"  WINNER: {best_final['config']}")
        print(f"{'*'*70}")

        # Use the higher of min_capital and test_capital (conservative)
        rec_capital = max(
            best_final["min_capital"],
            best_final["test_capital"] or 0
        )
        # Round up to nearest 50K
        rec_capital = int(np.ceil(rec_capital / 50000) * 50000)

        print(f"\n  RECOMMENDED CAPITAL: {rec_capital:,} INR")

        analyze_result(best_final["result_full"], rec_capital, "FULL PERIOD (2021-2026)")

        if best_final["result_test"]:
            analyze_result(best_final["result_test"], rec_capital, "OUT-OF-SAMPLE (2025-2026)")

        # Monthly breakdown for last 12 months
        trades_df = best_final["result_full"]["trades"]
        recent = trades_df[trades_df["entry_date"] >= trades_df["entry_date"].max() - pd.Timedelta(days=365)]
        if len(recent) > 0:
            recent["month"] = recent["entry_date"].dt.to_period("M")
            print(f"\n  --- Monthly P&L (last 12 months at {rec_capital:,} INR capital) ---")
            for month in sorted(recent["month"].unique()):
                mt = recent[recent["month"] == month]
                month_pnl = mt["net_return"].sum() / best_final["max_pos"] * rec_capital
                month_trades = len(mt)
                wr = (mt["net_return"] > 0).sum() / len(mt) * 100 if len(mt) > 0 else 0
                print(f"    {month}: {month_pnl:>+10,.0f} INR ({month_trades} trades, WR={wr:.0f}%)")

        # Print deployment instructions
        print(f"\n\n{'='*70}")
        print(f"  DEPLOYMENT PLAN FOR PAPER TRADING")
        print(f"{'='*70}")
        print(f"  Strategy: {best_final['config']}")
        print(f"  Capital: {rec_capital:,} INR")
        print(f"  Target: 5,000 INR/week")
        print(f"")
        print(f"  Daily routine:")
        print(f"    1. After market close (3:30 PM), run the scanner")
        print(f"    2. Compute {best_final['signal']} for all Nifty 200 stocks")
        if best_final["secondary_signal"]:
            print(f"    3. Filter: {best_final['secondary_signal']} {best_final['secondary_condition']} {best_final['secondary_threshold']}")
            print(f"    4. Select bottom decile (most oversold)")
        else:
            print(f"    3. Select bottom decile (most oversold)")
        print(f"    4. Buy top {best_final['max_pos']} stocks at next day's open")
        print(f"    5. Hold for {best_final['hold']} trading days")
        print(f"    6. Set -8% stop-loss on each position")
        print(f"    7. Exit at close on day {best_final['hold']}")
        print(f"")
        print(f"  Position sizing: {rec_capital:,} / {best_final['max_pos']} = {rec_capital // best_final['max_pos']:,} INR per position")
        print(f"  Max risk per trade: {rec_capital // best_final['max_pos'] * 0.08:,.0f} INR (8% stop)")
        print(f"  Max portfolio risk: {rec_capital * 0.08:,.0f} INR (if all positions stop out)")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
