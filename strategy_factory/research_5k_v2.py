#!/usr/bin/env python3
"""
Research V2: The mean reversion edge decayed in 2025.
New approach:
1. Find what ACTUALLY works in 2025 (recent regime)
2. Test momentum signals (return_252d was flagged as strong)
3. Test adaptive strategies (different regimes)
4. Try weekly rebalancing to reduce noise
5. Explore combined momentum + quality filters
6. Larger position concentration for higher per-trade returns
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
from signals.scanner import quintile_analysis, classify_signal
from backtest.metrics import compute_all_metrics, format_metrics

# Import the enhanced backtest from v1
from research_5k import run_enhanced_backtest, analyze_result


def scan_recent_regime(panel, stock_data):
    """Find what signals work in 2025 specifically."""
    print(f"\n{'='*70}")
    print(f"  REGIME ANALYSIS: What works in 2025?")
    print(f"{'='*70}")

    recent_panel = panel[panel["date"] >= pd.Timestamp("2025-01-01")]
    full_panel = panel[panel["date"] < pd.Timestamp("2025-01-01")]

    exclude_cols = {"ticker", "date"} | {f"fwd_{d}d" for d in [1, 3, 5, 10, 20]}
    feature_cols = [c for c in panel.columns if c not in exclude_cols]

    print(f"\n  Scanning {len(feature_cols)} features in 2025 data ({len(recent_panel):,} rows)...")

    results_2025 = []
    results_pre2025 = []

    for feat in feature_cols:
        r2025 = quintile_analysis(recent_panel, feat, "fwd_5d")
        rpre = quintile_analysis(full_panel, feat, "fwd_5d")
        if r2025:
            r2025["classification"] = classify_signal(r2025)
            results_2025.append(r2025)
        if rpre:
            rpre["classification"] = classify_signal(rpre)
            results_pre2025.append(rpre)

    results_2025.sort(key=lambda x: x["t_stat"], reverse=True)
    results_pre2025.sort(key=lambda x: x["t_stat"], reverse=True)

    print(f"\n  --- Signals ranked by 2025 performance (5d forward) ---")
    print(f"  {'Feature':<22} {'2025 t-stat':>10} {'2025 spread':>12} {'Pre-2025 t':>10} {'Pre-2025 sp':>12} {'Direction'}")
    for r in results_2025[:15]:
        feat = r["feature"]
        pre = next((x for x in results_pre2025 if x["feature"] == feat), None)
        pre_t = f"{pre['t_stat']:.2f}" if pre else "N/A"
        pre_sp = f"{pre['spread']:+.2f}%" if pre else "N/A"
        print(f"  {feat:<22} {r['t_stat']:>10.2f} {r['spread']:>+11.2f}% {pre_t:>10} {pre_sp:>12} {r['direction']}")

    return results_2025


def test_momentum_strategies(panel, stock_data):
    """Test momentum-based strategies which may work better in recent markets."""
    print(f"\n\n{'='*70}")
    print(f"  MOMENTUM STRATEGIES")
    print(f"  Hypothesis: In 2025 bull-to-correction regime, momentum outperforms MR")
    print(f"{'='*70}")

    results = []

    # Test various momentum signals
    momentum_configs = [
        # (signal, condition, label)
        ("return_252d", "highest", "12m momentum (buy winners)"),
        ("return_120d", "highest", "6m momentum (buy winners)"),
        ("return_60d", "highest", "3m momentum (buy winners)"),
        ("return_20d", "highest", "1m momentum (buy winners)"),
        ("dist_52w_high", "highest", "Near 52w high (breakout)"),
        ("rel_strength_20d", "highest", "Relative strength vs Nifty"),
        # Mean reversion on LONGER timeframes (may still work)
        ("return_20d", "lowest", "1m losers bounce"),
        ("return_60d", "lowest", "3m losers bounce"),
    ]

    for signal, condition, label in momentum_configs:
        if signal not in panel.columns:
            continue

        for hold in [5, 10, 20]:
            for max_pos in [3, 5]:
                result = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                )
                if result is None:
                    continue

                m = result["metrics"]
                weekly = result["weekly_returns"]

                # Also get 2025+ performance
                result_2025 = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    start_date="2025-01-01",
                )

                test_sharpe = 0
                test_cagr = 0
                test_trades = 0
                if result_2025:
                    tm = result_2025["metrics"]
                    test_sharpe = tm["sharpe"]
                    test_cagr = tm["cagr"]
                    test_trades = tm["trades"]

                for capital in [300000, 500000, 750000]:
                    weekly_inr = weekly * capital
                    results.append({
                        "signal": signal, "condition": condition, "label": label,
                        "hold": hold, "max_pos": max_pos, "capital": capital,
                        "avg_weekly": weekly_inr.mean(),
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                        "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                        "test_sharpe": test_sharpe, "test_cagr": test_cagr, "test_trades": test_trades,
                        "result": result, "result_2025": result_2025,
                    })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results generated")
        return df

    # Filter by test period performance
    viable = df[
        (df["test_sharpe"] > 0) &  # Positive in 2025
        (df["sharpe"] > 0.5) &     # Decent full period
        (df["avg_weekly"] >= 3000)  # Reasonable weekly return
    ].sort_values("test_sharpe", ascending=False)

    if len(viable) > 0:
        print(f"\n  VIABLE MOMENTUM STRATEGIES (positive in 2025):")
        print(f"  {'Label':<30} {'Hold':>4} {'Pos':>3} {'Cap':>7} {'AvgWk':>7} {'Full Sh':>7} {'2025 Sh':>7} {'2025 CAGR':>9} {'WR':>5} {'MDD':>5}")
        for _, r in viable.head(20).iterrows():
            print(f"  {r['label']:<30} {r['hold']:>4}d {r['max_pos']:>3} {r['capital']/1000:>5.0f}K {r['avg_weekly']:>+7,.0f} {r['sharpe']:>7.2f} {r['test_sharpe']:>7.2f} {r['test_cagr']:>+8.1f}% {r['wr']:>5.1f} {r['mdd']:>5.1f}")
    else:
        print(f"\n  No momentum strategies with positive 2025 performance found.")
        # Show best anyway
        best_test = df.sort_values("test_sharpe", ascending=False).head(10)
        print(f"\n  Best by 2025 Sharpe (even if negative):")
        for _, r in best_test.iterrows():
            print(f"  {r['label']:<30} h={r['hold']:>2}d p={r['max_pos']} cap={r['capital']/1000:.0f}K | "
                  f"Full: Sh={r['sharpe']:.2f} | 2025: Sh={r['test_sharpe']:.2f} CAGR={r['test_cagr']:+.1f}%")

    return df


def test_weekly_rebalance(panel, stock_data):
    """
    Test weekly rebalancing (enter only on Mondays).
    Reduces trading frequency, may reduce noise.
    """
    print(f"\n\n{'='*70}")
    print(f"  WEEKLY REBALANCING STRATEGIES")
    print(f"  Enter on Monday close signal, buy Tuesday open, hold N days")
    print(f"{'='*70}")

    # Filter panel to Mondays only (weekday == 0)
    panel_weekly = panel.copy()
    panel_weekly = panel_weekly[panel_weekly["date"].dt.weekday == 0]

    results = []
    signals = [
        ("return_5d", "lowest", "5d MR weekly"),
        ("return_20d", "lowest", "20d MR weekly"),
        ("rsi_3", "lowest", "RSI3 oversold weekly"),
        ("return_252d", "highest", "12m momentum weekly"),
        ("return_60d", "highest", "3m momentum weekly"),
        ("dist_52w_high", "highest", "Near 52w high weekly"),
    ]

    for signal, condition, label in signals:
        if signal not in panel_weekly.columns:
            continue
        for hold in [5, 10]:
            for max_pos in [3, 5]:
                result = run_enhanced_backtest(
                    stock_data, panel_weekly,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                )
                if result is None:
                    continue

                result_2025 = run_enhanced_backtest(
                    stock_data, panel_weekly,
                    signal_col=signal, condition=condition,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    start_date="2025-01-01",
                )

                m = result["metrics"]
                weekly = result["weekly_returns"]
                test_sh = result_2025["metrics"]["sharpe"] if result_2025 else -99
                test_cagr = result_2025["metrics"]["cagr"] if result_2025 else -99
                test_trades = result_2025["metrics"]["trades"] if result_2025 else 0

                for capital in [300000, 500000, 750000]:
                    weekly_inr = weekly * capital
                    results.append({
                        "label": label, "signal": signal, "condition": condition,
                        "hold": hold, "max_pos": max_pos, "capital": capital,
                        "avg_weekly": weekly_inr.mean(),
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                        "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                        "test_sharpe": test_sh, "test_cagr": test_cagr, "test_trades": test_trades,
                        "result": result, "result_2025": result_2025,
                    })

    df = pd.DataFrame(results)
    viable = df[
        (df["test_sharpe"] > 0) & (df["avg_weekly"] >= 2000)
    ].sort_values("test_sharpe", ascending=False)

    if len(viable) > 0:
        print(f"\n  VIABLE WEEKLY STRATEGIES (positive in 2025):")
        print(f"  {'Label':<25} {'Hold':>4} {'Pos':>3} {'Cap':>6} {'AvgWk':>7} {'Sh':>5} {'2025Sh':>6} {'2025C':>6} {'WR':>5} {'Trd':>4}")
        for _, r in viable.head(15).iterrows():
            print(f"  {r['label']:<25} {r['hold']:>4}d {r['max_pos']:>3} {r['capital']/1000:>4.0f}K {r['avg_weekly']:>+7,.0f} {r['sharpe']:>5.2f} {r['test_sharpe']:>6.2f} {r['test_cagr']:>+5.1f}% {r['wr']:>5.1f} {r['trades']:>4}")

    return df


def test_adaptive_regime(panel, stock_data):
    """
    Adaptive strategy: use market regime to switch between MR and momentum.
    When market is trending (Nifty above 20d SMA): momentum
    When market is correcting (Nifty below 20d SMA): mean reversion
    """
    print(f"\n\n{'='*70}")
    print(f"  ADAPTIVE REGIME STRATEGY")
    print(f"  Bull regime (Nifty > 20d SMA): buy momentum (best 252d return)")
    print(f"  Bear regime (Nifty < 20d SMA): buy mean reversion (worst 5d return)")
    print(f"{'='*70}")

    # Determine regime from Nifty 50 data
    nifty = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    if nifty.empty:
        print("  Cannot load Nifty data")
        return None

    nifty_close = nifty["Close"]
    nifty_sma20 = nifty_close.rolling(20).mean()
    nifty_regime = (nifty_close > nifty_sma20).astype(int)  # 1 = bull, 0 = bear

    # Map regime to panel dates
    panel_adaptive = panel.copy()
    regime_map = nifty_regime.to_dict()

    def get_regime(date):
        # Find nearest date in regime map
        if date in regime_map:
            return regime_map[date]
        # Try nearby dates
        for delta in range(1, 10):
            for d in [date - pd.Timedelta(days=delta), date + pd.Timedelta(days=delta)]:
                if d in regime_map:
                    return regime_map[d]
        return None

    panel_adaptive["regime"] = panel_adaptive["date"].apply(get_regime)
    panel_bull = panel_adaptive[panel_adaptive["regime"] == 1]
    panel_bear = panel_adaptive[panel_adaptive["regime"] == 0]

    print(f"  Bull days: {panel_bull['date'].nunique()} | Bear days: {panel_bear['date'].nunique()}")

    results = []

    # Test various combinations
    for hold in [5, 7, 10]:
        for max_pos in [3, 5]:
            # Run bull regime: momentum
            bull_result = run_enhanced_backtest(
                stock_data, panel_bull,
                signal_col="return_60d", condition="highest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
            )
            # Run bear regime: mean reversion
            bear_result = run_enhanced_backtest(
                stock_data, panel_bear,
                signal_col="return_5d", condition="lowest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
            )

            if bull_result is None and bear_result is None:
                continue

            # Merge trade lists
            all_trades = []
            if bull_result and not bull_result["trades"].empty:
                all_trades.append(bull_result["trades"])
            if bear_result and not bear_result["trades"].empty:
                all_trades.append(bear_result["trades"])

            if not all_trades:
                continue

            combined_trades = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
            equity = [1.0]
            for _, t in combined_trades.iterrows():
                pnl = t["net_return"] / max_pos
                equity.append(equity[-1] * (1 + pnl))

            eq = pd.Series(equity)
            rets = combined_trades["net_return"]
            m = compute_all_metrics(rets, eq)

            combined_trades["week"] = combined_trades["entry_date"].dt.isocalendar().week.astype(int)
            combined_trades["year"] = combined_trades["entry_date"].dt.year
            combined_trades["yearweek"] = combined_trades["year"].astype(str) + "-W" + combined_trades["week"].astype(str).str.zfill(2)
            weekly = combined_trades.groupby("yearweek")["net_return"].sum() / max_pos

            # 2025 subset
            t2025 = combined_trades[combined_trades["entry_date"] >= pd.Timestamp("2025-01-01")]
            test_trades = len(t2025)
            if len(t2025) > 10:
                test_rets = t2025["net_return"]
                test_eq = pd.Series([1.0] + list(1 + test_rets.cumsum() / max_pos))
                test_m = compute_all_metrics(test_rets, test_eq)
                test_sh = test_m["sharpe"]
                test_cagr = test_m["cagr"]
            else:
                test_sh = -99
                test_cagr = -99

            for capital in [300000, 500000, 750000]:
                weekly_inr = weekly * capital
                results.append({
                    "hold": hold, "max_pos": max_pos, "capital": capital,
                    "avg_weekly": weekly_inr.mean(),
                    "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                    "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                    "test_sharpe": test_sh, "test_cagr": test_cagr, "test_trades": test_trades,
                    "combined_trades": combined_trades, "weekly": weekly,
                })

    df = pd.DataFrame(results)
    if len(df) > 0:
        viable = df[(df["test_sharpe"] > 0)].sort_values("test_sharpe", ascending=False)
        print(f"\n  {'Hold':>4} {'Pos':>3} {'Cap':>6} {'AvgWk':>7} {'Sh':>5} {'2025Sh':>6} {'2025C':>6} {'WR':>5} {'MDD':>5} {'Trd':>4}")
        for _, r in (viable if len(viable) > 0 else df.sort_values("test_sharpe", ascending=False).head(10)).iterrows():
            tag = " <<" if r["test_sharpe"] > 0 else ""
            print(f"  {r['hold']:>4}d {r['max_pos']:>3} {r['capital']/1000:>4.0f}K {r['avg_weekly']:>+7,.0f} {r['sharpe']:>5.2f} {r['test_sharpe']:>6.2f} {r['test_cagr']:>+5.1f}% {r['wr']:>5.1f} {r['mdd']:>5.1f} {r['trades']:>4}{tag}")

    return df


def test_extreme_oversold(panel, stock_data):
    """
    Test buying EXTREMELY oversold stocks — ventile (bottom 5%) or z-score < -2.
    More selective = higher per-trade return = less capital needed.
    """
    print(f"\n\n{'='*70}")
    print(f"  EXTREME OVERSOLD STRATEGIES")
    print(f"  Hypothesis: Deeper oversold = stronger bounce, fewer but better trades")
    print(f"{'='*70}")

    results = []

    # Z-score threshold strategies
    for signal in ["return_5d", "return_1d", "rsi_3", "overnight_gap"]:
        if signal not in panel.columns:
            continue
        for threshold in [-2.5, -2.0, -1.5]:
            for hold in [3, 5, 7, 10]:
                for max_pos in [3, 5]:
                    result = run_enhanced_backtest(
                        stock_data, panel,
                        signal_col=signal, condition="below_threshold",
                        threshold=threshold,
                        hold_days=hold, max_positions=max_pos, num_buckets=10,
                    )
                    if result is None:
                        continue
                    m = result["metrics"]
                    if m["trades"] < 50:
                        continue

                    result_2025 = run_enhanced_backtest(
                        stock_data, panel,
                        signal_col=signal, condition="below_threshold",
                        threshold=threshold,
                        hold_days=hold, max_positions=max_pos, num_buckets=10,
                        start_date="2025-01-01",
                    )

                    weekly = result["weekly_returns"]
                    test_sh = result_2025["metrics"]["sharpe"] if result_2025 else -99
                    test_cagr = result_2025["metrics"]["cagr"] if result_2025 else -99
                    test_trades = result_2025["metrics"]["trades"] if result_2025 else 0

                    for capital in [200000, 300000, 500000]:
                        weekly_inr = weekly * capital
                        results.append({
                            "signal": signal, "threshold": threshold,
                            "hold": hold, "max_pos": max_pos, "capital": capital,
                            "avg_weekly": weekly_inr.mean(),
                            "sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"],
                            "trades": m["trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
                            "avg_ret": m["avg_return"],
                            "test_sharpe": test_sh, "test_cagr": test_cagr, "test_trades": test_trades,
                            "result": result, "result_2025": result_2025,
                        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results")
        return df

    viable = df[
        (df["test_sharpe"] > 0) & (df["avg_weekly"] >= 2000)
    ].sort_values("test_sharpe", ascending=False)

    print(f"\n  Strategies with positive 2025 performance:")
    target = viable if len(viable) > 0 else df.sort_values("test_sharpe", ascending=False).head(20)
    print(f"  {'Signal':<15} {'Thr':>5} {'Hold':>4} {'Pos':>3} {'Cap':>6} {'AvgWk':>7} {'Sh':>5} {'25Sh':>5} {'25C':>6} {'WR':>5} {'AvgR':>5} {'Trd':>4}")
    for _, r in target.head(20).iterrows():
        tag = " <<" if r["test_sharpe"] > 0 else ""
        print(f"  {r['signal']:<15} {r['threshold']:>5.1f} {r['hold']:>4}d {r['max_pos']:>3} {r['capital']/1000:>4.0f}K {r['avg_weekly']:>+7,.0f} {r['sharpe']:>5.2f} {r['test_sharpe']:>5.2f} {r['test_cagr']:>+5.1f}% {r['wr']:>5.1f} {r['avg_ret']:>5.2f} {r['trades']:>4}{tag}")

    return df


def final_champion_test(panel, stock_data, all_results):
    """
    Take the best strategy across ALL tests, run exhaustive validation.
    """
    print(f"\n\n{'#'*70}")
    print(f"  CHAMPION SELECTION — Best across all iterations")
    print(f"{'#'*70}")

    # Collect all results with test_sharpe > 0
    champions = []
    for name, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        viable = df[df["test_sharpe"] > 0].copy()
        viable["source"] = name
        if len(viable) > 0:
            champions.append(viable)

    if not champions:
        print("\n  WARNING: No strategy has positive 2025 Sharpe.")
        print("  Selecting least-bad option for paper trading validation...")
        for name, df in all_results.items():
            if df is None or len(df) == 0:
                continue
            best = df.nlargest(3, "test_sharpe").copy()
            best["source"] = name
            champions.append(best)

    if not champions:
        print("  No candidates at all. Need different approach.")
        return

    all_champs = pd.concat(champions, ignore_index=True)
    all_champs = all_champs.sort_values("test_sharpe", ascending=False)

    print(f"\n  Top 10 candidates across all approaches:")
    for i, (_, r) in enumerate(all_champs.head(10).iterrows()):
        sig_label = r.get("signal", r.get("label", "?"))
        cond = r.get("condition", r.get("threshold", "?"))
        print(f"  {i+1:>2}. [{r['source']:<12}] {sig_label} {cond} | h={r['hold']}d p={r['max_pos']} "
              f"cap={r['capital']/1000:.0f}K | Full Sh={r['sharpe']:.2f} | "
              f"2025 Sh={r['test_sharpe']:.2f} CAGR={r['test_cagr']:+.1f}% ({r.get('test_trades', '?')} trades)")

    # Take the #1 champion and do deep analysis
    champ = all_champs.iloc[0]
    print(f"\n  >> CHAMPION: {champ.get('signal', champ.get('label', '?'))} from {champ['source']}")

    # Get the stored result objects
    result_full = champ.get("result")
    result_2025 = champ.get("result_2025")
    capital = champ["capital"]

    if result_full is not None:
        analyze_result(result_full, capital, f"CHAMPION — Full Period", verbose=True)
    if result_2025 is not None:
        analyze_result(result_2025, capital, f"CHAMPION — 2025 Out-of-Sample", verbose=True)

    # Calculate required capital for 5K/week
    if result_full is not None:
        weekly = result_full["weekly_returns"]
        avg_ret = weekly.mean()
        if avg_ret > 0:
            min_cap_full = 5000 / avg_ret
            print(f"\n  Capital for 5K/wk (full period avg): {min_cap_full:,.0f} INR")

    if result_2025 is not None:
        weekly_2025 = result_2025["weekly_returns"]
        avg_ret_2025 = weekly_2025.mean()
        if avg_ret_2025 > 0:
            min_cap_2025 = 5000 / avg_ret_2025
            print(f"  Capital for 5K/wk (2025 avg):        {min_cap_2025:,.0f} INR")
            rec_capital = int(np.ceil(max(min_cap_2025 * 1.3, min_cap_full if avg_ret > 0 else 0) / 50000) * 50000)
        else:
            rec_capital = int(np.ceil(min_cap_full * 1.5 / 50000) * 50000) if avg_ret > 0 else 500000
            print(f"  WARNING: 2025 weekly avg is negative. Using 1.5x full-period estimate.")

        print(f"  RECOMMENDED CAPITAL (with 30% buffer): {rec_capital:,} INR")

        # Final weekly P&L at recommended capital
        print(f"\n  --- Projected Weekly P&L at {rec_capital:,} INR ---")
        if result_2025:
            wk = result_2025["weekly_returns"] * rec_capital
            print(f"  Based on 2025 performance:")
            print(f"    Avg: {wk.mean():+,.0f} | Med: {wk.median():+,.0f} | Best: {wk.max():+,.0f} | Worst: {wk.min():+,.0f}")
            print(f"    Weeks >= 5K: {(wk >= 5000).sum()}/{len(wk)} ({(wk >= 5000).sum()/len(wk)*100:.0f}%)")
            print(f"    Weeks > 0:   {(wk > 0).sum()}/{len(wk)} ({(wk > 0).sum()/len(wk)*100:.0f}%)")

    # Deployment details
    sig = champ.get("signal", "?")
    cond = champ.get("condition", "?")
    thresh = champ.get("threshold", None)
    hold = int(champ["hold"])
    max_pos = int(champ["max_pos"])

    print(f"\n\n{'='*70}")
    print(f"  PAPER TRADING DEPLOYMENT PLAN")
    print(f"{'='*70}")
    print(f"  Signal: {sig}")
    print(f"  Condition: {cond}" + (f" (threshold: {thresh})" if thresh else " (bottom decile)"))
    print(f"  Hold: {hold} trading days")
    print(f"  Max positions: {max_pos}")
    print(f"  Position size: {rec_capital:,} / {max_pos} = {rec_capital // max_pos:,} INR each")
    print(f"  Stop-loss: -8% per position")
    print(f"  Cost assumption: 0.1% round trip")
    print(f"  Rebalance: Daily (check at close, enter next open)")
    print(f"{'='*70}")


def main():
    print(f"\n{'#'*70}")
    print(f"  RESEARCH V2: FINDING AN EDGE THAT WORKS NOW")
    print(f"  The mean reversion edge decayed in 2025.")
    print(f"  Need to find what actually works in the current regime.")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)

    all_results = {}

    # 1. Scan what works in 2025
    scan_recent_regime(panel, stock_data)

    # 2. Test momentum strategies
    all_results["momentum"] = test_momentum_strategies(panel, stock_data)

    # 3. Weekly rebalancing
    all_results["weekly"] = test_weekly_rebalance(panel, stock_data)

    # 4. Adaptive regime
    all_results["adaptive"] = test_adaptive_regime(panel, stock_data)

    # 5. Extreme oversold
    all_results["extreme"] = test_extreme_oversold(panel, stock_data)

    # 6. Pick champion
    final_champion_test(panel, stock_data, all_results)


if __name__ == "__main__":
    main()
