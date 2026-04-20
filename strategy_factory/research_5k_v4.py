#!/usr/bin/env python3
"""
Research V4: Nifty 500 universe — 2.5x more stocks = more opportunities.

Why Nifty 500 should be better:
- Small/mid-caps have higher volatility = bigger moves
- Less institutional coverage = more inefficiency
- More stocks in extreme quintiles = better selection
- Still liquid enough for 1-2L position sizes

Strategy search:
1. Re-run regime scan on Nifty 500 (what works in 2025?)
2. Test champion (3m momentum) on larger universe
3. Test small-cap momentum (may be stronger in smaller stocks)
4. Test mean reversion on small/mid-caps (more extreme dislocations)
5. Final: find minimum capital for 5K/week
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_500, NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from signals.scanner import quintile_analysis, classify_signal
from backtest.metrics import compute_all_metrics, format_metrics
from research_5k import run_enhanced_backtest, analyze_result
from research_5k_v3 import detailed_weekly_analysis


def main():
    print(f"\n{'#'*70}")
    print(f"  RESEARCH V4: NIFTY 500 UNIVERSE")
    print(f"  500 stocks — more opportunities, more inefficiency")
    print(f"{'#'*70}")

    # Download all Nifty 500 data
    stock_data = download_universe(NIFTY_500, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)

    if panel.empty:
        print("ERROR: No data")
        sys.exit(1)

    n_stocks = panel["ticker"].nunique()
    print(f"\n  Universe: {n_stocks} stocks | Panel: {len(panel):,} rows")

    # ============================================================
    # 1. REGIME SCAN: What works in 2025 on Nifty 500?
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  REGIME SCAN: Nifty 500, 2025 vs pre-2025")
    print(f"{'='*70}")

    recent_panel = panel[panel["date"] >= pd.Timestamp("2025-01-01")]
    exclude_cols = {"ticker", "date"} | {f"fwd_{d}d" for d in [1, 3, 5, 10, 20]}
    feature_cols = [c for c in panel.columns if c not in exclude_cols]

    results_2025 = []
    for feat in feature_cols:
        r = quintile_analysis(recent_panel, feat, "fwd_5d")
        if r:
            r["classification"] = classify_signal(r)
            results_2025.append(r)

    results_2025.sort(key=lambda x: x["t_stat"], reverse=True)
    print(f"\n  Top signals in 2025 (Nifty 500):")
    print(f"  {'Feature':<22} {'t-stat':>8} {'Spread':>8} {'Direction':<12} {'Class'}")
    for r in results_2025[:15]:
        print(f"  {r['feature']:<22} {r['t_stat']:>8.2f} {r['spread']:>+7.2f}% {r['direction']:<12} {r['classification']}")

    # ============================================================
    # 2. COMPREHENSIVE STRATEGY GRID on Nifty 500
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  COMPREHENSIVE STRATEGY GRID — Nifty 500")
    print(f"{'='*70}")

    strategies = [
        # (signal, condition, threshold, label)
        ("return_60d", "highest", None, "3m momentum"),
        ("return_20d", "highest", None, "1m momentum"),
        ("return_252d", "highest", None, "12m momentum"),
        ("return_5d", "lowest", None, "5d mean reversion"),
        ("rsi_3", "lowest", None, "RSI3 oversold"),
        ("return_20d", "lowest", None, "1m losers bounce"),
        ("vol_5d_vs_20d", "lowest", None, "Volume drying up"),
        ("overnight_gap", "highest", None, "Gap up continuation"),
        ("dist_52w_high", "highest", None, "Near 52w high"),
        ("rel_strength_20d", "highest", None, "Rel strength vs Nifty"),
        # Extreme z-score thresholds
        ("return_5d", "below_threshold", -2.0, "5d crash (z<-2)"),
        ("rsi_3", "below_threshold", -2.0, "RSI3 extreme (z<-2)"),
        ("return_60d", "above_threshold", 2.0, "3m surge (z>2)"),
    ]

    all_results = []

    for signal, condition, threshold, label in strategies:
        if signal not in panel.columns:
            continue

        for hold in [5, 7, 10]:
            for max_pos in [3, 5, 7]:
                result = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col=signal, condition=condition,
                    threshold=threshold,
                    hold_days=hold, max_positions=max_pos,
                    num_buckets=10,
                )
                if result is None:
                    continue
                m = result["metrics"]
                if m["trades"] < 100:
                    continue

                result_2025 = run_enhanced_backtest(
                    stock_data, panel,
                    signal_col=signal, condition=condition,
                    threshold=threshold,
                    hold_days=hold, max_positions=max_pos,
                    num_buckets=10,
                    start_date="2025-01-01",
                )

                weekly = result["weekly_returns"]
                test_sh = result_2025["metrics"]["sharpe"] if result_2025 else -99
                test_cagr = result_2025["metrics"]["cagr"] if result_2025 else -99
                test_wr = result_2025["metrics"]["win_rate"] if result_2025 else 0
                test_trades = result_2025["metrics"]["trades"] if result_2025 else 0

                # Min capital for 5K/week (2025)
                if result_2025:
                    tw = result_2025["weekly_returns"]
                    avg_2025 = tw.mean()
                    min_cap = 5000 / avg_2025 if avg_2025 > 0 else 9999999
                    pct_pos_2025 = (tw > 0).sum() / len(tw) * 100
                else:
                    min_cap = 9999999
                    pct_pos_2025 = 0

                all_results.append({
                    "label": label, "signal": signal, "condition": condition,
                    "threshold": threshold,
                    "hold": hold, "max_pos": max_pos,
                    "full_sharpe": m["sharpe"], "full_cagr": m["cagr"],
                    "full_mdd": m["max_drawdown"], "full_wr": m["win_rate"],
                    "full_pf": m["profit_factor"], "full_trades": m["trades"],
                    "test_sharpe": test_sh, "test_cagr": test_cagr,
                    "test_wr": test_wr, "test_trades": test_trades,
                    "min_cap_5k": min_cap, "pct_pos_2025": pct_pos_2025,
                    "result": result, "result_2025": result_2025,
                })

    df = pd.DataFrame(all_results)

    # Sort by 2025 Sharpe
    viable = df[df["test_sharpe"] > 0].sort_values("test_sharpe", ascending=False)

    print(f"\n  ALL STRATEGIES WITH POSITIVE 2025 SHARPE ({len(viable)} configs):")
    print(f"  {'Label':<22} {'Hold':>4} {'Pos':>3} {'FSh':>5} {'FCAGR':>6} {'TSh':>5} {'TCAGR':>6} {'TWR':>5} {'TTrd':>4} {'MinCap':>8} {'%Pos25':>6}")
    for _, r in viable.head(30).iterrows():
        print(f"  {r['label']:<22} {r['hold']:>4}d {r['max_pos']:>3} "
              f"{r['full_sharpe']:>5.2f} {r['full_cagr']:>+5.1f}% "
              f"{r['test_sharpe']:>5.2f} {r['test_cagr']:>+5.1f}% "
              f"{r['test_wr']:>5.1f} {r['test_trades']:>4} "
              f"{r['min_cap_5k']:>8,.0f} {r['pct_pos_2025']:>6.1f}")

    # ============================================================
    # 3. TOP 5 DEEP ANALYSIS
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  DEEP ANALYSIS: Top 5 strategies")
    print(f"{'='*70}")

    seen_strategies = set()
    top_candidates = []

    for _, r in viable.iterrows():
        key = (r["label"], r["hold"])
        if key in seen_strategies:
            continue
        seen_strategies.add(key)
        top_candidates.append(r)
        if len(top_candidates) >= 5:
            break

    for i, r in enumerate(top_candidates):
        label = f"#{i+1} {r['label']} | hold={r['hold']}d | pos={r['max_pos']}"
        capital_levels = [300000, 400000, 500000, 600000, 750000]

        print(f"\n  --- {label} ---")
        print(f"  Full: Sharpe={r['full_sharpe']:.2f} CAGR={r['full_cagr']:.1f}% WR={r['full_wr']:.1f}% PF={r['full_pf']:.2f} Trades={r['full_trades']}")
        print(f"  2025: Sharpe={r['test_sharpe']:.2f} CAGR={r['test_cagr']:.1f}% WR={r['test_wr']:.1f}% Trades={r['test_trades']}")
        print(f"  Min capital for 5K/wk: {r['min_cap_5k']:,.0f}")

        # Capital sensitivity
        result_2025 = r["result_2025"]
        if result_2025:
            wk = result_2025["weekly_returns"]
            print(f"  {'Capital':>10} {'Avg/wk':>8} {'%>=5K':>6} {'%>0':>5} {'Worst':>9}")
            for cap in capital_levels:
                wk_inr = wk * cap
                print(f"  {cap:>10,} {wk_inr.mean():>+8,.0f} {(wk_inr>=5000).sum()/len(wk_inr)*100:>5.0f}% "
                      f"{(wk_inr>0).sum()/len(wk_inr)*100:>4.0f}% {wk_inr.min():>+9,.0f}")

    # ============================================================
    # 4. PICK THE WINNER — Full validation
    # ============================================================
    if len(top_candidates) > 0:
        winner = top_candidates[0]
        print(f"\n\n{'*'*70}")
        print(f"  WINNER: {winner['label']} | hold={winner['hold']}d | pos={winner['max_pos']}")
        print(f"{'*'*70}")

        result_2025 = winner["result_2025"]
        result_full = winner["result"]

        # Determine capital
        wk_2025 = result_2025["weekly_returns"] if result_2025 else None
        if wk_2025 is not None and wk_2025.mean() > 0:
            exact_cap = 5000 / wk_2025.mean()
            rec_cap = int(np.ceil(exact_cap * 1.2 / 50000) * 50000)
        else:
            rec_cap = 750000

        print(f"\n  RECOMMENDED CAPITAL: {rec_cap:,} INR")

        # Full period analysis
        analyze_result(result_full, rec_cap, "FULL PERIOD (2021-2026)")

        # 2025 out-of-sample
        if result_2025:
            analyze_result(result_2025, rec_cap, "2025 OUT-OF-SAMPLE")
            detailed_weekly_analysis(result_2025, rec_cap, "2025 WEEK-BY-WEEK")

        # 2024 H2 validation
        result_2024h2 = run_enhanced_backtest(
            stock_data, panel,
            signal_col=winner["signal"], condition=winner["condition"],
            threshold=winner["threshold"],
            hold_days=int(winner["hold"]), max_positions=int(winner["max_pos"]),
            num_buckets=10,
            start_date="2024-07-01", end_date="2024-12-31",
        )
        if result_2024h2:
            m24 = result_2024h2["metrics"]
            w24 = result_2024h2["weekly_returns"] * rec_cap
            print(f"\n  2024 H2 Cross-Validation: Sharpe={m24['sharpe']:.2f} WR={m24['win_rate']:.1f}% "
                  f"Avg/wk={w24.mean():+,.0f} Weeks>0={((w24>0).sum()/len(w24)*100):.0f}%")

        # Monthly breakdown 2024-2026
        trades_all = result_full["trades"]
        recent = trades_all[trades_all["entry_date"] >= pd.Timestamp("2024-01-01")].copy()
        if len(recent) > 0:
            recent["month"] = recent["entry_date"].dt.to_period("M")
            max_pos = int(winner["max_pos"])
            print(f"\n  --- Monthly P&L (2024-2026) at {rec_cap:,} INR ---")
            print(f"  {'Month':<10} {'Trades':>6} {'WR':>5} {'P&L INR':>10}")
            months_positive = 0
            months_total = 0
            for month in sorted(recent["month"].unique()):
                mt = recent[recent["month"] == month]
                pnl = mt["net_return"].sum() / max_pos * rec_cap
                wr = (mt["net_return"] > 0).sum() / len(mt) * 100
                marker = " +" if pnl >= 20000 else (" *" if pnl < 0 else "")
                print(f"  {str(month):<10} {len(mt):>6} {wr:>4.0f}% {pnl:>+10,.0f}{marker}")
                months_total += 1
                if pnl > 0:
                    months_positive += 1
            print(f"  Profitable months: {months_positive}/{months_total}")

        # Compare vs Nifty 200 result
        print(f"\n  --- Nifty 500 vs Nifty 200 Comparison ---")
        result_n200 = run_enhanced_backtest(
            stock_data, panel[panel["ticker"].isin(NIFTY_200)],
            signal_col=winner["signal"], condition=winner["condition"],
            threshold=winner["threshold"],
            hold_days=int(winner["hold"]), max_positions=int(winner["max_pos"]),
            num_buckets=10,
        )
        result_n200_2025 = run_enhanced_backtest(
            stock_data, panel[panel["ticker"].isin(NIFTY_200)],
            signal_col=winner["signal"], condition=winner["condition"],
            threshold=winner["threshold"],
            hold_days=int(winner["hold"]), max_positions=int(winner["max_pos"]),
            num_buckets=10,
            start_date="2025-01-01",
        )

        if result_n200:
            m200 = result_n200["metrics"]
            m500 = result_full["metrics"]
            print(f"  {'':>15} {'Nifty 200':>12} {'Nifty 500':>12}")
            print(f"  {'Sharpe':>15} {m200['sharpe']:>12.2f} {m500['sharpe']:>12.2f}")
            print(f"  {'CAGR':>15} {m200['cagr']:>+11.1f}% {m500['cagr']:>+11.1f}%")
            print(f"  {'WR':>15} {m200['win_rate']:>11.1f}% {m500['win_rate']:>11.1f}%")
            print(f"  {'MDD':>15} {m200['max_drawdown']:>11.1f}% {m500['max_drawdown']:>11.1f}%")
            print(f"  {'Trades':>15} {m200['trades']:>12} {m500['trades']:>12}")

        if result_n200_2025 and result_2025:
            m200t = result_n200_2025["metrics"]
            m500t = result_2025["metrics"]
            w200 = result_n200_2025["weekly_returns"] * rec_cap
            w500 = result_2025["weekly_returns"] * rec_cap
            print(f"\n  2025 Out-of-Sample:")
            print(f"  {'':>15} {'Nifty 200':>12} {'Nifty 500':>12}")
            print(f"  {'Sharpe':>15} {m200t['sharpe']:>12.2f} {m500t['sharpe']:>12.2f}")
            print(f"  {'Avg/wk INR':>15} {w200.mean():>+12,.0f} {w500.mean():>+12,.0f}")
            print(f"  {'Weeks>=5K':>15} {(w200>=5000).sum()/len(w200)*100:>11.0f}% {(w500>=5000).sum()/len(w500)*100:>11.0f}%")

        # DEPLOYMENT SPEC
        sig = winner["signal"]
        cond = winner["condition"]
        thresh = winner["threshold"]
        hold = int(winner["hold"])
        max_pos = int(winner["max_pos"])

        print(f"\n\n{'*'*70}")
        print(f"  DEPLOYMENT SPECIFICATION")
        print(f"{'*'*70}")
        print(f"  Strategy: {winner['label']}")
        print(f"  Universe: Nifty 500")
        print(f"  Signal: {sig} | Condition: {cond}" + (f" | Threshold: {thresh}" if thresh else " | Top decile"))
        print(f"  Hold: {hold} trading days")
        print(f"  Positions: {max_pos} equal-weight")
        print(f"  Capital: {rec_cap:,} INR")
        print(f"  Per position: {rec_cap // max_pos:,} INR")
        print(f"  Stop-loss: -8% per position ({rec_cap // max_pos * 8 // 100:,} INR)")
        print(f"  Cost: 0.1% round trip")
        print(f"")
        print(f"  Expected (2025 out-of-sample):")
        if result_2025:
            wk = result_2025["weekly_returns"] * rec_cap
            print(f"    Avg weekly: {wk.mean():+,.0f} INR")
            print(f"    Weeks >= 5K: {(wk>=5000).sum()}/{len(wk)} ({(wk>=5000).sum()/len(wk)*100:.0f}%)")
            print(f"    Weeks > 0: {(wk>0).sum()}/{len(wk)} ({(wk>0).sum()/len(wk)*100:.0f}%)")
            print(f"    Worst week: {wk.min():+,.0f} INR")
        print(f"{'*'*70}")


if __name__ == "__main__":
    main()
