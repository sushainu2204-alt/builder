#!/usr/bin/env python3
"""
Research: Does adding shorts improve the bear dip strategy?

Tests:
1. Short-only strategies in bear regime (short the strongest momentum stocks)
2. Short-only in bull regime (short overbought stocks when market is extended)
3. Combined long+short portfolio (hedge)
4. Compare: long-only vs long+short
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from backtest.metrics import compute_all_metrics, format_metrics
from research_5k_v5 import get_nifty_regime


def run_short_backtest(
    stock_data, panel, nifty_regime,
    signal_col, condition, threshold=None,
    hold_days=5, max_positions=5, num_buckets=10,
    allowed_regime=None, block_reentry_days=5,
    start_date=None, end_date=None,
):
    """Short backtest: sell high, buy back low. Profit = entry - exit."""
    df = panel.copy()
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    regime_map = nifty_regime.to_dict()
    all_dates = sorted(df["date"].unique())
    trades = []
    open_pos = {}
    blocked = {}

    for date in all_dates:
        regime = None
        for delta in range(0, 10):
            d = date - pd.Timedelta(days=delta)
            if d in regime_map:
                regime = regime_map[d]
                break
        if allowed_regime is not None and regime != allowed_regime:
            continue

        date_idx = all_dates.index(date)
        open_pos = {t: info for t, info in open_pos.items() if info["target_idx"] > date_idx}
        blocked = {t: d for t, d in blocked.items() if d > date}

        available = max_positions - len(open_pos)
        if available <= 0:
            continue

        day_data = df[df["date"] == date].dropna(subset=[signal_col])
        if len(day_data) < num_buckets * 2:
            continue

        try:
            day_data = day_data.copy()
            day_data["_bucket"] = pd.qcut(day_data[signal_col], num_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue

        # For shorts: we want to short the HIGHEST (overbought/overextended)
        if condition == "highest":
            selected = day_data[day_data["_bucket"] == day_data["_bucket"].max()]
            selected = selected.sort_values(signal_col, ascending=False)
        elif condition == "above_threshold" and threshold is not None:
            selected = day_data[day_data[signal_col] > threshold]
            selected = selected.sort_values(signal_col, ascending=False)
        else:
            continue

        candidates = [t for t in selected["ticker"].tolist()
                      if t not in open_pos and t not in blocked][:available]

        for ticker in candidates:
            if ticker not in stock_data:
                continue
            sdf = stock_data[ticker]
            try:
                entry_idx = sdf.index.get_indexer([date], method="nearest")[0]
            except Exception:
                continue

            actual_entry = entry_idx + 1
            if actual_entry >= len(sdf):
                continue

            entry_price = sdf.iloc[actual_entry]["Open"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            exit_idx = min(actual_entry + hold_days, len(sdf) - 1)
            exit_price = entry_price
            stopped = False
            # Short stop-loss: price goes UP 8%
            stop_price = entry_price * 1.08

            for di in range(actual_entry, exit_idx + 1):
                if sdf.iloc[di]["High"] >= stop_price:
                    exit_price = stop_price
                    stopped = True
                    break
                if di == exit_idx:
                    exit_price = sdf.iloc[di]["Close"]

            # Short P&L: profit when price drops
            gross = (entry_price - exit_price) / entry_price
            net = gross - 0.002  # Higher cost for shorts (borrow + STT)

            trades.append({
                "ticker": ticker,
                "entry_date": sdf.index[actual_entry],
                "exit_date": sdf.index[min(exit_idx, len(sdf) - 1)],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": net,
                "stopped_out": stopped,
                "side": "SHORT",
            })

            target = date_idx + hold_days + 1
            open_pos[ticker] = {"target_idx": min(target, len(all_dates) - 1)}
            if stopped:
                blocked[ticker] = date + pd.Timedelta(days=block_reentry_days * 1.5)

    if not trades:
        return None

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    equity = [1.0]
    for _, t in trades_df.iterrows():
        equity.append(equity[-1] * (1 + t["net_return"] / max_positions))

    metrics = compute_all_metrics(trades_df["net_return"], pd.Series(equity))
    trades_df["week"] = trades_df["entry_date"].dt.isocalendar().week.astype(int)
    trades_df["year"] = trades_df["entry_date"].dt.year
    trades_df["yearweek"] = trades_df["year"].astype(str) + "-W" + trades_df["week"].astype(str).str.zfill(2)
    weekly = trades_df.groupby("yearweek")["net_return"].sum() / max_positions

    return {"trades": trades_df, "metrics": metrics, "equity_curve": pd.Series(equity), "weekly_returns": weekly}


def combine_long_short(long_result, short_result, long_cap, short_cap, max_pos):
    """Combine long and short trade streams into one portfolio."""
    all_trades = []
    if long_result and not long_result["trades"].empty:
        lt = long_result["trades"].copy()
        lt["side"] = "LONG"
        all_trades.append(lt)
    if short_result and not short_result["trades"].empty:
        all_trades.append(short_result["trades"])

    if not all_trades:
        return None

    combined = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
    total_cap = long_cap + short_cap

    equity = [1.0]
    for _, t in combined.iterrows():
        cap = long_cap if t["side"] == "LONG" else short_cap
        pos_size = cap / (max_pos * total_cap)
        equity.append(equity[-1] * (1 + t["net_return"] * pos_size))

    metrics = compute_all_metrics(combined["net_return"], pd.Series(equity))
    combined["week"] = combined["entry_date"].dt.isocalendar().week.astype(int)
    combined["year"] = combined["entry_date"].dt.year
    combined["yearweek"] = combined["year"].astype(str) + "-W" + combined["week"].astype(str).str.zfill(2)
    weekly = combined.groupby("yearweek")["net_return"].apply(
        lambda x: sum(r * (long_cap if s == "LONG" else short_cap) / total_cap / max_pos
                      for r, s in zip(x, combined.loc[x.index, "side"]))
    )

    return {"trades": combined, "metrics": metrics, "equity_curve": pd.Series(equity), "weekly_returns": weekly}


def main():
    print(f"\n{'#'*70}")
    print(f"  RESEARCH: ADDING SHORTS")
    print(f"  Rule #4 says long-only. Let's test if data agrees.")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)
    nifty_regime = get_nifty_regime(index_data)

    from research_5k_v5 import run_regime_filtered_backtest

    # ============================================================
    # 1. SHORT-ONLY STRATEGIES
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  SHORT-ONLY STRATEGIES")
    print(f"{'='*70}")

    short_configs = [
        # (signal, condition, threshold, regime, label)
        ("rsi_3", "highest", None, 1, "Short overbought RSI3, bull"),
        ("rsi_3", "highest", None, 0, "Short overbought RSI3, bear"),
        ("rsi_3", "highest", None, None, "Short overbought RSI3, any"),
        ("return_5d", "highest", None, 1, "Short 5d winners, bull"),
        ("return_5d", "highest", None, 0, "Short 5d winners, bear"),
        ("return_60d", "highest", None, 0, "Short 3m momentum, bear"),
        ("return_60d", "highest", None, None, "Short 3m momentum, any"),
        ("rsi_3", "above_threshold", 2.0, 0, "Short RSI3 z>2, bear"),
        ("rsi_3", "above_threshold", 2.0, None, "Short RSI3 z>2, any"),
        ("overnight_gap", "highest", None, 0, "Short gap up, bear"),
    ]

    short_results = []
    for signal, condition, threshold, regime, label in short_configs:
        for hold in [3, 5]:
            for max_pos in [3, 5]:
                r = run_short_backtest(
                    stock_data, panel, nifty_regime,
                    signal_col=signal, condition=condition, threshold=threshold,
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    allowed_regime=regime, block_reentry_days=5,
                )
                if r is None or r["metrics"]["trades"] < 30:
                    continue

                m = r["metrics"]
                weekly = r["weekly_returns"]

                # 2025+ test
                t2025 = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
                if len(t2025) > 0:
                    t25_ret = t2025["net_return"]
                    t25_eq = pd.Series([1.0] + list(1 + t25_ret.cumsum() / max_pos))
                    t25_m = compute_all_metrics(t25_ret, t25_eq)
                    t25_sh = t25_m["sharpe"]
                    t25_wr = t25_m["win_rate"]
                else:
                    t25_sh = -99
                    t25_wr = 0

                # Crash period
                crash = r["trades"][r["trades"]["entry_date"] >= pd.Timestamp("2026-02-01")]
                crash_pnl = crash["net_return"].sum() / max_pos if len(crash) > 0 else 0

                for cap in [300000, 600000]:
                    wk = weekly * cap
                    short_results.append({
                        "label": label, "hold": hold, "pos": max_pos, "cap": cap,
                        "sharpe": m["sharpe"], "cagr": m["cagr"], "wr": m["win_rate"],
                        "mdd": m["max_drawdown"], "trades": m["trades"],
                        "avg_wk": wk.mean(), "t25_sh": t25_sh, "t25_wr": t25_wr,
                        "crash_pnl": crash_pnl * cap,
                        "result": r,
                    })

    sdf = pd.DataFrame(short_results)
    if len(sdf) > 0:
        sdf_sorted = sdf.sort_values("t25_sh", ascending=False)
        print(f"\n  {'Label':<30} {'H':>2} {'P':>2} {'Cap':>5} {'Sh':>5} {'CAGR':>6} {'WR':>5} {'25Sh':>5} {'25WR':>5} {'CrPnL':>8} {'Trd':>4}")
        for _, r in sdf_sorted.head(20).iterrows():
            tag = " <<" if r["t25_sh"] > 0 else ""
            print(f"  {r['label']:<30} {r['hold']:>2} {r['pos']:>2} {r['cap']/1000:>4.0f}K "
                  f"{r['sharpe']:>5.2f} {r['cagr']:>+5.1f}% {r['wr']:>5.1f} "
                  f"{r['t25_sh']:>5.2f} {r['t25_wr']:>5.1f} {r['crash_pnl']:>+8,.0f} {r['trades']:>4}{tag}")

    # ============================================================
    # 2. COMBINED LONG + SHORT
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  COMBINED LONG + SHORT PORTFOLIOS")
    print(f"  Long: RSI3 bear dip | Short: best from above")
    print(f"{'='*70}")

    # Long baseline (our champion)
    long_result = run_regime_filtered_backtest(
        stock_data, panel, nifty_regime,
        signal_col="rsi_3", condition="lowest",
        hold_days=5, max_positions=5, num_buckets=10,
        allowed_regime=0, block_reentry_days=5,
    )

    if long_result is None:
        print("  Long strategy returned no results")
        return

    long_m = long_result["metrics"]
    long_wk = long_result["weekly_returns"]

    print(f"\n  LONG-ONLY BASELINE (RSI3 bear dip, 5d, 5 pos):")
    print(f"    Sharpe={long_m['sharpe']:.2f} CAGR={long_m['cagr']:.1f}% WR={long_m['win_rate']:.1f}% Trades={long_m['trades']}")
    for cap in [400000, 600000]:
        wk = long_wk * cap
        print(f"    At {cap/1000:.0f}K: Avg/wk={wk.mean():+,.0f} | %>=5K={((wk>=5000).sum()/len(wk)*100):.0f}% | %>0={((wk>0).sum()/len(wk)*100):.0f}%")

    # Test short additions
    best_short_configs = [
        ("rsi_3", "highest", None, 0, 3, 5, "RSI3 overbought, bear"),
        ("rsi_3", "highest", None, 0, 5, 5, "RSI3 overbought, bear"),
        ("rsi_3", "highest", None, 1, 3, 5, "RSI3 overbought, bull"),
        ("rsi_3", "highest", None, 1, 5, 5, "RSI3 overbought, bull"),
        ("return_5d", "highest", None, 0, 3, 5, "5d winners, bear"),
        ("return_5d", "highest", None, 0, 5, 5, "5d winners, bear"),
        ("return_60d", "highest", None, 0, 5, 5, "3m mom, bear"),
        ("rsi_3", "above_threshold", 2.0, 0, 3, 3, "RSI3 z>2, bear"),
        ("rsi_3", "above_threshold", 2.0, None, 5, 5, "RSI3 z>2, any"),
    ]

    combo_results = []
    for sig, cond, thresh, regime, hold, pos, slabel in best_short_configs:
        short_r = run_short_backtest(
            stock_data, panel, nifty_regime,
            signal_col=sig, condition=cond, threshold=thresh,
            hold_days=hold, max_positions=pos, num_buckets=10,
            allowed_regime=regime, block_reentry_days=5,
        )
        if short_r is None:
            continue

        # Combine: 70% long, 30% short
        for total_cap in [500000, 600000, 750000]:
            long_alloc = int(total_cap * 0.7)
            short_alloc = int(total_cap * 0.3)

            combo = combine_long_short(long_result, short_r, long_alloc, short_alloc, 5)
            if combo is None:
                continue

            cm = combo["metrics"]
            cwk = combo["weekly_returns"] * total_cap

            # 2025 subset
            t25 = combo["trades"][combo["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
            crash = combo["trades"][combo["trades"]["entry_date"] >= pd.Timestamp("2026-02-01")]
            crash_pnl = sum(
                t["net_return"] * (long_alloc if t["side"] == "LONG" else short_alloc) / 5
                for _, t in crash.iterrows()
            ) if len(crash) > 0 else 0

            combo_results.append({
                "short_label": slabel, "hold": hold, "pos": pos,
                "total_cap": total_cap,
                "sharpe": cm["sharpe"], "cagr": cm["cagr"], "wr": cm["win_rate"],
                "mdd": cm["max_drawdown"], "trades": cm["trades"],
                "avg_wk": cwk.mean(),
                "pct_pos": (cwk > 0).sum() / len(cwk) * 100,
                "pct_5k": (cwk >= 5000).sum() / len(cwk) * 100,
                "crash_pnl": crash_pnl,
            })

    cdf = pd.DataFrame(combo_results)
    if len(cdf) > 0:
        cdf_sorted = cdf.sort_values("avg_wk", ascending=False)
        print(f"\n  LONG+SHORT COMBOS (70% long / 30% short):")
        print(f"  {'Short side':<22} {'H':>2} {'P':>2} {'Cap':>5} {'Sh':>5} {'CAGR':>6} {'WR':>5} {'AvgWk':>7} {'%>0':>5} {'%5K':>4} {'CrPnL':>8}")
        for _, r in cdf_sorted.head(15).iterrows():
            tag = " <<" if r["avg_wk"] >= 5000 and r["crash_pnl"] > -20000 else ""
            print(f"  {r['short_label']:<22} {r['hold']:>2} {r['pos']:>2} {r['total_cap']/1000:>4.0f}K "
                  f"{r['sharpe']:>5.2f} {r['cagr']:>+5.1f}% {r['wr']:>5.1f} "
                  f"{r['avg_wk']:>+7,.0f} {r['pct_pos']:>5.1f} {r['pct_5k']:>4.0f} {r['crash_pnl']:>+8,.0f}{tag}")

    # ============================================================
    # 3. VERDICT
    # ============================================================
    print(f"\n\n{'='*70}")
    print(f"  VERDICT: LONG-ONLY vs LONG+SHORT")
    print(f"{'='*70}")

    # Best long-only at 600K
    l600 = long_wk * 600000
    print(f"\n  LONG-ONLY (RSI3 bear dip, 600K):")
    print(f"    Avg/wk: {l600.mean():+,.0f} | %>0: {(l600>0).sum()/len(l600)*100:.0f}% | %>=5K: {(l600>=5000).sum()/len(l600)*100:.0f}%")

    # Best combo
    if len(cdf) > 0:
        best_combo = cdf_sorted.iloc[0]
        print(f"\n  BEST LONG+SHORT ({best_combo['short_label']}, {best_combo['total_cap']/1000:.0f}K):")
        print(f"    Avg/wk: {best_combo['avg_wk']:+,.0f} | %>0: {best_combo['pct_pos']:.0f}% | %>=5K: {best_combo['pct_5k']:.0f}%")
        print(f"    Crash P&L: {best_combo['crash_pnl']:+,.0f}")

        improvement = best_combo["avg_wk"] - l600.mean()
        print(f"\n  Improvement from adding shorts: {improvement:+,.0f} INR/week")
        if improvement > 1000:
            print(f"  >> SHORTS ADD VALUE — recommend long+short approach")
        elif improvement > 0:
            print(f"  >> MARGINAL IMPROVEMENT — shorts help slightly but add complexity")
        else:
            print(f"  >> SHORTS DON'T HELP — Rule #4 was right. Stay long-only.")

    # Short-only standalone verdict
    if len(sdf) > 0:
        best_short_standalone = sdf.sort_values("t25_sh", ascending=False).iloc[0]
        print(f"\n  BEST SHORT-ONLY ({best_short_standalone['label']}):")
        print(f"    Sharpe: {best_short_standalone['sharpe']:.2f} | 2025 Sharpe: {best_short_standalone['t25_sh']:.2f}")
        print(f"    WR: {best_short_standalone['wr']:.1f}% | Crash P&L: {best_short_standalone['crash_pnl']:+,.0f}")
        if best_short_standalone["t25_sh"] < 0:
            print(f"    >> Short-only is NEGATIVE in 2025. Rule #4 confirmed.")


if __name__ == "__main__":
    main()
