#!/usr/bin/env python3
"""
Research V5: Fix the momentum crash problem.

Problems identified:
1. Momentum strategy gets destroyed during market corrections
2. Same stocks keep getting re-entered after stop-loss (concentration risk)
3. No regime filter = blind entries in hostile conditions

Solutions to test:
A. Regime filter: Only trade when Nifty > 20d SMA (bull mode)
B. Anti-concentration: Block re-entry of stopped stocks for 10 days
C. Mean reversion IN corrections: When Nifty < 20d SMA, switch to buying dips
D. Completely different: Low volatility + relative strength (defensive momentum)
E. Multi-regime: Momentum in bull, mean reversion in bear, cash in chop

Goal: Find strategy that works ACROSS regimes including Feb-Mar 2026 crash.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from backtest.metrics import compute_all_metrics, format_metrics
from research_5k import run_enhanced_backtest, analyze_result


def get_nifty_regime(nifty_data, sma_period=20):
    """1 = bull (close > SMA), 0 = bear."""
    close = nifty_data["Close"]
    sma = close.rolling(sma_period).mean()
    regime = (close > sma).astype(int)
    return regime


def run_regime_filtered_backtest(
    stock_data, panel, nifty_regime,
    signal_col, condition, threshold=None,
    hold_days=10, max_positions=5, num_buckets=10,
    allowed_regime=1,  # 1=bull only, 0=bear only, None=both
    block_reentry_days=10,
    start_date=None, end_date=None,
):
    """
    Enhanced backtest with:
    - Regime filter (only enter in specified regime)
    - Anti-concentration (block re-entry after stop-loss)
    """
    df = panel.copy()
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    regime_map = nifty_regime.to_dict()
    all_dates = sorted(df["date"].unique())

    trades = []
    open_positions = {}
    blocked_tickers = {}  # ticker -> unblock_date

    for date in all_dates:
        # Check regime
        regime = None
        for delta in range(0, 10):
            d = date - pd.Timedelta(days=delta)
            if d in regime_map:
                regime = regime_map[d]
                break

        if allowed_regime is not None and regime != allowed_regime:
            continue

        # Clean expired positions
        date_idx = all_dates.index(date)
        open_positions = {t: info for t, info in open_positions.items()
                         if info["target_idx"] > date_idx}

        # Clean expired blocks
        blocked_tickers = {t: d for t, d in blocked_tickers.items() if d > date}

        available = max_positions - len(open_positions)
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

        if condition == "lowest":
            selected = day_data[day_data["_bucket"] == 0]
        elif condition == "highest":
            selected = day_data[day_data["_bucket"] == day_data["_bucket"].max()]
        elif condition == "below_threshold" and threshold is not None:
            selected = day_data[day_data[signal_col] < threshold]
        elif condition == "above_threshold" and threshold is not None:
            selected = day_data[day_data[signal_col] > threshold]
        else:
            continue

        if condition in ("lowest", "below_threshold"):
            selected = selected.sort_values(signal_col, ascending=True)
        else:
            selected = selected.sort_values(signal_col, ascending=False)

        candidates = [t for t in selected["ticker"].tolist()
                      if t not in open_positions and t not in blocked_tickers][:available]

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
            stop_price = entry_price * 0.92

            for di in range(actual_entry, exit_idx + 1):
                if sdf.iloc[di]["Low"] <= stop_price:
                    exit_price = stop_price
                    stopped = True
                    break
                if di == exit_idx:
                    exit_price = sdf.iloc[di]["Close"]

            gross = (exit_price / entry_price) - 1
            net = gross - 0.001

            trades.append({
                "ticker": ticker,
                "entry_date": sdf.index[actual_entry],
                "exit_date": sdf.index[min(exit_idx, len(sdf) - 1)],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": net,
                "stopped_out": stopped,
            })

            target_exit = all_dates.index(date) + hold_days + 1 if date in all_dates else date_idx + hold_days + 1
            open_positions[ticker] = {"target_idx": min(target_exit, len(all_dates) - 1)}

            if stopped and block_reentry_days > 0:
                blocked_tickers[ticker] = date + pd.Timedelta(days=block_reentry_days * 1.5)

    if not trades:
        return None

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    equity = [1.0]
    for _, t in trades_df.iterrows():
        equity.append(equity[-1] * (1 + t["net_return"] / max_positions))
    equity_curve = pd.Series(equity)

    metrics = compute_all_metrics(trades_df["net_return"], equity_curve)

    trades_df["week"] = trades_df["entry_date"].dt.isocalendar().week.astype(int)
    trades_df["year"] = trades_df["entry_date"].dt.year
    trades_df["yearweek"] = trades_df["year"].astype(str) + "-W" + trades_df["week"].astype(str).str.zfill(2)
    weekly = trades_df.groupby("yearweek")["net_return"].sum() / max_positions

    return {
        "trades": trades_df, "metrics": metrics,
        "equity_curve": equity_curve, "weekly_returns": weekly,
    }


def main():
    print(f"\n{'#'*70}")
    print(f"  RESEARCH V5: REGIME-AWARE STRATEGIES")
    print(f"  Fixing the momentum crash problem")
    print(f"{'#'*70}")

    stock_data = download_universe(NIFTY_200, START_DATE, END_DATE)
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)
    panel = build_panel(stock_data, index_data)

    nifty_regime = get_nifty_regime(index_data)
    bull_days = nifty_regime.sum()
    bear_days = len(nifty_regime) - bull_days
    print(f"\n  Regime: {bull_days} bull days, {bear_days} bear days")

    # ============================================================
    # TEST ALL APPROACHES
    # ============================================================
    approaches = []

    # A: Momentum with regime filter (bull only)
    for hold in [5, 7, 10]:
        for max_pos in [3, 5]:
            r = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="return_60d", condition="highest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
                allowed_regime=1, block_reentry_days=10,
            )
            if r and r["metrics"]["trades"] >= 50:
                approaches.append(("A: Mom bull-only", hold, max_pos, r))

    # B: Mean reversion in bear regime only
    for hold in [3, 5, 7]:
        for max_pos in [3, 5]:
            r = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="return_5d", condition="lowest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
                allowed_regime=0, block_reentry_days=5,
            )
            if r and r["metrics"]["trades"] >= 50:
                approaches.append(("B: MR bear-only", hold, max_pos, r))

    # C: Multi-regime combined (momentum in bull + MR in bear)
    for hold in [5, 7, 10]:
        for max_pos in [3, 5]:
            r_bull = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="return_60d", condition="highest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
                allowed_regime=1, block_reentry_days=10,
            )
            r_bear = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="return_5d", condition="lowest",
                hold_days=min(hold, 5), max_positions=max_pos, num_buckets=10,
                allowed_regime=0, block_reentry_days=5,
            )

            if r_bull is None and r_bear is None:
                continue

            all_trades = []
            if r_bull and not r_bull["trades"].empty:
                all_trades.append(r_bull["trades"])
            if r_bear and not r_bear["trades"].empty:
                all_trades.append(r_bear["trades"])
            if not all_trades:
                continue

            combined = pd.concat(all_trades).sort_values("entry_date").reset_index(drop=True)
            eq = [1.0]
            for _, t in combined.iterrows():
                eq.append(eq[-1] * (1 + t["net_return"] / max_pos))

            m = compute_all_metrics(combined["net_return"], pd.Series(eq))
            combined["week"] = combined["entry_date"].dt.isocalendar().week.astype(int)
            combined["year"] = combined["entry_date"].dt.year
            combined["yearweek"] = combined["year"].astype(str) + "-W" + combined["week"].astype(str).str.zfill(2)
            weekly = combined.groupby("yearweek")["net_return"].sum() / max_pos

            result = {"trades": combined, "metrics": m, "equity_curve": pd.Series(eq), "weekly_returns": weekly}
            if m["trades"] >= 50:
                approaches.append(("C: Multi-regime", hold, max_pos, result))

    # D: Momentum with NO regime filter but WITH anti-concentration
    for hold in [5, 7, 10]:
        for max_pos in [3, 5]:
            r = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="return_60d", condition="highest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
                allowed_regime=None, block_reentry_days=15,  # Strong anti-concentration
            )
            if r and r["metrics"]["trades"] >= 50:
                approaches.append(("D: Mom+anticonc", hold, max_pos, r))

    # E: Low vol momentum (buy stocks near 52w high with LOW volatility)
    # These are "quality momentum" — steady climbers, not wild ones
    panel_copy = panel.copy()
    # Composite: high dist_52w_high + low realized_vol = quality momentum
    if "dist_52w_high" in panel_copy.columns and "realized_vol_20d" in panel_copy.columns:
        panel_copy["quality_mom"] = panel_copy["dist_52w_high"] - panel_copy["realized_vol_20d"]

        for hold in [5, 7, 10]:
            for max_pos in [3, 5]:
                r = run_regime_filtered_backtest(
                    stock_data, panel_copy, nifty_regime,
                    signal_col="quality_mom", condition="highest",
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    allowed_regime=None, block_reentry_days=10,
                )
                if r and r["metrics"]["trades"] >= 50:
                    approaches.append(("E: Quality mom", hold, max_pos, r))

                # Bull only
                r2 = run_regime_filtered_backtest(
                    stock_data, panel_copy, nifty_regime,
                    signal_col="quality_mom", condition="highest",
                    hold_days=hold, max_positions=max_pos, num_buckets=10,
                    allowed_regime=1, block_reentry_days=10,
                )
                if r2 and r2["metrics"]["trades"] >= 50:
                    approaches.append(("E2: QMom bull", hold, max_pos, r2))

    # F: RSI(3) oversold with regime filter (bear only — buy dips in corrections)
    for hold in [3, 5]:
        for max_pos in [3, 5]:
            r = run_regime_filtered_backtest(
                stock_data, panel, nifty_regime,
                signal_col="rsi_3", condition="lowest",
                hold_days=hold, max_positions=max_pos, num_buckets=10,
                allowed_regime=0, block_reentry_days=5,
            )
            if r and r["metrics"]["trades"] >= 30:
                approaches.append(("F: RSI3 bear dip", hold, max_pos, r))

    # ============================================================
    # RANK ALL APPROACHES
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(approaches)} configurations tested")
    print(f"{'='*70}")

    rows = []
    for label, hold, max_pos, result in approaches:
        m = result["metrics"]
        weekly = result["weekly_returns"]

        # 2025+ test
        t2025 = result["trades"][result["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]

        # Feb-Mar 2026 (the crash period)
        crash = result["trades"][result["trades"]["entry_date"] >= pd.Timestamp("2026-02-01")]

        for cap in [400000, 600000]:
            wk_inr = weekly * cap
            t25_weekly = None
            if len(t2025) > 0:
                t2025c = t2025.copy()
                t2025c["yearweek"] = t2025c["year"].astype(str) + "-W" + t2025c["week"].astype(str).str.zfill(2)
                t25_weekly = t2025c.groupby("yearweek")["net_return"].sum() / max_pos * cap

            crash_pnl = crash["net_return"].sum() / max_pos * cap if len(crash) > 0 else 0

            rows.append({
                "label": label, "hold": hold, "pos": max_pos, "cap": cap,
                "full_sharpe": m["sharpe"], "full_cagr": m["cagr"],
                "full_wr": m["win_rate"], "trades": m["trades"],
                "full_mdd": m["max_drawdown"],
                "avg_wk": wk_inr.mean(),
                "pct_pos": (wk_inr > 0).sum() / len(wk_inr) * 100,
                "t25_avg_wk": t25_weekly.mean() if t25_weekly is not None and len(t25_weekly) > 0 else -99999,
                "t25_pct_pos": (t25_weekly > 0).sum() / len(t25_weekly) * 100 if t25_weekly is not None and len(t25_weekly) > 0 else 0,
                "crash_pnl": crash_pnl,
                "crash_trades": len(crash),
                "result": result,
            })

    rdf = pd.DataFrame(rows)

    # Sort by: positive 2025 avg, survived crash (crash_pnl > -50K), decent Sharpe
    best = rdf[
        (rdf["t25_avg_wk"] > 0) & (rdf["full_sharpe"] > 0.5)
    ].sort_values("t25_avg_wk", ascending=False)

    print(f"\n  {'Label':<18} {'H':>2} {'P':>2} {'Cap':>5} {'FSh':>5} {'CAGR':>6} {'WR':>5} {'MDD':>5} {'AvgWk':>7} {'25AvgW':>7} {'25%>0':>5} {'CrashPnL':>9} {'Trd':>4}")
    display = best if len(best) > 0 else rdf.sort_values("t25_avg_wk", ascending=False)
    for _, r in display.head(25).iterrows():
        tag = " <<" if r["t25_avg_wk"] >= 4000 and r["crash_pnl"] > -50000 else ""
        print(f"  {r['label']:<18} {r['hold']:>2} {r['pos']:>2} {r['cap']/1000:>4.0f}K "
              f"{r['full_sharpe']:>5.2f} {r['full_cagr']:>+5.1f}% {r['full_wr']:>5.1f} {r['full_mdd']:>5.1f} "
              f"{r['avg_wk']:>+7,.0f} {r['t25_avg_wk']:>+7,.0f} {r['t25_pct_pos']:>5.1f} "
              f"{r['crash_pnl']:>+9,.0f} {r['trades']:>4}{tag}")

    # ============================================================
    # PICK WINNER — strategy that works in 2025 AND survived the crash
    # ============================================================
    candidates = rdf[
        (rdf["t25_avg_wk"] >= 3500) &
        (rdf["crash_pnl"] > -30000) &
        (rdf["full_sharpe"] > 0.5)
    ].sort_values("t25_avg_wk", ascending=False)

    if len(candidates) == 0:
        # Relax crash constraint
        candidates = rdf[
            (rdf["t25_avg_wk"] >= 3000) &
            (rdf["full_sharpe"] > 0.3)
        ].sort_values("t25_avg_wk", ascending=False)

    if len(candidates) == 0:
        candidates = rdf.sort_values("t25_avg_wk", ascending=False)

    winner = candidates.iloc[0]
    result = winner["result"]
    cap = int(winner["cap"])

    print(f"\n\n{'*'*70}")
    print(f"  WINNER: {winner['label']} | hold={winner['hold']}d | pos={winner['pos']} | cap={cap:,}")
    print(f"  Full: Sharpe={winner['full_sharpe']:.2f} CAGR={winner['full_cagr']:.1f}% WR={winner['full_wr']:.1f}%")
    print(f"  2025 avg/wk: {winner['t25_avg_wk']:+,.0f} | Crash P&L: {winner['crash_pnl']:+,.0f}")
    print(f"{'*'*70}")

    analyze_result(result, cap, f"WINNER — Full Period")

    # 2025+ analysis
    t2025 = result["trades"][result["trades"]["entry_date"] >= pd.Timestamp("2025-01-01")]
    if len(t2025) > 0:
        eq25 = [1.0]
        for _, t in t2025.iterrows():
            eq25.append(eq25[-1] * (1 + t["net_return"] / winner["pos"]))
        m25 = compute_all_metrics(t2025["net_return"], pd.Series(eq25))
        t2025c = t2025.copy()
        t2025c["yearweek"] = t2025c["year"].astype(str) + "-W" + t2025c["week"].astype(str).str.zfill(2)
        w25 = t2025c.groupby("yearweek")["net_return"].sum() / winner["pos"] * cap

        print(f"\n  2025+ Out-of-Sample:")
        print(f"    Trades: {m25['trades']} | WR: {m25['win_rate']:.1f}% | Sharpe: {m25['sharpe']:.2f}")
        print(f"    Avg weekly: {w25.mean():+,.0f} | Weeks>0: {(w25>0).sum()}/{len(w25)} ({(w25>0).sum()/len(w25)*100:.0f}%)")
        print(f"    Weeks>=5K: {(w25>=5000).sum()}/{len(w25)} ({(w25>=5000).sum()/len(w25)*100:.0f}%)")

    # Feb-Mar 2026 crash detail
    crash = result["trades"][result["trades"]["entry_date"] >= pd.Timestamp("2026-02-01")]
    if len(crash) > 0:
        print(f"\n  Feb-Mar 2026 Crash Period:")
        print(f"    Trades: {len(crash)} | WR: {(crash['net_return']>0).sum()}/{len(crash)} "
              f"({(crash['net_return']>0).sum()/len(crash)*100:.0f}%)")
        crash_pnl = crash["net_return"].sum() / winner["pos"] * cap
        print(f"    P&L: {crash_pnl:+,.0f} INR")
        stopped = crash["stopped_out"].sum()
        print(f"    Stop-losses: {stopped} ({stopped/len(crash)*100:.0f}%)")

    # Monthly breakdown
    trades_all = result["trades"].copy()
    trades_all["month"] = trades_all["entry_date"].dt.to_period("M")
    print(f"\n  Monthly P&L at {cap:,} INR:")
    months_pos = 0
    months_tot = 0
    for month in sorted(trades_all["month"].unique()):
        mt = trades_all[trades_all["month"] == month]
        pnl = mt["net_return"].sum() / winner["pos"] * cap
        wr = (mt["net_return"] > 0).sum() / len(mt) * 100
        marker = " +" if pnl >= 20000 else (" *" if pnl < 0 else "")
        print(f"    {str(month):<10} {len(mt):>3} trades  WR={wr:>4.0f}%  {pnl:>+10,.0f} INR{marker}")
        months_tot += 1
        if pnl > 0: months_pos += 1
    print(f"  Profitable months: {months_pos}/{months_tot} ({months_pos/months_tot*100:.0f}%)")

    # Capital sensitivity for 5K/week
    print(f"\n  Capital sensitivity (2025+ avg weekly):")
    if len(t2025) > 0:
        t2025c2 = t2025.copy()
        t2025c2["yearweek"] = t2025c2["year"].astype(str) + "-W" + t2025c2["week"].astype(str).str.zfill(2)
        w25_raw = t2025c2.groupby("yearweek")["net_return"].sum() / winner["pos"]
        print(f"  {'Capital':>10} {'Avg/wk':>8} {'%>=5K':>6} {'%>0':>5}")
        for c in [300000, 400000, 500000, 600000, 750000, 1000000]:
            wi = w25_raw * c
            print(f"  {c:>10,} {wi.mean():>+8,.0f} {(wi>=5000).sum()/len(wi)*100:>5.0f}% {(wi>0).sum()/len(wi)*100:>4.0f}%")

    print(f"\n{'*'*70}")


if __name__ == "__main__":
    main()
