#!/usr/bin/env python3
"""
Sweep target % and stop % to find the optimal combination.
Reuses the same trade entry data from combined strategy backtest.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import NIFTY_200, NIFTY_500, START_DATE, END_DATE, NIFTY50_TICKER
from signals.scanner import download_universe, download_stock_data, build_panel
from research_5k_v5 import get_nifty_regime
from research_deep import add_advanced_features
from research_bull_fresh import add_bull_features
from backtest_combined_1y import (
    compute_regime_for_date, get_bear_picks, get_bull_breakout_picks,
    get_bull_accel_picks, get_bull_relstr_picks, get_bull_dip_picks,
    select_finalists,
)

CAPITAL = 400000
MAX_POS = 5
COST = 0.001


def simulate_entries(stock_data_n500, stock_data_n200, panel_n500, panel_n200, nifty_data, test_dates, all_dates):
    """Get all entry points — same logic as combined backtest, but store raw OHLC for flexible exit testing."""
    entries = []
    open_tickers = set()
    cooldown = {}  # ticker -> unblock_date (prevent re-entry of same stock for 5 days)

    for date in test_dates:
        # Simple cooldown: don't re-enter same stock within 5 days
        cooldown = {t: d for t, d in cooldown.items() if d > date}

        available = MAX_POS - len(open_tickers)
        if available <= 0:
            # Check if any open positions would have exited by now (rough estimate)
            # For sweep, we'll be more lenient — just cap entries per day
            open_tickers.clear()
            available = MAX_POS

        regime = compute_regime_for_date(date, nifty_data, panel_n200)
        if regime is None:
            continue

        all_picks = []
        all_picks.extend(get_bear_picks(date, panel_n200, regime))
        all_picks.extend(get_bull_breakout_picks(date, panel_n500, regime))
        all_picks.extend(get_bull_accel_picks(date, panel_n200, regime))
        all_picks.extend(get_bull_relstr_picks(date, panel_n500, regime))
        all_picks.extend(get_bull_dip_picks(date, panel_n200, regime))

        all_picks = [p for p in all_picks if p["ticker"] not in open_tickers and p["ticker"] not in cooldown]

        finalists = select_finalists(all_picks, max_pos=available)

        for pick in finalists:
            ticker = pick["ticker"]
            if ticker not in stock_data_n500:
                continue

            df = stock_data_n500[ticker]
            try:
                idx = df.index.get_indexer([date], method="nearest")[0]
            except Exception:
                continue

            entry_idx = idx + 1
            if entry_idx >= len(df) - 1:
                continue

            entry_price = df.iloc[entry_idx]["Open"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Store the forward price data for this entry (up to 30 days)
            forward_data = []
            for di in range(entry_idx, min(entry_idx + 31, len(df))):
                forward_data.append({
                    "day": di - entry_idx,
                    "open": df.iloc[di]["Open"],
                    "high": df.iloc[di]["High"],
                    "low": df.iloc[di]["Low"],
                    "close": df.iloc[di]["Close"],
                })

            entries.append({
                "signal_date": date,
                "entry_date": df.index[entry_idx],
                "ticker": ticker,
                "entry_price": entry_price,
                "n_strats": pick["n_strats"],
                "forward_data": forward_data,
            })

            open_tickers.add(ticker)
            cooldown[ticker] = date + pd.Timedelta(days=7)

    return entries


def simulate_target_stop(entries, target_pct, stop_pct, max_hold, per_pos):
    """Simulate target/stop exit for all entries. Returns summary stats."""
    results = []

    for entry in entries:
        entry_price = entry["entry_price"]
        target_price = entry_price * (1 + target_pct)
        stop_price = entry_price * (1 + stop_pct)  # stop_pct is negative

        exit_reason = None
        exit_day = 0
        exit_price = entry_price

        for fd in entry["forward_data"]:
            day = fd["day"]
            if day == 0:
                continue  # Entry day

            # Check stop (intraday low)
            if fd["low"] <= stop_price:
                exit_reason = "STOP"
                exit_price = stop_price
                exit_day = day
                break

            # Check target (close)
            if fd["close"] >= target_price:
                exit_reason = "TARGET"
                exit_price = target_price
                exit_day = day
                break

            if day >= max_hold:
                exit_reason = "MAX_HOLD"
                exit_price = fd["close"]
                exit_day = day
                break

        if exit_reason is None:
            # Ran out of data
            last = entry["forward_data"][-1]
            exit_reason = "END"
            exit_price = last["close"]
            exit_day = last["day"]

        ret = (exit_price / entry_price - 1) - COST
        pnl = ret * per_pos

        results.append({
            "ticker": entry["ticker"],
            "entry_date": entry["entry_date"],
            "ret": ret,
            "pnl": pnl,
            "days": exit_day,
            "reason": exit_reason,
            "n_strats": entry["n_strats"],
        })

    return pd.DataFrame(results)


def main():
    print(f"Loading data...")
    stock_data_n500 = download_universe(NIFTY_500, START_DATE, END_DATE)
    stock_data_n200 = {k: v for k, v in stock_data_n500.items() if k in NIFTY_200}
    index_data = download_stock_data(NIFTY50_TICKER, START_DATE, END_DATE)

    print(f"Building panels...")
    panel_n500 = build_panel(stock_data_n500, index_data)
    panel_n500 = add_advanced_features(panel_n500, stock_data_n500, index_data)
    panel_n500 = add_bull_features(panel_n500, stock_data_n500, index_data)
    panel_n200 = panel_n500[panel_n500["ticker"].isin(NIFTY_200)].copy()

    all_dates = sorted(panel_n200["date"].unique())
    one_year_ago = pd.Timestamp(datetime.now() - timedelta(days=365))
    test_dates = [d for d in all_dates if d >= one_year_ago]

    print(f"Generating entries...")
    entries = simulate_entries(stock_data_n500, stock_data_n200, panel_n500, panel_n200, index_data, test_dates, all_dates)
    print(f"  Total entries: {len(entries)}")

    PER_POS = CAPITAL // MAX_POS

    # ============================================================
    # SWEEP
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  TARGET % vs STOP % SWEEP")
    print(f"  Capital: {CAPITAL:,} | Per pos: {PER_POS:,} | Entries: {len(entries)}")
    print(f"{'='*90}")

    targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    stops = [-2.0, -3.0, -4.0, -5.0, -6.0, -8.0]
    max_holds = [5, 10, 15, 20, 30]

    # First: sweep target vs stop with fixed max_hold=20
    print(f"\n  Target vs Stop (max hold: 20 days):")
    print(f"  {'':>8}", end="")
    for s in stops:
        print(f"  Stop {s}%", end="")
    print()

    grid = {}
    for t in targets:
        print(f"  Tgt {t}%", end="")
        for s in stops:
            rdf = simulate_target_stop(entries, t/100, s/100, 20, PER_POS)
            total_pnl = rdf["pnl"].sum()
            n_target = (rdf["reason"] == "TARGET").sum()
            n_stop = (rdf["reason"] == "STOP").sum()
            tgt_rate = n_target / len(rdf) * 100
            grid[(t, s)] = {"pnl": total_pnl, "tgt_rate": tgt_rate, "n_target": n_target, "n_stop": n_stop}

            if total_pnl >= 0:
                print(f"  {total_pnl/1000:>+6.0f}K", end="")
            else:
                print(f"  {total_pnl/1000:>+6.0f}K", end="")
        print()

    # Find best combination
    best = max(grid.items(), key=lambda x: x[1]["pnl"])
    best_t, best_s = best[0]
    best_pnl = best[1]["pnl"]

    print(f"\n  BEST: Target {best_t}% / Stop {best_s}% → P&L: {best_pnl:+,.0f}")

    # Now sweep max_hold for the best target/stop
    print(f"\n  Max Hold sweep for Target {best_t}% / Stop {best_s}%:")
    print(f"  {'MaxHold':>8} {'P&L':>10} {'Trades':>6} {'TgtHit':>6} {'StpHit':>6} {'MxHold':>6} {'Tgt%':>5} {'AvgDays':>7}")

    best_overall = None
    best_overall_pnl = -999999

    for mh in max_holds:
        rdf = simulate_target_stop(entries, best_t/100, best_s/100, mh, PER_POS)
        total_pnl = rdf["pnl"].sum()
        n_target = (rdf["reason"] == "TARGET").sum()
        n_stop = (rdf["reason"] == "STOP").sum()
        n_max = (rdf["reason"] == "MAX_HOLD").sum()
        tgt_rate = n_target / len(rdf) * 100
        avg_days = rdf[rdf["reason"] == "TARGET"]["days"].mean() if n_target > 0 else 0
        marker = " <<" if total_pnl > best_overall_pnl else ""
        if total_pnl > best_overall_pnl:
            best_overall = (best_t, best_s, mh)
            best_overall_pnl = total_pnl
        print(f"  {mh:>6}d {total_pnl:>+10,.0f} {len(rdf):>6} {n_target:>6} {n_stop:>6} {n_max:>6} {tgt_rate:>4.0f}% {avg_days:>6.1f}d{marker}")

    # Also test top 5 grid combos with all max_holds
    print(f"\n  TOP 10 OVERALL CONFIGURATIONS:")
    all_configs = []
    for t in targets:
        for s in stops:
            for mh in max_holds:
                rdf = simulate_target_stop(entries, t/100, s/100, mh, PER_POS)
                total_pnl = rdf["pnl"].sum()
                n_target = (rdf["reason"] == "TARGET").sum()
                n_stop = (rdf["reason"] == "STOP").sum()
                tgt_rate = n_target / len(rdf) * 100
                avg_days_tgt = rdf[rdf["reason"] == "TARGET"]["days"].mean() if n_target > 0 else 0
                all_configs.append({
                    "target": t, "stop": s, "max_hold": mh,
                    "pnl": total_pnl, "trades": len(rdf),
                    "tgt_hit": n_target, "stp_hit": n_stop,
                    "tgt_rate": tgt_rate, "avg_days": avg_days_tgt,
                })

    cdf = pd.DataFrame(all_configs).sort_values("pnl", ascending=False)

    print(f"  {'Tgt%':>5} {'Stp%':>5} {'MaxH':>5} {'P&L':>10} {'Trades':>6} {'TgtHit':>6} {'StpHit':>6} {'Tgt%':>5} {'AvgD':>5}")
    for _, r in cdf.head(10).iterrows():
        print(f"  {r['target']:>4.1f}% {r['stop']:>4.0f}% {r['max_hold']:>4}d {r['pnl']:>+10,.0f} {r['trades']:>6} "
              f"{r['tgt_hit']:>6} {r['stp_hit']:>6} {r['tgt_rate']:>4.0f}% {r['avg_days']:>4.1f}d")

    # Deep dive on winner
    winner = cdf.iloc[0]
    print(f"\n\n{'*'*70}")
    print(f"  OPTIMAL: Target +{winner['target']:.1f}% / Stop {winner['stop']:.0f}% / Max hold {winner['max_hold']:.0f}d")
    print(f"{'*'*70}")

    rdf = simulate_target_stop(entries, winner["target"]/100, winner["stop"]/100, int(winner["max_hold"]), PER_POS)

    total_pnl = rdf["pnl"].sum()
    n_target = (rdf["reason"] == "TARGET").sum()
    n_stop = (rdf["reason"] == "STOP").sum()
    n_max = (rdf["reason"] == "MAX_HOLD").sum()

    print(f"\n  Total P&L:       {total_pnl:>+12,.0f} INR")
    print(f"  Total trades:    {len(rdf):>12}")
    print(f"  Target hits:     {n_target:>12} ({n_target/len(rdf)*100:.0f}%)")
    print(f"  Stop hits:       {n_stop:>12} ({n_stop/len(rdf)*100:.0f}%)")
    print(f"  Max hold exits:  {n_max:>12} ({n_max/len(rdf)*100:.0f}%)")
    print(f"  Avg days to tgt: {rdf[rdf['reason']=='TARGET']['days'].mean():>11.1f}d")
    print(f"  Avg P&L/trade:   {rdf['pnl'].mean():>+11,.0f} INR")

    # P&L breakdown
    tgt_pnl = rdf[rdf["reason"] == "TARGET"]["pnl"].sum()
    stp_pnl = rdf[rdf["reason"] == "STOP"]["pnl"].sum()
    max_pnl = rdf[rdf["reason"] == "MAX_HOLD"]["pnl"].sum()
    print(f"\n  From targets:    {tgt_pnl:>+12,.0f}")
    print(f"  From stops:      {stp_pnl:>+12,.0f}")
    print(f"  From max holds:  {max_pnl:>+12,.0f}")

    # Weekly P&L
    rdf["week"] = rdf["entry_date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = rdf.groupby("week")["pnl"].sum()
    print(f"\n  Weekly breakdown:")
    print(f"  Weeks with trades: {len(weekly)}")
    print(f"  Weeks profitable:  {(weekly>0).sum()} ({(weekly>0).sum()/len(weekly)*100:.0f}%)")
    print(f"  Avg week P&L:      {weekly.mean():+,.0f}")
    print(f"  Best week:         {weekly.max():+,.0f}")
    print(f"  Worst week:        {weekly.min():+,.0f}")

    # Compare: target exit vs fixed hold (no target, just hold max_hold days)
    rdf_fixed = simulate_target_stop(entries, 999, winner["stop"]/100, int(winner["max_hold"]), PER_POS)
    fixed_pnl = rdf_fixed["pnl"].sum()
    print(f"\n  COMPARISON:")
    print(f"  Target exit ({winner['target']:.1f}%/{winner['stop']:.0f}%/{winner['max_hold']:.0f}d): {total_pnl:>+10,.0f}")
    print(f"  Fixed hold ({winner['max_hold']:.0f}d, stop {winner['stop']:.0f}%):        {fixed_pnl:>+10,.0f}")
    diff = total_pnl - fixed_pnl
    print(f"  Target exit advantage:             {diff:>+10,.0f} {'BETTER' if diff > 0 else 'WORSE'}")


if __name__ == "__main__":
    main()
