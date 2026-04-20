"""
Quick Backtest Engine.

Given a signal condition, runs a simple long-only backtest:
- Buy stocks in the best quintile (or matching a threshold)
- Hold for N days
- Equal weight, max 10 stocks at a time
- 0.1% transaction cost
- -8% hard stop-loss
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_POSITIONS, COST_PER_TRADE, STOP_LOSS,
    DEFAULT_HOLD_DAYS, MIN_TRADES_PER_YEAR, FORWARD_RETURN_DAYS,
)
from backtest.metrics import compute_all_metrics, format_metrics


class BacktestEngine:
    """
    Simple event-driven backtest engine for long-only strategies.
    """

    def __init__(
        self,
        stock_data: dict,
        panel: pd.DataFrame,
        signal_col: str,
        signal_condition: str = "quintile_1",  # "quintile_1", "quintile_5", "below_threshold", "above_threshold"
        threshold: float = None,
        hold_days: int = DEFAULT_HOLD_DAYS,
        max_positions: int = MAX_POSITIONS,
        cost: float = COST_PER_TRADE,
        stop_loss: float = STOP_LOSS,
        rebalance_freq: str = "daily",  # "daily", "weekly", "monthly"
    ):
        self.stock_data = stock_data
        self.panel = panel
        self.signal_col = signal_col
        self.signal_condition = signal_condition
        self.threshold = threshold
        self.hold_days = hold_days
        self.max_positions = max_positions
        self.cost = cost
        self.stop_loss = stop_loss
        self.rebalance_freq = rebalance_freq

    def _get_eligible_dates(self) -> list:
        """Get dates when we check for new entries based on rebalance frequency."""
        all_dates = sorted(self.panel["date"].unique())
        if self.rebalance_freq == "daily":
            return all_dates
        elif self.rebalance_freq == "weekly":
            # First trading day of each week
            dates_series = pd.Series(all_dates)
            weeks = dates_series.dt.isocalendar().week
            mask = weeks != weeks.shift(1)
            return dates_series[mask].tolist()
        elif self.rebalance_freq == "monthly":
            dates_series = pd.Series(all_dates)
            months = dates_series.dt.month
            mask = months != months.shift(1)
            return dates_series[mask].tolist()
        return all_dates

    def _select_stocks(self, date) -> list:
        """Select stocks meeting signal condition on a given date."""
        day_data = self.panel[self.panel["date"] == date].dropna(subset=[self.signal_col])

        if len(day_data) == 0:
            return []

        if self.signal_condition == "quintile_1":
            # Bottom quintile (lowest signal values)
            try:
                day_data = day_data.copy()
                day_data["_q"] = pd.qcut(day_data[self.signal_col], 5, labels=False, duplicates="drop")
                selected = day_data[day_data["_q"] == 0]["ticker"].tolist()
            except ValueError:
                return []
        elif self.signal_condition == "quintile_5":
            try:
                day_data = day_data.copy()
                day_data["_q"] = pd.qcut(day_data[self.signal_col], 5, labels=False, duplicates="drop")
                selected = day_data[day_data["_q"] == 4]["ticker"].tolist()
            except ValueError:
                return []
        elif self.signal_condition == "below_threshold" and self.threshold is not None:
            selected = day_data[day_data[self.signal_col] < self.threshold]["ticker"].tolist()
        elif self.signal_condition == "above_threshold" and self.threshold is not None:
            selected = day_data[day_data[self.signal_col] > self.threshold]["ticker"].tolist()
        else:
            return []

        return selected[:self.max_positions]

    def _get_stock_return(self, ticker: str, entry_date, hold_days: int) -> dict:
        """
        Compute actual return for a single trade with stop-loss.
        Uses daily OHLCV data to check for stop-loss hits.
        """
        if ticker not in self.stock_data:
            return None

        df = self.stock_data[ticker]
        # Find the entry date index
        try:
            entry_idx = df.index.get_indexer([entry_date], method="nearest")[0]
        except Exception:
            return None

        if entry_idx >= len(df) - 1:
            return None

        # Entry on next day's open (realistic: signal computed at close, enter next day)
        actual_entry_idx = entry_idx + 1
        if actual_entry_idx >= len(df):
            return None

        entry_price = df.iloc[actual_entry_idx]["Open"]
        if pd.isna(entry_price) or entry_price <= 0:
            return None

        # Track daily to check stop-loss
        exit_idx = min(actual_entry_idx + hold_days, len(df) - 1)
        exit_price = entry_price  # default
        stopped_out = False

        for day_idx in range(actual_entry_idx, exit_idx + 1):
            low = df.iloc[day_idx]["Low"]
            close = df.iloc[day_idx]["Close"]

            # Check stop-loss against intraday low
            if (low / entry_price - 1) <= self.stop_loss:
                # Stopped out at stop-loss level
                exit_price = entry_price * (1 + self.stop_loss)
                stopped_out = True
                break

            # On the last day, exit at close
            if day_idx == exit_idx:
                exit_price = close

        gross_return = (exit_price / entry_price) - 1
        net_return = gross_return - self.cost  # Deduct transaction costs

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

    def run(self) -> dict:
        """
        Run the backtest. Returns dict with trades, metrics, equity curve.
        """
        eligible_dates = self._get_eligible_dates()
        trades = []
        open_positions = {}  # ticker -> exit_date

        print(f"\n  Running backtest: {self.signal_col} | {self.signal_condition} | Hold: {self.hold_days}d")
        print(f"  Dates to check: {len(eligible_dates)}")

        for date in eligible_dates:
            # Clean up expired positions
            open_positions = {
                t: ed for t, ed in open_positions.items()
                if ed > date
            }

            # How many slots available?
            available_slots = self.max_positions - len(open_positions)
            if available_slots <= 0:
                continue

            # Select stocks
            candidates = self._select_stocks(date)
            # Remove stocks we already hold
            candidates = [t for t in candidates if t not in open_positions]
            candidates = candidates[:available_slots]

            for ticker in candidates:
                result = self._get_stock_return(ticker, date, self.hold_days)
                if result is not None:
                    trades.append(result)
                    open_positions[ticker] = result["exit_date"]

        if not trades:
            print("  No trades generated!")
            return {"trades": pd.DataFrame(), "metrics": {}, "equity_curve": pd.Series(dtype=float)}

        trades_df = pd.DataFrame(trades)
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

        # Build equity curve
        equity = [1.0]
        for _, trade in trades_df.iterrows():
            # Equal weight per position (simplified)
            position_size = 1.0 / self.max_positions
            pnl = trade["net_return"] * position_size
            equity.append(equity[-1] * (1 + pnl))

        equity_curve = pd.Series(equity)
        trade_returns = trades_df["net_return"]

        metrics = compute_all_metrics(trade_returns, equity_curve)

        return {
            "trades": trades_df,
            "metrics": metrics,
            "equity_curve": equity_curve,
        }


def run_and_report(
    stock_data: dict,
    panel: pd.DataFrame,
    signal_col: str,
    signal_condition: str = "quintile_1",
    threshold: float = None,
    hold_days_list: list = None,
    universe_name: str = "Nifty 200",
):
    """
    Run backtest for multiple hold periods and print a full report.
    """
    if hold_days_list is None:
        hold_days_list = [DEFAULT_HOLD_DAYS]

    dates = panel["date"]
    start = dates.min().strftime("%Y-%m-%d") if hasattr(dates.min(), "strftime") else str(dates.min())
    end = dates.max().strftime("%Y-%m-%d") if hasattr(dates.max(), "strftime") else str(dates.max())

    print(f"\n{'='*70}")
    print(f"  BACKTEST REPORT")
    print(f"  Signal: {signal_col} | Condition: {signal_condition}")
    if threshold is not None:
        print(f"  Threshold: {threshold}")
    print(f"  Universe: {universe_name} | Period: {start} to {end}")
    print(f"{'='*70}")

    for hold_days in hold_days_list:
        engine = BacktestEngine(
            stock_data=stock_data,
            panel=panel,
            signal_col=signal_col,
            signal_condition=signal_condition,
            threshold=threshold,
            hold_days=hold_days,
        )
        result = engine.run()

        if result["trades"].empty:
            print(f"\n  Hold: {hold_days}d — No trades generated")
            continue

        trades_df = result["trades"]
        metrics = result["metrics"]

        # Rule #2: Check minimum trades per year
        date_range_years = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days / 365.25
        trades_per_year = metrics["trades"] / max(date_range_years, 1)

        print(f"\n  --- Hold Period: {hold_days} days ---")
        print(f"  Full Period:")
        print(f"    {format_metrics(metrics)}")
        print(f"    Trades/year: {trades_per_year:.0f}", end="")
        if trades_per_year < MIN_TRADES_PER_YEAR:
            print(f"  << WARNING: Below {MIN_TRADES_PER_YEAR} threshold (Rule #2)")
        else:
            print()

        # Rule #3: Last 12 months separately
        cutoff = trades_df["entry_date"].max() - pd.Timedelta(days=365)
        recent_trades = trades_df[trades_df["entry_date"] >= cutoff]
        if len(recent_trades) > 0:
            recent_returns = recent_trades["net_return"]
            recent_equity = pd.Series([1.0] + [1.0 + r / MAX_POSITIONS for r in recent_returns.cumsum()])
            recent_metrics = compute_all_metrics(recent_returns, recent_equity)
            print(f"  Last 12 Months:")
            print(f"    {format_metrics(recent_metrics)}")
            if recent_metrics["cagr"] < 0:
                print(f"    << WARNING: Negative recent performance (Rule #3 — edge may be dead)")

        # Year-by-year breakdown
        trades_df["year"] = trades_df["entry_date"].dt.year
        years = sorted(trades_df["year"].unique())
        print(f"  Year-by-Year:")
        yearly_results = []
        for year in years:
            year_trades = trades_df[trades_df["year"] == year]
            year_return = year_trades["net_return"].sum() * 100 / MAX_POSITIONS
            yearly_results.append(f"    {year}: {year_return:>+6.1f}% ({len(year_trades)} trades)")
        print("\n".join(yearly_results))
        profitable_years = sum(1 for y in years if trades_df[trades_df["year"] == y]["net_return"].sum() > 0)
        print(f"    Profitable years: {profitable_years}/{len(years)}")

        # Stop-loss stats
        stopped = trades_df["stopped_out"].sum()
        print(f"  Stop-loss hits: {stopped} ({stopped/len(trades_df)*100:.1f}% of trades)")

    print(f"\n{'='*70}")
