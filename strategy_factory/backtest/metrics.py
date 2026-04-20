"""
Backtest metrics: Sharpe, MDD, PF, WR, CAGR calculations.
"""
import numpy as np
import pandas as pd


def win_rate(returns: pd.Series) -> float:
    """Percentage of trades that were profitable."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns) * 100


def profit_factor(returns: pd.Series) -> float:
    """Sum of gains / abs(sum of losses)."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def sharpe_ratio(returns: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    # returns here are per-trade, so annualize based on avg hold period
    mean_ret = returns.mean()
    std_ret = returns.std()
    # Approximate: scale by sqrt of trades per year
    trades_per_year = min(len(returns), annual_factor)
    return (mean_ret / std_ret) * np.sqrt(trades_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative percentage."""
    if len(equity_curve) == 0:
        return 0.0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100


def cagr(equity_curve: pd.Series, periods_per_year: float = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_years = len(equity_curve) / periods_per_year
    if n_years <= 0 or total_return <= 0:
        return 0.0
    return (total_return ** (1 / n_years) - 1) * 100


def compute_all_metrics(trade_returns: pd.Series, equity_curve: pd.Series) -> dict:
    """Compute all backtest metrics."""
    return {
        "trades": len(trade_returns),
        "win_rate": win_rate(trade_returns),
        "profit_factor": profit_factor(trade_returns),
        "sharpe": sharpe_ratio(trade_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve),
        "avg_return": trade_returns.mean() * 100 if len(trade_returns) > 0 else 0,
        "median_return": trade_returns.median() * 100 if len(trade_returns) > 0 else 0,
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics as a readable string."""
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    return (
        f"Trades: {metrics['trades']} | "
        f"WR: {metrics['win_rate']:.1f}% | "
        f"PF: {pf_str} | "
        f"Sharpe: {metrics['sharpe']:.2f} | "
        f"MDD: {metrics['max_drawdown']:.1f}% | "
        f"CAGR: {metrics['cagr']:.1f}%"
    )
