"""Performance metrics computed from a NAV (net asset value) series."""

from __future__ import annotations
import pandas as pd
import numpy as np

TRADING_DAYS = 252


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def cumulative_return(nav: pd.Series) -> float:
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def annualized_return(nav: pd.Series) -> float:
    n = len(nav) - 1
    if n <= 0:
        return 0.0
    total = nav.iloc[-1] / nav.iloc[0]
    return float(total ** (TRADING_DAYS / n) - 1.0)


def annualized_vol(nav: pd.Series) -> float:
    r = daily_returns(nav)
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(nav: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe. rf is annual risk-free rate; default 0 (excess over cash)."""
    r = daily_returns(nav)
    if len(r) == 0 or r.std() == 0:
        return 0.0
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def win_rate(nav: pd.Series) -> float:
    r = daily_returns(nav)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def summary(nav: pd.Series, name: str = "strategy") -> dict:
    return {
        "strategy": name,
        "cum_return":  cumulative_return(nav),
        "ann_return":  annualized_return(nav),
        "ann_vol":     annualized_vol(nav),
        "sharpe":      sharpe_ratio(nav),
        "max_drawdown": max_drawdown(nav),
        "win_rate":    win_rate(nav),
    }


def summary_table(navs: dict[str, pd.Series]) -> pd.DataFrame:
    rows = [summary(nav, name) for name, nav in navs.items()]
    return pd.DataFrame(rows).set_index("strategy")
