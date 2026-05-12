"""
Main entry point for the INDENG 231 Project 1 backtesting system.

Run:
    python run_backtests.py

Produces:
    results/d3_single_stock_metrics.csv
    results/d4_portfolio_weighting_metrics.csv
    results/d5_benchmarks_vs_new_metrics.csv
    results/all_navs.csv                      <- raw NAV curves
    results/figures/*.png                     <- PNL/NAV plots
    results/run_log.txt                       <- experiment log
"""

from __future__ import annotations
import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running this file directly from the project root
sys.path.insert(0, str(Path(__file__).parent))

from backtest import DataHandler, BacktestEngine, metrics
from strategies.single_stock import (
    MomentumSingle, MeanReversionSingle, SMACrossoverSingle,
    RSIContrarianSingle, BollingerBreakoutSingle,
)
from strategies.portfolio_strategies import (
    EqualWeightSelected, InverseVolWeighted,
    SMACrossoverBenchmark, TopKMomentumBenchmark,
    TopKLongMomentumInvVolNew, RegimeFilteredMomentumNew,
)

# ----- config -----
DATA_PATH = Path(__file__).parent / "data.csv"
RESULTS = Path(__file__).parent / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

LOG_PATH = RESULTS / "run_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("main")


def buy_and_hold_nav(data: DataHandler, ticker: str, warmup: int) -> pd.Series:
    """Compute a buy-and-hold NAV starting from day `warmup` for one ticker.
    Used as a passive benchmark for single-stock comparisons."""
    s = data.close[ticker].iloc[warmup:].dropna()
    return (s / s.iloc[0]) * 1_000_000.0


def equal_weight_universe_nav(data: DataHandler, warmup: int) -> pd.Series:
    """Daily-rebalanced equal-weight portfolio across the tradable universe.
    Used as a passive benchmark for portfolio-level comparisons."""
    ch = data.close.iloc[warmup:]
    rets = ch.pct_change()
    # equal weight across whatever's tradable each day (drop NaN per row)
    avg_ret = rets.apply(lambda row: row.dropna().mean(), axis=1).fillna(0)
    nav = (1 + avg_ret).cumprod() * 1_000_000.0
    return pd.Series(nav.values, index=ch.index, name="EW_Universe")


def plot_navs(navs: dict, title: str, fname: str, normalize: bool = True):
    plt.figure(figsize=(11, 5.5))
    for name, nav in navs.items():
        s = nav / nav.iloc[0] if normalize else nav
        plt.plot(s.index, s.values, label=name, linewidth=1.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("NAV (normalized to 1.0)" if normalize else "NAV ($)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / fname, dpi=130)
    plt.close()


def run_strategy(data, strategy, warmup=60, cost_bps=0.0):
    log.info(f"  -> Running {strategy.name}")
    eng = BacktestEngine(data, strategy, warmup_days=warmup, cost_bps=cost_bps)
    res = eng.run()
    return res


# ============================================================
def main():
    log.info("Loading data...")
    data = DataHandler(DATA_PATH)
    log.info(f"Loaded: {data}")

    WARMUP = 210          # ~10 months — covers the regime filter's 200-day MA
    COST_BPS = 0.0        # frictionless headline; toggle to 5-10 to see realistic erosion

    # we'll save every NAV here to enable cross-section reporting
    all_navs = {}

    # ----------------------------------------------------------------------
    # DELIVERABLE 3: single-stock strategy comparison on AAPL
    # ----------------------------------------------------------------------
    log.info("\n=== DELIVERABLE 3: single-stock strategies on AAPL ===")
    TICKER = "AAPL"

    single_strats = [
        MomentumSingle(TICKER, lookback=20),
        MeanReversionSingle(TICKER, lookback=20, z_threshold=1.0),
        SMACrossoverSingle(TICKER, short=20, long=50),
        RSIContrarianSingle(TICKER, period=14, oversold=30, overbought=70),
        BollingerBreakoutSingle(TICKER, lookback=20, k=2.0),
    ]

    d3_navs = {}
    for s in single_strats:
        res = run_strategy(data, s, warmup=WARMUP, cost_bps=COST_BPS)
        d3_navs[s.name] = res["nav"]
        all_navs[s.name] = res["nav"]
    # passive buy-and-hold reference
    bh = buy_and_hold_nav(data, TICKER, WARMUP)
    d3_navs[f"BuyHold({TICKER})"] = bh
    all_navs[f"BuyHold({TICKER})"] = bh

    d3_metrics = metrics.summary_table(d3_navs).round(4)
    d3_metrics.to_csv(RESULTS / "d3_single_stock_metrics.csv")
    log.info("\n" + d3_metrics.to_string())
    plot_navs(d3_navs, f"Deliverable 3 — Single-stock strategies on {TICKER}",
              "d3_single_stock_navs.png")

    # ----------------------------------------------------------------------
    # DELIVERABLE 4: portfolio weighting comparison
    # ----------------------------------------------------------------------
    log.info("\n=== DELIVERABLE 4: portfolio weighting comparison ===")
    d4_strats = [
        EqualWeightSelected(short=20, long=50),
        InverseVolWeighted(short=20, long=50, vol_lookback=60),
    ]
    d4_navs = {}
    for s in d4_strats:
        res = run_strategy(data, s, warmup=WARMUP, cost_bps=COST_BPS)
        d4_navs[s.name] = res["nav"]
        all_navs[s.name] = res["nav"]
    ew_uni = equal_weight_universe_nav(data, WARMUP)
    ew_uni = ew_uni.reindex(next(iter(d4_navs.values())).index).ffill()
    d4_navs["EW_Universe(passive)"] = ew_uni
    all_navs["EW_Universe(passive)"] = ew_uni

    d4_metrics = metrics.summary_table(d4_navs).round(4)
    d4_metrics.to_csv(RESULTS / "d4_portfolio_weighting_metrics.csv")
    log.info("\n" + d4_metrics.to_string())
    plot_navs(d4_navs, "Deliverable 4 — Equal-weight vs Inverse-vol weighting",
              "d4_weighting.png")

    # ----------------------------------------------------------------------
    # DELIVERABLE 5: benchmarks vs two new strategies
    # ----------------------------------------------------------------------
    log.info("\n=== DELIVERABLE 5: benchmarks vs new strategies ===")
    d5_strats = [
        SMACrossoverBenchmark(short=20, long=50),
        TopKMomentumBenchmark(lookback=30, K=10),
        TopKLongMomentumInvVolNew(lookback=90, K=8, vol_lookback=60),
        RegimeFilteredMomentumNew(lookback=60, K=10, regime_short=50, regime_long=200),
    ]
    d5_navs = {}
    for s in d5_strats:
        res = run_strategy(data, s, warmup=WARMUP, cost_bps=COST_BPS)
        d5_navs[s.name] = res["nav"]
        all_navs[s.name] = res["nav"]

    d5_metrics = metrics.summary_table(d5_navs).round(4)
    d5_metrics.to_csv(RESULTS / "d5_benchmarks_vs_new_metrics.csv")
    log.info("\n" + d5_metrics.to_string())
    plot_navs(d5_navs, "Deliverable 5 — Benchmarks vs New Strategies",
              "d5_benchmarks_vs_new.png")

    # bar chart of Sharpe ratios for D5 (the headline metric)
    plt.figure(figsize=(9, 4.5))
    sr = d5_metrics["sharpe"].sort_values()
    colors = ["#888" if "BM" in n else "#1f77b4" for n in sr.index]
    plt.barh(sr.index, sr.values, color=colors)
    plt.title("Deliverable 5 — Sharpe ratio (blue = new, gray = benchmark)")
    plt.xlabel("Sharpe ratio")
    plt.tight_layout()
    plt.savefig(FIGS / "d5_sharpe_bars.png", dpi=130)
    plt.close()

    # ----------------------------------------------------------------------
    # final: dump all NAVs together so the report can re-plot anything
    # ----------------------------------------------------------------------
    nav_df = pd.DataFrame(all_navs)
    nav_df.to_csv(RESULTS / "all_navs.csv")

    log.info(f"\nAll done. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
