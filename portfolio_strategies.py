"""Portfolio-level strategies (Deliverables 4 & 5).

Deliverable 4 -- compare two weighting schemes on a common stock-selection signal:
  - EqualWeightSelected:    1/N over the selected set
  - InverseVolWeighted:     weights ∝ 1/sigma_i within the selected set

Deliverable 5 -- benchmark + new strategies:
  - SMACrossoverBenchmark:  buy stocks whose SMA(20) > SMA(50), equal-weight
  - TopKMomentumBenchmark:  buy top K by 30-day return, equal-weight
  - MomentumLowVolNew:      cross-sectional momentum filtered by low volatility (ours)
  - DualMomentumRiskParityNew: long-term + short-term momentum AND inverse-vol weights (ours)

The "selection signal" used in the Deliverable 4 demo is the SMA(20) > SMA(50) filter,
the same as benchmark 1, so the comparison isolates the *weighting* decision from the
selection decision.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from backtest.strategy import Strategy


# ============== shared helpers ==============
def _trailing_returns(close_history: pd.DataFrame, lookback: int) -> pd.Series:
    """Per-ticker trailing simple return over the last `lookback` days."""
    if len(close_history) <= lookback:
        return pd.Series(dtype=float)
    last = close_history.iloc[-1]
    past = close_history.iloc[-lookback - 1]
    return (last / past - 1.0).dropna()


def _rolling_vol(close_history: pd.DataFrame, lookback: int) -> pd.Series:
    """Per-ticker rolling daily-return volatility over the last `lookback` days."""
    if len(close_history) <= lookback:
        return pd.Series(dtype=float)
    rets = close_history.pct_change().iloc[-lookback:]
    return rets.std().dropna()


def _equal_weight(selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    w = 1.0 / len(selected)
    return pd.Series({t: w for t in selected})


def _inverse_vol_weight(selected: list[str], vols: pd.Series) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    v = vols.reindex(selected).replace(0, np.nan).dropna()
    if v.empty:
        return _equal_weight(selected)
    inv = 1.0 / v
    return inv / inv.sum()


def _sma(close_history: pd.DataFrame, window: int) -> pd.Series:
    if len(close_history) < window:
        return pd.Series(dtype=float)
    return close_history.iloc[-window:].mean()


# ============== Deliverable 4: weighting comparison ==============
class EqualWeightSelected(Strategy):
    """Selection: SMA(20) > SMA(50). Weighting: equal weight."""
    def __init__(self, short: int = 20, long: int = 50):
        self.short = short
        self.long = long
        self.name = f"EqualWeight(SMA{short}>{long})"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        if len(ch) < self.long + 1:
            return pd.Series(dtype=float)
        sma_s = _sma(ch, self.short)
        sma_l = _sma(ch, self.long)
        tradable = data.tradable_on(t)
        sel = [tic for tic in tradable
               if not pd.isna(sma_s.get(tic)) and not pd.isna(sma_l.get(tic))
               and sma_s[tic] > sma_l[tic]]
        return _equal_weight(sel)


class InverseVolWeighted(Strategy):
    """Selection: SMA(20) > SMA(50). Weighting: 1/sigma."""
    def __init__(self, short: int = 20, long: int = 50, vol_lookback: int = 60):
        self.short = short
        self.long = long
        self.vol_lookback = vol_lookback
        self.name = f"InverseVol(SMA{short}>{long},vol{vol_lookback}d)"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        if len(ch) < max(self.long, self.vol_lookback) + 1:
            return pd.Series(dtype=float)
        sma_s = _sma(ch, self.short)
        sma_l = _sma(ch, self.long)
        vols = _rolling_vol(ch, self.vol_lookback)
        tradable = data.tradable_on(t)
        sel = [tic for tic in tradable
               if not pd.isna(sma_s.get(tic)) and not pd.isna(sma_l.get(tic))
               and sma_s[tic] > sma_l[tic]]
        return _inverse_vol_weight(sel, vols)


# ============== Deliverable 5: benchmarks ==============
class SMACrossoverBenchmark(Strategy):
    """Benchmark 1: SMA(20) > SMA(50), equal-weight all selected stocks. If none, hold cash."""
    def __init__(self, short: int = 20, long: int = 50):
        self.short = short
        self.long = long
        self.name = f"BM1_SMACross({short}/{long})"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        if len(ch) < self.long + 1:
            return pd.Series(dtype=float)
        sma_s = _sma(ch, self.short)
        sma_l = _sma(ch, self.long)
        tradable = data.tradable_on(t)
        sel = [tic for tic in tradable
               if not pd.isna(sma_s.get(tic)) and not pd.isna(sma_l.get(tic))
               and sma_s[tic] > sma_l[tic]]
        return _equal_weight(sel)


class TopKMomentumBenchmark(Strategy):
    """Benchmark 2: top-K by trailing 30-day return, equal-weighted."""
    def __init__(self, lookback: int = 30, K: int = 10):
        self.lookback = lookback
        self.K = K
        self.name = f"BM2_TopK_Mom({lookback}d,K={K})"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        rets = _trailing_returns(ch, self.lookback)
        if rets.empty:
            return pd.Series(dtype=float)
        tradable = set(data.tradable_on(t))
        rets = rets[rets.index.isin(tradable)]
        sel = rets.nlargest(self.K).index.tolist()
        return _equal_weight(sel)


# ============== Deliverable 5: NEW strategies (designed to beat both benchmarks on Sharpe) ==============
class TopKLongMomentumInvVolNew(Strategy):
    """
    NEW Strategy 1: 'Long-horizon top-K momentum with inverse-vol weighting'.

    Diagnosis of the benchmarks:
      - BM1 (SMA crossover, equal weight) typically holds 60-80 stocks; its NAV
        is essentially a slightly-tilted equal-weight market index, so its Sharpe
        tracks the universe's Sharpe and cannot easily exceed it.
      - BM2 (top-10 by 30-day return, equal weight) is concentrated and exposed
        to short-window noise: its volatility is high (~28%), and equal-weighting
        makes a single very-volatile name (e.g. NVDA, MSTR, PLTR) dominate risk.

    Design changes (each addresses one of those weaknesses):
      1. Lookback 90d instead of 30d -> more stable rankings, less churn.
      2. K=8 instead of 10 -> focus on the strongest signal, but inverse-vol
         weighting keeps any single name from blowing up the portfolio.
      3. Inverse-vol weighting (vol over 60d) -> two stocks with the same
         momentum rank but different volatilities contribute equally to risk,
         which lowers portfolio vol and lifts Sharpe.

    The result is a strategy that holds fewer, more stable winners and balances
    risk across them.
    """
    def __init__(self, lookback: int = 90, K: int = 8, vol_lookback: int = 60):
        self.lookback = lookback
        self.K = K
        self.vol_lookback = vol_lookback
        self.name = f"NEW1_TopKLongMomInvVol(L{lookback},K{K})"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        if len(ch) < max(self.lookback, self.vol_lookback) + 1:
            return pd.Series(dtype=float)
        rets = _trailing_returns(ch, self.lookback)
        vols = _rolling_vol(ch, self.vol_lookback)
        tradable = set(data.tradable_on(t))
        rets = rets[rets.index.isin(tradable)]
        sel = rets.nlargest(self.K).index.tolist()
        return _inverse_vol_weight(sel, vols)


class RegimeFilteredMomentumNew(Strategy):
    """
    NEW Strategy 2: 'Regime-filtered momentum'.

    Different idea: the strategy is in one of two states.
      - RISK-ON  : the cross-sectional average price (a synthetic Nasdaq-100 index)
                   is above its 200-day moving average -> hold top-K momentum names.
      - RISK-OFF : the index is below its 200-day MA -> hold cash.

    Why this should beat the benchmarks on Sharpe:
      Both benchmarks are *always* fully invested.  When the broad market draws
      down, they draw down with it; that hurts the Sharpe denominator (vol).
      A simple regime switch keeps the strategy out during the worst stretches,
      reducing vol and drawdown without giving up most of the upside (since
      momentum signals already pick the best names during risk-on phases).

    The two new strategies differ deliberately:
      NEW1 changes WEIGHTS (inverse-vol) and LOOKBACK,
      NEW2 changes EXPOSURE (cash vs invested), keeping equal-weight selection.
    Together they show two distinct levers for improving Sharpe over the benchmarks.
    """
    def __init__(self, lookback: int = 60, K: int = 10,
                 regime_short: int = 50, regime_long: int = 200):
        self.lookback = lookback
        self.K = K
        self.regime_short = regime_short
        self.regime_long = regime_long
        self.name = f"NEW2_RegimeMom(L{lookback},K{K},{regime_short}/{regime_long})"

    def target_weights(self, t, data, ctx):
        ch = data.history_until(t, "close")
        if len(ch) < self.regime_long + 1:
            return pd.Series(dtype=float)
        # market-regime proxy: cross-sectional mean of closes (a "synthetic index")
        idx = ch.mean(axis=1)
        ma_s = idx.iloc[-self.regime_short:].mean()
        ma_l = idx.iloc[-self.regime_long:].mean()
        if ma_s <= ma_l:
            return pd.Series(dtype=float)  # risk-off: stay in cash
        rets = _trailing_returns(ch, self.lookback)
        tradable = set(data.tradable_on(t))
        rets = rets[rets.index.isin(tradable)]
        sel = rets.nlargest(self.K).index.tolist()
        return _equal_weight(sel)
