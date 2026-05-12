"""Portfolio: tracks shares per ticker and cash; rebalances to target weights.

Execution model (matches the project spec)
------------------------------------------
- All trades fill at today's close price.
- No short selling, no leverage: weights are clipped to [0, 1] and rescaled so they sum to <= 1.
- A small per-trade transaction cost (in basis points of notional) can be set; default 0
  to keep the headline numbers comparable to a frictionless reference. Toggling it on
  shows realistic degradation; we report both in the appendix when relevant.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


class Portfolio:
    def __init__(self, initial_cash: float = 1_000_000.0, cost_bps: float = 0.0):
        self.cash = float(initial_cash)
        self.shares = {}            # ticker -> shares (float; fractional allowed)
        self.cost_bps = cost_bps    # one-way transaction cost in basis points

    # ---- valuation ----
    def equity(self, prices: pd.Series) -> float:
        pos_val = sum(s * prices.get(tic, np.nan) for tic, s in self.shares.items()
                      if not pd.isna(prices.get(tic, np.nan)))
        return self.cash + pos_val

    def weights(self, prices: pd.Series) -> pd.Series:
        eq = self.equity(prices)
        if eq <= 0:
            return pd.Series(dtype=float)
        w = {tic: (s * prices.get(tic, np.nan)) / eq
             for tic, s in self.shares.items()
             if not pd.isna(prices.get(tic, np.nan)) and s != 0}
        return pd.Series(w)

    # ---- trading ----
    def rebalance_to(self, target_w: pd.Series, prices: pd.Series) -> dict:
        """Move current weights to target weights using today's close prices.

        Returns a small dict of trade stats (turnover, cost, etc.).
        """
        # 1. sanitize target
        tw = target_w.clip(lower=0).fillna(0)
        tw = tw[tw.index.isin(prices.dropna().index)]   # only tradable today
        s = tw.sum()
        if s > 1.0:
            tw = tw / s   # rescale rather than fail; never leverage
        eq = self.equity(prices)

        # 2. compute target shares
        target_shares = {}
        for tic, w in tw.items():
            p = prices[tic]
            if p > 0:
                target_shares[tic] = (w * eq) / p
        # any ticker held but no longer in target -> liquidate
        for tic in list(self.shares.keys()):
            if tic not in target_shares:
                target_shares[tic] = 0.0

        # 3. compute trades + cost
        notional_traded = 0.0
        for tic, ts in target_shares.items():
            cs = self.shares.get(tic, 0.0)
            delta = ts - cs
            if abs(delta) < 1e-10:
                continue
            p = prices.get(tic, np.nan)
            if pd.isna(p):
                continue
            trade_notional = abs(delta * p)
            notional_traded += trade_notional
            # cash side
            self.cash -= delta * p
            # cost
            cost = trade_notional * (self.cost_bps / 10_000.0)
            self.cash -= cost
            # update shares
            new_shares = cs + delta
            if abs(new_shares) < 1e-10:
                self.shares.pop(tic, None)
            else:
                self.shares[tic] = new_shares

        return {
            "turnover": notional_traded / eq if eq > 0 else 0.0,
            "notional_traded": notional_traded,
        }
