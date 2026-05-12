"""BacktestEngine: drives the day-by-day simulation loop.

Each trading day t the engine does, in order:
  1. Mark portfolio to today's close (record NAV).
  2. Ask the strategy for target weights given history up to t.
  3. Rebalance to those weights at today's close.

This ordering is internally consistent: at the end of the loop iteration for day t,
the portfolio reflects weights chosen by information available through t, which is
exactly the spec ("decisions based only on data up to today's closing price").

The convention "buy at today's close" means the strategy benefits from the next day's
return -- standard practice for daily-close backtests. We log NAV BEFORE rebalancing
so it reflects yesterday's positions valued at today's close.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from .portfolio import Portfolio
from .data_handler import DataHandler
from .strategy import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        cost_bps: float = 0.0,
        warmup_days: int = 60,
        start: str | None = None,
        end: str | None = None,
    ):
        self.data = data
        self.strategy = strategy
        self.cost_bps = cost_bps
        self.initial_cash = initial_cash
        self.warmup_days = warmup_days
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None

    def run(self) -> dict:
        port = Portfolio(self.initial_cash, cost_bps=self.cost_bps)
        dates = self.data.dates
        if self.start is not None:
            dates = dates[dates >= self.start]
        if self.end is not None:
            dates = dates[dates <= self.end]
        # warmup -- strategies need history to compute signals
        dates = dates[self.warmup_days:]

        nav_records = []
        weight_records = []
        trade_records = []
        ctx = {"portfolio": port}

        for t in dates:
            prices = self.data.close_on(t)
            # 1. mark to market BEFORE rebalancing
            nav = port.equity(prices)
            nav_records.append((t, nav))

            # 2. strategy decision (only uses history through t)
            try:
                tw = self.strategy.target_weights(t, self.data, ctx)
            except Exception as e:
                logger.warning(f"{self.strategy.name} failed on {t}: {e}; holding")
                tw = pd.Series(dtype=float)
            tw = tw.reindex(self.data.tickers).fillna(0)

            # 3. execute
            stats = port.rebalance_to(tw, prices)
            trade_records.append((t, stats["turnover"]))
            weight_records.append((t, tw))

        nav_series = pd.Series(
            [v for _, v in nav_records],
            index=pd.DatetimeIndex([d for d, _ in nav_records]),
            name=self.strategy.name,
        )
        weights_df = pd.DataFrame(
            {d: w for d, w in weight_records}
        ).T.fillna(0)
        weights_df.index = pd.DatetimeIndex(weights_df.index)
        turnover_series = pd.Series(
            [tv for _, tv in trade_records],
            index=pd.DatetimeIndex([d for d, _ in trade_records]),
            name="turnover",
        )

        return {
            "nav": nav_series,
            "weights": weights_df,
            "turnover": turnover_series,
            "strategy": self.strategy.name,
        }
