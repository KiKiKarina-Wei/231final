"""Data handler: loads the 5y Nasdaq-100 panel, builds aligned price matrices.

Design notes
------------
- All time-series are stored as wide DataFrames indexed by date with one column per ticker.
- The handler exposes ONLY information up to and including a given date `t`.
  This is the only place where look-ahead is policed; downstream code (strategies)
  must request data via `history_until(t)` rather than touch the raw frame.
- A stock that has not yet IPO'd will have NaN before its first trading day.
  The "tradable universe" on day t is the set of tickers with non-NaN close on t.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


class DataHandler:
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self._raw = df
        self.close = df.pivot(index="date", columns="ticker", values="close").sort_index()
        self.open  = df.pivot(index="date", columns="ticker", values="open").sort_index()
        self.high  = df.pivot(index="date", columns="ticker", values="high").sort_index()
        self.low   = df.pivot(index="date", columns="ticker", values="low").sort_index()
        self.volume = df.pivot(index="date", columns="ticker", values="volume").sort_index()

        self.dates = self.close.index
        self.tickers = list(self.close.columns)

    # -------- accessors used by the engine / strategies --------
    def history_until(self, t: pd.Timestamp, field: str = "close") -> pd.DataFrame:
        """Return the panel of `field` from start through and including day t (no future)."""
        frame = getattr(self, field)
        return frame.loc[:t]

    def tradable_on(self, t: pd.Timestamp) -> list[str]:
        """Tickers that have a real (non-NaN) close on day t."""
        row = self.close.loc[t]
        return row.dropna().index.tolist()

    def close_on(self, t: pd.Timestamp) -> pd.Series:
        return self.close.loc[t]

    def __repr__(self):
        return (f"DataHandler(days={len(self.dates)}, tickers={len(self.tickers)}, "
                f"range={self.dates.min().date()}..{self.dates.max().date()})")
