"""Strategy abstract base class.

Every strategy implements one method: `target_weights(t, data, context)`.
The contract is:

    inputs
    ------
    t        : pandas Timestamp  -- "today" (the close on which trades execute)
    data     : DataHandler       -- only history_until(t) should be used
    context  : dict              -- optional state (e.g. portfolio cash/positions)

    output
    ------
    pd.Series indexed by ticker, values in [0, 1], sum <= 1.
    Tickers omitted from the Series are treated as weight 0.

This single contract drives BOTH single-stock and portfolio backtests:
- a single-stock momentum strategy returns {"AAPL": 1.0} or {"AAPL": 0.0}
- a portfolio strategy returns a long vector across the universe
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    name: str = "BaseStrategy"

    @abstractmethod
    def target_weights(self, t: pd.Timestamp, data, context: dict) -> pd.Series:
        ...

    def __repr__(self):
        return f"<Strategy {self.name}>"
