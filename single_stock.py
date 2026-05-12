"""Single-stock strategies (Deliverable 3).

Each strategy decides 0% or 100% allocation in ONE chosen ticker; the rest is cash.
This makes them comparable head-to-head on the same stock.

Strategies implemented
----------------------
1. Momentum         : long when trailing N-day return > 0
2. MeanReversion    : long when z-score of price vs MA(N) is < -threshold
3. SMACrossover     : long when SMA(short) > SMA(long)             [extension #1]
4. RSIContrarian    : long when 14-day RSI < 30                     [extension #2]
5. BollingerBreakout: long when close breaks above upper band       [extension #3]
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from backtest.strategy import Strategy


# ------------ helper ------------
def _series(data, t, ticker, field="close") -> pd.Series:
    return data.history_until(t, field)[ticker].dropna()


# ------------ 1. Momentum ------------
class MomentumSingle(Strategy):
    def __init__(self, ticker: str, lookback: int = 20):
        self.ticker = ticker
        self.lookback = lookback
        self.name = f"Momentum({ticker},{lookback}d)"

    def target_weights(self, t, data, ctx):
        s = _series(data, t, self.ticker)
        if len(s) <= self.lookback:
            return pd.Series(dtype=float)
        ret = s.iloc[-1] / s.iloc[-self.lookback - 1] - 1
        return pd.Series({self.ticker: 1.0 if ret > 0 else 0.0})


# ------------ 2. Mean Reversion ------------
class MeanReversionSingle(Strategy):
    def __init__(self, ticker: str, lookback: int = 20, z_threshold: float = 1.0):
        self.ticker = ticker
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.name = f"MeanReversion({ticker},{lookback}d,z>{z_threshold})"

    def target_weights(self, t, data, ctx):
        s = _series(data, t, self.ticker)
        if len(s) <= self.lookback:
            return pd.Series(dtype=float)
        window = s.iloc[-self.lookback:]
        z = (s.iloc[-1] - window.mean()) / window.std()
        # buy when oversold (z very negative); flat otherwise
        return pd.Series({self.ticker: 1.0 if z < -self.z_threshold else 0.0})


# ------------ 3. SMA Crossover (extension #1) ------------
class SMACrossoverSingle(Strategy):
    def __init__(self, ticker: str, short: int = 20, long: int = 50):
        self.ticker = ticker
        self.short = short
        self.long = long
        self.name = f"SMACross({ticker},{short}/{long})"

    def target_weights(self, t, data, ctx):
        s = _series(data, t, self.ticker)
        if len(s) <= self.long:
            return pd.Series(dtype=float)
        sma_s = s.iloc[-self.short:].mean()
        sma_l = s.iloc[-self.long:].mean()
        return pd.Series({self.ticker: 1.0 if sma_s > sma_l else 0.0})


# ------------ 4. RSI Contrarian (extension #2) ------------
def _rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) <= period:
        return np.nan
    delta = prices.diff().dropna().iloc[-period:]
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - 100 / (1 + rs)


class RSIContrarianSingle(Strategy):
    def __init__(self, ticker: str, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.ticker = ticker
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI({ticker},{period}d,buy<{oversold})"

    def target_weights(self, t, data, ctx):
        s = _series(data, t, self.ticker)
        rsi = _rsi(s, self.period)
        if np.isnan(rsi):
            return pd.Series(dtype=float)
        # We use a simple "long when oversold, exit when overbought, otherwise hold what we had"
        port = ctx["portfolio"]
        currently_held = port.shares.get(self.ticker, 0) > 0
        if rsi < self.oversold:
            return pd.Series({self.ticker: 1.0})
        elif rsi > self.overbought:
            return pd.Series({self.ticker: 0.0})
        else:
            return pd.Series({self.ticker: 1.0 if currently_held else 0.0})


# ------------ 5. Bollinger Breakout (extension #3) ------------
class BollingerBreakoutSingle(Strategy):
    def __init__(self, ticker: str, lookback: int = 20, k: float = 2.0):
        self.ticker = ticker
        self.lookback = lookback
        self.k = k
        self.name = f"BollingerBreakout({ticker},{lookback}d,k={k})"

    def target_weights(self, t, data, ctx):
        s = _series(data, t, self.ticker)
        if len(s) <= self.lookback:
            return pd.Series(dtype=float)
        window = s.iloc[-self.lookback:]
        upper = window.mean() + self.k * window.std()
        lower = window.mean() - self.k * window.std()
        port = ctx["portfolio"]
        currently_held = port.shares.get(self.ticker, 0) > 0
        last = s.iloc[-1]
        if last > upper:
            return pd.Series({self.ticker: 1.0})        # breakout up -> long
        elif last < lower:
            return pd.Series({self.ticker: 0.0})        # break down -> exit
        else:
            return pd.Series({self.ticker: 1.0 if currently_held else 0.0})
