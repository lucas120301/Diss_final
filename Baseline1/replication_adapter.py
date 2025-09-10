"""
Preprocessing
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def realized_volatility(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    RV_t = (1/T) * sum_{j=t..t+T-1} (R_j - Rbar)^2, with T=22 by default.
    """
    rbar = returns.rolling(window).mean()
    rv = ((returns - rbar) ** 2).rolling(window).sum() / window
    return rv

def load_series_like_paper(
    csv_path: str,
    date_col: str = "Date",
    price_col: str = "Close",
    start: str | None = None,
    end: str | None = None,
    rv_window: int = 22,
):
    """
    Loads a [Date, Close] CSV, computes LR and RV_22, and returns standardized RV.
    Returns dict: df, returns, rv_22, scaler (fit on the whole series here; for modeling
    you should fit MinMax on train split only — the training scripts below handle that).
    """
    df = pd.read_csv(csv_path)
    if date_col not in df or price_col not in df:
        raise ValueError(f"Expected columns {date_col}, {price_col}. Got {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    price = df[price_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    price = price[price > 0]
    if start: price = price.loc[pd.to_datetime(start):]
    if end:   price = price.loc[:pd.to_datetime(end)]

    lr = np.log(price).diff().rename("LR")
    rv = realized_volatility(lr, rv_window).rename(f"RV_{rv_window}")

    out = pd.concat([price.rename("Price"), lr, rv], axis=1).dropna()
    return {
        "df": out,
        "returns": out["LR"].copy(),
        f"rv_{rv_window}": out[f"RV_{rv_window}"].copy(),
    }

def make_windows(series_1d: np.ndarray, window: int):
    """
    Converts a 1D array to (X, y) for LSTM: X.shape=(n, window, 1), y.shape=(n, 1)
    """
    X, y = [], []
    for i in range(len(series_1d) - window):
        X.append(series_1d[i:i+window])
        y.append(series_1d[i+window])
    X = np.asarray(X).reshape(-1, window, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y
