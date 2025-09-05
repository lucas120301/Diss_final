# generate_garch_forecasts.py
"""
Generate 252-day rolling one-step-ahead variance forecasts:
  - sGARCH(1,1), eGARCH(1,1), tGARCH(1,1)
Input: a CSV with Date & Close (or a CSV with LR). We’ll compute LR if needed.
Requires: pip install arch
Usage:
  python generate_garch_forecasts.py --csv data/extracted_0_1.csv --date_col Date --price_col Close --symbol SPX
Or (if you already have LR in the file):
  python generate_garch_forecasts.py --csv data/extracted_0_1_RV22.csv --date_col Date --lr_col LR --symbol SPX
"""

import argparse
import numpy as np
import pandas as pd
from arch import arch_model

def load_returns(csv, date_col="Date", price_col=None, lr_col=None):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv, parse_dates=[date_col]).sort_values(date_col)

    if lr_col and lr_col in df:
        dates = df[date_col].values
        r = pd.to_numeric(df[lr_col], errors="coerce").values
        # DROP NaN/Inf returns and align dates
        mask = np.isfinite(r)
        dates, r = dates[mask], r[mask]

    elif price_col and price_col in df:
        dates = df[date_col].values
        price = pd.to_numeric(df[price_col], errors="coerce").values
        mask = np.isfinite(price)
        dates, price = dates[mask], price[mask]
        # Log-returns; first value becomes NaN -> drop via slicing
        r = np.diff(np.log(price))
        dates = dates[1:]

        # Just in case: remove any residual non-finite values
        mask = np.isfinite(r)
        dates, r = dates[mask], r[mask]

    else:
        raise ValueError("Provide either --lr_col (preferred) or --price_col")

    return dates, r


def rolling_forecast(returns, model, window=252):
    """
    returns: 1D numpy array (original units)
    model: 'sGARCH' | 'eGARCH' | 'tGARCH'
    window: rolling refit window
    Returns variance forecasts back in the ORIGINAL units
    """
    out = []
    n = len(returns)
    scale = 100.0  # rescale returns to improve optimizer numerics

    for t in range(window, n):
        r_win = returns[t-window:t]
        r_win_s = r_win * scale  # scaled

        if model == "sGARCH":
            am = arch_model(r_win_s, vol="GARCH", p=1, q=1, mean="Constant", dist="normal", rescale=False)
        elif model == "eGARCH":
            am = arch_model(r_win_s, vol="EGARCH", p=1, q=1, mean="Constant", dist="normal", rescale=False)
        elif model == "tGARCH":
            am = arch_model(r_win_s, vol="GARCH", p=1, o=1, q=1, power=1.0, mean="Constant", dist="normal", rescale=False)
        else:
            raise ValueError("Unknown model")

        res = am.fit(disp="off")
        f = res.forecast(horizon=1)
        v_scaled = f.variance.values[-1, 0]     # variance in scaled units
        v = v_scaled / (scale ** 2)             # back to ORIGINAL units
        out.append(v)

    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (Date & Close, or Date & LR)")
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--price_col", default=None)
    ap.add_argument("--lr_col", default=None)
    ap.add_argument("--symbol", required=True, help="Symbol tag for filenames (e.g., SPX)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--window", type=int, default=252)
    args = ap.parse_args()

    dates, r = load_returns(args.csv, date_col=args.date_col, price_col=args.price_col, lr_col=args.lr_col)
    L = len(r)
    if L <= args.window:
        raise ValueError(f"Not enough data: {L} <= window {args.window}")

    print(f"Returns length: {L} | Rolling window: {args.window} | Forecasts: {L-args.window}")

    sg = rolling_forecast(r, "sGARCH", window=args.window)
    eg = rolling_forecast(r, "eGARCH", window=args.window)
    tg = rolling_forecast(r, "tGARCH", window=args.window)

    # align forecast dates (start at dates[window:])
    out_dates = dates[args.window:]
    pd.DataFrame({"Date": out_dates, "x": sg}).to_csv(f"{args.outdir}/sGarch_{args.symbol}_252roll.csv", index=False)
    pd.DataFrame({"Date": out_dates, "x": eg}).to_csv(f"{args.outdir}/eGarch_{args.symbol}_252roll.csv", index=False)
    pd.DataFrame({"Date": out_dates, "x": tg}).to_csv(f"{args.outdir}/tGarch_{args.symbol}_252roll.csv", index=False)
    print("Saved:",
          f"{args.outdir}/sGarch_{args.symbol}_252roll.csv,",
          f"{args.outdir}/eGarch_{args.symbol}_252roll.csv,",
          f"{args.outdir}/tGarch_{args.symbol}_252roll.csv")

if __name__ == "__main__":
    main()
