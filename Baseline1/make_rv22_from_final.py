# make_rv22_from_final.py
"""
Build RV_22 from a 'Final.csv'-style file (their dataset).
You can pass either --lr_col (preferred, e.g. LR_Cop) or --price_col (e.g. Close).
We’ll write <outdir>/<tag>_RV22.csv with columns: Date, Close?, LR, RV_22
Usage examples:
  python make_rv22_from_final.py --csv Final.csv --date_col Date --lr_col LR_Cop --tag COPPER --outdir data
  python make_rv22_from_final.py --csv Final.csv --date_col Date --price_col Close --tag SPX --outdir data
"""

import argparse, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--lr_col", default=None, help="If present in Final.csv (e.g. LR_Cop)")
    ap.add_argument("--price_col", default=None, help="Fallback if LR not given (e.g. Close)")
    ap.add_argument("--tag", required=True, help="Name prefix for outputs (e.g. COPPER)")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=[args.date_col]).sort_values(args.date_col)

    if args.lr_col and args.lr_col in df:
        lr = df[args.lr_col].astype(float).values
        close = df[args.price_col].astype(float).values if args.price_col and args.price_col in df else None
        dates = df[args.date_col].values
    elif args.price_col and args.price_col in df:
        price = df[args.price_col].astype(float).values
        dates = df[args.date_col].values
        lr = np.diff(np.log(price))
        # align dates & price down by 1 after diff
        dates = dates[1:]
        price = price[1:]
        close = price
    else:
        raise ValueError("Pass --lr_col (existing in CSV) or --price_col.")

    # RV_22 = rolling variance around rolling mean over T=22
    s = pd.Series(lr)
    rbar = s.rolling(22).mean()
    rv22 = ((s - rbar)**2).rolling(22).sum() / 22
    out = pd.DataFrame({"Date": dates, "LR": lr, "RV_22": rv22})
    if close is not None:
        out.insert(1, "Close", close)

    out = out.dropna()
    path = f"{args.outdir}/{args.tag}_RV22.csv"
    out.to_csv(path, index=False)
    print("✔ Wrote", path, "rows:", len(out))

if __name__ == "__main__":
    main()
