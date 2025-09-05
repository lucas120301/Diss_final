# bloomberg_to_rv22.py
"""
Extract [Date, Close] from Bloomberg-style wide Excel and build RV_22 CSVs.

Usage:
  python bloomberg_to_rv22.py --xlsx "terminal data 1.xlsx" --pairs "0,1" "3,4" "6,7" --outdir data
Options:
  --sheet SHEETNAME   (optional; default=first sheet)
"""

import argparse, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def excel_serial_to_datetime_series(series: pd.Series) -> pd.Series:
    origin = datetime(1899, 12, 30)
    def conv(x):
        try:
            return origin + timedelta(days=float(x))
        except Exception:
            return pd.NaT
    return series.apply(conv)

def _find_first_data_row(raw: pd.DataFrame, date_col: int, px_col: int, max_scan: int = 10) -> int:
    for r in range(min(max_scan, len(raw))):
        d = raw.iat[r, date_col]
        p = raw.iat[r, px_col]
        try:
            _ = float(d)  # excel serial
            float(p)      # numeric price
            return r
        except Exception:
            continue
    return 3  # fallback (header rows at 0–2)

def extract_pair(xlsx_path: str, date_col: int, px_col: int, out_prefix: str, sheet_name=None):
    print(f"\n=== Pair (Date col {date_col}, Price col {px_col}) ===")
    raw = pd.read_excel(xlsx_path, header=None, sheet_name=(0 if sheet_name is None else sheet_name))
    # If user passed None explicitly and pandas still returns a dict (older versions), handle it:
    if isinstance(raw, dict):
        # take the first sheet
        raw = next(iter(raw.values()))

    print(f"Loaded sheet with shape {raw.shape}")

    start_row = _find_first_data_row(raw, date_col, px_col)
    print(f"First data row detected at index: {start_row}")

    df = raw.iloc[start_row:, [date_col, px_col]].copy()
    df.columns = ["Date", "Close"]

    # Convert + clean
    df["Date"] = excel_serial_to_datetime_series(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    before = len(df)
    df = df.dropna().sort_values("Date")
    df = df[df["Close"] > 0]
    after = len(df)
    print(f"Rows before clean: {before}, after clean: {after}")

    if after == 0:
        print("No valid rows found for this pair; skipping.")
        return None, None

    base = f"{out_prefix}.csv"
    df.to_csv(base, index=False)
    print(f"✔ Wrote {base}")

    # Realized volatility (T=22)
    out = df.copy()
    out["LR"] = np.log(out["Close"]).diff()
    rbar = out["LR"].rolling(22).mean()
    out["RV_22"] = ((out["LR"] - rbar)**2).rolling(22).sum() / 22
    out = out.dropna()

    rv_path = f"{out_prefix}_RV22.csv"
    out.to_csv(rv_path, index=False)
    print(f"✔ Wrote {rv_path} (rows: {len(out)})")

    return base, rv_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to Bloomberg-style Excel")
    ap.add_argument("--pairs", nargs="+", required=True, help='Pairs like "0,1" (DateCol,PriceCol)')
    ap.add_argument("--outdir", default="data", help="Output directory (created if missing)")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet)")
    args = ap.parse_args()

    # 1) Validate Excel
    if not os.path.exists(args.xlsx):
        print(f"ERROR: Excel file not found: {args.xlsx}")
        print("Tip: run `pwd` then `ls -l` to confirm the filename and path. If the path has spaces, keep the quotes.")
        return

    # 2) Ensure outdir
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.outdir)}")

    # 3) Process each pair
    summary = []
    for pair in args.pairs:
        try:
            d, p = map(int, pair.split(","))
        except Exception:
            print(f"Skipping invalid pair string: {pair} (expected 'i,j')")
            continue

        out_prefix = os.path.join(args.outdir, f"extracted_{d}_{p}")
        try:
            base, rv = extract_pair(args.xlsx, d, p, out_prefix, sheet_name=args.sheet)
            summary.append({"pair": pair, "base_csv": base, "rv_csv": rv})
        except Exception as e:
            print(f"❌ Error for pair {pair}: {e}")
            summary.append({"pair": pair, "base_csv": None, "rv_csv": None})

    # 4) Summary
    print("\n=== Summary ===")
    for row in summary:
        print(f"{row['pair']}: base={row['base_csv']}, rv={row['rv_csv']}")

if __name__ == "__main__":
    main()
