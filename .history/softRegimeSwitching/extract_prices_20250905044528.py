# extract_prices.py
# Usage:
#   pip install pandas openpyxl
#   python3 extract_prices.py "terminal data 1.xlsx"          # -> terminal data 1_cleaned.csv
#   python3 extract_prices.py "terminal data 1.xlsx" Sheet1   # optional sheet
#   python3 extract_prices.py "terminal data 1.xlsx" Sheet1 out.csv

import sys
from pathlib import Path
import pandas as pd


# ----------------- Helpers -----------------

def norm(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def is_date_label(s) -> bool:
    return norm(s) == "date"


def is_last_px_label(s) -> bool:
    s = norm(s)
    return (
        ("last" in s and "px" in s)
        or s in {"px_last", "last_px", "lastpx", "last price", "px last", "px last", "lastpx"}
    )


def find_label_row(df: pd.DataFrame, max_scan_rows: int = 12) -> int | None:
    best_row, best_score = None, -1
    for r in range(min(max_scan_rows, df.shape[0])):
        row = df.iloc[r]
        score = sum(is_date_label(v) or is_last_px_label(v) for v in row)
        if score > best_score:
            best_score, best_row = score, r
    return best_row if best_row is not None and best_score >= 2 else None


def nearest_name_above(df: pd.DataFrame, label_row: int, col: int) -> str:
    for r in range(label_row - 1, -1, -1):
        val = df.iat[r, col] if col < df.shape[1] else None
        s = str(val).strip() if pd.notna(val) else ""
        if s:
            return s
    # fallback: try left cells on the row above
    if label_row - 1 >= 0:
        for c in range(col - 1, -1, -1):
            val = df.iat[label_row - 1, c]
            s = str(val).strip() if pd.notna(val) else ""
            if s:
                return s
    return f"Security_{col}"


def find_date_col_left(df: pd.DataFrame, label_row: int, last_px_col: int) -> int | None:
    # same row
    for c in range(last_px_col - 1, -1, -1):
        if is_date_label(df.iat[label_row, c]):
            return c
    # slight misalignment tolerance: check row above/below
    for rr in (label_row - 1, label_row + 1):
        if 0 <= rr < df.shape[0]:
            for c in range(last_px_col - 1, -1, -1):
                if is_date_label(df.iat[rr, c]):
                    return c
    return None


def to_datetime_series_mixed(series: pd.Series) -> pd.Series:
    """
    Convert a column that may contain:
    - Excel serial numbers (ints/floats or numeric strings), and/or
    - textual dates (e.g., '2020-01-01', '01/02/2020')
    into a single datetime64[ns] series.
    """
    s = series.copy()

    # 1) Numeric path (handles 38720 or "38720")
    nums = pd.to_numeric(s, errors="coerce")
    dt_from_nums = pd.to_datetime(nums, errors="coerce", unit="d", origin="1899-12-30")

    # 2) Text path (only where numeric failed but value exists)
    text_mask = nums.isna() & s.notna()
    # Normalize text a bit
    s_text = s[text_mask].astype(str).str.strip().replace({"": pd.NA})
    dt_from_text = pd.to_datetime(s_text, errors="coerce", infer_datetime_format=True)

    # 3) Merge
    out = dt_from_nums.copy()
    out.loc[text_mask] = dt_from_text
    return out  # may contain NaT for bad rows


# ----------------- Main extraction -----------------

def extract_prices_to_csv(xlsx_path: str, sheet_name=0, out_path: str | None = None) -> str:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    if df.empty:
        raise ValueError("The sheet is empty.")

    label_row = find_label_row(df, max_scan_rows=12)
    if label_row is None:
        raise ValueError("Could not locate a header row containing 'Date' / 'Last PX' labels in the first ~12 rows.")

    data_start = label_row + 1

    # Find all Last PX columns on the detected label row
    last_px_cols = [c for c in range(df.shape[1]) if is_last_px_label(df.iat[label_row, c])]
    if not last_px_cols:
        raise ValueError(f'No "Last PX" columns detected on label row {label_row}.')

    records = []
    for c in last_px_cols:
        date_col = find_date_col_left(df, label_row, c)
        if date_col is None:
            continue  # skip if we cannot pair a date column

        name = nearest_name_above(df, label_row, c)

        date_series = to_datetime_series_mixed(df.iloc[data_start:, date_col])
        price_series = pd.to_numeric(df.iloc[data_start:, c], errors="coerce")

        mask = date_series.notna() & price_series.notna()
        if mask.any():
            part = pd.DataFrame(
                {
                    "Date": date_series[mask].dt.date,  # keep date (no time)
                    "Name": name,
                    "Price": price_series[mask].astype(float),
                }
            )
            records.append(part)

    if not records:
        raise ValueError("Found Last PX columns, but no rows with valid Date+Price pairs.")

    tidy = pd.concat(records, ignore_index=True)
    tidy.sort_values(["Name", "Date"], inplace=True)

    if out_path is None:
        out_path = Path(xlsx_path).with_suffix("").as_posix() + "_cleaned.csv"

    tidy.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_prices.py <path_to_excel> [sheet_name_or_index] [out_csv]")
        sys.exit(1)

    xlsx = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv) >= 3 else 0
    try:
        sheet = int(sheet)
    except Exception:
        pass

    out = sys.argv[3] if len(sys.argv) >= 4 else None
    extract_prices_to_csv(xlsx, sheet, out)
