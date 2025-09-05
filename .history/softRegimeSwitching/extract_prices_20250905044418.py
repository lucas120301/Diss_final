# save as extract_prices.py
# Usage:
#   pip install pandas openpyxl
#   python extract_prices.py "terminal data 1.xlsx"
#   # optionally specify sheet and output:
#   python extract_prices.py "terminal data 1.xlsx" "Sheet1" "out.csv"

import sys
from pathlib import Path
import pandas as pd


# ----------------- Helpers -----------------

def norm(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def is_date_label(s) -> bool:
    s = norm(s)
    return s == "date"


def is_last_px_label(s) -> bool:
    s = norm(s)
    # tolerate common variants
    return (
        ("last" in s and "px" in s) or
        s in {"px_last", "last_px", "lastpx", "last price", "px last"}
    )


def find_label_row(df: pd.DataFrame, max_scan_rows: int = 10) -> int | None:
    """
    Look for the row that contains many 'date' and 'last px' tokens.
    Return the row index, or None if not found.
    """
    best_row = None
    best_score = -1
    rows_to_scan = min(max_scan_rows, df.shape[0])
    for r in range(rows_to_scan):
        row = df.iloc[r]
        score = 0
        for val in row:
            if is_date_label(val) or is_last_px_label(val):
                score += 1
        if score > best_score:
            best_score = score
            best_row = r

    # Require at least 2 matches to be confident we found the header row
    if best_row is not None and best_score >= 2:
        return best_row
    return None


def nearest_name_above(df: pd.DataFrame, label_row: int, col: int) -> str:
    """
    Walk upward from label_row-1 to 0 to find a non-empty string for the security name.
    If nothing found, try left neighbor names (same row) as a last resort.
    """
    for r in range(label_row - 1, -1, -1):
        val = df.iat[r, col] if col < df.shape[1] else None
        s = str(val).strip() if pd.notna(val) else ""
        if s:
            return s
    # fallback: try to the left on the same row
    for c in range(col - 1, -1, -1):
        val = df.iat[label_row - 1, c] if label_row - 1 >= 0 else None
        s = str(val).strip() if pd.notna(val) else ""
        if s:
            return s
    return f"Security_{col}"


def find_date_col_left(df: pd.DataFrame, label_row: int, last_px_col: int) -> int | None:
    """
    Find the nearest Date column to the LEFT of last_px_col on the same label row.
    If not found on the same row, look one row up/down as a tolerance for misaligned labels.
    """
    # same row
    for c in range(last_px_col - 1, -1, -1):
        if is_date_label(df.iat[label_row, c]):
            return c
    # tolerate slight misalignment
    for delta in (-1, 1):
        rr = label_row + delta
        if 0 <= rr < df.shape[0]:
            for c in range(last_px_col - 1, -1, -1):
                if is_date_label(df.iat[rr, c]):
                    return c
    return None


def to_datetime_series(series: pd.Series) -> pd.Series:
    # Try Excel serial dates first
    dt = pd.to_datetime(series, errors="coerce", unit="d", origin="1899-12-30")
    if dt.notna().sum() == 0:  # fallback if not serials
        dt = pd.to_datetime(series, errors="coerce")
    return dt


# ----------------- Main extraction -----------------

def extract_prices_to_csv(xlsx_path: str, sheet_name=0, out_path: str | None = None) -> str:
    # Read raw with no header; we’ll detect the label row
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")

    if df.empty:
        raise ValueError("The sheet is empty.")

    label_row = find_label_row(df, max_scan_rows=12)
    if label_row is None:
        raise ValueError("Could not locate a header row containing 'Date' / 'Last PX' labels in the first ~12 rows.")

    data_start = label_row + 1

    # Identify all Last PX columns on the detected label row
    last_px_cols = [c for c in range(df.shape[1]) if is_last_px_label(df.iat[label_row, c])]
    if not last_px_cols:
        raise ValueError(f'Could not find any "Last PX" columns on the detected label row {label_row}.')

    records = []
    for c in last_px_cols:
        date_col = find_date_col_left(df, label_row, c)
        if date_col is None:
            # Skip if we can't pair with a date
            continue

        name = nearest_name_above(df, label_row, c)

        # Extract series
        date_series = to_datetime_series(df.iloc[data_start:, date_col])
        price_series = pd.to_numeric(df.iloc[data_start:, c], errors="coerce")

        mask = date_series.notna() & price_series.notna()
        if mask.any():
            part = pd.DataFrame({
                "Date": date_series[mask].dt.date,  # date only
                "Name": name,
                "Price": price_series[mask].astype(float)
            })
            records.append(part)

    if not records:
        raise ValueError("Found 'Last PX' labels, but no matching (Date, Price) rows with data.")

    tidy = pd.concat(records, ignore_index=True)
    tidy.sort_values(["Name", "Date"], inplace=True)

    if out_path is None:
        out_path = Path(xlsx_path).with_suffix("").as_posix() + "_cleaned.csv"

    tidy.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_prices.py <path_to_excel> [sheet_name_or_index] [out_csv]")
        sys.exit(1)

    xlsx = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv) >= 3 else 0
    try:
        sheet = int(sheet)
    except Exception:
        pass  # treat as name if not int

    out = sys.argv[3] if len(sys.argv) >= 4 else None
    extract_prices_to_csv(xlsx, sheet, out)
