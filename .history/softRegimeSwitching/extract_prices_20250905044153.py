# save as extract_prices.py
# Usage:
#   pip install pandas openpyxl
#   python extract_prices.py "terminal data 1.xlsx"  # outputs <input>_cleaned.csv
#
# What it does:
# - Reads the first sheet
# - Uses row 2 (0-based index 1) to find "Date" and "Last PX" columns
# - Pairs each "Last PX" with the nearest "Date" column to its left
# - Uses row 1 (index 0) to name each security
# - Converts Excel serial dates to real dates
# - Writes a tidy CSV: Date, Name, Price

import sys
from pathlib import Path
import pandas as pd


def _looks_like(label: object, target: str) -> bool:
    if not isinstance(label, str):
        return False
    s = label.strip().lower()
    if target == "date":
        return s == "date"
    if target == "last_px":
        # be tolerant of variations like "last px", "px_last"
        return ("last" in s and "px" in s) or s in {"px_last", "last_px"}
    return False


def _to_datetime(series: pd.Series) -> pd.Series:
    # Try Excel serial first, then general parsing
    dt = pd.to_datetime(series, errors="coerce", unit="d", origin="1899-12-30")
    if dt.notna().sum() == 0:  # fallback if not serials
        dt = pd.to_datetime(series, errors="coerce")
    return dt


def extract_prices_to_csv(xlsx_path: str, sheet_name=0, out_path: str | None = None) -> str:
    # Read with NO header: we will use row 0 and row 1 as metadata
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")

    if df.shape[0] < 3:
        raise ValueError("Sheet is too short to contain headers + data (need at least 3 rows).")

    header_names_row = 0      # holds security names ("SPX Index", "SKEW", etc.)
    header_labels_row = 1     # holds "Date" / "Last PX"
    data_start_row = header_labels_row + 1

    labels_row = df.iloc[header_labels_row]
    names_row = df.iloc[header_names_row]

    # Identify all "Last PX" columns
    last_px_cols = [c for c in df.columns if _looks_like(labels_row[c], "last_px")]
    if not last_px_cols:
        raise ValueError('Could not find any "Last PX" columns in row 2 (index 1).')

    records = []

    for c in last_px_cols:
        # Find nearest "Date" column to the LEFT of this Last PX column
        date_col = None
        for k in range(c - 1, -1, -1):
            if _looks_like(labels_row[k], "date"):
                date_col = k
                break
        if date_col is None:
            # No date column found to the left; skip this block
            continue

        # Security name from the first row (index 0) of the Last PX column
        raw_name = names_row[c]
        name = str(raw_name).strip() if pd.notna(raw_name) else f"Security_{c}"

        # Extract and convert series
        date_series = _to_datetime(df.iloc[data_start_row:, date_col])
        price_series = pd.to_numeric(df.iloc[data_start_row:, c], errors="coerce")

        mask = date_series.notna() & price_series.notna()
        if mask.any():
            sub = pd.DataFrame(
                {"Date": date_series[mask].dt.date, "Name": name, "Price": price_series[mask].astype(float)}
            )
            records.append(sub)

    if not records:
        raise ValueError("No (Date, Price) pairs found. Check that row 2 contains 'Date'/'Last PX' labels.")

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
    # Try to convert sheet to int if it's a number
    try:
        sheet = int(sheet)
    except Exception:
        pass

    out = sys.argv[3] if len(sys.argv) >= 4 else None
    extract_prices_to_csv(xlsx, sheet, out)
