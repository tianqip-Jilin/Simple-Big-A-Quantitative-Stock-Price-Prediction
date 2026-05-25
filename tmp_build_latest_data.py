from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def main():
    base_df = pd.read_csv(DATA_DIR / "stock_data.csv")
    base_columns = list(base_df.columns)
    code_col = base_columns[0]
    date_col = base_columns[1]

    frames = [base_df]
    for path in sorted(DATA_DIR.glob("online_backfill_*.csv")):
        frames.append(pd.read_csv(path))

    latest = pd.concat(frames, ignore_index=True)
    latest[code_col] = latest[code_col].astype(str).str.zfill(6)
    latest[date_col] = pd.to_datetime(latest[date_col], errors="coerce", format="mixed")
    latest = (
        latest.sort_values([code_col, date_col])
        .drop_duplicates([code_col, date_col], keep="last")
        .reset_index(drop=True)
    )
    print(f"latest_date {latest[date_col].max().date()}")
    print(f"rows {len(latest)} stocks {latest[code_col].nunique()}")
    latest[date_col] = latest[date_col].dt.strftime("%Y-%m-%d")
    latest.to_csv(DATA_DIR / "latest_stock_data.csv", index=False)


if __name__ == "__main__":
    main()
