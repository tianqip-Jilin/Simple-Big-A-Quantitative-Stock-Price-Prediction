import time
from pathlib import Path

import baostock as bs
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
END_DATE = "2026-05-25"


def fetch_hist(bs_code, start_date, end_date, base_columns):
    last_error = None
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="1",
            )
            if rs.error_code != "0":
                raise RuntimeError(f"{rs.error_code} {rs.error_msg}")

            rows = []
            while (rs.error_code == "0") and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=rs.fields)
            numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            out = pd.DataFrame()
            out[base_columns[0]] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)
            out[base_columns[1]] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            out[base_columns[2]] = df["open"]
            out[base_columns[3]] = df["close"]
            out[base_columns[4]] = df["high"]
            out[base_columns[5]] = df["low"]
            out[base_columns[6]] = df["volume"]
            out[base_columns[7]] = df["amount"]
            out[base_columns[8]] = ((df["high"] - df["low"]) / (df["preclose"] + 1e-12) * 100).round(2)
            out[base_columns[9]] = (df["close"] - df["preclose"]).round(2)
            out[base_columns[10]] = df["turn"]
            out[base_columns[11]] = df["pctChg"]
            return out
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.6 * (attempt + 1))
    return ("ERR", bs_code, last_error or "unknown error")


def main():
    base_df = pd.read_csv(DATA_DIR / "stock_data.csv")
    base_columns = list(base_df.columns)
    date_col = base_columns[1]

    existing_parts = [base_df]
    for path in sorted(DATA_DIR.glob("online_backfill_*.csv")):
        existing_parts.append(pd.read_csv(path))

    existing = pd.concat(existing_parts, ignore_index=True)
    existing[base_columns[0]] = existing[base_columns[0]].astype(str).str.zfill(6)
    existing[date_col] = pd.to_datetime(existing[date_col], errors="coerce", format="mixed")
    last_date = existing[date_col].max()
    start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    target_path = DATA_DIR / f"online_backfill_{END_DATE}.csv"
    if last_date >= pd.Timestamp(END_DATE) and target_path.exists():
        latest_full = (
            existing.sort_values([base_columns[0], date_col])
            .drop_duplicates([base_columns[0], date_col], keep="last")
            .reset_index(drop=True)
        )
        latest_full[date_col] = latest_full[date_col].dt.strftime("%Y-%m-%d")
        latest_full_path = DATA_DIR / "latest_stock_data.csv"
        latest_full.to_csv(latest_full_path, index=False)
        print(f"already_up_to_date {last_date.date()} {target_path}")
        print(f"latest_full {latest_full_path} rows {len(latest_full)}")
        return

    hs300 = pd.read_csv(DATA_DIR / "hs300_stock_list.csv")
    codes = sorted(set(hs300["code"].astype(str)))

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")

    try:
        results = []
        failures = []
        for idx, code in enumerate(codes, start=1):
            result = fetch_hist(code, start_date, END_DATE, base_columns)
            if isinstance(result, tuple) and result and result[0] == "ERR":
                failures.append(result[1:])
            elif result is not None and not result.empty:
                results.append(result)
            if idx % 25 == 0:
                print(f"fetch_progress {idx}/{len(codes)} success={len(results)} fail={len(failures)}", flush=True)
            time.sleep(0.12)
    finally:
        bs.logout()

    new_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=base_columns)
    merged = pd.concat(existing_parts[1:] + [new_df], ignore_index=True) if existing_parts[1:] else new_df
    if not merged.empty:
        merged[base_columns[0]] = merged[base_columns[0]].astype(str).str.zfill(6)
        merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce", format="mixed")
        merged = (
            merged.sort_values([base_columns[0], date_col])
            .drop_duplicates([base_columns[0], date_col], keep="last")
            .reset_index(drop=True)
        )
        merged[date_col] = merged[date_col].dt.strftime("%Y-%m-%d")
    merged.to_csv(target_path, index=False)
    latest_full = pd.concat([base_df, merged], ignore_index=True)
    latest_full[base_columns[0]] = latest_full[base_columns[0]].astype(str).str.zfill(6)
    latest_full[date_col] = pd.to_datetime(latest_full[date_col], errors="coerce", format="mixed")
    latest_full = (
        latest_full.sort_values([base_columns[0], date_col])
        .drop_duplicates([base_columns[0], date_col], keep="last")
        .reset_index(drop=True)
    )
    latest_full[date_col] = latest_full[date_col].dt.strftime("%Y-%m-%d")
    latest_full_path = DATA_DIR / "latest_stock_data.csv"
    latest_full.to_csv(latest_full_path, index=False)
    print(f"last_existing {last_date.date()} start {start_date} end {END_DATE}")
    print(f"new_rows {len(new_df)} merged_rows {len(merged)} failed {len(failures)}")
    print(f"written {target_path}")
    print(f"latest_full {latest_full_path} rows {len(latest_full)}")
    if failures:
        print(f"fail_samples {failures[:5]}")


if __name__ == "__main__":
    main()
