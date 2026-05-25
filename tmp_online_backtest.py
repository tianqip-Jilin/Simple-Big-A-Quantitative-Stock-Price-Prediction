import pickle
import sys
import time
from pathlib import Path

import baostock as bs
import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"C:\Users\22980\Desktop\机器学习股票预测\THU-BDC2026-main")
SRC_DIR = ROOT / "code" / "src"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model" / "60_158+39"

sys.path.insert(0, str(SRC_DIR))

from config import config
from model import StockTransformer
from predict import build_inference_sequences, preprocess_predict_data


def fetch_hist(code: str, start_date: str, end_date: str):
    last_error = None
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                code,
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
            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["振幅"] = ((df["high"] - df["low"]) / (df["preclose"] + 1e-12) * 100).round(2)
            df["涨跌额"] = (df["close"] - df["preclose"]).round(2)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["code"] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)
            df = df.rename(
                columns={
                    "code": "股票代码",
                    "date": "日期",
                    "open": "开盘",
                    "close": "收盘",
                    "high": "最高",
                    "low": "最低",
                    "volume": "成交量",
                    "amount": "成交额",
                    "turn": "换手率",
                    "pctChg": "涨跌幅",
                }
            )
            df = df[
                ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
            ].copy()
            return df
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.6 * (attempt + 1))
    return ("ERR", code, last_error or "unknown error")


def load_latest_online_panel(stock_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    cons = pd.read_csv(DATA_DIR / "hs300_stock_list.csv")
    codes = sorted(set(cons["code"].astype(str)))
    last_local = pd.to_datetime(stock_df["日期"]).max()
    start_date = (last_local + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = "2026-05-19"
    cache_path = DATA_DIR / f"online_backfill_{end_date}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return cached, []

    results = []
    failures = []
    for idx, code in enumerate(codes, start=1):
        result = fetch_hist(code, start_date, end_date)
        if isinstance(result, tuple) and result and result[0] == "ERR":
            _, bad_code, message = result
            failures.append((bad_code, message))
        elif result is not None and not result.empty:
            results.append(result)
        if idx % 25 == 0:
            print(f"fetch_progress {idx}/{len(codes)} success={len(results)} fail={len(failures)}", flush=True)
        time.sleep(0.12)

    online_df = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(
            columns=[
                "股票代码",
                "日期",
                "开盘",
                "收盘",
                "最高",
                "最低",
                "成交量",
                "成交额",
                "振幅",
                "涨跌额",
                "换手率",
                "涨跌幅",
            ]
        )
    )
    if not online_df.empty:
        online_df.to_csv(cache_path, index=False)
    return online_df, failures


def build_eval_windows(trading_dates, start_after):
    candidates = []
    for idx, current_date in enumerate(trading_dates):
        if idx + 5 >= len(trading_dates):
            continue
        if current_date <= start_after:
            continue
        candidates.append((current_date, trading_dates[idx + 1], trading_dates[idx + 5]))

    if len(candidates) >= 4:
        return candidates[-4:]

    fallback = []
    for idx, current_date in enumerate(trading_dates):
        if idx + 5 < len(trading_dates):
            fallback.append((current_date, trading_dates[idx + 1], trading_dates[idx + 5]))
    return fallback[-4:]


def main():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
    stock_df = pd.read_csv(DATA_DIR / "stock_data.csv")
    stock_df["股票代码"] = stock_df["股票代码"].astype(str).str.zfill(6)
    stock_df["日期"] = pd.to_datetime(stock_df["日期"])
    last_local = stock_df["日期"].max()

    online_df, failures = load_latest_online_panel(stock_df)
    if not online_df.empty:
        online_df["股票代码"] = online_df["股票代码"].astype(str).str.zfill(6)
        online_df["日期"] = pd.to_datetime(online_df["日期"])

    all_df = (
        pd.concat([stock_df, online_df], ignore_index=True)
        if not online_df.empty
        else stock_df.copy()
    )
    all_df = (
        all_df.sort_values(["股票代码", "日期"])
        .drop_duplicates(["股票代码", "日期"], keep="last")
        .reset_index(drop=True)
    )

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    stockid2idx = {
        stock_id: idx for idx, stock_id in enumerate(sorted(all_df["股票代码"].astype(str).str.zfill(6).unique()))
    }
    processed, feature_columns = preprocess_predict_data(all_df.copy(), stockid2idx)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[feature_columns] = scaler.transform(processed[feature_columns])
    trading_dates = sorted(processed["日期"].dropna().unique())
    target_dates = {
        pd.Timestamp("2026-05-07"),
        pd.Timestamp("2026-05-08"),
        pd.Timestamp("2026-05-11"),
        pd.Timestamp("2026-05-12"),
    }
    windows = [window for window in build_eval_windows(trading_dates, last_local) if pd.Timestamp(window[0]) in target_dates]
    if len(windows) < 4:
        all_windows = []
        for idx, current_date in enumerate(trading_dates):
            if idx + 5 >= len(trading_dates):
                continue
            ts = pd.Timestamp(current_date)
            if ts in target_dates:
                all_windows.append((current_date, trading_dates[idx + 1], trading_dates[idx + 5]))
        windows = all_windows

    model = StockTransformer(
        input_dim=len(feature_columns),
        config=config,
        num_stocks=len(stockid2idx),
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    model.eval()

    returns = []
    rows = []
    for pred_date, buy_date, sell_date in windows:
        available_stock_ids = sorted(processed[processed["日期"] == pred_date]["股票代码"].astype(str).str.zfill(6).unique())
        seq_data, stock_ids = build_inference_sequences(
            processed,
            feature_columns,
            config["sequence_length"],
            available_stock_ids,
            pred_date,
        )
        if seq_data is None or not stock_ids:
            continue

        with torch.no_grad():
            scores = (
                model(torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0))
                .squeeze(0)
                .numpy()
            )

        top_indices = np.argsort(-scores)[:5]
        selected = [stock_ids[idx] for idx in top_indices]

        buy_open = all_df[all_df["日期"] == buy_date].set_index("股票代码")["开盘"]
        sell_open = all_df[all_df["日期"] == sell_date].set_index("股票代码")["开盘"]

        portfolio_return = 0.0
        valid_selected = []
        for stock_id in selected:
            if stock_id not in buy_open.index or stock_id not in sell_open.index:
                continue
            open_price = float(buy_open.loc[stock_id])
            if open_price <= 0:
                continue
            close_return = (float(sell_open.loc[stock_id]) - open_price) / open_price
            portfolio_return += 0.2 * close_return
            valid_selected.append(stock_id)

        returns.append(portfolio_return)
        rows.append(
            {
                "pred_date": str(pd.Timestamp(pred_date).date()),
                "buy_date": str(pd.Timestamp(buy_date).date()),
                "sell_date": str(pd.Timestamp(sell_date).date()),
                "stocks": "|".join(valid_selected),
                "return": portfolio_return,
            }
        )

    try:
        print(f"last_local {pd.Timestamp(last_local).date()}", flush=True)
        print(f"online_rows {len(online_df)} failed {len(failures)}", flush=True)
        print(f"latest_all_date {pd.Timestamp(all_df['日期'].max()).date()}", flush=True)
        for row in rows:
            print(
                row["pred_date"],
                row["buy_date"],
                row["sell_date"],
                row["stocks"],
                row["return"],
                flush=True,
            )
        print(f"avg_return {float(np.mean(returns)) if returns else None}", flush=True)
        print(f"fail_samples {failures[:5]}", flush=True)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
