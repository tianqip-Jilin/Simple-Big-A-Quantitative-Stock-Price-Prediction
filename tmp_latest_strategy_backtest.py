import sys
from pathlib import Path

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
from predict import (
    build_inference_sequences,
    build_portfolio,
    build_signal_frame,
    fit_theme_profile,
    preprocess_predict_data,
)


TARGET_DATES = [
    pd.Timestamp("2026-05-07"),
    pd.Timestamp("2026-05-08"),
    pd.Timestamp("2026-05-11"),
    pd.Timestamp("2026-05-12"),
]


def load_all_data() -> pd.DataFrame:
    base_df = pd.read_csv(DATA_DIR / "stock_data.csv")
    backfill_path = DATA_DIR / "online_backfill_2026-05-19.csv"
    if not backfill_path.exists():
        raise FileNotFoundError(f"Missing cached online backfill: {backfill_path}")
    online_df = pd.read_csv(backfill_path)

    for df in (base_df, online_df):
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        df["日期"] = pd.to_datetime(df["日期"])

    all_df = pd.concat([base_df, online_df], ignore_index=True)
    all_df = (
        all_df.sort_values(["股票代码", "日期"])
        .drop_duplicates(["股票代码", "日期"], keep="last")
        .reset_index(drop=True)
    )
    return all_df


def build_windows(trading_dates):
    windows = []
    target_set = set(TARGET_DATES)
    for idx, current_date in enumerate(trading_dates):
        if idx + 5 >= len(trading_dates):
            continue
        ts = pd.Timestamp(current_date)
        if ts in target_set:
            windows.append((current_date, trading_dates[idx + 1], trading_dates[idx + 5]))
    return windows


def main():
    raw_df = load_all_data()
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    stock_ids = sorted(raw_df["股票代码"].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}

    processed, feature_columns = preprocess_predict_data(raw_df.copy(), stockid2idx)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[feature_columns] = scaler.transform(processed[feature_columns])

    model = StockTransformer(
        input_dim=len(feature_columns),
        config=config,
        num_stocks=len(stock_ids),
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    model.eval()

    trading_dates = sorted(processed["日期"].dropna().unique())
    windows = build_windows(trading_dates)
    theme_profile, theme_scale, _ = fit_theme_profile(raw_df)

    rows = []
    returns = []
    for pred_date, buy_date, sell_date in windows:
        current_raw = raw_df[raw_df["日期"] <= pred_date].copy()
        current_processed = processed[processed["日期"] <= pred_date].copy()
        available_stock_ids = sorted(
            current_processed[current_processed["日期"] == pred_date]["股票代码"].astype(str).str.zfill(6).unique()
        )
        sequences_np, sequence_stock_ids = build_inference_sequences(
            current_processed,
            feature_columns,
            config["sequence_length"],
            available_stock_ids,
            pred_date,
        )

        with torch.no_grad():
            scores = (
                model(torch.from_numpy(sequences_np).unsqueeze(0).float())
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        signal_df = build_signal_frame(
            current_raw,
            sequence_stock_ids,
            scores,
            theme_profile,
            theme_scale,
            realtime_df=pd.DataFrame(),
        )
        portfolio = build_portfolio(signal_df, top_k=5)

        buy_open = raw_df[raw_df["日期"] == buy_date].set_index("股票代码")["开盘"]
        sell_open = raw_df[raw_df["日期"] == sell_date].set_index("股票代码")["开盘"]

        portfolio_return = 0.0
        selected = []
        for row in portfolio.itertuples(index=False):
            stock_id = row.stock_id
            weight = float(row.weight)
            if stock_id not in buy_open.index or stock_id not in sell_open.index:
                continue
            buy_price = float(buy_open.loc[stock_id])
            if buy_price <= 0:
                continue
            stock_return = (float(sell_open.loc[stock_id]) - buy_price) / buy_price
            portfolio_return += weight * stock_return
            selected.append(f"{stock_id}:{weight:.4f}")

        returns.append(portfolio_return)
        rows.append(
            {
                "pred_date": str(pd.Timestamp(pred_date).date()),
                "buy_date": str(pd.Timestamp(buy_date).date()),
                "sell_date": str(pd.Timestamp(sell_date).date()),
                "portfolio": "|".join(selected),
                "return": portfolio_return,
            }
        )

    for row in rows:
        print(row["pred_date"], row["buy_date"], row["sell_date"], row["portfolio"], row["return"], flush=True)
    print(f"avg_return {float(np.mean(returns)) if returns else None}", flush=True)


if __name__ == "__main__":
    main()
