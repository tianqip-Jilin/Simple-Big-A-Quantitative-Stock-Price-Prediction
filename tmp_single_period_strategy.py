import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "code" / "src"
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


def main():
    base_df = pd.read_csv(ROOT / "data" / "stock_data.csv")
    online_df = pd.read_csv(ROOT / "data" / "online_backfill_2026-05-19.csv")

    for df in (base_df, online_df):
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        df["日期"] = pd.to_datetime(df["日期"])

    raw_df = pd.concat([base_df, online_df], ignore_index=True)
    raw_df = (
        raw_df.sort_values(["股票代码", "日期"])
        .drop_duplicates(["股票代码", "日期"], keep="last")
        .reset_index(drop=True)
    )

    pred_date = pd.Timestamp("2026-04-24")
    buy_date = pd.Timestamp("2026-04-27")
    sell_date = pd.Timestamp("2026-04-30")

    scaler = joblib.load(ROOT / "model" / "60_158+39" / "scaler.pkl")
    stock_ids = sorted(raw_df["股票代码"].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}

    processed, feature_columns = preprocess_predict_data(raw_df.copy(), stockid2idx)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[feature_columns] = scaler.transform(processed[feature_columns])

    model = StockTransformer(input_dim=len(feature_columns), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(ROOT / "model" / "60_158+39" / "best_model.pth", map_location="cpu"))
    model.eval()

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

    theme_profile, theme_scale, _ = fit_theme_profile(current_raw)
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

    rows = []
    total_return = 0.0
    for row in portfolio.itertuples(index=False):
        stock_id = row.stock_id
        weight = float(row.weight)
        buy_price = float(buy_open.loc[stock_id])
        sell_price = float(sell_open.loc[stock_id])
        stock_return = (sell_price - buy_price) / buy_price
        weighted_return = weight * stock_return
        total_return += weighted_return
        rows.append(
            {
                "stock_id": stock_id,
                "weight": weight,
                "buy_open_2026_04_27": buy_price,
                "sell_open_2026_04_30": sell_price,
                "stock_return": stock_return,
                "weighted_return": weighted_return,
                "core_growth": bool(row.core_growth),
                "resistance_score": float(row.resistance_score),
                "resistance_trap": bool(row.resistance_trap),
            }
        )

    print(f"pred_date {pred_date.date()}")
    print(f"buy_date {buy_date.date()}")
    print(f"sell_date {sell_date.date()}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"total_return {total_return}")


if __name__ == "__main__":
    main()
