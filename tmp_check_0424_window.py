import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
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


PRED_DATE = pd.Timestamp("2026-04-24")
BUY_DATE = pd.Timestamp("2026-04-27")
SELL_DATE = pd.Timestamp("2026-04-30")
THEME_PROFILE_END_DATE = pd.Timestamp("2026-05-19")


def main():
    raw_df = pd.read_csv(DATA_DIR / "latest_stock_data.csv")
    code_col, date_col, open_col = raw_df.columns[0], raw_df.columns[1], raw_df.columns[2]
    raw_df[code_col] = raw_df[code_col].astype(str).str.zfill(6)
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], format="mixed")

    current_raw = raw_df[raw_df[date_col] <= PRED_DATE].copy()
    stock_ids = sorted(raw_df[code_col].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    processed, feature_columns = preprocess_predict_data(current_raw.copy(), stockid2idx)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[feature_columns] = scaler.transform(processed[feature_columns])

    available_stock_ids = sorted(
        processed[processed[date_col] == PRED_DATE][code_col].astype(str).str.zfill(6).unique()
    )
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        feature_columns,
        config["sequence_length"],
        available_stock_ids,
        PRED_DATE,
    )

    model = StockTransformer(input_dim=len(feature_columns), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(sequences_np).unsqueeze(0).float()).squeeze(0).detach().cpu().numpy()

    theme_raw = raw_df[raw_df[date_col] <= THEME_PROFILE_END_DATE].copy()
    theme_profile, theme_scale, _ = fit_theme_profile(theme_raw)
    signal_df = build_signal_frame(
        current_raw,
        sequence_stock_ids,
        scores,
        theme_profile,
        theme_scale,
        realtime_df=pd.DataFrame(),
    )
    portfolio_df = build_portfolio(signal_df, top_k=5)

    buy_open = raw_df[raw_df[date_col] == BUY_DATE].set_index(code_col)[open_col]
    sell_open = raw_df[raw_df[date_col] == SELL_DATE].set_index(code_col)[open_col]

    rows = []
    total = 0.0
    for row in portfolio_df.itertuples(index=False):
        stock_id = row.stock_id
        weight = float(row.weight)
        buy_price = float(buy_open.loc[stock_id])
        sell_price = float(sell_open.loc[stock_id])
        ret = (sell_price - buy_price) / buy_price
        contrib = weight * ret
        total += contrib
        rows.append({
            "stock_id": stock_id,
            "weight": weight,
            "buy_open_2026_04_27": buy_price,
            "sell_open_2026_04_30": sell_price,
            "return": ret,
            "weighted_return": contrib,
        })

    detail_df = pd.DataFrame(rows)
    out_path = ROOT / "output" / "check_0424_to_0430.csv"
    detail_df.to_csv(out_path, index=False)
    print(detail_df.to_string(index=False))
    print(f"portfolio_return {total}")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
