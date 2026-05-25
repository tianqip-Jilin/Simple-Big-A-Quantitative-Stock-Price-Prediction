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
    build_signal_frame,
    fit_theme_profile,
    preprocess_predict_data,
)


PRED_DATE = pd.Timestamp("2026-04-24")
THEME_PROFILE_END_DATE = pd.Timestamp("2026-05-19")
TARGET = "300394"


def main():
    raw_df = pd.read_csv(DATA_DIR / "latest_stock_data.csv")
    code_col, date_col, open_col = raw_df.columns[0], raw_df.columns[1], raw_df.columns[2]
    volume_col, amount_col, turnover_col, pct_col = raw_df.columns[6], raw_df.columns[7], raw_df.columns[10], raw_df.columns[11]
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
        processed, feature_columns, config["sequence_length"], available_stock_ids, PRED_DATE
    )

    model = StockTransformer(input_dim=len(feature_columns), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(sequences_np).unsqueeze(0).float()).squeeze(0).detach().cpu().numpy()

    theme_raw = raw_df[raw_df[date_col] <= THEME_PROFILE_END_DATE].copy()
    theme_profile, theme_scale, _ = fit_theme_profile(theme_raw)
    signal_df = build_signal_frame(current_raw, sequence_stock_ids, scores, theme_profile, theme_scale, realtime_df=pd.DataFrame())

    cols = [
        "stock_id", "model_score", "adjusted_score", "theme_similarity", "tech_momentum_score",
        "tech_regime", "tech_subsector_strength", "ret_3", "ret_5", "ret_10", "ret_20",
        "vol_20", "ma5_gap", "ma20_gap", "drawdown_20", "rsi14", "resistance_score",
        "resistance_trap", "breakdown", "oversold_rebound",
    ]
    target_signal = signal_df[signal_df["stock_id"] == TARGET][cols]
    rank_df = signal_df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    rank = int(rank_df.index[rank_df["stock_id"] == TARGET][0]) + 1

    hist = current_raw[current_raw[code_col] == TARGET].sort_values(date_col).tail(25).copy()
    hist["amount_ma5"] = hist[amount_col].rolling(5).mean()
    hist["amount_ma20"] = hist[amount_col].rolling(20).mean()
    hist["amount_ratio_5_20"] = hist["amount_ma5"] / (hist["amount_ma20"] + 1e-12)
    hist["turnover_ma5"] = hist[turnover_col].rolling(5).mean()
    hist["turnover_ma20"] = hist[turnover_col].rolling(20).mean()
    hist["turnover_ratio_5_20"] = hist["turnover_ma5"] / (hist["turnover_ma20"] + 1e-12)

    print(f"rank_by_adjusted_score {rank}")
    print(target_signal.to_string(index=False))
    print()
    print(hist[[date_col, open_col, pct_col, volume_col, amount_col, turnover_col, "amount_ratio_5_20", "turnover_ratio_5_20"]].tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
