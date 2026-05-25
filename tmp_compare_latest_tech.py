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
from predict import build_inference_sequences, build_portfolio, build_signal_frame, fit_theme_profile, preprocess_predict_data


START_PRED_DATE = pd.Timestamp("2026-04-01")
END_PRED_DATE = pd.Timestamp("2026-05-19")


def main():
    raw_df = pd.read_csv(DATA_DIR / "latest_stock_data.csv")
    code_col, date_col, open_col = raw_df.columns[0], raw_df.columns[1], raw_df.columns[2]
    raw_df[code_col] = raw_df[code_col].astype(str).str.zfill(6)
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], format="mixed")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    stock_ids = sorted(raw_df[code_col].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}

    processed, feature_columns = preprocess_predict_data(raw_df.copy(), stockid2idx)
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[feature_columns] = scaler.transform(processed[feature_columns])

    model = StockTransformer(input_dim=len(feature_columns), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location="cpu"))
    model.eval()

    trading_dates = sorted(processed[date_col].dropna().unique())
    windows = []
    for idx, current_date in enumerate(trading_dates):
        ts = pd.Timestamp(current_date)
        if ts < START_PRED_DATE or ts > END_PRED_DATE:
            continue
        if idx + 5 < len(trading_dates):
            windows.append((current_date, trading_dates[idx + 1], trading_dates[idx + 5]))

    theme_profile, theme_scale, _ = fit_theme_profile(raw_df)
    baseline_returns = []
    latest_returns = []
    rows = []

    for pred_date, buy_date, sell_date in windows:
        current_raw = raw_df[raw_df[date_col] <= pred_date].copy()
        current_processed = processed[processed[date_col] <= pred_date].copy()
        available_stock_ids = sorted(current_processed[current_processed[date_col] == pred_date][code_col].astype(str).str.zfill(6).unique())
        sequences_np, sequence_stock_ids = build_inference_sequences(
            current_processed,
            feature_columns,
            config["sequence_length"],
            available_stock_ids,
            pred_date,
        )
        with torch.no_grad():
            scores = model(torch.from_numpy(sequences_np).unsqueeze(0).float()).squeeze(0).detach().cpu().numpy()

        order = np.argsort(scores)[::-1]
        baseline_portfolio = [(sequence_stock_ids[idx], 0.2) for idx in order[:5]]

        signal_df = build_signal_frame(current_raw, sequence_stock_ids, scores, theme_profile, theme_scale, realtime_df=pd.DataFrame())
        strategy_df = build_portfolio(signal_df, top_k=5)
        strategy_portfolio = [(row.stock_id, float(row.weight)) for row in strategy_df.itertuples(index=False)]

        buy_open = raw_df[raw_df[date_col] == buy_date].set_index(code_col)[open_col]
        sell_open = raw_df[raw_df[date_col] == sell_date].set_index(code_col)[open_col]

        def calc(portfolio):
            total = 0.0
            valid = []
            for stock_id, weight in portfolio:
                if stock_id not in buy_open.index or stock_id not in sell_open.index:
                    continue
                buy_price = float(buy_open.loc[stock_id])
                if buy_price <= 0:
                    continue
                ret = (float(sell_open.loc[stock_id]) - buy_price) / buy_price
                total += weight * ret
                valid.append(f"{stock_id}:{weight:.4f}")
            return total, "|".join(valid)

        baseline_ret, baseline_names = calc(baseline_portfolio)
        latest_ret, latest_names = calc(strategy_portfolio)
        baseline_returns.append(baseline_ret)
        latest_returns.append(latest_ret)
        rows.append({
            "pred_date": str(pd.Timestamp(pred_date).date()),
            "buy_date": str(pd.Timestamp(buy_date).date()),
            "sell_date": str(pd.Timestamp(sell_date).date()),
            "baseline_return": baseline_ret,
            "latest_return": latest_ret,
            "baseline_portfolio": baseline_names,
            "latest_portfolio": latest_names,
        })

    detail_df = pd.DataFrame(rows)
    out_path = ROOT / "output" / "rolling_compare_latest_tech.csv"
    detail_df.to_csv(out_path, index=False)
    print(f"windows {len(rows)}")
    print(f"baseline_avg {float(np.mean(baseline_returns))}")
    print(f"latest_avg {float(np.mean(latest_returns))}")
    print(f"baseline_sum {float(np.sum(baseline_returns))}")
    print(f"latest_sum {float(np.sum(latest_returns))}")
    print(f"latest_win_rate {float(np.mean(np.array(latest_returns) > np.array(baseline_returns)))}")
    print(detail_df.tail(8).to_string(index=False))
    print(f"detail_csv {out_path}")


if __name__ == "__main__":
    main()
