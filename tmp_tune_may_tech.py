import itertools
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

import predict
from config import config
from model import StockTransformer
from predict import build_inference_sequences, build_portfolio, build_signal_frame, fit_theme_profile, preprocess_predict_data


START_PRED_DATE = pd.Timestamp("2026-05-01")
END_PRED_DATE = pd.Timestamp("2026-05-19")


def calc_return(portfolio, raw_df, code_col, date_col, open_col, buy_date, sell_date):
    buy_open = raw_df[raw_df[date_col] == buy_date].set_index(code_col)[open_col]
    sell_open = raw_df[raw_df[date_col] == sell_date].set_index(code_col)[open_col]
    total = 0.0
    names = []
    for stock_id, weight in portfolio:
        if stock_id not in buy_open.index or stock_id not in sell_open.index:
            continue
        buy_price = float(buy_open.loc[stock_id])
        if buy_price <= 0:
            continue
        ret = (float(sell_open.loc[stock_id]) - buy_price) / buy_price
        total += weight * ret
        names.append(f"{stock_id}:{weight:.4f}")
    return total, "|".join(names)


def refresh_adjusted_score(signal_df):
    cfg = predict.RANK_BLEND_CONFIG
    out = signal_df.copy()
    tech_mask = out["tech_attack"].astype(bool)
    tech_regime = float(out["tech_regime"].iloc[0]) if "tech_regime" in out else 0.0
    out["adjusted_score"] = (
        cfg["model_weight"] * predict._zscore(out["model_score"])
        + cfg["momentum_5_weight"] * predict._zscore(out["ret_5"])
        + cfg["momentum_20_weight"] * predict._zscore(out["ret_20"])
        + cfg["volatility_20_weight"] * predict._zscore(out["vol_20"])
        + cfg["oversold_rebound_weight"] * out["oversold_rebound"]
        + cfg["theme_profile_weight"] * predict._zscore(out["theme_similarity"])
        + cfg["core_growth_bonus"] * out["core_growth"].astype(float)
        + cfg["tech_attack_bonus"] * out["tech_attack"].astype(float)
        + cfg["tech_momentum_weight"] * predict._zscore(out["tech_momentum_score"]) * tech_mask.astype(float)
        + cfg["tech_regime_weight"] * tech_regime * tech_mask.astype(float)
        + cfg["tech_subsector_weight"] * out["tech_subsector_strength"] * tech_mask.astype(float)
    )
    out["adjusted_score"] += cfg["realtime_weight"] * predict._zscore(out["realtime_score"])
    out.loc[out["breakdown"], "adjusted_score"] -= 3.0
    out.loc[out["rt_breakdown"], "adjusted_score"] -= 1.5
    out["adjusted_score"] -= cfg["resistance_penalty_weight"] * out["resistance_score"]
    out.loc[out["resistance_trap"], "adjusted_score"] -= 1.2
    if "distribution_risk" in out:
        out["adjusted_score"] -= cfg["distribution_penalty_weight"] * out["distribution_risk"]
    if "distribution_trap" in out:
        out.loc[out["distribution_trap"], "adjusted_score"] -= 1.0
    out["target_growth_weight"] = cfg["tech_total_weight"] if tech_regime > 0 else cfg["quiet_tech_total_weight"]
    return out


def main():
    raw_df = pd.read_csv(DATA_DIR / "latest_stock_data.csv")
    code_col, date_col, open_col = raw_df.columns[0], raw_df.columns[1], raw_df.columns[2]
    raw_df[code_col] = raw_df[code_col].astype(str).str.zfill(6)
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], format="mixed")

    stock_ids = sorted(raw_df[code_col].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
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

    print(f"windows {len(windows)}")
    base_config = dict(predict.RANK_BLEND_CONFIG)
    theme_profile, theme_scale, _ = fit_theme_profile(raw_df)

    # Cache expensive model scores once per prediction date.
    cached = []
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
        baseline_return, _ = calc_return(baseline_portfolio, raw_df, code_col, date_col, open_col, buy_date, sell_date)
        signal_df = build_signal_frame(current_raw, sequence_stock_ids, scores, theme_profile, theme_scale, realtime_df=pd.DataFrame())
        cached.append((pred_date, buy_date, sell_date, signal_df, baseline_return))

    print(f"cached_windows {len(cached)}", flush=True)
    grid = [
        (0.45, 0.50, 0.65, 0.75, 0.82, 0.40),
        (0.35, 0.85, 0.85, 0.85, 0.88, 0.40),
        (0.55, 0.85, 0.85, 0.85, 0.88, 0.40),
        (0.35, 0.85, 0.45, 0.85, 0.82, 0.35),
        (0.55, 0.45, 0.85, 0.85, 0.82, 0.35),
        (0.35, 0.85, 0.85, 0.45, 0.88, 0.35),
        (0.55, 0.85, 0.45, 0.45, 0.78, 0.40),
        (0.45, 0.85, 0.85, 0.85, 0.82, 0.35),
    ]

    rows = []
    for core_bonus, tech_bonus, momentum_weight, subsector_weight, tech_total, max_single in grid:
        predict.RANK_BLEND_CONFIG.update(base_config)
        predict.RANK_BLEND_CONFIG.update({
            "core_growth_bonus": core_bonus,
            "tech_attack_bonus": tech_bonus,
            "tech_momentum_weight": momentum_weight,
            "tech_subsector_weight": subsector_weight,
            "tech_total_weight": tech_total,
            "max_single_weight": max_single,
        })

        returns = []
        baseline_returns = []
        portfolios = []
        for pred_date, buy_date, sell_date, signal_df, baseline_return in cached:
            signal_df = refresh_adjusted_score(signal_df)
            portfolio_df = build_portfolio(signal_df, top_k=5)
            portfolio = [(row.stock_id, float(row.weight)) for row in portfolio_df.itertuples(index=False)]
            ret, names = calc_return(portfolio, raw_df, code_col, date_col, open_col, buy_date, sell_date)
            returns.append(ret)
            baseline_returns.append(baseline_return)
            portfolios.append(names)

        returns_np = np.array(returns)
        baseline_np = np.array(baseline_returns)
        rows.append({
            "avg": float(returns_np.mean()),
            "sum": float(returns_np.sum()),
            "win_rate": float((returns_np > baseline_np).mean()),
            "min": float(returns_np.min()),
            "std": float(returns_np.std()),
            "score": float(returns_np.mean() + 0.25 * (returns_np > baseline_np).mean() - 0.2 * returns_np.std() + 0.1 * returns_np.min()),
            "core_growth_bonus": core_bonus,
            "tech_attack_bonus": tech_bonus,
            "tech_momentum_weight": momentum_weight,
            "tech_subsector_weight": subsector_weight,
            "tech_total_weight": tech_total,
            "max_single_weight": max_single,
            "last_portfolio": portfolios[-1] if portfolios else "",
        })

    result_df = pd.DataFrame(rows).sort_values(["score", "avg"], ascending=False).reset_index(drop=True)
    out_path = ROOT / "output" / "may_tech_param_search.csv"
    result_df.to_csv(out_path, index=False)
    print(result_df.head(12).to_string(index=False))
    print(f"written {out_path}")

    best = result_df.iloc[0].to_dict()
    print("best_params", {k: best[k] for k in [
        "core_growth_bonus", "tech_attack_bonus", "tech_momentum_weight",
        "tech_subsector_weight", "tech_total_weight", "max_single_weight"
    ]})


if __name__ == "__main__":
    main()
