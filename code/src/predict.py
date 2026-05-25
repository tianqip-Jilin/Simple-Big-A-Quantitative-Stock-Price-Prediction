import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}

THEME_PROFILE_SEEDS = {
	'grid_equipment': {'600406', '002028', '600089', '600312'},
	'communication': {'300308', '300502', '000063', '300394', '601138'},
	'battery': {'300750', '300014'},
	'robotics': {'002050', '300124'},
}
THEME_SEED_CODES = {code for codes in THEME_PROFILE_SEEDS.values() for code in codes}
TECH_ATTACK_POOL = {
	'002371', '688981', '688012', '688008', '688041',
	'603501', '603986', '300782', '002049', '600584',
	'688126', '688396', '688256',
	'300308', '300502', '300394', '300476', '601138', '000063',
}
TECH_SUBSECTORS = {
	'ai_hardware': {'300308', '300502', '300394', '300476', '601138', '000063'},
	'semicap': {'002371', '688012', '688008'},
	'chip_design': {'688041', '603501', '603986', '300782', '002049', '688256'},
	'foundry_material': {'688981', '600584', '688126', '688396'},
}
CORE_GROWTH_POOL = {
	'600406', '002028', '600089', '600312',
	'300308', '300502', '000063', '300394', '601138',
	'300750', '300014',
	'002050', '300124',
} | TECH_ATTACK_POOL

RANK_BLEND_CONFIG = {
	'model_weight': 1.0,
	'momentum_5_weight': 0.18,
	'momentum_20_weight': 0.22,
	'volatility_20_weight': -0.16,
	'oversold_rebound_weight': 0.35,
	'theme_profile_weight': 1.15,
	'realtime_weight': 0.45,
	'resistance_penalty_weight': 0.85,
	'core_growth_bonus': 0.35,
	'tech_attack_bonus': 0.85,
	'tech_momentum_weight': 0.45,
	'tech_regime_weight': 0.55,
	'tech_subsector_weight': 0.85,
	'distribution_penalty_weight': 1.15,
	'theme_candidate_count': 3,
	'theme_candidate_pool': 120,
	'theme_total_weight': 0.65,
	'tech_total_weight': 0.82,
	'quiet_tech_total_weight': 0.62,
	'non_core_single_cap': 0.15,
	'max_single_weight': 0.35,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def _zscore(values):
	values = np.asarray(values, dtype=np.float64)
	std = np.nanstd(values)
	if not np.isfinite(std) or std < 1e-12:
		return np.zeros_like(values, dtype=np.float64)
	return (values - np.nanmean(values)) / (std + 1e-12)


def _pct_change(first, last):
	if not np.isfinite(first) or abs(first) < 1e-12:
		return 0.0
	return float((last - first) / first)


def _rsi(prices, period=14):
	if len(prices) <= period:
		return 50.0
	delta = np.diff(prices[-(period + 1):])
	gain = np.clip(delta, 0, None).mean()
	loss = (-np.clip(delta, None, 0)).mean()
	if loss < 1e-12:
		return 100.0
	rs = gain / loss
	return float(100.0 - 100.0 / (1.0 + rs))


def _oversold_rebound_signal(open_prices):
	if len(open_prices) < 60:
		return 0.0, False

	last = float(open_prices[-1])
	ma20 = float(np.mean(open_prices[-20:]))
	ma60 = float(np.mean(open_prices[-60:]))
	low20 = float(np.min(open_prices[-20:]))
	low60 = float(np.min(open_prices[-60:]))
	high20 = float(np.max(open_prices[-20:]))
	ret_5 = _pct_change(open_prices[-6], last)
	ret_20 = _pct_change(open_prices[-21], last)
	rsi14 = _rsi(open_prices, period=14)

	breakdown = (
		last < low20 * 0.985
		or last < ma60 * 0.94
		or (last < low60 * 1.01 and ret_20 < -0.08)
	)
	if breakdown:
		return -1.0, True

	drawdown_from_20d_high = (last / (high20 + 1e-12)) - 1.0
	near_support = last >= low20 * 1.015 or last >= ma60 * 0.97
	ma20_not_collapsing = ma20 >= ma60 * 0.96
	short_term_stabilizing = ret_5 > -0.035
	oversold = ret_20 < -0.025 or rsi14 < 42 or drawdown_from_20d_high < -0.05

	if oversold and near_support and ma20_not_collapsing and short_term_stabilizing:
		return 1.0, False
	if oversold and not near_support:
		return -0.35, False
	return 0.0, False


def _resistance_trap_signal(open_prices):
	if len(open_prices) < 30:
		return 0.0, False

	last = float(open_prices[-1])
	ma20 = float(np.mean(open_prices[-20:]))
	ma60 = float(np.mean(open_prices[-60:])) if len(open_prices) >= 60 else ma20
	high20 = float(np.max(open_prices[-20:]))
	high60 = float(np.max(open_prices[-60:])) if len(open_prices) >= 60 else high20
	ret_3 = _pct_change(open_prices[-4], last) if len(open_prices) >= 4 else 0.0
	ret_5 = _pct_change(open_prices[-6], last) if len(open_prices) >= 6 else 0.0
	ret_10 = _pct_change(open_prices[-11], last) if len(open_prices) >= 11 else 0.0
	ret_20 = _pct_change(open_prices[-21], last) if len(open_prices) >= 21 else ret_10
	rsi14 = _rsi(open_prices, period=14)

	near_high20 = last >= high20 * 0.975
	near_high60 = last >= high60 * 0.965
	extended_from_ma20 = last >= ma20 * 1.06
	extended_from_ma60 = last >= ma60 * 1.12
	short_term_fading = ret_3 < 0.005 and ret_5 < 0.015
	momentum_rollover = ret_5 < ret_10 - 0.025 or ret_3 < -0.005

	score = 0.0
	if near_high20 and ret_20 > 0.05:
		score += 0.35
	if near_high60 and ret_10 > 0.035:
		score += 0.2
	if extended_from_ma20:
		score += 0.2
	if extended_from_ma60:
		score += 0.15
	if short_term_fading:
		score += 0.25
	if momentum_rollover:
		score += 0.25
	if rsi14 > 67:
		score += 0.15

	score = min(score, 1.25)
	return score, score >= 0.75


def _price_profile_features(open_prices):
	if len(open_prices) == 0:
		return None
	last = float(open_prices[-1])
	ma5 = float(np.mean(open_prices[-5:])) if len(open_prices) >= 5 else last
	ma20 = float(np.mean(open_prices[-20:])) if len(open_prices) >= 20 else last
	ma60 = float(np.mean(open_prices[-60:])) if len(open_prices) >= 60 else ma20
	high20 = float(np.max(open_prices[-20:])) if len(open_prices) >= 20 else last
	low20 = float(np.min(open_prices[-20:])) if len(open_prices) >= 20 else last
	ret_3 = _pct_change(open_prices[-4], last) if len(open_prices) >= 4 else 0.0
	ret_5 = _pct_change(open_prices[-6], last) if len(open_prices) >= 6 else 0.0
	ret_10 = _pct_change(open_prices[-11], last) if len(open_prices) >= 11 else 0.0
	ret_20 = _pct_change(open_prices[-21], last) if len(open_prices) >= 21 else 0.0
	if len(open_prices) >= 21:
		daily_returns = np.diff(open_prices[-21:]) / (open_prices[-21:-1] + 1e-12)
		vol_20 = float(np.nanstd(daily_returns))
	else:
		vol_20 = 0.0
	high60 = float(np.max(open_prices[-60:])) if len(open_prices) >= 60 else high20
	oversold_rebound, breakdown = _oversold_rebound_signal(open_prices)
	resistance_score, resistance_trap = _resistance_trap_signal(open_prices)
	return {
		'ret_3': ret_3,
		'ret_5': ret_5,
		'ret_10': ret_10,
		'ret_20': ret_20,
		'vol_20': vol_20,
		'ma5_gap': _pct_change(ma5, last),
		'ma20_gap': _pct_change(ma20, last),
		'ma60_gap': _pct_change(ma60, last),
		'drawdown_20': _pct_change(high20, last),
		'drawdown_60': _pct_change(high60, last),
		'support_gap_20': _pct_change(low20, last),
		'rsi14': _rsi(open_prices, period=14) / 100.0,
		'oversold_rebound': oversold_rebound,
		'resistance_score': resistance_score,
		'resistance_trap': resistance_trap,
		'breakdown': breakdown,
	}


PROFILE_FEATURE_COLUMNS = [
	'ret_3', 'ret_5', 'ret_10', 'ret_20', 'vol_20',
	'ma5_gap', 'ma20_gap', 'ma60_gap', 'drawdown_20', 'support_gap_20',
	'rsi14', 'oversold_rebound',
]


def fit_theme_profile(raw_df):
	code_col = raw_df.columns[0]
	date_col = raw_df.columns[1]
	open_col = raw_df.columns[2]
	seed_codes = sorted({code for codes in THEME_PROFILE_SEEDS.values() for code in codes})
	rows = []
	for stock_id in seed_codes:
		history = raw_df[raw_df[code_col] == stock_id].sort_values(date_col)
		open_prices = history[open_col].astype(float).to_numpy()
		features = _price_profile_features(open_prices)
		if features is None or features['breakdown']:
			continue
		features['stock_id'] = stock_id
		rows.append(features)

	if not rows:
		raise ValueError('No valid theme seed stocks were available to fit the theme profile.')

	seed_df = pd.DataFrame(rows)
	profile = seed_df[PROFILE_FEATURE_COLUMNS].mean()
	scale = seed_df[PROFILE_FEATURE_COLUMNS].std().replace(0, np.nan).fillna(1.0)
	return profile, scale, seed_df


def _profile_similarity(row, profile, scale):
	vec = row[PROFILE_FEATURE_COLUMNS].astype(float)
	dist = np.sqrt(np.mean(((vec - profile) / (scale + 1e-12)) ** 2))
	return float(1.0 / (1.0 + dist))


def _pick_column(df, candidates):
	for name in candidates:
		if name in df.columns:
			return name
	return None


def fetch_realtime_snapshot():
	try:
		import akshare as ak
	except Exception as exc:
		print(f'AkShare unavailable, fallback to historical signals: {exc}')
		return pd.DataFrame()

	spot = pd.DataFrame()
	for fetcher_name in ['stock_zh_a_spot_em', 'stock_zh_a_spot']:
		try:
			spot = getattr(ak, fetcher_name)()
			if not spot.empty:
				print(f'Loaded realtime snapshot from AkShare: {fetcher_name}, rows={len(spot)}')
				break
		except Exception as exc:
			print(f'AkShare {fetcher_name} failed: {exc}')
	if spot.empty:
		print('AkShare realtime snapshot unavailable, fallback to historical signals.')
		return pd.DataFrame()

	code_col = _pick_column(spot, ['\u4ee3\u7801', 'code', 'symbol'])
	price_col = _pick_column(spot, ['\u6700\u65b0\u4ef7', 'trade', 'latest'])
	pct_col = _pick_column(spot, ['\u6da8\u8dcc\u5e45', 'changepercent', 'pct_chg'])
	amount_col = _pick_column(spot, ['\u6210\u4ea4\u989d', 'amount'])
	turnover_col = _pick_column(spot, ['\u6362\u624b\u7387', 'turnoverratio', 'turnover'])
	volume_ratio_col = _pick_column(spot, ['\u91cf\u6bd4', 'volume_ratio'])
	if code_col is None:
		return pd.DataFrame()

	normalized_code = spot[code_col].astype(str).str.extract(r'(\d{6})', expand=False).fillna('')
	out = pd.DataFrame({'stock_id': normalized_code.str.zfill(6)})
	out = out[out['stock_id'].str.match(r'^\d{6}$')].copy()
	for target, source in [
		('rt_price', price_col),
		('rt_pct_chg', pct_col),
		('rt_amount', amount_col),
		('rt_turnover', turnover_col),
		('rt_volume_ratio', volume_ratio_col),
	]:
		out[target] = pd.to_numeric(spot[source], errors='coerce') if source is not None else 0.0
	return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_realtime_signals(signal_df, realtime_df):
	for col in ['rt_price', 'rt_pct_chg', 'rt_amount', 'rt_turnover', 'rt_volume_ratio']:
		signal_df[col] = 0.0
	signal_df['rt_breakdown'] = False
	signal_df['realtime_score'] = 0.0

	if realtime_df.empty:
		return signal_df

	out = signal_df.drop(columns=['rt_price', 'rt_pct_chg', 'rt_amount', 'rt_turnover', 'rt_volume_ratio'], errors='ignore')
	out = out.merge(realtime_df, on='stock_id', how='left')
	for col in ['rt_price', 'rt_pct_chg', 'rt_amount', 'rt_turnover', 'rt_volume_ratio']:
		out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)

	historical_last = out['rt_price'] <= 0
	out.loc[historical_last, 'rt_price'] = np.nan
	rt_ma60_gap = _pct_change_series(out['rt_price'] / (1.0 + out['ma60_gap']), out['rt_price'])
	rt_support_gap = _pct_change_series(out['rt_price'] / (1.0 + out['support_gap_20']), out['rt_price'])
	out['rt_breakdown'] = (
		(out['rt_price'].notna())
		& (
			(rt_ma60_gap < -0.06)
			| (rt_support_gap < -0.015)
		)
	)

	out['realtime_score'] = (
		0.35 * _zscore(out['rt_pct_chg'])
		+ 0.20 * _zscore(np.log1p(out['rt_amount'].clip(lower=0)))
		+ 0.20 * _zscore(out['rt_turnover'])
		+ 0.15 * _zscore(out['rt_volume_ratio'])
		- 0.75 * out['rt_breakdown'].astype(float)
	)
	return out


def _pct_change_series(first, last):
	first = pd.Series(first, dtype='float64')
	last = pd.Series(last, dtype='float64')
	return (last - first) / (first.abs() + 1e-12)


def _tech_momentum_signal(open_prices, amount_values):
	if len(open_prices) < 30:
		return 0.0

	last = float(open_prices[-1])
	ma10 = float(np.mean(open_prices[-10:])) if len(open_prices) >= 10 else last
	ma20 = float(np.mean(open_prices[-20:])) if len(open_prices) >= 20 else last
	high20 = float(np.max(open_prices[-20:]))
	ret_3 = _pct_change(open_prices[-4], last) if len(open_prices) >= 4 else 0.0
	ret_5 = _pct_change(open_prices[-6], last) if len(open_prices) >= 6 else 0.0
	ret_10 = _pct_change(open_prices[-11], last) if len(open_prices) >= 11 else 0.0
	ret_20 = _pct_change(open_prices[-21], last) if len(open_prices) >= 21 else 0.0
	rsi14 = _rsi(open_prices, period=14)

	if len(amount_values) >= 20:
		amount_recent = float(np.nanmean(amount_values[-5:]))
		amount_base = float(np.nanmean(amount_values[-20:-5]))
		amount_ratio = amount_recent / (amount_base + 1e-12)
	else:
		amount_ratio = 1.0

	breakout_quality = 0.0
	if last >= high20 * 0.985 and ret_20 > 0.03:
		breakout_quality += 0.35
	if last > ma10 > ma20:
		breakout_quality += 0.25
	if ret_5 > 0 and ret_10 > 0:
		breakout_quality += 0.2
	if amount_ratio > 1.15:
		breakout_quality += 0.15
	if ret_3 < -0.025 or rsi14 > 78:
		breakout_quality -= 0.35

	score = (
		0.35 * ret_5
		+ 0.45 * ret_10
		+ 0.25 * ret_20
		+ 0.12 * np.log1p(max(amount_ratio, 0.0))
		+ breakout_quality
	)
	return float(score)


def _distribution_risk_signal(open_prices, amount_values, turnover_values):
	if len(open_prices) < 20:
		return 0.0, False

	last = float(open_prices[-1])
	ma5 = float(np.mean(open_prices[-5:]))
	ret_3 = _pct_change(open_prices[-4], last) if len(open_prices) >= 4 else 0.0
	ret_5 = _pct_change(open_prices[-6], last) if len(open_prices) >= 6 else 0.0
	ret_20 = _pct_change(open_prices[-21], last) if len(open_prices) >= 21 else 0.0
	daily_returns = np.diff(open_prices[-21:]) / (open_prices[-21:-1] + 1e-12) if len(open_prices) >= 21 else np.array([])
	vol_20 = float(np.nanstd(daily_returns)) if len(daily_returns) else 0.0

	if len(amount_values) >= 20:
		amount_recent = float(np.nanmean(amount_values[-5:]))
		amount_base = float(np.nanmean(amount_values[-20:]))
		amount_ratio = amount_recent / (amount_base + 1e-12)
	else:
		amount_ratio = 1.0

	if len(turnover_values) >= 20:
		turnover_recent = float(np.nanmean(turnover_values[-5:]))
		turnover_base = float(np.nanmean(turnover_values[-20:]))
		turnover_ratio = turnover_recent / (turnover_base + 1e-12)
	else:
		turnover_ratio = 1.0

	risk = 0.0
	if ret_3 < -0.025 and amount_ratio > 1.1:
		risk += 0.45
	if ret_5 < -0.04 and turnover_ratio > 1.15:
		risk += 0.35
	if last < ma5 * 0.98 and ret_20 > 0.05:
		risk += 0.25
	if vol_20 > 0.035 and amount_ratio > 1.05:
		risk += 0.15
	if ret_5 < -0.06:
		risk += 0.2

	risk = min(risk, 1.25)
	return risk, risk >= 0.75


def _tech_subsector(stock_id):
	for name, codes in TECH_SUBSECTORS.items():
		if stock_id in codes:
			return name
	return 'non_tech'


def build_signal_frame(raw_df, sequence_stock_ids, model_scores, theme_profile, theme_scale, realtime_df=None):
	code_col = raw_df.columns[0]
	date_col = raw_df.columns[1]
	open_col = raw_df.columns[2]
	amount_col = raw_df.columns[7]
	turnover_col = raw_df.columns[10]

	rows = []
	for stock_id, model_score in zip(sequence_stock_ids, model_scores):
		history = raw_df[raw_df[code_col] == stock_id].sort_values(date_col)
		open_prices = history[open_col].astype(float).to_numpy()
		if len(open_prices) == 0:
			continue

		features = _price_profile_features(open_prices)
		if features is None:
			continue
		row = {
			'stock_id': stock_id,
			'model_score': float(model_score),
			'theme_seed': stock_id in THEME_SEED_CODES,
			'core_growth': stock_id in CORE_GROWTH_POOL,
			'tech_attack': stock_id in TECH_ATTACK_POOL,
			'tech_subsector': _tech_subsector(stock_id),
		}
		row.update(features)
		amount_values = pd.to_numeric(history[amount_col], errors='coerce').fillna(0.0).to_numpy()
		row['tech_momentum_score'] = _tech_momentum_signal(open_prices, amount_values)
		turnover_values = pd.to_numeric(history[turnover_col], errors='coerce').fillna(0.0).to_numpy()
		row['distribution_risk'], row['distribution_trap'] = _distribution_risk_signal(open_prices, amount_values, turnover_values)
		rows.append(row)

	signal_df = pd.DataFrame(rows)
	if signal_df.empty:
		raise ValueError('No usable ranking signals were generated.')

	cfg = RANK_BLEND_CONFIG
	signal_df['theme_similarity'] = signal_df.apply(
		lambda row: _profile_similarity(row, theme_profile, theme_scale),
		axis=1,
	)
	tech_mask = signal_df['tech_attack'].astype(bool)
	if tech_mask.any():
		tech_regime = (
			float(signal_df.loc[tech_mask, 'ret_5'].mean() - signal_df['ret_5'].mean())
			+ 0.6 * float((signal_df.loc[tech_mask, 'ret_5'] > 0).mean() - (signal_df['ret_5'] > 0).mean())
			+ 0.25 * float(signal_df.loc[tech_mask, 'tech_momentum_score'].mean())
		)
	else:
		tech_regime = 0.0
	signal_df['tech_regime'] = tech_regime
	signal_df['tech_subsector_strength'] = 0.0
	for subsector, part in signal_df[tech_mask].groupby('tech_subsector'):
		if subsector == 'non_tech' or part.empty:
			continue
		strength = (
			float(part['ret_5'].mean() - signal_df['ret_5'].mean())
			+ 0.45 * float((part['ret_5'] > 0).mean() - (signal_df['ret_5'] > 0).mean())
			+ 0.35 * float(part['tech_momentum_score'].mean())
		)
		signal_df.loc[part.index, 'tech_subsector_strength'] = strength

	signal_df['adjusted_score'] = (
		cfg['model_weight'] * _zscore(signal_df['model_score'])
		+ cfg['momentum_5_weight'] * _zscore(signal_df['ret_5'])
		+ cfg['momentum_20_weight'] * _zscore(signal_df['ret_20'])
		+ cfg['volatility_20_weight'] * _zscore(signal_df['vol_20'])
		+ cfg['oversold_rebound_weight'] * signal_df['oversold_rebound']
		+ cfg['theme_profile_weight'] * _zscore(signal_df['theme_similarity'])
		+ cfg['core_growth_bonus'] * signal_df['core_growth'].astype(float)
		+ cfg['tech_attack_bonus'] * signal_df['tech_attack'].astype(float)
		+ cfg['tech_momentum_weight'] * _zscore(signal_df['tech_momentum_score']) * signal_df['tech_attack'].astype(float)
		+ cfg['tech_regime_weight'] * tech_regime * signal_df['tech_attack'].astype(float)
		+ cfg['tech_subsector_weight'] * signal_df['tech_subsector_strength'] * signal_df['tech_attack'].astype(float)
	)
	signal_df = add_realtime_signals(signal_df, realtime_df if realtime_df is not None else pd.DataFrame())
	signal_df['adjusted_score'] += cfg['realtime_weight'] * _zscore(signal_df['realtime_score'])
	signal_df.loc[signal_df['breakdown'], 'adjusted_score'] -= 3.0
	signal_df.loc[signal_df['rt_breakdown'], 'adjusted_score'] -= 1.5
	signal_df['adjusted_score'] -= cfg['resistance_penalty_weight'] * signal_df['resistance_score']
	signal_df.loc[signal_df['resistance_trap'], 'adjusted_score'] -= 1.2
	signal_df['adjusted_score'] -= cfg['distribution_penalty_weight'] * signal_df['distribution_risk']
	signal_df.loc[signal_df['distribution_trap'], 'adjusted_score'] -= 1.0
	signal_df['growth_bucket'] = signal_df['core_growth'] | signal_df['tech_attack']
	signal_df['target_growth_weight'] = cfg['tech_total_weight'] if tech_regime > 0 else cfg['quiet_tech_total_weight']
	return signal_df


def _apply_weight_caps(weights, cap_values):
	weights = np.asarray(weights, dtype=np.float64).copy()
	cap_values = np.asarray(cap_values, dtype=np.float64)
	for _ in range(10):
		over = weights > cap_values
		if not over.any():
			break
		excess = float((weights[over] - cap_values[over]).sum())
		weights[over] = cap_values[over]
		under = ~over
		room = np.maximum(cap_values[under] - weights[under], 0.0)
		if room.sum() <= 1e-12:
			break
		weights[under] += excess * room / (room.sum() + 1e-12)
	weights = weights / (weights.sum() + 1e-12)
	return weights


def build_portfolio(signal_df, top_k=5):
	eligible_df = signal_df[
		(~signal_df['breakdown']) & (~signal_df['rt_breakdown']) & (~signal_df['resistance_trap']) & (~signal_df['distribution_trap'])
	].copy()
	if len(eligible_df) < top_k:
		eligible_df = signal_df[(~signal_df['breakdown']) & (~signal_df['rt_breakdown'])].copy()
	if len(eligible_df) < top_k:
		eligible_df = signal_df.copy()

	ranked = eligible_df.sort_values('model_score', ascending=False).reset_index(drop=True)
	if len(ranked) < top_k:
		raise ValueError(f'Not enough stocks for prediction: {len(ranked)}')

	cfg = RANK_BLEND_CONFIG
	theme_pool = eligible_df.sort_values('adjusted_score', ascending=False).head(cfg['theme_candidate_pool'])
	seed_candidates = theme_pool[theme_pool['growth_bucket']].sort_values(
		['adjusted_score', 'theme_similarity'],
		ascending=False,
	).head(cfg['theme_candidate_count'])
	theme_candidates = seed_candidates
	if len(theme_candidates) < cfg['theme_candidate_count']:
		fallback_candidates = theme_pool[~theme_pool['stock_id'].isin(theme_candidates['stock_id'])].sort_values(
			['theme_similarity', 'adjusted_score'],
			ascending=False,
		).head(cfg['theme_candidate_count'] - len(theme_candidates))
		theme_candidates = pd.concat([theme_candidates, fallback_candidates], ignore_index=True)
	theme_candidates = theme_candidates.sort_values(
		['theme_similarity', 'adjusted_score'],
		ascending=False,
	)
	# Keep core growth names preferred, but allow obviously stronger non-core stocks
	# to fill the portfolio through the ranked bucket instead of forcing the core list.
	selected = pd.concat([theme_candidates, ranked], ignore_index=True)
	selected = selected.drop_duplicates(subset=['stock_id'], keep='first').head(top_k).copy()

	score = selected['adjusted_score'].to_numpy(dtype=np.float64)
	score = score - score.max()
	raw_weights = np.exp(score / 0.75)

	growth_mask = selected['growth_bucket'].astype(bool).to_numpy()
	if growth_mask.any() and (~growth_mask).any():
		weights = np.zeros(len(selected), dtype=np.float64)
		growth_total = float(selected['target_growth_weight'].iloc[0]) if 'target_growth_weight' in selected else cfg['theme_total_weight']
		weights[growth_mask] = growth_total * raw_weights[growth_mask] / (raw_weights[growth_mask].sum() + 1e-12)
		weights[~growth_mask] = (1.0 - growth_total) * raw_weights[~growth_mask] / (raw_weights[~growth_mask].sum() + 1e-12)
	else:
		weights = raw_weights / (raw_weights.sum() + 1e-12)

	caps = np.where(growth_mask, cfg['max_single_weight'], cfg['non_core_single_cap'])
	weights = _apply_weight_caps(weights, caps)
	weights = np.round(weights, 4)
	diff = round(1.0 - float(weights.sum()), 4)
	weights[np.argmax(weights)] = round(float(weights[np.argmax(weights)]) + diff, 4)

	selected['weight'] = weights
	selected['theme_bucket'] = growth_mask
	return selected


def load_prediction_source_data(default_file):
	base_file = os.path.join(config['data_path'], 'stock_data.csv')
	data_file = base_file if os.path.exists(base_file) else default_file
	raw_df = pd.read_csv(data_file)

	backfill_files = []
	if os.path.isdir(config['data_path']):
		for name in os.listdir(config['data_path']):
			if name.startswith('online_backfill_') and name.endswith('.csv'):
				backfill_files.append(os.path.join(config['data_path'], name))

	if backfill_files:
		frames = [raw_df]
		for path in sorted(backfill_files):
			frames.append(pd.read_csv(path))
		raw_df = pd.concat(frames, ignore_index=True)
		code_col = raw_df.columns[0]
		date_col = raw_df.columns[1]
		raw_df[code_col] = raw_df[code_col].astype(str).str.zfill(6)
		raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce', format='mixed')
		raw_df = (
			raw_df.sort_values([code_col, date_col])
			.drop_duplicates([code_col, date_col], keep='last')
			.reset_index(drop=True)
		)
	return raw_df


def main():
	latest_data_file = os.path.join(config['data_path'], 'latest_stock_data.csv')
	data_file = latest_data_file if os.path.exists(latest_data_file) else os.path.join(config['data_path'], 'train.csv')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'], format='mixed')
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]

	# 仅输出前5，权重固定 0.2
	if len(ranked_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')
	top5 = ranked_stock_ids[:5]
	theme_profile, theme_scale, theme_seed_df = fit_theme_profile(raw_df)
	realtime_df = fetch_realtime_snapshot()
	signal_df = build_signal_frame(raw_df, sequence_stock_ids, scores, theme_profile, theme_scale, realtime_df)
	portfolio = build_portfolio(signal_df, top_k=5)
	top5 = portfolio['stock_id'].tolist()
	portfolio_weights = portfolio['weight'].tolist()
	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': portfolio_weights,
	})
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	output_df.to_csv(output_path, index=False)
	portfolio.to_csv(os.path.join('./output/', 'selection_details.csv'), index=False)
	theme_seed_df.to_csv(os.path.join('./output/', 'theme_profile_seeds.csv'), index=False)
	signal_df.sort_values('adjusted_score', ascending=False).to_csv(
		os.path.join('./output/', 'signal_scores.csv'),
		index=False,
	)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
