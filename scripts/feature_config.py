"""Shared config for the 81-dim -> compressed feature pipeline.

Data source is condition_diffusion/data ONLY (raw per-day caches: stock_market,
stock_status, barra_factors, trading_descriptors, trade_cal.pkl). We rebuild the
81-dim cross-sectional snapshot ourselves from these, replicating the feature
definitions used by deep_learning_framework's sample_construction.py (read for
reference only, never imported/executed):
  - 7 base price/volume: open/high/low/close/vwap as the ratio to the price 39
    trading days earlier (adjusted by adj_factor); volume/amt as the log
    day-over-day change (log(x_t) - log(x_{t-1}))
  - 10 Barra style factors + 64 trading_descriptors factors: robust-standardized
    (median / IQR) cross-sectionally, separately for each trading day
All outputs of this pipeline are written under condition_diffusion only.
"""
from __future__ import annotations

from pathlib import Path

CD_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CD_ROOT / "data"
STOCK_MARKET_DIR = DATA_DIR / "stock_market"
STOCK_STATUS_DIR = DATA_DIR / "stock_status"
BARRA_FACTORS_DIR = DATA_DIR / "barra_factors"
TRADING_DESCRIPTORS_DIR = DATA_DIR / "trading_descriptors"
TRADE_CAL_PATH = DATA_DIR / "trade_cal.pkl"

REQUEST_START = "2023-01-01"
REQUEST_END = "2026-07-31"
PRICE_LOOKBACK_SESSIONS = 39  # ratio to the price 39 trading days earlier (40-day window)
UNIVERSE = "winda"

# --- write targets (condition_diffusion only) ---
TRAIN_DATA_DIR = CD_ROOT / "train_data"
SCHEME_A_CSV = TRAIN_DATA_DIR / "features_schemeA.csv"
SCHEME_B_CSV = TRAIN_DATA_DIR / "features_schemeB.csv"
ANALYSIS_DIR = CD_ROOT / "analysis"
CLIP_BOUNDS_PATH = ANALYSIS_DIR / "descriptor_clip_bounds.pkl"
PCA_MODEL_PATH = ANALYSIS_DIR / "scheme_b_pca.pkl"
CORR_REPORT_PATH = ANALYSIS_DIR / "feature_correlation_report.md"
CORR_MATRIX_PATH = ANALYSIS_DIR / "feature_correlation_matrix.pkl"

K_PROPS = ["open", "high", "low", "close", "vwap"]

# --- the 81 raw feature names, in the exact column order sample_construction.py
# uses when building dataset_X (see deep_learning_framework/codes/sample_construction.py:149):
#   ['open','high','low','close','volume','amt','vwap'] + style_factors + factor_list
FEATURE_NAMES = [
    # 1-7: base price/volume
    "open", "high", "low", "close", "volume", "amt", "vwap",
    # 8-17: 10 Barra style factors (const_var.StyleFactors enum order)
    "beta", "earnings_yield", "growth", "leverage", "liquidity",
    "momentum", "nlsize", "size", "value", "volatility",
    # 18-81: 64 trading_descriptors factors (cfg/trading_descriptors.xlsx order)
    "amt_1m_3m", "swap_1m", "swap_3m", "swap_1y", "corr_close_turnover",
    "ivol", "ivr", "spread_bias", "est_num_diff", "illiq",
    "ln_volume_mean_1m", "ln_volume_mean_3m", "ln_volume_mean_6m", "ln_volume_mean_12m",
    "ln_volume_std_1m", "ln_volume_std_3m", "ln_volume_std_6m", "ln_volume_std_12m",
    "turnover_mean_1m", "turnover_mean_3m", "turnover_mean_6m",
    "turnover_std_1m", "turnover_std_3m", "turnover_std_6m",
    "turnover_stdrate_1m", "turnover_stdrate_3m", "turnover_stdrate_6m",
    "volume_1m_div_12m", "volume_1m_minus_12m", "volume_std_1m_div_12m",
    "clo_5d_60d", "close_max_div_min_1m", "close_max_div_min_3m", "close_max_div_min_6m",
    "vwap_5d_60d",
    "return_max_1m", "return_std_1m", "return_std_1w", "return_std_3m",
    "return_std_6m", "return_std_12m",
    "mom_1y", "ideal_reversal", "down_list", "mom_3m", "mom_6m", "mom_2y",
    "mom_1y_1m", "reverse_1m", "up_list",
    "duvol", "ncskew", "ana_cov", "specific_mom1", "specific_mom6", "specific_mom12",
    "io_to_float_a_share", "stk_quantity_g", "mean_stkvaluetonav",
    "delta_io_to_float_share", "top_ten_io_to_float_a_share",
    "top_ten_stk_quantity_g", "top_ten_mean_stkvaluetonav", "delta_top_ten_io",
]
assert len(FEATURE_NAMES) == 81, len(FEATURE_NAMES)
STYLE_FACTORS = FEATURE_NAMES[7:17]
DESCRIPTOR_FACTORS = FEATURE_NAMES[17:81]
assert len(STYLE_FACTORS) == 10 and len(DESCRIPTOR_FACTORS) == 64

# --- Scheme A: cluster-representative selection (see analysis/feature_correlation_report.md) ---
# One representative per correlation cluster; the two oversized clusters
# (>=13 members) get two representatives each. duvol, delta_top_ten_io and
# stk_quantity_g are dropped as redundant/weak. 19 features.
SCHEME_A_FEATURES = [
    "turnover_stdrate_3m",   # turnover-std-change cluster
    "amt",                    # volume/amt cluster
    "volatility",              # big turnover-activity/volatility cluster (rep 1)
    "swap_3m",                  # big turnover-activity/volatility cluster (rep 2)
    "down_list",                 # standalone
    "vwap",                       # price-level + short reversal cluster (rep 1)
    "reverse_1m",                  # price-level + short reversal cluster (rep 2)
    "volume_1m_div_12m",            # volume-trend-ratio cluster
    "ivr",                            # standalone
    "corr_close_turnover",             # standalone
    "momentum",                         # momentum family (barra composite)
    "est_num_diff",                      # standalone (analyst)
    "delta_io_to_float_share",            # standalone (institutional flow)
    "size",                                # size / volume-level cluster
    "value",                                # valuation cluster
    "io_to_float_a_share",                   # institutional-holding-level cluster
    "growth",                                 # standalone
    "nlsize",                                  # standalone
    "top_ten_stk_quantity_g",                   # standalone
]
assert len(SCHEME_A_FEATURES) == 19, len(SCHEME_A_FEATURES)
assert set(SCHEME_A_FEATURES) <= set(FEATURE_NAMES)

# --- Scheme B: PCA on standardized 81-dim features ---
SCHEME_B_N_COMPONENTS = 17
