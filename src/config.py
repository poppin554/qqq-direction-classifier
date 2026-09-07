"""
Central configuration: file paths and shared constants.

Every script that previously hardcoded an absolute Windows path
(C:/Users/User/Documents/DS PROJECT/...) now imports from here instead.
This is the only file you need to touch if the raw data filename changes.
"""
from pathlib import Path

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV_PATH = RAW_DATA_DIR / "qqq_1999-03-10_to_2026-08-26.csv"

# Train/test split boundary (see decision_log.md, Phase 5)
TRAIN_TEST_CUTOFF = "2018-01-01"

# The four leakage-safe, .shift(1)-disciplined features (see decision_log.md, Phase 4)
FEATURE_COLS = ["close_ma_5", "intraday_move", "lagged_return", "rolling_volatility"]

ROLLING_WINDOW = 5  # 5 trading days = weekly window for MA / volatility features
