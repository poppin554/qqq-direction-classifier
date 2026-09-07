"""
Chronological train/test split via boolean date masking (not .iloc[] positional
slicing, see decision_log.md, Phase 5, for why: row positions shift whenever
upstream rows are dropped, dates don't).
"""
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import TRAIN_TEST_CUTOFF


def split_train_test(df: pd.DataFrame, cutoff: str = TRAIN_TEST_CUTOFF):
    cutoff_ts = pd.Timestamp(cutoff)

    train_df = df[df['Date'] < cutoff_ts]
    test_df = df[df['Date'] >= cutoff_ts]

    return train_df, test_df
