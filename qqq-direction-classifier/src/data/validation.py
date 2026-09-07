"""
OHLCV sanity checks: negative values, High < Low, Open/Close outside High/Low range.
See decision_log.md, Phase 2, for the reasoning behind inclusive bounds and
np.isclose() tolerance (instead of exact equality or rounding raw data).
"""
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV_PATH


def load_and_validate(csv_path=RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    negative_check = df[['Open', 'High', 'Low', 'Close', 'Volume']].any(axis=1) < 0

    high_low_check = df['High'] < df['Low']

    open_range_check = (
        (df['Open'] > df['High']) & ~np.isclose(df['Open'], df['High']) |
        (df['Open'] < df['Low']) & ~np.isclose(df['Open'], df['Low'])
    )

    close_range_check = (
        (df['Close'] > df['High']) & ~np.isclose(df['Close'], df['High']) |
        (df['Close'] < df['Low']) & ~np.isclose(df['Close'], df['Low'])
    )

    df['fails_negative'] = negative_check
    df['fails_high_low'] = high_low_check
    df['fails_open_range'] = open_range_check
    df['fails_close_range'] = close_range_check

    df['any_violation'] = df[['fails_negative', 'fails_high_low',
                               'fails_open_range', 'fails_close_range']].any(axis=1)

    return df


if __name__ == "__main__":
    df = load_and_validate()
    violations = df[df['any_violation']]

    print(f"Total violating rows: {len(violations)}")
    print(violations[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'fails_negative', 'fails_high_low',
                       'fails_open_range', 'fails_close_range']])
