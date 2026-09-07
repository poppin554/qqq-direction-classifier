"""
Four technical features, each derived purely from OHLCV data and shifted by
1 day (.shift(1)) so today's row only reflects information available before
today's close. See decision_log.md, Phase 4, for the shift-discipline
reasoning and the two bugs caught while building these (double return
calculation, missing final shift on rolling_volatility).
"""
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import ROLLING_WINDOW


def build_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    df = df.copy()

    df['close_ma_5'] = (
        df['Close']
        .shift(1)
        .rolling(window)
        .mean()
    )

    df['intraday_move'] = (
        (df['Close'] - df['Open'])
        .shift(1)
    )

    df['lagged_return'] = (
        ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))
        .shift(1)
    )

    df['rolling_volatility'] = (
        (((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).rolling(window).std())
        .shift(1)
    )

    return df
