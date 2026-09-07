"""
Binary target: 1 if next day's Close is higher than today's, else 0.
See decision_log.md, Phase 3, for why the last row is dropped rather than
fabricated.

Order matters here: .astype(int) runs first (matches the original, tested
pipeline), then the last row is explicitly overwritten with np.nan, which
upcasts the whole column from int64 to float64. This is why `target` ends up
as float64 (1.0/0.0) after dropna(), not int64, downstream.
"""
import pandas as pd
import numpy as np


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.iloc[-1, df.columns.get_loc('target')] = np.nan  # last row has no next-day close, don't fabricate

    return df
