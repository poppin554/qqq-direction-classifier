"""
Loads the raw CSV, builds the target and the four features, and drops rows
with NaN (the last row, from the target's shift(-1), and the first `window`
rows, from the rolling features' shift(1)+rolling window warmup).
"""
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV_PATH
from src.features.build_target import build_target
from src.features.build_features import build_features


def prepare_dataset(csv_path=RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    df = build_target(df)
    df = build_features(df)

    df = df.dropna()
    return df


if __name__ == "__main__":
    df = prepare_dataset()
    print(f"Final row count: {len(df)}")
    print(f"Target class balance:\n{df['target'].value_counts()}")
