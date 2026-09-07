"""
SCRATCH FILE, not part of the project pipeline.
Original ad-hoc check for exact Close-price ties (used to produce the tie
count cited in decision_log.md, Phase 3 evidence: 33/6900 rows ~0.5%).
Requires two dated raw CSV snapshots that may not exist in data/raw/ anymore.
Kept for reference; safe to delete once the finding is trusted from the log.
"""
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR

df = pd.read_csv(RAW_DATA_DIR / "qqq_1999-03-10_to_2026-08-14.csv")

same_close = df['Close'].shift(-1) == df['Close']
print(df[same_close])
print(len(same_close))
