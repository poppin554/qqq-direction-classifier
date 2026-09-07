"""
Majority-class baseline accuracy. Computed separately for train and test
(never pooled), see decision_log.md, Phase 6, for why: different market
periods have different up/down base rates, a pooled number misrepresents both.
"""
import pandas as pd


def majority_class_baseline(y: pd.Series) -> dict:
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    accuracy = class_counts.max() / len(y)

    return {
        "majority_class": majority_class,
        "accuracy": accuracy,
        "class_counts": class_counts.to_dict(),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.features.prepare_dataset import prepare_dataset
    from src.data.split import split_train_test

    df = prepare_dataset()
    train_df, test_df = split_train_test(df)

    train_baseline = majority_class_baseline(train_df['target'])
    test_baseline = majority_class_baseline(test_df['target'])

    print(f"Train baseline: {train_baseline}")
    print(f"Test baseline: {test_baseline}")
