"""
Decision tree, no StandardScaler (trees split on thresholds per feature,
not sensitive to scale, see decision_log.md, Phase 8).

max_depth is swept from 1-15 with a FIXED random_state, so depth is the only
variable changing across iterations, this isolates the depth-vs-accuracy
trend from noise (a bug in an earlier version of this script accidentally
held max_depth fixed while varying random_state, producing misleading,
falsely-high, flat accuracy across all "depths").
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import FEATURE_COLS

RANDOM_STATE = 42  # fixed across the whole sweep, only max_depth varies


def sweep_max_depth(train_df, test_df, depths=range(1, 16), feature_cols=FEATURE_COLS) -> pd.DataFrame:
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    results = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results.append({"max_depth": depth, "train_accuracy": train_acc, "test_accuracy": test_acc})

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.features.prepare_dataset import prepare_dataset
    from src.data.split import split_train_test

    df = prepare_dataset()
    train_df, test_df = split_train_test(df)

    sweep_results = sweep_max_depth(train_df, test_df)
    print(sweep_results.to_string(index=False))

    best_row = sweep_results.loc[sweep_results['test_accuracy'].idxmax()]
    print(f"\nBest test accuracy: {best_row['test_accuracy']:.4f} at max_depth={int(best_row['max_depth'])}")
