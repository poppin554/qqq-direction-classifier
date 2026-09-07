"""
Logistic regression, default hyperparameters, StandardScaler inside the
Pipeline so scaling is fit on train only (never sees test data).
See decision_log.md, Phase 7.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import FEATURE_COLS


def train_logistic_regression(train_df, test_df, feature_cols=FEATURE_COLS):
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression()),
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


if __name__ == "__main__":
    from src.features.prepare_dataset import prepare_dataset
    from src.data.split import split_train_test

    df = prepare_dataset()
    train_df, test_df = split_train_test(df)

    result = train_logistic_regression(train_df, test_df)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Confusion matrix:\n{result['confusion_matrix']}")
