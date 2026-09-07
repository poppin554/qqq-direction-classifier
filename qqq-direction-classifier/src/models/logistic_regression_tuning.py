"""
Hyperparameter tuning on logistic regression: class_weight='balanced' and
regularization strength (C). See decision_log.md, Phase 7b.

Neither meaningfully changed the result, class_weight='balanced' improved
class-0 recall marginally (0.5% -> 2.4%) at a net accuracy cost, and C had
zero measurable effect at either extreme (C=5.0, C=np.inf). This supports
the conclusion that the limitation is a lack of separating signal in the
features, not miscalibration or overfitting.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import FEATURE_COLS


def train_with_class_weight(train_df, test_df, class_weight='balanced', feature_cols=FEATURE_COLS):
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight=class_weight)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def train_with_C(train_df, test_df, C=1.0, feature_cols=FEATURE_COLS):
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=C)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


if __name__ == "__main__":
    from src.features.prepare_dataset import prepare_dataset
    from src.data.split import split_train_test

    df = prepare_dataset()
    train_df, test_df = split_train_test(df)

    balanced_result = train_with_class_weight(train_df, test_df, class_weight='balanced')
    print(f"class_weight='balanced': accuracy={balanced_result['accuracy']:.4f}")
    print(balanced_result['confusion_matrix'])

    for c_value in [5.0, np.inf]:
        c_result = train_with_C(train_df, test_df, C=c_value)
        print(f"C={c_value}: accuracy={c_result['accuracy']:.4f}")
