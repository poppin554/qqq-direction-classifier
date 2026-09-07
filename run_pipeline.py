"""
Runs the full pipeline end to end: load -> prepare features/target -> split
-> baseline -> logistic regression -> tuning -> decision tree.

Run from the project root: python run_pipeline.py
(Assumes data/raw/ already contains the CSV, run `python -m src.data.load`
first if not.)
"""
import numpy as np

from src.features.prepare_dataset import prepare_dataset
from src.data.split import split_train_test
from src.evaluation.baseline import majority_class_baseline
from src.models.logistic_regression import train_logistic_regression
from src.models.logistic_regression_tuning import train_with_class_weight, train_with_C
from src.models.decision_tree import sweep_max_depth


def main():
    df = prepare_dataset()
    train_df, test_df = split_train_test(df)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}\n")

    # Phase 6: baseline
    train_baseline = majority_class_baseline(train_df['target'])
    test_baseline = majority_class_baseline(test_df['target'])
    print(f"Train baseline: {train_baseline['accuracy']:.4f} (class {train_baseline['majority_class']})")
    print(f"Test baseline: {test_baseline['accuracy']:.4f} (class {test_baseline['majority_class']})\n")

    # Phase 7: logistic regression
    logreg_result = train_logistic_regression(train_df, test_df)
    print(f"Logistic regression accuracy: {logreg_result['accuracy']:.4f}")
    print(f"Confusion matrix:\n{logreg_result['confusion_matrix']}\n")

    # Phase 7b: tuning
    balanced_result = train_with_class_weight(train_df, test_df, class_weight='balanced')
    print(f"class_weight='balanced' accuracy: {balanced_result['accuracy']:.4f}")
    for c_value in [5.0, np.inf]:
        c_result = train_with_C(train_df, test_df, C=c_value)
        print(f"C={c_value} accuracy: {c_result['accuracy']:.4f}")
    print()

    # Phase 8: decision tree depth sweep
    sweep_results = sweep_max_depth(train_df, test_df)
    print(sweep_results.to_string(index=False))
    best_row = sweep_results.loc[sweep_results['test_accuracy'].idxmax()]
    print(f"\nBest tree test accuracy: {best_row['test_accuracy']:.4f} at max_depth={int(best_row['max_depth'])}")


if __name__ == "__main__":
    main()
