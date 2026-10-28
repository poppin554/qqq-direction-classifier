# QQQ Next-Day Direction Classifier
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

A leakage-safe test of whether public technical indicators can predict next-day stock direction, and an honest accounting of what happens when they don't.

## Motivation

Weak-form market efficiency is the idea that past prices and volume are already reflected in current prices, so they can't be used to reliably beat the market. It's a common claim, but individuals rarely test it rigorously. Most self-directed attempts at "can I predict the market" skip a real baseline comparison, leak future information into their features without realizing it, or compute evaluation metrics incorrectly across train/test splits.

This project is a concrete, falsifiable test of that claim on one ticker, QQQ, using only OHLCV-derived technical features (no options data, no fundamentals). The question: **can a leakage-safe classifier meaningfully beat a naive majority-class baseline on next-day direction?**

A negative result is treated as a valid finding, not a failed project. If a small set of public technical indicators reliably beat the market, that would itself be surprising, and worth doubting before celebrating.

## Data

- **Ticker:** QQQ (Invesco QQQ Trust, Nasdaq-100 ETF)
- **Source:** Yahoo Finance via `yfinance`, `auto_adjust=True` (adjusted close, handles splits and dividends)
- **Range:** Full available history, ~1999 to 2026 (~6,900 trading days)
- **Columns stored:** Open, High, Low, Close, Volume

## Target

Binary classification: `1` if the next day's close is higher than today's close, else `0` (flat days count as `0`). The final row of the dataset is dropped, since it has no "next day" to compare against.

Class balance on the full dataset: ~54.5% up / ~45.5% down.

## Features

Four technical features, all derived from OHLCV data and all lagged by one day (`.shift(1)`), so today's row never uses today's own close, only information available before it.

| Feature | Description |
|---|---|
| `close_ma_5` | 5-day rolling average of Close |
| `intraday_move` | Yesterday's Close minus Open |
| `lagged_return` | Yesterday's daily return |
| `rolling_volatility` | 5-day rolling standard deviation of daily returns |

Note: `close_ma_5` and `intraday_move` are in price units, not scale-free. See [Known limitations](#known-limitations).

## Methodology

- **Split:** Chronological, not random. Train = everything before 2018-01-01 (4,729 rows); test = 2018-01-01 onward (2,173 rows). Boolean date masking is used instead of positional (`.iloc[]`) slicing, because row positions shift whenever upstream rows are dropped (e.g. NaN rows from feature engineering), while date-based filtering doesn't.
- **Baseline:** Majority-class accuracy, computed separately for train and test rather than pooled. Markets aren't stationary and the up/down base rate differs across periods (train: 53.73%, test: 56.09%), so a pooled number would misstate both.
- **Leakage discipline:**
  - All features are shifted before use.
  - The target is computed once on the full dataset before splitting, not recomputed inside subsets. Recomputing inside a subset corrupts the last row, since there is no "next day" within that subset even though real future data exists in the other split.
  - Scaling (`StandardScaler`) is fit only on training data, inside an sklearn `Pipeline`, so it never sees test data.

## Results

### Baseline

| Split | Majority class | Accuracy |
|---|---|---|
| Train | 1 (up) | 53.73% |
| Test | 1 (up) | 56.09% |

With about 2,170 test rows, the standard error of an accuracy estimate is roughly 1 percentage point, so differences of a few tenths of a point below are within noise.

### Logistic Regression

Default hyperparameters, features scaled via `StandardScaler` inside a `Pipeline`.

| Metric | Value |
|---|---|
| Test accuracy | 55.96% |
| vs. test baseline | Below baseline (56.09%) |
| Confusion matrix | TN=5, FP=949, FN=8, TP=1211 |
| Recall (class 0) | ~0.5% |

The model almost never predicts "down." Predicted probabilities were varied and non-degenerate (so the pipeline is not broken), but consistently favored class 1 at every confidence level. The model found no useful decision boundary and effectively collapsed to the majority-class strategy.

**Hyperparameter tuning:**

- `class_weight='balanced'` raised class-0 recall marginally (0.5% to 2.4%) but lowered accuracy to 55.68%.
- `C=5.0` and `C=np.inf` produced results identical to the untuned model.

Neither calibration nor regularization appears to be the problem, which is consistent with a lack of separating signal in these features.

### Decision Tree

Same features, same split. `StandardScaler` dropped (not needed for threshold-based splits). `max_depth` swept from 1 to 15 to separate genuine signal from overfitting.

| max_depth | Train accuracy | Test accuracy |
|---|---|---|
| 1 | 54.47% | 54.99% |
| 5 | 57.03% | 55.78% |
| 10 | 62.66% | 53.66% |
| 15 | 71.41% | 52.14% |

Train accuracy climbs steadily with depth (54.5% to 71.4%), which is classic overfitting: the tree memorizes training noise. Test accuracy is highest at depth 5, then declines. No depth beats the 56.09% baseline. Depth 5 was identified by looking at test accuracy, so 55.78% is an optimistic figure; the v2 plan uses a validation split instead.

## Conclusion

A linear model (logistic regression) and a non-linear-capable model (decision tree) were tested on the same features against the same baseline. Neither beat it. Tuning found no evidence that calibration or regularization was the cause for logistic regression, and the depth sweep found no evidence that the tree simply needed to be pushed toward non-linear patterns.

The convergent result is consistent with the limitation lying in the **feature set** rather than the model class: these four basic OHLCV-derived indicators show no exploitable signal for next-day QQQ direction. Two caveats weaken that reading: two of the four features are not scale-free, and no significance test was run (see below).

This does **not** show that QQQ's direction is unpredictable, or that no technical signal could work. It shows that this specific, easily replicated feature set did not clear the bar, which is consistent with weak-form efficiency and with why a persistent edge is hard to find in public daily data.

## What this project is (and isn't)

- **Is:** a leakage-safe test of a specific, falsifiable hypothesis, with an honest negative result.
- **Is:** a demonstration of train/test discipline, baseline methodology, and diagnosing model behavior (overfitting, feature scale, class imbalance) instead of reporting a single accuracy number.
- **Isn't:** a trading strategy. Predicting direction is not predicting magnitude, and even a real directional edge would need to survive transaction costs, slippage, and position sizing.
- **Isn't:** a claim that markets are unpredictable in general.

## Known limitations

- **Non-stationary features:** `close_ma_5` and `intraday_move` are in price units. QQQ rose from roughly $20 to $500+, so thresholds learned on 1999-2017 do not transfer to 2018+. The null result may partly reflect broken features rather than uninformative ones.
- **Model selection on test data:** `max_depth` was chosen by looking at test accuracy, which contradicts the leakage discipline above.
- **No significance testing:** Accuracy gaps are compared by eye. They are within sampling noise.
- **Accuracy only:** Accuracy rewards always predicting "up." AUC and log loss are not yet reported.
- **No strategy evaluation:** No backtest, costs, drawdown, or Sharpe.

## Roadmap (v2)

1. Scale-free features (e.g. `Close / MA5 - 1`, intraday move as a % of Open).
2. Walk-forward evaluation with a validation split; report the test result once.
3. AUC and log loss, plus a binomial test against the baseline.
4. A costed strategy backtest with drawdown and Sharpe.
5. The same pipeline on a second asset (e.g. a crypto series).

Ensemble methods are lower priority: they share the same hypothesis space as the single tree and are unlikely to find signal the depth sweep did not hint at.

## Tech stack

Python, pandas, numpy, scikit-learn, yfinance, pandas-market-calendars

## Decision log

Every methodological choice, including ones that were initially wrong and corrected (e.g. a NaN-to-int casting bug, a train/test date-masking bug), is documented in [`decision_log.md`](decision_log.md) with alternatives considered, reasoning, evidence, and what would invalidate it. Open issues are listed at the end of that file.

## How to run

```bash
git clone <repo-url>
cd qqq-direction-classifier
python -m venv venv
venv\Scripts\activate      # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python run_pipeline.py
```
