# QQQ Next-Day Direction Classifier

A leakage-safe test of whether public technical indicators can predict next-day stock direction, and an honest accounting of what happens when they don't.

## Motivation

Semi-strong market efficiency, the idea that publicly available information (like daily price and volume history) is already priced in and can't be used to reliably beat the market, is a common claim. It's rarely tested rigorously by individuals, though. Most self-directed attempts at "can I predict the market" skip a real baseline comparison, leak future information into their features without realizing it, or compute their evaluation metrics incorrectly across train/test splits.

This project is a concrete, falsifiable test of that claim on one ticker, QQQ, using only OHLCV-derived technical features, no options data, no fundamentals. The question: **can a leakage-safe classifier meaningfully beat a naive majority-class baseline on next-day direction?**

A negative result is treated as a valid, credible finding here, not a failed project. If a small set of public technical indicators reliably beat the market, that would itself be surprising, and worth doubting before celebrating.

## Data

- **Ticker:** QQQ (Invesco QQQ Trust, Nasdaq-100 ETF)
- **Source:** Yahoo Finance via `yfinance`, `auto_adjust=True` (adjusted close, handles splits/dividends)
- **Range:** Full available history, ~1999 to 2026 (~6,900 trading days)
- **Columns stored:** Open, High, Low, Close, Volume

## Target

Binary classification: `1` if next day's close is higher than today's close, else `0` (flat days count as `0`). The final row of the full dataset is dropped, since it has no "next day" to compare against.

Class balance on the full dataset: ~54.5% up / ~45.5% down.

## Features

Four features, all technical, all derived purely from OHLCV data, and all deliberately lagged by one day (`.shift(1)`) so that today's row never uses today's own closing price, only information available *before* today's close:

| Feature | Description |
|---|---|
| `close_ma_5` | 5-day rolling average of Close |
| `intraday_move` | Yesterday's Close minus Open |
| `lagged_return` | Yesterday's daily return |
| `rolling_volatility` | 5-day rolling standard deviation of daily returns |

## Methodology

- **Split:** Chronological, not random. Train = everything before 2018-01-01 (4,729 rows), test = everything from 2018-01-01 onward (2,173 rows). Boolean date masking was used instead of positional (`.iloc[]`) slicing, since row positions shift whenever upstream rows are dropped (e.g. NaN rows from feature engineering), while date-based filtering doesn't.
- **Baseline:** Majority-class accuracy, computed *separately* for train and test, not pooled. Markets aren't stationary, the up/down base rate differs across time periods (train baseline: 53.73%, test baseline: 56.09%), so a single pooled number wouldn't reflect either period's actual base rate.
- **Leakage discipline:** All features shifted before use. Target computed once on the full dataset before splitting, not recomputed inside subsets (recomputing inside a subset corrupts the last row, since there's no "next day" *within* that subset, even though real future data exists in the other split). Any scaling (`StandardScaler`) is fit only on training data, inside an sklearn `Pipeline`, so it never sees test data.

## Results

### Baseline

| Split | Majority class | Accuracy |
|---|---|---|
| Train | 1 (up) | 53.73% |
| Test | 1 (up) | 56.09% |

### Logistic Regression

Default hyperparameters, features scaled via `StandardScaler` inside a `Pipeline`.

| Metric | Value |
|---|---|
| Test accuracy | 55.96% |
| vs. test baseline | **Below baseline** (56.09%) |
| Confusion matrix | TN=5, FP=949, FN=8, TP=1211 |
| Recall (class 0) | ~0.5% |

The model essentially never predicts "down." Predicted probabilities were varied and non-degenerate (ruling out a broken pipeline), but consistently favored class 1 across all confidence levels, the model found no real decision boundary and effectively collapsed to the majority-class strategy.

**Hyperparameter tuning:** `class_weight='balanced'` was tested to directly penalize class-0 errors more heavily. Result: recall for class 0 improved marginally (0.5% → 2.4%), but accuracy dropped further (55.68%), since the majority class is larger, trading false positives for false negatives costs more than it gains. Regularization strength (`C`) was tested at `C=5.0` and `C=np.inf`, both produced identical results to the untuned model. Neither calibration nor overfitting was the actual issue, consistent with a lack of separating signal in the features themselves, not a tuning problem.

### Decision Tree

Same features, same split, `StandardScaler` dropped (not needed for threshold-based splits). `max_depth` swept from 1 to 15 to distinguish genuine signal from overfitting.

| max_depth | Train accuracy | Test accuracy |
|---|---|---|
| 1 | 54.47% | 54.99% |
| 5 | 57.03% | **55.78%** (best on test) |
| 10 | 62.66% | 53.66% |
| 15 | 71.41% | 52.14% |

Train accuracy climbs steadily with depth (54.5% → 71.4%), classic overfitting, the tree memorizes training-specific noise as depth increases. Test accuracy peaks early (depth 5) then declines, never beating the 56.09% baseline at any depth.

## Conclusion

Two structurally different model classes, one linear (logistic regression), one non-linear-capable (decision tree), were tested against the same features and the same baseline. Neither beat it. Hyperparameter tuning ruled out calibration and overfitting as fixable causes for logistic regression; the depth sweep ruled out "the tree just needs to be told to look for non-linear patterns" as an explanation for the decision tree.

The convergent result across both model families points to the limitation being in the **feature set**, not the model class: these four basic OHLCV-derived technical indicators do not appear to contain exploitable signal for next-day direction on QQQ, at least not one either a linear or non-linear model applied to them could find.

This does **not** prove that QQQ's direction is fundamentally unpredictable, or that no technical signal could ever work. It shows that this specific, common, easily-replicable feature set doesn't clear the bar, a result broadly consistent with what would be expected under semi-strong market efficiency, and consistent with why this space is hard to find a genuine, persistent edge in using only public daily data.

## What this project is (and isn't)

- **Is:** a rigorous, leakage-safe test of a specific, falsifiable hypothesis, with an honest negative result.
- **Is:** a demonstration of correct train/test discipline, baseline methodology, and diagnosing model behavior (overfitting, feature-scale sensitivity, class imbalance) rather than reporting a single accuracy number in isolation.
- **Isn't:** a trading strategy. Predicting *direction* is not the same as predicting *magnitude*, and even a genuine directional edge wouldn't automatically translate into a profitable strategy once transaction costs, slippage, and position sizing are considered.
- **Isn't:** a claim that markets are unpredictable in general, only that this specific setup found nothing.

## Possible next steps

- Different or expanded feature sets (volume-based features, longer/shorter rolling windows, cross-asset signals)
- Ensemble methods (random forest, gradient boosting), lower priority, since they share the same hypothesis space as the single decision tree already tested here and are unlikely to find signal the depth sweep didn't hint at
- Statistical significance testing on any future edge found (is it real or noise), risk-adjusted metrics, and market microstructure awareness, prerequisites for a more quant-oriented follow-up, deliberately out of scope here

## Tech stack

Python, pandas, numpy, scikit-learn, yfinance, pandas-market-calendars

## Project structure / decision log

Every methodological choice in this project, including ones that were initially wrong and had to be corrected (e.g. a NaN-to-int casting bug, a train/test date-masking bug), is documented with alternatives considered, reasoning, evidence, and what would invalidate it, in `decision_log.md`.
