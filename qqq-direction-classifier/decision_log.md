# QQQ Direction Classifier — Decision Log

## Phase 1 — Project Setup

| # | Decision | Alternatives considered | Why chosen | Evidence | What could invalidate it |
|---|----------|--------------------------|------------|----------|----------------------------|
| 1 | Ticker: QQQ | VT, SOXX | VT's low volatility risked too little signal to detect; SOXX had insufficient history; QQQ balances signal-to-noise and long history | Reasoned through volatility/signal tradeoff | If model shows QQQ's volatility itself dominates results in a way that doesn't generalize to broad-market ETFs |
| 2 | Task: Classification (next-day direction) | Regression (predict return magnitude) | Daily return regression typically has near-zero R² on liquid ETFs; classification gives richer, calibratable outputs (probabilities) even with a modest edge | Reasoned through what a weak regression R² vs. a calibrated classifier communicates | If literature/EDA shows magnitude prediction is more tractable than assumed |
| 3 | Price basis: Adjusted close (`auto_adjust=True`) | Unadjusted close | Unadjusted close creates false "down" labels on ex-dividend dates — an artifact, not real market movement | Verified empirically that `auto_adjust=True` overwrites Close with adjusted values, no separate Adj Close column | N/A — this is a correctness fix, not a judgment call |
| 4 | Historical window: full available history (`period="max"`, ~1999–present) | Short/recent window (e.g., post-2020) | Long window spans multiple regimes (crashes, bulls, bears), enabling generalization testing; cherry-picking a calm window would bias results and remove the regime analysis planned for Phase 8 | Reasoned through selection-bias risk of narrow windows | If regime shifts are so extreme that pre-2010 data is judged non-representative of current market microstructure |
| 5 | Raw columns stored: Open, High, Low, Close, Volume | Date/Open/Close only; full yfinance output incl. Dividends/Splits/Capital Gains | OHLCV costs nothing extra to store (same API call) and preserves optionality for later features (volume, range-based volatility); Dividends/Splits/Capital Gains dropped as not currently justified | N/A | If a future feature genuinely needs dividend/split data, this would need revisiting |
| 6 | Date format in raw CSV: ISO 8601 (YYYY-MM-DD) | DD-MM-YYYY (labeled) | ISO avoids parser ambiguity risk (`pd.read_csv(parse_dates=...)` auto-inference), is unambiguous without needing a header label, matches filename convention already in use | Reasoned through pandas date-parsing failure modes | Unlikely to be invalidated — this is close to a strict best practice |
| 7 | Storage format: CSV | Parquet | Dataset is small (~6,900 rows, single ticker); CSV's simplicity outweighs Parquet's advantages at this scale | N/A | — |
| 8 | Treat split-adjustment correctness as a documented limitation, not a verifiable check | Adj Close continuity check (rejected by construction); volume-doubling check (rejected — conflates with outstanding shares traded) | No independently-sourced, genuinely pre-adjustment historical price series for QQQ was found. Confidence rests on yfinance's source code (confirms `auto_adjust` computes a Close/Adj-Close ratio transformation) and community consensus (GitHub issues, Stack Overflow) that Yahoo's backend bakes in split adjustments before yfinance receives the data — not on first-party empirical verification | `.splits` confirms one 2:1 QQQ split on 2000-03-20; yfinance source code for `auto_adjust`/`back_adjust`; community discussion of the parameter's behavior | A genuinely independent data vendor (different upstream from Yahoo) providing a pre-split QQQ price for March 2000 that contradicts the current figures |

## Phase 2 — Data Integrity Checks

| Field | Decision A: Inclusive vs. Exclusive Bounds | Decision B: Tolerance vs. Rounding for Float Precision |
|---|---|---|
| **Decision** | Use inclusive comparisons (`>=`, `<=`) for High ≥ Low, and Open/Close-within-range checks | Use `np.isclose()` tolerance in range checks instead of rounding raw data |
| **Alternatives** | Exclusive comparisons (`>`, `<`); assuming yfinance data has no edge cases | Round raw price data to fewer decimal places; leave strict equality as-is (do nothing) |
| **Why** | High == Low is a valid, real trading day (flat/low-volatility session), not an error — exclusive comparison would falsely flag valid data | Rounding discards real source precision just to satisfy a check — the check should adapt to the data, not the other way around |
| **Evidence** | N/A (design decision, no violation triggered this one directly) | Row 3283 (2012-03-26): Close = 60.26075744628906 vs High = 60.260757446289055, diff = 7.105427357601002e-15 — pure floating-point noise from `auto_adjust=True` |
| **What could invalidate it** | If a future check needs to treat boundary-touching prices as suspicious for a different reason (unlikely for this project's scope) | The tolerance is a judgment call: `np.isclose` defaults (`rtol=1e-5`, `atol=1e-8`) could be too loose for higher-priced stocks — e.g. at $1000/share, up to ~$0.01 of real discrepancy could silently pass as "noise" and go undetected |

## Phase 3 — Target Variable Construction

| Field | Detail |
|---|---|
| **Decision** | Binary target = 1 if Close[t+1] > Close[t], else 0 (down and flat both map to 0). Last row (no next-day close available) is dropped, not fabricated. |
| **Alternatives considered** | Three-way label (up/down/flat); defaulting the last row to 0 instead of dropping it |
| **Why** | Three-way label adds unneeded complexity for a binary classification scope. Fabricating a label for the last row would insert a false answer for an outcome that hasn't happened yet. |
| **Evidence** | Tie check found 33/6900 rows (~0.5%) with exact Close equality; volume on those rows checked normal, ruling out a data artifact. Post-fix: row count = 6900, class balance ≈ 54.5% up / 45.5% down, manual spot-checks confirmed correct. |
| **What could invalidate this** | If the 33 tie rows are later found to be a data quality issue rather than genuine flat days. Also: operation order matters — boolean comparison → drop NaN row → cast to int. Casting to int first would have silently turned the missing label into a fabricated 0. |

## Phase 4 — Feature Engineering

### Entry 1 — Feature shift discipline

| Field | Detail |
|---|---|
| **Decision** | Every engineered feature (`close_ma_5`, `intraday_move`, `lagged_return`, `rolling_volatility`) is shifted with `.shift(1)` so today's row only reflects data through yesterday's close |
| **Alternatives considered** | Leaving features unshifted (using today's own Close/Open directly) |
| **Why** | The target uses Close[t] directly. Any feature also using Close[t] shares that ingredient with the label, entangling feature and answer instead of describing genuine past history |
| **Evidence** | Applied across all 4 features; verified via manual row-by-row cross-checks (e.g., `close_ma_5` row 8 vs. `Close.iloc[3:8]`; `rolling_volatility` row 6 vs. `rv_check` rows 1–5) |
| **What could invalidate this** | Any future feature added without reapplying this same shift discipline — not automatic, must be deliberately repeated each time |

### Entry 2 — Double return-calculation bug (`lagged_return`)

| Field | Detail |
|---|---|
| **Decision** | Use the literal return formula `(Close - Close.shift(1)) / Close.shift(1)` rather than `.pct_change()`, then apply one final `.shift(1)` |
| **Alternatives considered** | Using pandas' built-in `.pct_change()` |
| **Why** | Literal formula is easier to debug/trace errors in than a black-box method call |
| **Evidence** | Initial draft mistakenly stacked `.pct_change()` on top of an already-computed return formula, double-applying return logic; caught and corrected before finalizing |
| **What could invalidate this** | If a future feature reuses this same formula pattern, the same double-application mistake could recur without a rerun of manual verification |

### Entry 3 — Missing final shift (`rolling_volatility`)

| Field | Detail |
|---|---|
| **Decision** | Every rolling/lag feature must end with an explicit final `.shift(1)`, even if the calculation already contains an internal shift |
| **Alternatives considered** | Relying on the internal shift already present in the return formula, assuming it was "already safe" |
| **Why** | Internal shifts only protect the raw calculation itself, not the positioning of the final result relative to today's row |
| **Evidence** | Initial `rolling_volatility` was missing the final shift — first valid value appeared at row 5 instead of expected row 6, caught via cross-check against `rv_check` |
| **What could invalidate this** | Any new rolling/derived feature added later without repeating this same "does the result need its own shift" check |

## Phase 5 — Train/Test Split

| Field | Detail |
|---|---|
| **Decision** | Split `train_df`/`test_df` using boolean date masking (`cutoff = pd.Timestamp('2018-01-01')`, then `df[df['Date'] < cutoff]` / `df[df['Date'] >= cutoff]`) rather than positional index slicing |
| **Alternatives** | `.iloc[]` split using a manually located row position for the 2017/2018 boundary |
| **Why** | `.iloc[]` needs a manually-found row position, and that position shifts any time upstream rows are dropped (e.g. reruns with different NaN counts from feature engineering), making it fragile and non-reproducible. Boolean date masking filters on actual `Date` values, which don't shift regardless of dropped rows elsewhere |
| **Evidence** | `len(train_df)` = 4729, `len(test_df)` = 2173. Last train date = 2017-12-29. First test date = 2018-01-02. Boundary confirmed clean, no overlap |
| **What could invalidate this** | If date format is a string, boolean comparison will not work correctly, since `pd.Timestamp` comparisons require datetime dtype to validate properly. Also, a corrupted or missing date (NaT) will not crash the split, it will silently be excluded from both train and test, since `NaT < cutoff` and `NaT >= cutoff` both evaluate False. This isn't currently caught end-to-end, the NYSE calendar check runs independently on raw data and isn't reapplied after downstream transformations |

## Phase 7 — Logistic Regression (Pipeline)

| Field | Detail |
|---|---|
| **Decision** | Test whether a logistic regression model, using the four leakage-safe features (`close_ma_5`, `intraday_move`, `lagged_return`, `rolling_volatility`) inside an sklearn Pipeline (StandardScaler + LogisticRegression), can meaningfully beat the test-period majority-class baseline (56.09%) on QQQ next-day direction |
| **Alternatives** | Not yet tried: other model classes (e.g. tree-based models, which don't require scaling and can capture non-linear interactions), additional or different OHLCV-derived features beyond the current four, threshold tuning instead of the default 0.5 cutoff |
| **Why** | Logistic regression was chosen as the first model because it's simple, interpretable via coefficients, and establishes a baseline-comparable result before adding model complexity. StandardScaler was required inside the Pipeline specifically because logistic regression's gradient-based optimization is sensitive to feature scale, and features here have very different natural ranges (raw price differences vs. percentages) |
| **Evidence** | Accuracy = 55.96% vs test baseline 56.09%, model did not beat baseline. Confusion matrix: TN=5, FP=949, FN=8, TP=1211. Recall for class 0 ≈ 0.5% (5/954), indicating the model essentially never predicts "down" correctly. Predicted probabilities (via `predict_proba`) were varied (not flat/degenerate), ruling out a broken pipeline, model consistently favored class 1 across all confidence levels, meaning it found no real decision boundary separating the classes and effectively degenerated into predicting the majority class |
| **What could invalidate this** | This conclusion is scoped to one model (logistic regression) and one feature set (four basic OHLCV-derived technical indicators). A different model class (e.g. tree-based, capturing non-linear feature interactions) or additional/different OHLCV-derived features could still find separation these four features and this linear model did not. This result does not prove QQQ direction is unpredictable in general, only that this specific setup found no exploitable signal |

## Phase 7b — Hyperparameter Tuning (Logistic Regression)

| Field | Detail |
|---|---|
| **Decision** | Tune two hyperparameters on the Phase 7 logistic regression pipeline: `class_weight='balanced'` (targets class imbalance calibration) and `C` (regularization strength, tested at `C=5.0` and `C=np.inf`) |
| **Alternatives** | `solver` — not tuned, changes only the optimization algorithm, which converges to the same solution for a simple, well-scaled, 4-feature problem. `l1_ratio` — not tuned, only relevant under `penalty='elasticnet'` for mixing L1/L2 penalties, and since `C` itself had no effect, adjusting how the penalty is weighted was pointless. `random_state` — not tuned, only affects reproducibility of solver initialization, not model capacity |
| **Why** | Tuning further would chase marginal gains without altering the overall verdict (model does not beat the 56.09% test baseline). The two parameters tested target calibration (`class_weight`) and overfitting/large coefficients (`C`), neither of which is the root cause. The model's failure comes from an inability to find separating signal in the four features provided, not from miscalibration or overfitting, so no amount of reweighting or regularization strength adjustment can create signal that isn't present in the input features |
| **Evidence** | `class_weight='balanced'`: accuracy dropped to 55.68% (from 55.96%), confusion matrix TN=23, FP=931, FN=32, TP=1187 — class 0 recall improved marginally (0.5% → 2.4%) but at a net accuracy cost, since the majority class is larger, the trade of false negatives for fewer false positives costs more than it gains. `C=5.0` and `C=np.inf`: both produced identical results to the untuned model (55.96% accuracy, same confusion matrix as Phase 7), confirming regularization strength had zero effect |
| **What could invalidate this** | If a tree-based model finds meaningful separation using these same four features, it wouldn't invalidate the finding that logistic regression found no *linear* separation, that result stands either way. It would instead show that the signal may exist but be non-linear, which a linear model like logistic regression is structurally incapable of capturing regardless of tuning |

## Phase 8 — Tree-Based Model Comparison (Decision Tree)

| Field | Detail |
|---|---|
| **Decision** | Test whether a `DecisionTreeClassifier`, using the same four features and same train/test split, can find non-linear separation between classes that logistic regression (linear) could not. Swept `max_depth` from 1 to 15 (fixed `random_state=42`) to compare train vs. test accuracy at each depth, rather than testing a single unconstrained tree |
| **Alternatives** | Random forest — not started here, considered but a single decision tree was chosen first: performs reasonably on both small and large datasets, produces an interpretable model, and random forest's main advantage (variance reduction across many trees) is a response to overfitting risk that this depth-sweep already investigates directly for a single tree. An unconstrained single tree (no `max_depth` limit) was also considered and rejected as a first test, since it was predicted in advance to simply memorize training noise (~100% train accuracy) without answering whether real non-linear signal exists |
| **Why** | StandardScaler was dropped from this pipeline (unlike Phase 7), since decision trees split on threshold values per feature independently and are not sensitive to feature scale, unlike gradient-based linear models. `max_depth` was swept rather than fixed at one value because an unconstrained tree was predicted (and later confirmed) to overfit, a fixed depth wouldn't reveal where the train/test accuracy gap actually starts, which is the diagnostic needed to separate "the tree overfit" from "the tree found no real signal" |
| **Evidence** | Bug caught and fixed mid-sweep: initial loop mistakenly held `max_depth` fixed at 15 via `max(depths)` while varying `random_state`, producing misleadingly high, roughly flat train accuracy (~71%) across all "depths." After fixing (loop `depth` in `max_depth`, fixed `random_state=42`), true sweep showed: train accuracy climbs steadily from 54.47% (depth=1) to 71.41% (depth=15). Test accuracy peaks at depth=5 (55.78%), then declines steadily to 52.14% by depth=15, both below train accuracy and below the test baseline of 56.09% at every depth. Best test-depth result (55.78% at depth=5) does not beat baseline |
| **What could invalidate this** | This tests one tree-based model (single decision tree) on the same four OHLCV-derived features already ruled out by Phase 7. It does not prove no non-linear signal exists in the underlying market relationship, only that these four specific features contain no non-linear separation exploitable by a single tree either. An ensemble method (random forest, gradient boosting) or a different/expanded feature set could still find something this setup did not |

## Phase 6 — Majority-Class Baseline

| Field | Detail |
|---|---|
| **Decision** | Compute majority-class baseline from the existing `target` column, computed separately for train and test rather than pooled |
| **Alternatives** | Considered recomputing `target` inside each subset — rejected. Considered a single pooled baseline across the full dataset — rejected |
| **Why** | Recomputing `target` inside `train_df` via `.shift(-1)` corrupts the last row to a false 0 (NaN → int cast bug) even though real next-day data exists in `test_df`; using the existing full-dataset `target` avoids this. Separate baselines are needed because different market periods have different up/down base rates — a pooled number misrepresents both |
| **Evidence** | Train: 2541/4729 = 53.73% majority (class 1). Test: 1219/2173 = 56.09% majority (class 1) |
| **What could invalidate** | New data added to the dataset changes test composition and reopens this number. Changing the train/test cutoff date redraws the split and can shift both baselines |
