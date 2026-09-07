"""
SCRATCH FILE, not part of the project pipeline.
Confirms QQQ's dividend and split history via yfinance (feeds decision_log.md,
Phase 1 #8: confirms one 2:1 QQQ split on 2000-03-20). Renamed from the
original "tester_divident.py" (typo) to "dividend_and_split_check.py".
"""
import yfinance as yf

qqq = yf.Ticker("QQQ")
dividends = qqq.dividends
splits = qqq.splits

print(dividends)
print(splits)
