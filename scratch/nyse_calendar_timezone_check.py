"""
SCRATCH FILE, not part of the project pipeline.
Exploratory check of pandas_market_calendars' schedule() output (timezone
behavior). Superseded by src/data/market_calendar.py. Kept for reference;
safe to delete.
"""
import pandas_market_calendars as mcal

nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2020-01-01', end_date='2020-01-10')
print(schedule.index)
print(schedule.index.tz)
