"""
Checks the raw dataset against the NYSE trading calendar for missing trading days.
Two checks: (1) dates present in our data that NYSE doesn't recognize as trading days,
(2) NYSE trading days whose "next trading day" is missing from our data.

NOTE: this check runs independently on the raw CSV. It is NOT re-run after
downstream transformations (dropna, feature engineering, split). See
decision_log.md, Phase 5, "what could invalidate" for why that matters.
"""
import pandas as pd
import pandas_market_calendars as mcal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV_PATH

pd.set_option('display.max_columns', None)


def check_trading_calendar(csv_path=RAW_CSV_PATH):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    df['Date'] = df['Date'].dt.normalize()

    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.normalize()

    calendar_df = pd.DataFrame({'Date': trading_days})
    calendar_df['Next_Trading_Day'] = calendar_df['Date'].shift(-1)

    data_dates = pd.Index(df['Date'].unique())

    # Check 1: NYSE trading days whose next trading day is missing from our data
    calendar_df['Next_Day_Missing'] = ~calendar_df['Next_Trading_Day'].isin(data_dates)
    next_day_missing = calendar_df[calendar_df['Next_Day_Missing']]

    # Check 2: dates in our data that NYSE doesn't recognize as a trading day
    is_missing_from_calendar = ~data_dates.isin(calendar_df['Date'])
    dates_not_in_nyse_calendar = data_dates[is_missing_from_calendar]

    return {
        "is_monotonic_increasing": df['Date'].is_monotonic_increasing,
        "duplicated_rows": df[df['Date'].duplicated(keep=False)],
        "data_dates_count": len(data_dates),
        "calendar_days_count": len(calendar_df),
        "dates_not_in_nyse_calendar": dates_not_in_nyse_calendar,
        "next_day_missing": next_day_missing,
    }


if __name__ == "__main__":
    results = check_trading_calendar()

    print("Monotonic increasing dates:", results["is_monotonic_increasing"])
    print("\nDuplicated date rows:")
    print(results["duplicated_rows"])

    print("\nData date count:", results["data_dates_count"])
    print("NYSE calendar day count:", results["calendar_days_count"])

    print("\nDates in data not recognized by NYSE calendar:")
    print(results["dates_not_in_nyse_calendar"])

    print("\nTrading days whose next trading day is missing from data:")
    print(results["next_day_missing"])
