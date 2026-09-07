"""
Fetches raw QQQ OHLCV data from yfinance and saves it to data/raw/.
Run this directly to (re)download the dataset: python -m src.data.load
"""
import yfinance as yf
import pandas as pd
from pathlib import Path


def fetch_qqq_data(ticker: str) -> pd.DataFrame:
    dat = yf.Ticker(ticker)
    prices = dat.history(period="max", auto_adjust=True)  # auto_adjust: see decision_log.md, Phase 1 #3
    df = pd.DataFrame(prices)
    return df


def save_raw_data(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"qqq_{start_date}_to_{end_date}.csv"
    df.to_csv(output_path, columns=["Open", "High", "Low", "Close", "Volume"], index=True, index_label="Date")
    return str(output_path)


def summarise(df: pd.DataFrame, ticker: str) -> None:
    print("=====SANITY CHECK=====")
    print("Ticker:", ticker)
    print("Shape:", df.shape)
    print("\nData Type:")
    print(df.dtypes)
    print("Row count:", len(df))
    print("Column names:", df.columns.tolist())
    print("Date Range:", df.index[0], "to", df.index[-1])
    print("\nFirst 2 rows:")
    print(df.head(2))
    print("\nLast 2 rows:")
    print(df.tail(2))
    print("=========END==========")


if __name__ == "__main__":
    ticker = "QQQ"
    df = fetch_qqq_data(ticker)
    summarise(df, ticker)
    path = save_raw_data(df)
    print(f"Saved to {path}")
