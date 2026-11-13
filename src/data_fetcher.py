"""
data_fetcher.py
---------------
Download and cache OHLCV price data via yfinance.

Strategy: Use 5-minute interval (allowed up to 60 days of history per ticker)
to build a >10,000 row dataset by combining multiple tickers.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OHLCV_ORDER = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex and remove the 'Price' label."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    present = [c for c in OHLCV_ORDER if c in df.columns]
    return df[present]


def fetch_market_data(
    tickers: list[str],
    interval: str = "5m",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download data for multiple tickers, add a 'Ticker' column, and combine.
    Uses 5m interval to bypass Yahoo's strict 1m/7-day limit.
    """
    cache_name = f"multi_ticker_{interval}.csv"
    cache_path = RAW_DIR / cache_name

    if cache_path.exists() and not force_refresh:
        print(f"Loading from cache: {cache_name}")
        return pd.read_csv(cache_path, index_col="Date", parse_dates=True)

    all_data = []
    # For 5m, 15m, 30m data, Yahoo allows 60 days in one go.
    period = "60d" if interval != "1m" else "7d"

    for ticker in tickers:
        print(f"Downloading {ticker} ({interval} interval)...")
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            actions=False,
        )

        if df is not None and not df.empty:
            df = _flatten_columns(df)
            df.index.name = "Date"
            df["Ticker"] = ticker  # Tag the stock
            all_data.append(df)
        else:
            print(f"Warning: No data for {ticker}")

    if not all_data:
        raise ValueError("No data could be fetched for any of the provided tickers.")

    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()
    final_df.to_csv(cache_path)
    print(f"Saved {len(final_df)} rows to {cache_name}")
    return final_df


if __name__ == "__main__":
    # Fetch 3 major Indian stocks using 5-minute intervals
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    df = fetch_market_data(tickers, interval="5m")

    print(f"\n--- SUCCESS ---")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nLast 3 rows:")
    print(df.tail(3))