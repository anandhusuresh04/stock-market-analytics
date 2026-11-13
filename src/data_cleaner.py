"""
data_cleaner.py
---------------
Transform raw multi-ticker OHLCV data into an analysis-ready format.
Calculates technical indicators securely grouped by Ticker and saves to disk.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Simple_Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df.groupby("Ticker")["Close"].shift(1))
    return df

def add_moving_averages(df: pd.DataFrame, windows: list[int] = [20, 50]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=w).mean())
    return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f"Volatility_{window}"] = df.groupby("Ticker")["Simple_Return"].transform(
        lambda x: x.rolling(window=window).std()
    )
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    df = df.copy()
    df["BB_Middle"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window).mean())
    df["BB_Std"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window).std())
    df["BB_Upper"] = df["BB_Middle"] + (num_std * df["BB_Std"])
    df["BB_Lower"] = df["BB_Middle"] - (num_std * df["BB_Std"])
    df.drop(columns=["BB_Std"], inplace=True)
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df.groupby("Ticker")["Close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(window=window).mean())
    avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(window=window).mean())
    rs = avg_gain / avg_loss
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df

# ---------------------------------------------------------------------------
# Pipeline & Save
# ---------------------------------------------------------------------------
def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df.groupby("Ticker")[price_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    
    df = add_returns(df)
    df = add_moving_averages(df, windows=[20, 50])
    df = add_volatility(df, window=20)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    
    return df

def save_processed_data(df: pd.DataFrame, filename: str = "cleaned_market_data.csv") -> Path:
    """Saves the cleaned DataFrame to the data/processed directory."""
    save_path = PROCESSED_DIR / filename
    df.to_csv(save_path)
    print(f"Successfully saved processed data ({len(df)} rows) to: {save_path}")
    return save_path

def describe_quality(df: pd.DataFrame) -> pd.Series:
    report = {
        "Total Rows": len(df),
        "Unique Tickers": df["Ticker"].nunique(),
        "Total Columns": len(df.columns),
        "Missing Values": int(df.isna().sum().sum()),
        "Start Date": df.index.min().strftime("%Y-%m-%d %H:%M"),
        "End Date": df.index.max().strftime("%Y-%m-%d %H:%M"),
    }
    return pd.Series(report)

# ---------------------------------------------------------------------------
# Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_fetcher import fetch_market_data

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    # FIX APPLIED HERE: force_refresh=True tells the fetcher to bypass the 
    # old cache and download the newest live market data from Yahoo Finance.
    raw_df = fetch_market_data(tickers, interval="5m", force_refresh=True)

    # Clean data and calculate indicators
    clean_df = clean_pipeline(raw_df)

    # Display data quality report
    print("\n--- DATA QUALITY REPORT ---")
    print(describe_quality(clean_df))

    # Display final data
    print("\n--- CLEANED DATA ---")
    print(clean_df.tail())

    # Save processed dataset
    save_processed_data(clean_df)