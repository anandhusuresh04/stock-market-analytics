"""
analysis.py
-----------
Statistical analysis of the cleaned multi-ticker dataset.
Calculates returns, correlations, summary statistics, and risk metrics.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics (Mean, Std, Min, Max) for EACH ticker individually.
    We select key columns to avoid overwhelming the output.
    """
    # Drop NaNs so the stats are accurate
    clean_df = df.dropna()
    
    # Group by Ticker to get individual statistics for each stock
    key_cols = ["Close", "Volume", "Simple_Return"]
    return clean_df.groupby("Ticker")[key_cols].describe().T


def calculate_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Calculate the correlation matrix of Simple Returns between different tickers.
    Pearson correlation ranges from -1.0 to +1.0.
    """
    # Pivot the data so columns are Tickers and rows are Dates
    pivot_df = df.pivot_table(index="Date", columns="Ticker", values="Simple_Return")
    return pivot_df.corr(method=method)


def calculate_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the covariance matrix of Simple Returns between tickers."""
    pivot_df = df.pivot_table(index="Date", columns="Ticker", values="Simple_Return")
    return pivot_df.cov()


def get_annualized_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate annualized volatility for each ticker.
    NSE has 75 5-minute bars per day (6.25 hours). 75 * 252 days = 18,900 bars/year.
    """
    daily_vol = df.groupby("Ticker")["Simple_Return"].std()
    annualized_vol = daily_vol * np.sqrt(18900) 
    return annualized_vol.round(4)


def calculate_expected_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the average 5-minute return for each stock.
    This tells us which stock makes the most money on average per bar.
    """
    expected_returns = df.groupby("Ticker")["Simple_Return"].mean()
    return expected_returns


def calculate_sharpe_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Sharpe Ratio (Risk-Adjusted Return).
    Formula: Expected Return / Annualized Volatility.
    (Assuming Risk-Free Rate = 0 for this academic baseline).
    """
    expected_return = calculate_expected_returns(df)
    volatility = get_annualized_volatility(df)
    
    # We multiply by sqrt(18900) to annualize the expected return as well
    # so the units (yearly return vs yearly volatility) match.
    annualized_return = expected_return * 18900
    sharpe = annualized_return / volatility
    
    return sharpe.round(4)


def get_return_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Skewness and Kurtosis of returns to test for Normal Distribution.
    """
    # We apply a lambda function to avoid Pandas string-lookup bugs in .agg()
    dist_stats = df.groupby("Ticker")["Simple_Return"].apply(
        lambda x: pd.Series({
            'Skewness': x.skew(),
            'Kurtosis': x.kurt()
        })
    )
    return dist_stats.round(4)