"""
visualization.py
----------------
Generate professional charts for the academic report.
Saves all plots as PNG files in the visualizations/ directory.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
from pathlib import Path

# Path to save the images
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = PROJECT_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Set a professional visual theme
sns.set_theme(style="darkgrid")

def _format_time_axis():
    """Helper function to rotate and format the X-axis so it is readable."""
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    plt.gcf().autofmt_xdate()

def plot_moving_averages(df: pd.DataFrame, ticker: str, save: bool = True):
    """Plot Close Price with SMA_20 and SMA_50 for a specific ticker."""
    ticker_df = df[df["Ticker"] == ticker].tail(500).copy()

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_df.index, ticker_df["Close"], label="Close Price", color="black", alpha=0.5)
    plt.plot(ticker_df.index, ticker_df["SMA_20"], label="SMA 20", color="blue", linewidth=1.5)
    plt.plot(ticker_df.index, ticker_df["SMA_50"], label="SMA 50", color="red", linewidth=1.5)
    
    plt.title(f"{ticker} - Price and Moving Averages (Last 500 Bars)", fontsize=16, fontweight='bold')
    plt.ylabel("Price (INR)", fontsize=12)
    plt.legend()
    _format_time_axis()
    plt.tight_layout()

    if save:
        path = VIS_DIR / f"{ticker.replace('.', '_')}_moving_averages.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()

def plot_bollinger_bands(df: pd.DataFrame, ticker: str, save: bool = True):
    """Plot Close Price with Bollinger Bands for a specific ticker."""
    ticker_df = df[df["Ticker"] == ticker].tail(500).copy()

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_df.index, ticker_df["Close"], label="Close Price", color="black")
    plt.plot(ticker_df.index, ticker_df["BB_Upper"], label="Upper Band", color="green", linestyle="--", alpha=0.7)
    plt.plot(ticker_df.index, ticker_df["BB_Lower"], label="Lower Band", color="green", linestyle="--", alpha=0.7)
    
    plt.fill_between(ticker_df.index, ticker_df["BB_Lower"], ticker_df["BB_Upper"], color="green", alpha=0.1)

    plt.title(f"{ticker} - Bollinger Bands (Last 500 Bars)", fontsize=16, fontweight='bold')
    plt.ylabel("Price (INR)", fontsize=12)
    plt.legend(loc="upper left")
    _format_time_axis()
    plt.tight_layout()

    if save:
        path = VIS_DIR / f"{ticker.replace('.', '_')}_bollinger_bands.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()

def plot_rsi(df: pd.DataFrame, ticker: str, save: bool = True):
    """Plot the Relative Strength Index (RSI) with Overbought/Oversold lines."""
    ticker_df = df[df["Ticker"] == ticker].tail(500).copy()

    plt.figure(figsize=(14, 5))
    plt.plot(ticker_df.index, ticker_df["RSI_14"], color="purple", label="RSI (14)")

    # Add Overbought (70) and Oversold (30) limits
    plt.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    plt.fill_between(ticker_df.index, 70, 100, color="red", alpha=0.05)
    plt.fill_between(ticker_df.index, 0, 30, color="green", alpha=0.05)

    plt.title(f"{ticker} - Relative Strength Index (RSI)", fontsize=16, fontweight='bold')
    plt.ylabel("RSI Value", fontsize=12)
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    _format_time_axis()
    plt.tight_layout()

    if save:
        path = VIS_DIR / f"{ticker.replace('.', '_')}_rsi.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """Plot a heatmap of the correlation matrix."""
    pivot_df = df.pivot_table(index="Date", columns="Ticker", values="Simple_Return")
    corr_matrix = pivot_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title("Correlation of Simple Returns (5-min)", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save:
        path = VIS_DIR / "correlation_heatmap.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()

def plot_return_distribution(df: pd.DataFrame, save: bool = True):
    """
    Plot the distribution of returns and compare it to a theoretical Normal curve.
    This visually proves the high Kurtosis (fat tails) calculated in analysis.py.
    """
    returns = df["Simple_Return"].dropna()
    
    plt.figure(figsize=(12, 6))
    # 1. Plot actual returns (Purple Histogram)
    sns.histplot(returns, bins=200, kde=True, color="purple", stat="density", 
                 line_kws={"linewidth": 2}, label="Actual Returns")
    
    # 2. Plot theoretical Normal Distribution (Red Bell Curve)
    mu, std = returns.mean(), returns.std()
    xmin, xmax = -0.005, 0.005 # Zoomed in
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2, label="Theoretical Normal Curve")

    plt.title("Distribution of 5-min Returns: Actual vs Normal (Fat Tails)", fontsize=16, fontweight='bold')
    plt.xlabel("Simple Return", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(xmin, xmax) 
    plt.legend()
    plt.tight_layout()

    if save:
        path = VIS_DIR / "return_distribution.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()