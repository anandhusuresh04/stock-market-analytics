# NSE Stock Market Data Analytics and Predictive Modeling

## Overview
This repository contains a modular, end-to-end data science pipeline designed for high-frequency (5-minute interval) intraday stock market analysis. The pipeline fetches historical data for major Indian equities (RELIANCE, TCS, INFY), engineers technical indicators, performs statistical risk analysis, and applies machine learning to predict short-term price movements. 

The project specifically addresses a common pitfall in financial machine learning: distinguishing between the illusion of high accuracy in price regression (due to autocorrelation) and the realistic challenge of directional classification.

## Technical Stack
- **Language:** Python 3.10
- **Data Manipulation:** Pandas, NumPy
- **Financial Data API:** yfinance
- **Machine Learning:** Scikit-Learn
- **Data Visualization:** Matplotlib, Seaborn
- **Environment Management:** venv, Jupyter Notebook

## Project Architecture
The codebase is designed with modularity in mind, separating data acquisition, processing, analysis, and visualization into distinct modules.

```text
Stock_Market_Data_Exploring/
│
├── data/
│   ├── raw/                  # Cached raw OHLCV CSV files
│   └── processed/            # Cleaned data with engineered features
│
├── notebooks/
│   └── Stock_Market_Exploring.ipynb  # Main analysis and ML pipeline presentation
│
├── src/
│   ├── data_fetcher.py       # Downloads and caches market data
│   ├── data_cleaner.py       # Calculates technical indicators
│   ├── analysis.py           # Computes statistical and financial metrics
│   └── visualization.py      # Generates and saves charts
│
├── visualizations/           # Output directory for saved PNG charts
├── requirements.txt          # Project dependencies
└── README.md