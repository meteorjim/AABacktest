# AABacktest Project Analysis

## 1. Project Overview
**AABacktest** is a specialized asset allocation backtesting system designed for ETF portfolios in the Chinese A-share market. It allows users to simulate various investment strategies, including periodic rebalancing and Dollar Cost Averaging (DCA), with comprehensive performance analysis and visualization.

## 2. Architecture & Core Components

### 2.1. Core Class: `PortfolioBacktester`
Located in `portfolio_backtester.py`, this class is the heart of the system.
-   **Initialization**: Accepts ETF codes, weights, dates, capital, and strategy detailed parameters (rebalancing freq, DCA settings, transaction costs).
-   **Data Fetching**: `fetch_data()` retrieves historical data for specified ETFs using `akshare`.
-   **Backtesting Engine**: `run_backtest()` executes the simulation day-by-day:
    -   **Initial Buy**: Buys initial positions on the start date (or first available trading day).
    -   **Rebalancing**: Checks `_rebalance_portfolio()` daily.Triggers based on time frequency (monthly/quarterly/yearly) or threshold deviation.
    -   **DCA**: Executes `_dca_buy()` on scheduled dates (monthly/yearly).
    -   **Tracking**: Records daily values, positions, and transactions.
-   **Analysis**: `calculate_performance_metrics()` computes Total Return, CAGR, Sharpe Ratio, Max Drawdown, Volatility, etc.
-   **Reporting**: `generate_report()` creates an HTML report with interactive Plotly charts.

### 2.2. Data Module
-   **Source**: [AkShare](https://akshare.xyz/) is used to fetch A-share stock, ETF, and index data.
-   **Caching**: Custom caching mechanism in `get_price_akshare`:
    -   Files saved in `data/` directory as CSVs.
    -   Format: `{stock_code}_{start_date}_{end_date}.csv`.
    -   Smart matching: Can reuse larger time range caches for smaller queries.
    -   Fallback logic: Stock -> ETF -> Index data lookup sequence.

### 2.3. Market Analysis
Located in `market_period_analysis.py`:
-   **Purpose**: Tests strategies across defined historical market cycles (e.g., "2014-2015 Bull Market", "2018-2019 Bear Market").
-   **Configuration**: Pre-defined asset allocation mixes (e.g., "Balanced", "Aggressive", "Gold-Heavy").
-   **Output**: Generates `market_periods_analysis.csv` and heatmaps to compare strategy robustness.

## 3. Key Features
-   **Rebalancing**: Time-based (Monthly, Quarterly, Yearly) or Threshold-based (e.g., >5% deviation).
-   **DCA (Fixed Investment)**: Supports regular fixed-amount contributions.
-   **Transaction Costs**: Configurable buy/sell cost rates.
-   **Visualization**: Plotly-based dashboards showing Wealth Curve, Drawdown, Monthly/Yearly Returns Heatmaps.
-   **Metrics**: Sharpe Ratio, Max Drawdown Recovery Days, Annualized Volatility.

## 4. Environment & Dependencies
-   **Python Version**: >=3.12
-   **Dependency Manager**: `uv` or `pip`.
-   **Key Libs**: `pandas` (Data manipulation), `numpy` (Math), `akshare` (Financial Data), `plotly` (Visualization).

## 5. Usage for Agents

### Running a Basic Backtest
```python
from portfolio_backtester import PortfolioBacktester

# Initialize
bt = PortfolioBacktester(
    etf_codes=['510300', '510500'], # HS300, CSI500
    weights=[0.5, 0.5],
    start_date='2020-01-01',
    end_date='2023-12-31',
    rebalance_freq='quarterly',
    enable_rebalancing=True
)

# Run
bt.run_backtest()

# Get Results
results = bt.get_results()
print(results['total_return'])
```

### Running Market Analysis
Execute the script directly to generate comparison CSVs:
```bash
uv run python market_period_analysis.py
```

## 6. File Structure
-   `portfolio_backtester.py`: Core logic.
-   `market_period_analysis.py`: Cycle analysis script.
-   `example_usage.py`: Simple usage demo.
-   `test_verbose.py`: Demo with verbose logging enabled.
-   `data/`: Local data cache (git-ignored usually, but present in structure).
-   `pyproject.toml`: Project config.

## 7. Notes for Development
-   **Data Consistency**: AkShare data might sometimes return empty frames for specific dates/tickers. The system handles this by printing warnings or trying alternative data sources (Stock/ETF/Index).
-   **Date Handling**: The system auto-aligns trading dates. DCA and Rebalancing logic skips non-trading days to the next available one.
-   **Performance**: `force_refresh=False` is default to speed up repeated runs. Set to `True` if data seems stale or corrupted.
