# AABacktest Project Guide for AI Agents

This document is designed to help AI agents quickly understand the **AABacktest** project structure, core logic, and intended usage patterns.

## 1. Project Overview
**AABacktest** is a Python-based Asset Allocation Backtesting framework. It simulates the performance of an ETF portfolio over time with features like **Periodic Rebalancing**, **Threshold Rebalancing**, **Dollar-Cost Averaging (DCA)**, and **Synthetic Dividend Yields**.

### Tech Stack
-   **Language**: Python 3.12+
-   **Data Source**: `akshare` (Chinese A-share market data)
-   **Data Processing**: `pandas`, `numpy`
-   **Visualization**: `plotly` (Interactive HTML reports)
-   **Package Manager**: `uv` (recommended) or `pip`

## 2. File Structure

| File | Purpose | Key Classes/Functions |
| :--- | :--- | :--- |
| `portfolio_backtester.py` | **CORE ENGINE**. Contains all logic for data fetching, simulation, and reporting. | `PortfolioBacktester` |
| `example_usage.py` | **Entry Point**. Demonstrates how to configure and run a backtest. | `if __name__ == "__main__":` |
| `test_verbose.py` | Usage example focusing on detailed logging output. | |
| `data/` | **Cache Directory**. Stores downloaded ETF historical data CSVs (format: `{code}_{start}_{end}.csv`). | |

## 3. Core Logic: `PortfolioBacktester`

The `PortfolioBacktester` class orchestrates the entire lifecycle.

### 3.1 Initialization (`__init__`)
-   **Inputs**: ETF codes, target weights, dates, capital, strategy settings (DCA, Rebalance).
-   **Key Params**:
    -   `rebalance_freq` ('quarterly', etc.): Time-based rebalancing.
    -   `rebalance_threshold` (float): Deviation-based rebalancing (e.g., 0.05 for 5%).
    -   `verbose_trading` (bool): Detailed transaction logs.
    -   `show_daily_logs` (bool): Daily output switch.

### 3.2 Data Loading (`fetch_data`)
-   Iterates through `self.etf_codes`.
-   **Synthetic Dividend Parsing**: Detects if an `@yield` suffix is present (e.g., `000015@0.05`). Strips the suffix for fetch logic.
-   Checks `data/` for valid cached CSVs.
-   If missing/stale/forced: Calls `ak.fund_etf_hist_em` via `get_price_akshare`.
-   **Synthetic Dividend Processing**: Converts the parsed annual yield into a daily compounded factor, mutating the trailing DataFrame `open`, `close`, `high`, `low` rows.
-   Calculates the full simulated dataset, storing it with the original suffix-appended string key for consistent tracking.

### 3.3 Simulation Loop (`run_backtest`)
The heart of the simulation.
1.  **Pre-calculation**:
    -   `_get_time_rebalance_dates()`: Identifies scheduled rebalance days.
    -   `_get_dca_dates()`: Identifies DCA days.
    -   `_initial_buy()`: Executes initial position sizing on Day 0.
2.  **Daily Iteration** (`for date in trading_dates`):
    -   **Rebalance Check**:
        -   Is it a `time_rebalance_date`? -> Yes, Rebalance.
        -   Is `rebalance_threshold` > 0 AND `_check_rebalance_threshold(date)` is True? -> Yes, Rebalance (Dynamic).
        -   *Action*: `_rebalance_portfolio(date)` targets `self.effective_weights` (which filters out missing assets). Valuation target mapping is rigorously pinned to `close` prices (using a **Past-Date Fallback Engine** if today's data is missing). Actual execution is converted into **Pending Orders**, which are rigorously pinned to the `open` price of the next available trading day. This decoupling ensures accurate target calculation while respecting realistic execution constraints.
    -   **DCA Check**:
        -   Is it a `dca_date`? -> Yes, `_dca_buy(date)`. Uses the same target setting and pending order mechanism pinned to `open` prices.
    -   **Position Update**:
        -   Updates daily market value based on `close` prices.
        -   Records state to `self.daily_positions` and `self.daily_values`.

### 3.4 Reporting (`_calculate_results` & `generate_report`)
-   Calculates metrics: Total Return, CAGR, Max Drawdown, Sharpe Ratio, Volatility.
-   **Max Drawdown Details**: Strict conversion from TWR (Time-Weighted Return) index logic instead of absolute variations, protecting the curve against DCA value spikes.
-   Generates Plotly HTML:
    -   Portfolio Value Curve
    -   Drawdown Curve
    -   Asset Allocation Area Chart
    -   Top 10 Gains/Losses
-   Saves to `backtest_report_{timestamp}.html`.

## 4. Key Data Structures

### `self.positions` (Dict)
Tracks current holdings.
```python
{
    '510300': {'shares': 1200.5, 'cost': 50000.0},
    '510500': {'shares': 800.0, 'cost': 40000.0}
}
```

### `self.etf_data` (Dict of DataFrames)
Historical price data.
-   Key: ETF Code (str)
-   Value: `pd.DataFrame` (Index=Date, Cols=['open', 'close', 'high', 'low', 'volume'])

### `self.rebalance_dates` (List)
-   Accumulates both *Time-based* and *Threshold-based* rebalance events used for metrics.

## 5. Common Modification Tasks

| Goal | Action |
| :--- | :--- |
| **Add New Strategy** | Modify `run_backtest` loop. Add new condition calls (e.g., `_check_momentum_signal(date)`). |
| **Change Data Source** | Modify `_load_data` or `get_price_akshare`. Ensure return format matches `pd.DataFrame` with standard columns. |
| **Custom Fees** | Modify `transaction_cost` application in `_rebalance_portfolio` (lines ~820, ~870) and `_dca_buy` (~660). |
| **Add Metrics** | Update `_calculate_results`. accessing `self.daily_values` series. |

## 6. Development Tips
-   **Testing**: Use `uv run python example_usage.py` for quick end-to-end verification.
-   **Logging**: Toggle `show_daily_logs=True`/`False` in `example_usage.py` to inspect logic details vs. clean output.
-   **Caching**: If you suspect data issues, set `force_refresh=True` or delete `data/` folder.
