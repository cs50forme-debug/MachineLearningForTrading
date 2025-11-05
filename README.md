# Pairs Trading Strategy with Neural Networks

A Python-based statistical arbitrage system that identifies cointegrated stock pairs and executes mean-reversion trades with LSTM-enhanced spread prediction.

## Overview

This project implements a pairs trading strategy that:
- Identifies statistically related stock pairs using cointegration tests
- Monitors price spreads between paired securities
- Executes trades when spreads deviate from historical norms
- Tracks performance with detailed trade analytics

## What is Pairs Trading?

Pairs trading is a market-neutral strategy that exploits temporary price divergences between two historically correlated assets. When one stock becomes relatively expensive compared to its pair, the strategy shorts the expensive stock and longs the cheap one, profiting when prices converge back to their historical relationship.

## Key Concepts

### Cointegration
Two stocks are cointegrated when their prices move together over time, even if each individual price series is non-stationary. The strategy uses the Augmented Dickey-Fuller test to identify pairs with stable long-term relationships.

### Spread and Z-Score
The spread is the difference between two stock prices after adjusting for their hedge ratio (beta). The z-score standardizes this spread, showing how many standard deviations it is from its mean. Trading signals are generated when z-scores cross predefined thresholds.

### Hedge Ratio
The beta coefficient from linear regression that determines the optimal quantity ratio for pairing two stocks. This ensures the position is market-neutral.

## Requirements

```bash
pip install numpy pandas yfinance statsmodels matplotlib tensorflow scikit-learn tqdm rich
```

## Installation

```bash
git clone <repository-url>
cd pairs-trading-strategy
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python pairs_trading.py
```

### Input Parameters

When prompted, provide:

1. **Virtual cash amount**: Starting capital in INR (default: 100,000)
2. **Tickers**: Comma-separated stock symbols with .NS suffix for NSE stocks
   ```
   Example: RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS
   ```
3. **Start date**: Historical data start (format: YYYY-MM-DD)
4. **End date**: Historical data end (format: YYYY-MM-DD)

### Example Run

```
Enter virtual cash amount (INR): 100000
Enter tickers (comma-separated, use .NS): RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS
Start date (YYYY-MM-DD): 2020-01-01
End date (YYYY-MM-DD): 2025-01-01
```

## Strategy Logic

### Step 1: Data Collection
```python
data = fetch_history(tickers, start, end)
```
Downloads historical price data for specified tickers using yfinance API.

### Step 2: Cointegration Testing
```python
pairs = find_cointegrated_pairs(data, significance=0.05)
```
Tests all possible ticker pairs for cointegration using the ADF test. Pairs with p-values below 0.05 are considered statistically significant.

### Step 3: Spread Calculation
```python
spread, beta = compute_spread(x, y)
z = (spread - spread.mean()) / spread.std()
```
Calculates the price spread using OLS regression and converts it to a z-score for standardized thresholds.

### Step 4: Signal Generation

**Entry Signals:**
- Long spread when z-score < -1.0 (spread unusually low)
- Short spread when z-score > +1.0 (spread unusually high)

**Exit Signals:**
- Close long when z-score > -0.2 (approaching mean)
- Close short when z-score < +0.2 (approaching mean)

### Step 5: Position Sizing
```python
size = cash * 0.1  # Risk 10% per trade
```
Each trade risks 10% of available capital, scaled by spread volatility.

## Output

### Visual Analysis
The system generates three stacked plots:

1. **Portfolio Performance**: Strategy returns vs buy-and-hold benchmark
2. **Spread Chart**: Price spread with trade entry/exit markers
3. **Z-Score Chart**: Standardized spread with threshold lines

**Trade Markers:**
- Green triangles (↑): Long entries
- Red triangles (↓): Short entries
- Inverted triangles with black borders: Exit points

### Trade Summary
Detailed statistics including:
- Total number of trades executed
- Win rate and profit factor
- Average win/loss amounts
- Individual trade breakdown with dates and P&L
  
## Code Structure

### Core Functions

**fetch_history(tickers, start, end)**
Downloads and cleans historical price data from Yahoo Finance.

**find_cointegrated_pairs(data_dict)**
Identifies statistically cointegrated pairs using the ADF test on regression residuals.

**compute_spread(x, y)**
Calculates the spread between two price series using OLS regression.

**train_spread_predictor(spread)**
Trains an LSTM neural network to predict future spread movements (experimental feature).

**backtest_pair(x, y, cash, entry_z, exit_z)**
Simulates trading strategy and tracks all executed trades with detailed metrics.

**plot_performance(...)**
Generates comprehensive visualization of strategy performance and trade markers.

**print_trade_summary(trades, initial_capital, final_capital)**
Outputs detailed trade statistics and individual trade information.

## Strategy Parameters

### Tunable Variables

```python
entry_z = 1.0      # Z-score threshold for entry
exit_z = 0.2       # Z-score threshold for exit
size = cash * 0.1  # Position size (10% of capital)
lookback = 20      # LSTM lookback window
significance = 0.05 # Cointegration p-value threshold
```

### Risk Management

The strategy implements several risk controls:
- Position sizing limited to 10% per trade
- Mean-reversion exit triggers prevent unlimited losses
- Market-neutral positioning reduces directional exposure
- Cointegration requirement ensures statistical validity

## Limitations

1. **Transaction Costs**: The backtest does not account for brokerage fees, slippage, or taxes
2. **Lookahead Bias**: Uses full-period statistics; live trading requires rolling windows
3. **Market Regime Changes**: Cointegration relationships can break down during structural shifts
4. **Liquidity Assumptions**: Assumes perfect execution at closing prices
5. **Overfitting Risk**: Neural network component may overfit to historical patterns

## Performance Notes

- Strategy performs best in range-bound markets with stable correlations
- Trending markets may trigger early exits or false signals
- Requires sufficient historical data (minimum 250 trading days)
- Win rates typically range from 55-65% for robust pairs

## Future Enhancements

- Real-time data integration via broker APIs
- Dynamic position sizing based on volatility
- Multiple pair portfolios for diversification
- Machine learning for adaptive thresholds
- Out-of-sample testing framework
- Transaction cost modeling

## Disclaimer

This code is for educational purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Users should conduct thorough backtesting and paper trading before deploying real capital.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome. Please submit pull requests with clear descriptions of changes and appropriate test coverage.

## Contact

For questions or suggestions, please open an issue on the repository.
