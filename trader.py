import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from rich import print as rprint

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def fetch_history(tickers, start, end, interval="1d"):
    out = {}
    for t in tqdm(tickers, desc="Fetching data"):
        try:
            df = yf.download(t, start=start, end=end, interval=interval, progress=False)
            if df.empty:
                rprint(f"[yellow]No data for {t}[/yellow]")
                continue

            # Fix any MultiIndex or malformed columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
            # If columns are the ticker name repeated
            if all(str(t).lower() in str(c).lower() for c in df.columns):
                n = len(df.columns)
                df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"][:n]

            df.columns = [c.lower().strip() for c in df.columns]
            if "adj close" not in df.columns and "close" in df.columns:
                df["adj close"] = df["close"]

            df = df.dropna(how="any")
            out[t] = df
        except Exception as e:
            rprint(f"[red]Error fetching {t}: {e}[/red]")
    return out


def safe_price(df: pd.DataFrame) -> pd.Series:
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for key in ("adj close", "close"):
        if key in df.columns:
            return df[key].dropna()
    raise KeyError(f"No usable price column found: {df.columns}")

# ---------------------------------------------------------------------
# Cointegration Search
# ---------------------------------------------------------------------

def find_cointegrated_pairs(data_dict, significance=0.05, max_pairs=10):
    keys = list(data_dict.keys())
    n = len(keys)
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = keys[i], keys[j]
            s_a, s_b = safe_price(data_dict[a]), safe_price(data_dict[b])
            df = pd.concat([s_a, s_b], axis=1).dropna()
            if len(df) < 250:
                continue
            x, y = df.iloc[:, 0], df.iloc[:, 1]
            y_const = sm.add_constant(x)
            model = sm.OLS(y, y_const).fit()
            resid = model.resid
            adf_p = adfuller(resid)[1]
            if adf_p < significance:
                candidates.append((a, b, adf_p))
            if len(candidates) >= max_pairs:
                break
        if len(candidates) >= max_pairs:
            break
    candidates.sort(key=lambda x: x[2])
    return candidates

# ---------------------------------------------------------------------
# Spread + Neural augmentation
# ---------------------------------------------------------------------

def compute_spread(x, y):
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    beta = model.params[1]
    spread = y - beta * x
    return spread, beta


def train_spread_predictor(spread):
    scaler = StandardScaler()
    s_scaled = scaler.fit_transform(spread.values.reshape(-1, 1))
    lookback = 20
    X, y = [], []
    for i in range(lookback, len(s_scaled)):
        X.append(s_scaled[i - lookback:i])
        y.append(s_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=25, batch_size=16, verbose=0, callbacks=[es])
    return model, scaler, lookback

# ---------------------------------------------------------------------
# Backtest with Trade Tracking
# ---------------------------------------------------------------------

def backtest_pair(x, y, cash=100000, entry_z=1.0, exit_z=0.2):
    spread, beta = compute_spread(x, y)
    z = (spread - spread.mean()) / spread.std()

    position = 0  # +1 long spread, -1 short spread
    pnl = [cash]
    capital = cash
    size = cash * 0.1  # risk 10% per trade
    
    # Track all trades
    trades = []
    entry_price = None
    entry_date = None

    for i in range(1, len(z)):
        # trading logic
        if position == 0:
            if z.iloc[i] > entry_z:
                position = -1
                entry_price = spread.iloc[i]
                entry_date = spread.index[i]
            elif z.iloc[i] < -entry_z:
                position = 1
                entry_price = spread.iloc[i]
                entry_date = spread.index[i]

        elif position == 1 and z.iloc[i] > -exit_z:
            profit = (spread.iloc[i] - entry_price) * (size / spread.std())
            capital += profit
            trades.append({
                'entry_date': entry_date,
                'exit_date': spread.index[i],
                'position': 'LONG',
                'entry_spread': entry_price,
                'exit_spread': spread.iloc[i],
                'profit': profit,
                'entry_z': z.loc[entry_date],
                'exit_z': z.iloc[i]
            })
            position = 0
            entry_price = None
            entry_date = None
            
        elif position == -1 and z.iloc[i] < exit_z:
            profit = (entry_price - spread.iloc[i]) * (size / spread.std())
            capital += profit
            trades.append({
                'entry_date': entry_date,
                'exit_date': spread.index[i],
                'position': 'SHORT',
                'entry_spread': entry_price,
                'exit_spread': spread.iloc[i],
                'profit': profit,
                'entry_z': z.loc[entry_date],
                'exit_z': z.iloc[i]
            })
            position = 0
            entry_price = None
            entry_date = None

        pnl.append(capital)

    return np.array(pnl), spread, z, beta, trades

# ---------------------------------------------------------------------
# Enhanced Plot with Trade Markers
# ---------------------------------------------------------------------

def plot_performance(portfolio, benchmark, spread, z, trades, title, ticker_a, ticker_b):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Portfolio vs Benchmark
    ax1 = axes[0]
    ax1.plot(portfolio.index, portfolio / portfolio.iloc[0], label="Strategy", linewidth=2, color='blue')
    ax1.plot(benchmark.index, benchmark / benchmark.iloc[0], "--", label="Buy & Hold", linewidth=1.5, color='orange')
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spread with Trade Markers
    ax2 = axes[1]
    ax2.plot(spread.index, spread, label='Spread', linewidth=1, color='gray', alpha=0.7)
    ax2.axhline(spread.mean(), color='black', linestyle='--', linewidth=0.8, label='Mean')
    ax2.axhline(spread.mean() + spread.std(), color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.axhline(spread.mean() - spread.std(), color='green', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Mark trades on spread
    for trade in trades:
        color = 'green' if trade['position'] == 'LONG' else 'red'
        marker_entry = '^' if trade['position'] == 'LONG' else 'v'
        marker_exit = 'v' if trade['position'] == 'LONG' else '^'
        
        ax2.scatter(trade['entry_date'], trade['entry_spread'], 
                   color=color, marker=marker_entry, s=100, zorder=5, alpha=0.8)
        ax2.scatter(trade['exit_date'], trade['exit_spread'], 
                   color=color, marker=marker_exit, s=100, zorder=5, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax2.set_ylabel('Spread Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Z-score with Trade Markers
    ax3 = axes[2]
    ax3.plot(z.index, z, label='Z-Score', linewidth=1, color='purple')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.axhline(1, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Entry Threshold')
    ax3.axhline(-1, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.axhline(0.2, color='orange', linestyle=':', linewidth=0.8, alpha=0.5, label='Exit Threshold')
    ax3.axhline(-0.2, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Mark trades on z-score
    for trade in trades:
        color = 'green' if trade['position'] == 'LONG' else 'red'
        marker_entry = '^' if trade['position'] == 'LONG' else 'v'
        marker_exit = 'v' if trade['position'] == 'LONG' else '^'
        
        ax3.scatter(trade['entry_date'], trade['entry_z'], 
                   color=color, marker=marker_entry, s=100, zorder=5, alpha=0.8)
        ax3.scatter(trade['exit_date'], trade['exit_z'], 
                   color=color, marker=marker_exit, s=100, zorder=5, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_trade_summary(trades, initial_capital, final_capital):
    """Print detailed trade statistics"""
    if not trades:
        rprint("[yellow]No trades executed during the period.[/yellow]")
        return
    
    trades_df = pd.DataFrame(trades)
    
    rprint("\n" + "="*70)
    rprint("[bold cyan]TRADE SUMMARY[/bold cyan]")
    rprint("="*70)
    
    rprint(f"\n[bold]Total Trades:[/bold] {len(trades)}")
    rprint(f"[bold]Long Trades:[/bold] {len(trades_df[trades_df['position'] == 'LONG'])}")
    rprint(f"[bold]Short Trades:[/bold] {len(trades_df[trades_df['position'] == 'SHORT'])}")
    
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]
    
    rprint(f"\n[green]Winning Trades:[/green] {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    rprint(f"[red]Losing Trades:[/red] {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    
    if len(winning_trades) > 0:
        rprint(f"[green]Avg Win:[/green] ₹{winning_trades['profit'].mean():,.2f}")
        rprint(f"[green]Largest Win:[/green] ₹{winning_trades['profit'].max():,.2f}")
    
    if len(losing_trades) > 0:
        rprint(f"[red]Avg Loss:[/red] ₹{losing_trades['profit'].mean():,.2f}")
        rprint(f"[red]Largest Loss:[/red] ₹{losing_trades['profit'].min():,.2f}")
    
    total_profit = trades_df['profit'].sum()
    rprint(f"\n[bold]Total Profit/Loss:[/bold] ₹{total_profit:,.2f}")
    rprint(f"[bold]Return:[/bold] {(final_capital/initial_capital - 1)*100:.2f}%")
    
    # Trade details table
    rprint("\n[bold cyan]INDIVIDUAL TRADES:[/bold cyan]")
    rprint("-"*70)
    for i, trade in enumerate(trades, 1):
        profit_color = "green" if trade['profit'] > 0 else "red"
        rprint(f"\nTrade #{i} - [bold]{trade['position']}[/bold]")
        rprint(f"  Entry: {trade['entry_date'].strftime('%Y-%m-%d')} | Z-Score: {trade['entry_z']:.2f}")
        rprint(f"  Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} | Z-Score: {trade['exit_z']:.2f}")
        rprint(f"  [{profit_color}]P&L: ₹{trade['profit']:,.2f}[/{profit_color}]")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    cash = float(input("Enter virtual cash amount (INR): ") or 100000)
    tickers_str = input("Enter tickers (comma-separated, use .NS): ") or "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    start = input("Start date (YYYY-MM-DD): ") or "2020-01-01"
    end = input("End date (YYYY-MM-DD): ") or "2025-01-01"
    tickers = [t.strip() for t in tickers_str.split(",")]

    rprint("\nFetching historical data...")
    data = fetch_history(tickers, start, end)
    if not data:
        rprint("[red]No data fetched. Exiting.[/red]")
        return

    rprint("\nScanning candidate pairs for cointegration...")
    pairs = find_cointegrated_pairs(data)
    if not pairs:
        rprint("[red]No cointegrated pairs found.[/red]")
        return

    a, b, p = pairs[0]
    rprint(f"\nBest pair: [bold]{a}[/bold] vs [bold]{b}[/bold]  (p={p:.4f})")

    s_a = safe_price(data[a])
    s_b = safe_price(data[b])
    aligned = pd.concat([s_a, s_b], axis=1).dropna()
    x, y = aligned.iloc[:, 0], aligned.iloc[:, 1]

    portfolio, spread, z, beta, trades = backtest_pair(x, y, cash)

    # Align dates for plotting
    dates = aligned.index[1:]
    portfolio_series = pd.Series(portfolio[1:], index=dates)
    benchmark_series = pd.Series(x.values[1:], index=dates)

    plot_performance(portfolio_series, benchmark_series, spread, z, trades,
                    f"{a} vs {b} | Hedge Ratio {beta:.3f}", a, b)

    rprint(f"\n[green]Final Portfolio Value:[/green] ₹{portfolio[-1]:,.2f}")
    print_trade_summary(trades, cash, portfolio[-1])

if __name__ == "__main__":
    main()
