# Pairs Trading Strategy with Neural Networks

A Python-based statistical arbitrage system that identifies cointegrated stock pairs and executes mean-reversion trades with LSTM-enhanced spread prediction.

## Overview

This project implements a pairs trading strategy that:
- Identifies statistically related stock pairs using cointegration tests
- Monitors price spreads between paired securities
- Executes trades when spreads deviate from historical norms
- Tracks performance with detailed trade analytics

## What is Pairs Trading?

Imagine you have two friends who always walk together at the same speed. Suddenly, one friend runs ahead while the other lags behind. You know they'll eventually catch up to each other again. Pairs trading works the same way with stocks.

**The Basic Idea:**
When two stocks that normally move together suddenly drift apart in price, we bet that they'll come back together. We make money when they do.

**How It Works:**
1. Find two stocks that historically move together (like Coca-Cola and Pepsi)
2. Wait for their prices to drift apart
3. Buy the cheaper one and sell the expensive one
4. When prices come back together, close both positions and pocket the profit

**Why "Market-Neutral"?**
This strategy doesn't care if the overall market goes up or down. We only care about the relationship between our two stocks. If the market crashes but our two stocks crash by the same amount, we still make money because their relationship stayed the same.

## Key Concepts Explained Simply

### Cointegration (Finding Stock Buddies)

**What It Means:**
Cointegration is a fancy word for "stocks that stick together." Think of it like best friends - they might wander around individually, but they always end up hanging out together.

**Why It Matters:**
Not all stock pairs work for this strategy. We need stocks that have a reliable, long-term relationship. Maybe they're in the same industry, or they depend on the same raw materials, or they compete for the same customers.

**How We Find Them:**
The code uses something called the "Augmented Dickey-Fuller test" (don't worry about the name). It's just a statistical test that tells us if two stocks have a stable relationship. If the test gives us a p-value below 0.05, it means there's less than a 5% chance the relationship is random.

**Real Example:**
Think about Maruti and Tata Motors. Both car companies are affected by:
- Steel prices
- Fuel costs
- Interest rates (people buy cars on loans)
- Consumer confidence

So their stock prices tend to move together, even though each company has its own ups and downs.

### Spread (Measuring the Gap)

**What It Is:**
The spread is simply the price difference between our two stocks, but adjusted fairly. We can't just subtract one price from another because one stock might be Rs. 2,000 per share while another is Rs. 500 per share.

**The Adjustment (Hedge Ratio):**
We use something called a "hedge ratio" or "beta" to make the comparison fair. It's like converting currencies before comparing prices in different countries.

**Example:**
- Stock A costs Rs. 1,000
- Stock B costs Rs. 500
- The hedge ratio is 2.0

This means Stock A is worth about 2 shares of Stock B. So the spread is: Rs. 1,000 - (2.0 × Rs. 500) = Rs. 0

When this spread grows to Rs. 100 or shrinks to Rs. -100, that's when we trade.

**The Code:**
```python
spread = price_of_stock_A - (beta × price_of_stock_B)
```

### Z-Score (The "Weirdness" Meter)

**What It Tells Us:**
A z-score tells us how unusual the current spread is compared to history. It's measured in "standard deviations," which is just a fancy way of saying "how far from normal."

**Understanding Z-Scores:**
- Z-score of 0: Everything is normal, prices are at their average relationship
- Z-score of +1: The spread is bigger than usual (Stock A is expensive relative to B)
- Z-score of -1: The spread is smaller than usual (Stock A is cheap relative to B)
- Z-score of +2: Very unusual, only happens about 5% of the time
- Z-score of -2: Very unusual in the opposite direction

**Why We Use It:**
Instead of saying "the spread is Rs. 50," which doesn't tell us if that's normal or weird, we say "the z-score is 2.0," which immediately tells us this is unusual and worth trading.

**Trading Decision:**
```
If z-score > +1.0  → Stock A is too expensive, Stock B is too cheap
                   → SHORT A, LONG B (bet on them coming together)

If z-score < -1.0  → Stock A is too cheap, Stock B is too expensive  
                   → LONG A, SHORT B (bet on them coming together)

If z-score near 0  → Exit the trade, they're back to normal
```

### Hedge Ratio (The Magic Number)

**What It Does:**
The hedge ratio tells us exactly how many shares of Stock B we need for every share of Stock A to make our position balanced.

**Simple Example:**
- You buy 100 shares of Stock A
- The hedge ratio is 1.5
- So you sell 150 shares of Stock B

Now you're protected from market movements because you own and owe stocks in the right proportion.

**How It's Calculated:**
The code runs a linear regression (basically drawing the best-fit line) between the two stock prices. The slope of that line is our hedge ratio.

**Why It's Important:**
Without the right hedge ratio, if the market goes up, you might lose money on your short position more than you gain on your long position. The hedge ratio keeps everything balanced.

## Requirements

```bash
pip install numpy pandas yfinance statsmodels matplotlib tensorflow scikit-learn tqdm rich
```

## Installation

```bash
git clone github.com/cs50forme-debug/MachineLearningForTrading.git
cd MachineLearningForTrading
```

## Usage

Run the main script:

```bash
python trader.py
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

## Strategy Logic (Step-by-Step)

### Step 1: Data Collection - Getting Stock Prices

**What Happens:**
```python
data = fetch_history(tickers, start, end)
```

The code connects to Yahoo Finance and downloads historical price data for every stock you entered. For each stock, it gets:
- Open price (price when market opened)
- High price (highest price that day)
- Low price (lowest price that day)
- Close price (price when market closed)
- Volume (how many shares were traded)

**Why Yahoo Finance?**
It's free and has reliable data for thousands of stocks worldwide. The code uses a library called `yfinance` to do this automatically.

**Data Cleaning:**
Sometimes the downloaded data has problems (missing days, weird column names, etc.). The `fetch_history` function cleans all this up so every stock has the same format.

**What You'll See:**
```
Fetching data: 100%|████████████████████| 4/4 [00:03<00:00,  1.5it/s]
```
A progress bar showing how many stocks have been downloaded.

### Step 2: Finding Stock Buddies - Cointegration Testing

**What Happens:**
```python
pairs = find_cointegrated_pairs(data, significance=0.05)
```

Now the code tests EVERY possible pair combination to find which stocks have stable relationships.

**The Process:**
1. Take Stock A and Stock B
2. Run a mathematical regression (draw a best-fit line between their prices)
3. Look at the "errors" (how far actual prices are from the line)
4. Test if these errors are stable over time
5. If the p-value is less than 0.05, the pair passes the test

**Example Math:**
Let's say we test RELIANCE vs TCS:
- The code runs 1,000+ days of price data through the test
- If the p-value comes out as 0.03 (less than 0.05), it's a good pair
- If the p-value is 0.15 (more than 0.05), skip this pair

**What You'll See:**
```
Scanning candidate pairs for cointegration...
Best pair: RELIANCE.NS vs TCS.NS (p=0.0234)
```

The pair with the lowest p-value is the most cointegrated (strongest relationship).

### Step 3: Calculating the Spread

**What Happens:**
```python
spread, beta = compute_spread(x, y)
z = (spread - spread.mean()) / spread.std()
```

Now that we have our winning pair, we need to calculate how far apart they are at any moment.

**The Hedge Ratio (Beta) Calculation:**

Think of this like a recipe ratio. If you're making pasta and the recipe says "2 cups of water for every 1 cup of pasta," that's your ratio. Here, beta tells us the ratio between our two stocks.

**Example:**
```
RELIANCE is trading at Rs. 2,400
TCS is trading at Rs. 3,600
Beta calculated is 0.85

Spread = Rs. 2,400 - (0.85 × Rs. 3,600)
Spread = Rs. 2,400 - Rs. 3,060
Spread = Rs. -660
```

This spread of -660 doesn't mean much by itself. That's where the z-score comes in.

**Converting to Z-Score:**

The code looks at the average spread over the entire time period and the standard deviation (how much it typically varies).

```
Average spread = Rs. -500
Standard deviation = Rs. 200

Current spread = Rs. -660

Z-score = (-660 - (-500)) / 200
Z-score = -160 / 200
Z-score = -0.8
```

A z-score of -0.8 means "the spread is 0.8 standard deviations below average." This is pretty normal, so we don't trade yet.

### Step 4: Trading Signals - When to Buy and Sell

**Entry Signals (When to Open a Trade):**

The code watches the z-score constantly. When it crosses certain thresholds, it triggers a trade.

**Signal 1: Z-Score Drops Below -1.0**
```
Interpretation: Stock A is CHEAP compared to Stock B
Action: LONG the spread
        → Buy Stock A
        → Sell Stock B
Bet: The spread will increase back to normal
```

**Signal 2: Z-Score Rises Above +1.0**
```
Interpretation: Stock A is EXPENSIVE compared to Stock B  
Action: SHORT the spread
        → Sell Stock A
        → Buy Stock B
Bet: The spread will decrease back to normal
```

**Exit Signals (When to Close a Trade):**

We don't wait for the spread to return all the way to zero. We exit when it gets close enough.

**If You're LONG the Spread (bought A, sold B):**
```
Exit when: Z-score crosses above -0.2
Why: The spread has mostly recovered, lock in profits
```

**If You're SHORT the Spread (sold A, bought B):**
```
Exit when: Z-score crosses below +0.2
Why: The spread has mostly recovered, lock in profits
```

**Visual Example Timeline:**

```
Day 1: Z-score = 0.5    (normal, no action)
Day 5: Z-score = 1.2    (HIGH! Enter SHORT spread trade)
Day 12: Z-score = 0.8   (coming down, hold position)
Day 18: Z-score = 0.1   (below 0.2, EXIT trade, take profit)
```

### Step 5: Position Sizing - How Much to Trade

**The Rule:**
```python
size = cash * 0.1  # Risk 10% per trade
```

**What This Means:**
If you have Rs. 1,00,000, each trade will use Rs. 10,000 of capital. This is a safety mechanism so you don't bet everything on one trade.

**Why 10%?**
This is a conservative approach. If a trade goes wrong, you lose at most 10% of your money. Professional traders often use even smaller percentages (5% or less).

**The Math Behind Each Trade:**

Let's work through a complete example:

```
Starting Cash: Rs. 1,00,000
Risk Per Trade: Rs. 10,000 (10%)
Current Spread: -660
Spread Standard Deviation: 200

Z-Score = -1.2 (below -1.0, so ENTER LONG spread)

Position Size Calculation:
Size = Rs. 10,000 / Standard Deviation
Size = Rs. 10,000 / 200
Size = 50 "units" of the spread

This means:
- Buy 50 shares of Stock A
- Sell (50 × hedge_ratio) shares of Stock B
```

**Profit Calculation Example:**

```
Entry spread: -660 (z-score = -1.2)
Exit spread: -420 (z-score = -0.1)
Spread moved: 240 points

Profit = (spread_change) × (size / std)
Profit = 240 × (10,000 / 200)
Profit = 240 × 50
Profit = Rs. 12,000
```

You'd make Rs. 12,000 on this trade.

## Understanding the Output

### Visual Analysis (The Three Charts)

When the code finishes, it creates a window with three stacked graphs. Here's how to read each one:

**Chart 1: Portfolio Performance (Top Graph)**

This shows how your money grew (or shrank) over time.

```
What you see:
- Blue line: Your pairs trading strategy
- Orange dashed line: Buy and hold (just buying Stock A)
- Y-axis: Normalized returns (starts at 1.0 = your starting capital)
- X-axis: Time (dates)
```

**How to Read It:**
- If blue line is above orange line: Your strategy beat just holding the stock
- If blue line ends at 1.25: You made 25% profit
- If blue line ends at 0.85: You lost 15%
- Flat sections: No trades happening, just holding cash

**Chart 2: The Spread (Middle Graph)**

This shows the actual price difference between your two stocks over time, with markers showing when you traded.

```
What you see:
- Gray line: The spread value over time
- Black dashed line: Average spread (the "normal" level)
- Red dotted lines: +1 and -1 standard deviation boundaries
- Green/Red triangles: Your trades
```

**Understanding the Markers:**
- **Green triangle pointing UP (△)**: You entered a LONG spread trade here
  - You bet the spread would go up
  - Appeared when spread was very low
  
- **Red triangle pointing DOWN (▽)**: You entered a SHORT spread trade here
  - You bet the spread would go down
  - Appeared when spread was very high

- **Triangle with black border**: You EXITED the trade here
  - Flipped direction from entry (down if entry was up, up if entry was down)
  - This is where you took your profit (or loss)

**Example Reading:**
```
You see a green △ at Rs. -800, then a green ▽ with black border at Rs. -400

Translation: 
→ Entered LONG when spread was very negative (Rs. -800)
→ Exited when it recovered to Rs. -400
→ Made money because spread increased by Rs. 400
```

**Chart 3: Z-Score (Bottom Graph)**

This is the "standardized" version of the spread chart - easier to read because it's always on the same scale.

```
What you see:
- Purple line: Z-score over time
- Black solid line: Zero line (perfectly normal)
- Red dashed lines: +1 and -1 (entry thresholds)
- Orange dotted lines: +0.2 and -0.2 (exit thresholds)
- Green/Red triangles: Same trades as Chart 2
```

**Reading the Z-Score:**
```
Z-score = +2.0: Extremely high, happens rarely (2-3% of time)
Z-score = +1.0: High enough to trade (SHORT signal)
Z-score = +0.2: Slightly high, time to exit SHORT trades
Z-score = 0.0:  Perfectly normal, no action
Z-score = -0.2: Slightly low, time to exit LONG trades  
Z-score = -1.0: Low enough to trade (LONG signal)
Z-score = -2.0: Extremely low, happens rarely (2-3% of time)
```

**What Good Performance Looks Like:**
- Triangles appear near the +1/-1 lines (good entries)
- Exit triangles appear closer to zero (spread recovered)
- Multiple profitable trades (spread mean-reverts reliably)

**What Bad Performance Looks Like:**
- Triangles scattered randomly
- Spread keeps trending in one direction (no mean reversion)
- Many exits at even worse z-scores (losses)

### Trade Summary (Text Output)

After the charts, you'll see detailed text output in your terminal. Let's break down each section:

**Section 1: Overall Statistics**

```
======================================================================
TRADE SUMMARY
======================================================================

Total Trades: 15
Long Trades: 8
Short Trades: 7
```

**What This Tells You:**
- Total number of complete round-trip trades (entry + exit)
- How many times you bet spread would go UP (Long)
- How many times you bet spread would go DOWN (Short)

A good strategy should have both types of trades, showing the spread moved in both directions.

**Section 2: Win/Loss Breakdown**

```
Winning Trades: 9 (60.0%)
Losing Trades: 6 (40.0%)

Avg Win: ₹3,245.67
Largest Win: ₹8,120.34

Avg Loss: ₹-1,842.50
Largest Loss: ₹-4,328.00
```

**Understanding Win Rate:**
- 60% win rate means 6 out of every 10 trades made money
- Professional strategies typically have 50-65% win rates
- Below 45%: Strategy probably doesn't work
- Above 70%: Very good (or might be overfitted to historical data)

**Understanding Average Win vs Loss:**
Notice the average win (₹3,245) is bigger than the average loss (₹1,842). This is GOOD. Even with a 50% win rate, you'd make money because your wins are bigger than your losses.

**The Profit Factor:**
If you calculate: `(Total money from wins) / (Total money from losses)`, you get the profit factor.
- Profit Factor > 1.5: Excellent
- Profit Factor = 1.0 to 1.5: Acceptable  
- Profit Factor < 1.0: Losing strategy

**Section 3: Overall Performance**

```
Total Profit/Loss: ₹12,450.00
Return: 12.45%
```

**What This Means:**
You started with Rs. 1,00,000 and ended with Rs. 1,12,450. That's a 12.45% return.

**Judging Returns:**
Compare this to:
- Indian Fixed Deposit: ~6-7% per year (safe)
- Nifty 50 Index: ~10-12% per year (average market)
- Your Strategy: 12.45% (in this example)

But remember: These are backtested returns on historical data. Real trading is harder.

**Section 4: Individual Trade Details**

```
INDIVIDUAL TRADES:
----------------------------------------------------------------------

Trade #1 - LONG
  Entry: 2020-03-15 | Z-Score: -1.23
  Exit:  2020-04-02 | Z-Score: -0.18
  P&L: ₹2,847.50

Trade #2 - SHORT  
  Entry: 2020-05-20 | Z-Score: 1.45
  Exit:  2020-06-08 | Z-Score: 0.15
  P&L: ₹3,192.00
```

**Reading Each Trade:**

**Trade #1 Breakdown:**
- Type: LONG (we bet spread would increase)
- Entry Date: March 15, 2020
- Entry Z-Score: -1.23 (spread was LOW, below -1.0 threshold)
- Exit Date: April 2, 2020 (held for 18 days)
- Exit Z-Score: -0.18 (spread recovered, close to zero)
- Profit: Rs. 2,847.50 (green color in terminal)

**What Happened:** The spread was unusually low on March 15 (-1.23 standard deviations). We bought Stock A and sold Stock B. Over 18 days, the spread increased back toward normal (-0.18). We exited near the mean and made Rs. 2,847.50.

**Trade #2 Breakdown:**
- Type: SHORT (we bet spread would decrease)
- Entry: May 20, 2020, z-score +1.45 (spread was HIGH)
- Exit: June 8, 2020, z-score +0.15 (spread recovered downward)
- Profit: Rs. 3,192.00

**What Happened:** The spread was unusually high. We sold Stock A and bought Stock B. The spread decreased back to normal over 19 days, and we profited.

**Red Numbers (Losses):**
```
Trade #5 - LONG
  Entry: 2020-08-10 | Z-Score: -1.05
  Exit:  2020-08-25 | Z-Score: -1.42
  P&L: ₹-2,150.00
```

This trade LOST money because the spread didn't revert - it actually got MORE negative. This happens sometimes. The strategy isn't perfect.

Example output:
```
======================================================================
TRADE SUMMARY
======================================================================

Total Trades: 15
Long Trades: 8
Short Trades: 7

Winning Trades: 9 (60.0%)
Losing Trades: 6 (40.0%)

Avg Win: ₹3,245.67
Largest Win: ₹8,120.34

Total Profit/Loss: ₹12,450.00
Return: 12.45%
```

## Code Structure (What Each Function Does)

### Core Functions Explained

**1. fetch_history(tickers, start, end)**

**Purpose:** Downloads stock price data from the internet.

**What It Does:**
- Connects to Yahoo Finance for each stock ticker
- Downloads daily price data (Open, High, Low, Close, Volume)
- Cleans up any messy or missing data
- Returns a dictionary with all the data organized

**Example:**
```python
data = fetch_history(['RELIANCE.NS', 'TCS.NS'], '2020-01-01', '2025-01-01')
# Returns: {'RELIANCE.NS': DataFrame with prices, 'TCS.NS': DataFrame with prices}
```

**Why It's Important:** Without good quality historical data, the whole strategy falls apart. This function makes sure the data is clean and complete.

---

**2. find_cointegrated_pairs(data_dict, significance=0.05)**

**Purpose:** Finds which stock pairs have stable long-term relationships.

**What It Does:**
- Takes every possible pair combination of your stocks
- Runs the Augmented Dickey-Fuller test on each pair
- Tests if the "errors" between the two stocks are stationary
- Returns the best pairs ranked by p-value

**The Process:**
```
Input: 4 stocks (A, B, C, D)

Tests performed:
A vs B → p-value = 0.08 (too high, rejected)
A vs C → p-value = 0.02 (GOOD, accepted)
A vs D → p-value = 0.15 (rejected)
B vs C → p-value = 0.04 (GOOD, accepted)
B vs D → p-value = 0.22 (rejected)
C vs D → p-value = 0.31 (rejected)

Result: Returns [(A,C,0.02), (B,C,0.04)]
```

**Parameters:**
- `significance=0.05`: Only accept pairs with p-value below 5%
- `max_pairs=10`: Stop after finding 10 good pairs (saves time)

**Why It's Important:** This is the MOST crucial step. Trading non-cointegrated pairs is like gambling - the relationship might be random.

---

**3. compute_spread(x, y)**

**Purpose:** Calculates how far apart two stock prices are at any moment.

**What It Does:**
- Runs a linear regression (finds the best-fit line) between Stock A and Stock B
- Extracts the "beta" (hedge ratio) from the regression
- Calculates spread = Price_A - (beta × Price_B)
- Returns both the spread series and the beta value

**The Math:**
```python
# Regression equation: y = alpha + beta*x + error
# We want to find beta

Example:
Stock A = [100, 102, 101, 105]
Stock B = [50, 51, 50.5, 52.5]

Regression finds: beta = 2.0

Spread = [100 - 2.0*50, 102 - 2.0*51, 101 - 2.0*50.5, 105 - 2.0*52.5]
Spread = [0, 0, 0, 0]  ← Perfect cointegration!
```

**Why It's Important:** The hedge ratio ensures we're comparing apples to apples, not apples to oranges.

---

**4. train_spread_predictor(spread)**

**Purpose:** Trains a neural network to predict future spread movements (experimental).

**What It Does:**
- Takes the historical spread data
- Scales it to a standard range (neural networks work better this way)
- Creates "sequences" - windows of 20 days of spread history
- Trains an LSTM (Long Short-Term Memory) neural network
- Returns the trained model

**How LSTM Works (Simplified):**

Think of LSTM like studying for an exam. You don't just memorize today's lesson - you remember patterns from the past weeks and use them to predict what might appear on the test.

```
Input: [Spread Day 1, Day 2, Day 3, ... Day 20]
Output: Predicted spread on Day 21

The network learns patterns like:
"When spread is very negative for 5 days, it usually bounces back"
"When spread oscillates rapidly, a big move is coming"
```

**Network Architecture:**
```
Input: 20 days of spread data
    ↓
LSTM Layer (32 neurons) - Remembers patterns
    ↓  
Dropout (10%) - Prevents overfitting
    ↓
Dense Layer (16 neurons) - Processes information
    ↓
Output: Predicted spread value for tomorrow
```

**Why It's Experimental:** This feature is included but not actively used in trading decisions. It's there for future enhancement. The current strategy relies on z-scores, not predictions.

---

**5. backtest_pair(x, y, cash, entry_z=1.0, exit_z=0.2)**

**Purpose:** Simulates the entire trading strategy on historical data.

**What It Does:**
- Goes through every day in your historical data
- Checks if the z-score crosses entry thresholds (-1.0 or +1.0)
- Opens positions when signals appear
- Tracks the profit/loss while position is open
- Closes positions when z-score crosses exit thresholds (-0.2 or +0.2)
- Records every trade with entry/exit details
- Returns final portfolio value and all trade records

**The Day-by-Day Loop:**
```python
For each day:
    Calculate current z-score
    
    If no position and z-score > 1.0:
        → Enter SHORT spread (sell A, buy B)
        → Record entry details
    
    If no position and z-score < -1.0:
        → Enter LONG spread (buy A, sell B)
        → Record entry details
    
    If holding LONG and z-score > -0.2:
        → Exit LONG spread
        → Calculate profit
        → Record trade
        → Update cash
    
    If holding SHORT and z-score < 0.2:
        → Exit SHORT spread
        → Calculate profit
        → Record trade
        → Update cash
    
    Record portfolio value for this day
```

**Position Tracking:**
```python
position = 0   # No trade
position = 1   # Long spread (betting it will increase)
position = -1  # Short spread (betting it will decrease)
```

**Profit Calculation:**
```python
# For LONG spread:
profit = (exit_spread - entry_spread) × (position_size / volatility)

# For SHORT spread:
profit = (entry_spread - exit_spread) × (position_size / volatility)
```

**Why It's Important:** This function is the heart of the strategy. It shows you exactly how the strategy would have performed if you had traded it in the past.

---

**6. plot_performance(...)**

**Purpose:** Creates the three charts showing strategy performance.

**What It Does:**
- Sets up a figure with 3 subplots stacked vertically
- **Plot 1:** Draws portfolio value vs benchmark (buy-and-hold)
- **Plot 2:** Draws spread over time with trade markers
- **Plot 3:** Draws z-score over time with threshold lines
- Adds all the visual elements (markers, lines, labels)
- Displays the figure

**Marker Logic:**
```python
For each completed trade:
    If trade was LONG:
        Entry marker: Green triangle pointing UP (△)
        Exit marker: Green triangle pointing DOWN (▽)
    
    If trade was SHORT:
        Entry marker: Red triangle pointing DOWN (▽)
        Exit marker: Red triangle pointing UP (△)
```

**Why Different Markers:** The inverted triangle at exit makes it easy to visually match entry and exit points - they're always opposite directions.

---

**7. print_trade_summary(trades, initial_capital, final_capital)**

**Purpose:** Prints detailed statistics about all your trades.

**What It Does:**
- Converts trade list to a pandas DataFrame for easy analysis
- Calculates win rate, average win/loss, total P&L
- Prints nicely formatted summary with colors
- Lists every individual trade with details

**Calculations:**
```python
Total Trades = len(trades)
Winning Trades = trades where profit > 0
Losing Trades = trades where profit <= 0
Win Rate = (Winning Trades / Total Trades) × 100
Average Win = Sum of all wins / Number of wins
Average Loss = Sum of all losses / Number of losses
Total Return = (Final Capital / Initial Capital - 1) × 100
```

**Color Coding:**
- Green text: Positive numbers (wins, profits)
- Red text: Negative numbers (losses)
- Yellow text: Warnings
- Cyan text: Headers and important info

**Why It's Important:** Numbers on a chart are hard to interpret. This summary breaks everything down into clear statistics you can understand and compare.

## Strategy Parameters (Knobs You Can Turn)

### Tunable Variables

These are the settings you can change to modify how the strategy behaves. Think of them like difficulty settings in a video game.

**1. entry_z = 1.0**

**What It Controls:** How extreme the spread must be before you enter a trade.

**Current Setting:** 1.0 (one standard deviation from mean)

**If You Increase It (e.g., to 2.0):**
- Fewer trades (you wait for more extreme situations)
- Higher win rate (only trading very clear signals)
- Less total profit (missing smaller opportunities)

**If You Decrease It (e.g., to 0.5):**
- More trades (trading more often)
- Lower win rate (some trades might be false signals)
- Potentially higher total profit (catching more moves)

**Recommendation:** Keep between 0.8 and 1.5. Below 0.8 is too aggressive, above 2.0 is too conservative.

---

**2. exit_z = 0.2**

**What It Controls:** How close to "normal" the spread must get before you exit.

**Current Setting:** 0.2 (exit when spread is within 0.2 standard deviations of mean)

**If You Increase It (e.g., to 0.5):**
- Hold positions longer (wait for spread to get closer to mean)
- Risk of spread reversing before exit
- Potentially higher profits per trade

**If You Decrease It (e.g., to 0.1):**
- Exit trades faster
- Lock in profits sooner
- Less risk of giving back gains
- Lower profit per trade

**Recommendation:** Keep between 0.1 and 0.3. The default 0.2 is a good balance.

---

**3. size = cash * 0.1**

**What It Controls:** How much money you risk per trade.

**Current Setting:** 10% of total capital per trade

**If You Increase It (e.g., to 0.2 = 20%):**
- Bigger potential profits
- Bigger potential losses
- More aggressive (risky)
- Could lose money faster

**If You Decrease It (e.g., to 0.05 = 5%):**
- Smaller potential profits
- Smaller potential losses
- More conservative (safer)
- Takes longer to grow account

**Recommendation:** Never go above 15%. Professional traders often use 2-5%. The 10% default is moderate.

---

**4. lookback = 20**

**What It Controls:** How many days of history the LSTM neural network looks at.

**Current Setting:** 20 days (one trading month)

**If You Increase It (e.g., to 50):**
- Network considers longer-term patterns
- Slower to react to recent changes
- Might catch bigger trends

**If You Decrease It (e.g., to 10):**
- Network focuses on recent data only
- Faster reactions
- Might miss longer-term patterns

**Note:** This only affects the LSTM predictor, which is experimental and not used in actual trading decisions yet.

---

**5. significance = 0.05**

**What It Controls:** How strict the cointegration test is.

**Current Setting:** 0.05 (5% significance level - standard in statistics)

**If You Increase It (e.g., to 0.10):**
- More pairs pass the test
- Some pairs might not be truly cointegrated
- More trading opportunities but lower quality

**If You Decrease It (e.g., to 0.01):**
- Fewer pairs pass the test
- Only very strong relationships accepted
- Higher quality but fewer opportunities

**Recommendation:** Keep at 0.05. This is the standard used in academic research.

---

### How to Change Parameters

To modify these settings, edit the values in the code:

```python
# In the backtest_pair function:
def backtest_pair(x, y, cash=100000, entry_z=1.5, exit_z=0.15):  # Changed!
    # ... rest of code
```

Or in the main function:

```python
# In main():
portfolio, spread, z, beta, trades = backtest_pair(
    x, y, 
    cash,
    entry_z=1.2,   # Custom entry threshold
    exit_z=0.25    # Custom exit threshold
)
```

### Risk Management Explained

**What is Risk Management?**

Risk management is like wearing a seatbelt. You hope you never need it, but it protects you when things go wrong.

**Built-in Safety Features:**

**1. Position Sizing Limit (10% per trade)**

**Why It Helps:**
If you lose 5 trades in a row (worst case), you've only lost ~45% of your capital, not everything. You can still recover.

**The Math:**
```
Starting: Rs. 1,00,000
After Loss 1: Rs. 90,000 (lost 10%)
After Loss 2: Rs. 81,000 (lost 10% of 90k)
After Loss 3: Rs. 72,900
After Loss 4: Rs. 65,610
After Loss 5: Rs. 59,049

Still have 59% of original capital to trade with!
```

**2. Mean-Reversion Exit Triggers**

**Why It Helps:**
Instead of waiting for the spread to reverse completely, we exit when it gets close to normal. This prevents the "hope and pray" scenario where you watch profits turn into losses.

**Example:**
```
Entry: Spread at -1.2 standard deviations
Target would be: 0.0 (perfect mean reversion)
Exit trigger at: -0.2 (80% of the way there)

Result: You capture most of the move but don't risk waiting too long
```

**3. Market-Neutral Positioning**

**Why It Helps:**
Because you're long one stock and short another, you don't care if the overall market goes up or down.

**Example:**
```
Market crashes 10%:
Stock A drops 10%: You lose Rs. 5,000 on your long position
Stock B drops 10%: You gain Rs. 5,000 on your short position
Net effect: Rs. 0

Your profit only depends on the RELATIONSHIP between the stocks
```

**4. Cointegration Requirement**

**Why It Helps:**
By only trading pairs that pass the statistical test, we avoid random relationships that might be coincidental.

**Without This Rule:**
You might trade any two stocks that "look" related, but have no real connection. This is basically gambling.

**With This Rule:**
You only trade pairs with proven, stable relationships backed by statistical evidence.

---

### What Could Go Wrong? (Risk Factors)

**1. Relationship Breakdown**

**The Risk:**
The two stocks stop moving together. Maybe one company gets acquired, goes bankrupt, or changes its business model.

**Example:**
You're trading ICICI Bank vs HDFC Bank. Suddenly HDFC Bank merges with HDFC Ltd. The whole relationship changes overnight.

**Protection:**
Monitor news for your stocks. If something major happens, close your positions even if the technical signals haven't triggered.

**2. Trending Spreads (Non-Mean-Reverting)**

**The Risk:**
The spread starts trending in one direction instead of reverting to the mean.

**Example:**
```
Day 1: Z-score = -1.5 (enter LONG)
Day 5: Z-score = -1.8 (getting worse)
Day 10: Z-score = -2.3 (still worse)
Day 15: Z-score = -2.7 (losses mounting)
```

**Protection:**
Add a stop-loss. If z-score moves 0.5 in the wrong direction from entry, exit immediately.

**3. Low Liquidity**

**The Risk:**
When you try to buy or sell, there aren't enough buyers/sellers at good prices. You get terrible execution prices.

**Example:**
The strategy says "Buy 1,000 shares at Rs. 2,400."
But the market only has:
- 100 shares available at Rs. 2,400
- 200 shares at Rs. 2,405
- 700 shares at Rs. 2,412

You end up paying an average of Rs. 2,409, which ruins your profit.

**Protection:**
Only trade stocks with high daily volume (10 lakh+ shares traded per day).

**4. Transaction Costs Killing Returns**

**The Risk:**
Every trade costs money in brokerage fees. If you trade too often, fees eat all your profits.

**Example:**
```
Trade 1 Profit: Rs. 500
Brokerage: Rs. 40 (buying) + Rs. 40 (selling) = Rs. 80
Net: Rs. 420

Trade 2 Profit: Rs. 300
Brokerage: Rs. 80
Net: Rs. 220

Trade 3 Loss: Rs. -400
Brokerage: Rs. 80
Net: Rs. -480

Total: Rs. 400 gross profit → Rs. 160 net profit after fees
Fees ate 60% of profits!
```

**Protection:**
- Trade less frequently (use higher entry_z thresholds)
- Use discount brokers with low fees
- Account for fees in your backtest (this code doesn't do this yet)

## Limitations (What This Code Doesn't Do)

Understanding limitations is crucial. Here's what you need to know:

### 1. Transaction Costs Not Included

**The Problem:**
The backtest assumes you can buy and sell stocks for free. In reality, every trade costs money.

**Real-World Costs:**
```
Typical costs per trade in India:
- Brokerage: Rs. 20-40 per trade (discount brokers)
- STT (Securities Transaction Tax): 0.025% on sell side
- Exchange charges: ~0.003%
- GST: 18% on brokerage
- Stamp duty: 0.015% on buy side

Example:
Trade value: Rs. 10,000
Total costs: ~Rs. 50-80 per trade (both buy and sell)

If you make 20 trades:
Cost = Rs. 1,000-1,600 in fees
```

**Impact on Results:**
A strategy showing 15% returns might actually make only 10% after costs.

**How to Fix It:**
Manually subtract approximate costs from your backtest profits, or modify the code to include `transaction_cost` parameter.

---

### 2. Lookahead Bias

**The Problem:**
The code calculates spread statistics (mean and standard deviation) using the ENTIRE time period, including future data.

**Why This is Unrealistic:**
In real trading, you don't know what the spread will be next month or next year. But the backtest "peeks" at future data to calculate current z-scores.

**Example of the Bias:**
```
Trading on Jan 1, 2020:
Code uses: Mean spread from Jan 2020 to Dec 2024 (5 years)
Reality: You only know mean spread up to Dec 2019

The code has "future knowledge" which makes it look better than reality.
```

**How Bad Is It?**
If the spread trend changes significantly, this can overestimate returns by 20-50%.

**How to Fix It:**
Use a rolling window. Instead of:
```python
z = (spread - spread.mean()) / spread.std()
```

Use:
```python
z = (spread - spread.rolling(250).mean()) / spread.rolling(250).std()
```

This calculates mean and std using only the last 250 days (one trading year).

---

### 3. Market Regime Changes

**The Problem:**
Cointegration relationships can break down permanently.

**Why It Happens:**
- Company fundamentals change (new CEO, business model shift)
- Industry disruption (new technology, regulations)
- Mergers and acquisitions
- Major scandals or corporate governance issues

**Real Example:**
```
2015-2018: HDFC Bank and ICICI Bank are highly cointegrated
2018-2019: ICICI Bank has leadership crisis and scandals
Result: Relationship breaks down, pair trading fails

If you kept trading this pair, you'd lose money even though historical 
data showed strong cointegration.
```

**Warning Signs:**
- P-value increases over time (check monthly)
- Spread stops mean-reverting (trends for 30+ days)
- One stock has major news events

**How to Handle It:**
Re-test cointegration every month. If p-value rises above 0.10, stop trading that pair.

---

### 4. Liquidity Assumptions (Perfect Execution)

**The Problem:**
The code assumes you can buy or sell any quantity at the closing price. Real markets aren't that simple.

**Reality Checks:**

**Slippage:**
The difference between expected price and actual execution price.
```
You want to buy at: Rs. 2,400
You actually get: Rs. 2,403
Slippage: Rs. 3 per share

On 100 shares: Rs. 300 loss
Over 20 trades: Rs. 6,000 loss
```

**Market Impact:**
Large orders move the price against you.
```
Example: You want to buy 10,000 shares

Market depth:
1,000 shares @ Rs. 2,400
2,000 shares @ Rs. 2,401
3,000 shares @ Rs. 2,402
4,000 shares @ Rs. 2,404

Your average price: Rs. 2,402.30 (worse than Rs. 2,400)
```

**Bid-Ask Spread:**
The difference between buying and selling price.
```
Bid (you sell at): Rs. 2,399
Ask (you buy at): Rs. 2,401
Spread: Rs. 2

Every round trip (buy then sell): You lose Rs. 2 per share automatically
```

**How to Minimize:**
- Only trade highly liquid stocks (₹50+ crore daily volume)
- Use limit orders instead of market orders
- Split large orders into smaller chunks
- Trade during market hours with good liquidity (10:00 AM - 2:30 PM)

---

### 5. Overfitting Risk (The Neural Network)

**The Problem:**
The LSTM neural network might "memorize" historical patterns that don't repeat in the future.

**What Overfitting Looks Like:**
```
Backtest: 95% win rate, amazing returns
Real trading: 45% win rate, losing money

The network learned patterns specific to that exact historical data 
but can't generalize to new data.
```

**Why It Happens:**
- Training on too little data (less than 5 years)
- Network too complex (too many neurons)
- Testing on same data used for training
- Cherry-picking the best parameters

**Good News:**
The current strategy doesn't actually use the LSTM for trading decisions. It's just trained but not applied. So this isn't affecting your backtest yet.

**If You Want to Use the LSTM:**
1. Split data: 70% training, 30% testing (test on data NEVER seen in training)
2. Use simple networks (the current 32-neuron LSTM is reasonable)
3. Validate on multiple time periods
4. Check if it beats the simple z-score approach (often it won't)

---

### 6. No Stop-Losses

**The Problem:**
If a trade goes really wrong, the strategy just holds and hopes for mean reversion.

**Worst Case Scenario:**
```
Entry: Z-score = -1.2 (LONG spread)
Day 5: Z-score = -1.8 (down Rs. 2,000)
Day 10: Z-score = -2.5 (down Rs. 5,000)
Day 20: Z-score = -3.2 (down Rs. 8,000)

The spread never reverts. You just watch losses grow.
```

**Solution:**
Add a stop-loss rule:
```python
if abs(current_z - entry_z) > 1.0:
    # Z-score moved 1.0 in the wrong direction
    exit_trade()  # Take the loss and move on
```

This limits maximum loss per trade.

---

### 7. Single Pair Only

**The Problem:**
The code only trades the single best cointegrated pair.

**Why This is Risky:**
- If that one pair's relationship breaks, you're done
- Missing out on other good opportunities
- All eggs in one basket

**Professional Approach:**
Trade a portfolio of 5-10 cointegrated pairs simultaneously.

**Benefits:**
```
Pair 1: +5% return
Pair 2: -2% return (relationship broke)
Pair 3: +7% return
Pair 4: +3% return
Pair 5: -1% return

Average: +2.4% (still positive despite 2 failures)
```

**How to Implement:**
Instead of:
```python
pairs = find_cointegrated_pairs(data)
a, b, p = pairs[0]  # Take only the best pair
```

Do:
```python
pairs = find_cointegrated_pairs(data, max_pairs=5)
# Trade all 5 pairs simultaneously
```

Then divide your capital equally among pairs (Rs. 20,000 per pair if you have Rs. 1 lakh).

## Performance Notes

- Strategy performs best in range-bound markets with stable correlations
- Trending markets may trigger early exits or false signals
- Requires sufficient historical data (minimum 250 trading days)
- Win rates typically range from 55-65% for robust pairs

## Future Enhancements (How to Make It Better)

These are improvements you can make to the code to make it more realistic and profitable.

### 1. Real-Time Data Integration

**Current State:**
The code only works with historical data from Yahoo Finance. You run it once and see past results.

**Enhancement:**
Connect to live market data through broker APIs.

**Popular Indian Broker APIs:**
```python
# Zerodha Kite API
from kiteconnect import KiteConnect

# Upstox API
from upstox_api.api import Upstox

# 5Paisa API
from py5paisa import FivePaisaClient
```

**What You'd Add:**
```python
def get_live_price(ticker):
    # Connect to broker API
    # Get current price
    # Return price
    pass

def check_signals_live():
    while market_is_open():
        current_spread = compute_spread(...)
        current_z = calculate_z_score(...)
        
        if current_z > entry_threshold:
            place_order()
        
        time.sleep(60)  # Check every minute
```

**Benefits:**
- Trade automatically in real-time
- No manual execution needed
- Faster reaction to signals

---

### 2. Dynamic Position Sizing

**Current State:**
Always risk 10% per trade, regardless of market conditions.

**Enhancement:**
Adjust position size based on:
- Volatility (risk less when volatility is high)
- Account size (risk more as you grow capital)
- Confidence level (risk more on strong signals)

**Implementation:**
```python
def calculate_position_size(cash, volatility, z_score):
    # Base size
    base_size = cash * 0.10
    
    # Adjust for volatility
    # High volatility = smaller position
    vol_adjustment = 1.0 / (1.0 + volatility)
    
    # Adjust for signal strength
    # Stronger signal = larger position
    signal_strength = abs(z_score) - 1.0
    signal_adjustment = 1.0 + (0.2 * signal_strength)
    
    final_size = base_size * vol_adjustment * signal_adjustment
    
    # Never exceed 15% of capital
    return min(final_size, cash * 0.15)

# Example:
# Normal volatility, z-score = -1.5:
# Size = 10% × 1.0 × 1.1 = 11%

# High volatility, z-score = -1.5:
# Size = 10% × 0.7 × 1.1 = 7.7%

# Normal volatility, weak signal (z-score = -1.1):
# Size = 10% × 1.0 × 1.02 = 10.2%
```

**Benefits:**
- Better risk management
- Adapt to changing market conditions
- Potentially higher returns

---

### 3. Multiple Pair Portfolios

**Current State:**
Trade only the single best pair.

**Enhancement:**
Trade 5-10 pairs simultaneously for diversification.

**Implementation:**
```python
def trade_multiple_pairs(data, capital):
    # Find top 5 pairs
    pairs = find_cointegrated_pairs(data, max_pairs=5)
    
    # Divide capital equally
    capital_per_pair = capital / len(pairs)
    
    results = {}
    for pair in pairs:
        ticker_a, ticker_b, p_value = pair
        
        # Run backtest for this pair
        pnl, spread, z, beta, trades = backtest_pair(
            data[ticker_a], 
            data[ticker_b], 
            capital_per_pair
        )
        
        results[f"{ticker_a}-{ticker_b}"] = {
            'pnl': pnl[-1],
            'trades': len(trades),
            'win_rate': calculate_win_rate(trades)
        }
    
    return results
```

**Benefits:**
```
Risk Reduction:
If one pair fails, others might succeed

Portfolio Example:
Pair 1: +Rs. 8,000
Pair 2: -Rs. 2,000
Pair 3: +Rs. 5,000
Pair 4: +Rs. 3,000
Pair 5: -Rs. 1,000

Total: +Rs. 13,000 (diversification smooths results)
```

---

### 4. Machine Learning for Adaptive Thresholds

**Current State:**
Entry/exit thresholds are fixed (1.0 and 0.2).

**Enhancement:**
Use ML to learn optimal thresholds for each pair and adapt over time.

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

def train_threshold_model(spread_history, trades_history):
    features = []
    labels = []
    
    for trade in trades_history:
        # Features: volatility, trend, p-value, etc.
        features.append([
            spread_volatility,
            recent_trend,
            cointegration_strength
        ])
        
        # Label: was this a profitable trade?
        labels.append(1 if trade['profit'] > 0 else 0)
    
    model = RandomForestClassifier()
    model.fit(features, labels)
    
    return model

def get_adaptive_threshold(current_conditions, model):
    prediction = model.predict_proba(current_conditions)
    
    # If model is confident, use lower threshold
    # If model is uncertain, use higher threshold
    
    if prediction[1] > 0.7:  # High confidence
        return 0.8
    elif prediction[1] > 0.5:  # Medium confidence
        return 1.0
    else:  # Low confidence
        return 1.5
```

**Benefits:**
- Thresholds adapt to market conditions
- Trade more when conditions are favorable
- Trade less when conditions are poor

---

### 5. Out-of-Sample Testing Framework

**Current State:**
Test on same data used to find pairs (in-sample bias).

**Enhancement:**
Split data into training and testing periods.

**Implementation:**
```python
def walk_forward_test(data, window_size=252, test_size=63):
    """
    Walk-forward testing: train on 1 year, test on 3 months, 
    then roll forward
    """
    results = []
    
    for start in range(0, len(data) - window_size - test_size, test_size):
        # Training period
        train_start = start
        train_end = start + window_size
        
        # Testing period
        test_start = train_end
        test_end = test_start + test_size
        
        # Find pairs on training data
        train_data = {
            ticker: df.iloc[train_start:train_end] 
            for ticker, df in data.items()
        }
        pairs = find_cointegrated_pairs(train_data)
        
        # Test on unseen data
        test_data = {
            ticker: df.iloc[test_start:test_end] 
            for ticker, df in data.items()
        }
        test_results = backtest_pair(test_data, pairs[0])
        
        results.append(test_results)
    
    return results

# Example timeline:
# Train: Jan 2020 - Dec 2020 → Find best pairs
# Test: Jan 2021 - Mar 2021 → Trade those pairs
# Train: Apr 2020 - Mar 2021 → Find new best pairs
# Test: Apr 2021 - Jun 2021 → Trade new pairs
# ... and so on
```

**Benefits:**
- More realistic performance estimates
- Catches overfitting
- Shows if strategy actually works on new data

---

### 6. Transaction Cost Modeling

**Current State:**
Assumes free trading.

**Enhancement:**
Deduct realistic costs from each trade.

**Implementation:**
```python
def calculate_transaction_costs(trade_value, trade_type):
    """
    Calculate realistic costs for Indian markets
    """
    # Brokerage (discount broker)
    brokerage = min(20, trade_value * 0.0003)
    
    # STT (only on sell side for delivery)
    stt = trade_value * 0.001 if trade_type == 'sell' else 0
    
    # Exchange charges
    exchange = trade_value * 0.0000325
    
    # Clearing charges
    clearing = trade_value * 0.00002
    
    # GST (18% on brokerage + exchange + clearing)
    gst = (brokerage + exchange + clearing) * 0.18
    
    # SEBI charges
    sebi = trade_value * 0.000001
    
    # Stamp duty (on buy side)
    stamp = trade_value * 0.00015 if trade_type == 'buy' else 0
    
    total = brokerage + stt + exchange + clearing + gst + sebi + stamp
    
    return total

# Use in backtest:
def backtest_with_costs(x, y, cash):
    # ... existing code ...
    
    # When entering trade
    entry_cost = calculate_transaction_costs(trade_value, 'buy')
    capital -= entry_cost
    
    # When exiting trade
    exit_cost = calculate_transaction_costs(trade_value, 'sell')
    profit -= exit_cost
    
    # ... rest of code ...
```

**Example Impact:**
```
Without costs: +15% annual return
With costs: +11% annual return

Difference: 4% eaten by fees
```

---

### 7. Stop-Loss Implementation

**Current State:**
Trades can lose unlimited amounts while waiting for mean reversion.

**Enhancement:**
Add stop-loss to limit maximum loss per trade.

**Implementation:**
```python
def backtest_with_stoploss(x, y, cash, stop_loss_z=1.0):
    # ... existing code ...
    
    for i in range(1, len(z)):
        # Check stop-loss for open positions
        if position == 1:  # Long spread
            # If z-score moved 1.0 in wrong direction, exit
            if z.iloc[i] < (entry_z - stop_loss_z):
                # Stop-loss hit
                loss = (spread.iloc[i] - entry_spread) * size
                capital += loss  # Will be negative
                position = 0
                trades.append({'type': 'STOP_LOSS', ...})
        
        elif position == -1:  # Short spread
            if z.iloc[i] > (entry_z + stop_loss_z):
                # Stop-loss hit
                loss = (entry_spread - spread.iloc[i]) * size
                capital += loss  # Will be negative
                position = 0
                trades.append({'type': 'STOP_LOSS', ...})
        
        # ... rest of existing logic ...
```

**Benefits:**
```
Without stop-loss:
Worst trade: -Rs. 15,000 loss (waited 90 days for mean reversion)

With stop-loss (1.0 z-score):
Worst trade: -Rs. 5,000 loss (exited after 10 days)

Risk controlled: Max loss limited to 5% of capital
```

---

### 8. Performance Dashboard

**Current State:**
Just prints numbers to terminal.

**Enhancement:**
Create interactive dashboard with key metrics.

**Tools to Use:**
```python
import streamlit as st
import plotly.graph_objects as go

def create_dashboard(results):
    st.title("Pairs Trading Dashboard")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{results['return']:.2f}%")
    col2.metric("Win Rate", f"{results['win_rate']:.1f}%")
    col3.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
    
    # Interactive equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results['dates'],
        y=results['portfolio_values'],
        name='Portfolio'
    ))
    st.plotly_chart(fig)
    
    # Trade table
    st.dataframe(results['trades'])
```

**Benefits:**
- Easy to visualize performance
- Share results with others
- Track multiple strategies

## Disclaimer (Read This Carefully)

**Educational Purpose Only**

This code is designed for learning and educational purposes. It demonstrates concepts in quantitative finance, statistical arbitrage, and algorithmic trading. It is NOT:
- Investment advice
- A guaranteed money-making system
- Ready for live trading without modifications
- Endorsed by any financial institution

**Past Performance ≠ Future Results**

The backtest shows how the strategy would have performed on historical data. This does NOT mean it will work in the future because:

```
Historical data tells you what ALREADY happened
Future markets might behave completely differently

Example:
Strategy made 20% in 2020-2024 backtest
Doesn't mean it will make 20% in 2025-2029

Markets change, relationships break, what worked before might fail
```

**Risk of Loss**

Trading involves substantial risk. You can lose some or ALL of your invested capital:
- Bad trades can lose money quickly
- Leverage amplifies losses
- Market crashes can exceed stop-losses
- Technical failures can cause unintended positions
- Slippage and costs reduce returns

**Statistics:**
```
Professional fact: 80-90% of retail traders lose money
Not because they're dumb, but because:
- Underestimating transaction costs
- Poor risk management
- Emotional decision-making
- Insufficient capital
- Unrealistic expectations
```

**Before Trading Real Money**

1. **Paper Trade for 6+ Months**
   - Use a simulator or paper trading account
   - Track results as if they were real
   - Understand the emotional aspect
   - Refine your strategy

2. **Start Small**
   - Begin with money you can afford to lose
   - Never trade with:
     - Emergency funds
     - Loan money
     - Money needed for expenses
     - Money that would stress you if lost

3. **Understand All Costs**
   - Brokerage fees
   - Taxes (short-term capital gains = 15%)
   - Exchange charges
   - Time investment

4. **Have Realistic Goals**
   ```
   Realistic: 10-15% annual returns
   Unrealistic: "Double my money in 3 months"
   
   If someone promises guaranteed high returns, it's a scam.
   ```

**Technical Disclaimers**

**No Warranty:**
This code is provided "as is" without any warranty. The authors are not responsible for:
- Bugs or errors in the code
- Financial losses from using this code
- Data accuracy issues
- Missed trades or incorrect signals
- Any damages arising from use

**Not Financial Advice:**
Nothing in this README constitutes financial, investment, legal, or tax advice. Consult qualified professionals:
- Financial advisor for investment decisions
- Tax consultant for tax implications
- Lawyer for legal questions

**Regulatory Compliance:**
Check your local regulations:
- SEBI regulations in India
- Some strategies might require special licenses
- Tax reporting requirements
- Anti-money laundering rules

**Data Usage:**
- Yahoo Finance data is for personal use
- Check yfinance terms of service
- Don't redistribute data commercially
- Real-time data might require paid subscriptions

---

## How to Use This Responsibly

**Step 1: Learn the Concepts**
```
Spend 2-3 weeks understanding:
- What is cointegration?
- Why does mean reversion work?
- What are the risks?
- How do statistics work in finance?

Resources:
- "Algorithmic Trading" by Ernest Chan
- Investopedia articles on pairs trading
- YouTube tutorials on statistical arbitrage
```

**Step 2: Backtest Thoroughly**
```
Don't just run once and trust results:
- Test on different time periods (2020, 2021, 2022, etc.)
- Test on different sectors (banking, IT, pharma)
- Test with different parameters (entry_z = 0.5, 1.0, 1.5, 2.0)
- Calculate worst-case scenarios

If strategy fails in ANY period, understand why before proceeding
```

**Step 3: Paper Trade**
```
Use broker's paper trading platform:
- Zerodha Kite has paper trading
- Upstox has simulation mode
- Sharekhan has practice account

Trade for 6 months minimum:
- Follow every signal
- Track emotions
- Document mistakes
- Calculate real costs
- See if you can follow the strategy consistently
```

**Step 4: Start Tiny (If You Proceed)**
```
First real trade:
- Use 1-2% of total capital
- Trade for 3 months
- If profitable, gradually increase to 5%
- Never go above 10-15% of total capital in pairs trading

Example progression:
Month 1-3: Rs. 10,000 (2% of 5 lakh capital)
Month 4-6: Rs. 25,000 (5% if doing well)
Month 7+: Rs. 50,000 (10% max)
```

**Step 5: Continuous Monitoring**
```
Weekly checks:
- Win rate still above 50%?
- P-values still below 0.05?
- Any major news about your stocks?
- Are costs eating profits?

Monthly review:
- Compare returns to benchmark
- Calculate Sharpe ratio
- Review all trades
- Adjust parameters if needed

Quarterly deep-dive:
- Is strategy still working?
- Should you find new pairs?
- Has market regime changed?
- Time to pause or continue?
```

**Red Flags to Stop Trading:**
```
STOP IMMEDIATELY if:
- 3 consecutive months of losses
- Win rate drops below 40%
- Drawdown exceeds 25%
- P-value rises above 0.10
- You're feeling emotional stress
- You're deviating from the plan

Take a break, analyze what went wrong, only restart after fixing issues
```

---

## Questions to Ask Yourself

Before using this code with real money:

1. **Do I understand how it works?**
   - Can you explain cointegration to a friend?
   - Do you know what a z-score is?
   - Can you interpret the charts?

2. **Can I afford to lose this money?**
   - Would losing 25% affect your life?
   - Is this money earmarked for anything important?
   - Do you have an emergency fund separate from this?

3. **Do I have realistic expectations?**
   - Are you expecting 50% annual returns? (unrealistic)
   - Are you prepared for 10-15% drawdowns?
   - Can you handle 3-6 months of no profits?

4. **Am I emotionally ready?**
   - Can you follow rules without emotion?
   - Will you panic during a losing streak?
   - Can you avoid revenge trading after losses?

5. **Do I have enough time?**
   - Can you monitor positions daily?
   - Can you react to signals promptly?
   - Do you have time for monthly reviews?

**If you answered "no" or "unsure" to ANY of these, DO NOT trade with real money yet.**

---

## Final Words

Pairs trading is a sophisticated strategy used by hedge funds and professional traders. It requires:
- Strong statistical knowledge
- Disciplined execution
- Risk management skills
- Emotional control
- Sufficient capital

This code gives you the tools, but success depends on:
- Your understanding
- Your discipline
- Your risk management
- Market conditions
- Luck (yes, luck is always a factor)

**The best way to use this code:**
1. Learn from it
2. Understand the concepts
3. Experiment with parameters
4. Test extensively before going live
5. Start small if you proceed
6. Be prepared to lose money while learning
7. Treat it as an education expense, not guaranteed income

Good luck, trade safely, and never stop learning!

## Contributing

Contributions are welcome. Please submit pull requests with clear descriptions of changes and appropriate test coverage.

## Contact

For questions or suggestions, please open an issue on the repository.
