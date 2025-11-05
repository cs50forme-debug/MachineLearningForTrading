import pandas as pd
import ta

df = pd.read_csv("BSEData.csv")

# Normalize column names
df.columns = [c.lower().strip() for c in df.columns]
rename_map = {'adj close': 'adj_close'}
df.rename(columns=rename_map, inplace=True)

# Convert numeric columns safely
for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs caused by bad data
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).sort_values('date').reset_index(drop=True)

# Momentum
df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
stoch = ta.momentum.StochRSIIndicator(df['close'])
df['stoch_rsi'] = stoch.stochrsi()
df['stoch_rsi_k'] = stoch.stochrsi_k()
df['stoch_rsi_d'] = stoch.stochrsi_d()
df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()
df['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
df['roc'] = ta.momentum.ROCIndicator(df['close'], 12).roc()
df['willr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

# Trend
df['sma_10'] = ta.trend.SMAIndicator(df['close'], 10).sma_indicator()
df['sma_50'] = ta.trend.SMAIndicator(df['close'], 50).sma_indicator()
df['ema_20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
df['ema_100'] = ta.trend.EMAIndicator(df['close'], 100).ema_indicator()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

# Volatility
bb = ta.volatility.BollingerBands(df['close'])
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
df['bb_width'] = df['bb_high'] - df['bb_low']
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

# Volume
df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

df = df.dropna().reset_index(drop=True)
df.to_csv("BSEDataWithIndicators.csv", index=False)

print(f"Saved: BSEDataWithIndicators.csv | Rows: {df.shape[0]} | Cols: {df.shape[1]}")

