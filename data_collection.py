import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
assert API_KEY is not None, "No API KEY specified"

directory = 'data/'
symbols = [
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOGL',
    'JPM',
    'JNJ',
    'XOM',
    'PG',
    'TSLA',
    'BRK-B',
    'IBM'
]

os.makedirs(directory, exist_ok=True)

def calculate_returns(data):
    data.index = pd.to_datetime(data.index)
    
    # Fill any missing close prices (carry forward last value)
    # data['close'] = data['close'].fillna(method='ffill')
    
    # Daily returns
    data['daily_return'] = data['close'].pct_change()
    
    # Weekly returns (based on a 5-trading-day window)
    data['weekly_return'] = data['close'].pct_change(periods=5)
    
    # Monthly returns (based on first and last trading day of each month)
    data['month'] = data.index.to_period('M')  # Add a column for grouping by month
    monthly_returns = (
        data.groupby('month')['close']
        .agg(['first', 'last'])  # Get the first and last closing price in each month
        .assign(monthly_return=lambda x: (x['last'] - x['first']) / x['first'])
    )
    # Merge monthly returns back into the original dataframe
    data = data.merge(monthly_returns[['monthly_return']], left_on='month', right_index=True, how='left')
    
    return data

# Download and process data
for symbol in symbols:
    print(f"Processing {symbol}...")
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    
    # Download daily data
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    
    data.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    }, inplace=True)
    
    # Calculate returns
    data = calculate_returns(data)
    
    # Save to CSV
    file_path = os.path.join(directory, f"{symbol}.csv")
    data.to_csv(file_path)
    print(f"Data for {symbol} saved to {file_path}")
