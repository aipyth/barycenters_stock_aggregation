import pandas as pd
import os

data_folder = 'stocks/'
stock_data = {}

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        filename = file.split('.')[:-1]
        filename = '.'.join(filename)
        symbol = filename.split('_')[0]
        file_path = os.path.join(data_folder, file)
        stock_data[symbol] = pd.read_csv(file_path, index_col=0, parse_dates=True)

print(stock_data['AAPL'].info())

# for symbol, df in stock_data.items():
#     df.rename(columns={
#         'Close/Last': 'close',
#         'Volume': 'volume',
#         'Open': 'open',
#         'High': 'high',
#         'Low': 'low',
#     }, inplace=True)
#     df['close'] = pd.to_numeric(df['close'].str.replace('$', ''))
#     df['open'] = pd.to_numeric(df['open'].str.replace('$', ''))
#     df['high'] = pd.to_numeric(df['high'].str.replace('$', ''))
#     df['low'] = pd.to_numeric(df['low'].str.replace('$', ''))

#     file_path = os.path.join(data_folder, symbol + '.csv')
#     df.to_csv(file_path)


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

for symbol, df in stock_data.items():
    df = calculate_returns(df)

    file_path = os.path.join(data_folder, symbol + '.csv')
    df.to_csv(file_path)