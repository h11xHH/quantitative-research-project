'''
如果放量上涨就跟涨，持有一段时间后卖出
'''
import pandas as pd
import numpy as np


#########################
file_name = "BTCUSDT_1h_kline.csv"
hold_period = 8
rise_window = 60
volume_window = 60*4
PER = 24 * 365
#########################
def rolling_rank(series):
    """返回窗口中最后一个值在该窗口内的百分比排名"""
    return series.rank(pct=True).iloc[-1]

def momentum_detect(data, rise_window=30, volume_window = 60*4):
    rise_up = data['close']\
        .rolling(window=rise_window, min_periods=1)\
        .apply(rolling_rank, raw=False)
    rise_up = rise_up.apply(lambda x: 1 if x > 0.99 else 0)

    mean_volume = data['volume'].rolling(window=volume_window, min_periods=1).mean()
    std_volume = data['volume'].rolling(window=volume_window, min_periods=1).std()
    volume_up = (data['volume'] - mean_volume) / (std_volume + 1e-9)
    volume_up = volume_up.apply(lambda x: 1 if x > 3 else 0)

    signal = rise_up * volume_up
    return signal

print("reading data...")
data = pd.read_csv(file_name, index_col=0)
print(f"hold_period={hold_period}, rise_window={rise_window}, volume_window={volume_window}")

data['signal'] = momentum_detect(data, rise_window=rise_window, volume_window=volume_window)
data['signal'] = data['signal'].shift(1)

print("\n基准收益")
ret = np.log(data['close'] / data['open'])
print(f"年化收益: {ret.mean() * PER*100:.2f}%")
print(f"夏普比率: {ret.mean() / ret.std() * np.sqrt(PER):.2f}")

target_position = data['signal'].shift(1).rolling(window=hold_period).max()
ret = target_position * np.log(data['close'] / data['open'])
data['ret'] = ret
annual_retrun = ret.mean() * PER
print(f"\n年化收益: {annual_retrun*100:.2f}%")
sharpe_ratio = ret.mean() / ret.std() * np.sqrt(PER)
print(f"夏普比率: {sharpe_ratio:.2f}")
data.to_csv("BTCUSDT_1m_kline_signal.csv")