import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def create_features(df):
    """
    Crée des features ML à partir des données OHLCV
    """
    df = df.copy()

    # === TREND FEATURES ===
    df['SMA_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()

    # SMA Crossover signals
    df['SMA_10_20_cross'] = (df['SMA_10'] > df['SMA_20']).astype(int)
    df['SMA_20_50_cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)

    # === MOMENTUM FEATURES ===
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()

    # === VOLATILITY FEATURES ===
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # === PRICE ACTION ===
    df['returns'] = df['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # === TARGET ===
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # ✅ Pandas 2.0+ - bfill().ffill().fillna(0)
    df = df.bfill().ffill().fillna(0)
    
    return df


def get_feature_columns():
    return [
        'SMA_10', 'SMA_20', 'SMA_50',
        'SMA_10_20_cross', 'SMA_20_50_cross',
        'RSI',
        'volatility', 'volume_ratio',
        'returns', 'high_low_ratio', 'close_position'
    ]