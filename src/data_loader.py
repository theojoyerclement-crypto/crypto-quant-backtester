"""Téléchargement de données historiques OHLCV depuis Binance via ccxt."""

import os
import sys
import time

import ccxt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1d", since="2020-01-01", limit=1000):
    """Télécharge toutes les bougies depuis une date donnée (pagination incluse)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    now_ms = exchange.milliseconds()
    all_candles = []

    while since_ms < now_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        print(f"  {symbol} : {len(all_candles)} bougies récupérées...")
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").set_index("date").drop(columns="timestamp")
    return df


def save_csv(df, symbol, timeframe):
    os.makedirs(DATA_DIR, exist_ok=True)
    name = symbol.replace("/", "_") + f"_{timeframe}.csv"
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path)
    return path


def load_csv(symbol="BTC/USDT", timeframe="1d"):
    """Recharge un CSV déjà téléchargé (utilisé par le backtester)."""
    name = symbol.replace("/", "_") + f"_{timeframe}.csv"
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path, index_col="date", parse_dates=True)


if __name__ == "__main__":
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1d"
    for sym in symbols:
        print(f"Téléchargement de {sym} ({timeframe})...")
        try:
            data = fetch_ohlcv(sym, timeframe)
        except Exception as e:
            print(f"Erreur pour {sym} : {e}")
            sys.exit(1)
        path = save_csv(data, sym, timeframe)
        print(f"OK : {len(data)} lignes, du {data.index[0].date()} au {data.index[-1].date()}")
        print(f"Sauvegardé : {path}\n")