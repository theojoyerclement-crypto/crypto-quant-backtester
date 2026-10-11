"""Signaux live : recupere les dernieres bougies et affiche la position de chaque strategie."""

import os
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import rsi_meanreversion, sma_crossover

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1d"
LIMIT = 300  # assez de bougies pour chauffer SMA 50 et RSI 14
REFRESH_SECONDS = 300  # utilise avec --loop


def fetch_recent(exchange, symbol):
    raw = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def describe_signals(df):
    """Retourne le dernier prix et la position de chaque strategie."""
    price = df["close"].iloc[-1]

    sma_df = sma_crossover(df.copy(), short=20, long=50)
    sma_pos = "LONG" if sma_df["signal"].iloc[-1] == 1 else "HORS MARCHE"
    sma_detail = f"SMA20 {sma_df['sma_short'].iloc[-1]:,.0f} / SMA50 {sma_df['sma_long'].iloc[-1]:,.0f}"

    rsi_df = rsi_meanreversion(df.copy())
    rsi_pos = "LONG" if rsi_df["signal"].iloc[-1] == 1 else "HORS MARCHE"
    rsi_value = rsi_df["rsi"].iloc[-1]

    return price, sma_pos, sma_detail, rsi_pos, rsi_value


def print_signals(exchange):
    print(f"\n=== SIGNAUX LIVE ({datetime.now():%Y-%m-%d %H:%M}) ===")
    for symbol in SYMBOLS:
        df = fetch_recent(exchange, symbol)
        price, sma_pos, sma_detail, rsi_pos, rsi_value = describe_signals(df)
        print(f"\n{symbol}  prix : {price:,.2f}")
        print(f"  SMA 20/50 : {sma_pos:<12} ({sma_detail})")
        print(f"  RSI 14    : {rsi_pos:<12} (RSI = {rsi_value:.1f})")
    print("\nNote : bougie du jour en cours, le signal peut encore changer a la cloture.")


def main():
    exchange = ccxt.binance({"enableRateLimit": True})
    loop = "--loop" in sys.argv
    while True:
        print_signals(exchange)
        if not loop:
            break
        print(f"Prochain rafraichissement dans {REFRESH_SECONDS // 60} min (Ctrl+C pour arreter)")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()