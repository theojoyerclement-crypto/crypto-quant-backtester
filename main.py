"""Point d'entree : backtest de plusieurs strategies sur BTC et ETH, comparaison."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd

from backtest import compute_metrics, run_backtest
from data_loader import load_csv
from plots import plot_backtest, plot_comparison
from strategy import rsi_meanreversion, sma_crossover

STRATEGIES = {
    "SMA 20/50": lambda df: sma_crossover(df, short=20, long=50),
    "RSI 14 (30/70)": lambda df: rsi_meanreversion(df, period=14, oversold=30, overbought=70),
}


def main():
    symbols = ["BTC/USDT", "ETH/USDT"]
    for symbol in symbols:
        print(f"\n########## {symbol} ##########")
        df = load_csv(symbol, "1d")

        results = {}
        rows = {}
        for name, strategy in STRATEGIES.items():
            bt = run_backtest(strategy(df), fee=0.001)
            results[name] = bt
            rows[name] = compute_metrics(bt)
            slug = name.split(" ")[0] + "_" + symbol.replace("/", "_")
            plot_backtest(bt, symbol, name, filename=f"{slug}_backtest.png")

        table = pd.DataFrame(rows)
        print(table.to_string())
        path = plot_comparison(results, symbol)
        print(f"Comparaison sauvegardee : {path}")


if __name__ == "__main__":
    main()
