"""Point d'entrée : charge les données, backtest la stratégie, affiche et sauvegarde les résultats."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from backtest import compute_metrics, print_metrics, run_backtest
from data_loader import load_csv
from plots import plot_backtest
from strategy import sma_crossover


def main():
    symbols = ["BTC/USDT", "ETH/USDT"]
    for symbol in symbols:
        print(f"\n########## {symbol} ##########")
        df = load_csv(symbol, "1d")
        signals = sma_crossover(df, short=20, long=50)
        bt = run_backtest(signals, fee=0.001)
        print_metrics(compute_metrics(bt))
        path = plot_backtest(bt, symbol, "SMA 20/50")
        print(f"Graphique sauvegarde : {path}")


if __name__ == "__main__":
    main()