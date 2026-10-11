"""Optimisation train/test des parametres SMA : evite l'overfitting."""

import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import compute_metrics, run_backtest
from data_loader import load_csv
from strategy import sma_crossover

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
SHORTS = range(10, 55, 5)    # 10, 15, ..., 50
LONGS = range(50, 225, 25)   # 50, 75, ..., 200


def evaluate(df, short, long):
    """Backtest sur toute la serie, retourne le resultat complet."""
    return run_backtest(sma_crossover(df, short=short, long=long), fee=0.001)


def optimize(symbol):
    df = load_csv(symbol, "1d")

    # 1. Recherche du meilleur couple (short, long) sur le TRAIN uniquement
    results = []
    for short, long in itertools.product(SHORTS, LONGS):
        if short >= long:
            continue
        bt = evaluate(df, short, long)
        train = bt.loc[:TRAIN_END]
        m = compute_metrics(train)
        results.append({"short": short, "long": long, "sharpe": m["Sharpe ratio"]})

    grid = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    best = grid.iloc[0]
    best_short, best_long = int(best["short"]), int(best["long"])

    # 2. Evaluation des parametres figes sur train ET test
    bt = evaluate(df, best_short, best_long)
    train_m = compute_metrics(bt.loc[:TRAIN_END])
    test_m = compute_metrics(bt.loc[TEST_START:])

    # 3. Reference : parametres par defaut (20/50) sur le test
    bt_def = evaluate(df, 20, 50)
    default_test = compute_metrics(bt_def.loc[TEST_START:])

    return grid, (best_short, best_long), train_m, test_m, default_test


def main():
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        print(f"\n########## {symbol} ##########")
        grid, (s, l), train_m, test_m, default_test = optimize(symbol)

        print(f"\nTop 5 sur le train (2020-2023) :")
        print(grid.head(5).to_string(index=False))
        print(f"\nMeilleurs parametres : SMA {s}/{l}")

        table = pd.DataFrame({
            f"TRAIN SMA {s}/{l}": train_m,
            f"TEST SMA {s}/{l}": test_m,
            "TEST SMA 20/50": default_test,
        })
        pd.options.display.float_format = "{:.3f}".format
        print("\n" + table.to_string())

        drop = test_m["Sharpe ratio"] - train_m["Sharpe ratio"]
        print(f"\nVariation du Sharpe train -> test : {drop:+.2f}")
        if drop < -0.3:
            print("=> Signe d'overfitting : les parametres optimises generalisent mal.")
        else:
            print("=> Pas de degradation majeure hors echantillon.")


if __name__ == "__main__":
    main()