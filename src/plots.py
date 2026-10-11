"""Graphiques du backtest (matplotlib, sauvegarde en PNG)."""

import os

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre, sauvegarde directe en fichier
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def plot_backtest(bt, symbol="BTC/USDT", strategy_name="SMA 20/50"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # 1. Prix + SMA + zones en position
    ax = axes[0]
    ax.plot(bt.index, bt["close"], label="Prix", color="black", linewidth=1)
    ax.plot(bt.index, bt["sma_short"], label="SMA courte", color="tab:blue", linewidth=1)
    ax.plot(bt.index, bt["sma_long"], label="SMA longue", color="tab:orange", linewidth=1)
    ax.fill_between(
        bt.index, bt["close"].min(), bt["close"].max(),
        where=bt["position"] > 0, color="green", alpha=0.08, label="En position",
    )
    ax.set_yscale("log")
    ax.set_title(f"{symbol} - Prix et signaux ({strategy_name})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 2. Equity curve vs buy & hold
    ax = axes[1]
    ax.plot(bt.index, bt["equity"], label="Strategie", color="tab:green")
    ax.plot(bt.index, bt["buy_hold"], label="Buy & Hold", color="gray", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Equity curve (capital initial 10 000 $)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 3. Drawdown
    ax = axes[2]
    ax.fill_between(bt.index, bt["drawdown"] * 100, 0, color="tab:red", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{symbol.replace('/', '_')}_backtest.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path