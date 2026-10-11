"""Moteur de backtest : equity curve et métriques de performance."""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365  # crypto = marché ouvert 7j/7


def run_backtest(df, fee=0.001, initial_capital=10_000):
    """Calcule la performance de la stratégie à partir de la colonne 'signal'.

    - Le signal est décalé d'1 période (pas de lookahead bias).
    - Des frais sont prélevés à chaque changement de position.
    """
    out = df.copy()
    out["returns"] = out["close"].pct_change().fillna(0)
    out["position"] = out["signal"].shift(1).fillna(0)
    out["trade"] = out["position"].diff().abs().fillna(0)
    out["strategy_returns"] = out["position"] * out["returns"] - out["trade"] * fee
    out["equity"] = initial_capital * (1 + out["strategy_returns"]).cumprod()
    out["buy_hold"] = initial_capital * (1 + out["returns"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1
    return out


def compute_metrics(bt):
    """Métriques clés : rendement, Sharpe, max drawdown, win rate..."""
    n = len(bt)
    years = n / PERIODS_PER_YEAR
    total_return = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1
    bh_return = bt["buy_hold"].iloc[-1] / bt["buy_hold"].iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1

    std = bt["strategy_returns"].std()
    sharpe = bt["strategy_returns"].mean() / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else 0.0
    max_dd = bt["drawdown"].min()

    # Win rate par trade : un trade = une séquence continue en position
    in_pos = bt["position"] > 0
    trade_id = (in_pos != in_pos.shift()).cumsum()
    trade_returns = (
        (1 + bt.loc[in_pos, "strategy_returns"]).groupby(trade_id[in_pos]).prod() - 1
    )
    n_trades = len(trade_returns)
    win_rate = (trade_returns > 0).mean() if n_trades else 0.0

    return {
        "Rendement total strategie": total_return,
        "Rendement total buy & hold": bh_return,
        "CAGR (annualise)": cagr,
        "Sharpe ratio": sharpe,
        "Max drawdown": max_dd,
        "Nombre de trades": n_trades,
        "Win rate": win_rate,
        "Temps en position": in_pos.mean(),
    }


def print_metrics(metrics):
    print("\n=== RESULTATS DU BACKTEST ===")
    for name, val in metrics.items():
        if name in ("Sharpe ratio",):
            print(f"{name:<30} {val:>10.2f}")
        elif name == "Nombre de trades":
            print(f"{name:<30} {val:>10d}")
        else:
            print(f"{name:<30} {val:>10.2%}")