"""Stratégies de trading : génération de signaux à partir des prix."""

import pandas as pd


def sma_crossover(df, short=20, long=50):
    """Croisement de moyennes mobiles.

    signal = 1 (long) quand SMA courte > SMA longue, sinon 0 (hors marché).
    """
    out = df.copy()
    out["sma_short"] = out["close"].rolling(short).mean()
    out["sma_long"] = out["close"].rolling(long).mean()
    out["signal"] = (out["sma_short"] > out["sma_long"]).astype(int)
    # Pas de signal tant que les moyennes ne sont pas calculables
    out.loc[out["sma_long"].isna(), "signal"] = 0
    return out