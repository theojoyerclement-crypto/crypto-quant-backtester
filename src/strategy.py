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

def rsi_meanreversion(df, period=14, oversold=30, overbought=70):
    """Strategie RSI : on achete quand le marche est survendu,
    on sort quand il est surachete."""
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Lissage de Wilder (standard pour le RSI)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Signal avec memoire : entree sous 'oversold', sortie au-dessus de 'overbought'
    signal = []
    position = 0
    for value in df["rsi"]:
        if value < oversold:
            position = 1
        elif value > overbought:
            position = 0
        signal.append(position)

    df["signal"] = signal
    return df