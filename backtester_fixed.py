#!/usr/bin/env python3
"""Backtester crypto temporel, sans fuite de données."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


RANDOM_STATE = 42
FEATURES = ["return_1", "sma_ratio", "rsi", "volatility"]


def synthetic_ohlcv(n=1500, seed=RANDOM_STATE):
    """Produit un OHLCV déterministe uniquement si aucun CSV n'est fourni."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    shocks = rng.normal(0.00015, 0.012, n)
    close = 100 * np.exp(np.cumsum(shocks))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    volume = rng.lognormal(10, 0.35, n)
    return pd.DataFrame({"timestamp": dates, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})


def normalize_columns(frame):
    """Tolère casse, espaces et variantes usuelles de noms OHLCV."""
    aliases = {
        "timestamp": {"timestamp", "time", "date", "datetime", "open_time"},
        "open": {"open", "opening", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "closing", "c", "price", "last"},
        "volume": {"volume", "vol", "v", "qty", "quantity"},
    }
    normalized = {}
    for original in frame.columns:
        key = str(original).strip().lower().replace(" ", "_")
        for target, names in aliases.items():
            if key in names:
                normalized[target] = original
                break
    required = ["open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError("Colonnes OHLCV manquantes: " + ", ".join(missing))
    result = frame.rename(columns={value: key for key, value in normalized.items()}).copy()
    if "timestamp" not in result:
        result["timestamp"] = np.arange(len(result))
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for col in required:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    if len(result) < 100:
        raise ValueError("Le CSV doit contenir au moins 100 lignes OHLCV valides.")
    return result


def load_data(csv_path=None):
    if csv_path is None:
        print("Source: données synthétiques (aucun --csv fourni).")
        return synthetic_ohlcv()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError("CSV introuvable: " + str(path))
    print("Source:", path)
    return normalize_columns(pd.read_csv(path))


def make_features(data):
    """Les fenêtres sont causales; la cible est le rendement de la bougie suivante."""
    out = data.copy()
    close = out["close"]
    out["return_1"] = close.pct_change()
    out["sma_ratio"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["volatility"] = out["return_1"].rolling(20, min_periods=20).std()
    out["future_return"] = close.shift(-1) / close - 1.0
    out["target"] = (out["future_return"] > 0).astype(int)
    return out.dropna(subset=FEATURES + ["future_return"]).reset_index(drop=True)


def fit_model(x_train, y_train):
    """Utilise XGBoost si présent, sinon sklearn, sinon un fallback numpy explicite."""
    if XGBClassifier is not None:
        model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9,
                              random_state=RANDOM_STATE, eval_metric="logloss",
                              verbosity=0)
        model.fit(x_train, y_train)
        return model, "XGBoost"
    if LogisticRegression is not None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        return model, "LogisticRegression"
    print("Avertissement: scikit-learn absent; fallback logistique numpy utilisé.")
    return NumpyLogistic().fit(x_train, y_train), "LogisticRegression (numpy fallback)"


class NumpyLogistic:
    """Fallback léger pour que le script reste exécutable sans scikit-learn."""
    def fit(self, x, y):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1
        z = (x - self.mean_) / self.std_
        z = np.c_[np.ones(len(z)), z]
        self.weights_ = np.zeros(z.shape[1])
        for _ in range(700):
            p = 1 / (1 + np.exp(-np.clip(z @ self.weights_, -30, 30)))
            self.weights_ -= 0.05 * (z.T @ (p - y)) / len(y)
        return self

    def predict(self, x):
        z = (x - self.mean_) / self.std_
        z = np.c_[np.ones(len(z)), z]
        return ((1 / (1 + np.exp(-np.clip(z @ self.weights_, -30, 30)))) >= 0.5).astype(int)


def max_drawdown(equity):
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def evaluate(test, predictions):
    """Évalue exclusivement la période test; signal décalé d'une bougie."""
    frame = test.copy()
    frame["prediction"] = predictions
    frame["signal"] = frame["prediction"].shift(1).fillna(0)
    frame["position_change"] = frame["signal"].diff().abs().fillna(frame["signal"].abs())
    frame["strategy_return"] = frame["signal"] * frame["return_1"] - frame["position_change"] * 0.0015
    frame["equity"] = (1 + frame["strategy_return"]).cumprod()
    bh = (1 + frame["return_1"]).cumprod()
    daily_scale = np.sqrt(24 * 365)
    std = frame["strategy_return"].std()
    sharpe = daily_scale * frame["strategy_return"].mean() / std if std > 0 else 0.0
    trades = int((frame["position_change"] > 0).sum())
    active = frame.loc[frame["signal"] != 0, "strategy_return"]
    wins = float((active > 0).mean()) if len(active) else 0.0
    return {
        "Rendement stratégie": float(frame["equity"].iloc[-1] - 1),
        "Buy & hold": float(bh.iloc[-1] - 1),
        "Sharpe annualisé": float(sharpe),
        "Max drawdown": max_drawdown(frame["equity"]),
        "Win-rate": wins,
        "Nombre de trades": trades,
        "Accuracy": float((frame["prediction"] == frame["target"]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtester crypto sans fuite de données")
    parser.add_argument("--csv", help="Chemin vers le CSV OHLCV")
    args = parser.parse_args()
    try:
        data = make_features(load_data(args.csv))
        split = int(len(data) * 0.70)
        train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
        if len(train) < 30 or len(test) < 10:
            raise ValueError("Pas assez de données après préparation train/test.")
        x_train, x_test = train[FEATURES].to_numpy(), test[FEATURES].to_numpy()
        y_train = train["target"].to_numpy()
        model, model_name = fit_model(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate(test, predictions)
        print("\n=== Résultats hors échantillon (test temporel 30 %) ===")
        print("Modèle:", model_name)
        print("Frais: 0,10 %; slippage: 0,05 % par changement de position")
        for name, value in metrics.items():
            if name == "Nombre de trades":
                print(f"{name}: {value}")
            else:
                print(f"{name}: {value:.2%}" if name != "Sharpe annualisé" else f"{name}: {value:.3f}")
    except (FileNotFoundError, ValueError, OSError) as exc:
        print("Erreur:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
