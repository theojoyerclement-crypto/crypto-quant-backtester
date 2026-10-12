#!/usr/bin/env python3
"""Génère les fichiers de base du projet crypto-quant-backtester.

Usage depuis la racine du projet (Windows cmd):
    python setup_files.py
"""
from pathlib import Path

BACKTESTER = r'''#!/usr/bin/env python3
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
'''

README = r'''# Crypto Quant Backtester

Backtester pédagogique de stratégies crypto à partir de données OHLCV. Le projet calcule quelques indicateurs techniques, entraîne un classifieur et mesure la stratégie uniquement sur une période de test chronologiquement postérieure à l'apprentissage.

## Méthodologie anti-fuite de données

- Les données sont triées par timestamp et séparées dans le temps : 70 % entraînement, puis 30 % test.
- Les fenêtres d'indicateurs sont causales : elles n'utilisent que les bougies disponibles au moment considéré.
- Le signal est décalé d'une bougie avant l'application au rendement.
- Les frais (0,10 %) et le slippage (0,05 %) sont appliqués à chaque changement de position.
- Les métriques publiées sont calculées hors échantillon, sur le test uniquement.

## Installation sous Windows (cmd)

Depuis la racine du projet :

    python -m venv .venv
    .venv\\Scripts\\activate
    python -m pip install --upgrade pip
    python -m pip install pandas numpy scikit-learn
    REM Optionnel : python -m pip install xgboost

## Utilisation

Avec un CSV OHLCV :

    python backtester_fixed.py --csv data/BTC.csv

Sans argument, le script génère des données synthétiques pour vérifier le fonctionnement :

    python backtester_fixed.py

Le CSV doit contenir timestamp, open, high, low, close et volume. Des variantes courantes de casse et de noms sont tolérées.

## Résultats

| Métrique | Valeur |
| --- | --- |
| Rendement stratégie | à compléter |
| Buy & hold | à compléter |
| Sharpe annualisé | à compléter |
| Max drawdown | à compléter |
| Win-rate | à compléter |
| Nombre de trades | à compléter |
| Accuracy | à compléter |

Aucun chiffre de performance n'est inventé dans cette documentation : les valeurs doivent être renseignées après exécution sur un jeu de données identifié.

## Limites

Ce projet est un outil pédagogique et ne constitue pas un conseil financier. Les données synthétiques ne représentent pas nécessairement le marché. Les coûts, la liquidité, le spread, les latences, les pannes et les contraintes d'exécution réelles sont simplifiés. Le risque de surapprentissage, de biais de sélection et de changement de régime demeure. Les performances passées ou simulées ne garantissent pas les performances futures.

## Structure du projet

    setup_files.py          Générateur des fichiers de base
    backtester_fixed.py     Backtester exécutable
    data/                   Données OHLCV locales (non fournies)
    README.md               Documentation
    COMMANDES_GIT.md        Aide Git pour Windows cmd
'''

GIT = r'''# Commandes Git (Windows cmd)

À exécuter depuis la racine du dépôt :

    git status
    git add setup_files.py backtester_fixed.py README.md COMMANDES_GIT.md
    git commit -m "Ajoute le backtester crypto sans fuite de donnees"
    git push origin main

Pour vérifier le commit avant l'envoi :

    git log -1 --oneline
    git status

Ces commandes supposent que le dépôt distant `origin` et l'authentification Git sont déjà configurés.
'''


def main():
    files = {"backtester_fixed.py": BACKTESTER, "README.md": README, "COMMANDES_GIT.md": GIT}
    root = Path.cwd()
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"Écrit: {name}")


if __name__ == "__main__":
    main()
