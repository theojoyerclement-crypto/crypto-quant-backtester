# Crypto Quant Backtester

> Backtesting reproductible de stratégies crypto sur **BTC/USDT** et **ETH/USDT**, combinant indicateurs techniques, XGBoost et métriques de risque.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-research-orange?style=flat-square)](#limites-et-précautions)

## Sommaire

- [Présentation](#présentation)
- [Résultats](#résultats)
- [Méthodologie](#méthodologie)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Limites et précautions](#limites-et-précautions)
- [Prochaines étapes](#prochaines-étapes)
- [Contribution et licence](#contribution-et-licence)

## Présentation

Ce projet explore une question simple : **un modèle supervisé apporte-t-il une amélioration mesurable par rapport à une approche technique classique sur les cryptomonnaies ?**

Le dépôt fournit :

- un moteur de backtesting sur données OHLCV journalières ;
- une stratégie technique fondée sur le croisement de moyennes mobiles et le RSI ;
- une stratégie ML XGBoost utilisant 11 variables techniques ;
- des prédictions, signaux live, graphiques et métriques de performance ;
- une base de travail pour mettre en place une validation quantitative plus robuste.

> **Important :** les chiffres ci-dessous sont les résultats d’une exécution capturée du projet, et non une promesse de performance future. Ils doivent être reproduits et validés hors échantillon avant toute conclusion d’investissement.

## Résultats

### Résultats ML capturés — fenêtre de 365 jours, frais de 0,1 %

| Actif | Rendement stratégie | Buy & Hold | Surperformance | Sharpe | Max drawdown | Win rate | Trades |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **BTC/USDT** | **+805,96 %** | -27,16 % | **+833,12 %** | 6,52 | -5,40 % | 94,61 % | 167 |
| **ETH/USDT** | **+1 378,21 %** | -38,92 % | **+1 417,13 %** | 5,61 | -8,88 % | 97,08 % | 171 |

**Lecture rapide :** le backtest capturé affiche une forte surperformance de la stratégie ML par rapport au buy & hold sur cette fenêtre. Ces valeurs sont exceptionnellement élevées pour un backtest journalier ; elles doivent donc être considérées comme un **signal à auditer**, notamment sur la séparation entraînement/test, le décalage des signaux, le calcul des trades et le risque de fuite de données.

### Qualité de classification observée

| Actif | Accuracy entraînement | Accuracy test | Nombre de features |
|:---|---:|---:|---:|
| **BTC/USDT** | 100,00 % | 56,16 % | 11 |
| **ETH/USDT** | 100,00 % | 49,32 % | 11 |

L’écart entre l’accuracy d’entraînement et l’accuracy de test est un indicateur de surapprentissage potentiel. La performance de portefeuille ne doit pas être interprétée indépendamment des métriques de classification et des contrôles de fuite de données.

### Graphiques

Les graphiques générés par le backtest sont disponibles dans `results/` :

- [BTCUSDT — courbe d’équité et drawdown](results/BTCUSDT_ML_xgboost_backtest.png)
- [ETHUSDT — courbe d’équité et drawdown](results/ETHUSDT_ML_xgboost_backtest.png)

## Méthodologie

### Données

- Source : données OHLCV Binance via l’API de marché.
- Actifs : `BTC/USDT` et `ETH/USDT`.
- Timeframe du run présenté : **1 jour**.
- Fenêtre du run présenté : **365 jours**.
- Coût de transaction indiqué dans la sortie : **0,1 %**.

### Variables explicatives

Le modèle XGBoost est alimenté par 11 variables dérivées des prix et volumes :

1. SMA 20 jours ;
2. SMA 50 jours ;
3. RSI 14 ;
4. volatilité glissante ;
5. MACD ;
6. largeur des bandes de Bollinger ;
7. ATR 14 ;
8. momentum 10 jours ;
9. ratio de volume ;
10. position du cours dans la bougie ;
11. rendement.

La cible est une classification directionnelle : `1` lorsque la clôture suivante est supérieure à la clôture courante, `0` sinon.

### Stratégies et métriques

- **SMA + RSI** : stratégie technique de référence.
- **XGBoost** : classification supervisée des mouvements futurs.
- **Buy & Hold** : benchmark passif.
- **Métriques** : rendement cumulé, surperformance, Sharpe, maximum drawdown, win rate et nombre de trades.

## Fonctionnalités

- Backtest BTC/ETH automatisé.
- Entraînement et sauvegarde des modèles XGBoost.
- Comparaison avec des stratégies techniques et le buy & hold.
- Génération de graphiques de performance.
- Prédictions sur les prochaines bougies.
- Signaux live affichés périodiquement.
- Scripts d’optimisation et tests unitaires présents dans `src/`.

## Installation

### Prérequis

- Python 3.9 ou plus récent ;
- `pip` et `venv` ;
- connexion Internet pour récupérer les données Binance en mode API.

### Mise en place

```bash
git clone https://github.com/theojoyerclement-crypto/crypto-quant-backtester.git
cd crypto-quant-backtester

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

Depuis la racine du dépôt :

```bash
# Backtest ML sur BTC et ETH
python main.py

# Signaux live
python src/live.py

# Prédictions ML
python src/predict.py --symbol BTCUSDT --model xgboost --days 5

# Génération de graphiques
python src/generate_plots.py

# Tests
pytest
```

Les sorties sont écrites dans `results/` et les modèles sérialisés dans `models/` lorsque les scripts correspondants sont exécutés.

## Architecture

```text
crypto-quant-backtester/
├── main.py                         # Point d’entrée du backtest ML
├── requirements.txt                # Dépendances Python
├── data/                           # Données historiques locales
│   ├── BTC_USDT_1d.csv
│   └── ETH_USDT_1d.csv
├── src/
│   ├── data_fetcher.py             # Récupération des données marché
│   ├── data_loader.py              # Chargement et préparation des données
│   ├── features.py                 # Construction des variables techniques
│   ├── strategy.py                 # Stratégies techniques
│   ├── backtest.py                 # Moteur de backtesting classique
│   ├── ml_model.py                 # Modèle et entraînement ML
│   ├── ml_backtest.py              # Simulation XGBoost
│   ├── predict.py                  # Prédictions futures
│   ├── live.py                     # Signaux live
│   ├── optimize.py                 # Optimisation des paramètres
│   ├── generate_plots.py            # Graphiques de comparaison
│   ├── plots.py                    # Visualisations du backtest ML
│   └── test_backtest.py            # Tests unitaires
├── models/                         # Modèles sauvegardés, générés localement
└── results/                        # Résultats graphiques
```

## Limites et précautions

Ce projet est une **base de recherche et d’apprentissage**, pas un système de trading prêt pour la production.

- Les résultats présentés proviennent d’une fenêtre historique limitée et d’un run capturé.
- La forte performance observée est atypique et doit être auditée avant d’être considérée comme crédible.
- Le dépôt doit encore démontrer une validation walk-forward et plusieurs fenêtres hors échantillon.
- Les contrôles de data leakage doivent être vérifiés : les features doivent utiliser uniquement l’information disponible avant la décision.
- Les opérations d’exécution doivent être strictement décalées après le signal ; sinon le backtest peut bénéficier d’une information future.
- Le slippage, la liquidité, les spreads, les latences et les rejets d’ordres ne sont pas entièrement modélisés.
- Le win rate n’est interprétable que si les entrées, sorties et trades fermés sont comptés correctement.
- Les rendements passés ne garantissent pas les rendements futurs.

**Aucun conseil financier. Ne pas utiliser ce projet avec des fonds réels sans validation indépendante, paper trading et contrôles de risque.**

## Prochaines étapes

### Priorité 1 — fiabilité du backtest

- [ ] Remplacer tout remplissage rétrospectif des valeurs manquantes par un traitement causal (`dropna` après création des features).
- [ ] Mettre en place un split temporel strict : entraînement avant test, sans mélange aléatoire.
- [ ] Produire des prédictions véritablement out-of-sample, sans évaluer le modèle sur les données utilisées pour son entraînement.
- [ ] Décaler chaque signal d’au moins une bougie avant l’exécution.
- [ ] Recalculer le win rate à partir du P&L des trades fermés.
- [ ] Ajouter des assertions et tests anti-leakage.

### Priorité 2 — validation quantitative

- [ ] Ajouter une validation walk-forward avec plusieurs fenêtres.
- [ ] Comparer XGBoost, SMA + RSI, buy & hold et un benchmark aléatoire.
- [ ] Mesurer l’incertitude : intervalles de confiance, bootstrap et sensibilité aux paramètres.
- [ ] Tester plusieurs régimes de marché, actifs et timeframes.
- [ ] Documenter précisément période, timezone, source des données et définition de chaque métrique.

### Priorité 3 — réalisme d’exécution et production

- [ ] Modéliser spread, slippage dépendant de la volatilité et liquidité.
- [ ] Ajouter position sizing, exposition maximale, stop-loss et limites de risque.
- [ ] Ajouter un journal de trades exportable et un rapport HTML/CSV reproductible.
- [ ] Mettre en place CI, linting, tests de non-régression et verrouillage des versions.
- [ ] Ajouter alertes Telegram/Discord et un dashboard de suivi après validation paper trading.

## Contribution et licence

Les issues, pull requests et retours méthodologiques sont bienvenus. Pour contribuer :

```bash
git checkout -b feature/ma-contribution
git add .
git commit -m "feat: description concise"
git push origin feature/ma-contribution
```

Projet distribué sous licence **MIT**. Voir `LICENSE`.
