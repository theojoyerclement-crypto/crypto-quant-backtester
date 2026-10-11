\# Crypto Quant Backtester



Backtester de stratégies de trading sur BTC et ETH en Python : récupération des données, génération de signaux, simulation avec frais, métriques de performance et graphiques.



!\[Backtest BTC](results/BTC\_USDT\_backtest.png)



\## Fonctionnalités



\- Données historiques journalières via l'API Binance (`ccxt`), du 01/01/2020 à aujourd'hui

\- Stratégie : croisement de moyennes mobiles (SMA 20 / SMA 50)

\- Moteur de backtest \*\*sans lookahead bias\*\* (signal décalé d'une période) et \*\*avec frais de trading\*\* (0,1 % par changement de position)

\- Métriques : rendement total, CAGR, Sharpe ratio, max drawdown, nombre de trades, win rate, temps en position

\- Comparaison systématique avec le buy \& hold

\- Graphiques : prix + signaux, equity curve, drawdown



\## Résultats (SMA 20/50, frais 0,1 %, capital initial 10 000 $)



| Métrique | BTC/USDT | ETH/USDT |

|---|---|---|

| Rendement stratégie | 679,4 % | 1 168,8 % |

| Rendement buy \& hold | 1 053,4 % | 1 813,3 % |

| CAGR | 35,4 % | 45,4 % |

| Sharpe ratio | 0,93 | 0,93 |

| Max drawdown | -58,7 % | -60,7 % |

| Nombre de trades | 28 | 24 |

| Win rate | 42,9 % | 54,2 % |



\## Analyse



La stratégie ne bat pas le buy \& hold en rendement brut sur la période, et subit un drawdown important (\~-59 %), principalement en 2022 : les moyennes mobiles lentes réagissent tard aux retournements. C'est le compromis classique du trend-following : peu de trades, win rate modéré, gains concentrés sur les grandes tendances.



\*\*Limites\*\* : pas de slippage, pas de gestion du risque (stop-loss, taille de position), paramètres non optimisés, résultats passés non prédictifs.



\## Installation



```cmd

git clone https://github.com/TON\_PSEUDO/crypto-quant-backtester.git

cd crypto-quant-backtester

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



\## Utilisation



```cmd

python src\\data\_loader.py   :: télécharge les données

python main.py              :: lance les backtests et génère les graphiques

```



Les graphiques sont sauvegardés dans `results/`.



\## Structure



```

src/data\_loader.py   récupération et chargement des données

src/strategy.py      génération des signaux

src/backtest.py      moteur de backtest et métriques

src/plots.py         graphiques

main.py              point d'entrée

```



\## Pistes d'amélioration



\- Stratégies RSI et mean-reversion

\- Optimisation des paramètres avec validation out-of-sample

\- Stop-loss et dimensionnement de position

\- Mode live avec signaux en temps réel



\## Stack



Python, pandas, numpy, matplotlib, ccxt

