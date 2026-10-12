"""Backtest avec strategie ML (hors echantillon, avec frais)."""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import create_features, get_feature_columns
from src.ml_model import MLBacktester
from src.data_fetcher import fetch_binance_data
import matplotlib.pyplot as plt

FEES = 0.001  # 0,1 % par changement de position


def backtest_ml_strategy(symbol, timeframe='1d', days=365, model_type='xgboost'):
    print(f"\n{'=' * 60}")
    print(f"ML BACKTEST : {symbol} | {timeframe} | {days}j | {model_type}")
    print(f"{'=' * 60}\n")

    try:
        print("Recuperation des donnees...")
        df = fetch_binance_data(symbol, timeframe, days)
        if df is None or len(df) < 100:
            print("Pas assez de donnees !")
            return None
        print(f"{len(df)} bougies chargees\n")

        print("Creation des features...")
        df = create_features(df)
        feature_columns = get_feature_columns()
        print(f"{len(feature_columns)} features creees\n")

        print("Entrainement du modele...")
        ml_model = MLBacktester(model_type=model_type)
        metrics, _ = ml_model.train(df, feature_columns)

        print("\nSimulation du trading (periode de TEST uniquement)...\n")

        # Backtest uniquement sur la periode jamais vue a l'entrainement
        df_bt = df.loc[ml_model.test_index].copy()
        df_bt['ml_signal'] = ml_model.predict(df_bt)
        df_bt['position'] = df_bt['ml_signal']
        df_bt['returns'] = df_bt['close'].pct_change()

        trades_mask = df_bt['position'].diff().abs().fillna(0)
        df_bt['strategy_returns'] = (
            df_bt['position'].shift(1) * df_bt['returns'] - trades_mask * FEES
        )
        df_bt = df_bt.dropna(subset=['strategy_returns'])

        cumulative = (1 + df_bt['strategy_returns']).cumprod()
        buy_hold = (1 + df_bt['returns']).cumprod()

        total_return = (cumulative.iloc[-1] - 1) * 100
        total_return_bh = (buy_hold.iloc[-1] - 1) * 100

        in_position = df_bt['position'].shift(1) == 1
        winning = ((df_bt['strategy_returns'] > 0) & in_position).sum()
        days_in_pos = in_position.sum()
        win_rate = (winning / days_in_pos * 100) if days_in_pos > 0 else 0
        total_trades = int(trades_mask.sum())

        std = df_bt['strategy_returns'].std()
        sharpe = (df_bt['strategy_returns'].mean() / std * np.sqrt(365)) if std > 0 else 0

        cummax = cumulative.expanding().max()
        drawdown = (cumulative - cummax) / cummax
        max_dd = drawdown.min() * 100

        print(f"{'=' * 60}")
        print("RESULTATS BACKTEST ML (hors echantillon, frais 0.1%)")
        print(f"{'=' * 60}")
        print(f"Periode test:         {df_bt.index[0].date()} -> {df_bt.index[-1].date()} ({len(df_bt)}j)")
        print(f"Retour Strategie ML:  {total_return:>8.2f}%")
        print(f"Retour Buy & Hold:    {total_return_bh:>8.2f}%")
        print(f"Outperformance:       {total_return - total_return_bh:>8.2f}%")
        print(f"\nWin Rate (jours):     {win_rate:>8.2f}%")
        print(f"Sharpe Ratio:         {sharpe:>8.4f}")
        print(f"Max Drawdown:         {max_dd:>8.2f}%")
        print(f"Changements position: {total_trades:>8d}")
        print(f"{'=' * 60}\n")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        axes[0].plot(df_bt.index, df_bt['close'], label='Prix', linewidth=2, color='black')
        buys = df_bt[df_bt['ml_signal'] == 1].index
        sells = df_bt[df_bt['ml_signal'] == 0].index
        axes[0].scatter(buys, df_bt.loc[buys, 'close'], color='green', marker='^', s=60, label='LONG', zorder=5)
        axes[0].scatter(sells, df_bt.loc[sells, 'close'], color='red', marker='v', s=60, label='COURT', zorder=5)
        axes[0].set_title(f'{symbol} - Signaux ML (periode de test)', fontweight='bold')
        axes[0].set_ylabel('Prix (USDT)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(cumulative.index, (cumulative - 1) * 100, label='Strategie ML', linewidth=2.5, color='blue')
        axes[1].plot(buy_hold.index, (buy_hold - 1) * 100, label='Buy & Hold', linewidth=2, linestyle='--', color='orange')
        axes[1].set_title(f'Performance cumulative hors echantillon - {symbol}', fontweight='bold')
        axes[1].set_ylabel('Retour (%)')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linewidth=0.8)

        axes[2].fill_between(drawdown.index, drawdown * 100, 0, alpha=0.4, color='red', label='Drawdown')
        axes[2].set_title(f'Drawdown - {symbol}', fontweight='bold')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='lower left')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        filepath = f'results/{symbol}_ML_{model_type}_backtest.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Graphique sauvegarde : {filepath}")
        plt.close()

        os.makedirs('models', exist_ok=True)
        ml_model.save(f'models/{symbol}_{model_type}_model.pkl')

        return {
            'symbol': symbol,
            'model': ml_model,
            'metrics': metrics,
            'performance': {
                'return_strategy': total_return,
                'return_bh': total_return_bh,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'total_trades': total_trades,
            },
            'df_backtest': df_bt,
        }

    except Exception as e:
        print(f"Erreur lors du backtest {symbol} : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = []
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        r = backtest_ml_strategy(symbol, timeframe='1d', days=365, model_type='xgboost')
        if r:
            results.append(r)

    print("\n" + "=" * 60)
    print("COMPARAISON")
    print("=" * 60)
    for r in results:
        p = r['performance']
        print(f"\n{r['symbol']}")
        print(f"  Retour Strategie: {p['return_strategy']:>8.2f}%")
        print(f"  Retour Buy&Hold:  {p['return_bh']:>8.2f}%")
        print(f"  Win Rate:         {p['win_rate']:>8.2f}%")
        print(f"  Sharpe Ratio:     {p['sharpe']:>8.4f}")
        print(f"  Max Drawdown:     {p['max_drawdown']:>8.2f}%")