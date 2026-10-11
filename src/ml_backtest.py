"""Backtest avec stratégie ML."""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import create_features, get_feature_columns
from src.ml_model import MLBacktester
from src.data_fetcher import fetch_binance_data
import matplotlib.pyplot as plt


def backtest_ml_strategy(symbol, timeframe='1d', days=365, model_type='xgboost'):
    """
    Backtest avec stratégie ML
    
    Args:
        symbol: ex 'BTCUSDT'
        timeframe: ex '1d', '4h', '1h'
        days: nombre de jours d'historique
        model_type: 'xgboost' ou 'random_forest'
        
    Returns:
        Dict avec résultats du backtest ou None si erreur
    """
    
    print(f"\n{'='*60}")
    print(f"ML BACKTEST : {symbol} | {timeframe} | {days}j | {model_type}")
    print(f"{'='*60}\n")
    
    try:
        # 1. Récupérer les données
        print("📊 Récupération des données...")
        df = fetch_binance_data(symbol, timeframe, days)
        
        if df is None or len(df) < 100:
            print("❌ Pas assez de données !")
            return None
        
        print(f"✅ {len(df)} bougies chargées\n")
        
        # 2. Créer les features
        print("⚙️ Création des features...")
        df = create_features(df)
        feature_columns = get_feature_columns()
        print(f"✅ {len(feature_columns)} features créées\n")
        
        # 3. Entraîner le modèle
        print("🤖 Entraînement du modèle...")
        ml_model = MLBacktester(model_type=model_type)
        metrics, (X_test, y_test, y_pred) = ml_model.train(df, feature_columns)
        
        # 4. Simuler le trading avec les prédictions
        print("\n📈 Simulation du trading...\n")
        
        # Utiliser les prédictions sur tout le dataset pour le backtest
        df_backtest = df.copy()
        df_backtest['ml_signal'] = ml_model.predict(df_backtest)
        
        # Appliquer la stratégie : acheter si signal=1, vendre si signal=0
        df_backtest['position'] = df_backtest['ml_signal']
        df_backtest['returns'] = df_backtest['close'].pct_change()
        df_backtest['strategy_returns'] = df_backtest['position'].shift(1) * df_backtest['returns']
        
        # Calculs de performance
        cumulative_returns = (1 + df_backtest['strategy_returns']).cumprod()
        buy_hold_returns = (1 + df_backtest['returns']).cumprod()
        
        total_return_strategy = (cumulative_returns.iloc[-1] - 1) * 100
        total_return_bh = (buy_hold_returns.iloc[-1] - 1) * 100
        
        # Win rate
        winning_trades = (df_backtest['strategy_returns'] > 0).sum()
        total_trades = (df_backtest['strategy_returns'] != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Sharpe Ratio
        strategy_std = df_backtest['strategy_returns'].std()
        sharpe = (df_backtest['strategy_returns'].mean() / strategy_std * np.sqrt(252)) if strategy_std > 0 else 0
        
        # Max Drawdown
        cummax = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        print(f"{'='*60}")
        print(f"RÉSULTATS BACKTEST ML")
        print(f"{'='*60}")
        print(f"Retour Stratégie ML:  {total_return_strategy:>8.2f}%")
        print(f"Retour Buy & Hold:    {total_return_bh:>8.2f}%")
        print(f"Outperformance:       {total_return_strategy - total_return_bh:>8.2f}%")
        print(f"\nWin Rate:             {win_rate:>8.2f}%")
        print(f"Sharpe Ratio:         {sharpe:>8.4f}")
        print(f"Max Drawdown:         {max_drawdown:>8.2f}%")
        print(f"Trades:               {total_trades:>8.0f}")
        print(f"{'='*60}\n")
        
        # 5. Visualiser
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Graphique 1 : Prix + signaux
        axes[0].plot(df_backtest.index, df_backtest['close'], label='Prix', linewidth=2, color='black')
        buy_signals = df_backtest[df_backtest['ml_signal'] == 1].index
        sell_signals = df_backtest[df_backtest['ml_signal'] == 0].index
        axes[0].scatter(buy_signals, df_backtest.loc[buy_signals, 'close'], 
                       color='green', marker='^', s=100, label='Signal LONG', zorder=5, edgecolors='darkgreen', linewidth=1)
        axes[0].scatter(sell_signals, df_backtest.loc[sell_signals, 'close'], 
                       color='red', marker='v', s=100, label='Signal COURT', zorder=5, edgecolors='darkred', linewidth=1)
        axes[0].set_title(f'{symbol} - Signaux ML (Prédictions)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Prix (USDT)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Graphique 2 : Performance cumulative
        axes[1].plot(cumulative_returns.index, (cumulative_returns - 1) * 100, 
                    label='Stratégie ML', linewidth=2.5, color='blue')
        axes[1].plot(buy_hold_returns.index, (buy_hold_returns - 1) * 100, 
                    label='Buy & Hold', linewidth=2, linestyle='--', color='orange')
        axes[1].fill_between(cumulative_returns.index, 0, (cumulative_returns - 1) * 100, 
                            alpha=0.1, color='blue')
        axes[1].set_title(f'Performance Cumulative - {symbol}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Retour (%)')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Graphique 3 : Drawdown
        axes[2].fill_between(drawdown.index, drawdown * 100, 0, alpha=0.4, color='red', label='Drawdown')
        axes[2].plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1.5)
        axes[2].set_title(f'Drawdown - {symbol}', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='lower left')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        os.makedirs('results', exist_ok=True)
        filepath = f'results/{symbol}_ML_{model_type}_backtest.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {filepath}")
        plt.close()
        
        # Sauvegarder le modèle
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{symbol}_{model_type}_model.pkl'
        ml_model.save(model_path)
        
        return {
            'symbol': symbol,
            'model': ml_model,
            'metrics': metrics,
            'performance': {
                'return_strategy': total_return_strategy,
                'return_bh': total_return_bh,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades
            },
            'df_backtest': df_backtest
        }
    
    except Exception as e:
        print(f"❌ Erreur lors du backtest {symbol} : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Tester sur BTC et ETH
    results = []
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        result = backtest_ml_strategy(symbol, timeframe='1d', days=365, model_type='xgboost')
        if result:
            results.append(result)
    
    # Comparaison
    print("\n" + "="*60)
    print("📊 COMPARAISON MODÈLES")
    print("="*60)
    for r in results:
        print(f"\n{r['symbol']}")
        print(f"  📈 Retour Stratégie:  {r['performance']['return_strategy']:>8.2f}%")
        print(f"  📊 Retour Buy&Hold:   {r['performance']['return_bh']:>8.2f}%")
        print(f"  🎯 Win Rate:          {r['performance']['win_rate']:>8.2f}%")
        print(f"  📉 Sharpe Ratio:      {r['performance']['sharpe']:>8.4f}")
        print(f"  ⚠️  Max Drawdown:      {r['performance']['max_drawdown']:>8.2f}%")
        print(f"  💰 Nombre Trades:     {r['performance']['total_trades']:>8.0f}")