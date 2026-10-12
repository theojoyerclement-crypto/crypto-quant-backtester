import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Style pro
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'XGBoost': '#2ecc71', 'SMA': '#3498db', 'RSI': '#e74c3c', 'Hold': '#95a5a6'}

def plot_equity_curves(results_dict, symbol='BTCUSDT'):
    """Equity curves: XGBoost vs SMA vs RSI vs Buy&Hold"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for strategy, data in results_dict.items():
        if 'equity_curve' in data:
            ax.plot(data['dates'], data['equity_curve'], 
                   label=f"{strategy} ({data['return']:.1%})",
                   linewidth=2.5, color=colors.get(strategy, '#9b59b6'))
    
    ax.set_title(f'Equity Curve - {symbol} (Rendement total, capital initial 10k $)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{symbol}_equity_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique equity curves sauvegardé: results/{symbol}_equity_curves.png")
    plt.close()


def plot_strategy_comparison(metrics_dict, symbol='BTCUSDT'):
    """Comparaison multi-stratégies (barplots)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparaison Stratégies - {symbol}', fontsize=18, fontweight='bold', y=1.00)
    
    strategies = list(metrics_dict.keys())
    
    # Return
    returns = [metrics_dict[s].get('return_strategy', 0) for s in strategies]
    ax1.bar(strategies, returns, color=[colors.get(s, '#9b59b6') for s in strategies], alpha=0.8)
    ax1.set_title('Rendement Total (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('%', fontweight='bold')
    for i, v in enumerate(returns):
        ax1.text(i, v + 20, f'{v:.1f}%', ha='center', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sharpe Ratio
    sharpes = [metrics_dict[s].get('sharpe', 0) for s in strategies]
    ax2.bar(strategies, sharpes, color=[colors.get(s, '#9b59b6') for s in strategies], alpha=0.8)
    ax2.set_title('Sharpe Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Sharpe', fontweight='bold')
    for i, v in enumerate(sharpes):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Win Rate
    win_rates = [metrics_dict[s].get('win_rate', 0) * 100 for s in strategies]
    ax3.bar(strategies, win_rates, color=[colors.get(s, '#9b59b6') for s in strategies], alpha=0.8)
    ax3.set_title('Win Rate (%)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('%', fontweight='bold')
    ax3.set_ylim(0, 100)
    for i, v in enumerate(win_rates):
        ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Max Drawdown
    max_dds = [metrics_dict[s].get('max_drawdown', 0) * 100 for s in strategies]
    ax4.bar(strategies, max_dds, color=[colors.get(s, '#9b59b6') for s in strategies], alpha=0.8)
    ax4.set_title('Max Drawdown (%)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('%', fontweight='bold')
    for i, v in enumerate(max_dds):
        ax4.text(i, v - 3, f'{v:.1f}%', ha='center', fontweight='bold', color='white')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{symbol}_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique comparaison sauvegardé: results/{symbol}_strategy_comparison.png")
    plt.close()


def plot_drawdown_analysis(dates, equity_curve, strategy_name='XGBoost', symbol='BTCUSDT'):
    """Drawdown analysis"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Analyse du Drawdown - {strategy_name} ({symbol})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Equity curve
    ax1.plot(dates, equity_curve, linewidth=2.5, color=colors.get(strategy_name, '#2ecc71'), label=strategy_name)
    ax1.fill_between(dates, equity_curve, alpha=0.2, color=colors.get(strategy_name, '#2ecc71'))
    ax1.set_title('Courbe de capitalisation', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Capital ($)', fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Drawdown
    running_max = pd.Series(equity_curve).expanding().max().values
    drawdown = (np.array(equity_curve) - running_max) / running_max * 100
    ax2.fill_between(dates, drawdown, alpha=0.6, color='#e74c3c', label='Drawdown %')
    ax2.set_title('Drawdown par rapport au pic', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{symbol}_drawdown_{strategy_name}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique drawdown sauvegardé: results/{symbol}_drawdown_{strategy_name}.png")
    plt.close()


def plot_price_with_signals(dates, prices, signals, strategy_name='XGBoost', symbol='BTCUSDT'):
    """Prix + signaux d'entrée/sortie"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(dates, prices, linewidth=2, color='#2c3e50', label='Prix', zorder=1)
    
    # Signaux LONG (entrée)
    long_signals = signals[signals['signal'] == 1]
    if len(long_signals) > 0:
        ax.scatter(long_signals['date'], long_signals['price'], 
                  color='#2ecc71', marker='^', s=150, label='LONG (Achat)', zorder=3)
    
    # Signaux SHORT (sortie)
    short_signals = signals[signals['signal'] == -1]
    if len(short_signals) > 0:
        ax.scatter(short_signals['date'], short_signals['price'], 
                  color='#e74c3c', marker='v', s=150, label='SHORT (Vente)', zorder=3)
    
    ax.set_title(f'Prix + Signaux - {strategy_name} ({symbol})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prix ($)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{symbol}_signals_{strategy_name}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique signaux sauvegardé: results/{symbol}_signals_{strategy_name}.png")
    plt.close()


if __name__ == '__main__':
    print("✅ Module de génération graphiques chargé")
    print("Utilisation : from src.generate_plots import plot_equity_curves, plot_strategy_comparison, etc.")