import sys
from src.ml_backtest import backtest_ml_strategy
from src.data_fetcher import fetch_binance_data
from src.features import create_features

def main():
    """
    Script principal pour lancer les backtests
    """
    
    print("\n" + "="*60)
    print("CRYPTO QUANT BACKTESTER - ML VERSION")
    print("="*60 + "\n")
    
    # Config
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframe = '1d'
    days = 365
    model_type = 'xgboost'
    
    print(f"📊 Configuration:")
    print(f"   Symboles: {symbols}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Jours: {days}")
    print(f"   Modèle: {model_type}\n")
    
    # Lancer les backtests
    results = []
    for symbol in symbols:
        print(f"\n🔄 Processing {symbol}...")
        result = backtest_ml_strategy(symbol, timeframe, days, model_type)
        if result:
            results.append(result)
    
    # Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    
    for r in results:
        print(f"\n{r['symbol']}:")
        print(f"  Return Stratégie: {r['performance']['return_strategy']:.2f}%")
        print(f"  Return Buy&Hold:  {r['performance']['return_bh']:.2f}%")
        print(f"  Win Rate:         {r['performance']['win_rate']:.2f}%")
        print(f"  Sharpe Ratio:     {r['performance']['sharpe']:.4f}")
        print(f"  Max Drawdown:     {r['performance']['max_drawdown']:.2f}%")

if __name__ == '__main__':
    main()