import pandas as pd
import ccxt
from datetime import datetime, timedelta

def fetch_binance_data(symbol, timeframe='1d', days=365):
    """
    Récupère les données OHLCV de Binance via ccxt
    
    Args:
        symbol: ex 'BTCUSDT', 'ETHUSDT'
        timeframe: ex '1d', '4h', '1h'
        days: nombre de jours d'historique
    
    Returns:
        DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
    """
    
    try:
        exchange = ccxt.binance()
        
        # Convertir le symbole (BTCUSDT -> BTC/USDT)
        if '/' not in symbol:
            symbol = symbol[:-4] + '/' + symbol[-4:]
        
        print(f"  Récupération de {symbol} ({days}j)...")
        
        # Calculer la date de départ
        since = exchange.parse8601(
            (datetime.now() - timedelta(days=days)).isoformat()
        )
        
        # Récupérer les données par batch (ccxt limite à ~500 bougies par requête)
        all_candles = []
        while since < exchange.milliseconds():
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
            except Exception as e:
                print(f"  ⚠️  Erreur lors de la récupération : {e}")
                break
        
        if not all_candles:
            print(f"  ❌ Aucune donnée récupérée pour {symbol}")
            return None
        
        # Convertir en DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convertir timestamp en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index.name = 'date'
        
        # Garder seulement les derniers `days` jours
        df = df.tail(days)
        
        print(f"  ✅ {len(df)} bougies chargées (de {df.index[0]} à {df.index[-1]})")
        
        return df
    
    except Exception as e:
        print(f"  ❌ Erreur ccxt : {e}")
        return None