"""Prédictions live avec modèle ML sauvegardé."""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import create_features, get_feature_columns
from src.ml_model import MLBacktester
from src.data_fetcher import fetch_binance_data


def predict_future(symbol, model_type='xgboost', n_future=5, lookback_days=30):
    """
    Prédire les mouvements futurs avec un modèle ML sauvegardé.

    Args:
        symbol: ex 'BTCUSDT'
        model_type: 'xgboost' ou 'random_forest'
        n_future: nombre de bougies futures à prédire
        lookback_days: nombre de jours d'historique à charger

    Returns:
        Dict avec prédictions ou None si erreur
    """

    print(f"\n{'='*60}")
    print(f"PRÉDICTIONS ML : {symbol}")
    print(f"{'='*60}\n")

    try:
        # 1. Récupérer les données
        print(f"📊 Récupération des {lookback_days} derniers jours...")
        df = fetch_binance_data(symbol, '1d', lookback_days)

        if df is None or len(df) < 20:
            print("❌ Pas assez de données pour les prédictions !")
            return None

        print(f"✅ {len(df)} bougies chargées\n")

        # 2. Créer les features
        print("⚙️ Création des features...")
        print(f"DEBUG: df.shape AVANT create_features = {df.shape}")
        print(f"DEBUG: df.index = {df.index}\n")
        
        df = create_features(df)
        
        print(f"DEBUG: df.shape APRÈS create_features = {df.shape}")
        if len(df) > 0:
            print(f"DEBUG: df.index = {df.index}")
            print(f"DEBUG: df.columns = {df.columns.tolist()}\n")
        else:
            print("DEBUG: ⚠️ DataFrame vide après create_features !\n")
        
        feature_columns = get_feature_columns()
        print(f"✅ {len(feature_columns)} features créées\n")

        # 3. Charger le modèle
        model_path = f'models/{symbol}_{model_type}_model.pkl'

        print(f"🤖 Chargement du modèle : {model_path}")

        if not os.path.exists(model_path):
            print(f"❌ Modèle non trouvé : {model_path}")
            print(f"   Solution : Lancer d'abord : python main.py")
            return None

        # Charger avec MLBacktester
        try:
            ml_model = MLBacktester(model_type=model_type)
            ml_model.load(model_path)
            print(f"✅ Modèle chargé avec succès\n")
        except Exception as e:
            print(f"❌ Erreur de chargement : {e}")
            print(f"   Les modèles sont peut-être corrompus.")
            print(f"   Solution : supprimer models/*.pkl et relancer python main.py")
            return None

        # 4. Prédire sur les dernières données
        print(f"🔮 Prédictions pour les {n_future} prochaines bougies :\n")

        # Prendre les dernières n_future données (mais max len(df))
        n_predict = min(n_future, len(df))
        
        if n_predict < n_future:
            print(f"⚠️  Attention : seulement {len(df)} bougies disponibles")
            print(f"   Prédiction sur {n_predict} bougies au lieu de {n_future}\n")
        
        latest_data = df.iloc[-n_predict:].copy()
        
        # Vérifier qu'on a des données
        if len(latest_data) == 0:
            print("❌ Aucune donnée à prédire !")
            return None
        
        print(f"📊 Shape : {latest_data.shape}")
        print(f"   Features : {len(feature_columns)} colonnes\n")

        # Prédictions
        try:
            predictions = ml_model.predict(latest_data)
            probabilities = ml_model.predict_proba(latest_data)
        except Exception as e:
            print(f"❌ Erreur lors des prédictions : {e}")
            import traceback
            traceback.print_exc()
            return None

        # Afficher les résultats
        print(f"{'Date':<20} {'Prix':<12} {'Signal':<15} {'Confiance':<12}")
        print("=" * 60)

        results = []
        for i, (idx, row) in enumerate(latest_data.iterrows()):
            pred = predictions[i]
            signal = "🟢 LONG" if pred == 1 else "🔴 SHORT"
            prob_down, prob_up = probabilities[i]
            confidence = max(prob_down, prob_up) * 100
            price = row['close']
            
            # Format date
            if hasattr(idx, 'strftime'):
                date_str = idx.strftime('%Y-%m-%d')
            else:
                date_str = str(idx)

            print(f"{date_str:<20} ${price:<11.2f} {signal:<15} {confidence:<11.1f}%")

            results.append({
                'date': idx,
                'price': price,
                'signal': 'LONG' if pred == 1 else 'SHORT',
                'confidence': confidence,
                'prob_down': prob_down * 100,
                'prob_up': prob_up * 100,
                'prediction': pred
            })

        print("=" * 60)

        # Résumé
        longs = sum(1 for r in results if r['signal'] == 'LONG')
        shorts = sum(1 for r in results if r['signal'] == 'SHORT')
        avg_confidence = np.mean([r['confidence'] for r in results])

        print(f"\n📊 Résumé:")
        print(f"   🟢 LONG:  {longs} ({longs/len(results)*100:.1f}%)")
        print(f"   🔴 SHORT: {shorts} ({shorts/len(results)*100:.1f}%)")
        print(f"   Confiance moyenne: {avg_confidence:.1f}%")
        print(f"\n✅ Prédictions complétées !\n")

        return {
            'symbol': symbol,
            'model_type': model_type,
            'predictions': predictions,
            'probabilities': probabilities,
            'results': results,
            'summary': {
                'longs': longs,
                'shorts': shorts,
                'avg_confidence': avg_confidence
            }
        }

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prédictions ML pour crypto')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbole (ex: BTCUSDT)')
    parser.add_argument('--model', type=str, default='xgboost', help='Modèle (xgboost ou random_forest)')
    parser.add_argument('--future', type=int, default=5, help='Nombre de bougies à prédire')
    parser.add_argument('--lookback', type=int, default=30, help='Jours d\'historique à charger')

    args = parser.parse_args()

    result = predict_future(
        symbol=args.symbol,
        model_type=args.model,
        n_future=args.future,
        lookback_days=args.lookback
    )

    if result:
        print(f"✅ Prédictions pour {result['symbol']} réussies !")
    else:
        print(f"❌ Prédictions échouées.")