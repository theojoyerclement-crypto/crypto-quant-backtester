import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class MLBacktester:
    """
    Machine Learning Backtester pour prédire les mouvements de prix.
    Supporte XGBoost et Random Forest avec sérialisation robuste via joblib.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialise le backtester ML.
        
        Args:
            model_type: 'xgboost' ou 'random_forest'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        if self.model_type not in ['xgboost', 'random_forest']:
            raise ValueError(f"model_type doit être 'xgboost' ou 'random_forest', got {model_type}")
        
    def train(self, df, feature_columns, test_size=0.2, random_state=42):
        """
        Entraîne le modèle ML sur les données historiques.
        
        Args:
            df: DataFrame avec features + colonne 'target' (0 ou 1)
            feature_columns: Liste des colonnes de features à utiliser
            test_size: Proportion test (0.2 = 20%)
            random_state: Seed pour reproducibilité
            
        Returns:
            metrics: Dict avec accuracy, precision, recall, f1
            (X_test, y_test, y_pred): Données de test et prédictions
        """
        
        # Vérifier que la colonne target existe
        if 'target' not in df.columns:
            raise ValueError("DataFrame doit contenir une colonne 'target'")
        
        # Vérifier que les features existent
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Colonnes manquantes : {missing_features}")
        
        self.feature_columns = feature_columns
        
        # Récupérer X et y
        X = df[feature_columns].dropna().values
        y = df.loc[df[feature_columns].dropna().index, 'target'].values
        
        # Vérifier qu'il y a assez de données
        if len(X) < 50:
            raise ValueError(f"Pas assez de données : {len(X)} < 50")
        
        print(f"\n📊 Données d'entraînement : {len(X)} samples")
        print(f"   Distribution : {(y==1).sum()} UP, {(y==0).sum()} DOWN")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normaliser les features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Choisir et configurer le modèle
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
                eval_metric='logloss'
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        
        # Entraîner le modèle
        print(f"🤖 Entraînement du modèle {self.model_type.upper()}...")
        self.model.fit(X_train, y_train)
        
        # Évaluer sur le test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Afficher les résultats
        print(f"\n{'='*50}")
        print(f"RÉSULTATS {self.model_type.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"{'='*50}\n")
        
        return metrics, (X_test, y_test, y_pred)
    
    def predict(self, df):
        """
        Prédit sur de nouvelles données (0 = DOWN, 1 = UP).
        
        Args:
            df: DataFrame avec les mêmes features que l'entraînement
            
        Returns:
            Array de prédictions (0 ou 1)
            
        Raises:
            ValueError si le modèle n'est pas entraîné
        """
        if self.model is None:
            raise ValueError("❌ Modèle non entraîné ! Appelez train() d'abord.")
        
        if self.feature_columns is None:
            raise ValueError("❌ Colonnes de features non définies !")
        
        # Vérifier les features manquantes
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"❌ Colonnes manquantes : {missing_features}")
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].mean()).values
        X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, df):
        """
        Retourne les probabilités de prédiction.
        
        Args:
            df: DataFrame avec les mêmes features que l'entraînement
            
        Returns:
            Array de shape (n_samples, 2) avec [prob_DOWN, prob_UP]
            
        Raises:
            ValueError si le modèle n'est pas entraîné
        """
        if self.model is None:
            raise ValueError("❌ Modèle non entraîné ! Appelez train() d'abord.")
        
        if self.feature_columns is None:
            raise ValueError("❌ Colonnes de features non définies !")
        
        # Vérifier les features manquantes
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"❌ Colonnes manquantes : {missing_features}")
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].mean()).values
        X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """
        Sauvegarde le modèle ET le scaler avec joblib (robuste).
        
        Args:
            filepath: Chemin de sauvegarde (ex: 'models/BTCUSDT_xgboost_model.pkl')
        """
        if self.model is None:
            raise ValueError("❌ Rien à sauvegarder : modèle non entraîné !")
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Sauvegarder tout dans un dict
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath, compress=3)
        print(f"💾 Modèle sauvegardé : {filepath}")
    
    def load(self, filepath):
        """
        Charge un modèle précédemment sauvegardé.
        
        Args:
            filepath: Chemin du modèle (ex: 'models/BTCUSDT_xgboost_model.pkl')
            
        Returns:
            self (pour method chaining)
            
        Raises:
            FileNotFoundError si le fichier n'existe pas
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Modèle non trouvé : {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler', StandardScaler())
            self.feature_columns = model_data.get('feature_columns')
            self.model_type = model_data.get('model_type', self.model_type)
            
            if self.model is None:
                raise ValueError("Modèle corrompu : 'model' manquant")
            
            print(f"✅ Modèle chargé : {filepath}")
            return self
            
        except Exception as e:
            raise RuntimeError(f"❌ Erreur lors du chargement : {e}")
    
    def get_feature_importance(self, top_n=10):
        """
        Retourne les features les plus importantes.
        
        Args:
            top_n: Nombre de top features à retourner
            
        Returns:
            DataFrame avec features et importances
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné !")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Le modèle {self.model_type} n'a pas feature_importances_")
        
        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_imp_df.head(top_n)