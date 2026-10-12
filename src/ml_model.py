import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class MLBacktester:
    """ML Backtester avec split temporel (pas de fuite de donnees)."""

    def __init__(self, model_type='xgboost'):
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.train_index = None
        self.test_index = None

        if self.model_type not in ['xgboost', 'random_forest']:
            raise ValueError(f"model_type doit etre 'xgboost' ou 'random_forest', got {model_type}")

    def train(self, df, feature_columns, test_size=0.2, random_state=42):
        if 'target' not in df.columns:
            raise ValueError("DataFrame doit contenir une colonne 'target'")

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        self.feature_columns = feature_columns

        valid = df[feature_columns + ['target']].dropna()
        if len(valid) < 50:
            raise ValueError(f"Pas assez de donnees : {len(valid)} < 50")

        X = valid[feature_columns].values
        y = valid['target'].values

        print(f"\nDonnees : {len(X)} samples")
        print(f"   Distribution : {(y == 1).sum()} UP, {(y == 0).sum()} DOWN")

        # Split TEMPOREL : passe = train, futur = test (pas de melange)
        split = int(len(valid) * (1 - test_size))
        self.train_index = valid.index[:split]
        self.test_index = valid.index[split:]
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        print(f"   Train : {len(X_train)} jours | Test : {len(X_test)} jours (hors echantillon)")

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbosity=0, eval_metric='logloss'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )

        print(f"Entrainement du modele {self.model_type.upper()}...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }

        print(f"\n{'=' * 50}")
        print(f"RESULTATS {self.model_type.upper()} (test hors echantillon)")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"{'=' * 50}\n")

        return metrics, (X_test, y_test, y_pred)

    def _prepare_X(self, df):
        if self.model is None:
            raise ValueError("Modele non entraine ! Appelez train() d'abord.")
        if self.feature_columns is None:
            raise ValueError("Colonnes de features non definies !")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")
        X = df[self.feature_columns].fillna(df[self.feature_columns].mean()).values
        return self.scaler.transform(X)

    def predict(self, df):
        return self.model.predict(self._prepare_X(df))

    def predict_proba(self, df):
        return self.model.predict_proba(self._prepare_X(df))

    def save(self, filepath):
        if self.model is None:
            raise ValueError("Rien a sauvegarder : modele non entraine !")
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
        }, filepath, compress=3)
        print(f"Modele sauvegarde : {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modele non trouve : {filepath}")
        try:
            data = joblib.load(filepath)
            self.model = data.get('model')
            self.scaler = data.get('scaler', StandardScaler())
            self.feature_columns = data.get('feature_columns')
            self.model_type = data.get('model_type', self.model_type)
            if self.model is None:
                raise ValueError("Modele corrompu : 'model' manquant")
            print(f"Modele charge : {filepath}")
            return self
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement : {e}")

    def get_feature_importance(self, top_n=10):
        if self.model is None:
            raise ValueError("Modele non entraine !")
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Le modele {self.model_type} n'a pas feature_importances_")
        df_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        return df_imp.head(top_n)