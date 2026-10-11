"""Tests unitaires pour le moteur de backtest."""

import numpy as np
import pandas as pd
from backtest import run_backtest, compute_metrics


def create_dummy_df(n=100, trend=1.0):
    """Crée un DataFrame de test avec prix montants ou descendants."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * (1 + trend * np.random.randn(n).cumsum() / 100)
    df = pd.DataFrame(
        {"close": close, "signal": np.random.randint(0, 2, n)}, index=dates
    )
    return df


def test_backtest_runs_without_error():
    """Vérifier que le backtest s'exécute sans crash."""
    df = create_dummy_df(100)
    bt = run_backtest(df)
    assert "equity" in bt.columns
    assert "strategy_returns" in bt.columns
    assert len(bt) == 100


def test_no_lookahead_bias():
    """Vérifier que le signal d'aujourd'hui n'utilise pas le prix de demain.
    
    Position shift(1) signifie qu'on entre en position DEMAIN à partir du signal d'AUJOURD'HUI.
    """
    df = create_dummy_df(50)
    bt = run_backtest(df)
    
    # La position doit être décalée d'une période (lookahead = 0)
    position_shifted = df["signal"].shift(1).fillna(0)
    assert np.allclose(bt["position"].values, position_shifted.values)


def test_fees_reduce_returns():
    """Vérifier que les frais réduisent bien le rendement final."""
    df = create_dummy_df(100)
    
    bt_no_fee = run_backtest(df, fee=0.0)
    bt_with_fee = run_backtest(df, fee=0.001)
    
    equity_no_fee = bt_no_fee["equity"].iloc[-1]
    equity_with_fee = bt_with_fee["equity"].iloc[-1]
    
    assert equity_with_fee <= equity_no_fee, "Les frais doivent réduire le rendement"


def test_compute_metrics_structure():
    """Vérifier que compute_metrics retourne toutes les clés attendues."""
    df = create_dummy_df(365)
    bt = run_backtest(df)
    metrics = compute_metrics(bt)
    
    expected_keys = {
        "Rendement total strategie",
        "Rendement total buy & hold",
        "CAGR (annualise)",
        "Sharpe ratio",
        "Max drawdown",
        "Nombre de trades",
        "Win rate",
        "Temps en position",
    }
    
    assert set(metrics.keys()) == expected_keys, f"Clés manquantes : {expected_keys - set(metrics.keys())}"


def test_max_drawdown_is_negative():
    """Vérifier que le max drawdown est toujours négatif ou 0."""
    df = create_dummy_df(100)
    bt = run_backtest(df)
    metrics = compute_metrics(bt)
    
    assert metrics["Max drawdown"] <= 0, "Max drawdown doit être ≤ 0"


def test_win_rate_in_range():
    """Vérifier que le win rate est entre 0 et 1."""
    df = create_dummy_df(100)
    bt = run_backtest(df)
    metrics = compute_metrics(bt)
    
    assert 0 <= metrics["Win rate"] <= 1, f"Win rate hors range : {metrics['Win rate']}"


def test_temps_position_in_range():
    """Vérifier que le temps en position est entre 0 et 1."""
    df = create_dummy_df(100)
    bt = run_backtest(df)
    metrics = compute_metrics(bt)
    
    assert 0 <= metrics["Temps en position"] <= 1, f"Temps en position hors range : {metrics['Temps en position']}"


def test_equity_curve_starts_at_initial_capital():
    """Vérifier que la courbe d'équité commence au capital initial."""
    df = create_dummy_df(100)
    initial_capital = 10_000
    bt = run_backtest(df, initial_capital=initial_capital)
    
    assert np.isclose(bt["equity"].iloc[0], initial_capital, rtol=1e-2), \
        f"Équité initiale ≠ capital initial : {bt['equity'].iloc[0]} vs {initial_capital}"


if __name__ == "__main__":
    # Exécution simple (sans pytest)
    test_backtest_runs_without_error()
    test_no_lookahead_bias()
    test_fees_reduce_returns()
    test_compute_metrics_structure()
    test_max_drawdown_is_negative()
    test_win_rate_in_range()
    test_temps_position_in_range()
    test_equity_curve_starts_at_initial_capital()
    
    print("✅ Tous les tests passent!")