"""Randfälle: Gruppe ohne Ereignis, alle Nullen, n = p, n < p, konstante Spalte, Unterdispersion, große Zählwerte, Ausreißer, scheiternde Anpassung."""

import math

import numpy as np
import pytest

import zae_models as M
import zae_sample as S
import zae_simulation as SIM


def test_group_without_event_poisson_does_not_converge():
    """Ohne ein einziges Ereignis in einer Gruppe existiert der Schätzer nicht (β → −∞): die IRLS meldet kein Konvergieren (Abbruch an der Kappung), die App-Meldung steht in zae_hand."""
    X = np.array([[1.0, 0.0]] * 4 + [[1.0, 1.0]] * 4)
    y = np.array([0, 0, 0, 0, 1, 3, 6, 10], float)
    f = M.irls(X, y, maxit=40)
    assert not f.converged or f.beta[0] < -15


def test_all_zero_counts():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    y = np.zeros(10)
    try:
        f = M.irls(X, y, maxit=40)
        assert not f.converged or f.beta[0] < -15
    except M.FitError:
        pass


def test_n_equals_p_and_fewer_observations_than_columns():
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([3.0, 5.0])
    with pytest.raises(M.FitError):
        M.poisson_inference(X, y)                      # n = p: keine Freiheitsgrade für φ̂
    with pytest.raises(M.FitError):
        M.irls(X[:1], y[:1])                           # n < p


def test_constant_column_is_singular():
    X = np.column_stack([np.ones(20), np.ones(20)])
    with pytest.raises(M.FitError):
        M.irls(X, np.arange(20.0) % 4)


def test_underdispersion_puts_alpha_on_the_boundary():
    """Zählungen mit Varianz < Mittel: die Profilgleichung hat keine Nullstelle, α̂ liegt am Rand (Randlösung gemeldet), kleiner als 1e-5."""
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    mu = np.exp(1.8 + 0.2 * x)
    y = rng.binomial(60, np.minimum(mu / 60, 0.99)).astype(float)
    nb = M.nb2_fit(X, y)
    assert nb.at_boundary and nb.alpha < 1e-5


def test_huge_counts_do_not_overflow():
    """Kappung |η| ≤ 30: auch bei Zählwerten bis 1e9 ein endliches Ergebnis (kein Überlauf)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=50)
    X = np.column_stack([np.ones(50), x])
    y = np.round(np.exp(18 + 0.5 * x)).astype(float)
    f = M.irls(X, y)
    assert np.all(np.isfinite(f.beta)) and np.all(np.isfinite(f.mu))
    assert f.beta[0] == pytest.approx(18, abs=0.1) and f.beta[1] == pytest.approx(0.5, abs=0.05)


def test_all_methods_reports_a_nb2_failure_without_losing_the_poisson_row(monkeypatch):
    def boom(*a, **k):
        raise M.FitError("erfundener Fehler")

    monkeypatch.setattr(M, "nb2_fit", boom)
    X, y, _ = S.draw_world(1, 100, "NB2", 0.5)
    r = M.all_methods(X, y, [1])
    assert "NB2" in r["fehler"] and np.isnan(r["est"][0, 6]) and np.isfinite(r["est"][0, 0])
    assert "NB2" not in M.coefficient_table(X, y, 1)


def test_run_cell_counts_failures_instead_of_hiding_them(monkeypatch):
    calls = {"n": 0}
    orig = M.nb2_fit

    def flaky(X, y, *a, **k):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise M.FitError("erfundener Fehler")
        return orig(X, y, *a, **k)

    monkeypatch.setattr(M, "nb2_fit", flaky)
    cell = SIM.run_cell("NB2", 0.5, 50, 40, 12)
    assert cell["nb2_ausfaelle"] == 10 and cell["n_valid"]["NB2"] == 30 and cell["n_valid"]["Poisson"] == 40
    assert 0 <= cell["coverage"]["NB2"] <= 1


def test_misspecification_outlier_datasets_are_not_averaged_in():
    """Bei c = 0,3 und n = 200 gibt es Datensätze mit riesigem Unterschied der Schätzer; die Auswertung nennt Median und Quartile und zählt die Ausreißer (Betrag > 1)."""
    r = SIM.run_missspec(0.3, 200, 200, 5)
    assert r["diff_q1"] <= r["diff_median"] <= r["diff_q3"] and r["diff_gross"] >= 0
    assert "diff_mean" not in r and "diff_sd" not in r


def test_inversion_handles_mean_zero_and_large_uniform():
    y = S.counts_from_uniform(np.array([0.0, 0.5, 0.999999999]), np.array([0.0, 1e-9, 2.0]), "nb2", 0.5)
    assert y[0] == 0 and y[1] == 0 and y[2] >= 3


def test_hand_values_with_all_equal_counts_have_zero_dispersion_residuals():
    """Beide Gruppen konstant (3, 3, 3, 3 und 6, 6, 6, 6): kein Rest, φ̂ = 0, Sandwich-Varianz 0: die Anzeige muss damit umgehen (keine Division durch 0 in der Fraction-Rechnung)."""
    import zae_hand as H
    ex = H.hand_exact((3, 3, 3, 3), (6, 6, 6, 6))
    assert ex["phi"] == 0 and ex["var_hc0"] == 0 and ex["var_naive"] == 1 / 12 + 1 / 24
    r = H.hand_numeric((3, 3, 3, 3), (6, 6, 6, 6))
    assert math.isfinite(r["beta"][1]) and r["z"]["naive"] > 0 and math.isnan(r["z"]["hc0"])      # HC0: Standardfehler 0, z nicht definiert
