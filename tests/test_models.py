"""Einheiten des Kerns an der Mini-Instanz von Hand (acht Standtage): IRLS, Pearson-Dispersion, Sandwich, NB2, Exposition."""

import math

import numpy as np
import pytest

import zae_models as M


def test_poisson_ml_is_the_group_means(mini):
    X, y = mini
    f = M.irls(X, y)
    assert f.converged
    assert f.beta == pytest.approx([math.log(1.5), math.log(10 / 3)], abs=1e-9)
    assert f.mu[:4] == pytest.approx(1.5) and f.mu[4:] == pytest.approx(5.0)


def test_first_irls_step_from_beta_zero_is_exact(mini):
    """Bei β = 0 ist μ = 1, W = 1, z = y − 1: der Schritt ist (gewichtetes) Gruppenmittel von z: (1/2; 7/2). Start beta0 = 0, ein Schritt (maxit = 1 nach dem Start)."""
    X, y = mini
    f = M.irls(X, y, beta0=np.zeros(2), maxit=1, tol=0.0)
    assert f.iterations == 1
    # Ein Schritt: der Prädiktor bei β = 0 ist 0, μ = 1; die neue Lösung steht in f.beta
    assert f.beta == pytest.approx([0.5, 3.5], abs=1e-12)


def test_naive_quasi_and_sandwich_variances_by_hand(mini):
    """Von Hand: Var naiv = 1/6 + 1/20 = 13/60; Pearson-χ² = 17/(3/2) + 46/5 = 308/15, φ̂ = χ²/6 = 154/45; Quasi = φ̂ · 13/60; HC0 = 17/36 + 46/400 = 1057/1800; HC1 = HC0 · 8/6."""
    X, y = mini
    inf = M.poisson_inference(X, y)
    assert inf.cov_naive[1, 1] == pytest.approx(13 / 60, rel=1e-9)
    assert inf.phi == pytest.approx(154 / 45, rel=1e-9)
    assert inf.cov_quasi[1, 1] == pytest.approx(1001 / 1350, rel=1e-9)
    assert inf.cov_hc0[1, 1] == pytest.approx(1057 / 1800, rel=1e-9)
    assert inf.cov_hc1[1, 1] == pytest.approx(1057 / 1350, rel=1e-9)


def test_leverage_of_the_two_group_model(mini):
    """Hebel h_i = μ_i x_i' A⁻¹ x_i: bei einem Gruppenmodell 1/T_g · μ_g = 1/n_g = 1/4 je Beobachtung; die Summe der Hebel ist p = 2."""
    X, y = mini
    inf = M.poisson_inference(X, y)
    assert inf.leverage == pytest.approx(0.25, abs=1e-9)
    assert inf.leverage.sum() == pytest.approx(2.0, abs=1e-9)


def test_hc3a_by_hand_for_two_groups(mini):
    """Bei h = 1/4 für alle ist der Beitrag r/(1 − h) = r · 4/3: HC3a = HC0 · 16/9."""
    X, y = mini
    inf = M.poisson_inference(X, y)
    assert inf.cov_hc3a[1, 1] == pytest.approx(inf.cov_hc0[1, 1] * 16 / 9, rel=1e-9)


def test_nb2_alpha_and_variance_of_the_hand_example(mini):
    """α̂ aus der Profilgleichung gleich dem Gitter-Maximum der NB-Likelihood (auf 1e-3 relativ), Var(β̂₁) = 1/T_A + 1/T_B + α̂ (1/n_A + 1/n_B)."""
    X, y = mini
    nb = M.nb2_fit(X, y)
    assert nb.alpha == pytest.approx(0.606, abs=2e-3)
    assert nb.cov[1, 1] == pytest.approx(13 / 60 + nb.alpha * (1 / 4 + 1 / 4), rel=1e-7)
    assert nb.beta == pytest.approx([math.log(1.5), math.log(10 / 3)], abs=1e-7)       # bei jedem festen α bleiben die Gruppenmittel die Lösung


def test_nb2_weights_leave_the_group_means_unchanged_for_any_alpha(mini):
    X, y = mini
    for a in (0.1, 0.5, 2.0):
        assert M.irls(X, y, alpha=a).beta == pytest.approx([math.log(1.5), math.log(10 / 3)], abs=1e-9)


def test_offset_is_the_rate_with_exposure():
    """Kunde 1: 21 Standtage bei 3 Containern, Kunde 2: 20 bei 5: Raten 7 und 4, β = (ln 7, ln 4/7)."""
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    f = M.irls(X, np.array([21.0, 20.0]), offset=np.log([3.0, 5.0]))
    assert f.beta == pytest.approx([math.log(7), math.log(4 / 7)], abs=1e-9)


def test_all_methods_table_for_the_hand_example(mini):
    X, y = mini
    r = M.all_methods(X, y, [1])
    se = r["se"][0]
    # Reihenfolge METHODS: Poisson, Quasi z, Quasi t, HC0, HC1, HC3a, NB2
    assert se[0] == pytest.approx(math.sqrt(13 / 60), rel=1e-9) and se[1] == pytest.approx(math.sqrt(1001 / 1350), rel=1e-9) and se[3] == pytest.approx(math.sqrt(1057 / 1800), rel=1e-9)
    assert r["crit"][2] == pytest.approx(2.4469, abs=5e-4)        # t mit 6 Freiheitsgraden
    assert np.all(np.isfinite(r["est"])) and np.all(np.isfinite(se))


def test_coefficient_table_marks_intervals(mini):
    X, y = mini
    t = M.coefficient_table(X, y, 1)
    assert t["Poisson"]["lo"] > 0 and t["Quasi t"]["lo"] < 0     # Poisson schließt 0 aus, Quasi (t) nicht
    assert t["Poisson"]["est"] == pytest.approx(t["Quasi t"]["est"], abs=1e-12)


def test_nb2_loglik_for_one_observation():
    """NB2 mit μ = 3, α = 1/2, y = 0: P(0) = (1 + αμ)^(−1/α) = (5/2)^(−2) = 4/25."""
    assert math.exp(M.nb2_loglik(np.array([0.0]), np.array([3.0]), 0.5)) == pytest.approx(4 / 25, rel=1e-12)
