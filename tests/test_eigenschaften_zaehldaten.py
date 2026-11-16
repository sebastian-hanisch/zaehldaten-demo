"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die ohne Kenntnis des richtigen Ergebnisses gelten müssen (je 20 Zufallsinstanzen, fester Zufalls-Seed)."""

import math

import numpy as np
import pytest

import zae_models as M
import zae_sample as S


def _data(seed, n=60, alpha=0.5, c=0.0):
    X, y, _ = S.draw_world(seed, n, "NB2", alpha, c)
    return X, y


SEEDS = range(300, 320)


def test_quasi_poisson_has_the_same_coefficients_and_se_scaled_by_sqrt_phi():
    for seed in SEEDS:
        X, y = _data(seed)
        inf = M.poisson_inference(X, y)
        assert np.sqrt(np.diag(inf.cov_quasi)) == pytest.approx(math.sqrt(inf.phi) * np.sqrt(np.diag(inf.cov_naive)), rel=1e-12)
        t = M.coefficient_table(X, y, 1)
        assert t["Quasi z"]["est"] == pytest.approx(t["Poisson"]["est"], abs=1e-12)         # dieselben Koeffizienten (Konstruktion)


def test_nb2_does_not_have_the_same_coefficients_as_poisson_when_the_mean_model_is_wrong():
    """Gegenstück: „der Punktschätzer ist gleich“ gilt nur für Quasi gegen Poisson, nicht für NB2 (hier bei einer Krümmung c = 0,3 und n = 400)."""
    diffs = []
    for seed in SEEDS:
        X, y = _data(seed, n=400, c=0.3)
        diffs.append(abs(M.nb2_fit(X, y).beta[1] - M.irls(X, y).beta[1]))
    assert np.median(diffs) > 0.05


def test_permuting_the_observations_changes_nothing():
    for seed in SEEDS:
        X, y = _data(seed)
        perm = np.random.default_rng(seed).permutation(len(y))
        a, b = M.all_methods(X, y, [1]), M.all_methods(X[perm], y[perm], [1])
        assert a["est"] == pytest.approx(b["est"], abs=1e-9) and a["se"] == pytest.approx(b["se"], rel=1e-8)


def test_shifting_a_feature_changes_only_the_intercept():
    for seed in SEEDS:
        X, y = _data(seed)
        X2 = X.copy()
        X2[:, 1] += 2.5
        b1, b2 = M.irls(X, y).beta, M.irls(X2, y).beta
        assert b2[1] == pytest.approx(b1[1], abs=1e-8)
        assert b2[0] == pytest.approx(b1[0] - 2.5 * b1[1], abs=1e-8)       # log μ = b0 + b1 (x + 2,5 − 2,5) → neuer Achsenabschnitt b0 − 2,5 b1


def test_doubling_every_observation():
    """β̂ bleibt gleich, der naive Poisson-SE und HC0 werden durch √2 geteilt; φ̂ bleibt nur ungefähr gleich (Freiheitsgrade 2n − p statt n − p)."""
    for seed in SEEDS:
        X, y = _data(seed)
        X2, y2 = np.vstack([X, X]), np.concatenate([y, y])
        a, b = M.poisson_inference(X, y), M.poisson_inference(X2, y2)
        assert b.fit.beta == pytest.approx(a.fit.beta, abs=1e-8)
        assert np.sqrt(b.cov_naive[1, 1]) == pytest.approx(np.sqrt(a.cov_naive[1, 1]) / math.sqrt(2), rel=1e-8)
        assert np.sqrt(b.cov_hc0[1, 1]) == pytest.approx(np.sqrt(a.cov_hc0[1, 1]) / math.sqrt(2), rel=1e-8)
        assert b.phi == pytest.approx(a.phi * (len(y) - 2) / (2 * len(y) - 2) * 2, rel=1e-9)   # Σ(...) verdoppelt, Nenner 2n − p
        assert abs(b.phi / a.phi - 1) < 0.1


def test_offset_log_t_equals_rate_with_weight_t():
    """Poisson mit Offset log t: die Score-Gleichung ist dieselbe wie die der Rate y/t mit Gewicht t (Rechenweg: gewichtetes IRLS für die Rate)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = rng.integers(1, 8, 50).astype(float)
        x = rng.normal(size=50)
        X = np.column_stack([np.ones(50), x])
        y = rng.poisson(t * np.exp(0.4 + 0.3 * x)).astype(float)
        b = M.irls(X, y, offset=np.log(t)).beta
        b_r = np.zeros(2)
        for _ in range(60):              # gewichtetes IRLS für die Rate y/t, Gewicht t · μ_rate (anderer Rechenweg: kein Offset)
            eta = X @ b_r
            mu_r = np.exp(eta)
            z = eta + (y / t - mu_r) / mu_r
            w = t * mu_r
            b_r = np.linalg.solve((X.T * w) @ X, (X.T * w) @ z)
        assert b == pytest.approx(b_r, abs=1e-7)


def test_scaling_the_exposure_by_c_shifts_only_the_intercept_by_minus_ln_c():
    for seed in SEEDS:
        X, y = _data(seed)
        t = np.random.default_rng(seed).integers(1, 6, len(y)).astype(float)
        b1 = M.irls(X, y, offset=np.log(t)).beta
        b2 = M.irls(X, y, offset=np.log(3.0 * t)).beta
        assert b2[1] == pytest.approx(b1[1], abs=1e-8) and b2[0] == pytest.approx(b1[0] - math.log(3.0), abs=1e-8)


def test_nb2_with_alpha_to_zero_is_poisson():
    for seed in SEEDS:
        X, y = _data(seed, alpha=0.0)
        a = M.irls(X, y, alpha=1e-9).beta
        assert a == pytest.approx(M.irls(X, y).beta, abs=1e-6)


def test_group_models_return_the_group_means():
    """Bei einem Gruppenmodell löst das Gruppenmittel die Score-Gleichung (Score je Gruppe null), für Poisson und für NB2 mit jedem α."""
    rng = np.random.default_rng(9)
    for _ in range(20):
        g = rng.integers(0, 3, 90)
        X = np.column_stack([np.ones(90), g == 1, g == 2]).astype(float)
        y = rng.poisson(np.where(g == 0, 1.5, np.where(g == 1, 4.0, 9.0)) * rng.gamma(2, 0.5, 90)).astype(float)
        if min(y[g == k].sum() for k in range(3)) == 0:
            continue
        for a in (0.0, 0.7):
            f = M.irls(X, y, alpha=a)
            for k in range(3):
                assert f.mu[g == k] == pytest.approx(y[g == k].mean(), rel=1e-7)


def test_poisson_coverage_falls_with_overdispersion():
    """Mittel der Überdeckung über 60 Datensätze je α (Zähleinheit Wiederholung): α = 0 > 0,5 > 1 mit großem Abstand (Band: das Mittel bei α = 0 über 0,85, bei α = 1 unter 0,8)."""
    cov = {}
    for a in (0.0, 0.5, 1.0):
        hit = 0
        for r in range(150):
            X, y, _ = S.draw_world(5000 + r, 200, "NB2", a)
            t = M.coefficient_table(X, y, 1, with_nb2=False)["Poisson"]
            hit += t["lo"] <= 0.4 <= t["hi"]
        cov[a] = hit / 150
    assert cov[0.0] > 0.85 and cov[0.5] < cov[0.0] and cov[1.0] < cov[0.5] and cov[1.0] < 0.8


def test_quasi_poisson_with_phi_below_one_narrows_the_interval():
    """Unterdispersion (Zählung aus 1 + Binomial): φ̂ < 1, das Quasi-Intervall ist dann schmaler als das Poisson-Intervall (wie in R, nicht auf 1 begrenzt)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    X = np.column_stack([np.ones(300), x])
    mu = np.exp(1.5 + 0.2 * x)
    y = rng.binomial(40, np.minimum(mu / 40, 0.99)).astype(float)         # Varianz μ (1 − μ/40) < μ
    inf = M.poisson_inference(X, y)
    assert inf.phi < 1
    assert inf.cov_quasi[1, 1] < inf.cov_naive[1, 1]
