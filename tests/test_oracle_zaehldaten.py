"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg.

- IRLS (Poisson) gegen scikit-learn `PoissonRegressor(alpha=0, solver="newton-cholesky")` und gegen BFGS mit analytischem Gradienten (nie ohne `jac`: Vorwärtsdifferenz streut bis 1e-4).
- Offset log t gegen scikit-learn mit Rate y/t und Gewicht t.
- NB2-Log-Likelihood gegen `scipy.stats.nbinom.logpmf(n = 1/α, p = 1/(1 + αμ))`; α̂ gegen das Gitter-Maximum der scipy-Likelihood und gegen die gemeinsame Maximierung über (β, log α) mit `jac="3-point"`.
- Sandwich gegen eine Schleife über die Beobachtungen und gegen die dichte Matrixform.
- Grenzwert der Überdeckung: Gauss-Hermite gegen `scipy.integrate.quad` (zwei Rechenwege).
- Zählziehung per Inversion gegen `scipy.stats.nbinom.ppf`; Generator gegen die geschlossene Verteilung (Verweilzeit lognormal, Mittel exp(μ + s²/2)).
scikit-learn gehört nur in requirements-dev.txt (`pytest.importorskip`)."""

import math

import numpy as np
import pytest
from scipy import optimize, special, stats

import zae_asymptotics as AS
import zae_generator as G
import zae_models as M
import zae_rng as R
import zae_sample as S


def _world(seed, n=60, welt="NB2", alpha=0.5, c=0.0):
    X, y, _ = S.draw_world(seed, n, welt, alpha, c)
    return X, y


# --- IRLS gegen scikit-learn und BFGS -------------------------------------------------------------------------------------------------------------

def test_irls_equals_sklearn_poisson_regressor():
    linear_model = pytest.importorskip("sklearn.linear_model")
    for seed in range(20):
        X, y = _world(seed, 80, alpha=0.5)
        mine = M.irls(X, y).beta
        sk = linear_model.PoissonRegressor(alpha=0.0, solver="newton-cholesky", tol=1e-12, max_iter=500).fit(X[:, 1:], y)
        assert mine == pytest.approx([sk.intercept_, sk.coef_[0]], abs=1e-8), seed


def test_irls_equals_bfgs_with_analytic_gradient():
    for seed in range(10):
        X, y = _world(seed, 70, alpha=1.0)

        def nll(b):
            eta = X @ b
            return float(np.sum(np.exp(eta) - y * eta))

        def grad(b):
            return X.T @ (np.exp(X @ b) - y)

        res = optimize.minimize(nll, np.zeros(2), jac=grad, method="BFGS", options={"gtol": 1e-10})
        assert M.irls(X, y).beta == pytest.approx(res.x, abs=1e-6), seed


def test_offset_equals_sklearn_rate_with_weight_t():
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    t = rng.integers(1, 9, 60).astype(float)
    x = rng.normal(size=60)
    X = np.column_stack([np.ones(60), x])
    y = rng.poisson(t * np.exp(0.5 + 0.3 * x)).astype(float)
    mine = M.irls(X, y, offset=np.log(t)).beta
    sk = linear_model.PoissonRegressor(alpha=0.0, solver="newton-cholesky", tol=1e-12, max_iter=500).fit(X[:, 1:], y / t, sample_weight=t)
    assert mine == pytest.approx([sk.intercept_, sk.coef_[0]], abs=1e-8)


# --- NB2 ------------------------------------------------------------------------------------------------------------------------------------------

def test_nb2_loglik_equals_scipy_nbinom():
    rng = np.random.default_rng(5)
    for _ in range(20):
        mu = rng.uniform(0.2, 15, 40)
        y = rng.poisson(mu * rng.gamma(2.0, 0.5, 40)).astype(float)
        a = float(rng.uniform(0.05, 2.0))
        ref = float(np.sum(stats.nbinom.logpmf(y, 1 / a, 1 / (1 + a * mu))))
        assert M.nb2_loglik(y, mu, a) == pytest.approx(ref, abs=1e-10)


def _nll_joint(par, X, y):
    b, la = par[:-1], par[-1]
    mu = np.exp(np.clip(X @ b, -30, 30))
    a = math.exp(la)
    return -float(np.sum(stats.nbinom.logpmf(y, 1 / a, 1 / (1 + a * mu))))


def test_nb2_alpha_equals_grid_maximum_and_joint_maximisation():
    for seed in range(8):
        X, y = _world(seed, 120, alpha=0.5)
        nb = M.nb2_fit(X, y)
        mu = nb.mu
        grid = np.exp(np.linspace(math.log(1e-3), math.log(20), 4001))
        ll = [float(np.sum(stats.nbinom.logpmf(y, 1 / a, 1 / (1 + a * mu)))) for a in grid]
        assert nb.alpha == pytest.approx(grid[int(np.argmax(ll))], rel=2e-3), seed
        start = np.concatenate([nb.beta + 0.05, [math.log(nb.alpha) + 0.1]])
        res = optimize.minimize(_nll_joint, start, args=(X, y), jac="3-point", method="BFGS", options={"gtol": 1e-7})
        assert nb.beta == pytest.approx(res.x[:-1], abs=1e-3), seed
        assert nb.alpha == pytest.approx(math.exp(res.x[-1]), rel=1e-3), seed
        assert -res.fun == pytest.approx(M.nb2_loglik(y, nb.mu, nb.alpha), abs=1e-6)


def test_nb2_fit_maximises_the_likelihood_at_its_result():
    """Gegenprobe am eigenen Ergebnis: der Score nach β ist null, die Log-Likelihood fällt bei Verschiebung von α und von β."""
    X, y = _world(11, 150, alpha=1.0)
    nb = M.nb2_fit(X, y)
    assert np.max(np.abs(X.T @ ((y - nb.mu) / (1 + nb.alpha * nb.mu)))) < 1e-6
    base = M.nb2_loglik(y, nb.mu, nb.alpha)
    for f in (0.9, 1.1):
        assert M.nb2_loglik(y, nb.mu, nb.alpha * f) < base
    for d in (0.01, -0.01):
        assert M.nb2_loglik(y, np.exp(X @ (nb.beta + np.array([0.0, d]))), nb.alpha) < base


# --- Sandwich -------------------------------------------------------------------------------------------------------------------------------------

def test_sandwich_equals_loop_over_observations_and_dense_form():
    for seed in range(10):
        X, y = _world(seed, 50, alpha=0.5)
        inf = M.poisson_inference(X, y)
        mu, n, p = inf.fit.mu, len(y), 2
        A = sum(mu[i] * np.outer(X[i], X[i]) for i in range(n))
        Ai = np.linalg.inv(A)
        B0 = sum((y[i] - mu[i]) ** 2 * np.outer(X[i], X[i]) for i in range(n))
        h = np.array([mu[i] * X[i] @ Ai @ X[i] for i in range(n)])
        B3 = sum(((y[i] - mu[i]) / (1 - h[i])) ** 2 * np.outer(X[i], X[i]) for i in range(n))
        assert inf.cov_hc0 == pytest.approx(Ai @ B0 @ Ai, rel=1e-9)
        assert inf.cov_hc1 == pytest.approx(Ai @ B0 @ Ai * n / (n - p), rel=1e-9)
        assert inf.cov_hc3a == pytest.approx(Ai @ B3 @ Ai, rel=1e-9)
        assert inf.cov_naive == pytest.approx(Ai, rel=1e-9)


# --- Grenzwert der Überdeckung ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("welt,par", [("NB2", 0.25), ("NB2", 0.5), ("NB2", 1.0), ("NB1", 0.5), ("NB1", 1.46), ("NB2", 0.0)])
def test_limit_coverage_two_integration_routes_agree(welt, par):
    a = AS.limit_coverage(welt, par, method="gauss_hermite")
    b = AS.limit_coverage(welt, par, method="quad")
    for k in ("ratio", "phi", "cov_poisson", "cov_quasi", "sd_true"):
        assert a[k] == pytest.approx(b[k], abs=1e-6), (welt, par, k)


def test_limit_coverage_known_values():
    """Gemessene Werte (Gauss-Hermite, α = 0,25 / 0,5 / 1): Poisson 83,4 / 74,2 / 61,9 %, Quasi-Poisson 93,2 / 92,5 / 91,8 %; α = 0: exakt 95 %; NB1: Quasi exakt 95 %."""
    for a, cp, cq in ((0.25, 0.834, 0.932), (0.5, 0.742, 0.925), (1.0, 0.619, 0.918)):
        r = AS.limit_coverage("NB2", a)
        assert r["cov_poisson"] == pytest.approx(cp, abs=6e-4) and r["cov_quasi"] == pytest.approx(cq, abs=6e-4)
    r0 = AS.limit_coverage("NB2", 0.0)
    assert r0["cov_poisson"] == pytest.approx(0.95, abs=1e-9) and r0["cov_quasi"] == pytest.approx(0.95, abs=1e-9)
    for d in (0.5, 1.46):
        assert AS.limit_coverage("NB1", d)["cov_quasi"] == pytest.approx(0.95, abs=1e-9)


def test_limit_ratio_matches_the_variance_formula_at_alpha_zero():
    """Bei V = μ ist B = A, also wahre SD gleich naive SD."""
    assert AS.limit_coverage("NB2", 0.0)["ratio"] == pytest.approx(1.0, abs=1e-12)


# --- Zählziehung ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,par", [("nb2", 0.5), ("nb2", 1.0), ("nb1", 1.46), ("poisson", 0.0)])
def test_inversion_equals_scipy_ppf_for_nearly_all_draws(kind, par):
    u = R.uniform_array(17, 20000)
    mu = np.exp(1.0 + 0.4 * R.normal_array(18, 20000))
    y = S.counts_from_uniform(u, mu, kind, par)
    if kind == "nb2":
        ref = stats.nbinom.ppf(u, 1 / par, 1 / (1 + par * mu))
    elif kind == "nb1":
        ref = stats.nbinom.ppf(u, mu / par, 1 / (1 + par))
    else:
        ref = stats.poisson.ppf(u, mu)
    assert np.mean(y == ref) > 0.9995


@pytest.mark.parametrize("kind,par,var_fn", [("nb2", 0.5, lambda m: m + 0.5 * m ** 2), ("nb1", 1.46, lambda m: 2.46 * m), ("poisson", 0.0, lambda m: m)])
def test_count_moments_given_the_mean(kind, par, var_fn):
    """Mittel und Varianz der Zählung bei festem μ = 6 über 300 000 Ziehungen (Band 4 Standardfehler)."""
    n = 300000
    u = R.uniform_array(23, n)
    y = S.counts_from_uniform(u, np.full(n, 6.0), kind, par).astype(float)
    v = var_fn(6.0)
    assert abs(y.mean() - 6.0) < 4 * math.sqrt(v / n)
    assert abs(y.var() / v - 1) < 0.04


def test_delta_for_alpha_matches_mean_dispersion():
    """δ = α E[μ] mit E[μ] = exp(1 + 0,08): die mittlere Dispersion E[V/μ] ist in beiden Welten gleich 1 + α E[μ]."""
    assert S.delta_for_alpha(0.5) == pytest.approx(0.5 * math.exp(1.08), rel=1e-12)
    xs, ws = np.polynomial.hermite_e.hermegauss(80)
    ws = ws / ws.sum()
    mu = np.exp(1.0 + 0.4 * xs)
    assert np.sum(ws * (1 + 0.5 * mu)) == pytest.approx(np.sum(ws * (1 + S.delta_for_alpha(0.5))), rel=1e-12)


# --- Generator ------------------------------------------------------------------------------------------------------------------------------------

def test_generator_log_dwell_is_normal_around_the_oracle():
    """Orakel (geschlossene Verteilung): log d ~ N(mu, s²) mit dem bedingten (mu, s) je Container: die standardisierte Größe ist N(0, 1)."""
    X, d = G.make_history(40000, 5)
    mu, s = G.log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    z = (np.log(d) - mu) / s
    assert abs(z.mean()) < 0.025 and abs(z.std() - 1) < 0.02
    assert abs(np.mean(z ** 3)) < 0.06 and abs(np.mean(z ** 4) - 3) < 0.2
    assert np.all(d > 0) and np.all(G.standtage(d) >= 1)             # begonnene Tage: nie null


def test_generator_mean_dwell_equals_lognormal_mean_by_art():
    X, d = G.make_history(200000, 6)
    mu, s = G.log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    art = X[:, 0].astype(int)
    for a in range(4):
        m = art == a
        rel_se = math.sqrt(math.exp(s[m][0] ** 2) - 1) / math.sqrt(m.sum()) * 1.3      # Streuung des Mittels grob nach oben genommen (Kunden- und Merkmalsvarianz)
        assert d[m].mean() / np.exp(mu[m] + s[m] ** 2 / 2).mean() == pytest.approx(1.0, abs=5 * rel_se + 0.01), a


def test_generator_feature_shares():
    X, _ = G.make_history(100000, 7)
    art = X[:, 0].astype(int)
    for a, p in enumerate((0.35, 0.30, 0.25, 0.10)):
        assert abs(np.mean(art == a) - p) < 0.01
    assert abs(np.mean(X[art == 0, 2]) - 0.25) < 0.015           # Zollhalt bei Import
    assert np.all(X[art != 0, 2] == 0)
    assert abs(np.mean(X[:, 3]) - 2 / 7) < 0.01                  # Wochenende
    assert abs(np.std(X[:, 5]) - 1) < 0.02                       # Rauschmerkmal Gewicht ~ N(0, 1)
    assert np.all(X[art == 0, 4] == 0) and X[art == 1, 4].min() >= 1 and X[art == 1, 4].max() <= 10


def test_special_functions_used_by_the_profile_equation_agree_with_scipy():
    """Der Score nach r = 1/α am Maximum ist null: Gegenprobe der Nullstellensuche mit anderem Verfahren (Bisektion)."""
    X, y = _world(2, 200, alpha=1.0)
    f = M.irls(X, y)
    a, rand = M.nb2_alpha_given_mu(y, f.mu)
    assert not rand

    def score(lr):
        r = math.exp(lr)
        return float(np.sum(special.digamma(y + r) - special.digamma(r) + np.log(r / (r + f.mu)) + (f.mu - y) / (r + f.mu)))

    root = optimize.bisect(score, -6, 14, xtol=1e-12)
    assert a == pytest.approx(math.exp(-root), rel=1e-7)
