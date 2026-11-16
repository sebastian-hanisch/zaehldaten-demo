"""Kern: Zählmodelle mit Log-Link. IRLS für Poisson und für NB2 mit festem α, Pearson-Dispersion (Quasi-Poisson), Sandwich-Kovarianz (HC0, HC1 und die Fassung mit
Hebelkorrektur, hier „HC3a“), NB2 mit α aus der Profilgleichung. Nur numpy und scipy (scipy nur für Digamma und die Nullstellensuche).

Konvention: y_i ≥ 0 Zählwert, μ_i = exp(x_i'β + offset_i), p = Zahl der Spalten (inklusive Achsenabschnitt). Gewicht im IRLS: μ / (1 + α μ) (α = 0: Poisson).
Quasi-Poisson hat dieselben Koeffizienten wie Poisson und die Varianz φ̂ · Var_Poisson mit φ̂ = Σ (y − μ̂)² / μ̂ / (n − p).
Sandwich: A = X'WX (W = μ̂), B = Σ r_i² x_i x_i'; HC0 = A⁻¹ B A⁻¹, HC1 = n/(n − p) · HC0; „HC3a“ ersetzt r_i durch r_i / (1 − h_i) mit h_i = μ̂_i x_i' A⁻¹ x_i: ein
Analogon zu HC3 der linearen Regression, nicht dessen Definition von MacKinnon und White (1985, dort ist HC3 das Jackknife)."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

ETA_MAX = 30.0          # Kappung des linearen Prädiktors (wie in R): exp(30) ≈ 1e13, darüber Überlauf
ALPHA_MIN = 1e-8        # untere Grenze der NB2-Streuung; bei Unterdispersion hat die Profilgleichung keine Nullstelle (Randlösung bei α ≈ e^-14)
ALPHA_LOG_RANGE = (-6.0, 14.0)   # Suchbereich für log r = −log α in der Profilgleichung


class FitError(ValueError):
    """Anpassung nicht möglich (singuläre Matrix, nicht endliche Werte, Ereignislosigkeit); die App zeigt eine Meldung statt einer Zahl."""


@dataclass(frozen=True)
class PoissonFit:
    beta: np.ndarray
    mu: np.ndarray
    w: np.ndarray
    iterations: int
    converged: bool


@dataclass(frozen=True)
class NB2Fit:
    beta: np.ndarray
    alpha: float
    mu: np.ndarray
    cov: np.ndarray
    outer_iterations: int
    at_boundary: bool          # α̂ am Rand ALPHA_MIN (Randlösung, Unterdispersion)


def _solve(X, W, z):
    XtW = X.T * W
    try:
        return np.linalg.solve(XtW @ X, XtW @ z)
    except np.linalg.LinAlgError as exc:
        raise FitError("Die Informationsmatrix ist singulär (konstante oder linear abhängige Spalte, oder zu wenige Beobachtungen).") from exc


def irls(X, y, alpha=0.0, offset=None, beta0=None, maxit=50, tol=1e-10) -> PoissonFit:
    """IRLS mit Log-Link: Poisson (alpha = 0) oder NB2 mit festem alpha. Start: μ = y + 1/2 (nie 0), bzw. beta0.
    Abbruch, wenn die größte Koeffizientenänderung unter `tol` liegt; `converged` sagt, ob das erreicht wurde (bei Ereignislosigkeit einer Gruppe nicht)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    if len(y) < X.shape[1]:
        raise FitError("Weniger Beobachtungen als Spalten.")
    if beta0 is None:
        mu = y + 0.5
        W = mu / (1 + alpha * mu)
        z = np.log(mu) - off + (y - mu) / mu
        b = _solve(X, W, z)
        it0 = 1
    else:
        b = np.asarray(beta0, float)
        it0 = 0
    converged = False
    it = it0
    for it in range(it0 + 1, maxit + 1):
        eta = np.clip(X @ b + off, -ETA_MAX, ETA_MAX)
        mu = np.exp(eta)
        W = mu / (1 + alpha * mu)
        z = eta - off + (y - mu) / mu
        bn = _solve(X, W, z)
        step = float(np.max(np.abs(bn - b)))
        b = bn
        if not np.isfinite(step):
            raise FitError("Die Anpassung divergiert (nicht endliche Werte).")
        if step < tol:
            converged = True
            break
    eta = np.clip(X @ b + off, -ETA_MAX, ETA_MAX)
    mu = np.exp(eta)
    return PoissonFit(b, mu, mu / (1 + alpha * mu), it, converged)


def nb2_loglik(y, mu, alpha) -> float:
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def nb2_alpha_given_mu(y, mu) -> tuple[float, bool]:
    """α aus der Profilgleichung (Score nach r = 1/α, Nullstelle in log r). Gibt (α, am_Rand) zurück: hat der Score im ganzen Suchbereich dasselbe Vorzeichen, ist die
    Lösung eine Randlösung (kleiner Rand ALPHA_MIN bei Unterdispersion)."""
    y = np.asarray(y, float)

    def score(lr):
        r = np.exp(lr)
        return float(np.sum(special.digamma(y + r) - special.digamma(r) + np.log(r / (r + mu)) + (mu - y) / (r + mu)))

    lo, hi = ALPHA_LOG_RANGE
    if score(hi) > 0:                          # die Likelihood wächst noch bei r = e^14: Randlösung am Poisson-Ende (α ≈ 8e-7)
        return max(float(np.exp(-hi)), ALPHA_MIN), True
    if score(lo) < 0:                          # extrem stark überstreut: Randlösung am anderen Ende (α ≈ 403)
        return float(np.exp(-lo)), True
    lr = optimize.brentq(score, lo, hi, xtol=1e-10)
    return float(np.exp(-lr)), False


def nb2_fit(X, y, offset=None, maxout=30, tol=1e-9) -> NB2Fit:
    """NB2 mit Log-Link: β und α abwechselnd (wie `glm.nb` in R): IRLS für β bei festem α, Profilgleichung für α bei festem μ; Abbruch, wenn die Log-Likelihood
    sich relativ um weniger als `tol` ändert. Die Kovarianz ist die inverse Informationsmatrix (X'WX)⁻¹ mit W = μ/(1 + αμ) bei α̂ (β und α sind orthogonal)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    f = irls(X, y, 0.0, offset)
    a, rand = nb2_alpha_given_mu(y, f.mu)
    ll_old = -np.inf
    outer = 0
    for outer in range(1, maxout + 1):
        f = irls(X, y, a, offset, beta0=f.beta)
        a, rand = nb2_alpha_given_mu(y, f.mu)
        ll = nb2_loglik(y, f.mu, a)
        if abs(ll - ll_old) < tol * (abs(ll) + 1):
            break
        ll_old = ll
    f = irls(X, y, a, offset, beta0=f.beta)
    try:
        cov = np.linalg.inv((X.T * f.w) @ X)
    except np.linalg.LinAlgError as exc:
        raise FitError("Die Informationsmatrix ist singulär.") from exc
    return NB2Fit(f.beta, a, f.mu, cov, outer, rand)


@dataclass(frozen=True)
class PoissonInference:
    """Alles, was aus einer Poisson-Anpassung folgt (gleiche Koeffizienten für Poisson und Quasi-Poisson)."""
    fit: PoissonFit
    n: int
    p: int
    phi: float                 # Pearson-Dispersion
    cov_naive: np.ndarray
    cov_quasi: np.ndarray
    cov_hc0: np.ndarray
    cov_hc1: np.ndarray
    cov_hc3a: np.ndarray
    leverage: np.ndarray


def poisson_inference(X, y, offset=None, fit: PoissonFit | None = None) -> PoissonInference:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    if n <= p:
        raise FitError("Mehr Beobachtungen als Spalten nötig (n > p), sonst gibt es keine Dispersionsschätzung.")
    f = fit if fit is not None else irls(X, y, 0.0, offset)
    mu = f.mu
    try:
        Ainv = np.linalg.inv((X.T * mu) @ X)
    except np.linalg.LinAlgError as exc:
        raise FitError("Die Informationsmatrix ist singulär.") from exc
    r = y - mu
    phi = float(np.sum(r ** 2 / mu) / (n - p))
    meat0 = (X.T * r ** 2) @ X
    hc0 = Ainv @ meat0 @ Ainv
    h = mu * np.einsum("ij,jk,ik->i", X, Ainv, X)
    r3 = r / np.maximum(1.0 - h, 1e-8)           # h ≥ 1 kommt nur bei ausgenutzten Beobachtungen vor; die Kappung verhindert die Division durch 0
    hc3 = Ainv @ ((X.T * r3 ** 2) @ X) @ Ainv
    return PoissonInference(f, n, p, phi, Ainv, phi * Ainv, hc0, hc0 * (n / (n - p)), hc3, h)


METHODS = ("Poisson", "Quasi z", "Quasi t", "HC0", "HC1", "HC3a", "NB2")
METHOD_LABELS = {
    "Poisson": "Poisson (naiv)",
    "Quasi z": "Quasi-Poisson (z)",
    "Quasi t": "Quasi-Poisson (t)",
    "HC0": "Sandwich HC0",
    "HC1": "Sandwich HC1",
    "HC3a": "Sandwich mit Hebelkorrektur (HC3-Analogon)",
    "NB2": "Negativ binomial (NB2)",
}
Z975 = 1.959963984540054


def t_quantile(df: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, df))




def all_methods(X, y, cols, offset=None, with_nb2=True) -> dict:
    """Schätzung, Standardfehler und kritischer Wert für die Koeffizienten `cols` bei allen sieben Verfahren (Reihenfolge METHODS).
    Felder: est, se (Felder der Form (len(cols), 7), NaN wo ein Verfahren scheitert), crit (7 kritische Werte: z, bei Quasi t das t-Quantil mit n − p Freiheitsgraden),
    phi (Pearson-Dispersion), alpha (NB2) und alpha_am_rand, fehler (Verfahren → Grund). Scheitert schon die Poisson-Anpassung, steigt FitError auf."""
    inf = poisson_inference(X, y, offset)
    cols = list(cols)
    b = inf.fit.beta[cols]
    covs = (inf.cov_naive, inf.cov_quasi, inf.cov_quasi, inf.cov_hc0, inf.cov_hc1, inf.cov_hc3a)
    est = np.full((len(cols), 7), np.nan)
    se = np.full((len(cols), 7), np.nan)
    for m, cov in enumerate(covs):
        est[:, m] = b
        se[:, m] = np.sqrt(np.diag(cov)[cols])
    crit = np.array([Z975, Z975, t_quantile(inf.n - inf.p), Z975, Z975, Z975, Z975])
    out = {"est": est, "se": se, "crit": crit, "phi": inf.phi, "n": inf.n, "p": inf.p, "alpha": float("nan"), "alpha_am_rand": False, "fehler": {}, "inference": inf}
    if with_nb2:
        try:
            nb = nb2_fit(X, y, offset)
            est[:, 6] = nb.beta[cols]
            se[:, 6] = np.sqrt(np.diag(nb.cov)[cols])
            out["alpha"] = nb.alpha
            out["alpha_am_rand"] = nb.at_boundary
            out["nb2"] = nb
        except FitError as exc:
            out["fehler"]["NB2"] = str(exc)
    return out


def coefficient_table(X, y, j, offset=None, with_nb2=True) -> dict:
    """Für die App: Koeffizient j, je Verfahren {est, se, crit, lo, hi}; fehlende Verfahren stehen in 'fehler'."""
    r = all_methods(X, y, [j], offset, with_nb2)
    out = {"phi": r["phi"], "n": r["n"], "p": r["p"], "alpha": r["alpha"], "alpha_am_rand": r["alpha_am_rand"], "fehler": r["fehler"]}
    for m, name in enumerate(METHODS):
        if np.isfinite(r["est"][0, m]) and np.isfinite(r["se"][0, m]):
            e, s, c = float(r["est"][0, m]), float(r["se"][0, m]), float(r["crit"][m])
            out[name] = {"est": e, "se": s, "crit": c, "lo": e - c * s, "hi": e + c * s}
    return out
