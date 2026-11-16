"""Orakel ohne Zufall: die Überdeckung der Intervalle im Grenzwert n → ∞, aus der Sandwich-Formel Var(β̂) = A⁻¹ B A⁻¹ mit A = E[μ x x'] und B = E[V(μ) x x'] (x ~ N(0, 1),
Zeilen (1, x), μ = exp(β0 + β1 x), Varianzfunktion V). Zwei Rechenwege für die Erwartungswerte: Gauss-Hermite (80 Knoten) und `scipy.integrate.quad`.

Poisson-Intervall: die naive Varianz ist A⁻¹ (richtig nur für V = μ). Quasi-Poisson: das Skalar φ = E[V/μ] mal A⁻¹. Die z-Intervalle haben die Überdeckung
2 Φ(1,96 · SE_modell / SD_wahr) − 1, mit SD_wahr = √(A⁻¹BA⁻¹)₁₁ (wenn das Mittelwertmodell stimmt, ist √n (β̂₁ − β1) im Grenzwert normalverteilt mit Mittel 0 und der Varianz (A⁻¹BA⁻¹)₁₁;
B enthält die wahre Varianzfunktion V; SE_modell steht auf derselben Skala, also ohne den Faktor 1/n)."""

import math

import numpy as np
from scipy import integrate, stats

import zae_constants as C

Z975 = 1.959963984540054


def variance_function(mu, welt: str, par: float):
    """V(μ): NB2 μ + αμ² (par = α), NB1 (1 + δ) μ (par = δ)."""
    return mu + par * mu ** 2 if welt == "NB2" else (1.0 + par) * mu


def _moments_gauss_hermite(welt, par, beta):
    xs, ws = np.polynomial.hermite_e.hermegauss(80)
    ws = ws / ws.sum()
    mu = np.exp(beta[0] + beta[1] * xs)
    V = variance_function(mu, welt, par)
    Xg = np.column_stack([np.ones_like(xs), xs])
    A = (Xg.T * (ws * mu)) @ Xg
    B = (Xg.T * (ws * V)) @ Xg
    return A, B, float(np.sum(ws * V / mu))


def _moments_quad(welt, par, beta):
    pdf = stats.norm.pdf
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            A[i, j] = integrate.quad(lambda x: math.exp(beta[0] + beta[1] * x) * x ** (i + j) * pdf(x), -12, 12, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
            B[i, j] = integrate.quad(lambda x: variance_function(math.exp(beta[0] + beta[1] * x), welt, par) * x ** (i + j) * pdf(x), -12, 12, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    phi = integrate.quad(lambda x: variance_function(math.exp(beta[0] + beta[1] * x), welt, par) / math.exp(beta[0] + beta[1] * x) * pdf(x), -12, 12, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    return A, B, float(phi)


def limit_coverage(welt: str, par: float, beta=(C.BETA0, C.BETA1), method: str = "gauss_hermite") -> dict:
    """Grenzwert-Überdeckung des Anstiegs für Poisson (naiv) und Quasi-Poisson (Skalar φ). `par`: α (NB2) bzw. δ (NB1); par = 0 gibt für beide 0,95."""
    A, B, phi = (_moments_gauss_hermite if method == "gauss_hermite" else _moments_quad)(welt, par, beta)
    Ai = np.linalg.inv(A)
    S = Ai @ B @ Ai
    ratio = math.sqrt(S[1, 1] / Ai[1, 1])               # wahre SD / naive SD
    return {
        "ratio": ratio,
        "phi": phi,
        "cov_poisson": float(2 * stats.norm.cdf(Z975 / ratio) - 1),
        "cov_quasi": float(2 * stats.norm.cdf(Z975 * math.sqrt(phi) / ratio) - 1),
        "sd_true": math.sqrt(S[1, 1]),
    }
