"""Zählwerte und Datensätze der Messwelt 1 (ein Merkmal x ~ N(0, 1)), nur aus dem ganzzahligen SplitMix64 (`zae_rng`).

Zählwerte per **Inversion der Verteilungsfunktion** aus einer Gleichverteilten, direkt aus der Wahrscheinlichkeitsfunktion (Rekursion P(k) = P(k−1) · Faktor): so braucht
die negativ binomiale Zählung keinen Gamma-Zieher. Verteilungen (alle mit Mittel μ):
  Poisson   P(0) = e^(−μ),                    P(k) = P(k−1) · μ / k                      Varianz μ
  NB2       r = 1/α, P(0) = (1 + αμ)^(−1/α),  P(k) = P(k−1) · (k − 1 + r) / k · αμ/(1+αμ)   Varianz μ + α μ²   (Gamma-Mischung der Poisson-Rate)
  NB1       r = μ/δ, P(0) = (1 + δ)^(−μ/δ),   P(k) = P(k−1) · (k − 1 + r) / k · δ/(1+δ)     Varianz (1 + δ) μ
Ein Grenzfall bleibt: `exp` und `log1p` aus der C-Bibliothek können in der letzten Stelle abweichen und einen Zählwert in etwa 1e-16 der Ziehungen umkippen; darum sind
alle „neu gerechnet“-Tests der Demo Bänder."""

import math

import numpy as np

import zae_constants as C
import zae_rng as R

K_MAX = 100000            # Obergrenze der Inversion (Schutz gegen Stillstand der Summe bei u nahe 1); in den Messwelten nie erreicht


def delta_for_alpha(alpha: float) -> float:
    """NB1-Streuung δ, die dieselbe mittlere Dispersion E[V/μ] = 1 + α E[μ] hat wie NB2 mit α (E[μ] = exp(β0 + β1²/2) für x ~ N(0, 1))."""
    return alpha * math.exp(C.BETA0 + C.BETA1 ** 2 / 2)


def counts_from_uniform(u: np.ndarray, mu: np.ndarray, kind: str, par: float = 0.0) -> np.ndarray:
    """Zählwerte per Inversion: kleinstes k mit u < F(k). kind: 'poisson' (par ignoriert), 'nb2' (par = α > 0), 'nb1' (par = δ > 0)."""
    u = np.asarray(u, float)
    mu = np.asarray(mu, float)
    if kind == "poisson" or (kind == "nb2" and par <= 0):
        p0 = np.exp(-mu)
        r = None
        q = None
    elif kind == "nb2":
        r = np.full(mu.shape, 1.0 / par)
        q = par * mu / (1.0 + par * mu)
        p0 = np.exp(-r * np.log1p(par * mu))
    elif kind == "nb1":
        r = mu / par
        q = np.full(mu.shape, par / (1.0 + par))
        p0 = np.exp(-r * math.log1p(par))
    else:
        raise ValueError(f"unbekannte Verteilung {kind!r}")
    y = np.zeros(u.shape, dtype=np.int64)
    cum = p0.copy()
    idx = np.nonzero(u >= cum)[0]
    pk = p0[idx]
    cm = cum[idx]
    mu_a = mu[idx]
    r_a = None if r is None else r[idx]
    q_a = None if q is None else q[idx]
    k = 0
    while idx.size and k < K_MAX:
        k += 1
        pk = pk * (mu_a / k if r_a is None else (k - 1 + r_a) / k * q_a)
        cm = cm + pk
        y[idx] = k
        keep = u[idx] >= cm
        if not keep.all():
            idx, pk, cm, mu_a = idx[keep], pk[keep], cm[keep], mu_a[keep]
            if r_a is not None:
                r_a, q_a = r_a[keep], q_a[keep]
    return y


def draw_world(seed: int, n: int, welt: str, alpha: float, c: float = 0.0):
    """Ein Datensatz der Messwelt 1: x ~ N(0, 1), μ = exp(1 + 0,4 x + c x²), Zählwert NB2 (Varianz μ + αμ²) bzw. NB1 (Varianz (1 + δ)μ mit δ = α·E[μ]); α = 0: Poisson.
    Gibt (X mit Spalten 1 und x, y, μ) zurück."""
    x = R.normal_array(seed, n)
    u = R.uniform_array(seed ^ 0x2545F4914F6CDD1D, n)
    mu = np.exp(C.BETA0 + C.BETA1 * x + c * x ** 2)
    if alpha <= 0:
        y = counts_from_uniform(u, mu, "poisson")
    elif welt == "NB2":
        y = counts_from_uniform(u, mu, "nb2", alpha)
    elif welt == "NB1":
        y = counts_from_uniform(u, mu, "nb1", delta_for_alpha(alpha))
    else:
        raise ValueError(f"unbekannte Welt {welt!r}")
    return np.column_stack([np.ones(n), x]), y.astype(float), mu
