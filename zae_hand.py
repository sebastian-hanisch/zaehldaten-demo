"""Von Hand: acht Standtage in zwei Gruppen (ohne und mit Zollhalt), die Standardfehler als exakte Brüche (`fractions`), dazu die Gewichte der Verfahren, die Exposition und die
Gamma-Mischung. Bei zwei Gruppen löst das Gruppenmittel die Score-Gleichung beider Gruppen (Σ (y − μ) = 0 je Gruppe); alles Weitere folgt in Brüchen.

Die Acht Werte sind **erfunden** (nicht aus dem Generator gezogen), weil nur so alles in Brüchen nachrechenbar bleibt."""

import math
from fractions import Fraction

import numpy as np
from scipy import stats

import zae_models as M

Fr = Fraction


class NoEventsError(ValueError):
    """Eine Gruppe ohne ein einziges Ereignis: der Poisson-Schätzer existiert nicht (β → −∞)."""


def _group_stats(a, b):
    A = [Fr(int(v)) for v in a]
    B = [Fr(int(v)) for v in b]
    return A, B


def hand_exact(a, b) -> dict:
    """Gruppenformeln in Brüchen für Gruppe A (ohne Zollhalt, Werte `a`) und Gruppe B (mit Zollhalt, Werte `b`), Modell 1 + Gruppenindikator.
    Gibt T (Summen), m (Mittel), Var(β̂₁) naiv / Quasi / HC0 / HC1, X², φ̂ als Fraction zurück."""
    A, B = _group_stats(a, b)
    nA, nB = len(A), len(B)
    if nA < 1 or nB < 1:
        raise ValueError("Jede Gruppe braucht mindestens eine Beobachtung.")
    TA, TB = sum(A), sum(B)
    if TA == 0 or TB == 0:
        raise NoEventsError("Eine Gruppe hat kein einziges Ereignis (Summe 0): der Poisson-Schätzer existiert nicht, β läuft gegen −∞.")
    mA, mB = TA / nA, TB / nB
    rssA = sum((y - mA) ** 2 for y in A)
    rssB = sum((y - mB) ** 2 for y in B)
    n, p = nA + nB, 2
    if n <= p:
        raise ValueError("Mehr Beobachtungen als Parameter nötig.")
    x2 = rssA / mA + rssB / mB
    phi = x2 / (n - p)
    var_naive = 1 / TA + 1 / TB
    var_hc0 = rssA / TA ** 2 + rssB / TB ** 2
    return {
        "nA": nA, "nB": nB, "TA": TA, "TB": TB, "mA": mA, "mB": mB, "rssA": rssA, "rssB": rssB, "x2": x2, "phi": phi,
        "var_naive": var_naive, "var_quasi": phi * var_naive, "var_hc0": var_hc0, "var_hc1": var_hc0 * Fr(n, n - p),
    }


def hand_numeric(a, b) -> dict:
    """Zahlen zum Handbeispiel: Koeffizienten (Gruppenmittel), z-Werte, Intervalle, NB2 (α aus der Profilgleichung, Var(β̂₁) = 1/T_A + 1/T_B + α (1/n_A + 1/n_B)) und die
    IRLS-Schritte aus β = 0. Wirft NoEventsError ohne Ereignis in einer Gruppe."""
    ex = hand_exact(a, b)
    mA, mB = float(ex["mA"]), float(ex["mB"])
    beta = (math.log(mA), math.log(mB / mA))
    se = {k: math.sqrt(float(ex["var_" + k])) for k in ("naive", "quasi", "hc0", "hc1")}
    n, p = ex["nA"] + ex["nB"], 2
    tcrit = float(stats.t.ppf(0.975, n - p))
    z975 = M.Z975
    y = np.array([float(v) for v in a] + [float(v) for v in b])
    mu = np.array([mA] * ex["nA"] + [mB] * ex["nB"])

    alpha, rand = M.nb2_alpha_given_mu(y, mu)
    var_nb = float(ex["var_naive"]) + alpha * (1 / ex["nA"] + 1 / ex["nB"])
    se["nb2"] = math.sqrt(var_nb)
    X = np.array([[1.0, 0.0]] * ex["nA"] + [[1.0, 1.0]] * ex["nB"])
    beta_t = np.zeros(2)
    pfad = []
    for _ in range(40):
        eta = X @ beta_t
        m_ = np.exp(eta)
        z_ = eta + (y - m_) / m_
        nb = np.linalg.solve((X.T * m_) @ X, (X.T * m_) @ z_)
        pfad.append(nb.copy())
        done = float(np.max(np.abs(nb - beta_t))) < 1e-13
        beta_t = nb
        if done:
            break
    b1 = beta[1]
    z = {k: (b1 / v if v > 0 else float("nan")) for k, v in se.items()}          # Standardfehler 0 (alle Reste null): z nicht definiert
    crit = {"naive": z975, "quasi": tcrit, "hc0": z975, "hc1": z975, "nb2": z975}
    iv = {k: (b1 - crit[k] * se[k], b1 + crit[k] * se[k]) for k in se}
    return {"exact": ex, "beta": beta, "se": se, "z": z, "crit": crit, "iv": iv, "alpha_nb2": alpha, "alpha_am_rand": rand,
            "irls_schritte": len(pfad), "irls_schritt1": [float(v) for v in pfad[0]], "faktor": mB / mA}


def weight_ratio(mu_gross, mu_klein, alpha):
    """Verhältnis der IRLS-Gewichte μ/(1 + αμ) zweier Fälle (exakt, wenn alle Argumente Fraction sind). Poisson und Quasi-Poisson: α = 0 gibt μ_groß/μ_klein."""
    return (mu_gross / (1 + alpha * mu_gross)) / (mu_klein / (1 + alpha * mu_klein))


def exposure_example(y1, t1, y2, t2) -> dict:
    """Exposition: Kunde 1 mit y1 Standtagen bei t1 Containern, Kunde 2 mit y2 bei t2. Poisson mit Offset log t gleich Rate: β = (ln(y1/t1), ln((y2/t2)/(y1/t1)))."""
    if y1 <= 0 or y2 <= 0 or t1 <= 0 or t2 <= 0:
        raise NoEventsError("Standtage und Container müssen positiv sein (ein Kunde ohne Standtage hat die Rate 0, β läuft gegen −∞).")
    r1, r2 = Fr(y1, t1), Fr(y2, t2)
    return {"rate1": r1, "rate2": r2, "verhaeltnis": r2 / r1, "beta": (math.log(r1), math.log(r2 / r1))}


def nb_mixture_pmf(mean: Fraction, alpha: Fraction, kmax: int) -> list:
    """Gamma-Mischung der Poisson-Rate = NB2: Wahrscheinlichkeiten P(0..kmax) in Brüchen. r = 1/α muss ganzzahlig sein (Mittel 3, α = 1/2: r = 2, p = 2/5, P(k) = (k + 1)(4/25)(3/5)^k)."""
    r = 1 / alpha
    if r.denominator != 1:
        raise ValueError("1/α muss ganzzahlig sein.")
    r = int(r)
    p = 1 / (1 + alpha * mean)
    q = 1 - p
    out = []
    for k in range(kmax + 1):
        out.append(Fr(math.comb(k + r - 1, k)) * p ** r * q ** k)
    return out
