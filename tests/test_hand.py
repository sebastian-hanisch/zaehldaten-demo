"""Von Hand in Brüchen: acht Standtage in zwei Gruppen, Gewichte, Exposition, Gamma-Mischung. Bruch gleich Bruch, wo es Brüche gibt; sonst Band."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import zae_hand as H

A, B = (0, 0, 1, 5), (1, 3, 6, 10)


def test_group_formulas_in_fractions():
    ex = H.hand_exact(A, B)
    assert ex["TA"] == 6 and ex["TB"] == 20 and ex["mA"] == Fr(3, 2) and ex["mB"] == 5
    assert ex["rssA"] == 17 and ex["rssB"] == 46
    assert ex["x2"] == Fr(308, 15) and ex["phi"] == Fr(154, 45)
    assert ex["var_naive"] == Fr(13, 60)
    assert ex["var_quasi"] == Fr(1001, 1350)
    assert ex["var_hc0"] == Fr(1057, 1800)
    assert ex["var_hc1"] == Fr(1057, 1350)


def test_sandwich_matrix_form_equals_group_formula_in_fractions():
    """A⁻¹ B A⁻¹ mit Brüchen (Matrixform) gleich der Gruppenformel RSS_A/T_A² + RSS_B/T_B²."""
    ex = H.hand_exact(A, B)
    rows = [(1, 0)] * 4 + [(1, 1)] * 4
    y = [Fr(v) for v in A + B]
    mu = [ex["mA"]] * 4 + [ex["mB"]] * 4
    I11, I12, I22 = ex["TA"] + ex["TB"], ex["TB"], ex["TB"]
    det = I11 * I22 - I12 * I12
    Ainv = [[I22 / det, -I12 / det], [-I12 / det, I11 / det]]
    meat = [[sum(((yy - m) ** 2) * x[i] * x[j] for x, yy, m in zip(rows, y, mu)) for j in range(2)] for i in range(2)]

    def mm(P, Q):
        return [[sum(P[i][k] * Q[k][j] for k in range(2)) for j in range(2)] for i in range(2)]

    S = mm(mm(Ainv, meat), Ainv)
    assert S[1][1] == ex["var_hc0"] == Fr(1057, 1800)


def test_numeric_part_of_the_hand_example():
    r = H.hand_numeric(A, B)
    assert r["beta"] == pytest.approx((math.log(1.5), math.log(10 / 3)), abs=1e-12)
    assert r["z"]["naive"] == pytest.approx(2.5863, abs=5e-4) and r["z"]["quasi"] == pytest.approx(1.3983, abs=5e-4)
    assert r["z"]["hc0"] == pytest.approx(1.5707, abs=5e-4) and r["z"]["hc1"] == pytest.approx(1.3603, abs=5e-4)
    assert r["z"]["quasi"] == pytest.approx(r["z"]["naive"] / math.sqrt(154 / 45), rel=1e-12)
    assert r["z"]["nb2"] == pytest.approx(1.67, abs=0.02)
    assert r["alpha_nb2"] == pytest.approx(0.606, abs=2e-3)
    assert r["irls_schritt1"] == pytest.approx([0.5, 3.5], abs=1e-12)
    assert r["iv"]["naive"][0] > 0 > r["iv"]["quasi"][0]          # das naive Intervall schließt 0 aus, das Quasi-Intervall (t, 6 Freiheitsgrade) nicht
    assert r["iv"]["quasi"] == pytest.approx((-0.90, 3.31), abs=0.01)
    assert r["iv"]["naive"] == pytest.approx((0.29, 2.12), abs=0.01)
    assert 5 <= r["irls_schritte"] <= 12                           # Schrittzahl nur als Band (hier gemessen: 9)


def test_nb2_alpha_equals_grid_maximum_of_the_scipy_likelihood():
    y = np.array(A + B, float)
    mu = np.array([1.5] * 4 + [5.0] * 4)
    grid = np.exp(np.linspace(np.log(1e-3), np.log(50), 20001))
    ll = [float(np.sum(stats.nbinom.logpmf(y, 1 / a, 1 / (1 + a * mu)))) for a in grid]
    assert H.hand_numeric(A, B)["alpha_nb2"] == pytest.approx(grid[int(np.argmax(ll))], rel=1e-3)


def test_weights_poisson_and_nb2_in_fractions():
    assert H.weight_ratio(Fr(20), Fr(1), Fr(0)) == 20
    assert H.weight_ratio(Fr(20), Fr(1), Fr(1, 2)) == Fr(30, 11)
    assert H.weight_ratio(Fr(20), Fr(1), Fr(1)) == Fr(40, 21)
    assert float(H.weight_ratio(Fr(20), Fr(1), Fr(1, 4))) == pytest.approx(4.167, abs=5e-4)


def test_exposure_example():
    r = H.exposure_example(21, 3, 20, 5)
    assert r["rate1"] == 7 and r["rate2"] == 4 and r["verhaeltnis"] == Fr(4, 7)
    assert r["beta"] == pytest.approx((math.log(7), math.log(4 / 7)), abs=1e-12)


def test_gamma_mixture_in_fractions():
    """Mittel 3, α = 1/2: r = 2, p = 2/5, P(k) = (k + 1)(4/25)(3/5)^k; Summe 1, Mittel 3, Varianz 15/2 (Brüche, 400 Glieder, Rest unter 1e-70)."""
    pm = H.nb_mixture_pmf(Fr(3), Fr(1, 2), 400)
    assert pm[0] == Fr(4, 25) and pm[1] == Fr(24, 125) and all(pm[k] == (k + 1) * Fr(4, 25) * Fr(3, 5) ** k for k in range(12))
    s0 = sum(pm)
    s1 = sum(k * v for k, v in enumerate(pm))
    s2 = sum(k * k * v for k, v in enumerate(pm))
    assert abs(float(s0) - 1) < 1e-70 and abs(float(s1) - 3) < 1e-60 and abs(float(s2 - s1 * s1) - 7.5) < 1e-50
    assert float(pm[0]) == 0.16 and math.exp(-3) < 0.0500          # Poisson mit Mittel 3 hat P(0) = 0,0498


def test_group_without_event_has_no_estimate():
    with pytest.raises(H.NoEventsError):
        H.hand_exact((0, 0, 0, 0), B)
    with pytest.raises(H.NoEventsError):
        H.hand_numeric(A, (0, 0, 0, 0))
    with pytest.raises(H.NoEventsError):
        H.exposure_example(0, 3, 20, 5)


def test_one_group_with_one_observation_each_has_no_dispersion():
    """n = p = 2: keine Freiheitsgrade, keine Dispersionsschätzung."""
    with pytest.raises(ValueError):
        H.hand_exact((3,), (4,))
