"""Claims-Tests: jede Zahl der Befunde der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Plattform (Lehre 2026-11-15): alle Überdeckungs-, Fehlalarm- und Nullzahlen sind Bänder (Monte-Carlo-Standardfehler mal 3 oder mehr), nie exakt; exakt nur, was in Brüchen oder `fractions`
gerechnet ist (tests/test_hand.py). Die schweren Messreihen stehen vorgerechnet in `data/zae_frozen.json` (`tools/zae_make_frozen.py`); dieser Test liest sie, und eine kleine Neu-Rechnung
(unten, Abschnitt B17) prüft, dass die Datei zum Code der Suite passt. Zufall nur aus SplitMix64.
Zähleinheit: Wiederholung = ein Datensatz bzw. eine Historie aus n Containern; Ziel der Intervalle im Container-Vehikel ist der Grenzwert des jeweiligen Schätzers."""

import math
from pathlib import Path

import numpy as np
import pytest

import zae_asymptotics as AS
import zae_constants as C
import zae_evaluation as ev
import zae_simulation as SIM

F = ev.load_frozen()
CELLS = F["grid"]
M = ["Poisson", "Quasi z", "Quasi t", "HC0", "HC1", "HC3a", "NB2"]


def cell(welt, alpha, n):
    return ev.grid_cell(F, welt, alpha, n)


def within(cov, target, cell_, k=4.0, extra=0.0):
    return abs(cov - target) <= k * cell_["se_mc"] + extra


# --- B1/B2: Poisson-Intervall und Grenzwert -----------------------------------------------------------------------------------------------------------

def test_b1_poisson_coverage_over_alpha_at_n_200():
    """NB2-Welt, n = 200: α = 0 / 0,25 / 0,5 / 1 gibt Poisson 94,8 / 84,0 / 74,1 / 63,6 % (4000, 4000, 20 000, 4000 Wiederholungen; Band 4 Standardfehler plus 1 Prozentpunkt)."""
    for a, want in ((0.0, 0.948), (0.25, 0.840), (0.5, 0.741), (1.0, 0.636)):
        c = cell("NB2", a, 200)
        assert within(c["coverage"]["Poisson"], want, c, extra=0.01), a
    covs = [cell("NB2", a, 200)["coverage"]["Poisson"] for a in (0.0, 0.25, 0.5, 1.0)]
    assert covs == sorted(covs, reverse=True)


def test_b1_main_cell_is_in_the_band_74_to_76_percent_within_monte_carlo_error():
    c = cell("NB2", 0.5, 200)
    assert c["R"] == 20000
    assert 0.72 <= c["coverage"]["Poisson"] <= 0.77        # die Zahl der README ist das Band „74 bis 76 %“ (frühere Messung mit 2000 Läufen: 76,1 %; dort 10 000 Läufe: 74,4 %)


def test_b2_limit_equals_formula_and_simulation_is_close():
    """Grenzwert aus der Sandwich-Formel: Poisson 83,4 / 74,2 / 61,9 %; die Simulation bei n = 200 liegt innerhalb von 4 Standardfehlern plus 1,5 Prozentpunkten (endliches n)."""
    for a, lim in ((0.25, 0.834), (0.5, 0.742), (1.0, 0.619)):
        o = AS.limit_coverage("NB2", a)
        assert o["cov_poisson"] == pytest.approx(lim, abs=6e-4)
        c = cell("NB2", a, 200)
        assert within(c["coverage"]["Poisson"], o["cov_poisson"], c, extra=0.015)


def test_b2_poisson_is_below_90_percent_in_all_18_cells_with_overdispersion():
    cs = [c for c in CELLS if c["alpha"] > 0]
    assert len(cs) == 18
    assert all(c["coverage"]["Poisson"] < 0.90 for c in cs)
    assert min(c["coverage"]["Poisson"] for c in cs) > 0.60 and max(c["coverage"]["Poisson"] for c in cs) < 0.88     # 63,6 bis 86,6 %


def test_b2_without_overdispersion_all_methods_hold_93_percent_at_n_200():
    """α = 0, n = 200: alle sieben Verfahren zwischen 93,5 und 95,5 % (gemessen 93,8 bis 95,2); das Preset „Keine Überstreuung“ meldet darum keinen Effekt."""
    c = cell("NB2", 0.0, 200)
    assert all(0.935 <= v <= 0.955 for v in c["coverage"].values())
    kind, text = ev.verdict_cover(c, ev.limit_cached("NB2", 0.0), "NB2", 0.0, 200)
    assert kind == "success" and "keinen Befund" in text


# --- B3: Quasi-Poisson, Grenzwert und NB1-Welt ----------------------------------------------------------------------------------------------------------

def test_b3_quasi_poisson_does_not_hold_95_in_the_limit_for_nb2_variance():
    for a, want in ((0.25, 0.932), (0.5, 0.925), (1.0, 0.918)):
        assert AS.limit_coverage("NB2", a)["cov_quasi"] == pytest.approx(want, abs=6e-4)
    nb2_cells = [c for c in CELLS if c["welt"] == "NB2" and c["alpha"] > 0]
    assert len(nb2_cells) == 9
    assert all(0.915 <= c["coverage"]["Quasi z"] <= 0.935 for c in nb2_cells)          # 91,9 bis 93,2 %, in keiner Zelle 95 %


def test_b3_in_the_nb1_world_quasi_poisson_is_exact_in_the_limit_and_close_in_simulation():
    for d in (0.5, 1.46, ev.welt_par("NB1", 0.5)):
        assert AS.limit_coverage("NB1", d)["cov_quasi"] == pytest.approx(0.95, abs=1e-9)
    n200 = [cell("NB1", a, 200) for a in (0.25, 0.5, 1.0)]
    assert [round(100 * c["coverage"]["Quasi z"], 1) for c in n200] == pytest.approx([94.8, 94.8, 93.7], abs=1.5)
    assert all(c["coverage"]["Quasi z"] > 0.93 for c in n200)


def test_b3_count_of_cells_within_two_standard_errors_of_95_percent():
    """Zählung über die 18 Zellen mit α > 0 (Abstand höchstens 2 Monte-Carlo-Standardfehler von 95 %): Poisson 0, HC0 0, HC1 0, HC3a 3, Quasi z 4, Quasi t 7, NB2 7 (Band ± 2)."""
    cs = [c for c in CELLS if c["alpha"] > 0]
    count = {m: sum(abs(c["coverage"][m] - 0.95) <= 2 * c["se_mc"] for c in cs) for m in M}
    assert count["Poisson"] == 0 and count["HC0"] == 0 and count["HC1"] == 0
    for m, want in (("HC3a", 3), ("Quasi z", 4), ("Quasi t", 7), ("NB2", 7)):
        assert abs(count[m] - want) <= 2, (m, count[m])
    assert max(count.values()) < 18                                  # kein Verfahren hält das Niveau in allen Zellen


# --- B4: kleine Stichprobe -----------------------------------------------------------------------------------------------------------------------------

def test_b4_hc0_undercovers_at_n_30_and_the_leverage_correction_lifts_it():
    cs = [c for c in CELLS if c["alpha"] > 0 and c["n"] == 30]
    assert len(cs) == 6
    assert all(c["coverage"]["HC0"] < c["coverage"]["HC3a"] - 0.02 for c in cs)                      # 6 von 6 Zellen
    assert all(0.85 <= c["coverage"]["HC0"] <= 0.91 for c in cs)                                    # 86,3 bis 89,9 %
    assert all(0.905 <= c["coverage"]["HC3a"] <= 0.945 for c in cs)                                 # 91,4 bis 94,1 %
    c = cell("NB2", 0.5, 30)
    assert 0.86 <= c["coverage"]["HC0"] <= 0.90 and 0.91 <= c["coverage"]["HC3a"] <= 0.945          # 88,0 und 92,8 %
    kind, text = ev.verdict_cover(c, ev.limit_cached("NB2", 0.5), "NB2", 0.5, 30)
    assert "Ausnahme im selben Satz" in text and "HC0" in text


# --- B5: Quasi-Poisson gegen NB2 -----------------------------------------------------------------------------------------------------------------------

def test_b5_nb2_is_ahead_in_the_nb2_world_and_not_in_the_nb1_world_at_alpha_05_n_200():
    c2, c1 = cell("NB2", 0.5, 200), cell("NB1", 0.5, 200)
    d2 = c2["coverage"]["NB2"] - c2["coverage"]["Quasi z"]
    d1 = c1["coverage"]["NB2"] - c1["coverage"]["Quasi z"]
    assert d2 > 2 * c2["se_mc"] and 0.015 < d2 < 0.035          # 2,3 Prozentpunkte (94,5 gegen 92,2)
    assert abs(d1) < 2 * c1["se_mc"] + 0.002                      # NB1-Welt: 94,7 gegen 94,8, kein Abstand über zwei Standardfehlern
    text = ev.verdict_two_worlds(c2, c1, 0.5, 200)[1]
    assert "NB2 liegt vorn" in text and "kein Abstand über zwei Standardfehlern" in text


def test_b5_weights_poisson_20_nb2_2_727():
    assert ev.weight_ratio_value(0.5) == pytest.approx(30 / 11, rel=1e-12) and float(ev.weight_ratio_value(1.0)) == pytest.approx(40 / 21, rel=1e-12)
    assert float(ev.weight_ratio_value(0.25)) == pytest.approx(4.1667, abs=5e-4)


def test_b5_rmse_ratio_poisson_over_nb2_is_small_with_correct_mean_model():
    """RMSE-Verhältnis Poisson/NB2: α = 0 → 1,000; α = 0,5 → 1,055 (n = 200, 20 000 Wdh.); α = 1 → 1,078; n = 30, α = 0,5: 1,011; Verzerrung beider Schätzer unter 0,01 in Betrag (n ≥ 100)."""
    assert cell("NB2", 0.0, 200)["rmse_ratio"] == pytest.approx(1.0, abs=0.01)
    assert cell("NB2", 0.5, 200)["rmse_ratio"] == pytest.approx(1.055, abs=0.02)
    assert cell("NB2", 1.0, 200)["rmse_ratio"] == pytest.approx(1.078, abs=0.03)
    assert cell("NB2", 0.5, 30)["rmse_ratio"] == pytest.approx(1.011, abs=0.03)
    for c in CELLS:
        if c["welt"] == "NB2" and c["n"] >= 100:
            assert abs(c["bias"]["Poisson"]) < 0.01 and abs(c["bias"]["NB2"]) < 0.01


def test_b5_nb2_estimate_has_outlier_datasets_at_alpha_1_n_30():
    """Negativbefund: bei α = 1, n = 30 hat die NB2-Schätzung einzelne Ausreißer-Datensätze (SD über 10 gegen 0,24 bei Poisson); das RMSE-Verhältnis ist dort wertlos und steht nicht in der README."""
    c = cell("NB2", 1.0, 30)
    assert c["sd_est"]["NB2"] > 10 and c["sd_est"]["Poisson"] < 0.5


def test_b5_nb2_degenerate_replicate():
    """Die Standardabweichung 79,6 (NB2, α = 1, n = 30, 4000 Wiederholungen) gibt im Wesentlichen einen einzigen Datensatz wieder: Wiederholung 3480 hat die Randlösung α̂ ≈ 403 und β̂₁ ≈ 5035 (Abbruchregel für α̂); 5035/√4000 ≈ 79,6."""
    import zae_models as MOD
    import zae_rng as RNG
    import zae_sample as SMP
    c = cell("NB2", 1.0, 30)
    assert c["R"] == 4000 and c["seed"] == 8110
    X, y, _ = SMP.draw_world(RNG.rep_seed(c["seed"], 3480), 30, "NB2", 1.0)
    fit = MOD.nb2_fit(X, y)
    assert fit.at_boundary and fit.alpha > 100 and 1000 < abs(fit.beta[1]) < 20000
    assert c["sd_est"]["NB2"] == pytest.approx(abs(fit.beta[1]) / math.sqrt(c["R"]), rel=0.05)


def test_b5_boundary_solutions_for_alpha_zero():
    """α = 0, n = 200: α̂ liegt in 2253 von 4000 Datensätzen am Rand (Randlösung, Band ± 5 %), bei α = 0,5 in keinem; keine Anpassung scheiterte."""
    c = cell("NB2", 0.0, 200)
    assert 0.45 <= c["nb2_randloesung"] / c["R"] <= 0.68
    assert cell("NB2", 0.5, 200)["nb2_randloesung"] == 0
    assert all(c["nb2_ausfaelle"] == 0 and c["poisson_ausfaelle"] == 0 for c in CELLS)
    assert sum(c["R"] for c in CELLS) == 106000                                    # Zahl der Wiederholungen des Rasters (README B9)


# --- B6: falsches Mittelwertmodell ----------------------------------------------------------------------------------------------------------------------

def test_b6_quasi_and_poisson_share_coefficients_but_nb2_differs_under_misspecification():
    lim = {e["c"]: e for e in F["missspec"]["limit"]}
    assert lim[0.0]["poisson"] == pytest.approx(0.4, abs=0.02) and lim[0.0]["nb2"] == pytest.approx(0.4, abs=0.02)
    assert lim[0.1]["poisson"] == pytest.approx(0.505, abs=0.03) and lim[0.1]["nb2"] == pytest.approx(0.436, abs=0.03)
    assert lim[0.2]["poisson"] == pytest.approx(0.678, abs=0.06) and lim[0.2]["nb2"] == pytest.approx(0.466, abs=0.03)
    assert lim[0.3]["nb2"] == pytest.approx(0.461, abs=0.04)
    assert 0.75 < lim[0.3]["poisson"] < 1.25                       # nicht stabil: μ² hat für c ≥ 0,25 keinen endlichen Erwartungswert (frühere Messung mit 10⁶: 1,04)
    assert all(lim[c]["poisson"] > lim[c]["nb2"] for c in (0.1, 0.2, 0.3))


def test_b6_poisson_limit_closed_form():
    """Grenzwert des Poisson-Anstiegs bei wahrem log μ = 1 + 0,4 x + c x², x ~ N(0, 1): aus E[μ] = E[exp(b0 + b1 x)] und E[x μ] = E[x exp(b0 + b1 x)] folgt b1 = 0,4/(1 − 2c); er existiert auch bei c = 0,3 (= 1,0).
    Gegenprobe: die Schätzgleichungen per Quadratur lösen (scipy) und die vorgerechneten Messwerte vergleichen."""
    from scipy import integrate, optimize

    def gleichungen(par, c):
        b0, b1 = par
        dichte = lambda x: math.exp(1 + 0.4 * x + (c - 0.5) * x * x) / math.sqrt(2 * math.pi)        # μ(x)·φ(x) in einem Exponenten (kein Überlauf)
        m0 = integrate.quad(dichte, -60, 60, limit=200)[0]
        m1 = integrate.quad(lambda x: x * dichte(x), -60, 60, limit=200)[0]
        e = math.exp(b0 + b1 * b1 / 2)
        return [m0 - e, m1 - b1 * e]

    lim = {e["c"]: e for e in F["missspec"]["limit"]}
    for c, tol in ((0.0, 0.02), (0.1, 0.03), (0.2, 0.06), (0.3, 0.1)):
        b1 = optimize.fsolve(gleichungen, [1.0, 0.4], args=(c,), xtol=1e-12)[1]
        assert b1 == pytest.approx(0.4 / (1 - 2 * c), abs=1e-6)
        if c < 0.3:
            assert lim[c]["poisson"] == pytest.approx(b1, abs=tol)
    assert abs(lim[0.3]["poisson"] - 1.0) < 0.1                                    # beide Messungen (0,970 und 1,038) liegen nahe 1,000
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "0,4/(1 − 2c)" in readme and "1,000 bei c = 0,3" in readme


def test_b6_n_200_distribution_of_the_difference():
    d = {e["c"]: e for e in F["missspec"]["n200"]}
    assert d[0.0]["diff_median"] == pytest.approx(0.002, abs=0.01)
    assert d[0.2]["diff_median"] == pytest.approx(-0.141, abs=0.02) and d[0.2]["diff_q1"] == pytest.approx(-0.203, abs=0.03) and d[0.2]["diff_q3"] == pytest.approx(-0.096, abs=0.03)
    assert d[0.2]["poisson_robust_sd"] == pytest.approx(0.152, abs=0.02)
    assert 3 <= d[0.2]["diff_gross"] <= 20                                   # 8 von 3000 Datensätzen mit Unterschied über 1 (Band)
    assert d[0.2]["R"] == 3000 and all(e["ausfaelle"] == 0 for e in d.values())
    assert 150 <= d[0.3]["diff_gross"] <= 400 and d[0.3]["diff_median"] == pytest.approx(-0.252, abs=0.03)     # c = 0,3: 268 von 3000 Datensätzen mit Unterschied über 1


# --- B7: Container-Vehikel -----------------------------------------------------------------------------------------------------------------------------

def test_b7_dispersion_of_standtage():
    v = F["vehicle"]["limit"]
    lim, lim2, limk = v["ohne_kunden"], v["ohne_kunden_zweiter_seed"], v["mit_kunden"]
    assert lim["n"] == 1_000_000 and lim["min"] == 1.0
    assert lim["mean"] == pytest.approx(6.15, abs=0.15) and lim["var"] == pytest.approx(41.7, rel=0.08)
    assert lim["var"] / lim["mean"] == pytest.approx(6.8, abs=0.6)
    assert lim["phi"] == pytest.approx(3.12, abs=0.2) and lim2["phi"] == pytest.approx(lim["phi"], abs=0.15) and limk["phi"] == pytest.approx(2.46, abs=0.2)
    pa = lim["phi_je_art"]
    assert pa[0] == pytest.approx(3.05, abs=0.3) and pa[1] == pytest.approx(0.43, abs=0.05) and pa[2] == pytest.approx(0.44, abs=0.05) and pa[3] == pytest.approx(18.0, abs=1.5)
    assert sum(p < 1 for p in pa) == 2                                      # in zwei von vier Arten ist die Streuung kleiner als Poisson


def test_b7_pseudo_true_coefficients_are_stable_across_two_seeds():
    v = F["vehicle"]["limit"]
    a, b = v["ohne_kunden"], v["ohne_kunden_zweiter_seed"]
    for col in (C.COL_ZOLL, C.COL_WE, C.COL_GEWICHT):
        assert abs(a["beta_poisson"][col] - b["beta_poisson"][col]) < 0.015
        assert abs(a["beta_nb2"][col] - b["beta_nb2"][col]) < 0.015
    assert a["beta_poisson"][C.COL_ZOLL] == pytest.approx(0.649, abs=0.02) and a["beta_nb2"][C.COL_ZOLL] == pytest.approx(0.648, abs=0.02)
    assert a["beta_poisson"][C.COL_WE] == pytest.approx(0.160, abs=0.02) and a["beta_nb2"][C.COL_WE] == pytest.approx(0.145, abs=0.02)
    assert abs(a["beta_poisson"][C.COL_GEWICHT]) < 0.005 and abs(a["beta_nb2"][C.COL_GEWICHT]) < 0.005     # Rauschmerkmal: Grenzwert 0


def test_b7_coverage_at_3000_containers_without_customers():
    c = ev.veh_cell(F, 3000, False)
    assert c["R"] == 2000 and c["phi_mean"] == pytest.approx(3.10, abs=0.25)
    z, w, g = c["spalten"]["Zoll"], c["spalten"]["Wochenende"], c["spalten"]["Gewicht (Rauschmerkmal)"]
    tol = 4 * c["se_mc"] + 0.01
    for m, want in (("Poisson", 0.689), ("Quasi z", 0.916), ("HC0", 0.953), ("NB2", 0.872)):
        assert abs(z[m]["coverage"] - want) <= tol, m
    for m, want in (("Poisson", 0.596), ("Quasi z", 0.842), ("HC0", 0.949), ("NB2", 0.873)):
        assert abs(w[m]["coverage"] - want) <= tol, m
    for m, want in (("Poisson", 0.382), ("Quasi z", 0.135), ("HC0", 0.047), ("NB2", 0.116)):
        assert abs(g[m]["fehlalarm"] - want) <= 0.03, m
    for m, want in (("Poisson", 0.51), ("Quasi z", 0.88), ("HC0", 1.01), ("NB2", 0.77)):
        assert z[m]["ratio"] == pytest.approx(want, abs=0.03), m


def test_b7_ranking_of_nb2_and_quasi_depends_on_the_coefficient():
    """Zählung über die vier Zellen (n = 3000 und 300, ohne und mit Kunden): beim Wochenende liegt NB2 über Quasi-Poisson in 4 von 4, bei Zoll darunter in 3 von 4
    (Ausnahme: n = 300 mit Kunden, 89,5 gegen 87,4 %). Die Rangfolge hängt also am Koeffizienten; „NB2 ist besser“ und „Quasi-Poisson ist besser“ gelten beide nicht allgemein."""
    zoll_unter, we_ueber = 0, 0
    for vn in C.VN_CHOICES:
        for kunden in (False, True):
            c = ev.veh_cell(F, vn, kunden)["spalten"]
            zoll_unter += c["Zoll"]["NB2"]["coverage"] < c["Zoll"]["Quasi z"]["coverage"]
            we_ueber += c["Wochenende"]["NB2"]["coverage"] > c["Wochenende"]["Quasi z"]["coverage"]
    assert zoll_unter == 3 and we_ueber == 4
    a = ev.veh_cell(F, 3000, False)["spalten"]
    assert a["Zoll"]["NB2"]["coverage"] < a["Zoll"]["Quasi z"]["coverage"] and a["Wochenende"]["NB2"]["coverage"] > a["Wochenende"]["Quasi z"]["coverage"]
    assert a["Wochenende"]["NB2"]["sd"] < a["Wochenende"]["Poisson"]["sd"]                   # NB2 hat die kleinere Streuung des Schätzers (0,031 gegen 0,041), schätzt aber eine andere Größe


def test_b7_noise_feature_false_alarm_in_all_four_cells():
    for vn in C.VN_CHOICES:
        for kunden in (False, True):
            g = ev.veh_cell(F, vn, kunden)["spalten"]["Gewicht (Rauschmerkmal)"]
            assert 0.25 < g["Poisson"]["fehlalarm"] < 0.45 and 0.10 < g["Quasi z"]["fehlalarm"] < 0.20 and 0.07 < g["NB2"]["fehlalarm"] < 0.13 and g["HC0"]["fehlalarm"] < 0.08
            assert g["HC3a"]["fehlalarm"] < 0.06                           # nominal 5 %


def test_b7_customers_in_the_model_lower_phi_but_do_not_repair_poisson():
    a, b = ev.veh_cell(F, 3000, False), ev.veh_cell(F, 3000, True)
    assert b["phi_mean"] < a["phi_mean"] - 0.4 and b["phi_mean"] == pytest.approx(2.38, abs=0.25)
    for k in ("Zoll", "Wochenende"):
        assert b["spalten"][k]["Poisson"]["coverage"] < 0.78                  # mit Kunden unter 78 % (72,7 und 60,5 %)
    assert b["spalten"]["Zoll"]["HC0"]["coverage"] > 0.93


def test_b7_hc0_at_300_containers_is_below_the_leverage_correction():
    c = ev.veh_cell(F, 300, False)["spalten"]
    assert c["Zoll"]["HC0"]["coverage"] < c["Zoll"]["HC3a"]["coverage"] and c["Wochenende"]["HC0"]["coverage"] < c["Wochenende"]["HC3a"]["coverage"]
    assert c["Zoll"]["HC3a"]["coverage"] == pytest.approx(0.939, abs=0.02)


def test_b7_phi_hat_scatters_widely_between_histories():
    """φ̂ je Historie (n = 3000): Mittel 3,10, Streuung 0,90 (2000 Historien), mit Kunden 2,38 ± 0,60: die Streuung hängt am Heavy Tail der Art „Leer“."""
    a, b = ev.veh_cell(F, 3000, False), ev.veh_cell(F, 3000, True)
    assert a["phi_sd"] == pytest.approx(0.90, abs=0.2) and b["phi_sd"] == pytest.approx(0.60, abs=0.15)
    a3, b3 = ev.veh_cell(F, 300, False), ev.veh_cell(F, 300, True)                  # n = 300: 2,75 ± 1,38 und 1,93 ± 0,70
    assert a3["phi_mean"] == pytest.approx(2.75, abs=0.25) and b3["phi_mean"] == pytest.approx(1.93, abs=0.25)
    assert a3["phi_sd"] == pytest.approx(1.38, abs=0.3) and b3["phi_sd"] == pytest.approx(0.70, abs=0.2)


# --- B8: Nullen ------------------------------------------------------------------------------------------------------------------------------------------

def test_b8_standtage_have_no_zeros_and_standgeld_tage_do():
    assert F["vehicle"]["limit"]["ohne_kunden"]["min"] == 1.0
    from zae_generator import make_history, standgeld_tage, standtage
    _, d = make_history(200000, 3)
    assert standtage(d).min() >= 1 and np.mean(standgeld_tage(d) == 0) > 0.7


def test_b8_zero_shares_of_standgeld_tage():
    z = F["nullen"]
    t = z["ohne_kunden"]
    assert t["R"] == 300 and t["beobachtet"] == pytest.approx(0.766, abs=0.01)
    assert t["poisson_erwartet"] == pytest.approx(0.562, abs=0.02) and t["nb2_erwartet"] == pytest.approx(0.758, abs=0.015)
    assert t["phi"] == pytest.approx(6.4, abs=0.6) and z["mit_kunden"]["phi"] == pytest.approx(4.5, abs=0.5)
    je = z["limit"]["je_art"]
    obs = {a: je[a]["beobachtet"] for a in C.ART_LABELS}
    assert obs["Import"] == pytest.approx(0.734, abs=0.01) and obs["Export"] == pytest.approx(0.733, abs=0.01) and obs["Umschlag"] == pytest.approx(0.963, abs=0.01) and obs["Leer"] == pytest.approx(0.477, abs=0.01)
    assert je["Import"]["poisson"] == pytest.approx(0.369, abs=0.02) and je["Leer"]["poisson"] < 0.01
    assert je["Import"]["nb2"] == pytest.approx(0.664, abs=0.02) and je["Export"]["nb2"] == pytest.approx(0.812, abs=0.02)
    worst = max(abs(je[a]["nb2"] - obs[a]) for a in C.ART_LABELS)
    assert 0.06 < worst < 0.10                                               # NB2 liegt je Art um bis zu 8 Prozentpunkte daneben


# --- B17: die Datei passt zum Code -----------------------------------------------------------------------------------------------------------------------

def test_b17_a_small_recomputation_matches_the_frozen_cell_within_monte_carlo_error():
    """Neu gerechnet mit anderem Zufalls-Seed und weniger Wiederholungen (n = 100, α = 0,5, 800 Wdh.): jede Überdeckung liegt innerhalb von 4 Standardfehlern der Differenz von der vorgerechneten Zelle."""
    ref = cell("NB2", 0.5, 100)
    new = SIM.run_cell("NB2", 0.5, 100, 800, 424242)
    for m in ("Poisson", "Quasi z", "HC0", "HC3a", "NB2"):
        se = math.sqrt(ref["se_mc"] ** 2 + new["se_mc"] ** 2)
        assert abs(new["coverage"][m] - ref["coverage"][m]) <= 4 * se + 0.005, m


def test_b17_the_frozen_probe_is_reproduced():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("zae_make_frozen", pathlib.Path(__file__).resolve().parent.parent / "tools" / "zae_make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == 0
