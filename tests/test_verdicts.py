"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds."""

import copy

import pytest

import zae_constants as C
import zae_evaluation as ev

M = ["Poisson", "Quasi z", "Quasi t", "HC0", "HC1", "HC3a", "NB2"]


def _cell(cov, R=1000, se_mc=0.007):
    return {"coverage": dict(zip(M, cov)), "R": R, "se_mc": se_mc, "nb2_ausfaelle": 0, "nb2_randloesung": 0}


# --- Format ----------------------------------------------------------------------------------------------------------------------------------------------

def test_number_format_uses_decimal_comma_and_true_minus():
    assert ev.de(2.5863, 2) == "2,59"
    assert ev.de(-1.234, 2) == "−1,23"
    assert ev.de(float("nan")) == "nicht definiert" and ev.de(float("inf")) == "nicht definiert"
    assert ev.pct(0.744) == "74,4 %" and ev.pct(0.9567, 0) == "96 %"
    assert ev.frac(ev.Fraction(30, 11)) == "30/11" and ev.frac(ev.Fraction(20, 1)) == "20"


# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------------

def test_hand_verdict_names_every_number_of_the_example():
    kind, text = ev.verdict_hand(ev.analyse_hand((0, 0, 1, 5, 1, 3, 6, 10)))
    assert kind == "warning"
    assert "Gleicher Punktschätzer (Faktor 3,33" in text
    assert "Poisson: z = 2,59, Intervall [0,29; 2,12] schließt 0 aus" in text
    assert "Quasi-Poisson (t, 6 Freiheitsgrade): z = 1,40, Intervall [−0,90; 3,31] enthält 0 (φ̂ = 3,42)" in text
    assert "Sandwich HC0: z = 1,57" in text and "NB2: z = 1,67" in text
    assert "Gilt für diese acht Werte, nichts über Häufigkeiten." in text


def test_hand_verdict_is_info_when_all_methods_agree():
    """Große Unterschiede zwischen den Gruppen: alle Intervalle schließen 0 aus, kein Widerspruch zu melden."""
    kind, text = ev.verdict_hand(ev.analyse_hand((1, 2, 1, 2, 30, 40, 35, 45)))
    assert kind == "info" and "verschiedene Schlüsse" not in text and text.count("schließt 0 aus") == 4


def test_hand_verdict_for_a_group_without_event_gives_no_number():
    kind, text = ev.verdict_hand_empty("Eine Gruppe hat kein einziges Ereignis.")
    assert kind == "warning" and "keinen endlichen Schätzer" in text
    with pytest.raises(ev.H.NoEventsError):
        ev.analyse_hand((0, 0, 0, 0, 1, 2, 3, 4))


def test_hand_fractions_rows():
    rows = dict(ev.hand_fractions(ev.analyse_hand((0, 0, 1, 5, 1, 3, 6, 10))))
    assert rows["naiv, 1/T_A + 1/T_B"] == "13/60" and rows["Quasi-Poisson, φ̂ · naiv"] == "1001/1350" and rows["Sandwich HC0, Σ RSS_g / T_g²"] == "1057/1800"
    assert rows["Sandwich HC1, n/(n − p) · HC0"] == "1057/1350" and rows["Pearson-φ̂"] == "154/45" and rows["Pearson-χ²"] == "308/15"


def test_exposure_verdict():
    kind, text = ev.verdict_exposure(ev.H.exposure_example(21, 3, 20, 5), (21, 3, 20, 5))
    assert kind == "info" and "Kunde 1 7,00" in text and "Kunde 2 4,00" in text and "4/7 = 0,571" in text and "(1,9459; −0,5596)" in text


# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------------

def test_cover_classes_have_inclusive_lower_bounds():
    cl = ev.cover_classes(dict(zip(M, (0.93, 0.9299, 0.90, 0.8999, 0.95, 0.5, 0.929))))
    assert cl["ab"] == ["Poisson", "HC1"]
    assert cl["mitte"] == ["Quasi z", "Quasi t", "NB2"]
    assert cl["unter"] == ["HC0", "HC3a"]


def test_cover_verdict_without_overdispersion_reports_no_effect_at_the_inclusive_threshold():
    limit = {"cov_poisson": 0.95, "cov_quasi": 0.95}
    kind, text = ev.verdict_cover(_cell([0.93, 0.94, 0.94, 0.935, 0.935, 0.94, 0.95]), limit, "NB2", 0.0, 200)
    assert kind == "success" and "keinen Befund" in text and "ab 93 %: Poisson" in text
    kind, text = ev.verdict_cover(_cell([0.929, 0.94, 0.94, 0.935, 0.935, 0.94, 0.95]), limit, "NB2", 0.0, 200)
    assert kind == "info" and "keinen Befund" not in text


def test_cover_verdict_without_overdispersion_names_an_exception_in_the_same_sentence():
    limit = {"cov_poisson": 0.95, "cov_quasi": 0.95}
    kind, text = ev.verdict_cover(_cell([0.95, 0.94, 0.94, 0.91, 0.92, 0.94, 0.95]), limit, "NB2", 0.0, 30)
    assert kind == "success" and "Ausnahme in derselben Welt: Sandwich HC0, Sandwich HC1 unter 93 %" in text


def test_cover_verdict_with_overdispersion_names_numbers_limit_and_interpretation():
    limit = {"cov_poisson": 0.7419, "cov_quasi": 0.9246}
    cov = [0.741, 0.922, 0.924, 0.930, 0.931, 0.937, 0.945]
    kind, text = ev.verdict_cover(_cell(cov, R=20000, se_mc=0.002), limit, "NB2", 0.5, 200)
    assert kind == "warning"
    for part in ("NB2-Welt, α = 0,50, n = 200, 20000 Wiederholungen", "Monte-Carlo-Standardfehler 0,2 Prozentpunkte", "Poisson 74,1 %", "Quasi-Poisson (z) 92,2 %", "Sandwich HC0 93,0 %", "HC3-Analogon 93,7 %", "NB2 94,5 %",
                 "Poisson 74,2 %, Quasi-Poisson 92,5 %", "Deutung aus der Formel, nicht eigens gemessen", "unter 90 %: Poisson (naiv)"):
        assert part in text, part
    assert "Ausnahme im selben Satz" not in text                          # nur bei n = 30


def test_cover_verdict_nb1_world_names_the_exact_limit():
    limit = {"cov_poisson": 0.787, "cov_quasi": 0.95}
    kind, text = ev.verdict_cover(_cell([0.787, 0.948, 0.95, 0.939, 0.939, 0.946, 0.947]), limit, "NB1", 0.5, 200)
    assert "NB1-Welt" in text and "Quasi-Poisson hält im Grenzwert 95,0 %" in text and kind == "warning"


def test_cover_verdict_small_sample_exception_only_when_hc0_is_below_quasi():
    limit = {"cov_poisson": 0.74, "cov_quasi": 0.92}
    kind, text = ev.verdict_cover(_cell([0.78, 0.92, 0.93, 0.88, 0.89, 0.928, 0.929]), limit, "NB2", 0.5, 30)
    assert "bei n = 30 liegt das Sandwich HC0 mit 88,0 % unter Quasi-Poisson, die Hebelkorrektur (HC3-Analogon) hebt es auf 92,8 %" in text
    kind, text = ev.verdict_cover(_cell([0.78, 0.92, 0.93, 0.93, 0.93, 0.93, 0.929]), limit, "NB2", 0.5, 30)
    assert "Ausnahme im selben Satz" not in text


def test_cover_verdict_poisson_threshold_for_the_warning_is_90_percent():
    limit = {"cov_poisson": 0.9, "cov_quasi": 0.95}
    assert ev.verdict_cover(_cell([0.90, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]), limit, "NB2", 0.25, 200)[0] == "info"
    assert ev.verdict_cover(_cell([0.8999, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]), limit, "NB2", 0.25, 200)[0] == "warning"


def test_live_verdict_names_both_series():
    kind, text = ev.verdict_live(_cell([0.75, 0.92, 0.92, 0.93, 0.93, 0.94, 0.95], R=200, se_mc=0.0154), _cell([0.741, 0.922, 0.92, 0.93, 0.93, 0.94, 0.945], R=20000, se_mc=0.0015))
    assert kind == "info" and "Live-Probe mit 200 Wiederholungen" in text and "1,5 Prozentpunkte" in text and "Vorgerechnet (20000 Wiederholungen)" in text and "Poisson 75,0 %" in text and "Poisson 74,1 %" in text


def test_single_draw_verdict_lists_intervals_that_miss_the_truth():
    tab = {m: {"lo": 0.3, "hi": 0.5, "est": 0.4} for m in ev.M.METHODS}
    assert "alle sieben Intervalle enthalten den wahren Anstieg 0,4" in ev.verdict_single(tab)[1]
    tab["Poisson"] = {"lo": 0.41, "hi": 0.5, "est": 0.455}
    tab["HC0"] = {"lo": 0.1, "hi": 0.39, "est": 0.3}
    text = ev.verdict_single(tab)[1]
    assert "nicht: Poisson (naiv), Sandwich HC0" in text and "0,455" in text and "Ein Datensatz sagt nichts über Häufigkeiten." in text


# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------------

def _frozen_vehicle():
    def sp(cov, alarm, ratio):
        return {m: {"coverage": cov[i], "fehlalarm": alarm[i], "sd": 0.05, "se_mean": 0.03, "ratio": ratio[i], "n_valid": 300} for i, m in enumerate(M)}
    cov = (0.697, 0.913, 0.913, 0.957, 0.957, 0.957, 0.88)
    alarm = (0.323, 0.117, 0.117, 0.043, 0.043, 0.043, 0.097)
    ratio = (0.52, 0.91, 0.91, 1.04, 1.04, 1.04, 0.80)
    cell = {"n": 3000, "R": 300, "kunden": False, "se_mc": 0.0126, "phi_mean": 3.04, "spalten": {"Zoll": sp(cov, alarm, ratio), "Wochenende": sp(cov, alarm, ratio), "Gewicht (Rauschmerkmal)": sp(cov, alarm, ratio)}}
    cell_k = copy.deepcopy(cell)
    cell_k["kunden"] = True
    cell_k["phi_mean"] = 2.5
    lim = {"n": 1_000_000, "mean": 6.157, "var": 43.41, "min": 1.0, "phi": 3.227, "phi_je_art": [3.02, 0.44, 0.44, 19.16]}
    return {"vehicle": {"cells": [cell, cell_k], "limit": {"ohne_kunden": lim, "mit_kunden": dict(lim, phi=2.536)}}}


def test_vehicle_verdict_for_a_real_column_names_coverage_ratio_and_target():
    kind, text = ev.verdict_vehicle(_frozen_vehicle(), 3000, False, "Zoll")
    assert kind == "warning"
    for part in ("n = 3000 Container, 300 Wiederholungen (vorgerechnet), ohne Kunden im Modell, Pearson-φ̂ im Mittel 3,04", "Zoll: Poisson 69,7 %, Quasi-Poisson 91,3 %, NB2 88,0 %, Sandwich HC0 95,7 % Überdeckung",
                 "Monte-Carlo-Standardfehler 1,3 Prozentpunkte", "Poisson 0,52, Quasi 0,91, HC0 1,04, NB2 0,80", "Grenzwert des jeweiligen Schätzers (n = 1 Million)", "nicht exakt log-linear"):
        assert part in text, part


def test_vehicle_verdict_for_the_noise_feature_reports_false_alarms():
    kind, text = ev.verdict_vehicle(_frozen_vehicle(), 3000, False, "Gewicht (Rauschmerkmal)")
    assert kind == "warning"
    assert "schlägt bei Poisson in 32,3 %, Quasi-Poisson 11,7 %, NB2 9,7 % und Sandwich HC0 4,3 %" in text and "nominal 5 %" in text


def test_vehicle_verdict_threshold_of_the_warning_for_the_noise_feature_is_10_percent():
    f = _frozen_vehicle()
    f["vehicle"]["cells"][0]["spalten"]["Gewicht (Rauschmerkmal)"]["Poisson"]["fehlalarm"] = 0.10
    assert ev.verdict_vehicle(f, 3000, False, "Gewicht (Rauschmerkmal)")[0] == "info"
    f["vehicle"]["cells"][0]["spalten"]["Gewicht (Rauschmerkmal)"]["Poisson"]["fehlalarm"] = 0.1001
    assert ev.verdict_vehicle(f, 3000, False, "Gewicht (Rauschmerkmal)")[0] == "warning"


def test_vehicle_verdict_with_customers_says_so():
    text = ev.verdict_vehicle(_frozen_vehicle(), 3000, True, "Zoll")[1]
    assert "mit 11 Kundenspalten im Modell, Pearson-φ̂ im Mittel 2,50" in text


def test_dispersion_verdict_counts_underdispersed_kinds():
    kind, text = ev.verdict_dispersion(_frozen_vehicle())
    assert kind == "info"
    for part in ("Mittel 6,16", "Varianz 43,41", "Varianz geteilt durch Mittel 7,05", "Pearson-φ̂ 3,23", "Import 3,02, Export 0,44, Umschlag 0,44, Leer 19,16", "in 2 von 4 Arten (Export, Umschlag)", "auf 2,54"):
        assert part in text, part


# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------------

def test_weights_verdict():
    assert "Faktor 20" in ev.verdict_weights(0.0)[1]
    t = ev.verdict_weights(0.5)[1]
    assert "das 20-Fache bei Poisson und Quasi-Poisson" in t and "2,727-Fache bei NB2 mit α = 0,50" in t and "30/11" in t
    assert "40/21" in ev.verdict_weights(1.0)[1] and "1,905" in ev.verdict_weights(1.0)[1]


def test_two_worlds_verdict_threshold_is_two_standard_errors_strict():
    se = 0.01
    base = [0.9] * 7
    q = 0.92
    c2 = _cell(base[:1] + [q] + base[2:], se_mc=se)
    c2["coverage"]["NB2"] = q + 2 * se                                     # genau zwei Standardfehler: noch kein Abstand
    c1 = _cell(base, se_mc=se)
    assert "kein Abstand über zwei Standardfehlern" in ev.verdict_two_worlds(c2, c1, 0.5, 200)[1]
    c2["coverage"]["NB2"] = q + 2.01 * se
    assert "NB2 liegt vorn" in ev.verdict_two_worlds(c2, c1, 0.5, 200)[1]
    c1["coverage"]["Quasi z"] = c1["coverage"]["NB2"] + 2.01 * se
    t = ev.verdict_two_worlds(c2, c1, 0.5, 200)[1]
    assert "Quasi-Poisson liegt vorn" in t and "Kein Verfahren ist in beiden Welten das beste." in t


def test_two_worlds_verdict_without_overdispersion():
    assert "dieselbe Poisson-Welt" in ev.verdict_two_worlds(None, None, 0.0, 200)[1]
    assert ev.collect_cells({"grid": []}, 0.0, 200) == (None, None)


def _frozen_miss():
    return {"missspec": {"limit": [{"c": c, "n": 200000, "poisson": p, "nb2": n} for c, p, n in ((0.0, 0.4004, 0.4009), (0.2, 0.6785, 0.4664), (0.3, 0.9703, 0.4614))],
                         "n200": [{"c": c, "n": 200, "R": 3000, "diff_median": m, "diff_q1": a, "diff_q3": b, "poisson_robust_sd": s, "diff_gross": g}
                                  for c, m, a, b, s, g in ((0.0, 0.0022, -0.0129, 0.0164, 0.0703, 0), (0.2, -0.1415, -0.2026, -0.0958, 0.1515, 8), (0.3, -0.2516, -0.3612, -0.1694, 0.2548, 268))]}}


def test_missspec_verdict_without_curvature_is_a_success():
    kind, text = ev.verdict_missspec(_frozen_miss(), 0.0)
    assert kind == "success" and "Poisson 0,400 und NB2 0,401" in text and "Unterschied im Median 0,002" in text


def test_missspec_verdict_with_curvature_names_median_quartiles_and_outliers():
    kind, text = ev.verdict_missspec(_frozen_miss(), 0.2)
    assert kind == "warning"
    for part in ("c = 0,2", "n = 200.000", "Poisson 0,678 gegen NB2 0,466", "im Median bei −0,14 (Quartile −0,20 und −0,10)", "robusten Streuung von β̂₁(Poisson) von 0,15", "in 8 von 3000 Datensätzen",
                 "Quasi-Poisson hat dieselben Koeffizienten wie Poisson, NB2 nicht"):
        assert part in text, part
    assert "Ab c = 0,25" not in text
    assert "Ab c = 0,25 hat μ² keinen endlichen Erwartungswert" in ev.verdict_missspec(_frozen_miss(), 0.3)[1]


# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------------

def _frozen_zeros():
    je = {"Import": (0.736, 0.373, 0.666), "Export": (0.732, 0.656, 0.812), "Umschlag": (0.963, 0.932, 0.950), "Leer": (0.476, 0.002, 0.440)}
    return {"vehicle": {"limit": {"ohne_kunden": {"min": 1.0, "n": 1_000_000}}},
            "nullen": {"limit": {"n": 1_000_000, "je_art": {a: {"beobachtet": o, "poisson": p, "nb2": n} for a, (o, p, n) in je.items()}},
                       "ohne_kunden": {"R": 300, "beobachtet": 0.765, "poisson_erwartet": 0.561, "nb2_erwartet": 0.757, "phi": 6.42},
                       "mit_kunden": {"R": 300, "beobachtet": 0.765, "poisson_erwartet": 0.6, "nb2_erwartet": 0.755, "phi": 4.5}}}


def test_zero_verdict_for_standtage_has_no_zero_claim_beyond_the_minimum():
    kind, text = ev.verdict_zeros(_frozen_zeros(), "Standtage")
    assert kind == "info" and "keine Nullen (kleinster Wert 1" in text and "Standgeld-Tage" in text


def test_zero_verdict_for_standgeld_tage_names_all_zero_shares_and_the_open_question():
    kind, text = ev.verdict_zeros(_frozen_zeros(), "Standgeld-Tage")
    assert kind == "warning"
    for part in ("Nullanteil beobachtet 0,765", "Poisson erwartet 0,561", "NB2 erwartet 0,757", "Pearson-φ̂ 6,42 ohne und 4,50 mit 11 Kundenspalten", "Import 0,736 / 0,373 / 0,666", "Leer 0,476 / 0,002 / 0,440",
                 "um bis zu 8 Prozentpunkte daneben", "ist nicht gemessen"):
        assert part in text, part


# --- Presets: je Preset die erwartete Verdikt-Zeile (§11a Punkt 7) --------------------------------------------------------------------------------------

def test_every_preset_yields_its_expected_verdict_kind_in_experiment_2():
    """Jedes Preset durchläuft die Verdikt-Zeile von Experiment 2: α = 0 gibt „keinen Befund“ (success), jedes Preset mit α > 0 eine Warnung ohne diese Wendung."""
    f = ev.load_frozen()
    for name, p in C.PRESETS.items():
        cell = ev.grid_cell(f, p["welt"], p["alpha"], p["n"])
        kind, text = ev.verdict_cover(cell, ev.limit_cached(p["welt"], p["alpha"]), p["welt"], p["alpha"], p["n"])
        if p["alpha"] == 0:
            assert kind == "success" and "keinen Befund" in text, name
        else:
            assert kind == "warning" and "keinen Befund" not in text, name


def test_the_preset_without_overdispersion_reports_no_effect_anywhere():
    """Preset „Keine Überstreuung“: Experiment 2 meldet keinen Befund (success). Die Experimente 3 bis 5 sind davon nicht abhängig; das Preset lässt c = 0, dort meldet Experiment 4 (c) „Richtiges Mittelwertmodell“ als success."""
    f = ev.load_frozen()
    p = C.PRESETS["Keine Überstreuung (α = 0)"]
    assert p["c"] == 0.0 and ev.verdict_missspec(f, p["c"])[0] == "success"
    assert ev.verdict_cover(ev.grid_cell(f, p["welt"], p["alpha"], p["n"]), ev.limit_cached(p["welt"], p["alpha"]), p["welt"], p["alpha"], p["n"])[0] == "success"
    assert ev.verdict_two_worlds(*ev.collect_cells(f, p["alpha"], p["n"]), p["alpha"], p["n"])[1].startswith("Bei α = 0")


def test_the_preset_with_a_wrong_mean_model_reports_the_effect():
    f = ev.load_frozen()
    p = C.PRESETS["Falsches Mittelwertmodell"]
    kind, text = ev.verdict_missspec(f, p["c"])
    assert kind == "warning" and "Falsches Mittelwertmodell (c = 0,2)" in text
