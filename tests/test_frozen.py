"""Die vorgerechnete Datei `data/zae_frozen.json`: Aufbau, Vollständigkeit, innere Konsistenz, Zusammenhang mit dem Grenzwert aus der Formel. Die Neu-Rechnung einer kleinen Zelle und
`tools/zae_make_frozen.py --check` stehen in tests/test_claims.py (B17)."""

import json
import math

import pytest

import zae_asymptotics as AS
import zae_constants as C
import zae_evaluation as ev

F = ev.load_frozen()


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, float):
        yield obj


def test_all_parts_present_and_json_has_no_nan_or_infinity():
    assert set(F) == {"grid", "vehicle", "missspec", "nullen", "meta"}
    assert all(math.isfinite(x) for x in _numbers(F))
    text = (ev.FROZEN_PATH).read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_the_grid_has_every_documented_cell_exactly_once():
    keys = [(c["welt"], c["alpha"], c["n"]) for c in F["grid"]]
    want = [(w, a, n) for w in C.WELT_CHOICES for a in C.ALPHA_CHOICES for n in C.N_CHOICES if not (w == "NB1" and a == 0.0)]
    assert sorted(keys) == sorted(want) and len(keys) == 21
    for c in F["grid"]:
        assert c["R"] >= 4000 and set(c["coverage"]) == set(ev.M.METHODS)
        assert c["se_mc"] == pytest.approx(math.sqrt(0.95 * 0.05 / c["R"]))
        assert all(0.0 <= v <= 1.0 for v in c["coverage"].values())
        assert all(c["n_valid"][m] <= c["R"] for m in ev.M.METHODS)


def test_the_grid_lookup_maps_nb1_with_alpha_zero_to_the_poisson_world():
    assert ev.grid_cell(F, "NB1", 0.0, 200) is ev.grid_cell(F, "NB2", 0.0, 200)
    with pytest.raises(KeyError):
        ev.grid_cell(F, "NB2", 0.5, 50)


def test_every_app_setting_combination_has_a_vehicle_cell():
    for vn in C.VN_CHOICES:
        for kunden in (False, True):
            c = ev.veh_cell(F, vn, kunden)
            assert set(c["spalten"]) == set(C.VEHICLE_COLUMNS) and c["R"] >= 2000
    with pytest.raises(KeyError):
        ev.veh_cell(F, 1000, False)


def test_missspec_has_every_curvature_of_the_slider():
    assert [e["c"] for e in F["missspec"]["limit"]] == list(C.C_CHOICES) == [e["c"] for e in F["missspec"]["n200"]]


def test_the_frozen_grid_agrees_with_the_formula_for_poisson_and_quasi_in_the_nb2_world():
    """Die Überdeckung in der Simulation (n = 200) liegt beim Grenzwert aus der Formel (Band: 4 Standardfehler plus 1,5 Prozentpunkte für das endliche n), für Poisson und für Quasi-Poisson."""
    for a in (0.25, 0.5, 1.0):
        c = ev.grid_cell(F, "NB2", a, 200)
        o = ev.limit_cached("NB2", a)
        assert abs(c["coverage"]["Poisson"] - o["cov_poisson"]) <= 4 * c["se_mc"] + 0.015
        assert abs(c["coverage"]["Quasi z"] - o["cov_quasi"]) <= 4 * c["se_mc"] + 0.015


def test_the_oracle_limit_for_the_nb1_world_uses_the_matching_mean_dispersion():
    for a in (0.25, 0.5, 1.0):
        o = ev.limit_cached("NB1", a)
        assert o["phi"] == pytest.approx(AS.limit_coverage("NB2", a)["phi"], rel=1e-9)        # gleiche mittlere Dispersion E[V/μ]
        assert o["cov_quasi"] == pytest.approx(0.95, abs=1e-9)


def test_meta_names_versions_and_the_script():
    assert "zae_make_frozen" in F["meta"]["hinweis"] and F["meta"]["numpy"] and F["meta"]["scipy"]
