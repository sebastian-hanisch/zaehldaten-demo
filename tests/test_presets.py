"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten), keine zwei gleichen Presets."""

import pytest

import zae_constants as C
import zae_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name in C.PRESETS:
        assert C.PRESET_HELP[name]
        flat = P.flat_preset(name)
        assert set(flat) == set(P.SETTING_SPECS), name


def test_preset_values_are_valid_for_their_controls():
    for name in C.PRESETS:
        for key, value in P.flat_preset(name).items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    flat = P.flat_preset(first)
    assert all(flat[k] == s.default for k, s in P.SETTING_SPECS.items())


def test_no_two_presets_are_equal_so_the_highlight_is_unique():
    flats = [tuple(sorted(P.flat_preset(n).items())) for n in C.PRESETS]
    assert len(set(flats)) == len(flats)


def test_each_non_default_preset_changes_exactly_the_documented_controls():
    base = P.flat_preset(next(iter(C.PRESETS)))
    diff = {n: {k for k, v in P.flat_preset(n).items() if v != base[k]} for n in C.PRESETS}
    assert diff["Keine Überstreuung (α = 0)"] == {"alpha"} and diff["Starke Überstreuung (α = 1)"] == {"alpha"}
    assert diff["Standtage der Container"] == {"spalte"} and diff["Kleine Stichprobe (n = 30)"] == {"n"} and diff["Falsches Mittelwertmodell"] == {"c"}


def test_alpha_and_n_values_of_the_presets_match_their_names():
    assert C.PRESETS["Keine Überstreuung (α = 0)"]["alpha"] == 0.0 and C.PRESETS["Starke Überstreuung (α = 1)"]["alpha"] == 1.0
    assert C.PRESETS["Kleine Stichprobe (n = 30)"]["n"] == 30 and C.PRESETS["Falsches Mittelwertmodell"]["c"] == 0.2
    assert C.PRESETS["Standtage der Container"]["spalte"].startswith("Gewicht") and C.PRESETS["Von Hand (8 Standtage)"]["h"] == (0, 0, 1, 5, 1, 3, 6, 10)


def test_setting_specs_cover_every_control_key_of_the_app():
    keys = set(P.SETTING_SPECS)
    assert {f"h{i}" for i in range(1, 9)} <= keys and {f"e{i}" for i in range(1, 5)} <= keys
    assert {"welt", "alpha", "n", "c", "vn", "spalte", "kunden", "ausgang", "live_r", "seed_input"} <= keys


@pytest.mark.parametrize("key,raw,want", [("alpha", "0.4", 0.5), ("alpha", "0.1", 0.0), ("alpha", "7", 1.0), ("n", "150", 100), ("n", "151", 200), ("vn", "1000", 300), ("vn", "2000", 3000),
                                          ("c", "0.17", 0.2), ("h1", "500", 99), ("h1", "-4", 0), ("live_r", "5", 50), ("live_r", "9999", 500), ("seed_input", "-1", 0),
                                          ("welt", "NB1", "NB1"), ("welt", "xyz", None), ("spalte", "Zoll", "Zoll"), ("spalte", "gibt es nicht", None), ("kunden", "true", True), ("kunden", "0", False)])
def test_permalink_values_are_limited_or_snapped(key, raw, want):
    assert P._clean(P.SETTING_SPECS[key], raw) == want


def test_permalink_rejects_non_numbers():
    with pytest.raises(ValueError):
        P._clean(P.SETTING_SPECS["alpha"], "abc")
