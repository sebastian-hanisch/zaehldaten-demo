"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS, auch die im Hauptbereich (Experimente): das aktive Preset ist hervorgehoben, solange JEDER Regler dem Preset entspricht (zustandslos).
Die Presets setzen die Regler aller Experimente zugleich; die Achtergruppe `h` (Handbeispiel) und die Exposition `e` werden auf je einzelne Regler (h1 … h8, e1 … e4) verteilt."""

import random
from dataclasses import dataclass

import streamlit as st

import zae_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


def _specs() -> dict:
    s = {}
    for i in range(8):
        s[f"h{i + 1}"] = SettingSpec(f"h{i + 1}", int, C.HAND_DEFAULT[i], 0, C.HAND_MAX)
    for i in range(4):
        s[f"e{i + 1}"] = SettingSpec(f"e{i + 1}", int, C.EXPO_DEFAULT[i], 0, 9999)
    s["welt"] = SettingSpec("welt", str, "NB2", choices=C.WELT_CHOICES)
    s["alpha"] = SettingSpec("alpha", float, 0.5, choices=C.ALPHA_CHOICES)
    s["n"] = SettingSpec("n", int, 200, choices=C.N_CHOICES)
    s["live_r"] = SettingSpec("live_r", int, C.LIVE_R_DEFAULT, C.LIVE_R_MIN, C.LIVE_R_MAX)
    s["vn"] = SettingSpec("vn", int, 3000, choices=C.VN_CHOICES)
    s["spalte"] = SettingSpec("spalte", str, "Zoll", choices=tuple(C.VEHICLE_COLUMNS))
    s["kunden"] = SettingSpec("kunden", bool, False)
    s["c"] = SettingSpec("c", float, 0.0, choices=C.C_CHOICES)
    s["ausgang"] = SettingSpec("ausgang", str, "Standtage", choices=C.AUSGANG_CHOICES)
    s["seed_input"] = SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX)
    return s


SETTING_SPECS = _specs()


def flat_preset(name: str) -> dict:
    """Preset als {Regler-Schlüssel: Wert}: `h` und `e` auf einzelne Regler verteilt, `seed` als `seed_input`."""
    p = dict(C.PRESETS[name])
    out = {}
    for i, v in enumerate(p.pop("h")):
        out[f"h{i + 1}"] = v
    for i, v in enumerate(p.pop("e")):
        out[f"e{i + 1}"] = v
    out["seed_input"] = p.pop("seed")
    out.update(p)
    return out


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: auf Typ, Grenzen oder erlaubte Werte gebracht; None, wenn nichts Brauchbares darin steht. Bei einer Auswahl rastet der Wert ein."""
    if spec.caster is bool:
        return str(raw).lower() in ("1", "true", "ja")
    value = spec.caster(raw)
    if spec.choices is not None:
        if spec.caster is str:
            return value if value in spec.choices else None
        return min(spec.choices, key=lambda c: abs(c - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf die erlaubten Werte eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _clean(spec, qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for state_key, value in flat_preset(name).items():
        st.session_state[state_key] = value


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos; jeder Regler aus SETTING_SPECS wird verglichen)."""
    for name in C.PRESETS:
        flat = flat_preset(name)
        if all(st.session_state.get(state_key, spec.default) == flat.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt die Regler aller Experimente):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
