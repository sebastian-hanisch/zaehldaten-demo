"""Konstanten: Regler-Grenzen, feste Annahmen der Messwelten, Container-Vehikel, Farben, Presets."""

# --- Messwelt 1: ein Merkmal x ~ N(0, 1), log μ = β0 + β1 x (+ c x²), Zählwert NB2 oder NB1 ---------------------------------------------------------------
BETA0, BETA1 = 1.0, 0.4          # wahrer Achsenabschnitt und Anstieg: der Anstieg 0,4 ist das Ziel der Intervalle
ALPHA_CHOICES = (0.0, 0.25, 0.5, 1.0)
N_CHOICES = (30, 100, 200)
C_CHOICES = (0.0, 0.1, 0.2, 0.3)
WELT_CHOICES = ("NB2", "NB1")
WELT_LABELS = {
    "NB2": "NB2: Varianz μ + α μ² (quadratisch im Mittel)",
    "NB1": "NB1: Varianz (1 + δ) μ (linear im Mittel, gleiche mittlere Dispersion)",
}
LIVE_R_MIN, LIVE_R_MAX, LIVE_R_DEFAULT = 50, 500, 200       # Wiederholungen der Live-Probe (Rechenzeit unter 2 s bei n = 200)

# --- Messwelt 2: das Container-Vehikel Standtage -------------------------------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)             # Anteile der Containerarten
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art
P_ZOLL = 0.25                                # Zollhalt bei Import
K_KUNDEN = 12
# Zwölf Kundeneffekte (log, Streuung 0,3) als feste Konstanten (17 Stellen): Normalzahlen aus einem numpy-Strom wären über Versionen nicht zugesagt.
UK = (-0.42714751093638936, 0.37911853743873308, -0.2611985213877257, -0.077751970480319274, -0.022602992103156291, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.58585891890365693, 0.70422289631365553, 0.29054907172557709)
SPALTEN = ("Achsenabschnitt", "Export", "Umschlag", "Leer", "Zoll", "Wochenende", "log(1+Vorlauf)", "Gewicht (Rauschmerkmal)")
COL_ZOLL, COL_WE, COL_GEWICHT = 4, 5, 7
VEHICLE_COLUMNS = {"Zoll": COL_ZOLL, "Wochenende": COL_WE, "Gewicht (Rauschmerkmal)": COL_GEWICHT}
VN_CHOICES = (300, 3000)
STANDGELD_FREI_TAGE = 7                      # Annahme der Messung: Standgeld fällig ab dem 8. Tag, Standgeld-Tage = max(0, Standtage − 7)
AUSGANG_CHOICES = ("Standtage", "Standgeld-Tage")

# --- vorgerechnete Datei und Zufalls-Seeds --------------------------------------------------------------------------------------------------------------
FROZEN_FILE = "zae_frozen.json"
SEED_MAX, DEFAULT_SEED = 999999, 2026
SEED_GRID, SEED_MAIN_NB2, SEED_MAIN_NB1 = 8100, 8200, 8300
SEED_VEHICLE, SEED_LIMIT, SEED_LIMIT2, SEED_NULL, SEED_MISSPEC, SEED_MISSPEC_LIMIT = 8400, 8500, 8550, 8600, 8700, 8800

# --- Handbeispiel: acht Standtage in zwei Gruppen (ohne / mit Zollhalt) -------------------------------------------------------------------------------------
HAND_DEFAULT = (0, 0, 1, 5, 1, 3, 6, 10)
HAND_MAX = 99
EXPO_DEFAULT = (21, 3, 20, 5)                # Kunde 1: 21 Standtage bei 3 Containern, Kunde 2: 20 bei 5

COLORS = {
    "Poisson": "#c0392b", "Quasi z": "#e08e0b", "Quasi t": "#c9a227", "HC0": "#2e7d4f", "HC1": "#4f9d6f", "HC3a": "#1f6f6b", "NB2": "#4c78a8",
}
COLOR_TARGET = "#555555"
COLOR_ORACLE = "#7a3fb0"

# --- Presets ----------------------------------------------------------------------------------------------------------------------------------------
_BASIS = {
    "h": HAND_DEFAULT, "e": EXPO_DEFAULT, "welt": "NB2", "alpha": 0.5, "n": 200, "c": 0.0, "vn": 3000, "spalte": "Zoll", "kunden": False,
    "ausgang": "Standtage", "live_r": LIVE_R_DEFAULT, "seed": DEFAULT_SEED,
}


def _preset(**kw):
    p = dict(_BASIS)
    p.update(kw)
    return p


PRESETS = {
    "Von Hand (8 Standtage)": _preset(),
    "Keine Überstreuung (α = 0)": _preset(alpha=0.0),
    "Starke Überstreuung (α = 1)": _preset(alpha=1.0),
    "Standtage der Container": _preset(spalte="Gewicht (Rauschmerkmal)"),
    "Kleine Stichprobe (n = 30)": _preset(n=30),
    "Falsches Mittelwertmodell": _preset(c=0.2),
}
PRESET_HELP = {
    "Von Hand (8 Standtage)": "Voreinstellung: acht erfundene Standtage in zwei Gruppen (ohne und mit Zollhalt), von Hand in Brüchen nachrechenbar. Die übrigen Experimente stehen auf α = 0,5 und n = 200.",
    "Keine Überstreuung (α = 0)": "Die Zählwerte sind genau Poisson-verteilt: Varianz gleich Mittel. Alle Verfahren sollten die Intervalle halten; die Demo meldet hier keinen Effekt.",
    "Starke Überstreuung (α = 1)": "Die Varianz wächst quadratisch mit dem Mittel (μ + μ²). Das Poisson-Intervall unterdeckt am stärksten.",
    "Standtage der Container": "Das Container-Vehikel mit n = 3000 Containern, Auswahl: das Rauschmerkmal Gewicht (wahre Wirkung null). Gemessen wird, wie oft ein Merkmal ohne Wirkung anschlägt.",
    "Kleine Stichprobe (n = 30)": "Dieselbe Überstreuung (α = 0,5) bei nur 30 Beobachtungen: das Sandwich HC0 unterdeckt hier, die Hebelkorrektur hebt es.",
    "Falsches Mittelwertmodell": "Die Wahrheit hat einen Krümmungsterm (c = 0,2), das Modell nur den Anstieg: Poisson und NB2 schätzen dann verschiedene Größen.",
}
