"""Container-Vehikel: Merkmale und Verweilzeit eines Containers, daraus die Standtage (begonnene Tage = aufgerundete Verweilzeit).

Nachbau (nicht bitgleich) des Merkmalsmodells der Nachbar-Demo `verweilzeit-demo` (dort `vwz_generator.py`, `draw_features` und `log_median_and_s`; nur gelesen, keine
Ereignissimulation, keine Stapelplanung): Art (Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %), Kunde (zwölf Effekte als feste Konstanten), Zollhalt bei Import, Wochenende,
Vorlauf bei Export und Umschlag, Rauschmerkmal Gewicht (Wirkung null). Die log-Verweilzeit ist normal mit je Art anderer Streuung. Alle Zufallszahlen aus SplitMix64
(`zae_rng`), darum gelten die Zahlen dieser Demo und nicht die der Nachbar-Demo (anderer Zufallsstrom). Alle Parameter sind erfunden, kein Echtdatenbezug."""

import numpy as np

import zae_constants as C
import zae_rng as R

_CUM_ART = np.cumsum(C.ART_P)
_UK = np.array(C.UK)
_SALT = 0x9E6C63D0876A9A47


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Bedingte Verteilung der log-Verweilzeit: N(mu, s²), vektorisiert. Das ist der Orakel-Wert des Generators (Test: Mittel von d gleich exp(mu + s²/2))."""
    art = np.asarray(art)
    u = _UK[kunde] * np.take(C.U_SCALE, art)
    med = np.select([art == 0, art == 1, art == 2], [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf], default=7.0 * np.exp(0.30 * we))
    return np.log(med) + u, np.take(C.S_BASE, art)


def make_history(n: int, seed: int):
    """Eine Historie aus n abgeschlossenen Containern: Merkmalsmatrix X (Spalten art, kunde, zoll, wochenende, vorlauf, gewicht) und Verweilzeit d in Tagen."""
    u = R.uniform_array(seed, 5 * n).reshape(5, n)
    z = R.normal_array(seed ^ _SALT, 2 * n)
    art = np.minimum(np.searchsorted(_CUM_ART, u[0], side="right"), 3)
    kunde = np.minimum((u[1] * C.K_KUNDEN).astype(int), C.K_KUNDEN - 1)
    zoll = ((u[2] < C.P_ZOLL) & (art == 0)).astype(int)
    tag = u[3] * 7000.0
    we = ((np.floor(tag).astype(int) % 7) >= 5).astype(int)
    vorlauf = np.where(art == 1, 1.0 + 9.0 * u[4], np.where(art == 2, 0.5 + 5.5 * u[4], 0.0))
    mu, s = log_median_and_s(art, kunde, zoll, we, vorlauf)
    d = np.exp(mu + s * z[:n])
    X = np.column_stack([art, kunde, zoll, we, vorlauf, z[n:]]).astype(float)
    return X, d


def standtage(d: np.ndarray) -> np.ndarray:
    """Begonnene Tage: aufgerundete Verweilzeit, immer mindestens 1 (keine Nullen)."""
    return np.ceil(d)


def standgeld_tage(d: np.ndarray, frei: int = C.STANDGELD_FREI_TAGE) -> np.ndarray:
    """Annahme der Messung: Standgeld ist ab dem 8. Tag fällig, Standgeld-Tage = max(0, Standtage − 7); viele Nullen."""
    return np.maximum(0.0, np.ceil(d) - frei)


def design(X: np.ndarray, kunden: bool = False) -> np.ndarray:
    """Entwurfsmatrix des Modells: Achsenabschnitt, Art-Dummies (Export, Umschlag, Leer), Zoll, Wochenende, log(1 + Vorlauf), Gewicht; mit `kunden` zusätzlich 11 Kundenspalten
    (Kunde 1 bis 11, Kunde 0 ist die Referenz)."""
    art = X[:, 0].astype(int)
    cols = [np.ones(len(X)), art == 1, art == 2, art == 3, X[:, 2], X[:, 3], np.log1p(X[:, 4]), X[:, 5]]
    if kunden:
        cols.extend(X[:, 1] == k for k in range(1, C.K_KUNDEN))
    return np.column_stack(cols).astype(float)
