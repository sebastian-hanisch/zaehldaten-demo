"""Auswertung und Verdikte. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen
und jede Zahl im Text an handgesetzten Werten. Die vorgerechneten Messreihen stehen in `data/zae_frozen.json` (`tools/zae_make_frozen.py`); Zähleinheit überall: eine Wiederholung
= ein Datensatz (bzw. eine Historie aus n Containern), wo nichts anderes steht."""

import json
import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import zae_asymptotics as AS
import zae_constants as C
import zae_hand as H
import zae_models as M
import zae_sample as S

FROZEN_PATH = Path(__file__).resolve().parent / "data" / C.FROZEN_FILE
HALT, KNAPP = 0.93, 0.90          # Klassen der Überdeckung in den Verdikten: ab 93 %, 90 bis unter 93 %, unter 90 % (nominal 95 %)
NO_EFFECT_COVER = 0.93            # ab dieser Überdeckung des Poisson-Intervalls meldet die App keinen Effekt der Überstreuung


# --- Zahlenformat (deutsches Dezimalkomma) -------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 1) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen; nicht endliche Werte als „nicht definiert“."""
    if not math.isfinite(x):
        return "nicht definiert"
    s = f"{x:.{nd}f}".replace(".", ",")
    return "−" + s[1:] if s.startswith("-") else s


def pct(x: float, nd: int = 1) -> str:
    """Anteil 0..1 als „74,4 %“."""
    return f"{de(100 * x, nd)} %"


def frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


# --- Experiment 1: von Hand ----------------------------------------------------------------------------------------------------------------------

HAND_NAMES = {"naive": "Poisson", "quasi": "Quasi-Poisson (t)", "hc0": "Sandwich HC0", "hc1": "Sandwich HC1", "nb2": "NB2"}


def analyse_hand(h) -> dict:
    """Die acht Werte: erste vier ohne Zollhalt, letzte vier mit Zollhalt. Wirft `H.NoEventsError`, wenn eine Gruppe kein Ereignis hat."""
    return H.hand_numeric(h[:4], h[4:])


def excludes_zero(iv) -> bool:
    return iv[0] > 0 or iv[1] < 0


def verdict_hand(res: dict) -> tuple[str, str]:
    ex = res["exact"]
    df = ex["nA"] + ex["nB"] - 2
    parts = []
    flags = {}
    for k in ("naive", "quasi", "hc0", "nb2"):
        lo, hi = res["iv"][k]
        flags[k] = excludes_zero(res["iv"][k])
        name = HAND_NAMES[k] if k != "quasi" else f"Quasi-Poisson (t, {df} Freiheitsgrade)"
        extra = f" (φ̂ = {de(float(ex['phi']), 2)})" if k == "quasi" else ""
        parts.append(f"{name}: z = {de(res['z'][k], 2)}, Intervall [{de(lo, 2)}; {de(hi, 2)}] {'schließt 0 aus' if flags[k] else 'enthält 0'}{extra}")
    text = "; ".join(parts) + ". Gilt für diese acht Werte, nichts über Häufigkeiten."
    kind = "warning" if len(set(flags.values())) > 1 else "info"
    if kind == "warning":
        text = f"Gleicher Punktschätzer (Faktor {de(res['faktor'], 2)} bei den Zählwerten), aber verschiedene Schlüsse: " + text
    return kind, text


def verdict_hand_empty(message: str) -> tuple[str, str]:
    return "warning", f"{message} Die App rechnet hier keine Zahl, weil es keinen endlichen Schätzer gibt."


def hand_fractions(res: dict) -> list[tuple[str, str]]:
    """Die Varianz von β̂₁ je Verfahren als Bruch (Zeilen für die Anzeige)."""
    ex = res["exact"]
    return [("naiv, 1/T_A + 1/T_B", frac(ex["var_naive"])), ("Quasi-Poisson, φ̂ · naiv", frac(ex["var_quasi"])), ("Sandwich HC0, Σ RSS_g / T_g²", frac(ex["var_hc0"])),
            ("Sandwich HC1, n/(n − p) · HC0", frac(ex["var_hc1"])), ("Pearson-φ̂", frac(ex["phi"])), ("Pearson-χ²", frac(ex["x2"]))]


def verdict_exposure(res: dict, y: tuple) -> tuple[str, str]:
    r1, r2 = res["rate1"], res["rate2"]
    return "info", (f"Raten: Kunde 1 {de(float(r1), 2)} Standtage je Container, Kunde 2 {de(float(r2), 2)}; Verhältnis {frac(res['verhaeltnis'])} = {de(float(res['verhaeltnis']), 3)}. "
                    f"Poisson mit Offset log(Container) gibt β = ({de(res['beta'][0], 4)}; {de(res['beta'][1], 4)}): dieselben Zahlen wie die Rate, ohne dass die Containerzahl als Merkmal auftaucht.")


# --- Experiment 2: wann kippt das Poisson-Intervall? ---------------------------------------------------------------------------------------------

def welt_par(welt: str, alpha: float) -> float:
    return alpha if welt == "NB2" else S.delta_for_alpha(alpha)


@lru_cache(maxsize=64)
def limit_cached(welt: str, alpha: float) -> dict:
    """Grenzwert der Überdeckung (Orakel ohne Zufall) für Poisson und Quasi-Poisson."""
    return AS.limit_coverage(welt, welt_par(welt, alpha))


def grid_cell(frozen: dict, welt: str, alpha: float, n: int) -> dict:
    """Vorgerechnete Zelle des Rasters; NB1 mit α = 0 ist dieselbe Welt wie NB2 mit α = 0."""
    if alpha == 0:
        welt = "NB2"
    for c in frozen["grid"]:
        if c["welt"] == welt and c["alpha"] == alpha and c["n"] == n:
            return c
    raise KeyError((welt, alpha, n))


def cover_classes(cov: dict) -> dict:
    """Verfahren nach Überdeckung: 'unter' 90 %, 'mitte' 90 bis unter 93 %, 'ab' 93 %."""
    out = {"unter": [], "mitte": [], "ab": []}
    for m in M.METHODS:
        v = cov[m]
        out["ab" if v >= HALT else "mitte" if v >= KNAPP else "unter"].append(m)
    return out


def _methods_list(ms):
    return ", ".join(M.METHOD_LABELS[m] if m not in ("Quasi z", "Quasi t", "HC3a") else {"Quasi z": "Quasi-Poisson (z)", "Quasi t": "Quasi-Poisson (t)", "HC3a": "HC3-Analogon"}[m] for m in ms) or "keines"


def verdict_cover(cell: dict, limit: dict, welt: str, alpha: float, n: int) -> tuple[str, str]:
    cov = cell["coverage"]
    se_mc = 100 * cell["se_mc"]
    kopf = (f"{'NB2-Welt' if (welt == 'NB2' or alpha == 0) else 'NB1-Welt'}, α = {de(alpha, 2)}, n = {n}, {cell['R']} Wiederholungen (vorgerechnet; Monte-Carlo-Standardfehler "
            f"{de(se_mc, 1)} Prozentpunkte): Poisson {pct(cov['Poisson'])}, Quasi-Poisson (z) {pct(cov['Quasi z'])}, Sandwich HC0 {pct(cov['HC0'])}, HC3-Analogon {pct(cov['HC3a'])}, NB2 {pct(cov['NB2'])} (nominal 95 %).")
    cl = cover_classes(cov)
    klassen = (f" Nach Klassen: unter 90 %: {_methods_list(cl['unter'])}; 90 bis unter 93 %: {_methods_list(cl['mitte'])}; ab 93 %: {_methods_list(cl['ab'])}.")
    if alpha == 0:
        if cov["Poisson"] >= NO_EFFECT_COVER:
            ausnahme = [m for m in M.METHODS if cov[m] < HALT]
            satz = " Hier gibt es keinen Befund gegen Poisson: das Poisson-Intervall hält 93 % oder mehr."
            if ausnahme:
                satz += f" Ausnahme in derselben Welt: {_methods_list(ausnahme)} unter 93 %."
            return "success", kopf + satz + klassen
        return "info", kopf + klassen
    grenz = (f" Grenzwert für n → ∞ aus der Sandwich-Formel (kein Zufall): Poisson {pct(limit['cov_poisson'])}, Quasi-Poisson (z) {pct(limit['cov_quasi'])}.")
    if welt == "NB2":
        deut = " Quasi-Poisson hält also schon im Grenzwert nicht 95 %: ein einziges φ passt nicht zu einer Varianz, die quadratisch im Mittel wächst (Deutung aus der Formel, nicht eigens gemessen)."
    else:
        deut = " In dieser Welt ist das Skalar φ die richtige Varianzfunktion: Quasi-Poisson hält im Grenzwert 95,0 %, das Poisson-Intervall nicht."
    ausnahme = ""
    if n <= 30 and cov["HC0"] < cov["Quasi z"]:
        ausnahme = f" Ausnahme im selben Satz: bei n = {n} liegt das Sandwich HC0 mit {pct(cov['HC0'])} unter Quasi-Poisson (z), die Hebelkorrektur (HC3-Analogon) hebt es auf {pct(cov['HC3a'])}."
    kind = "warning" if cov["Poisson"] < KNAPP else "info"
    return kind, kopf + grenz + deut + ausnahme + klassen


def verdict_live(cell: dict, ref: dict) -> tuple[str, str]:
    cov, rc = cell["coverage"], ref["coverage"]
    se = 100 * cell["se_mc"]
    return "info", (f"Live-Probe mit {cell['R']} Wiederholungen (Monte-Carlo-Standardfehler {de(se, 1)} Prozentpunkte je Zahl): Poisson {pct(cov['Poisson'])}, Quasi-Poisson (z) {pct(cov['Quasi z'])}, "
                    f"HC0 {pct(cov['HC0'])}, NB2 {pct(cov['NB2'])}. Vorgerechnet ({ref['R']} Wiederholungen): Poisson {pct(rc['Poisson'])}, Quasi-Poisson (z) {pct(rc['Quasi z'])}, HC0 {pct(rc['HC0'])}, NB2 {pct(rc['NB2'])}. "
                    f"Die Live-Probe hat einen anderen Zufalls-Seed und weniger Wiederholungen: Abweichungen von ein bis zwei Standardfehlern sind zu erwarten.")


def single_draw(seed: int, n: int, welt: str, alpha: float) -> dict:
    X, y, _ = S.draw_world(seed, n, welt, alpha)
    tab = M.coefficient_table(X, y, 1)
    tab["y"] = y
    tab["x"] = X[:, 1]
    return tab


def verdict_single(tab: dict, truth: float = C.BETA1) -> tuple[str, str]:
    miss = [m for m in M.METHODS if m in tab and not (tab[m]["lo"] <= truth <= tab[m]["hi"])]
    if not miss:
        return "info", f"Dieser eine Datensatz: alle sieben Intervalle enthalten den wahren Anstieg {de(truth, 1)}. Ein Datensatz sagt nichts über Häufigkeiten."
    return "info", (f"Dieser eine Datensatz: den wahren Anstieg {de(truth, 1)} enthalten nicht: {_methods_list(miss)}. Der Anstieg ist bei allen Poisson-Verfahren {de(tab['Poisson']['est'], 3)}, "
                    f"nur die Breite des Intervalls unterscheidet sich. Ein Datensatz sagt nichts über Häufigkeiten.")


# --- Experiment 3: Standtage der Container -------------------------------------------------------------------------------------------------------

def veh_cell(frozen: dict, vn: int, kunden: bool) -> dict:
    for c in frozen["vehicle"]["cells"]:
        if c["n"] == vn and c["kunden"] == kunden:
            return c
    raise KeyError((vn, kunden))


def verdict_vehicle(frozen: dict, vn: int, kunden: bool, spalte: str) -> tuple[str, str]:
    cell = veh_cell(frozen, vn, kunden)
    sp = cell["spalten"][spalte]
    lim = frozen["vehicle"]["limit"]["mit_kunden" if kunden else "ohne_kunden"]
    kund = "mit 11 Kundenspalten im Modell" if kunden else "ohne Kunden im Modell"
    kopf = f"n = {vn} Container, {cell['R']} Wiederholungen (vorgerechnet), {kund}, Pearson-φ̂ im Mittel {de(cell['phi_mean'], 2)}: "
    if spalte.startswith("Gewicht"):
        a = {m: sp[m]["fehlalarm"] for m in ("Poisson", "Quasi z", "HC0", "NB2")}
        text = (f"das Rauschmerkmal Gewicht (wahre Wirkung null) schlägt bei Poisson in {pct(a['Poisson'])}, Quasi-Poisson (z) {pct(a['Quasi z'])}, NB2 {pct(a['NB2'])} und Sandwich HC0 {pct(a['HC0'])} "
                f"der Wiederholungen an (nominal 5 %; Monte-Carlo-Standardfehler {de(100 * cell['se_mc'] * math.sqrt(5 / 95), 1)} Prozentpunkte).")
        kind = "warning" if a["Poisson"] > 0.10 else "info"
    else:
        c = {m: sp[m]["coverage"] for m in ("Poisson", "Quasi z", "HC0", "NB2")}
        text = (f"{spalte}: Poisson {pct(c['Poisson'])}, Quasi-Poisson (z) {pct(c['Quasi z'])}, NB2 {pct(c['NB2'])}, Sandwich HC0 {pct(c['HC0'])} Überdeckung (nominal 95 %; Monte-Carlo-Standardfehler "
                f"{de(100 * cell['se_mc'], 1)} Prozentpunkte).")
        kind = "warning" if c["Poisson"] < KNAPP else "info"
    r = {m: sp[m]["ratio"] for m in ("Poisson", "Quasi z", "HC0", "NB2")}
    text += (f" Mittlerer Standardfehler geteilt durch die wahre Streuung der Schätzung: Poisson {de(r['Poisson'], 2)}, Quasi (z) {de(r['Quasi z'], 2)}, HC0 {de(r['HC0'], 2)}, NB2 {de(r['NB2'], 2)}. "
             f"Ziel ist der Grenzwert des jeweiligen Schätzers (n = {de(lim['n'] / 1e6, 0)} Million), nicht ein wahrer Koeffizient: das Mittelwertmodell der Standtage ist nicht exakt log-linear.")
    return kind, kopf + text


def verdict_dispersion(frozen: dict) -> tuple[str, str]:
    lim = frozen["vehicle"]["limit"]["ohne_kunden"]
    lk = frozen["vehicle"]["limit"]["mit_kunden"]
    pa = lim["phi_je_art"]
    unter = [C.ART_LABELS[a] for a in range(4) if pa[a] < 1]
    return "info", (f"Standtage im Container-Vehikel (eine Historie aus {de(lim['n'] / 1e6, 0)} Million Containern): Mittel {de(lim['mean'], 2)}, Varianz {de(lim['var'], 2)}, Varianz geteilt durch Mittel {de(lim['var'] / lim['mean'], 2)} "
                    f"(roh über alle Container zusammen; das ist nicht die Überdispersion gegenüber dem Modell: schon ein exakt passendes Poisson-Modell zeigt roh mehr als 1, weil die erwarteten Werte der Container verschieden sind). "
                    f"Gegenüber dem angepassten Modell misst die Streuung das Pearson-φ̂ {de(lim['phi'], 2)} (bei Poisson 1); je Containerart " + ", ".join(f"{C.ART_LABELS[a]} {de(pa[a], 2)}" for a in range(4))
                    + f": ein einziges φ passt nicht, in {len(unter)} von 4 Arten ({', '.join(unter) or 'keiner'}) ist die Streuung gegen das angepasste Mittel kleiner als Poisson. Mit 11 Kundenspalten im Modell sinkt φ̂ auf {de(lk['phi'], 2)}.")


# --- Experiment 4: Quasi-Poisson ist nicht NB2 ---------------------------------------------------------------------------------------------------

def weight_ratio_value(alpha: float, mu_big: int = 20, mu_small: int = 1) -> Fraction:
    return H.weight_ratio(Fraction(mu_big), Fraction(mu_small), Fraction(str(alpha)))


def verdict_weights(alpha: float) -> tuple[str, str]:
    if alpha == 0:
        return "info", "Bei α = 0 ist NB2 gleich Poisson: beide gewichten einen Fall mit Mittel 20 gegenüber einem mit Mittel 1 mit dem Faktor 20."
    w = weight_ratio_value(alpha)
    return "info", (f"Ein Fall mit Mittel 20 zählt gegenüber einem mit Mittel 1 das 20-Fache bei Poisson und Quasi-Poisson, aber nur das {de(float(w), 3)}-Fache bei NB2 mit α = {de(alpha, 2)} "
                    f"(Bruch {frac(w)}; Gewicht μ/(1 + αμ) gegen μ).")


def _vorn(d: float, se: float) -> str:
    """Wer liegt vorn? Nur ein Abstand über zwei Monte-Carlo-Standardfehlern zählt (d = NB2 minus Quasi-Poisson (z), in Anteilen)."""
    if d > 2 * se:
        return "NB2 liegt vorn"
    if d < -2 * se:
        return "Quasi-Poisson (z) liegt vorn"
    return "kein Abstand über zwei Standardfehlern"


def verdict_two_worlds(c_nb2: dict | None, c_nb1: dict | None, alpha: float, n: int) -> tuple[str, str]:
    if c_nb2 is None or c_nb1 is None:
        return "info", "Bei α = 0 sind NB2- und NB1-Welt dieselbe Poisson-Welt: es gibt nichts zu vergleichen."
    q2, n2 = c_nb2["coverage"]["Quasi z"], c_nb2["coverage"]["NB2"]
    q1, n1 = c_nb1["coverage"]["Quasi z"], c_nb1["coverage"]["NB2"]
    se2, se1 = c_nb2["se_mc"], c_nb1["se_mc"]
    v2, v1 = _vorn(n2 - q2, se2), _vorn(n1 - q1, se1)
    fazit = ""
    if v2 == "NB2 liegt vorn" and v1 == "Quasi-Poisson (z) liegt vorn":
        fazit = " Kein Verfahren ist in beiden Welten das beste."
    return "info", (f"α = {de(alpha, 2)}, n = {n}. NB2-Welt ({c_nb2['R']} Wiederholungen, Monte-Carlo-Standardfehler {de(100 * se2, 1)} Prozentpunkte): NB2 {pct(n2)} gegen Quasi-Poisson (z) {pct(q2)}, {v2}. "
                    f"NB1-Welt ({c_nb1['R']} Wiederholungen, {de(100 * se1, 1)} Prozentpunkte): NB2 {pct(n1)} gegen Quasi-Poisson (z) {pct(q1)}, {v1}.{fazit}")


def missspec_entry(frozen: dict, c: float) -> tuple[dict, dict]:
    lim = next(e for e in frozen["missspec"]["limit"] if e["c"] == c)
    dist = next(e for e in frozen["missspec"]["n200"] if e["c"] == c)
    return lim, dist


def verdict_missspec(frozen: dict, c: float) -> tuple[str, str]:
    lim, dist = missspec_entry(frozen, c)
    if c == 0:
        return "success", (f"Richtiges Mittelwertmodell (c = 0): Poisson {de(lim['poisson'], 3)} und NB2 {de(lim['nb2'], 3)} liegen bei n = {de(lim['n'] / 1000, 0)} 000 beide bei 0,4 "
                           f"(Unterschied im Median {de(dist['diff_median'], 3)} bei n = 200, {dist['R']} Wiederholungen). Hier schätzen beide dieselbe Größe.")
    n_gross = f"{lim['n']:,}".replace(",", ".")
    return "warning", (f"Falsches Mittelwertmodell (c = {de(c, 1)}): ein großer Datensatz (n = {n_gross}) gibt für den Anstieg Poisson {de(lim['poisson'], 3)} gegen NB2 {de(lim['nb2'], 3)} (der wahre Anstieg bei x = 0 ist 0,4). "
                       f"Bei n = 200 ({dist['R']} Wiederholungen) liegt der Unterschied β̂₁(NB2) − β̂₁(Poisson) im Median bei {de(dist['diff_median'], 2)} (Quartile {de(dist['diff_q1'], 2)} und {de(dist['diff_q3'], 2)}) "
                       f"bei einer robusten Streuung von β̂₁(Poisson) von {de(dist['poisson_robust_sd'], 2)}; in {dist['diff_gross']} von {dist['R']} Datensätzen ist der Betrag des Unterschieds größer als 1 "
                       f"(Mittel und Standardabweichung des Unterschieds nennt die App nicht: einzelne Ausreißer dominieren sie). Quasi-Poisson hat dieselben Koeffizienten wie Poisson, NB2 nicht."
                       + (" Ab c = 0,25 hat μ² keinen endlichen Erwartungswert (das Integral von exp(2c x²) gegen die Normaldichte divergiert): die Schätzung des Poisson-Anstiegs konvergiert dann nicht mit der üblichen Rate, "
                          "und einzelne Datensätze weichen stark ab; der Grenzwert selbst existiert (0,4/(1 − 2c), bei c = 0,3 gleich 1,000)." if c >= 0.25 else ""))


# --- Experiment 5: Nullen und fehlende Merkmale -------------------------------------------------------------------------------------------------

def verdict_zeros(frozen: dict, ausgang: str) -> tuple[str, str]:
    z = frozen["nullen"]
    if ausgang == "Standtage":
        lim = frozen["vehicle"]["limit"]["ohne_kunden"]
        return "info", (f"Standtage sind begonnene Tage und haben keine Nullen (kleinster Wert {de(lim['min'], 0)} in einer Historie aus {de(lim['n'] / 1e6, 0)} Million Containern). "
                        "Die Nullfrage stellt sich erst beim abgeleiteten Zählausgang Standgeld-Tage (Auswahl oben).")
    t, tk = z["ohne_kunden"], z["mit_kunden"]
    lim = z["limit"]
    je = lim["je_art"]
    return "warning", (f"Standgeld-Tage (Annahme dieser Messung: max(0, Standtage − 7)), n = 3000, {t['R']} Wiederholungen: Nullanteil beobachtet {de(t['beobachtet'], 3)}, "
                       f"Poisson erwartet {de(t['poisson_erwartet'], 3)}, NB2 erwartet {de(t['nb2_erwartet'], 3)}; Pearson-φ̂ {de(t['phi'], 2)} ohne und {de(tk['phi'], 2)} mit 11 Kundenspalten im Modell. "
                       f"Je Containerart (eine Historie aus {de(lim['n'] / 1e6, 0)} Million Containern, beobachtet / Poisson / NB2): "
                       + "; ".join(f"{a} {de(je[a]['beobachtet'], 3)} / {de(je[a]['poisson'], 3)} / {de(je[a]['nb2'], 3)}" for a in C.ART_LABELS)
                       + f". NB2 trifft den Gesamtanteil, aber je Art liegt es um bis zu {de(100 * max(abs(je[a]['nb2'] - je[a]['beobachtet']) for a in C.ART_LABELS), 0)} Prozentpunkte daneben. "
                       "Ob ein Zero-Inflation- oder Hurdle-Modell das behebt, ist nicht gemessen.")


# --- Kernfragen -------------------------------------------------------------------------------------------------------------------------------------

def verdict_q2(cell: dict, limit: dict, welt: str, alpha: float, n: int) -> tuple[str, str]:
    return verdict_cover(cell, limit, welt, alpha, n)


def collect_cells(frozen: dict, alpha: float, n: int):
    """Beide Welten bei α und n für den Vergleich Quasi-Poisson gegen NB2; (None, None) bei α = 0."""
    if alpha == 0:
        return None, None
    return grid_cell(frozen, "NB2", alpha, n), grid_cell(frozen, "NB1", alpha, n)
