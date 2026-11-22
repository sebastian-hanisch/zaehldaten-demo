"""Zählen: Poisson, Überdispersion und die Frage, welcher Standardfehler stimmt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Standtage der Container als Zählvariable: Poisson nimmt Varianz = Mittel an, die Standtage streuen aber insgesamt um ein Vielfaches mehr (gegen das angepasste Modell gemessen etwa
dreimal so stark, bei Export und Umschlag dagegen weniger). Die Demo misst, wie schmal die Intervalle des
Poisson-Modells dadurch werden, was Quasi-Poisson, negativ binomiales Modell und Sandwich-Verfahren daran ändern, und wo sie selbst an Grenzen stoßen.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import zae_constants as C
import zae_evaluation as ev
import zae_generator as G
import zae_hand as H
import zae_models as M
import zae_simulation as SIM
import zae_visualization as V
from zae_layout import dunkles_design, kopf_css
from zae_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Zählen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(h):
    return ev.analyse_hand(h)


@st.cache_data(show_spinner=False)
def _live_cell(welt, alpha, n, reps, seed):
    return SIM.run_cell(welt, alpha, n, reps, seed)


@st.cache_data(show_spinner=False)
def _single(seed, n, welt, alpha):
    return ev.single_draw(seed, n, welt, alpha)


@st.cache_data(show_spinner=False)
def _vehicle_single(seed, n, kunden, col):
    X, d = G.make_history(n, seed)
    return M.coefficient_table(G.design(X, kunden), G.standtage(d), col)


def _de(x, nd=1):
    return ev.de(x, nd)


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


frozen = ev.load_frozen()

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zae_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔢 Zählen: Poisson, Überdispersion und die Frage, welcher Standardfehler stimmt")
_lim0 = frozen["vehicle"]["limit"]["ohne_kunden"]
st.markdown(
    f"""
Zählt man **Standtage** (begonnene Tage eines Containers im Hafen) und rechnet **Poisson**, nimmt man an: **Varianz = Mittel** (je Container, bei gegebenen Merkmalen). Im Container-Vehikel streuen die Standtage
insgesamt um ein Vielfaches mehr als ihr Mittel (Varianz geteilt durch Mittel {_de(_lim0['var'] / _lim0['mean'], 1)}); ein Teil davon kommt nur von den verschiedenen Erwartungswerten der Container. Gegen das angepasste Modell gemessen ist die Streuung
rund das {_de(_lim0['phi'], 1)}-Fache (Pearson-φ̂ {_de(_lim0['phi'], 2)}, **Überdispersion**), und bei Export und Umschlag streuen die Standtage sogar **weniger** als Poisson (Experiment 3).
Der Schätzer existiert, aber seine **Standardfehler sind zu klein**: die 95-%-Intervalle des Poisson-Modells decken viel seltener als 95 %, und ein Merkmal ohne Wirkung schlägt zu oft an.
Die Demo misst das mit Zähleinheiten (Wiederholung = ein Datensatz), prüft die üblichen Reparaturen (**Quasi-Poisson**, **negativ binomial**, **Sandwich-Standardfehler**) und zeigt,
dass sie die 95 % nicht überall herstellen und nicht dasselbe sind.

Weiter unten: die **Kernfragen**, fünf Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Schnellstart und Zufalls-Seed stehen in der Seitenleiste. Wer das Thema zuerst ohne Formeln verstehen will, öffnet „💬 In einfachen Worten“.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/ueberdispersion-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich vor, Sie zählen, wie viele Tage Container im Hafen stehen, und schätzen mit dem einfachsten Zählmodell (**Poisson**), wie Zollhalt oder Wochenende die **Standtage** verlängern. Dieses Modell nimmt an, dass die Zählwerte nicht stärker schwanken, als es ihr Durchschnitt vorgibt. Schwanken sie stärker (**Überdispersion**), bleibt die Schätzung brauchbar, aber die Angabe, wie sicher sie ist, wird zu klein: Die Intervalle treffen zu selten, und ein Merkmal ohne Wirkung schlägt zu oft an. Die Demo misst das an erfundenen Daten über viele Wiederholungen und prüft die üblichen Reparaturen: **Quasi-Poisson**, die **negative Binomialverteilung** und **Sandwich-Standardfehler**. Sie zeigt, dass die Reparaturen nicht dasselbe sind und keine von ihnen in allen gemessenen Fällen hält, was sie verspricht. Echte Hafendaten kommen nicht vor.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/ueberdispersion-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert Zählen mit Standardfehlern", expanded=True):
    st.markdown(
        """
1. **Poisson-Regression:** der Zählwert y hat das Mittel μ = exp(x'β). Der Schätzer β̂ löst die Score-Gleichung Σ (y − μ) x = 0 (iterativ gewichtete kleinste Quadrate, hier selbst gerechnet).
2. **Standardfehler:** gilt Varianz = Mittel, steht die Streuung von β̂ in der Informationsmatrix (X'WX)⁻¹. Gilt es nicht, hängt die wahre Streuung an der wahren Varianz.
3. **Reparaturen:** *Quasi-Poisson* skaliert die Varianz mit dem Pearson-φ̂ (gleiche Koeffizienten). *Sandwich* (HC0, HC1, mit Hebelkorrektur) schätzt die Streuung aus den Residuen. *NB2* nimmt die
   Varianz μ + αμ² an und schätzt β und α gemeinsam (andere Gewichte, andere Koeffizienten, wenn das Mittelwertmodell nicht stimmt).
4. **Prüfen:** nicht an einem Datensatz, sondern über viele Wiederholungen: wie oft deckt das Intervall den Grenzwert?
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Live-Datensätze", *(lambda b: (int(b[0]), int(b[1])))((SETTING_SPECS["seed_input"].lo, SETTING_SPECS["seed_input"].hi)), key="seed_input", step=1,
                           help="Gilt für den einen Datensatz und die Live-Probe in Experiment 2 und die eine Historie in Experiment 3. Die vorgerechneten Reihen haben feste Zufalls-Seeds.")
    st.button("🎲 Neue Datensätze ziehen", width="stretch", on_click=randomize_seed)
    st.caption("Die Regler der Experimente stehen im Hauptbereich beim jeweiligen Experiment; ein Preset setzt sie alle zugleich.")

S_ = st.session_state

# --- Experiment 1: Zählen von Hand ----------------------------------------------------------------------------------------------------------------------

st.subheader("🧮 Zählen von Hand: acht Zählwerte, zwei Gruppen")
st.markdown(
    "Acht **erfundene** Zählwerte (nicht aus dem Generator gezogen, es sind keine Standtage: sie enthalten Nullen, die echte Standtage nicht haben), damit alles in Brüchen nachrechenbar bleibt: "
    "vier Container **ohne** Zollhalt, vier **mit**. Gesucht ist der Faktor, um den der Zollhalt die erwarteten Zählwerte erhöht. Die Zahlen sind änderbar."
)
cA, cB = st.columns(2)
with cA:
    st.markdown("**Ohne Zollhalt** (Zählwert je Container)")
    cols = st.columns(4)
    for i in range(4):
        cols[i].number_input(f"A{i + 1}", min_value=0, max_value=C.HAND_MAX, step=1, key=f"h{i + 1}")
with cB:
    st.markdown("**Mit Zollhalt**")
    cols = st.columns(4)
    for i in range(4):
        cols[i].number_input(f"B{i + 1}", min_value=0, max_value=C.HAND_MAX, step=1, key=f"h{i + 5}")
hand_values = tuple(int(S_[f"h{i + 1}"]) for i in range(8))
hand_res = None
try:
    hand_res = _hand(hand_values)
except H.NoEventsError as exc:
    _show(ev.verdict_hand_empty(str(exc)))
except ValueError as exc:
    st.warning(str(exc))
if hand_res is not None:
    ex = hand_res["exact"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gruppenmittel ohne / mit Zollhalt", f"{ev.frac(ex['mA'])} / {ev.frac(ex['mB'])}")
    m2.metric("Faktor exp(β̂₁)", _de(hand_res["faktor"], 3))
    m3.metric("Pearson-φ̂", f"{ev.frac(ex['phi'])} = {_de(float(ex['phi']), 2)}")
    m4.metric("IRLS-Schritte aus β = 0", str(hand_res["irls_schritte"]), help="Abbruch bei Änderung unter 1e-13 (Gleitkomma); der erste Schritt ist exakt (1/2; 7/2) beim Voreinstellungs-Beispiel.")
    rows = [{"Verfahren": ev.HAND_NAMES[k], "Standardfehler": _de(hand_res["se"][k], 4), "z": _de(hand_res["z"][k], 2),
             "95-%-Intervall": f"[{_de(hand_res['iv'][k][0], 2)}; {_de(hand_res['iv'][k][1], 2)}]", "enthält 0": "ja" if not ev.excludes_zero(hand_res["iv"][k]) else "nein"} for k in ("naive", "quasi", "hc0", "hc1", "nb2")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.plotly_chart(V.build_hand_intervals(hand_res), width="stretch", key="hand_chart")
    st.markdown("**Varianz von β̂₁ als Bruch** (Gruppenformeln, `fractions`):")
    st.dataframe(pd.DataFrame(ev.hand_fractions(hand_res), columns=["Größe", "Bruch"]), hide_index=True, width="stretch")
    _show(ev.verdict_hand(hand_res))
    st.caption(f"NB2: α̂ = {_de(hand_res['alpha_nb2'], 3)} aus der Profilgleichung; Var(β̂₁) = 1/T_A + 1/T_B + α̂ (1/n_A + 1/n_B). Das Quasi-Intervall nutzt t mit {ex['nA'] + ex['nB'] - 2} Freiheitsgraden, alle anderen z.")

with st.expander("Exposition: Standtage je Container (Offset)"):
    st.markdown("Zählt man Standtage je Kunde und die Kunden haben verschieden viele Container, gehört die Containerzahl als **Exposition** t ins Modell: μ = t · Rate, also Offset log t. Zwei Kunden:")
    ec = st.columns(4)
    ec[0].number_input("Standtage Kunde 1", min_value=0, max_value=9999, step=1, key="e1")
    ec[1].number_input("Container Kunde 1", min_value=0, max_value=9999, step=1, key="e2")
    ec[2].number_input("Standtage Kunde 2", min_value=0, max_value=9999, step=1, key="e3")
    ec[3].number_input("Container Kunde 2", min_value=0, max_value=9999, step=1, key="e4")
    try:
        er = H.exposure_example(int(S_["e1"]), int(S_["e2"]), int(S_["e3"]), int(S_["e4"]))
        _show(ev.verdict_exposure(er, ()))
    except H.NoEventsError as exc:
        st.warning(str(exc))

st.markdown("---")

# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------------------

welt, alpha, n = S_["welt"], float(S_["alpha"]), int(S_["n"])
cell = ev.grid_cell(frozen, welt, alpha, n)
limit = ev.limit_cached("NB2" if alpha == 0 else welt, alpha)
st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie viel Streuung steckt in den Zählwerten gegenüber Poisson (Varianz = Mittel)?**")
_show(ev.verdict_dispersion(frozen))
st.markdown(f"**2. Welcher Standardfehler stimmt, wenn nicht Poisson?** (Einstellung aus Experiment 2: {welt}-Welt, α = {_de(alpha, 2)}, n = {n})")
_show(ev.verdict_cover(cell, limit, welt, alpha, n))
st.markdown("**3. Sind Quasi-Poisson und NB2 dasselbe?**")
c2, c1 = ev.collect_cells(frozen, alpha, n)
_show(ev.verdict_two_worlds(c2, c1, alpha, n))
_show(ev.verdict_weights(alpha))

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wann kippt das Poisson-Intervall?")
st.caption(
    "Welt: ein Merkmal x ~ N(0, 1), log μ = 1 + 0,4 x, der Zählwert ist negativ binomial verteilt. Ziel der Intervalle ist der wahre Anstieg 0,4. Zähleinheit: Wiederholung = ein Datensatz aus n Beobachtungen. "
    "Die Balken sind **vorgerechnet** (`tools/zae_make_frozen.py`), die Raute ist der Grenzwert aus der Sandwich-Formel (kein Zufall), die Kreise die Live-Probe."
)
rc = st.columns(3)
rc[0].selectbox("Welt der Streuung", C.WELT_CHOICES, key="welt", format_func=lambda w: C.WELT_LABELS[w], help="NB2: Varianz μ + αμ². NB1: Varianz (1 + δ) μ mit gleicher mittlerer Dispersion wie NB2 bei demselben α.")
rc[1].select_slider("Überdispersion α", options=C.ALPHA_CHOICES, key="alpha", help="α = 0: Poisson. α = 1: Varianz μ + μ².")
rc[2].select_slider("Beobachtungen n je Datensatz", options=C.N_CHOICES, key="n")

live = None
lr = st.columns([2, 1, 3])
lr[0].slider("Wiederholungen der Live-Probe", C.LIVE_R_MIN, C.LIVE_R_MAX, key="live_r", help="Rechenzeit bei n = 200 etwa 2 ms je Wiederholung.")
if lr[1].button("Live-Probe rechnen", key="live_go"):
    S_["live_params"] = (welt, alpha, n, int(S_["live_r"]), int(seed))
params = S_.get("live_params")
if params == (welt, alpha, n, int(S_["live_r"]), int(seed)):
    t0 = time.time()
    with st.spinner(f"Rechne {params[3]} Wiederholungen …"):
        live = _live_cell(*params)
    lr[2].caption(f"Live-Probe: {params[3]} Wiederholungen in {_de(time.time() - t0, 1)} s (beim ersten Rechnen; danach aus dem Zwischenspeicher).")
st.plotly_chart(V.build_coverage(cell, limit, live), width="stretch", key="cover_chart")
_show(ev.verdict_cover(cell, limit, welt, alpha, n))
if live is not None:
    _show(ev.verdict_live(live, cell))
st.markdown("**Ein Datensatz** (Zufalls-Seed der Seitenleiste): sieben Intervalle für den Anstieg")
single = _single(int(seed), n, welt, alpha)
st.plotly_chart(V.build_single(single), width="stretch", key="single_chart")
_show(ev.verdict_single(single))
st.caption(
    f"Pearson-φ̂ dieses Datensatzes {_de(single['phi'], 2)}"
    + (f", NB2-α̂ {_de(single['alpha'], 2)}" if np.isfinite(single["alpha"]) else "")
    + (" (Randlösung: α̂ am Rand des Suchbereichs)" if single.get("alpha_am_rand") else "")
    + f". In den vorgerechneten {cell['R']} Wiederholungen scheiterte NB2 {cell['nb2_ausfaelle']}-mal, lag {cell['nb2_randloesung']}-mal am Rand."
)

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚢 Experiment 3: Standtage der Container")
st.caption(
    "Das Container-Vehikel: Art, Kunde, Zollhalt, Wochenende, Vorlauf und ein Rauschmerkmal Gewicht (Wirkung null); Standtage = aufgerundete Verweilzeit (alle Parameter erfunden). "
    "Ziel der Intervalle ist der **Grenzwert des jeweiligen Schätzers** (aus einer Historie von einer Million Containern), nicht ein wahrer Koeffizient: das Mittelwertmodell ist nicht exakt log-linear."
)
st.plotly_chart(V.build_phi_art(frozen["vehicle"]["limit"]["ohne_kunden"]), width="stretch", key="phi_chart")
vc = st.columns(3)
vc[0].select_slider("Container je Historie", options=C.VN_CHOICES, key="vn")
vc[1].selectbox("Spalte", list(C.VEHICLE_COLUMNS), key="spalte")
vc[2].toggle("Kunden im Modell (11 Spalten)", key="kunden", help="Die zwölf Kunden wirken auf die Verweilzeit; mit ihren Spalten im Modell sinkt die Überdispersion.")
vn, spalte, kunden = int(S_["vn"]), S_["spalte"], bool(S_["kunden"])
vcell = ev.veh_cell(frozen, vn, kunden)
st.plotly_chart(V.build_vehicle(vcell, spalte), width="stretch", key="veh_chart")
_show(ev.verdict_vehicle(frozen, vn, kunden, spalte))
st.markdown("**Eine Historie live** (Zufalls-Seed der Seitenleiste)")
vt = _vehicle_single(int(seed), vn, kunden, C.VEHICLE_COLUMNS[spalte])
vrows = [{"Verfahren": V.LABEL[m], "Schätzung": _de(vt[m]["est"], 3), "Standardfehler": _de(vt[m]["se"], 4), "95-%-Intervall": f"[{_de(vt[m]['lo'], 3)}; {_de(vt[m]['hi'], 3)}]"} for m in M.METHODS if m in vt]
st.dataframe(pd.DataFrame(vrows), hide_index=True, width="stretch")
st.caption(f"Pearson-φ̂ dieser Historie {_de(vt['phi'], 2)} (n = {vn}). Der Schalter „Kunden im Modell“ oben ändert φ̂ und die vorgerechneten Zahlen.")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("⚖️ Experiment 4: Quasi-Poisson ist nicht NB2")
st.markdown("**(a) Gewichte.** Poisson (und Quasi-Poisson) gewichten einen Fall im IRLS mit μ, NB2 mit μ/(1 + αμ): große Zählwerte zählen bei NB2 weniger. α kommt aus Experiment 2.")
st.plotly_chart(V.build_weights(alpha), width="stretch", key="weight_chart")
_show(ev.verdict_weights(alpha))
st.markdown("**(b) Zwei Welten.** Dieselbe mittlere Dispersion, aber die Varianz wächst quadratisch (NB2-Welt) oder linear (NB1-Welt) im Mittel. Zahlen aus dem Raster von Experiment 2 (α und n von dort).")
_show(ev.verdict_two_worlds(c2, c1, alpha, n))
if c2 is not None:
    tw = pd.DataFrame([{"Welt": "NB2 (quadratisch)", "Wiederholungen": c2["R"], **{V.LABEL[m]: ev.pct(c2["coverage"][m]) for m in ("Poisson", "Quasi z", "HC0", "NB2")}},
                       {"Welt": "NB1 (linear)", "Wiederholungen": c1["R"], **{V.LABEL[m]: ev.pct(c1["coverage"][m]) for m in ("Poisson", "Quasi z", "HC0", "NB2")}}])
    st.dataframe(tw, hide_index=True, width="stretch")
st.markdown("**(c) Falsches Mittelwertmodell.** Die Wahrheit hat einen Krümmungsterm: log μ = 1 + 0,4 x + c x², angepasst wird nur 1 + x.")
st.select_slider("Krümmung c der Wahrheit", options=C.C_CHOICES, key="c")
c_val = float(S_["c"])
st.plotly_chart(V.build_missspec(frozen, c_val), width="stretch", key="miss_chart")
_show(ev.verdict_missspec(frozen, c_val))

st.markdown("---")

# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("0️⃣ Experiment 5: Nullen und fehlende Merkmale")
st.radio("Zählvariable", C.AUSGANG_CHOICES, key="ausgang", horizontal=True, help="Standgeld-Tage: Annahme dieser Messung, Standgeld fällig ab dem 8. Tag, also max(0, Standtage − 7).")
st.plotly_chart(V.build_zeros(frozen, False), width="stretch", key="zero_chart")
_show(ev.verdict_zeros(frozen, S_["ausgang"]))

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Was die Demo zeigt |
|---|---|---|
| **Varianz = Mittel** (Poisson) | Intervalle zu schmal, Merkmale ohne Wirkung schlagen zu oft an | Experiment 2 und 3 |
| **Ein einziges φ** (Quasi-Poisson) | passt nicht zu einer Varianz, die quadratisch im Mittel wächst; in zwei von vier Containerarten ist die Streuung gegen das angepasste Mittel kleiner als Poisson | Kernfrage 1, Experiment 2 und 3 |
| **Varianz μ + αμ²** (NB2) | hält die 95 % im Vehikel nicht und schätzt bei falschem Mittelwertmodell eine andere Größe als Poisson | Experiment 3 und 4 |
| **Große Stichprobe** (Sandwich) | bei n = 30 deckt HC0 in allen 6 gemessenen Zellen mit α > 0 mehr als 2 Prozentpunkte weniger als die Hebelkorrektur | Experiment 2 (Preset „Kleine Stichprobe“) |
| **Das Mittelwertmodell stimmt** | alle Verfahren schätzen sonst Grenzwerte, die sich unterscheiden können; „wahrer Koeffizient“ gibt es dann nicht | Experiment 3 und 4 |
| **Nullen sind Poisson- oder NB2-Nullen** | je Containerart weicht der Nullanteil ab | Experiment 5 (Zero-Inflation und Hurdle sind nicht gemessen) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/ueberdispersion-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Zählwerte $y_i \in \{0, 1, 2, \dots\}$ (Standtage) und Merkmale $x_i$ (erste Spalte konstant 1); gesucht sind Koeffizienten $\beta$ und die Frage, wie sicher sie sind.

| Symbol | Bedeutung |
|---|---|
| $y_i$, $x_i$ | Zählwert (Standtage) und Merkmalszeile des Containers $i$ |
| $\mu_i$ | erwarteter Zählwert, $\mu_i = \exp(x_i^\top \beta)$ (Log-Link; bei Exposition $t_i$: $\mu_i = t_i \exp(x_i^\top\beta)$) |
| $n$, $p$ | Zahl der Beobachtungen und der Spalten (inklusive Achsenabschnitt) |
| $V(\mu)$ | Varianzfunktion: Poisson $\mu$, Quasi-Poisson $\varphi\mu$, NB2 $\mu + \alpha\mu^2$, NB1 $(1+\delta)\mu$ |
| $\hat\varphi$ | Pearson-Dispersion $\sum_i (y_i - \hat\mu_i)^2 / \hat\mu_i \,/\, (n - p)$ |
| $A$, $B$ | $A = \sum_i \hat\mu_i x_i x_i^\top$ (Information), $B = \sum_i r_i^2 x_i x_i^\top$ mit $r_i = y_i - \hat\mu_i$ |
| $h_i$ | Hebel $h_i = \hat\mu_i x_i^\top A^{-1} x_i$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/ueberdispersion-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Schätzer.** Poisson-Quasi-Likelihood: $\sum_i (y_i - \mu_i)\, x_i = 0$, gelöst durch IRLS: $\beta^{+} = (X^\top W X)^{-1} X^\top W z$ mit Gewicht $W = \mu/(1+\alpha\mu)$ ($\alpha = 0$: Poisson) und Arbeitsantwort $z = \eta + (y - \mu)/\mu$.
Quasi-Poisson hat dieselben $\hat\beta$. NB2 maximiert die Log-Likelihood der negativ binomialen Verteilung mit $r = 1/\alpha$; $\beta$ und $\alpha$ werden abwechselnd bestimmt ($\alpha$ aus der Profilgleichung).

**Standardfehler.**

| Verfahren | Kovarianz von $\hat\beta$ |
|---|---|
| Poisson (naiv) | $A^{-1}$ |
| Quasi-Poisson | $\hat\varphi\, A^{-1}$ (Intervall mit $z$ oder $t_{n-p}$) |
| Sandwich HC0 | $A^{-1} B A^{-1}$ |
| Sandwich HC1 | $\tfrac{n}{n-p}\,\mathrm{HC0}$ |
| Hebelkorrektur (HC3-Analogon, in der App „HC3a“) | $A^{-1} \big(\sum_i \tfrac{r_i^2}{(1-h_i)^2} x_i x_i^\top\big) A^{-1}$ |
| NB2 | $(X^\top W X)^{-1}$ mit $W = \hat\mu/(1+\hat\alpha\hat\mu)$ |

**Namensfalle.** Bei MacKinnon und White (1985) heißt HC3 das Jackknife; die heutige Bedeutung (Hebelkorrektur $r_i/(1-h_i)$) stammt von Davidson und MacKinnon (1993). Die Fassung dieser Demo überträgt sie auf den
Poisson-Fall und heißt darum „Hebelkorrektur (HC3-Analogon)“, nicht „HC3“.

**Grenzwert der Überdeckung.** Stimmt das Mittelwertmodell, ist $\sqrt{n}\,(\hat\beta_1 - \beta_1)$ für $n \to \infty$ normalverteilt mit Mittel 0 und Varianz $(A^{-1} B A^{-1})_{11}$, $A = E[\mu\, x x^\top]$, $B = E[V(\mu)\, x x^\top]$ (Erwartung über das Merkmal).
Das z-Intervall des Modells mit Standardfehler $s$ deckt dann mit Wahrscheinlichkeit $2\Phi(1{,}96\, s / \mathrm{SD}) - 1$; für Poisson ist $s^2 = (A^{-1})_{11}$, für Quasi-Poisson $s^2 = \varphi (A^{-1})_{11}$ mit $\varphi = E[V(\mu)/\mu]$.
Dabei ist $V$ die wahre Varianzfunktion, $\Phi$ die Verteilungsfunktion der Standardnormalverteilung und $\mathrm{SD} = \sqrt{(A^{-1} B A^{-1})_{11}}$ die wahre Streuung auf derselben Skala wie $s$ ($s^2$ ist die vom Verfahren geschätzte Varianz auf dieser Skala).
Die App berechnet die Erwartungen auf zwei Wegen (Gauss-Hermite mit 80 Knoten und `scipy.integrate.quad`).

**Gewichte.** Im IRLS zählt ein Fall bei Poisson mit $\mu$, bei NB2 mit $\mu/(1+\alpha\mu)$. Für $\mu = 20$ gegen $\mu = 1$ ist das Verhältnis $20$ bzw. $\tfrac{20/(1+20\alpha)}{1/(1+\alpha)}$; bei $\alpha = \tfrac12$ ist es $\tfrac{30}{11}$.

**Gamma-Mischung.** Ist $y \mid G \sim \text{Poisson}(\mu G)$ mit $G \sim \text{Gamma}$ (Form $1/\alpha$, Mittel 1), so ist $y$ negativ binomial mit Mittel $\mu$ und Varianz $\mu + \alpha \mu^2$.
Bei Mittel 3 und $\alpha = \tfrac12$: $P(k) = (k+1)\tfrac{4}{25}\left(\tfrac35\right)^k$, $P(0) = \tfrac{4}{25} = 0{,}16$ gegen $e^{-3} \approx 0{,}0498$ bei Poisson.

**Quasi-Poisson bei $\hat\varphi < 1$.** Wie in R wird $\hat\varphi$ unverändert angewandt: die Intervalle werden dann schmaler als die Poisson-Intervalle. Die App zeigt $\hat\varphi < 1$ nur als Pearson-$\hat\varphi$ je Containerart (Experiment 3), nicht als Intervall und nicht als Regler.

**Literatur** (nur Gelesenes mit Stufe, Stufen: W gelesen, A Abstract, S Suchtreffer). Zeileis, Kleiber, Jackman (2008), Regression Models for Count Data in R (W). Cameron, Count Data Regression Made Simple (W).
Brooks u. a. (2017), glmmTMB (W). Ver Hoef und Boveng (2007), Quasi-Poisson gegen negativ binomial (nur S). Alle Zahlen dieser Demo sind eigene Nachrechnung in diesem Modell, bei diesen Parametern.
        """
    )

sync_query_params({k: S_[k] for k in SETTING_SPECS})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html)."
)
