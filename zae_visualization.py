"""Plotly-Figuren (alle Achsen fest, `fixedrange`: Touch-Scrollen bleibt nicht am Diagramm hängen)."""

import math

import numpy as np
import plotly.graph_objects as go

import zae_constants as C
import zae_models as M

LABEL = {"Poisson": "Poisson", "Quasi z": "Quasi (z)", "Quasi t": "Quasi (t)", "HC0": "HC0", "HC1": "HC1", "HC3a": "HC3-Analogon", "NB2": "NB2"}


def _fix(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_hand_intervals(res: dict):
    """95-%-Intervalle für β̂₁ (Zollhalt-Effekt) bei fünf Verfahren, dazu die Linie bei 0."""
    fig = go.Figure()
    keys = [k for k in ("naive", "quasi", "hc0", "hc1", "nb2")]
    names = {"naive": "Poisson (z)", "quasi": "Quasi-Poisson (t)", "hc0": "Sandwich HC0 (z)", "hc1": "Sandwich HC1 (z)", "nb2": "NB2 (z)"}
    b1 = res["beta"][1]
    for k in keys:
        lo, hi = res["iv"][k]
        fig.add_trace(go.Scatter(x=[lo, hi], y=[names[k]] * 2, mode="lines+markers", line=dict(width=6, color=C.COLORS[{"naive": "Poisson", "quasi": "Quasi t", "hc0": "HC0", "hc1": "HC1", "nb2": "NB2"}[k]]), showlegend=False))
    fig.add_trace(go.Scatter(x=[b1] * len(keys), y=[names[k] for k in keys], mode="markers", marker=dict(symbol="diamond", size=11, color="black"), name="β̂₁ (gleich bei allen Poisson-Verfahren)"))
    fig.add_vline(x=0, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="0 = kein Zollhalt-Effekt")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Koeffizient des Zollhalts (log-Skala)")
    return _fix(fig, 300, title="95-%-Intervalle für den Zollhalt-Effekt (acht Werte)")


def build_coverage(cell: dict, limit: dict | None, live: dict | None = None, title: str = "Überdeckung des 95-%-Intervalls"):
    """Überdeckung je Verfahren (vorgerechnet) mit ±2 Monte-Carlo-Standardfehlern; Raute: Grenzwert aus der Sandwich-Formel (Poisson, Quasi z); Punkte: Live-Probe."""
    ms = list(M.METHODS)
    cov = [100 * cell["coverage"][m] for m in ms]
    err = [200 * cell["se_mc"]] * len(ms)
    fig = go.Figure(go.Bar(x=[LABEL[m] for m in ms], y=cov, marker_color=[C.COLORS[m] for m in ms], error_y=dict(type="data", array=err, visible=True), name=f"vorgerechnet ({cell['R']} Wdh.)",
                           text=[f"{v:.1f}".replace(".", ",") for v in cov], textposition="outside", cliponaxis=False))
    if limit is not None:
        fig.add_trace(go.Scatter(x=[LABEL["Poisson"], LABEL["Quasi z"]], y=[100 * limit["cov_poisson"], 100 * limit["cov_quasi"]], mode="markers",
                                 marker=dict(symbol="diamond", size=13, color=C.COLOR_ORACLE, line=dict(width=1, color="white")), name="Grenzwert n → ∞ (Formel)"))
    if live is not None:
        fig.add_trace(go.Scatter(x=[LABEL[m] for m in ms], y=[100 * live["coverage"][m] for m in ms], mode="markers", marker=dict(symbol="circle-open", size=11, color="black", line=dict(width=2)),
                                 name=f"Live-Probe ({live['R']} Wdh.)"))
    fig.add_hline(y=95, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="95 %")
    fig.update_yaxes(range=[40, 105], title="Überdeckung in %")
    return _fix(fig, 380, title=title)


def build_single(tab: dict, truth: float = C.BETA1):
    fig = go.Figure()
    ms = [m for m in M.METHODS if m in tab]
    for m in ms:
        t = tab[m]
        fig.add_trace(go.Scatter(x=[t["lo"], t["hi"]], y=[LABEL[m]] * 2, mode="lines+markers", line=dict(width=6, color=C.COLORS[m]), showlegend=False))
    fig.add_trace(go.Scatter(x=[tab[m]["est"] for m in ms], y=[LABEL[m] for m in ms], mode="markers", marker=dict(symbol="diamond", size=10, color="black"), name="Schätzung"))
    fig.add_vline(x=truth, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text=f"wahr {str(truth).replace('.', ',')}")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Anstieg β₁")
    return _fix(fig, 330, title="Ein Datensatz: sieben Intervalle")


def build_vehicle(cell: dict, spalte: str):
    ms = ["Poisson", "Quasi z", "HC0", "HC3a", "NB2"]
    sp = cell["spalten"][spalte]
    rauschen = spalte.startswith("Gewicht")
    vals = [100 * sp[m]["fehlalarm" if rauschen else "coverage"] for m in ms]
    fig = go.Figure(go.Bar(x=[LABEL[m] for m in ms], y=vals, marker_color=[C.COLORS[m] for m in ms], error_y=dict(type="data", array=[200 * cell["se_mc"] * (math.sqrt(5 / 95) if rauschen else 1)] * len(ms), visible=True),
                           text=[f"{v:.1f}".replace(".", ",") for v in vals], textposition="outside", cliponaxis=False))
    if rauschen:
        fig.add_hline(y=5, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="nominal 5 %")
        fig.update_yaxes(range=[0, 45], title="Fehlalarm in % (|z| > 1,96, wahre Wirkung 0)")
    else:
        fig.add_hline(y=95, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="95 %")
        fig.update_yaxes(range=[40, 105], title="Überdeckung des Grenzwerts in %")
    return _fix(fig, 360, title=f"Container-Vehikel, n = {cell['n']}, {'mit' if cell['kunden'] else 'ohne'} Kunden: {spalte}")


def build_phi_art(limit: dict):
    pa = limit["phi_je_art"]
    fig = go.Figure(go.Bar(x=list(C.ART_LABELS), y=pa, marker_color=["#c0392b" if v > 1 else "#4c78a8" for v in pa], text=[f"{v:.2f}".replace(".", ",") for v in pa], textposition="outside", cliponaxis=False))
    fig.add_hline(y=1, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="Poisson: 1")
    fig.update_yaxes(title="Pearson-φ̂ je Containerart", range=[0, max(pa) * 1.2])
    return _fix(fig, 300, title="Streuung gegenüber Poisson je Containerart (Standtage)")


def build_weights(alpha: float):
    mu = np.linspace(0.5, 25, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mu, y=mu, name="Poisson und Quasi-Poisson: μ", line=dict(color=C.COLORS["Poisson"], width=3)))
    if alpha > 0:
        fig.add_trace(go.Scatter(x=mu, y=mu / (1 + alpha * mu), name=f"NB2: μ/(1 + αμ), α = {str(alpha).replace('.', ',')}", line=dict(color=C.COLORS["NB2"], width=3)))
    fig.update_xaxes(title="Mittel μ eines Falls")
    fig.update_yaxes(title="Gewicht im IRLS")
    return _fix(fig, 320, title="Wie die Verfahren Zählwerte gewichten")


def build_missspec(frozen: dict, c_marker: float):
    lim = frozen["missspec"]["limit"]
    cs = [e["c"] for e in lim]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs, y=[e["poisson"] for e in lim], mode="lines+markers", name="Poisson (= Quasi-Poisson)", line=dict(color=C.COLORS["Poisson"], width=3)))
    fig.add_trace(go.Scatter(x=cs, y=[e["nb2"] for e in lim], mode="lines+markers", name="NB2", line=dict(color=C.COLORS["NB2"], width=3)))
    fig.add_hline(y=0.4, line_dash="dash", line_color=C.COLOR_TARGET, annotation_text="wahrer Anstieg bei x = 0: 0,4")
    fig.add_vline(x=c_marker, line_dash="dot", line_color=C.COLOR_ORACLE)
    fig.update_xaxes(title="Krümmung c der Wahrheit (log μ = 1 + 0,4 x + c x²)", tickvals=cs)
    fig.update_yaxes(title=f"Anstieg β̂₁ bei n = {lim[0]['n']:,}".replace(",", "."))
    return _fix(fig, 340, title="Falsches Mittelwertmodell: zwei Schätzer, zwei Größen")


def build_zeros(frozen: dict, kunden: bool):
    lim = frozen["nullen"]["limit"]["je_art"]
    arts = list(C.ART_LABELS)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=arts, y=[lim[a]["beobachtet"] for a in arts], name="beobachtet", marker_color="#555555"))
    fig.add_trace(go.Bar(x=arts, y=[lim[a]["poisson"] for a in arts], name="Poisson erwartet", marker_color=C.COLORS["Poisson"]))
    fig.add_trace(go.Bar(x=arts, y=[lim[a]["nb2"] for a in arts], name="NB2 erwartet", marker_color=C.COLORS["NB2"]))
    fig.update_yaxes(title="Anteil Nullen (Standgeld-Tage)", range=[0, 1.05])
    return _fix(fig, 340, barmode="group", title="Nullanteil je Containerart: beobachtet gegen erwartet")
