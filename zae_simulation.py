"""Wiederholungs-Messungen: Überdeckung der Intervalle, Streuung der Schätzer, Standtage der Container, falsches Mittelwertmodell, Nullen.

Zähleinheit überall: **eine Wiederholung = ein Datensatz** (bzw. eine Historie aus n Containern). Wiederholung r eines Experiments hat den Zufalls-Seed `rep_seed(Basis, r)`:
die ersten R Wiederholungen sind also unabhängig von R (ein kürzerer Lauf ist ein Präfix des längeren; darauf stützt sich die Live-Probe und die Prüfung der vorgerechneten Datei).
Die schweren Reihen rechnet `tools/zae_make_frozen.py` einmal und friert sie in `data/zae_frozen.json` ein; die App rechnet nur kleine Stichproben live."""

import math

import numpy as np

import zae_constants as C
import zae_generator as G
import zae_models as M
import zae_rng as R
import zae_sample as S

METHODS = M.METHODS


def _finite_rows(est, se):
    return np.isfinite(est) & np.isfinite(se)


def summarize_cover(est, se, crit, truth):
    """Aus Feldern est, se (R × 7) und crit (7) die Überdeckung je Verfahren über die Wiederholungen, in denen das Verfahren eine endliche Antwort gab.
    truth: Skalar oder Feld (7,) (Ziel je Verfahren: bei NB2 der Grenzwert des NB2-Schätzers)."""
    ok = _finite_rows(est, se)
    hit = np.abs(est - truth) < crit * se
    n_valid = ok.sum(axis=0)
    cov = np.where(n_valid > 0, (hit & ok).sum(axis=0) / np.maximum(n_valid, 1), np.nan)
    return cov, n_valid


def replicate_world(seed, n, welt, alpha, c=0.0):
    """Eine Wiederholung der Messwelt 1: est, se (je 7), crit (7), phi, alpha_hat, alpha_am_rand. Scheitert die Poisson-Anpassung, sind alle Werte NaN."""
    X, y, _ = S.draw_world(seed, n, welt, alpha, c)
    try:
        r = M.all_methods(X, y, [1])
    except M.FitError:
        nan = np.full(7, np.nan)
        return nan, nan, np.full(7, M.Z975), float("nan"), float("nan"), False
    return r["est"][0], r["se"][0], r["crit"], r["phi"], r["alpha"], r["alpha_am_rand"]


def run_cell(welt, alpha, n, R_reps, base_seed, c=0.0, truth=C.BETA1, keep_first=0):
    """R_reps Wiederholungen der Messwelt 1; Ziel der Intervalle ist der wahre Anstieg `truth` (c = 0: das Mittelwertmodell stimmt). Gibt ein JSON-fähiges Wörterbuch zurück."""
    est = np.full((R_reps, 7), np.nan)
    se = np.full((R_reps, 7), np.nan)
    phis = np.full(R_reps, np.nan)
    alphas = np.full(R_reps, np.nan)
    rand = 0
    crit = None
    for r in range(R_reps):
        e, s, crit, phi, a, am_rand = replicate_world(R.rep_seed(base_seed, r), n, welt, alpha, c)
        est[r], se[r], phis[r], alphas[r] = e, s, phi, a
        rand += int(am_rand)
    cov, n_valid = summarize_cover(est, se, crit, truth)
    width = np.nanmean(2 * crit * se, axis=0)
    se_mean = np.nanmean(se, axis=0)
    ep, en = est[:, 0], est[:, 6]
    bias_p, bias_n = float(np.nanmean(ep) - truth), float(np.nanmean(en) - truth)
    rmse_p = math.sqrt(float(np.nanmean((ep - truth) ** 2)))
    rmse_n = math.sqrt(float(np.nanmean((en - truth) ** 2)))
    out = {
        "welt": welt, "alpha": alpha, "n": n, "R": R_reps, "seed": base_seed, "c": c,
        "coverage": {m: float(cov[i]) for i, m in enumerate(METHODS)},
        "n_valid": {m: int(n_valid[i]) for i, m in enumerate(METHODS)},
        "se_mc": math.sqrt(0.95 * 0.05 / R_reps),
        "width": {m: float(width[i]) for i, m in enumerate(METHODS)},
        "se_mean": {m: float(se_mean[i]) for i, m in enumerate(METHODS)},
        "sd_est": {"Poisson": float(np.nanstd(ep)), "NB2": float(np.nanstd(en))},
        "bias": {"Poisson": bias_p, "NB2": bias_n},
        "rmse_ratio": rmse_p / rmse_n if rmse_n > 0 else float("nan"),
        "phi_mean": float(np.nanmean(phis)),
        "alpha_median": float(np.nanmedian(alphas)),
        "nb2_randloesung": int(rand),
        "nb2_ausfaelle": int(R_reps - n_valid[6]),
        "poisson_ausfaelle": int(R_reps - n_valid[0]),
    }
    if keep_first:
        out["probe"] = {"R": keep_first, "est_poisson": [float(v) for v in est[:keep_first, 0]], "est_nb2": [float(v) for v in est[:keep_first, 6]]}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# falsches Mittelwertmodell
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def misspec_slope_pair(seed, n, c, alpha=0.5):
    """Anstieg β̂₁ von Poisson und NB2 auf einem Datensatz mit Krümmung c (NB2-Zählung, α = 0,5); angepasst wird immer 1 + x."""
    X, y, _ = S.draw_world(seed, n, "NB2", alpha, c)
    bp = M.irls(X, y).beta[1]
    bn = M.nb2_fit(X, y).beta[1]
    return float(bp), float(bn)


def run_missspec(c, n, R_reps, base_seed):
    """Wiederholungsverteilung des Unterschieds β̂₁(NB2) − β̂₁(Poisson) bei falschem Mittelwertmodell. Gibt Median und Quartile an, nicht Mittel und SD (einzelne
    Ausreißer-Datensätze dominieren sie)."""
    bp, bn, fail = [], [], 0
    for r in range(R_reps):
        try:
            a, b = misspec_slope_pair(R.rep_seed(base_seed, r), n, c)
        except M.FitError:
            fail += 1
            continue
        if np.isfinite(a) and np.isfinite(b):
            bp.append(a)
            bn.append(b)
        else:
            fail += 1
    bp, bn = np.array(bp), np.array(bn)
    d = bn - bp
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    iqr_p = np.subtract(*np.percentile(bp, [75, 25]))
    return {
        "c": c, "n": n, "R": R_reps, "seed": base_seed, "ausfaelle": fail,
        "diff_median": float(q2), "diff_q1": float(q1), "diff_q3": float(q3),
        "poisson_median": float(np.median(bp)), "nb2_median": float(np.median(bn)),
        "poisson_robust_sd": float(iqr_p / 1.349),
        "diff_gross": int(np.sum(np.abs(d) > 1.0)),
        "anteil_diff_ueber_sd": float(np.mean(np.abs(d) > iqr_p / 1.349)),
    }


def missspec_limit(c, n, seed, alpha=0.5):
    """Ein großer Datensatz: Grenzwert (Pseudo-Wahrheit) des Anstiegs bei Poisson und NB2 (Zufalls-Seed fest)."""
    X, y, _ = S.draw_world(seed, n, "NB2", alpha, c)
    bp = float(M.irls(X, y).beta[1])
    nb = M.nb2_fit(X, y)
    return {"c": c, "n": n, "seed": seed, "poisson": bp, "nb2": float(nb.beta[1]), "alpha": float(nb.alpha)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Container-Vehikel
# ---------------------------------------------------------------------------------------------------------------------------------------------------

def vehicle_limit(n, seed, kunden=False):
    """Grenzwert der Schätzer im Vehikel aus einer großen Historie (ein Datensatz): Pseudo-wahre Koeffizienten von Poisson und NB2, Pearson-φ̂ gesamt und je Art,
    Mittel und Varianz der Standtage. Das Mittelwertmodell ist nicht log-linear: es gibt keinen wahren Koeffizienten, nur den Grenzwert des jeweiligen Schätzers."""
    X, d = G.make_history(n, seed)
    y = G.standtage(d)
    F = G.design(X, kunden)
    inf = M.poisson_inference(F, y)
    nb = M.nb2_fit(F, y)
    art = X[:, 0].astype(int)
    mu = inf.fit.mu
    phi_art = [float(np.sum((y[art == a] - mu[art == a]) ** 2 / mu[art == a]) / np.sum(art == a)) for a in range(4)]
    return {
        "n": n, "seed": seed, "kunden": kunden, "mean": float(y.mean()), "var": float(y.var()), "min": float(y.min()),
        "phi": inf.phi, "alpha_nb2": nb.alpha,
        "beta_poisson": [float(v) for v in inf.fit.beta[:8]], "beta_nb2": [float(v) for v in nb.beta[:8]],
        "phi_je_art": phi_art,
    }


def vehicle_cell(n, R_reps, base_seed, truth_p, truth_n, kunden=False):
    """R_reps Historien aus je n Containern; je Spalte (Zoll, Wochenende, Gewicht) und Verfahren: Überdeckung des Grenzwerts des jeweiligen Schätzers (truth_p für alle
    Poisson-Verfahren, truth_n für NB2), SD der Schätzung, mittlerer Standardfehler; für das Rauschmerkmal zusätzlich der Fehlalarm (|z| > Quantil, Ziel 0)."""
    cols = list(C.VEHICLE_COLUMNS.values())
    est = np.full((R_reps, len(cols), 7), np.nan)
    se = np.full((R_reps, len(cols), 7), np.nan)
    phis, alphas = np.full(R_reps, np.nan), np.full(R_reps, np.nan)
    crit = None
    for r in range(R_reps):
        X, d = G.make_history(n, R.rep_seed(base_seed, r))
        try:
            res = M.all_methods(G.design(X, kunden), G.standtage(d), cols)
        except M.FitError:
            continue
        est[r], se[r], crit, phis[r], alphas[r] = res["est"], res["se"], res["crit"], res["phi"], res["alpha"]
    out = {"n": n, "R": R_reps, "seed": base_seed, "kunden": kunden, "se_mc": math.sqrt(0.95 * 0.05 / R_reps),
           "phi_mean": float(np.nanmean(phis)), "phi_sd": float(np.nanstd(phis)), "alpha_median": float(np.nanmedian(alphas)), "spalten": {}}
    for k, (name, col) in enumerate(C.VEHICLE_COLUMNS.items()):
        tr = np.array([truth_p[col]] * 6 + [truth_n[col]])
        cov, n_valid = summarize_cover(est[:, k, :], se[:, k, :], crit, tr)
        e, s = est[:, k, :], se[:, k, :]
        zero_hit = np.abs(e - 0.0) > crit * s
        ok = _finite_rows(e, s)
        alarm = np.where(ok.sum(axis=0) > 0, (zero_hit & ok).sum(axis=0) / np.maximum(ok.sum(axis=0), 1), np.nan)
        sd = np.nanstd(e, axis=0)
        sem = np.nanmean(s, axis=0)
        out["spalten"][name] = {
            m: {"coverage": float(cov[i]), "fehlalarm": float(alarm[i]), "sd": float(sd[i]), "se_mean": float(sem[i]), "ratio": float(sem[i] / sd[i]), "n_valid": int(n_valid[i])}
            for i, m in enumerate(METHODS)
        }
    return out


def vehicle_zeros(n, R_reps, base_seed, kunden=False):
    """Nullen bei den Standgeld-Tagen (Standgeld ab dem 8. Tag, Annahme dieser Messung): beobachteter Nullanteil gegen den, den Poisson und NB2 erwarten."""
    obs, pz, nz, phis, alphas = [], [], [], [], []
    for r in range(R_reps):
        X, d = G.make_history(n, R.rep_seed(base_seed, r))
        y = G.standgeld_tage(d)
        F = G.design(X, kunden)
        try:
            inf = M.poisson_inference(F, y)
            nb = M.nb2_fit(F, y)
        except M.FitError:
            continue
        obs.append(np.mean(y == 0))
        pz.append(np.mean(np.exp(-inf.fit.mu)))
        nz.append(np.mean((1 + nb.alpha * nb.mu) ** (-1 / nb.alpha)))
        phis.append(inf.phi)
        alphas.append(nb.alpha)
    return {"n": n, "R": R_reps, "seed": base_seed, "kunden": kunden, "beobachtet": float(np.mean(obs)), "poisson_erwartet": float(np.mean(pz)),
            "nb2_erwartet": float(np.mean(nz)), "phi": float(np.mean(phis)), "alpha_median": float(np.median(alphas))}


def zeros_limit(n, seed):
    """Nullanteile des Standgelds im Grenzwert (eine große Historie), gesamt und je Containerart: beobachtet, Poisson erwartet, NB2 erwartet."""
    X, d = G.make_history(n, seed)
    y = G.standgeld_tage(d)
    F = G.design(X)
    inf = M.poisson_inference(F, y)
    nb = M.nb2_fit(F, y)
    art = X[:, 0].astype(int)
    pz = np.exp(-inf.fit.mu)
    nz = (1 + nb.alpha * nb.mu) ** (-1 / nb.alpha)
    je_art = {C.ART_LABELS[a]: {"beobachtet": float(np.mean(y[art == a] == 0)), "poisson": float(np.mean(pz[art == a])), "nb2": float(np.mean(nz[art == a]))} for a in range(4)}
    return {"n": n, "seed": seed, "beobachtet": float(np.mean(y == 0)), "poisson_erwartet": float(np.mean(pz)), "nb2_erwartet": float(np.mean(nz)), "phi": inf.phi,
            "alpha": nb.alpha, "je_art": je_art}
