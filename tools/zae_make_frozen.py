"""Erzeugt data/zae_frozen.json: die vorgerechneten Wiederholungs-Messungen der App (Überdeckungsraster, Container-Vehikel, falsches Mittelwertmodell, Nullen).

Aufruf (aus dem Repo-Ordner, Python des Test- oder Laufzeit-venv):
    python tools/zae_make_frozen.py                 alles neu rechnen (Laufzeit je nach Rechner und Last 15 bis 40 Minuten), schreibt nach jedem Teil
    python tools/zae_make_frozen.py --only grid     nur einen Teil (grid, vehicle, missspec, nullen) neu rechnen und in die Datei einsetzen
    python tools/zae_make_frozen.py --check         Konsistenzprüfung: die ersten Wiederholungen der Hauptzelle und der Vehikel-Historien neu rechnen und mit der Datei vergleichen
                                                    (Band 1e-6; Exit-Code 1 bei Abweichung). Ändert nichts.
Jede Wiederholung r eines Experiments hat den Zufalls-Seed rep_seed(Basis, r): ein kürzerer Lauf ist ein Präfix des längeren."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zae_constants as C  # noqa: E402
import zae_generator as G  # noqa: E402
import zae_models as M  # noqa: E402
import zae_rng as R  # noqa: E402
import zae_simulation as SIM  # noqa: E402

OUT = ROOT / "data" / C.FROZEN_FILE
R_GRID, R_MAIN_NB2, R_MAIN_NB1 = 4000, 20000, 10000
R_VEH = {3000: 2000, 300: 4000}
R_NULL = 300
R_MISS, N_MISS_LIMIT = 3000, 200000
N_LIMIT = 1_000_000
PROBE = 40
CHECK_TOL = 1e-6


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def grid_cells():
    """(welt, alpha, n, R, Basis-Seed) für das Raster; NB1 mit α = 0 ist dasselbe wie NB2 mit α = 0 und fehlt."""
    cells = []
    k = 0
    for welt in C.WELT_CHOICES:
        for alpha in C.ALPHA_CHOICES:
            if welt == "NB1" and alpha == 0.0:
                continue
            for n in C.N_CHOICES:
                k += 1
                reps, seed = R_GRID, C.SEED_GRID + k
                if alpha == 0.5 and n == 200:
                    reps, seed = (R_MAIN_NB2, C.SEED_MAIN_NB2) if welt == "NB2" else (R_MAIN_NB1, C.SEED_MAIN_NB1)
                cells.append((welt, alpha, n, reps, seed))
    return cells


def part_grid():
    out = []
    for welt, alpha, n, reps, seed in grid_cells():
        t = time.time()
        cell = SIM.run_cell(welt, alpha, n, reps, seed, keep_first=PROBE if (alpha == 0.5 and n == 200 and welt == "NB2") else 0)
        out.append(cell)
        log(f"grid {welt} alpha={alpha} n={n} R={reps}: Poisson {cell['coverage']['Poisson']:.3f} Quasi {cell['coverage']['Quasi z']:.3f} NB2 {cell['coverage']['NB2']:.3f} ({time.time() - t:.0f} s)")
    return out


def part_vehicle():
    out = {"limit": {}, "cells": [], "probe": {}}
    for key, (n, seed, kunden) in {"ohne_kunden": (N_LIMIT, C.SEED_LIMIT, False), "ohne_kunden_zweiter_seed": (N_LIMIT, C.SEED_LIMIT2, False), "mit_kunden": (N_LIMIT, C.SEED_LIMIT, True)}.items():
        t = time.time()
        out["limit"][key] = SIM.vehicle_limit(n, seed, kunden)
        log(f"vehicle limit {key}: phi {out['limit'][key]['phi']:.3f} ({time.time() - t:.0f} s)")
    lim = out["limit"]["ohne_kunden"]
    lim_k = out["limit"]["mit_kunden"]
    for n in (3000, 300):
        for kunden in (False, True):
            t = time.time()
            ref = lim_k if kunden else lim
            cell = SIM.vehicle_cell(n, R_VEH[n], C.SEED_VEHICLE + n + (1 if kunden else 0), ref["beta_poisson"], ref["beta_nb2"], kunden)
            out["cells"].append(cell)
            z = cell["spalten"]["Zoll"]
            log(f"vehicle n={n} kunden={kunden} R={R_VEH[n]}: Zoll Poisson {z['Poisson']['coverage']:.3f} Quasi {z['Quasi z']['coverage']:.3f} HC0 {z['HC0']['coverage']:.3f} NB2 {z['NB2']['coverage']:.3f} ({time.time() - t:.0f} s)")
    out["probe"] = vehicle_probe()
    return out


def vehicle_probe(reps=5, n=3000):
    est = []
    for r in range(reps):
        X, d = G.make_history(n, R.rep_seed(C.SEED_VEHICLE + n, r))
        res = M.all_methods(G.design(X, False), G.standtage(d), [C.COL_ZOLL])
        est.append([float(res["est"][0, 0]), float(res["est"][0, 6])])
    return {"n": n, "base": C.SEED_VEHICLE + n, "kunden": False, "est_zoll_poisson_nb2": est}


def part_nullen():
    out = {"limit": SIM.zeros_limit(N_LIMIT, C.SEED_LIMIT), "ohne_kunden": SIM.vehicle_zeros(3000, R_NULL, C.SEED_NULL, False),
           "mit_kunden": SIM.vehicle_zeros(3000, R_NULL, C.SEED_NULL, True)}
    log(f"nullen: beobachtet {out['ohne_kunden']['beobachtet']:.3f} Poisson {out['ohne_kunden']['poisson_erwartet']:.3f} NB2 {out['ohne_kunden']['nb2_erwartet']:.3f}")
    return out


def part_missspec():
    out = {"limit": [], "n200": []}
    for k, c in enumerate(C.C_CHOICES):
        out["limit"].append(SIM.missspec_limit(c, N_MISS_LIMIT, C.SEED_MISSPEC_LIMIT + k))
        t = time.time()
        out["n200"].append(SIM.run_missspec(c, 200, R_MISS, C.SEED_MISSPEC + k))
        log(f"missspec c={c}: Grenzwert Poisson {out['limit'][-1]['poisson']:.3f} NB2 {out['limit'][-1]['nb2']:.3f}; n=200 Median {out['n200'][-1]['diff_median']:+.3f} ({time.time() - t:.0f} s)")
    return out


PARTS = {"grid": part_grid, "vehicle": part_vehicle, "missspec": part_missspec, "nullen": part_nullen}


def meta():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "plattform": platform.system(),
            "hinweis": "erzeugt von tools/zae_make_frozen.py; Zufalls-Seeds und Zahl der Wiederholungen stehen je Eintrag"}


def run(only):
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in PARTS:
        if only and name != only:
            continue
        t = time.time()
        data[name] = PARTS[name]()
        data["meta"] = meta()
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        log(f"Teil {name} geschrieben ({time.time() - t:.0f} s)")


def check() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    bad = 0
    main = next(c for c in data["grid"] if c["welt"] == "NB2" and c["alpha"] == 0.5 and c["n"] == 200)
    pr = main["probe"]
    for r in range(pr["R"]):
        e, _, _, _, _, _ = SIM.replicate_world(R.rep_seed(main["seed"], r), main["n"], "NB2", 0.5)
        for got, want, nm in ((e[0], pr["est_poisson"][r], "Poisson"), (e[6], pr["est_nb2"][r], "NB2")):
            if abs(got - want) > CHECK_TOL * max(1.0, abs(want)):
                bad += 1
                print(f"Abweichung Hauptzelle Wiederholung {r} {nm}: neu {got:.9f}, Datei {want:.9f}")
    vp = data["vehicle"]["probe"]
    for r, (p_want, n_want) in enumerate(vp["est_zoll_poisson_nb2"]):
        X, d = G.make_history(vp["n"], R.rep_seed(vp["base"], r))
        res = M.all_methods(G.design(X, vp["kunden"]), G.standtage(d), [C.COL_ZOLL])
        for got, want, nm in ((res["est"][0, 0], p_want, "Poisson"), (res["est"][0, 6], n_want, "NB2")):
            if abs(got - want) > CHECK_TOL * max(1.0, abs(want)):
                bad += 1
                print(f"Abweichung Vehikel Historie {r} {nm}: neu {got:.9f}, Datei {want:.9f}")
    print(f"zae_make_frozen --check: {bad} Abweichungen (Band {CHECK_TOL:g}); {pr['R']} Wiederholungen der Hauptzelle, {len(vp['est_zoll_poisson_nb2'])} Historien des Vehikels")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(PARTS))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(check())
    run(args.only)
