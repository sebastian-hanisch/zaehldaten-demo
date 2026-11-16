"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b): acht erfundene Standtage in zwei Gruppen."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

A_WERTE = (0, 0, 1, 5)      # ohne Zollhalt: Summe 6, Mittel 3/2
B_WERTE = (1, 3, 6, 10)     # mit Zollhalt: Summe 20, Mittel 5


@pytest.fixture
def mini():
    """Entwurfsmatrix (Achsenabschnitt, Zollhalt-Indikator) und Standtage der acht Container. Handrechnung: die Gruppenmittel 3/2 und 5 sind die Poisson-Schätzung,
    β̂ = (ln 3/2, ln 10/3); Var(β̂₁) = 1/6 + 1/20 = 13/60."""
    X = np.array([[1.0, 0.0]] * 4 + [[1.0, 1.0]] * 4)
    y = np.array(A_WERTE + B_WERTE, float)
    return X, y
