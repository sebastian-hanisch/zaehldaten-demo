"""Zufallsgenerator: dieselbe Folge in Klasse und Feld, Normalzahlen ohne C-Bibliothek gleich der exakten Fassung, Momente."""

import numpy as np

import zae_rng as R


def test_array_stream_equals_class_stream():
    g = R.SplitMix64(12345)
    assert [g.next_u64() for _ in range(50)] == [int(v) for v in R.splitmix_array(12345, 50)]
    g = R.SplitMix64(7)
    assert [g.random() for _ in range(50)] == list(R.uniform_array(7, 50))


def test_known_splitmix_values_for_seed_zero():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF
    assert g.next_u64() == 0x6E789E6AA1B965F4


def test_log_and_cos_series_match_the_exact_decimal_versions():
    u = R.uniform_array(3, 400)
    u = np.maximum(u, 1e-12)
    ex = np.array([R.exact_log(float(v)) for v in u])
    assert np.max(np.abs(R.log_array(u) - ex) / np.abs(ex).clip(1e-3)) < 1e-14
    ex = np.array([R.exact_cos(2.0 * np.pi * float(v)) for v in u])
    assert np.max(np.abs(R.cos_2pi_array(u) - ex)) < 2e-15


def test_log_array_on_special_arguments():
    for x, want in ((1.0, 0.0), (0.5, -0.6931471805599453), (2.0, 0.6931471805599453), (1e-12, -27.631021115928547)):
        assert abs(float(R.log_array(np.array([x]))[0]) - want) < 1e-14 * max(1.0, abs(want))


def test_cos_array_on_special_arguments():
    for u, want in ((0.0, 1.0), (0.25, 0.0), (0.5, -1.0), (0.75, 0.0), (1.0 / 6.0, 0.5), (1.0 / 8.0, 2 ** -0.5)):
        assert abs(float(R.cos_2pi_array(np.array([u]))[0]) - want) < 2e-15


def test_normal_array_matches_the_exact_class_value_by_value():
    g = R.SplitMix64(10)
    ex = np.array([g.normal() for _ in range(300)])
    assert np.max(np.abs(R.normal_array(10, 300) - ex)) < 1e-13


def test_normal_array_moments():
    z = R.normal_array(5, 200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1.0) < 0.01          # 200 000 Werte: Standardfehler des Mittels 0,0022
    assert abs(np.mean(z ** 3)) < 0.03 and abs(np.mean(z ** 4) - 3.0) < 0.1


def test_normal_array_is_a_prefix_stream():
    """Die ersten n Werte sind unabhängig von der Länge der angeforderten Folge (Lauf 0 bleibt bei mehr Läufen derselbe)."""
    assert np.array_equal(R.normal_array(10, 30), R.normal_array(10, 600)[:30])


def test_rep_seed_is_deterministic_and_distinct():
    seeds = [R.rep_seed(8200, r) for r in range(1000)]
    assert seeds == [R.rep_seed(8200, r) for r in range(1000)] and len(set(seeds)) == 1000
    assert R.rep_seed(8200, 0) != R.rep_seed(8201, 0) and all(0 <= s < 2 ** 64 for s in seeds)


def test_rep_seed_streams_are_not_shifted_copies():
    a, b = R.uniform_array(R.rep_seed(1, 0), 50), R.uniform_array(R.rep_seed(1, 1), 50)
    assert not np.array_equal(a[1:], b[:-1]) and abs(np.corrcoef(a, b)[0, 1]) < 0.4
