"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

import zae_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_hand_group_without_event_shows_a_message_instead_of_numbers():
    at = _fresh_app()
    for i in range(1, 5):
        at.number_input(key=f"h{i}").set_value(0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("kein einziges Ereignis" in w.value for w in at.warning)
    assert not any("Poisson: z =" in w.value for w in list(at.warning) + list(at.info))


def test_hand_values_without_residuals_do_not_crash():
    at = _fresh_app()
    for i, v in enumerate((3, 3, 3, 3, 6, 6, 6, 6), start=1):
        at.number_input(key=f"h{i}").set_value(v)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_live_probe_runs_and_adds_its_verdict():
    at = _fresh_app()
    at.slider(key="live_r").set_value(50)
    at.button(key="live_go").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Live-Probe mit 50 Wiederholungen" in i.value for i in at.info)


def test_exposure_block_with_zero_standtage_shows_a_message():
    at = _fresh_app()
    at.number_input(key="e1").set_value(0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("müssen positiv sein" in w.value for w in at.warning)


def test_every_choice_of_every_selection_runs():
    for key, choices in (("welt", C.WELT_CHOICES), ("alpha", C.ALPHA_CHOICES), ("n", C.N_CHOICES), ("c", C.C_CHOICES), ("vn", C.VN_CHOICES), ("spalte", tuple(C.VEHICLE_COLUMNS)), ("ausgang", C.AUSGANG_CHOICES)):
        for choice in choices:
            at = _fresh_app()
            for art in ("selectbox", "select_slider", "radio"):
                try:
                    getattr(at, art)(key=key).set_value(choice)
                    break
                except KeyError:
                    continue
            at.run(timeout=TIMEOUT)
            assert not at.exception, (key, choice, [str(e) for e in at.exception])


def test_kunden_toggle_and_permalink_run():
    at = _fresh_app()
    at.toggle(key="kunden").set_value(True)
    at.run(timeout=TIMEOUT)
    assert not at.exception and any("mit 11 Kundenspalten" in w.value for w in list(at.warning) + list(at.info))
    at = AppTest.from_file(APP_PATH)
    at.query_params["alpha"] = "1.0"
    at.query_params["n"] = "30"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="alpha").value == 1.0 and at.select_slider(key="n").value == 30
