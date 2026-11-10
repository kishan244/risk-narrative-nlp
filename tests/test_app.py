from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_every_tab_without_errors():
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Risk Radar"
    assert len(app.tabs) == 6
    assert len(app.metric) >= 4


def test_changing_company_and_theme_works():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox[0].select("Tesla").run()
    assert not app.exception
    app.text_input[0].input("semiconductor").run()
    assert not app.exception
