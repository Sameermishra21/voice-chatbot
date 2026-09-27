"""Smoke test: the Streamlit app starts, loads the model and shows the microphone widget."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_streamlit_app_loads():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("Model loaded" in c.value for c in at.caption)
    assert "Voice-Enabled AI Chatbot" in at.markdown[0].value
