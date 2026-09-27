"""
tests/test_app.py  –  automated tests (pytest)

Run from the project root:
    pytest -q

* Model tests use the real trained model.
* API tests send real WAV bytes to /process-audio.  Only the network call to
  the Google speech service is replaced (monkeypatched) so the tests run
  offline and deterministically; everything else — WAV validation, silence
  detection, error handling, the BiLSTM model — runs for real.
"""

import io
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

import speech_recognition as sr  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402
import test_cases  # noqa: E402


def make_wav(seconds=1.5, freq=220.0, amplitude=0.3, rate=16000) -> bytes:
    """Create a 16-bit mono WAV in memory (a tone stands in for a voice)."""
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(n)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # runs the lifespan -> loads the model once
        yield c


def fake_google(text=None, exc=None):
    def _recognize(self, audio, language="en-US", **kw):
        assert isinstance(audio, sr.AudioData) and len(audio.frame_data) > 0
        if exc:
            raise exc
        return text
    return _recognize


# ----------------------------- health / text ---------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["model_loaded"] is True and len(j["intents"]) >= 10


def test_homepage_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Start Speaking" in r.text
    assert client.get("/static/script.js").status_code == 200


@pytest.mark.parametrize("text,expected", test_cases.TEST_CASES)
def test_demo_cases(client, text, expected):
    r = client.post("/predict-text", json={"text": text})
    assert r.status_code == 200
    assert r.json()["intent"] == expected


def test_gibberish_is_low_confidence(client):
    j = client.post("/predict-text", json={"text": "zxqv blorp flarn"}).json()
    assert j["low_confidence"] is True
    assert j["response"].startswith("I'm not sure I understood that")


# ----------------------------- audio pipeline --------------------------------
def test_process_audio_success(client, monkeypatch):
    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake_google("what courses do you offer"))
    r = client.post("/process-audio", files={"audio": ("speech.wav", make_wav(), "audio/wav")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["recognized_text"] == "what courses do you offer"
    assert j["intent"] == "courses"
    assert 0.0 <= j["confidence"] <= 1.0
    assert set(["recognized_text", "intent", "confidence", "response"]) <= set(j)


def test_empty_recording(client):
    r = client.post("/process-audio", files={"audio": ("speech.wav", b"", "audio/wav")})
    assert r.status_code == 400 and r.json()["error"] == "empty_recording"


def test_too_short_recording(client):
    r = client.post("/process-audio", files={"audio": ("speech.wav", make_wav(seconds=0.1), "audio/wav")})
    assert r.status_code == 400 and r.json()["error"] == "empty_recording"


def test_silent_recording(client):
    r = client.post("/process-audio", files={"audio": ("speech.wav", make_wav(amplitude=0.0), "audio/wav")})
    assert r.status_code == 400 and r.json()["error"] == "silent_recording"


def test_invalid_audio(client):
    r = client.post("/process-audio", files={"audio": ("speech.wav", b"not a wav file" * 20, "audio/wav")})
    assert r.status_code == 400 and r.json()["error"] == "invalid_audio"


def test_speech_not_recognized(client, monkeypatch):
    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake_google(exc=sr.UnknownValueError()))
    r = client.post("/process-audio", files={"audio": ("speech.wav", make_wav(), "audio/wav")})
    assert r.status_code == 422 and r.json()["error"] == "speech_not_recognized"


def test_stt_service_failure(client, monkeypatch):
    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake_google(exc=sr.RequestError("network down")))
    r = client.post("/process-audio", files={"audio": ("speech.wav", make_wav(), "audio/wav")})
    assert r.status_code == 503 and r.json()["error"] == "stt_service_error"
