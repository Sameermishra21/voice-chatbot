"""
speech_to_text.py
-----------------
Server-side Speech-to-Text.

The browser records the microphone with the Web Audio API and sends a
16-bit PCM, mono, 16 kHz WAV file.  WAV is read natively by the
SpeechRecognition library, so NO ffmpeg / audio-conversion step is needed.

Recognition engine: Google Web Speech API (via the `SpeechRecognition`
package, `recognize_google`).  It needs outbound internet from the server,
which Hugging Face Spaces / Render / your laptop all have.

Every failure is raised as SpeechError with a clear code so the API can return
a precise message to the user.  Recognition results are never invented.
"""

from __future__ import annotations

import audioop  # stdlib up to Python 3.12 (audioop-lts provides it on 3.13+)
import io
import os
import wave

import speech_recognition as sr

STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")   # e.g. "en-IN" for Indian English
MIN_DURATION_S = 0.4      # shorter than this = empty recording
MIN_RMS = 60              # quieter than this (16-bit scale) = silence


class SpeechError(Exception):
    """Raised for any speech-recognition problem; `code` is machine-readable."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def inspect_wav(wav_bytes: bytes) -> dict:
    """Validate the WAV and return duration/RMS. Raises SpeechError if unusable."""
    if not wav_bytes or len(wav_bytes) < 100:
        raise SpeechError("empty_recording", "The recording is empty. Please speak and try again.", 400)
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frames = w.readframes(w.getnframes())
            rate, width = w.getframerate(), w.getsampwidth()
            duration = w.getnframes() / float(rate) if rate else 0.0
    except (wave.Error, EOFError) as exc:
        raise SpeechError("invalid_audio", f"Could not read the recorded audio ({exc}).", 400)

    if duration < MIN_DURATION_S:
        raise SpeechError("empty_recording",
                          "The recording was too short. Hold the button a little longer while speaking.", 400)
    rms = audioop.rms(frames, width) if frames else 0
    if rms < MIN_RMS:
        raise SpeechError("silent_recording",
                          "Only silence was captured. Check that the correct microphone is selected and speak louder.", 400)
    return {"duration_s": round(duration, 2), "sample_rate": rate, "rms": rms}


def transcribe_wav(wav_bytes: bytes, language: str = STT_LANGUAGE) -> tuple[str, dict]:
    """Return (recognized_text, audio_info). Raises SpeechError on any failure."""
    info = inspect_wav(wav_bytes)
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        raise SpeechError("speech_not_recognized",
                          "Speech was captured but could not be understood. Please speak clearly and try again.", 422)
    except sr.RequestError as exc:
        raise SpeechError("stt_service_error",
                          f"The speech recognition service is unavailable right now ({exc}).", 503)

    text = (text or "").strip()
    if not text:
        raise SpeechError("speech_not_recognized", "No words were recognized in the recording.", 422)
    return text, info
