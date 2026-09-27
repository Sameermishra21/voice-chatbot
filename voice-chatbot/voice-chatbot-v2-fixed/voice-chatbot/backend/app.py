"""
app.py
------
FastAPI backend for the Voice-Enabled Chatbot.

Routes
  GET  /                -> the chatbot web page (frontend/index.html)
  GET  /static/*        -> frontend CSS / JS
  GET  /health          -> status check used by the frontend on page load
  POST /process-audio   -> microphone WAV -> Speech-to-Text -> BiLSTM intent -> response
  POST /predict-text    -> text -> BiLSTM intent -> response   (for testing / viva demo only)

The trained model is loaded ONCE when the server starts; it is never retrained here.

Run locally (from the project root):
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
sys.path.insert(0, str(BACKEND_DIR))  # allow `import nlp_utils` however the app is launched

from nlp_utils import IntentClassifier  # noqa: E402
import speech_to_text as stt           # noqa: E402

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
MAX_AUDIO_BYTES = 5 * 1024 * 1024       # 5 MB  (~2.5 min of 16 kHz mono WAV)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voicebot")

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Loading trained intent model ...")
    STATE["classifier"] = IntentClassifier(BACKEND_DIR, threshold=CONFIDENCE_THRESHOLD)
    log.info("Model loaded: %d intents, vocab %d",
             len(STATE["classifier"].labels), STATE["classifier"].tokenizer.vocab_size)
    yield
    STATE.clear()


app = FastAPI(title="Voice-Enabled Chatbot", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})


# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    clf = STATE.get("classifier")
    return {
        "status": "ok" if clf else "loading",
        "model_loaded": clf is not None,
        "intents": clf.labels if clf else [],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stt_engine": "Google Web Speech API (SpeechRecognition)",
        "stt_language": stt.STT_LANGUAGE,
    }


@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(..., description="16-bit PCM WAV from the browser microphone")):
    t0 = time.perf_counter()
    data = await audio.read()
    if len(data) > MAX_AUDIO_BYTES:
        return error(413, "audio_too_large", "Recording is too long. Please keep it under a minute.")

    # 1. Speech -> text
    try:
        text, audio_info = stt.transcribe_wav(data)
    except stt.SpeechError as exc:
        log.info("STT failed: %s", exc.code)
        return error(exc.http_status, exc.code, exc.message)
    t1 = time.perf_counter()

    # 2. Text -> intent -> response
    result = STATE["classifier"].predict(text)
    t2 = time.perf_counter()
    log.info("heard=%r intent=%s conf=%.3f", text, result["predicted_intent"], result["confidence"])

    return {
        "recognized_text": text,
        **result,
        "audio": audio_info,
        "timing_ms": {"speech_to_text": round((t1 - t0) * 1000), "intent_model": round((t2 - t1) * 1000)},
    }


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@app.post("/predict-text")
def predict_text(body: TextIn):
    """Testing endpoint: bypasses the microphone and runs only the deep learning model."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")
    return {"recognized_text": text, **STATE["classifier"].predict(text)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
