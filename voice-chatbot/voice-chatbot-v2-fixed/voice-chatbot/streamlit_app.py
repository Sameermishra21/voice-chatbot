"""
streamlit_app.py
----------------
Streamlit version of the Voice-Enabled Chatbot (for Streamlit Community Cloud).

Flow:
  🎤 st.audio_input  -> browser asks for microphone permission, records live
                        (NO file upload) and returns a 16 kHz WAV
  -> backend/speech_to_text.py  : validate audio + Google Web Speech API -> text
  -> backend/nlp_utils.py       : Embedding + BiLSTM intent model -> intent + confidence
  -> response from intents.json -> shown on the page, 🔊 spoken by the browser

The trained model is loaded ONCE (st.cache_resource); it is never retrained here.

Run locally:
    streamlit run streamlit_app.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
import streamlit.components.v1 as components

BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND))

from nlp_utils import IntentClassifier  # noqa: E402
import speech_to_text as stt           # noqa: E402

CONFIDENCE_THRESHOLD = 0.60

st.set_page_config(page_title="Voice-Enabled AI Chatbot", page_icon="🎙️", layout="centered")


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained BiLSTM model…")
def load_classifier() -> IntentClassifier:
    return IntentClassifier(BACKEND, threshold=CONFIDENCE_THRESHOLD)


def speak_button(text: str) -> None:
    """🔊 Play Response – uses the browser's built-in speechSynthesis (no server audio)."""
    safe = json.dumps(text)
    components.html(
        f"""
        <button id="play" style="font-size:16px;font-weight:600;padding:10px 20px;border-radius:10px;
                border:2px solid #4f46e5;background:white;color:#4f46e5;cursor:pointer;">
          🔊 Play Response
        </button>
        <script>
          const btn = document.getElementById("play");
          if (!("speechSynthesis" in window)) {{ btn.disabled = true; btn.textContent = "🔇 TTS not supported"; }}
          btn.onclick = () => {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance({safe});
            u.lang = "en-US";
            window.speechSynthesis.speak(u);
          }};
        </script>
        """,
        height=56,
    )


def process(audio_bytes: bytes, clf: IntentClassifier) -> dict:
    """Speech -> text -> intent. Returns a result dict or {'error':..., 'message':...}."""
    try:
        text, info = stt.transcribe_wav(audio_bytes)
    except stt.SpeechError as exc:
        return {"error": exc.code, "message": exc.message}
    result = clf.predict(text)
    return {"recognized_text": text, **result, "audio": info}


# --------------------------------------------------------------------------- #
st.markdown(
    "<h1 style='text-align:center;margin-bottom:0'>🎙️ Voice-Enabled AI Chatbot</h1>"
    "<p style='text-align:center;color:gray'>Speech Recognition + Embedding/BiLSTM Intent Classification</p>",
    unsafe_allow_html=True,
)

try:
    clf = load_classifier()
    st.caption(f"🟢 Model loaded · {len(clf.labels)} intents · confidence threshold {CONFIDENCE_THRESHOLD:.0%}")
except Exception as exc:  # model files missing / corrupted
    st.error(f"❌ Could not load the trained model: {exc}. Run `python backend/train_model.py` first.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_hash = None
    st.session_state.last_result = None

st.markdown("#### 🎤 Speak your question")
st.caption(
    "Click the **microphone** to start speaking and click the **stop** button when you finish. "
    "Your browser will ask for microphone permission the first time. "
    "Example: *“What courses do you offer?”*"
)
audio = st.audio_input("Record your voice", label_visibility="collapsed", key="mic")

if audio is not None:
    audio_bytes = audio.getvalue()
    digest = hashlib.sha1(audio_bytes).hexdigest()
    if digest != st.session_state.last_hash:          # process each recording once
        with st.status("⏳ Processing… (speech-to-text + intent model)", expanded=False) as status:
            result = process(audio_bytes, clf)
            if "error" in result:
                status.update(label="❌ Could not process the recording", state="error")
            else:
                status.update(label="🟢 Response ready", state="complete")
        st.session_state.last_hash = digest
        st.session_state.last_result = result
        if "error" not in result:
            st.session_state.history.append(result)

result = st.session_state.last_result

st.divider()
if result is None:
    st.markdown("**Recognized Speech:**")
    st.info("Your converted speech will appear here.")
    st.markdown("**Chatbot Response:**")
    st.info("The chatbot's answer will appear here.")
elif "error" in result:
    st.error(f"❌ {result['message']}")
    if result["error"] == "stt_service_error":
        st.caption("The speech recognition service could not be reached. Please try again in a moment.")
else:
    st.markdown("**Recognized Speech:**")
    st.info(f"“{result['recognized_text']}”")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detected Intent:**")
        if result.get("out_of_scope"):
            st.warning("`out_of_scope` – not a college question")
        elif result["low_confidence"]:
            st.warning(f"unknown  (closest: `{result['predicted_intent']}`)")
        else:
            st.success(f"`{result['intent']}`")
    with c2:
        st.markdown("**Confidence:**")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%", label_visibility="collapsed")
        st.progress(min(1.0, float(result["confidence"])))

    st.markdown("**Chatbot Response:**")
    st.success(result["response"])
    speak_button(result["response"])

    with st.expander("Model details (top-3 predictions)"):
        for p in result["top_predictions"]:
            st.write(f"- `{p['intent']}` — {p['confidence'] * 100:.1f}%")
        st.caption(f"Audio: {result['audio']['duration_s']} s at {result['audio']['sample_rate']} Hz")

if st.session_state.history:
    st.divider()
    st.markdown("#### Conversation")
    for turn in reversed(st.session_state.history[-10:]):
        with st.chat_message("user"):
            st.write(turn["recognized_text"])
        with st.chat_message("assistant"):
            st.write(turn["response"])
            st.caption(f"{turn['intent']} · {turn['confidence'] * 100:.1f}%")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. 🎤 Browser records your microphone (16 kHz WAV)\n"
        "2. 🗣️ Speech-to-Text (Google Web Speech API)\n"
        "3. 🔤 Tokenize → pad → **Embedding → BiLSTM → Dense → Dropout → Softmax**\n"
        "4. 🎯 Intent + confidence (below 60 % → asks you to repeat)\n"
        "5. 💬 Response from `intents.json`, 🔊 spoken by the browser"
    )
    st.header("Try asking")
    st.markdown(
        "- Hello\n- What is your name?\n- What courses do you offer?\n- What are your working hours?\n"
        "- How much is the fee?\n- Is there a hostel?\n- How can I contact you?\n- Goodbye"
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()
