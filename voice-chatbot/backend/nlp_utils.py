"""
nlp_utils.py
------------
Shared text-processing code used by BOTH train_model.py and app.py.

Keeping preprocessing in one place guarantees that the text the model sees at
inference time is processed exactly the same way as during training.

Contents
  * clean_text()        – lower-case + strip punctuation
  * SimpleTokenizer     – builds a word -> index vocabulary, converts text to
                          integer sequences and pads them to a fixed length
  * IntentClassifier    – loads the trained Keras model + metadata and predicts
                          an intent with a confidence score
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

PAD_TOKEN = "<PAD>"   # index 0 – used for padding (masked by the Embedding layer)
OOV_TOKEN = "<OOV>"   # index 1 – any word not seen during training

LOW_CONFIDENCE_MESSAGE = "I'm not sure I understood that. Could you please say it again?"


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lower-case the text, keep only letters/digits/apostrophes, squash spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens."""
    return clean_text(text).split()


# --------------------------------------------------------------------------- #
# Tokenizer / vocabulary
# --------------------------------------------------------------------------- #
class SimpleTokenizer:
    """
    Minimal, transparent word-level tokenizer.

    fit()            -> builds vocabulary {word: index}; 0 = PAD, 1 = OOV
    texts_to_padded  -> list of texts -> 2-D numpy array (n_samples, max_len)
    """

    def __init__(self, word_index: dict[str, int] | None = None, max_len: int = 0):
        self.word_index: dict[str, int] = word_index or {PAD_TOKEN: 0, OOV_TOKEN: 1}
        self.max_len = max_len

    # ---- training ---------------------------------------------------------
    def fit(self, texts: list[str]) -> "SimpleTokenizer":
        counts: dict[str, int] = {}
        longest = 0
        for text in texts:
            words = tokenize(text)
            longest = max(longest, len(words))
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        # Most frequent words get the smallest indices (deterministic order).
        for word, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)
        # A little head-room so slightly longer spoken sentences still fit.
        self.max_len = longest + 4
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.word_index)

    # ---- inference --------------------------------------------------------
    def text_to_sequence(self, text: str) -> list[int]:
        oov = self.word_index[OOV_TOKEN]
        return [self.word_index.get(w, oov) for w in tokenize(text)]

    def pad(self, seq: list[int]) -> list[int]:
        """Post-padding with zeros; long inputs keep their FIRST max_len words."""
        seq = seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    def texts_to_padded(self, texts: list[str]) -> np.ndarray:
        return np.array([self.pad(self.text_to_sequence(t)) for t in texts], dtype="int32")

    def known_word_count(self, text: str) -> int:
        """How many words of `text` exist in the vocabulary (used for OOV check)."""
        return sum(1 for w in tokenize(text) if w in self.word_index)


# --------------------------------------------------------------------------- #
# Intent classifier used by the web backend
# --------------------------------------------------------------------------- #
class IntentClassifier:
    """
    Wraps the trained Embedding + BiLSTM model.

    predict(text) returns a dict:
        intent, confidence, response, low_confidence, top_predictions
    """

    def __init__(self, model_dir: str | Path, threshold: float = 0.60):
        # Imported here so that importing nlp_utils is cheap (e.g. in tests).
        import tensorflow as tf

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        intents = json.loads((model_dir / "intents.json").read_text(encoding="utf-8"))

        self.model = tf.keras.models.load_model(model_dir / "chatbot_model.keras")
        self.tokenizer = SimpleTokenizer(meta["word_index"], meta["max_len"])
        self.labels: list[str] = meta["labels"]           # index -> tag
        self.threshold = threshold
        self.responses = {i["tag"]: i["responses"] for i in intents["intents"]}

        # Warm-up call so the first real request is not slow.
        self.model.predict(self.tokenizer.texts_to_padded(["hello"]), verbose=0)

    def predict(self, text: str) -> dict:
        x = self.tokenizer.texts_to_padded([text])
        probs = self.model.predict(x, verbose=0)[0]        # softmax output
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        tag = self.labels[best]
        confidence = float(probs[best])

        top = [{"intent": self.labels[int(i)], "confidence": round(float(probs[int(i)]), 4)}
               for i in order[:3]]

        # Two safety checks before trusting the prediction:
        #  1. none of the words are in the vocabulary -> the model is guessing
        #  2. softmax probability below the threshold  -> not confident enough
        no_known_words = self.tokenizer.known_word_count(text) == 0
        low_conf = no_known_words or confidence < self.threshold

        if low_conf:
            return {
                "intent": "unknown",
                "predicted_intent": tag,
                "confidence": round(confidence, 4),
                "response": LOW_CONFIDENCE_MESSAGE,
                "low_confidence": True,
                "top_predictions": top,
            }

        return {
            "intent": tag,
            "predicted_intent": tag,
            "confidence": round(confidence, 4),
            "response": random.choice(self.responses[tag]),
            "low_confidence": False,
            "top_predictions": top,
        }
