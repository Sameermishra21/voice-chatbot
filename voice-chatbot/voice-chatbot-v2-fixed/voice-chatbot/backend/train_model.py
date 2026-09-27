"""
train_model.py
--------------
Trains the Embedding + Bidirectional-LSTM intent classifier.

Run ONCE (from the project root):
    python backend/train_model.py

It will:
  1. Load backend/intents.json
  2. Clean + tokenize every pattern and build the vocabulary
  3. Convert text to integer sequences and pad them
  4. Encode the intent tags as integer labels
  5. Split into training / validation sets (stratified 80 / 20)
  6. Build and train the neural network
  7. Retrain the final model on ALL patterns for the best number of epochs
  8. Save:
       backend/chatbot_model.keras      – trained (final) model
       backend/model_metadata.json      – vocabulary, max_len, label mapping
       backend/training_history.json    – per-epoch accuracy / loss
       backend/training_report.json     – final metrics + classification report
       backend/plots/accuracy.png       – accuracy curve
       backend/plots/loss.png           – loss curve
The web app (app.py) only LOADS these files – it never retrains.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from nlp_utils import SimpleTokenizer

BASE = Path(__file__).resolve().parent
SEED = 42

# Hyper-parameters (small on purpose: tiny dataset, fast CPU training)
EMBED_DIM = 64
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.5
EPOCHS = 200
BATCH_SIZE = 16
VAL_SPLIT = 0.20


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    texts, tags = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            tags.append(intent["tag"])
    labels = sorted({i["tag"] for i in data["intents"]})
    return texts, tags, labels


def build_model(vocab_size: int, max_len: int, n_classes: int) -> tf.keras.Model:
    """
    Input (padded word indices)
      -> Embedding      : each word index -> dense 64-d vector (mask_zero ignores padding)
      -> Bidirectional LSTM : reads the sentence left->right AND right->left
      -> Dense (ReLU)   : non-linear combination of the sentence features
      -> Dropout        : regularisation against over-fitting
      -> Dense (Softmax): probability for each intent
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(max_len,), dtype="int32", name="token_ids"),
            tf.keras.layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True, name="embedding"),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS), name="bilstm"),
            tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense"),
            tf.keras.layers.Dropout(DROPOUT, name="dropout"),
            tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax"),
        ],
        name="intent_bilstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_curve(history: dict, metric: str, out: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.2))
    plt.plot(history[metric], label=f"Training {metric}", linewidth=2)
    plt.plot(history[f"val_{metric}"], label=f"Validation {metric}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main() -> None:
    set_seeds()

    # 1. Load data -----------------------------------------------------------
    texts, tags, labels = load_dataset(BASE / "intents.json")
    label_to_idx = {tag: i for i, tag in enumerate(labels)}
    y = np.array([label_to_idx[t] for t in tags], dtype="int32")
    print(f"Loaded {len(texts)} patterns across {len(labels)} intents.")

    # 2. Train / validation split (stratified -> every intent in both sets) ----
    x_train_txt, x_val_txt, y_train, y_val = train_test_split(
        texts, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y
    )

    # 3. Vocabulary is built from TRAINING text only (no data leakage) --------
    tokenizer = SimpleTokenizer().fit(x_train_txt)
    x_train = tokenizer.texts_to_padded(x_train_txt)
    x_val = tokenizer.texts_to_padded(x_val_txt)
    print(f"Vocabulary size: {tokenizer.vocab_size} | max_len: {tokenizer.max_len}")
    print(f"Train samples: {len(x_train)} | Validation samples: {len(x_val)}")

    # 4. Model ----------------------------------------------------------------
    model = build_model(tokenizer.vocab_size, tokenizer.max_len, len(labels))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    # 5. Evaluate the restored best model --------------------------------------
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    report = classification_report(
        y_val, y_pred, labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    )
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    print("\n================ FINAL METRICS (best weights) ================")
    print(f"Epochs run          : {len(history['loss'])} (best epoch {best_epoch})")
    print(f"Training accuracy   : {train_acc:.4f}")
    print(f"Validation accuracy : {val_acc:.4f}")
    print(f"Training loss       : {train_loss:.4f}")
    print(f"Validation loss     : {val_loss:.4f}")
    print(classification_report(y_val, y_pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))

    # 6. Final deployment model ----------------------------------------------
    # The 80/20 split above is only used to MEASURE generalisation and to choose
    # the number of epochs. The model the app uses is then retrained from scratch
    # on 100 % of the patterns for `best_epoch` epochs, so no example (e.g. the
    # sentence "goodbye") is left unseen just because it was in the validation set.
    set_seeds()
    final_tok = SimpleTokenizer().fit(texts)
    x_all = final_tok.texts_to_padded(texts)
    final_model = build_model(final_tok.vocab_size, final_tok.max_len, len(labels))
    final_model.fit(x_all, y, epochs=best_epoch, batch_size=BATCH_SIZE, verbose=0)
    final_loss, final_acc = final_model.evaluate(x_all, y, verbose=0)
    print(f"Final model (all {len(texts)} patterns, {best_epoch} epochs): "
          f"accuracy {final_acc:.4f}, loss {final_loss:.4f}")

    # 7. Save artefacts --------------------------------------------------------
    final_model.save(BASE / "chatbot_model.keras")

    metadata = {
        "labels": labels,                       # index -> intent tag
        "label_to_index": label_to_idx,         # intent tag -> index
        "word_index": final_tok.word_index,     # vocabulary (all patterns)
        "max_len": final_tok.max_len,
        "vocab_size": final_tok.vocab_size,
        "architecture": {
            "embedding_dim": EMBED_DIM, "lstm_units": LSTM_UNITS,
            "dense_units": DENSE_UNITS, "dropout": DROPOUT,
        },
    }
    (BASE / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (BASE / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (BASE / "training_report.json").write_text(json.dumps({
        "samples_total": len(texts), "samples_train": len(x_train), "samples_val": len(x_val),
        "n_intents": len(labels), "vocab_size_split": tokenizer.vocab_size,
        "vocab_size_final": final_tok.vocab_size, "max_len": final_tok.max_len,
        "epochs_run": len(history["loss"]), "best_epoch": best_epoch,
        "train_accuracy": round(float(train_acc), 4), "val_accuracy": round(float(val_acc), 4),
        "train_loss": round(float(train_loss), 4), "val_loss": round(float(val_loss), 4),
        "final_model_accuracy_all_data": round(float(final_acc), 4),
        "final_model_loss_all_data": round(float(final_loss), 4),
        "total_params": int(final_model.count_params()),
        "classification_report": report,
    }, indent=2), encoding="utf-8")

    plots = BASE / "plots"
    plots.mkdir(exist_ok=True)
    plot_curve(history, "accuracy", plots / "accuracy.png", "Training vs Validation Accuracy")
    plot_curve(history, "loss", plots / "loss.png", "Training vs Validation Loss")
    print(f"\nSaved model, metadata, history and plots to {BASE}")


if __name__ == "__main__":
    main()
