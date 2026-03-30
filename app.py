"""
app.py  –  Flask application for English → French Neural Machine Translation
"""

import os
import re
import json
import pickle

import numpy as np
from flask import Flask, render_template, request, jsonify

# ─── Lazy-load model on first request ────────────────────────────────────────
_encoder_model      = None
_decoder_model      = None
_eng_tokenizer      = None
_fra_tokenizer      = None
_index_to_french    = None
_start_idx          = None
_end_idx            = None
_enc_max_len        = None
_lstm_units         = None

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)


def load_artifacts():
    """Load all model artifacts once and cache them in module-level globals."""
    global _encoder_model, _decoder_model
    global _eng_tokenizer, _fra_tokenizer
    global _index_to_french, _start_idx, _end_idx
    global _enc_max_len, _lstm_units

    if _encoder_model is not None:
        return  # already loaded

    # Import TF here to avoid slowing down the import of this module
    from tensorflow.keras.models import load_model

    print("Loading model artifacts…")
    _encoder_model = load_model(os.path.join(MODEL_DIR, "encoder_model.h5"))
    _decoder_model = load_model(os.path.join(MODEL_DIR, "decoder_model.h5"))

    with open(os.path.join(MODEL_DIR, "eng_tokenizer.pkl"), "rb") as f:
        _eng_tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "fra_tokenizer.pkl"), "rb") as f:
        _fra_tokenizer = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)

    _enc_max_len     = cfg["enc_max_len"]
    _lstm_units      = cfg["lstm_units"]
    _index_to_french = {idx: word for word, idx in _fra_tokenizer.word_index.items()}
    _start_idx       = _fra_tokenizer.word_index["<start>"]
    _end_idx         = _fra_tokenizer.word_index["<end>"]
    print("Model loaded.")


def clean_text(sentence: str) -> str:
    sentence = str(sentence).lower().strip()
    sentence = re.sub(r"([?.!,'])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ?.!,]+", " ", sentence)
    return sentence.strip()


def translate(sentence: str) -> str:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    load_artifacts()

    cleaned = clean_text(sentence)
    seq     = _eng_tokenizer.texts_to_sequences([cleaned])
    seq     = pad_sequences(seq, maxlen=_enc_max_len, padding="post")

    h, c = _encoder_model.predict(seq, verbose=0)

    target_seq = np.array([[_start_idx]])
    result_words = []

    for _ in range(50):
        out, h, c = _decoder_model.predict([target_seq, h, c], verbose=0)
        word_idx = int(np.argmax(out[0, -1, :]))

        if word_idx == _end_idx or word_idx == 0:
            break

        word = _index_to_french.get(word_idx, "")
        if word not in ("<start>", "<end>", "<oov>", ""):
            result_words.append(word)

        target_seq = np.array([[word_idx]])

    return " ".join(result_words).strip()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some English text."}), 400

    if len(text) > 300:
        return jsonify({"error": "Input too long. Please keep it under 300 characters."}), 400

    # Check model files exist
    required = ["encoder_model.h5", "decoder_model.h5",
                "eng_tokenizer.pkl", "fra_tokenizer.pkl", "config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return jsonify({
            "error": (
                "Model files not found. "
                "Please run  python save_model.py  first to train and save the model."
            )
        }), 503

    try:
        french = translate(text)
        if not french:
            french = "(no translation produced)"
        return jsonify({"translation": french})
    except Exception as exc:
        app.logger.error("Translation error: %s", exc)
        return jsonify({"error": "Translation failed. Please try again."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
