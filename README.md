# 🌐 English → French Neural Machine Translation

A **Sequence-to-Sequence LSTM** model that translates English sentences into French, served via a Flask REST API, with a polished responsive web UI — containerized with Docker for easy deployment.

> Built as part of the **ZAKA AI Machine Learning Specialization – Model Deployment project.**

---

## 📋 Table of Contents

- [What the Model Does](#what-the-model-does)
- [Model Architecture & Performance](#model-architecture--performance)
- [How Translation Works (BTS)](#how-translation-works-bts)
- [Project Structure](#project-structure)
- [Quick Start – Run Locally (No Docker)](#quick-start--run-locally-no-docker)
- [Quick Start – Run with Docker](#quick-start--run-with-docker)
- [API Reference](#api-reference)
- [How to Use the Interface](#how-to-use-the-interface)
- [Training the Model (save_model.py)](#training-the-model-save_modelpy)
- [Jupyter Notebook (Google Colab)](#jupyter-notebook-google-colab)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Known Issues & Limitations](#known-issues--limitations)
- [Troubleshooting](#troubleshooting)

---

## What the Model Does

The model takes an **English sentence** as input and produces a **French translation** word by word using a neural encoder–decoder architecture trained on ~40,000 real sentence pairs.

**Example translations:**

| English | French |
|---------|--------|
| She is reading a book. | elle lit un livre . |
| Can you help me please? | pouvez-vous m ' aider ? |
| The weather is beautiful today. | le temps est beau aujourd ' hui . |
| We need to go to the market. | nous devons aller au marché . |

---

## Model Architecture & Performance

### Architecture Summary

| Component | Detail |
|-----------|--------|
| Architecture | Seq2Seq with LSTM Encoder & Decoder |
| Training data | ~40,000 English–French sentence pairs |
| Max sentence length | 12 words (both languages) |
| English vocabulary | ~6,000 tokens |
| French vocabulary | ~8,000 tokens |
| Embedding dimension | 64 |
| LSTM units | 128 |
| Dropout | 0.2 (applied to embeddings & LSTM outputs) |
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Training epochs | Up to 25 (with early stopping, patience=4) |
| Batch size | 64 |
| Train / Val split | 90% / 10% |

### Training Results

| Metric | Training Set | Validation Set |
|--------|-------------|---------------|
| Accuracy | ~98.7% | ~98.9% |
| Loss | ~0.043 | ~0.037 |

> The model ran for 25 epochs and achieved excellent convergence. The validation accuracy surpassed the training accuracy (a good sign of generalization on short sentences).

---

## How Translation Works (BTS)

Here is the complete end-to-end journey of a single translation request:

```
USER types "She is reading a book." → clicks Translate
      ↓
[BROWSER — JavaScript fetch()]
  POST /translate  {"text": "She is reading a book."}
      ↓
[FLASK — app.py → /translate route]
  1. Validates text (not empty, not > 300 chars)
  2. Confirms model files exist in /model/
  3. Calls translate() function
      ↓
[translate() function]

  STEP 1 — CLEAN TEXT
    lowercase + normalize punctuation spacing
    "she is reading a book ."

  STEP 2 — TOKENIZE  (eng_tokenizer.pkl)
    Each word → integer index from the learned vocabulary
    [42, 3, 87, 2, 120, 5]  →  padded to length 12
    [42, 3, 87, 2, 120, 5, 0, 0, 0, 0, 0, 0]

  STEP 3 — ENCODER  (encoder_model.h5)
    Reads the padded integer sequence
    Outputs: hidden state h + cell state c
    (h, c) = the neural "memory" of the entire English sentence

  STEP 4 — DECODER LOOP  (decoder_model.h5)
    i=0: input=<start>, states=(h,c)  → predicts "elle"  → new (h,c)
    i=1: input="elle",  states=(h,c)  → predicts "lit"   → new (h,c)
    i=2: input="lit",   states=(h,c)  → predicts "un"    → new (h,c)
    i=3: input="un",    states=(h,c)  → predicts "livre" → new (h,c)
    i=4: input="livre", states=(h,c)  → predicts <end>   → STOP ✋

    result_words = ["elle", "lit", "un", "livre"]

  STEP 5 — JOIN & RETURN
    "elle lit un livre"
      ↓
[FLASK]
  Returns: {"translation": "elle lit un livre"}
      ↓
[BROWSER — JavaScript]
  showTranslation() → renders text in French pane with fade-in animation
```

---

## Project Structure

```
.
├── app.py                            # Flask web application & translation logic
├── save_model.py                     # One-time script: train & save all model artifacts
├── English_to_French_NMT.ipynb       # Original research notebook (Google Colab)
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker container definition
├── .dockerignore                     # Files excluded from Docker image
│
├── templates/
│   └── index.html                    # Responsive web UI (HTML + JS)
│
├── static/
│   └── style.css                     # Dark glassmorphic design system
│
├── model/                            # ⚠️ Created after running save_model.py
│   ├── translation_model.h5          # Full training model (reference)
│   ├── encoder_model.h5              # Inference encoder (English → hidden states)
│   ├── decoder_model.h5              # Inference decoder (hidden states → French words)
│   ├── eng_tokenizer.pkl             # English vocabulary & word-to-index mapping
│   ├── fra_tokenizer.pkl             # French vocabulary & word-to-index mapping
│   └── config.json                   # Hyperparameters (enc_max_len, lstm_units, etc.)
│
└── Dataset to be uploaded on colab/
    ├── en.csv                        # ~140,000 English sentences (raw)
    └── fr.csv                        # ~140,000 French sentences (raw)
```

---

## Quick Start – Run Locally (No Docker)

This is the simplest way to run the app on your own machine.

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Step 1 – Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Train and save the model (one-time, ~10–30 min on CPU)

```bash
python save_model.py
```

This reads `en.csv` and `fr.csv`, trains the LSTM model, and saves 6 files into the `model/` folder. You only need to do this **once**.

### Step 4 – Start the Flask server

```bash
python app.py
```

### Step 5 – Open your browser

Navigate to **http://localhost:5000** and start translating!

---

## Quick Start – Run with Docker

### Prerequisites
- Docker Desktop installed and running
- Model files already generated (run `python save_model.py` first — the `model/` folder must **not** be empty when you build the image)

### Step 1 – Build the Docker image

```bash
docker build -t en-fr-translator .
```

### Step 2 – Run the container

```bash
docker run -p 5000:5000 en-fr-translator
```

Open your browser at **http://localhost:5000**

---

## API Reference

The Flask app exposes the following REST endpoints:

### `POST /translate`

Translates an English sentence to French.

**Request body (JSON):**
```json
{
  "text": "She is reading a book."
}
```

**Success response `200`:**
```json
{
  "translation": "elle lit un livre ."
}
```

**Error responses:**

| Status | Meaning | Example message |
|--------|---------|-----------------|
| `400` | Empty or missing input | `"Please enter some English text."` |
| `400` | Input too long (> 300 chars) | `"Input too long. Please keep it under 300 characters."` |
| `500` | Internal prediction error | `"Translation failed. Please try again."` |
| `503` | Model files not found | `"Model files not found. Please run python save_model.py first."` |

---

### `GET /health`

Simple health check endpoint.

**Response `200`:**
```json
{
  "status": "ok"
}
```

---

## How to Use the Interface

1. Type an English sentence in the **left text box** (max 300 characters).
2. Optionally click one of the **example chips** below the card to auto-fill a sentence.
3. Press the **Translate** button or use `Ctrl + Enter`.
4. The French translation appears on the **right pane** with a smooth animation.
5. Use the **Copy** button to copy the French result to your clipboard.

> **Note:** The very first translation after starting the server will be slightly slower (~3–5 seconds) as the model loads into memory. All subsequent translations are fast.

---

## Training the Model (save_model.py)

`save_model.py` is a standalone script that:

1. **Loads data** from `en.csv` and `fr.csv`
2. **Cleans & normalizes** text (lowercase, punctuation spacing, character filtering)
3. **Filters** to sentences with ≤ 12 words, caps at 40,000 pairs
4. **Builds tokenizers** for English (6,000 vocab) and French (8,000 vocab)
5. **Encodes & pads** sequences for the encoder input, decoder input, and decoder target
6. **Splits** data: 90% training, 10% validation
7. **Builds the Seq2Seq LSTM model** with Dropout regularization
8. **Trains** for up to 25 epochs with:
   - `EarlyStopping` (patience=4, restores best weights)
   - `ReduceLROnPlateau` (patience=2, factor=0.5)
9. **Builds separate inference models** (encoder + decoder)
10. **Saves all artifacts** to the `model/` directory

**Output files:**

| File | Description |
|------|-------------|
| `translation_model.h5` | Complete training model |
| `encoder_model.h5` | Inference encoder only |
| `decoder_model.h5` | Inference decoder only |
| `eng_tokenizer.pkl` | English tokenizer |
| `fra_tokenizer.pkl` | French tokenizer |
| `config.json` | Vocabulary sizes, max lengths, LSTM units |

---

## Jupyter Notebook (Google Colab)

The file `English_to_French_NMT.ipynb` is the **original research notebook** used to explore, develop, and evaluate this model. It contains:

- 📊 **Exploratory Data Analysis** — sentence length distributions, most common words
- 🧹 **Data Preprocessing** — cleaning, tokenization, sequence encoding
- 🏗️ **Model Architecture** — full Seq2Seq LSTM with encoder & decoder
- 📈 **Training curves** — loss and accuracy plots across epochs
- 🧪 **Evaluation** — BLEU-1 and BLEU-2 scores on 300 validation examples
- 🔍 **Side-by-side prediction comparison** — English vs correct vs predicted French

The `save_model.py` and `app.py` files are a direct production-ready port of the logic built in this notebook. The models and tokenizers are **identical** in architecture and behavior.

To run on Google Colab:
1. Upload `en.csv` and `fr.csv` to your Colab session
2. Run all cells
3. Download the generated `.h5` and `.pkl` files from `/content/` or Google Drive
4. Drop them into the local `model/` folder

---

## Deployment

The app can be deployed on any platform that supports Docker containers:

| Platform | Notes |
|----------|-------|
| **Render** | Free tier available, connect GitHub repo, select Docker runtime |
| **Railway** | Simple `docker build` + deploy, good free tier |
| **Fly.io** | Run `fly launch` then `fly deploy` |
| **Hugging Face Spaces** | Use "Docker" Space type |
| **Google Cloud Run** | Serverless Docker, scales to zero |
| **AWS App Runner** | Managed Docker deployment |

> ⚠️ **Important:** You must build the Docker image **after** `save_model.py` has been run, so that the `model/` folder with all `.h5` and `.pkl` files is included in the image.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | TensorFlow 2.15 / Keras — LSTM Seq2Seq |
| **Backend** | Python 3.10, Flask 3 |
| **Production server** | Gunicorn |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (no frameworks) |
| **Containerization** | Docker |
| **Research** | Jupyter Notebook (Google Colab) |

---

## Known Issues & Limitations

- The model was trained on **short, everyday sentences (≤ 12 words)**. Long or complex sentences may produce poor or truncated translations.
- **French grammatical gender** is not always predicted correctly, since English lacks gendered nouns.
- **Rare or out-of-vocabulary words** are replaced with `<oov>` and silently skipped in the output.
- **Apostrophes** in contractions (e.g., *l'homme*, *don't*) are spaced out during cleaning, which can affect output formatting.
- **First request after cold start** may be slow (3–5 seconds) as the model loads into memory.
- The model does **not** support accent input from English — accented French output is dependent on training data coverage.

---

## Troubleshooting

### ❌ `503 Service Unavailable` when translating
The `model/` folder is empty. Run:
```bash
python save_model.py
```
Then restart `python app.py`.

### ❌ `ModuleNotFoundError: No module named 'numpy'` (or similar)
Your virtual environment is not activated. Run:
```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
Then install:
```bash
pip install -r requirements.txt
```

### ❌ `save_model.py` is taking a very long time
This is expected when training on a **CPU**. The model trains for up to 25 epochs over 40,000 sentence pairs. On a standard laptop CPU this takes **10–30 minutes**. Training on a **GPU** (e.g., Google Colab) is significantly faster (~2–5 minutes).

### ❌ Docker build fails with "model/ is empty"
You need to run `python save_model.py` **before** building the Docker image, so the trained model files are present to be copied into the image layer.

### ❌ Translation output is blank or `(no translation produced)`
The input sentence may contain words not in the training vocabulary (very rare or technical words). Try a simpler, everyday sentence.
