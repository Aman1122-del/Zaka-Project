# English to French Translator

This is a neural machine translation web app that translates English sentences into French. The model is based on an LSTM encoder-decoder architecture and was trained on around 40,000 sentence pairs. The frontend is a simple Flask web app and everything runs inside Docker.

Live demo is here: https://zaka-project-production.up.railway.app/

![App Preview](preview.png)

---

## What this project uses

- Python and Flask for the backend
- TensorFlow for the NMT model
- Gunicorn as the production server
- Docker to containerize everything

---

## How to run it locally with Docker

Make sure Docker Desktop is open and the engine is running before starting.

```bash
git clone https://github.com/Aman1122-del/Zaka-Project.git
cd Zaka-Project
docker build -t eng-to-fra-nmt .
docker run -p 5000:5000 eng-to-fra-nmt
```

Then open your browser and go to http://localhost:5000

The first time you translate something it might take a few seconds because the model loads on the first request. After that it is fast.

If you want to run it without Docker just install the dependencies and run the Flask app directly.

```bash
pip install -r requirements.txt
python app.py
```

---

## How to use it

Open the app in the browser. Type an English sentence in the left box and click the Translate button. The French translation will appear on the right side. You can also click any of the example sentences at the bottom and it will auto-translate.

There is a copy button on the output box if you want to copy the translation. The input is limited to 300 characters.

---

## Known issues and limitations

The model was trained on short and simple sentences so it works best with sentences that have basic grammar. Long or complex sentences might not translate well or might give weird output. It also does not handle slang or informal language very well.

The model only translates from English to French, there is no reverse direction.

Sometimes the output has small grammar mistakes, this is expected from a small LSTM model. For production use you would want something like a Transformer model.

---

## Project structure

```
Zaka-Project/
    app.py               the main Flask app
    Dockerfile           builds the Docker image
    requirements.txt     Python packages
    save_model.py        script used to train and save the model
    model/               saved model files and tokenizers
    templates/           HTML templates
    static/              CSS styles
```
