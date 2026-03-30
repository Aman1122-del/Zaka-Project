import os, json, pickle
import numpy as np
import pandas as pd
import re
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "Dataset to be uploaded on colab")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_LEN    = 12
MAX_PAIRS  = 40000
ENG_VOCAB  = 6000
FRA_VOCAB  = 8000
EMBED_DIM  = 64
LSTM_UNITS = 128
EPOCHS     = 25
BATCH_SIZE = 64

print("Loading data...")
en_df = pd.read_csv(os.path.join(DATA_DIR, "en.csv"), header=None, names=["english"])
fr_df = pd.read_csv(os.path.join(DATA_DIR, "fr.csv"), header=None, names=["french"])
df = pd.concat([en_df, fr_df], axis=1).dropna().reset_index(drop=True)
print(f"  Total pairs: {len(df)}")

def clean_text(sentence):
    sentence = str(sentence).lower().strip()
    sentence = re.sub(r"([?.!,'])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ?.!,]+", " ", sentence)
    return sentence.strip()

df["english_clean"] = df["english"].apply(clean_text)
df["french_clean"]  = df["french"].apply(clean_text)
df["french_in"]     = "<start> " + df["french_clean"]
df["french_out"]    = df["french_clean"] + " <end>"

df["eng_len"] = df["english_clean"].apply(lambda x: len(x.split()))
df["fra_len"] = df["french_clean"].apply(lambda x: len(x.split()))
df = df[(df["eng_len"] <= MAX_LEN) & (df["fra_len"] <= MAX_LEN)].reset_index(drop=True)

if len(df) > MAX_PAIRS:
    df = df.sample(MAX_PAIRS, random_state=42).reset_index(drop=True)

print(f"  After filtering: {len(df)} pairs")

print("Building tokenizers...")
eng_tokenizer = Tokenizer(num_words=ENG_VOCAB, oov_token="<oov>", filters="")
eng_tokenizer.fit_on_texts(df["english_clean"])

fra_tokenizer = Tokenizer(num_words=FRA_VOCAB, oov_token="<oov>", filters="")
fra_tokenizer.fit_on_texts(df["french_in"].tolist() + df["french_out"].tolist())

eng_vocab_size = min(ENG_VOCAB, len(eng_tokenizer.word_index)) + 1
fra_vocab_size = min(FRA_VOCAB, len(fra_tokenizer.word_index)) + 1
print(f"  English vocab: {eng_vocab_size}  |  French vocab: {fra_vocab_size}")

enc_seqs = pad_sequences(eng_tokenizer.texts_to_sequences(df["english_clean"]), padding="post")
dec_in   = pad_sequences(fra_tokenizer.texts_to_sequences(df["french_in"]),     padding="post")
dec_out  = pad_sequences(fra_tokenizer.texts_to_sequences(df["french_out"]),    padding="post")
dec_out  = np.expand_dims(dec_out, -1)

enc_max_len = enc_seqs.shape[1]

enc_train, enc_val, dec_in_train, dec_in_val, dec_out_train, dec_out_val = train_test_split(
    enc_seqs, dec_in, dec_out, test_size=0.1, random_state=42
)
print(f"  Train: {len(enc_train)}  |  Val: {len(enc_val)}")

print("Building model...")
encoder_inputs = Input(shape=(None,), name="encoder_input")
enc_emb        = Embedding(eng_vocab_size, EMBED_DIM, mask_zero=True)(encoder_inputs)
enc_emb        = Dropout(0.2)(enc_emb)
_, state_h, state_c = LSTM(LSTM_UNITS, return_state=True, name="encoder_lstm")(enc_emb)

decoder_inputs = Input(shape=(None,), name="decoder_input")
dec_emb        = Embedding(fra_vocab_size, EMBED_DIM, mask_zero=True)(decoder_inputs)
dec_emb        = Dropout(0.2)(dec_emb)
dec_out_seq, _, _ = LSTM(LSTM_UNITS, return_sequences=True, return_state=True, name="decoder_lstm")(
    dec_emb, initial_state=[state_h, state_c]
)
dec_out_seq    = Dropout(0.2)(dec_out_seq)
output         = Dense(fra_vocab_size, activation="softmax", name="output_layer")(dec_out_seq)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("Training...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1)

model.fit(
    [enc_train, dec_in_train], dec_out_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([enc_val, dec_in_val], dec_out_val),
    callbacks=[early_stop, reduce_lr],
)

print("Building inference models...")
encoder_model = Model(encoder_inputs, [state_h, state_c])

dec_state_h_inp = Input(shape=(LSTM_UNITS,), name="dec_state_h")
dec_state_c_inp = Input(shape=(LSTM_UNITS,), name="dec_state_c")

decoder_lstm_layer      = model.get_layer("decoder_lstm")
decoder_embedding_layer = model.layers[3]

dec_emb_inf         = decoder_embedding_layer(decoder_inputs)
dec_out_inf, h2, c2 = decoder_lstm_layer(dec_emb_inf, initial_state=[dec_state_h_inp, dec_state_c_inp])
dec_out_inf         = model.get_layer("output_layer")(dec_out_inf)

decoder_model = Model(
    [decoder_inputs, dec_state_h_inp, dec_state_c_inp],
    [dec_out_inf, h2, c2],
)

print("Saving artifacts...")
model.save(os.path.join(MODEL_DIR, "translation_model.h5"))
encoder_model.save(os.path.join(MODEL_DIR, "encoder_model.h5"))
decoder_model.save(os.path.join(MODEL_DIR, "decoder_model.h5"))

with open(os.path.join(MODEL_DIR, "eng_tokenizer.pkl"), "wb") as f:
    pickle.dump(eng_tokenizer, f)
with open(os.path.join(MODEL_DIR, "fra_tokenizer.pkl"), "wb") as f:
    pickle.dump(fra_tokenizer, f)

config = {
    "eng_vocab_size": int(eng_vocab_size),
    "fra_vocab_size": int(fra_vocab_size),
    "enc_max_len":    int(enc_max_len),
    "lstm_units":     LSTM_UNITS,
    "embed_dim":      EMBED_DIM,
}
with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("Done! All artifacts saved to ./model/")
print("  translation_model.h5")
print("  encoder_model.h5")
print("  decoder_model.h5")
print("  eng_tokenizer.pkl")
print("  fra_tokenizer.pkl")
print("  config.json")
