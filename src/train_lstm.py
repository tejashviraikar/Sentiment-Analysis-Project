"""OPTIONAL deep-learning model (Unit VI): Embedding + LSTM. Needs `pip install tensorflow`.
Usage: python src/train_lstm.py --sample 20000 --epochs 4
"""
import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from utils import clean_text, load_data

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/IMDB Dataset.csv")
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--epochs", type=int, default=4)
args = ap.parse_args()

df, _ = load_data(args.data, args.sample)
texts = df["review"].map(clean_text).values
X_tr, X_te, y_tr, y_te = train_test_split(texts, df["label"].values, test_size=0.2,
                                          stratify=df["label"], random_state=42)

vec = tf.keras.layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
vec.adapt(X_tr)
model = tf.keras.Sequential([
    vec,
    tf.keras.layers.Embedding(20000, 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64, validation_split=0.1)

y_pred = (model.predict(X_te).ravel() > 0.5).astype(int)
p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
print(f"LSTM  acc={accuracy_score(y_te, y_pred):.4f} precision={p:.4f} recall={r:.4f} f1={f:.4f}")
