"""Shared helpers: data loading and text cleaning (Unit II: preprocessing)."""
import re
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: "not good" must not become "good".
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "cannot", "neither"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase, strip HTML, expand n't, keep letters only, drop stop words."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)          # HTML tags (IMDB has <br />)
    text = re.sub(r"n't\b", " not", text)       # don't -> do not
    text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def _synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Tiny fake dataset, ONLY for smoke-testing the code without the real CSV."""
    rnd = random.Random(seed)
    pos = ["great", "excellent", "amazing", "loved", "brilliant", "wonderful", "superb", "enjoyable"]
    neg = ["terrible", "awful", "boring", "hated", "poor", "waste", "disappointing", "dull"]
    filler = ["the", "movie", "plot", "acting", "story", "film", "was", "really", "very", "scenes",
              "actor", "ending", "cast", "music", "director", "overall"]
    rows = []
    for _ in range(n):
        label = rnd.randint(0, 1)
        main, other = (pos, neg) if label else (neg, pos)
        words = [rnd.choice(filler) for _ in range(rnd.randint(8, 25))]
        for _ in range(rnd.randint(1, 3)):
            words.insert(rnd.randrange(len(words)), rnd.choice(main))
        if rnd.random() < 0.2:                   # label noise so scores aren't 100%
            words.insert(rnd.randrange(len(words)), rnd.choice(other))
        rows.append(("this " + " ".join(words), "positive" if label else "negative"))
    return pd.DataFrame(rows, columns=["review", "sentiment"])


def load_data(path: str = "data/IMDB Dataset.csv", sample: int = 0):
    """Load IMDB CSV (columns: review, sentiment). Falls back to synthetic demo data."""
    if Path(path).exists():
        df = pd.read_csv(path)
        demo = False
    else:
        print(f"[!] {path} not found -> using SYNTHETIC demo data (results are meaningless).")
        df, demo = _synthetic_data(), True
    df = df.dropna(subset=["review", "sentiment"]).drop_duplicates(subset="review")
    if sample and sample < len(df):
        df = df.sample(sample, random_state=42)
    df = df.reset_index(drop=True)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    return df, demo
