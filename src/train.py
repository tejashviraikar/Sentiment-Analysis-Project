"""Train and compare classical + neural models for sentiment analysis.

Usage:
    python src/train.py                       # full dataset
    python src/train.py --sample 10000 --cv 0 # quick run
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import clean_text, load_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS, MODELS = ROOT / "results", ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def tfidf(max_features=20000, ngram_range=(1, 2)):
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                           min_df=2, sublinear_tf=True)


def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0),
        "Linear SVM": LinearSVC(C=0.5),
        # Trees on sparse text are slow: fewer trees + shallower depth keeps it practical.
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=40,
                                                n_jobs=-1, random_state=42),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                                          early_stopping=True, max_iter=30, random_state=42),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "IMDB Dataset.csv"))
    ap.add_argument("--sample", type=int, default=0, help="subsample N rows (0 = all)")
    ap.add_argument("--cv", type=int, default=3, help="CV folds (0 to skip)")
    ap.add_argument("--skip-rf", action="store_true", help="skip Random Forest (slowest model)")
    args = ap.parse_args()

    df, demo = load_data(args.data, args.sample)
    print(f"Loaded {len(df)} reviews | positive share = {df['label'].mean():.2f}")
    df["clean"] = df["review"].map(clean_text)

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        df["clean"], df["label"], df.index, test_size=0.2, stratify=df["label"], random_state=42)

    # ---- Experiment 1: feature representation (Unit II) ----
    print("\n== Feature comparison (Logistic Regression) ==")
    feats = {
        "Bag of Words (unigram)": CountVectorizer(max_features=20000, min_df=2),
        "TF-IDF (unigram)": TfidfVectorizer(max_features=20000, min_df=2, sublinear_tf=True),
        "TF-IDF (uni+bigram)": tfidf(),
    }
    feat_rows = []
    for name, vec in feats.items():
        p = Pipeline([("vec", vec), ("clf", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, p.predict(X_te))
        feat_rows.append({"Features": name, "Accuracy": round(acc, 4)})
        print(f"{name:26s} acc={acc:.4f}")
    pd.DataFrame(feat_rows).to_csv(RESULTS / "feature_comparison.csv", index=False)

    # ---- Experiment 2: model comparison (Units IV, VI) ----
    print("\n== Model comparison (TF-IDF uni+bigram) ==")
    rows, fitted, preds = [], {}, {}
    for name, clf in get_models().items():
        is_rf = name == "Random Forest"
        if is_rf and args.skip_rf:
            continue
        # Random Forest gets a smaller feature set (5k unigrams) and no CV, for speed.
        vec = tfidf(5000, (1, 1)) if is_rf else tfidf()
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        train_s = time.time() - t0
        y_pred = pipe.predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
        cv = (cross_val_score(pipe, X_tr, y_tr, cv=args.cv, scoring="f1", n_jobs=1)
              if args.cv > 1 and not is_rf else [np.nan])
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, y_pred), "Precision": p,
                     "Recall": r, "F1": f, "CV F1 (mean)": np.mean(cv),
                     "CV F1 (std)": np.std(cv), "Train time (s)": train_s})
        fitted[name], preds[name] = pipe, y_pred
        print(f"{name:20s} acc={rows[-1]['Accuracy']:.4f} f1={f:.4f} time={train_s:.1f}s")

    res = pd.DataFrame(rows).round(4).sort_values("F1", ascending=False)
    res.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\n", res.to_string(index=False))

    # ---- Plots: F1 bar chart + confusion matrices ----
    plt.figure(figsize=(7, 4))
    plt.barh(res["Model"][::-1], res["F1"][::-1], color="#4C78A8")
    plt.xlabel("F1 score")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.6))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                               display_labels=["neg", "pos"]).plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrices.png", dpi=150)
    plt.close()

    # ---- Save best model for the web app ----
    best = res.iloc[0]["Model"]
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    print(f"\nBest model: {best} -> models/best_model.joblib")

    # ---- Interpretability: top words from Logistic Regression ----
    lr = fitted["Logistic Regression"]
    vocab = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({"most_negative": vocab[order[:20]], "most_positive": vocab[order[::-1][:20]]})
    top.to_csv(RESULTS / "top_words.csv", index=False)

    # ---- Error analysis on the best model ----
    y_pred = preds[best]
    wrong = (y_pred != y_te.values)
    err = df.loc[idx_te[wrong], ["review", "sentiment"]].copy()
    err["predicted"] = np.where(y_pred[wrong] == 1, "positive", "negative")
    err["review"] = err["review"].str.slice(0, 300)
    err.head(100).to_csv(RESULTS / "misclassified_examples.csv", index=False)
    wc = df.loc[idx_te, "review"].str.split().str.len().values
    print(f"Error analysis: {wrong.sum()} errors ({wrong.mean():.1%}). "
          f"Avg length wrong={wc[wrong].mean():.0f} words vs right={wc[~wrong].mean():.0f} words")
    if demo:
        print("\n[!] Demo data was used. Put the real IMDB CSV in data/ and re-run.")


if __name__ == "__main__":
    main()
