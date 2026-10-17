"""Text features, and the genre classifier that fills in genres the scrape did not capture."""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

import config

# Words that appear in many descriptions but say nothing about the book's content: audiobook vocabulary, bestseller
# blurbs and publisher/award boilerplate ("Penguin presents", "Random House presents", "Audie Award winner").
DOMAIN_STOP_WORDS = ["audiobook", "audiobooks", "audible", "listen", "listening", "narrated", "narrator", "book",
                     "books", "edition", "unabridged", "abridged", "read", "author", "bestselling", "bestseller",
                     "new", "york", "times", "best", "seller", "copyright", "pdf", "accompanying", "pages",
                     "penguin", "presents", "random", "audie", "award", "awards", "winner", "finalist", "nominee",
                     "harpercollins", "hachette", "macmillan", "simon", "schuster", "publishers", "publishing",
                     "recorded", "recording", "studios", "originals", "exclusive", "exclusively", "brought",
                     "international", "selling", "acclaimed", "million", "copies", "sold", "worldwide"]


def clean_text(s) -> str:
    """Lower case, letters only (keeps Devanagari), single spaces."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-zऀ-ॿ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def book_text(df: pd.DataFrame) -> pd.Series:
    """The text a book is described by: its title (twice, so short-description books still have weight) and its
    cleaned description. Genres and author are separate feature blocks, not text, so the models can weight them."""
    title = df["title"].map(clean_text)
    return (title + " " + title + " " + df["description"].map(clean_text)).str.strip()


def make_vectorizer(**overrides) -> TfidfVectorizer:
    params = dict(config.TFIDF_PARAMS)
    params["stop_words"] = list(TfidfVectorizer(stop_words="english").get_stop_words()) + DOMAIN_STOP_WORDS
    params.update(overrides)
    return TfidfVectorizer(**params)


# ---------------------------------------------------------------- genre inference
def genre_classifier():
    return make_pipeline(make_vectorizer(min_df=2, max_df=0.8),
                         LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))


def confidence_threshold(conf: np.ndarray, correct: np.ndarray) -> float:
    """Lowest probability cut-off at which the predictions above it are right GENRE_INFERENCE_TARGET_ACCURACY of
    the time (on cross-validation)."""
    order = np.argsort(-conf)
    running_accuracy = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(running_accuracy >= config.GENRE_INFERENCE_TARGET_ACCURACY)[0]
    ok = ok[ok >= 20]                       # ignore the first few, where accuracy is noise
    return float(conf[order][ok.max()]) if len(ok) else 1.0


def infer_missing_genres(books: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train on books whose genre the scrape captured, predict for books that have a real description but no listed
    genre. 5-fold cross-validation measures how far the inferred genres can be trusted and sets the confidence
    cut-off; less confident predictions stay 'Unknown'. Books with neither a genre nor a description stay 'Unknown'."""
    books = books.copy()
    text = book_text(books)
    labelled = books["genre_source"].eq("listed") & books["has_description"]
    target = books["genre_source"].eq("unknown") & books["has_description"]
    X, y = text[labelled], books.loc[labelled, "primary_genre"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    proba = cross_val_predict(genre_classifier(), X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(y.unique()))
    pred, conf = classes[proba.argmax(1)], proba.max(1)
    threshold = confidence_threshold(conf, pred == y.values)
    confident = conf >= threshold
    top3 = (np.argsort(-proba, axis=1)[:, :3] == np.searchsorted(classes, y.values)[:, None]).any(1)

    model = genre_classifier().fit(X, y)
    joblib.dump(model, config.GENRE_CLASSIFIER)
    new_proba = model.predict_proba(text[target])
    new_pred = model.classes_[new_proba.argmax(1)]
    keep = new_proba.max(1) >= threshold
    idx = books.index[target]
    books.loc[idx[keep], "primary_genre"] = new_pred[keep]
    books.loc[idx[keep], "genre_source"] = "inferred"

    report = {
        "trained_on": int(labelled.sum()),
        "classes": len(classes),
        "cv_accuracy": round(accuracy_score(y, pred), 4),
        "cv_macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "cv_top3_accuracy": round(float(top3.mean()), 4),
        "cv_accuracy_when_confident": round(accuracy_score(y[confident], pred[confident]), 4),
        "cv_share_confident": round(float(confident.mean()), 4),
        "min_probability": round(float(threshold), 3),
        "target_accuracy": config.GENRE_INFERENCE_TARGET_ACCURACY,
        "candidates": int(target.sum()),
        "inferred": int(keep.sum()),
        "left_unknown_low_confidence": int((~keep).sum()),
        "per_genre": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score", "support")}
                      for k, v in classification_report(y, pred, output_dict=True, zero_division=0).items()
                      if k in classes},
    }
    return books, report
