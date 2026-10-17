"""Data and model access for the app: cached loading, filters, and the 'why recommended' explanations."""
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config

ARTIFACTS = [config.RECOMMENDER_ARTIFACTS, config.MODEL_METRICS, config.CLUSTER_SUMMARY, config.CLEANING_REPORT]
METHODS = {
    "Hybrid": "hybrid",
    "Content": "content",
    "Cluster": "cluster",
}
METHOD_HELP = ("**Hybrid** (recommended): similarity of description, genres and author, plus a small boost for books in the same "
               "cluster and for well-rated books (weights tuned on validation data).  \n"
               "**Content**-based: pure similarity of description, genres and author.  \n"
               "**Cluster**-based: the best-rated books in the same cluster of similar books.")


def _stamp() -> tuple:
    """Changes whenever a model file changes, so the caches refresh after a redeploy
    (Streamlit Cloud pulls new files without restarting the server)."""
    return tuple(p.stat().st_mtime_ns for p in ARTIFACTS)


@st.cache_resource(show_spinner="Loading the recommender…", max_entries=1)
def _artifacts(stamp):
    return joblib.load(config.RECOMMENDER_ARTIFACTS)


def artifacts():
    return _artifacts(_stamp())


def books() -> pd.DataFrame:
    return artifacts().books


def recommender():
    return artifacts().recommender


@st.cache_data(show_spinner=False)
def _json(path, stamp):
    return json.loads(path.read_text())


def metrics():
    return _json(config.MODEL_METRICS, _stamp())


def cluster_summary():
    return _json(config.CLUSTER_SUMMARY, _stamp())


def cleaning_report():
    return _json(config.CLEANING_REPORT, _stamp())


@st.cache_data(show_spinner=False)
def _options(stamp):
    b = books()
    labels = (b["title"] + " — " + b["author"]).tolist()
    genres = sorted(g for g in b["primary_genre"].unique() if g != config.UNKNOWN_GENRE)
    fine = b["genres"].dropna().str.split(r" \| ").explode().str.strip().value_counts()
    authors = b["author"].value_counts()
    return labels, genres, list(fine.index), list(authors.index)


def options():
    """(book labels in book_id order, broad genres, fine genres by frequency, authors by frequency)."""
    return _options(_stamp())


# ---------------------------------------------------------------- filters
@dataclass(frozen=True)
class Filters:
    genres: tuple = ()
    min_rating: float = 0.0
    min_reviews: int = 0
    max_price: float | None = None
    max_hours: float | None = None
    include_unrated: bool = True
    gems_only: bool = False
    exclude_authors: tuple = ()


def mask(f: Filters) -> np.ndarray:
    b = books()
    m = np.ones(len(b), dtype=bool)
    if f.genres:
        m &= b["primary_genre"].isin(f.genres).to_numpy()
    rated = b["is_rated"].to_numpy()
    if f.min_rating > 0:
        m &= (b["rating"].fillna(0).to_numpy() >= f.min_rating) | (~rated & f.include_unrated)
    if not f.include_unrated:
        m &= rated
    if f.min_reviews:
        m &= b["num_reviews"].to_numpy() >= f.min_reviews
    if f.max_price is not None:
        m &= b["price"].fillna(0).to_numpy() <= f.max_price
    if f.max_hours is not None:   # books of unknown length are left out: they might be longer
        m &= (b["listening_minutes"].to_numpy() <= f.max_hours * 60)
    if f.gems_only:
        m &= b["hidden_gem"].to_numpy()
    if f.exclude_authors:
        m &= ~b["author"].isin(f.exclude_authors).to_numpy()
    return m


FILTER_DEFAULTS = {"genres": [], "rating": 0.0, "reviews": 0, "price": "Any", "hours": "Any", "unrated": True,
                   "gems": False}


def reset_filters(key: str, **overrides):
    """Put a page's filters back to their defaults (optionally presetting some); safe inside button callbacks."""
    for name, value in FILTER_DEFAULTS.items():
        st.session_state[f"{key}-{name}"] = overrides.get(name, value)


def filter_controls(key: str, show_genre=True) -> Filters:
    """The shared filter widgets (in an expander) used by the recommendation pages. Defaults live in session state,
    not in the widget calls, so buttons can preset a filter without conflicting with the widget."""
    _, genres, _, _ = options()
    for name, value in FILTER_DEFAULTS.items():
        st.session_state.setdefault(f"{key}-{name}", value)
    with st.expander("Filters", icon=":material/tune:"):
        c1, c2, c3 = st.columns(3)
        g = c1.multiselect("Only these genres", genres, key=f"{key}-genres") if show_genre else ()
        min_rating = c2.slider("Minimum rating", 0.0, 5.0, step=0.1, key=f"{key}-rating")
        min_reviews = c3.select_slider("Minimum reviews", [0, 5, 10, 50, 100, 500, 1000, 5000], key=f"{key}-reviews")
        c4, c5, c6 = st.columns(3)
        price = c4.select_slider("Maximum price", ["Any", "Free", 300, 500, 700, 1000, 1500, 2500], key=f"{key}-price",
                                 format_func=lambda v: v if isinstance(v, str) else f"₹{v:,}")
        hrs = c5.select_slider("Maximum length", ["Any", 2, 4, 6, 8, 12, 20, 40], key=f"{key}-hours",
                               format_func=lambda v: v if isinstance(v, str) else f"{v} hours",
                               help="Only listening time known for about a third of books; a length limit leaves "
                                    "out books of unknown length.")
        with c6:
            include_unrated = st.toggle("Include unrated books", key=f"{key}-unrated")
            gems = st.toggle("Hidden gems only", key=f"{key}-gems",
                             help="Rated 4.5+ by 5–65 reviewers: loved, but not yet widely found.")
    return Filters(tuple(g), min_rating, min_reviews,
                   None if price == "Any" else (0 if price == "Free" else price),
                   None if hrs == "Any" else hrs, include_unrated, gems)


# ---------------------------------------------------------------- explanations
def _genre_set(s):
    return {g.strip() for g in s.split("|")} if isinstance(s, str) else set()


def explain(rec_df: pd.DataFrame, query_ids) -> dict:
    """{book index: one-line reason} from what a recommendation shares with the query books."""
    b = books()
    q = b.iloc[list(query_ids)]
    q_genres = set().union(*[_genre_set(s) for s in q["genres"]]) if len(q) else set()
    q_broad = set(q["primary_genre"]) - {config.UNKNOWN_GENRE}
    q_authors, q_clusters = set(q["author"]), set(q["cluster"])
    out = {}
    for idx, r in rec_df.iterrows():
        reasons = []
        if r["author"] in q_authors:
            reasons.append("same author")
        shared = sorted(_genre_set(r["genres"]) & q_genres)
        if shared:
            reasons.append("shares " + ", ".join(f"“{g}”" for g in shared[:2]) + ("…" if len(shared) > 2 else ""))
        elif r["primary_genre"] in q_broad:
            reasons.append(f"also {r['primary_genre']}")
        if r["cluster"] in q_clusters and not shared:
            reasons.append("same cluster of similar books")
        if r.get("hidden_gem"):
            reasons.append("highly rated, little known")
        if not reasons:
            reasons.append("similar description")
        out[idx] = "Why: " + "; ".join(reasons)
    return out


def explain_profile(rec_df: pd.DataFrame, genres, authors, liked_ids) -> dict:
    base = explain(rec_df, liked_ids) if liked_ids else {}
    out = {}
    for idx, r in rec_df.iterrows():
        reasons = []
        if r["author"] in authors:
            reasons.append("an author you like")
        if r["primary_genre"] in genres:
            reasons.append(f"{r['primary_genre']}")
        txt = base.get(idx, "").removeprefix("Why: ")
        if txt and txt != "similar description":
            reasons.append(txt)
        if not reasons and r.get("hidden_gem"):
            reasons.append("highly rated, little known")
        out[idx] = "Why: " + "; ".join(reasons or ["matches your taste profile"])
    return out
