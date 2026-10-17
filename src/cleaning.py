"""Clean each raw file, de-duplicate it, and merge the two into one row per book."""
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config

_JUNK = re.compile("|".join(config.JUNK_DESCRIPTION_PATTERNS), re.IGNORECASE)
_BLURB = re.compile(config.BLURB_TRIGGER, re.IGNORECASE)
_GENRE_RULES = [(group, re.compile("|".join(r"\b" + re.escape(k) for k in keys), re.IGNORECASE))
                for group, keys in config.GENRE_RULES]

RAW_COLUMNS = {
    "Book Name": "title", "Author": "author", "Rating": "rating", "Number of Reviews": "num_reviews",
    "Price": "price", "Description": "description", "Listening Time": "listening_time",
    "Ranks and Genre": "ranks_and_genre",
}


# ---------------------------------------------------------------- field parsers
def normalise_text(s) -> str:
    """Unicode-normalise, unify quotes, collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[’‘`´]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    return re.sub(r"\s+", " ", s).strip()


def match_key(s) -> str:
    """Key used to recognise the same title/author across rows: lower case, letters and digits only."""
    s = normalise_text(s).lower()
    s = re.sub(r"^the ", "", s)
    return re.sub(r"[^0-9a-zऀ-ॿ]+", "", s)


def is_award_blurb(s: str) -> bool:
    """True for a short line that only announces awards or bestseller status (see config.BLURB_WORDS)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    if len(s) >= config.BLURB_MAX_CHARS or not _BLURB.search(s):
        return False
    words = [t for t in re.findall(r"[a-z']+", s.lower())
             if len(t) >= 3 and t not in ENGLISH_STOP_WORDS and t not in config.BLURB_WORDS]
    return len(words) < config.BLURB_MIN_CONTENT_WORDS


def clean_description(s):
    """Return the description, or NaN when it is missing, a scraping failure, an award blurb or too short to use."""
    if pd.isna(s):
        return np.nan
    s = normalise_text(s)
    if len(s) < config.MIN_DESCRIPTION_CHARS or _JUNK.search(s) or is_award_blurb(s):
        return np.nan
    return s


def parse_listening_minutes(s):
    """'10 hours and 54 minutes' -> 654.0; '-1' / 'Not yet known' -> NaN."""
    if pd.isna(s) or str(s).strip() in config.MISSING_SENTINELS:
        return np.nan
    s = str(s).lower()
    hours = re.search(r"(\d+)\s*hour", s)
    minutes = re.search(r"(\d+)\s*minute", s)
    if not hours and not minutes:
        return np.nan
    return float((int(hours.group(1)) if hours else 0) * 60 + (int(minutes.group(1)) if minutes else 0))


def parse_ranks(s):
    """',#2 in Audible Audiobooks & Originals (See Top 100 ...),#1 in Meditation (...),#1 in Self-Esteem'
    -> (overall Audible rank, [(rank, fine genre), ...]). Genre names can contain commas ("Mind, Body & Spirit"),
    so entries are split on ',#' rather than ','."""
    if pd.isna(s) or str(s).strip() in config.MISSING_SENTINELS:
        return np.nan, []
    overall, genres = np.nan, []
    for part in str(s).split(",#"):
        m = re.match(r"#?([\d,]+) in (.+)", part.strip(",# "))
        if not m:
            continue
        rank = int(m.group(1).replace(",", ""))
        name = re.sub(r"\s*\((?:See Top 100 in .*|Audible Audiobooks & Originals|Books)\)\s*$", "", m.group(2))
        name = normalise_text(name)
        if name == config.OVERALL_RANK_CATEGORY:
            overall = rank
        else:
            genres.append((rank, name))
    return overall, genres


def broad_genre(fine: str) -> str:
    """Map one of the 877 fine genres to a broad genre (first matching rule wins)."""
    for group, pattern in _GENRE_RULES:
        if pattern.search(fine):
            return group
    return config.OTHER_GENRE


def primary_genre(fine_genres: list) -> str:
    """A book's broad genre = the most common broad genre among its fine genres; ties go to the genre it ranks best in."""
    if not fine_genres:
        return config.UNKNOWN_GENRE
    groups = [broad_genre(g) for _, g in sorted(fine_genres)]
    known = [g for g in groups if g != config.OTHER_GENRE] or groups
    counts = pd.Series(known).value_counts()
    best = counts[counts == counts.max()].index
    return next(g for g in known if g in best)


# ---------------------------------------------------------------- per-file cleaning
def standardise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename columns, fix types, turn sentinels into NaN, parse the text fields."""
    df = df.rename(columns=RAW_COLUMNS).copy()
    df["title"] = df["title"].map(normalise_text)
    df["author"] = df["author"].map(normalise_text)
    # -1 rating always comes with a missing review count: the book has not been rated yet.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").where(lambda r: r >= 0)
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce")
    df.loc[df["rating"].isna(), "num_reviews"] = 0
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for col in ["description", "listening_time", "ranks_and_genre"]:
        if col not in df:
            df[col] = np.nan
    df["description"] = df["description"].map(clean_description)
    df["listening_minutes"] = df["listening_time"].map(parse_listening_minutes)
    parsed = df["ranks_and_genre"].map(parse_ranks)
    df["audible_rank"] = parsed.str[0]
    df["fine_genres"] = parsed.str[1]
    df["source"] = source
    df["title_key"] = df["title"].map(match_key)
    df["author_key"] = df["author"].map(match_key)
    return df.drop(columns=["listening_time", "ranks_and_genre"])


def completeness(df: pd.DataFrame) -> pd.Series:
    """How much usable information a row carries; used to pick the best copy of a duplicated book."""
    return (df["description"].notna() * 4 + df["fine_genres"].map(bool) * 2 + df["listening_minutes"].notna()
            + df["rating"].notna())


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates, then keep the most complete (then most reviewed) copy of each title/author."""
    n0 = len(df)
    exact_cols = ["title", "author", "rating", "num_reviews", "price", "description", "listening_minutes",
                  "audible_rank"]
    df = df.drop_duplicates(subset=exact_cols)
    n_exact = n0 - len(df)
    df = (df.assign(_score=completeness(df))
            .sort_values(["_score", "num_reviews"], ascending=False)
            .drop_duplicates(["title_key", "author_key"])
            .drop(columns="_score"))
    return df, {"rows_in": n0, "exact_duplicates": n_exact, "same_book_duplicates": n0 - n_exact - len(df),
                "rows_out": len(df)}


# ---------------------------------------------------------------- merging
def unify_author_spellings(cat: pd.DataFrame, adv: pd.DataFrame) -> int:
    """Where the same title appears under two near-identical author spellings (across or within the files),
    rewrite the rarer spelling's author_key to the other so both rows merge. Returns the number of keys rewritten."""
    both = pd.concat([cat[["title_key", "author_key"]], adv[["title_key", "author_key"]]])
    rename = {}
    for title, grp in both.drop_duplicates().groupby("title_key"):
        keys = list(grp["author_key"])
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a != b and SequenceMatcher(None, a, b).ratio() >= config.AUTHOR_FUZZY_MATCH:
                    rename[(title, b)] = a
    if not rename:
        return 0
    for df in (cat, adv):
        df["author_key"] = [rename.get((t, a), a) for t, a in zip(df["title_key"], df["author_key"])]
    return len(rename)


def merge_catalogs(cat: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """Outer-join on (title, author). The advanced file is the later, richer scrape (review counts are higher for
    the same book), so its values win; the basic catalog fills the gaps and contributes books the other lacks."""
    key = ["title_key", "author_key"]
    shared = ["title", "author", "rating", "num_reviews", "price"]
    m = adv.merge(cat[key + shared], on=key, how="outer", suffixes=("", "_cat"), indicator=True)
    # A book rated in the older scrape but listed as unrated in the newer one keeps its older rating and reviews.
    older_rating = m["rating"].isna() & m["rating_cat"].notna()
    m.loc[older_rating, ["rating", "num_reviews"]] = m.loc[older_rating, ["rating_cat", "num_reviews_cat"]].values
    for col in shared:
        m[col] = m[col].combine_first(m[f"{col}_cat"])
    m["in_advanced"] = m["_merge"].isin(["both", "left_only"])
    m["in_catalog"] = m["_merge"].isin(["both", "right_only"])
    m["fine_genres"] = m["fine_genres"].map(lambda v: v if isinstance(v, list) else [])
    return m.drop(columns=[f"{c}_cat" for c in shared] + ["_merge", "source"])


# ---------------------------------------------------------------- derived fields
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rated"] = df["rating"].notna()
    df["num_reviews"] = df["num_reviews"].fillna(0).astype(int)
    df["price_corrected"] = df["price"] > config.PRICE_DECIMAL_SHIFT_ABOVE
    df.loc[df["price_corrected"], "price"] = (df.loc[df["price_corrected"], "price"] / 10).round(1)
    df["is_free"] = df["price"].eq(0)
    df["has_description"] = df["description"].notna()
    df["genres"] = df["fine_genres"].map(lambda gs: " | ".join(g for _, g in sorted(gs)))
    df["num_genres"] = df["fine_genres"].map(len)
    df["best_genre_rank"] = df["fine_genres"].map(lambda gs: min(r for r, _ in gs) if gs else np.nan)
    df["primary_genre"] = df["fine_genres"].map(primary_genre)
    df["genre_source"] = np.where(df["primary_genre"].eq(config.UNKNOWN_GENRE), "unknown", "listed")
    df["short_title"] = df["title"].str.split(r":| - ", n=1, regex=True).str[0].str.strip()

    # Bayesian weighted rating: shrink each rating towards the catalogue mean by how few reviews it has.
    rated = df["is_rated"]
    c = df.loc[rated, "rating"].mean()
    v, m = df["num_reviews"], config.BAYES_MIN_REVIEWS
    df["weighted_rating"] = np.where(rated, (v / (v + m)) * df["rating"].fillna(0) + (m / (v + m)) * c, np.nan)
    df["popularity"] = np.log1p(df["num_reviews"])

    gem_max = df.loc[rated, "num_reviews"].quantile(config.HIDDEN_GEM_MAX_REVIEW_QUANTILE)
    df["hidden_gem"] = (rated & df["rating"].ge(config.HIDDEN_GEM_MIN_RATING)
                        & df["num_reviews"].between(config.HIDDEN_GEM_MIN_REVIEWS, max(gem_max,
                                                                                       config.HIDDEN_GEM_MIN_REVIEWS)))
    return df


FINAL_COLUMNS = [
    "book_id", "title", "short_title", "author", "rating", "num_reviews", "is_rated", "weighted_rating",
    "popularity", "price", "price_corrected", "is_free", "listening_minutes", "description", "has_description", "primary_genre",
    "genre_source", "genres", "num_genres", "best_genre_rank", "audible_rank", "hidden_gem", "in_catalog",
    "in_advanced",
]


def run() -> tuple[pd.DataFrame, dict]:
    """Raw CSVs -> one clean row per book, plus a report of every count."""
    raw_cat = pd.read_csv(config.CATALOG_CSV)
    raw_adv = pd.read_csv(config.ADVANCED_CSV)
    cat, adv = standardise(raw_cat, "catalog"), standardise(raw_adv, "advanced")

    report = {
        "raw": {
            "catalog": {"rows": len(raw_cat), "columns": list(raw_cat.columns)},
            "advanced": {"rows": len(raw_adv), "columns": list(raw_adv.columns)},
        },
        "missing_raw": {
            "catalog_unrated_(rating_-1)": int((raw_cat["Rating"] < 0).sum()),
            "catalog_price_missing": int(raw_cat["Price"].isna().sum()),
            "advanced_unrated_(rating_-1)": int((raw_adv["Rating"] < 0).sum()),
            "advanced_description_missing": int(raw_adv["Description"].isna().sum()),
            "advanced_description_junk": int((raw_adv["Description"].notna() & adv["description"].isna()).sum()),
            "advanced_description_award_blurb_only": int(raw_adv["Description"].dropna().map(normalise_text)
                                                         .map(is_award_blurb).sum()),
            "advanced_listening_time_missing": int(adv["listening_minutes"].isna().sum()),
            "advanced_ranks_and_genre_missing": int(raw_adv["Ranks and Genre"].astype(str).eq("-1").sum()),
        },
    }
    report["author_spellings_unified"] = unify_author_spellings(cat, adv)
    cat, report["dedup_catalog"] = deduplicate(cat)
    adv, report["dedup_advanced"] = deduplicate(adv)

    books = add_derived(merge_catalogs(cat, adv))
    books = books.sort_values(["num_reviews", "title"], ascending=[False, True]).reset_index(drop=True)
    books["book_id"] = np.arange(len(books))
    report["merge"] = {
        "books": len(books),
        "in_both_files": int((books["in_catalog"] & books["in_advanced"]).sum()),
        "only_catalog": int((books["in_catalog"] & ~books["in_advanced"]).sum()),
        "only_advanced": int((~books["in_catalog"] & books["in_advanced"]).sum()),
    }
    report["final_coverage"] = {
        "rated": int(books["is_rated"].sum()),
        "with_description": int(books["has_description"].sum()),
        "with_listed_genre": int(books["genre_source"].eq("listed").sum()),
        "with_listening_time": int(books["listening_minutes"].notna().sum()),
        "price_missing": int(books["price"].isna().sum()),
        "price_decimal_shift_corrected": int(books["price_corrected"].sum()),
        "fine_genres": int(books["fine_genres"].explode().dropna().map(lambda t: t[1]).nunique()),
    }
    return books[FINAL_COLUMNS + ["fine_genres"]], report
