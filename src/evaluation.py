"""Offline evaluation of the recommenders.

The data has no per-user ratings, so "did the user like it" cannot be measured. Two proxies are used instead:

1. Ranking quality (precision@k, recall@k, nDCG@k). A recommendation for book q is *relevant* when it shares at least
   one of q's fine Audible genres (876 of them, e.g. "Space Opera Science Fiction"); *strictly relevant* when it
   shares two. Queries and candidates are books with listed genres. Features built from the fine genres are the
   label source, so configurations that use them are marked as an optimistic upper bound.
2. Rating prediction (RMSE, MAE). Predict a held-out book's rating from the ratings of its nearest neighbours in the
   feature space, against global-, genre- and author-mean baselines.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

import config
from src import recommenders as R


def genre_matrix(books: pd.DataFrame) -> sparse.csr_matrix:
    """Binary book x fine-genre matrix; (M @ M[q].T) counts the genres each book shares with book q."""
    from sklearn.preprocessing import MultiLabelBinarizer
    sets = [[g.strip() for g in s.split("|") if g.strip()] if isinstance(s, str) else [] for s in books["genres"]]
    return MultiLabelBinarizer(sparse_output=True).fit_transform(sets).astype(np.float32).tocsr()


def ranking_metrics(ranked: np.ndarray, rel: np.ndarray, strict: np.ndarray, k: int) -> dict:
    hits = rel[ranked[:k]]
    gains = hits / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(k, rel.sum()) + 2))).sum()
    return {"precision": hits.mean() if len(hits) else 0.0,
            "recall": hits.sum() / rel.sum() if rel.sum() else np.nan,
            "ndcg": gains.sum() / ideal if ideal else np.nan,
            "strict_precision": strict[ranked[:k]].mean() if len(hits) else 0.0}


def evaluate_rankers(books: pd.DataFrame, rankers: dict, k=config.EVAL_K, n_queries=config.EVAL_QUERIES,
                     seed=config.SEED) -> pd.DataFrame:
    """rankers: name -> function(query_index, pool_mask) -> score array over all books.
    Returns one row per ranker with mean metrics, average rating of what it recommends, and catalogue coverage."""
    M = genre_matrix(books)
    pool = books["genre_source"].eq("listed").to_numpy()
    queries_all = np.where(pool & books["has_description"].to_numpy())[0]
    rng = np.random.RandomState(seed)
    queries = rng.choice(queries_all, size=min(n_queries, len(queries_all)), replace=False)

    # Same-book editions are never relevant targets (the recommender removes them too).
    st = books["short_title"].str.lower().to_numpy()
    au = books["author"].to_numpy()
    rating = books["rating"].to_numpy(dtype=float)

    cases = []
    for q in queries:
        mask = pool & ~((st == st[q]) & (au == au[q]))
        mask[q] = False
        shared = np.asarray((M @ M[q].T).todense()).ravel() * mask
        if (shared >= 1).any():
            cases.append((q, mask, shared >= 1, shared >= 2))

    rows = []
    for name, ranker in rankers.items():
        metrics, recommended, ratings = [], set(), []
        for q, mask, rel, strict in cases:
            scores = ranker(q, mask).astype(float)
            scores[~mask] = -np.inf
            ranked = np.argsort(-scores, kind="stable")[:k]
            metrics.append(ranking_metrics(ranked, rel, strict, k))
            recommended.update(ranked.tolist())
            ratings.extend(rating[ranked][~np.isnan(rating[ranked])].tolist())
        m = pd.DataFrame(metrics).mean()
        rows.append({"model": name, f"precision@{k}": m["precision"], f"recall@{k}": m["recall"],
                     f"ndcg@{k}": m["ndcg"], f"strict_precision@{k}": m["strict_precision"],
                     "avg_rating_of_recs": float(np.mean(ratings)) if ratings else np.nan,
                     "catalogue_coverage": len(recommended) / pool.sum(), "queries": len(metrics)})
    return pd.DataFrame(rows)


FEATURE_CONFIGS = {
    "text (title + description)": {"text": 1.0},
    "text + author": {"text": 1.0, "author": config.AUTHOR_WEIGHT},
    "text + broad genre": {"text": 1.0, "broad": config.GENRE_WEIGHT},
    "text + broad genre + author": {"text": 1.0, "broad": config.GENRE_WEIGHT, "author": config.AUTHOR_WEIGHT},
    "all, incl. fine genres": R.PRODUCTION_WEIGHTS,
}


def similarity_fn(C: sparse.csr_matrix):
    return lambda q: np.asarray((C @ C[q].T).todense()).ravel()


def build_rankers(books, blocks, quality, clusters, text_clusters, hybrid_weights) -> pd.DataFrame:
    """Every model to compare, as name -> (ranker, uses fine genres?). Genre-blind variants use only the text
    block and text-assigned clusters, so they cannot see the labels the evaluation scores against."""
    rng = np.random.RandomState(config.SEED)
    text_sim = similarity_fn(R.combine(blocks, {"text": 1.0}))
    full_sim = similarity_fn(R.combine(blocks, R.PRODUCTION_WEIGHTS))
    w = hybrid_weights

    def cluster_ranker(labels, sim):
        return lambda q, m: (labels == labels[q]) * (1 + quality) + 1e-3 * sim(q)

    def hybrid_ranker(labels, sim):
        return lambda q, m: w["content"] * sim(q) + w["cluster"] * (labels == labels[q]) + w["quality"] * quality

    rankers = {
        "Random": (lambda q, m: rng.rand(len(books)), False),
        "Most popular (weighted rating)": (lambda q, m: quality.copy(), False),
        "Content-based (genre-blind)": (lambda q, m: text_sim(q), False),
        "Clustering-based (genre-blind)": (cluster_ranker(text_clusters, text_sim), False),
        "Hybrid (genre-blind)": (hybrid_ranker(text_clusters, text_sim), False),
        "Content-based (full features)": (lambda q, m: full_sim(q), True),
        "Clustering-based (full features)": (cluster_ranker(clusters, full_sim), True),
        "Hybrid (full features)": (hybrid_ranker(clusters, full_sim), True),
    }
    return rankers


def feature_ablation(books, blocks) -> pd.DataFrame:
    """Which feature combination recommends best? Content-based model, one row per feature configuration."""
    rankers = {}
    for name, weights in FEATURE_CONFIGS.items():
        sim = similarity_fn(R.combine(blocks, weights))
        rankers[name] = (lambda sim: lambda q, m: sim(q))(sim)
    res = evaluate_rankers(books, rankers).rename(columns={"model": "features"})
    res["uses_fine_genres"] = res["features"].str.contains("fine")
    return res


# ---------------------------------------------------------------- hybrid weight tuning
def tune_hybrid(books, blocks, clusters, quality, grid=None, k=config.EVAL_K):
    """Grid-search the hybrid weights on a validation query set (different seed from the test queries). The
    objective is nDCG@k with a small reward for recommending well-rated books, so quality is not tuned to zero."""
    prod = R.combine(blocks, R.PRODUCTION_WEIGHTS)
    grid = grid or [(c, cl, 1 - c - cl) for c in (0.5, 0.6, 0.7, 0.8, 0.9) for cl in (0.0, 0.05, 0.1, 0.2)
                    if 1 - c - cl >= 0]
    rows = []
    for c, cl, qu in grid:
        f = (lambda c, cl, qu: lambda q, m: c * np.asarray((prod @ prod[q].T).todense()).ravel()
             + cl * (clusters == clusters[q]) + qu * quality)(c, cl, qu)
        r = evaluate_rankers(books, {"h": f}, k=k, n_queries=300, seed=config.SEED + 1).iloc[0]
        rows.append({"content": c, "cluster": cl, "quality": round(qu, 2), f"ndcg@{k}": r[f"ndcg@{k}"],
                     f"precision@{k}": r[f"precision@{k}"], "avg_rating_of_recs": r["avg_rating_of_recs"]})
    res = pd.DataFrame(rows)
    # Rating of recommendations on a 0-1 scale (ratings mostly lie in 3.5-5.0) adds up to 0.1 to the objective.
    res["objective"] = res[f"ndcg@{k}"] + 0.1 * ((res["avg_rating_of_recs"] - 3.5) / 1.5).clip(0, 1)
    best = res.loc[res["objective"].idxmax()]
    return {"content": float(best["content"]), "cluster": float(best["cluster"]),
            "quality": float(best["quality"])}, res


# ---------------------------------------------------------------- rating prediction (RMSE)
def rating_prediction(books: pd.DataFrame, blocks: dict, k=config.RATING_KNN_NEIGHBOURS) -> pd.DataFrame:
    """Hold out 20% of rated books (with >= 5 reviews, so the rating means something) and predict their ratings."""
    rated = np.where(books["is_rated"].to_numpy() & (books["num_reviews"].to_numpy() >= 5))[0]
    train, test = train_test_split(rated, test_size=config.TEST_SIZE, random_state=config.SEED)
    y = books["rating"].to_numpy(dtype=float)
    g_mean = y[train].mean()

    def group_mean(col):
        means = pd.Series(y[train]).groupby(books[col].to_numpy()[train]).agg(["mean", "size"])
        # shrink small groups towards the global mean
        shrunk = (means["mean"] * means["size"] + g_mean * 5) / (means["size"] + 5)
        return np.array([shrunk.get(v, g_mean) for v in books[col].to_numpy()[test]])

    def knn(weights):
        C = R.combine(blocks, weights)
        S = (C[test] @ C[train].T).toarray()
        top = np.argsort(-S, axis=1)[:, :k]
        s = np.take_along_axis(S, top, 1)
        r = y[train][top]
        pred = (s * r).sum(1) / np.where(s.sum(1) > 0, s.sum(1), 1)
        return np.where(s.sum(1) > 0, pred, g_mean)

    preds = {
        "Global mean (baseline)": np.full(len(test), g_mean),
        "Genre mean": group_mean("primary_genre"),
        "Author mean": group_mean("author"),
        "k-NN content (text only)": knn({"text": 1.0}),
        "k-NN content (all features)": knn(R.PRODUCTION_WEIGHTS),
    }
    preds["Hybrid (k-NN + author mean)"] = (preds["k-NN content (all features)"] + preds["Author mean"]) / 2
    rows = []
    for name, p in preds.items():
        err = p - y[test]
        rows.append({"model": name, "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())})
    return pd.DataFrame(rows).assign(test_books=len(test), train_books=len(train))
