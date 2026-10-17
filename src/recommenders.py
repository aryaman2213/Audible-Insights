"""Feature space, clustering and the recommendation models (content-based, clustering-based, hybrid).

Everything the app needs is in one `Artifacts` object, saved with joblib by train.py.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

import config
from src import nlp

BLOCKS = ("text", "broad", "fine", "author")
TITLE_ONLY_CLUSTER_CONFIDENCE = 0.3   # a cluster guessed from a title alone is a weak signal
PRODUCTION_WEIGHTS = {"text": config.TEXT_WEIGHT, "broad": config.GENRE_WEIGHT, "fine": config.GENRE_WEIGHT,
                      "author": config.AUTHOR_WEIGHT}


# ---------------------------------------------------------------- feature blocks
def build_blocks(books: pd.DataFrame) -> tuple[dict, TfidfVectorizer]:
    """One sparse, row-normalised matrix per feature block (all n_books rows; a book without that information has
    an all-zero row in the block)."""
    vec = nlp.make_vectorizer()
    text = vec.fit_transform(nlp.book_text(books))

    genres = sorted(g for g in books["primary_genre"].unique() if g != config.UNKNOWN_GENRE)
    gidx = {g: i for i, g in enumerate(genres)}
    rows = [i for i, g in enumerate(books["primary_genre"]) if g in gidx]
    broad = sparse.csr_matrix((np.ones(len(rows)), (rows, [gidx[books["primary_genre"].iat[i]] for i in rows])),
                              shape=(len(books), len(genres)))

    fine_vec = TfidfVectorizer(tokenizer=lambda s: [g.strip() for g in s.split("|") if g.strip()],
                               token_pattern=None, lowercase=False, sublinear_tf=True)
    fine = fine_vec.fit_transform(books["genres"].fillna(""))

    authors = books["author"].str.lower().str.replace(r"[^a-z]+", "", regex=True)
    codes, uniques = pd.factorize(authors)
    author = sparse.csr_matrix((np.ones(len(books)), (np.arange(len(books)), codes)),
                               shape=(len(books), len(uniques)))

    blocks = {name: normalize(m.astype(np.float32).tocsr()) for name, m in
              [("text", text), ("broad", broad), ("fine", fine), ("author", author)]}
    return blocks, vec


def combine(blocks: dict, weights: dict) -> sparse.csr_matrix:
    """Concatenate weighted blocks so that cosine(a, b) = sum_w weight * block-cosine / norm. sqrt(weight) is used
    because the dot product squares the scale."""
    parts = [blocks[b] * np.sqrt(w) for b, w in weights.items() if w > 0]
    return normalize(sparse.hstack(parts).tocsr())


# ---------------------------------------------------------------- clustering
@dataclass
class ClusterResult:
    labels: np.ndarray                 # KMeans cluster per book (all books)
    text_labels: np.ndarray            # nearest cluster by text alone, for every book (genre-blind evaluation)
    assigned_by: np.ndarray            # "fitted" | "description" | "title": how each book got its cluster
    fit_mask: np.ndarray               # books the clusters were learned on (listed genre)
    embedding: np.ndarray              # SVD embedding of the fit books, row-normalised
    k: int
    silhouette_by_k: dict
    dbscan: list                       # one dict per eps tried
    names: dict                        # cluster id -> short name
    top_terms: dict                    # cluster id -> list of terms
    sizes: dict                        # cluster id -> number of books (all books)
    map_xy: np.ndarray                 # 2-D t-SNE coordinates (NaN outside fit_mask)


def fit_clusters(books: pd.DataFrame, blocks: dict, vec: TfidfVectorizer, weights=None,
                 k_range=config.KMEANS_K_RANGE, with_dbscan=True, with_map=True) -> ClusterResult:
    """Two stages.
    1. Learn clusters where the information is complete: books with a listed genre, in an SVD topic space of
       text + genres. KMeans picks k by silhouette; DBSCAN is run on the same space for comparison.
    2. Every other book (description only, or title only) joins the cluster whose text centroid it is closest to,
       in a text-only SVD space. Clustering everything together instead leaves the genre-less books in one leftover
       cluster, because their feature vectors lack the genre blocks."""
    weights = weights or config.CLUSTER_WEIGHTS
    fit = books["genre_source"].eq("listed").to_numpy()
    X = combine(blocks, weights)[fit]
    svd = TruncatedSVD(n_components=config.SVD_COMPONENTS, random_state=config.SEED).fit(X)
    emb = normalize(svd.transform(X))

    sil, models = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(emb)
        sil[k] = float(silhouette_score(emb, km.labels_))
        models[k] = km
    best_k = max(sil, key=sil.get)
    fit_labels = models[best_k].labels_

    dbscan = []
    if with_dbscan:
        for eps in config.DBSCAN_EPS_GRID:
            lab = DBSCAN(eps=eps, min_samples=config.DBSCAN_MIN_SAMPLES).fit(emb).labels_
            n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
            clustered = lab != -1
            s = (float(silhouette_score(emb[clustered], lab[clustered]))
                 if n_clusters >= 2 and clustered.sum() > n_clusters else None)
            dbscan.append({"eps": eps, "clusters": n_clusters, "noise_share": round(float(1 - clustered.mean()), 4),
                           "largest_cluster_share": round(float(np.bincount(lab[clustered]).max() / len(lab)), 4)
                           if clustered.any() else 0.0,
                           "silhouette_clustered_only": None if s is None else round(s, 4),
                           "agreement_with_kmeans_ari": round(float(adjusted_rand_score(fit_labels, lab)), 4)})

    # Stage 2: text-only space for everyone; assign non-fit books to the nearest cluster text centroid.
    text = blocks["text"]
    tsvd = TruncatedSVD(n_components=config.SVD_COMPONENTS, random_state=config.SEED).fit(text)
    temb = normalize(tsvd.transform(text))
    centroids = normalize(np.vstack([temb[fit][fit_labels == c].mean(0) for c in range(best_k)]))
    text_labels = (temb @ centroids.T).argmax(1)
    labels = text_labels.copy()
    labels[fit] = fit_labels
    assigned_by = np.where(fit, "fitted", np.where(books["has_description"], "description", "title"))

    # Name each cluster by its dominant broad genre and its most characteristic terms.
    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(text[fit].mean(0)).ravel()
    names, top_terms = {}, {}
    for c in range(best_k):
        members = np.where(fit)[0][fit_labels == c]
        lift = np.asarray(text[members].mean(0)).ravel() - overall
        top = [t for t in terms[np.argsort(-lift)] if len(t) > 2][:12]
        top_terms[c] = top
        lead = books["primary_genre"].iloc[members].value_counts().index[0]
        names[c] = f"{lead} · {', '.join(top[:3])}"

    map_xy = np.full((len(books), 2), np.nan)
    if with_map:
        map_xy[fit] = TSNE(n_components=2, perplexity=35, init="pca", random_state=config.SEED,
                           metric="cosine").fit_transform(emb)
    return ClusterResult(labels=labels, text_labels=text_labels, assigned_by=assigned_by, fit_mask=fit, embedding=emb, k=best_k,
                         silhouette_by_k=sil, dbscan=dbscan, names=names, top_terms=top_terms,
                         sizes={int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))},
                         map_xy=map_xy)


# ---------------------------------------------------------------- the models
def quality_score(books: pd.DataFrame) -> np.ndarray:
    """Weighted rating scaled to 0-1; unrated books get the catalogue mean so they are neither boosted nor buried."""
    w = books["weighted_rating"].to_numpy(dtype=float)
    w = np.where(np.isnan(w), np.nanmean(w), w)
    return (w - w.min()) / (w.max() - w.min())


@dataclass
class Recommender:
    """Scores every book for a query vector. Methods return a score array over all books; `top` turns scores into
    a ranked list while removing the query books and other editions of them."""
    books: pd.DataFrame
    content: sparse.csr_matrix                    # combined, normalised feature matrix
    clusters: np.ndarray
    quality: np.ndarray
    cluster_confidence: np.ndarray = None         # 1.0 if the cluster came from genres/description, lower if title only
    hybrid_weights: dict = field(default_factory=lambda: {"content": 0.8, "cluster": 0.1, "quality": 0.1})

    # --- scores
    def content_scores(self, q: sparse.csr_matrix) -> np.ndarray:
        return np.asarray((self.content @ q.T).todense()).ravel()

    def membership(self, cluster_ids) -> np.ndarray:
        """How firmly each book belongs to the query's cluster(s): 0 outside, its assignment confidence inside."""
        conf = self.cluster_confidence if self.cluster_confidence is not None else np.ones(len(self.clusters))
        return np.isin(self.clusters, list(cluster_ids)) * conf

    def cluster_scores(self, cluster_ids) -> np.ndarray:
        """Clustering-based: books in the query's cluster(s), ranked by quality (weighted rating). Books placed in the
        cluster from their title alone rank below those placed by genre or description."""
        return self.membership(cluster_ids) * (1.0 + self.quality)

    def hybrid_scores(self, q: sparse.csr_matrix, cluster_ids, weights=None) -> np.ndarray:
        w = weights or self.hybrid_weights
        return (w["content"] * self.content_scores(q) + w["cluster"] * self.membership(cluster_ids)
                + w["quality"] * self.quality)

    # --- queries
    def query_vector(self, ids) -> sparse.csr_matrix:
        return normalize(sparse.csr_matrix(self.content[list(ids)].mean(0)))

    def scores_for_books(self, ids, method="hybrid", weights=None) -> np.ndarray:
        q = self.query_vector(ids)
        cl = set(self.clusters[list(ids)])
        if method == "content":
            return self.content_scores(q)
        if method == "cluster":
            return self.cluster_scores(cl) + 1e-3 * self.content_scores(q)   # similarity only breaks ties
        return self.hybrid_scores(q, cl, weights)

    def top(self, scores: np.ndarray, exclude_ids=(), n=10, mask=None) -> np.ndarray:
        scores = scores.astype(float).copy()
        if mask is not None:
            scores[~mask] = -np.inf
        if len(exclude_ids):
            ex = self.books.iloc[list(exclude_ids)]
            same_book = (self.books["short_title"].str.lower().isin(ex["short_title"].str.lower())
                         & self.books["author"].isin(ex["author"])).to_numpy()
            scores[same_book] = -np.inf
            scores[list(exclude_ids)] = -np.inf
        n_valid = int(np.isfinite(scores).sum())
        idx = np.argpartition(-scores, min(n, n_valid) - 1)[:min(n, n_valid)] if n_valid else np.array([], int)
        return idx[np.argsort(-scores[idx])]

    def recommend(self, ids, method="hybrid", n=10, mask=None) -> pd.DataFrame:
        scores = self.scores_for_books(ids, method)
        idx = self.top(scores, ids, n, mask)
        return self.books.iloc[idx].assign(score=scores[idx])

    def recommend_for_profile(self, liked_ids=(), genres=(), authors=(), n=10, mask=None,
                              sub_genres=()) -> pd.DataFrame:
        """Cold-start / preference query: any mix of liked books, broad genres, fine sub-genres and authors.
        With only genres, this ranks books of those genres by quality and cluster fit."""
        b = self.books
        pref = np.zeros(len(b), dtype=bool)
        if sub_genres:
            fine = b["genres"].fillna("").str.split(r" \| ")
            pref |= fine.map(lambda gs: any(g in sub_genres for g in gs)).to_numpy()
        if genres:
            pref |= b["primary_genre"].isin(genres).to_numpy()
        if authors:
            pref |= b["author"].isin(authors).to_numpy()
        seed_ids = list(liked_ids)
        if not seed_ids and pref.any():
            # Seed with the best-rated books matching the preferences: their content is what the user asked for.
            seed_ids = list(b.index[pref][np.argsort(-self.quality[pref])][:15])
        if not seed_ids:
            scores = self.quality.copy()
        else:
            scores = self.scores_for_books(seed_ids, "hybrid")
        scores = scores + 0.5 * pref                  # explicit preferences come first
        idx = self.top(scores, liked_ids, n, mask)
        return b.iloc[idx].assign(score=scores[idx])


@dataclass
class Artifacts:
    books: pd.DataFrame
    recommender: Recommender
    clusters: ClusterResult
    vectorizer: TfidfVectorizer
    weights: dict


def build_artifacts(books: pd.DataFrame, hybrid_weights=None, blocks=None, clusters=None) -> Artifacts:
    blocks, vec = blocks or build_blocks(books)
    clusters = clusters or fit_clusters(books, blocks, vec)
    content = combine(blocks, PRODUCTION_WEIGHTS)
    confidence = np.where(clusters.assigned_by == "title", TITLE_ONLY_CLUSTER_CONFIDENCE, 1.0)
    rec = Recommender(books=books, content=content, clusters=clusters.labels, quality=quality_score(books),
                      cluster_confidence=confidence)
    if hybrid_weights:
        rec.hybrid_weights = hybrid_weights
    books = books.assign(cluster=clusters.labels, cluster_name=[clusters.names[c] for c in clusters.labels],
                         map_x=clusters.map_xy[:, 0], map_y=clusters.map_xy[:, 1],
                         cluster_assigned_by=clusters.assigned_by)
    rec.books = books
    return Artifacts(books=books, recommender=rec, clusters=clusters, vectorizer=vec, weights=PRODUCTION_WEIGHTS)
