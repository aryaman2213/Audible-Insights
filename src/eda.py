"""One function per EDA question (returns a DataFrame or dict), plus the static charts for the notebook and report.
The Streamlit app re-uses the data functions and draws its own interactive charts."""
import json

import matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

PALETTE = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f97316"]
sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 13})

KNOWN = lambda df: df[df["primary_genre"] != config.UNKNOWN_GENRE]


def load_books() -> pd.DataFrame:
    import joblib
    return joblib.load(config.RECOMMENDER_ARTIFACTS).books


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=130)
    return fig


# ---------------------------------------------------------------- easy
def q1_popular_genres(b: pd.DataFrame) -> pd.DataFrame:
    """Popularity three ways: number of books, total reviews (a proxy for listeners), and average rating."""
    k = KNOWN(b)
    g = k.groupby("primary_genre").agg(books=("title", "size"), total_reviews=("num_reviews", "sum"),
                                       median_reviews=("num_reviews", "median"), avg_rating=("rating", "mean"))
    g["share_of_reviews"] = g["total_reviews"] / g["total_reviews"].sum()
    return g.sort_values("books", ascending=False)


def q1_fine_genres(b: pd.DataFrame, top=15) -> pd.Series:
    fine = b["genres"].dropna().str.split(r" \| ").explode().str.strip()
    return fine.value_counts().head(top)


def q2_top_authors(b: pd.DataFrame, min_books=3, min_reviews=20) -> pd.DataFrame:
    """Authors with at least `min_books` rated books (each with >= min_reviews), ranked by mean rating, ties broken
    by total reviews. A single 5-star book with 3 reviews does not make a top author."""
    r = b[b["is_rated"] & (b["num_reviews"] >= min_reviews)]
    a = r.groupby("author").agg(books=("title", "size"), avg_rating=("rating", "mean"),
                                avg_weighted_rating=("weighted_rating", "mean"), total_reviews=("num_reviews", "sum"))
    a = a[a["books"] >= min_books]
    return a.sort_values(["avg_rating", "total_reviews"], ascending=False)


def q3_rating_distribution(b: pd.DataFrame) -> dict:
    r = b.loc[b["is_rated"], "rating"]
    bands = pd.cut(r, [0, 3, 3.5, 4, 4.5, 5.01], right=False,
                   labels=["< 3.0", "3.0-3.4", "3.5-3.9", "4.0-4.4", "4.5-5.0"]).value_counts().sort_index()
    return {"rated_books": int(len(r)), "unrated_books": int((~b["is_rated"]).sum()), "mean": r.mean(),
            "median": r.median(), "std": r.std(), "skew": float(stats.skew(r)), "share_4_5_plus": (r >= 4.5).mean(),
            "share_below_4": (r < 4).mean(), "bands": bands}


def q4_publication_years(b: pd.DataFrame) -> dict:
    """Neither file has a publication date. Count what little year information the text holds."""
    d = b["description"].fillna("")
    explicit = d.str.extract(r"(?i)\b(?:first |originally )?published\s+(?:in\s+)?(1[5-9]\d\d|20[0-2]\d)\b")[0]
    return {"publication_date_column": False, "descriptions_stating_a_publication_year": int(explicit.notna().sum()),
            "titles_containing_a_year": int(b["title"].str.contains(r"\b(?:19|20)\d\d\b").sum())}


def q5_ratings_vs_reviews(b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    r = b[b["is_rated"]].copy()
    bins = [0, 9, 49, 199, 999, 4999, np.inf]
    labels = ["1-9", "10-49", "50-199", "200-999", "1k-5k", "5k+"]
    r["review_band"] = pd.cut(r["num_reviews"], bins, labels=labels)
    t = r.groupby("review_band", observed=True)["rating"].agg(books="size", avg_rating="mean", std="std",
                                                              share_5_star=lambda s: (s == 5).mean(),
                                                              share_below_4=lambda s: (s < 4).mean())
    rho, p = stats.spearmanr(r["num_reviews"], r["rating"])
    return t, {"spearman_rho": rho, "p_value": p}


# ---------------------------------------------------------------- medium
def q6_cluster_table(summary: dict) -> pd.DataFrame:
    rows = [{"cluster": int(c), "name": v["name"], "books": v["size"], "fitted": v["fitted_books"],
             "avg_rating": v["avg_rating"], "top_terms": ", ".join(v["top_terms"][:6]),
             "examples": " · ".join(v["examples"][:3])} for c, v in summary.items()]
    return pd.DataFrame(rows).sort_values("books", ascending=False)


def q6_most_similar_pairs(b: pd.DataFrame, top=12) -> pd.DataFrame:
    """Most similar book pairs by description alone (different authors, real descriptions)."""
    from src import recommenders as R
    blocks, _ = R.build_blocks(b)
    idx = np.where(b["has_description"].to_numpy())[0]
    T = blocks["text"][idx]
    S = (T @ T.T).toarray()
    np.fill_diagonal(S, 0)
    authors = b["author"].to_numpy()[idx]
    S[authors[:, None] == authors[None, :]] = 0
    st = b["short_title"].str.lower().to_numpy()[idx]
    S[st[:, None] == st[None, :]] = 0
    iu = np.triu_indices_from(S, 1)
    order = np.argsort(-S[iu])[:top]
    rows = []
    for o in order:
        i, j = idx[iu[0][o]], idx[iu[1][o]]
        rows.append({"book_a": b.at[i, "title"], "book_b": b.at[j, "title"], "similarity": S[iu[0][o], iu[1][o]],
                     "same_cluster": b.at[i, "cluster"] == b.at[j, "cluster"]})
    return pd.DataFrame(rows)


def q8_author_popularity(b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Author popularity = total reviews across the author's books. Each book is compared with its author's
    popularity *excluding that book*, so a single blockbuster does not correlate with itself."""
    r = b[b["is_rated"]].copy()
    tot = r.groupby("author")["num_reviews"].transform("sum")
    r["author_reviews_excl"] = tot - r["num_reviews"]
    r["author_books"] = r.groupby("author")["title"].transform("size")
    multi = r[r["author_books"] >= 2]
    bins = [-1, 0, 99, 999, 9999, np.inf]
    labels = ["0 (other books unreviewed)", "1-99", "100-999", "1k-10k", "10k+"]
    multi = multi.assign(author_popularity=pd.cut(multi["author_reviews_excl"], bins, labels=labels))
    t = multi.groupby("author_popularity", observed=True)["rating"].agg(books="size", avg_rating="mean",
                                                                       share_below_4=lambda s: (s < 4).mean())
    rho, p = stats.spearmanr(multi["author_reviews_excl"], multi["rating"])
    return t, {"spearman_rho": rho, "p_value": p, "books_by_multi_book_authors": len(multi)}


# ---------------------------------------------------------------- scenarios
def scenario_scifi(b: pd.DataFrame, rec, n=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) Cold start, no history: the best science-fiction titles by weighted rating (fine genre contains
    'Science Fiction'; children's titles excluded). (b) What the hybrid model suggests from that seed profile."""
    sf = b["genres"].fillna("").str.contains("Science Fiction|Sci-Fi") & (b["primary_genre"] != "Children & Teens")
    top = b[sf].sort_values("weighted_rating", ascending=False).head(n)
    seeds = b[sf].sort_values("weighted_rating", ascending=False).head(10).index
    mask = (b["primary_genre"] != "Children & Teens").to_numpy()
    more = rec.recommend(list(seeds), "hybrid", n=n, mask=mask & ~b.index.isin(top.index))
    return top, more


def scenario_thriller(b: pd.DataFrame, rec, n=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A user who rated thrillers highly: their history = well-reviewed thrillers rated >= 4.5. Recommend from the
    hybrid model, excluding the history."""
    thr = b["primary_genre"].eq("Mystery, Thriller & Horror") & b["genres"].fillna("").str.contains("Thriller")
    thr &= (b["rating"] >= 4.5) & (b["num_reviews"] >= 200)
    history = b[thr].sort_values("num_reviews", ascending=False).head(5)
    recs = rec.recommend(list(history.index), "hybrid", n=n)
    return history, recs


def scenario_hidden_gems(b: pd.DataFrame, n=15) -> pd.DataFrame:
    """Ranked by weighted rating, which already favours a 4.9 from 30 reviews over a 5.0 from 6."""
    g = b[b["hidden_gem"] & b["has_description"]]
    return g.sort_values(["weighted_rating", "num_reviews"], ascending=False).head(n)


# ---------------------------------------------------------------- own analysis
def correlation_table(b: pd.DataFrame) -> pd.DataFrame:
    r = b[b["is_rated"]].assign(log_reviews=lambda d: np.log1p(d["num_reviews"]),
                                log_audible_rank=lambda d: np.log1p(d["audible_rank"]))
    cols = {"rating": "Rating", "log_reviews": "Reviews (log)", "weighted_rating": "Weighted rating",
            "price": "Price", "listening_minutes": "Listening time", "log_audible_rank": "Audible rank (log)",
            "num_genres": "No. of genres"}
    return r[list(cols)].rename(columns=cols).corr(method="spearman")


def price_and_length(b: pd.DataFrame) -> pd.DataFrame:
    k = KNOWN(b)
    return k.groupby("primary_genre").agg(median_price=("price", "median"),
                                          median_hours=("listening_minutes", lambda m: m.median() / 60),
                                          free_share=("is_free", "mean")).sort_values("median_price")


# ---------------------------------------------------------------- static charts (notebook + report)
def chart_genres(b):
    t = q1_popular_genres(b)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    t["books"].sort_values().plot.barh(ax=ax[0], color=PALETTE[0])
    ax[0].set(title="Books per genre", xlabel="books", ylabel="")
    (t["total_reviews"] / 1000).sort_values().plot.barh(ax=ax[1], color=PALETTE[1])
    ax[1].set(title="Total reviews per genre (thousands)", xlabel="reviews (k)", ylabel="")
    fig.tight_layout()
    return save(fig, "q1_genres")


def chart_fine_genres(b):
    s = q1_fine_genres(b)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    s.sort_values().plot.barh(ax=ax, color=PALETTE[2])
    ax.set(title="Most common fine Audible genres", xlabel="books", ylabel="")
    fig.tight_layout()
    return save(fig, "q1_fine_genres")


def chart_top_authors(b, top=15):
    a = q2_top_authors(b).head(top)
    fig, ax = plt.subplots(figsize=(9, 6))
    a["avg_rating"][::-1].plot.barh(ax=ax, color=PALETTE[5])
    ax.set_xlim(4.3, 5.0)
    ax.set(title=f"Highest-rated authors (>= 3 books, each >= 20 reviews)", xlabel="average rating", ylabel="")
    for i, (r, n) in enumerate(zip(a["avg_rating"][::-1], a["books"][::-1])):
        ax.text(r + 0.005, i, f"{r:.2f} · {n} books", va="center", fontsize=9)
    fig.tight_layout()
    return save(fig, "q2_top_authors")


def chart_rating_distribution(b):
    r = b.loc[b["is_rated"], "rating"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    sns.histplot(r, bins=np.arange(1, 5.11, 0.1), ax=ax[0], color=PALETTE[0])
    ax[0].axvline(r.mean(), color=PALETTE[4], ls="--", label=f"mean {r.mean():.2f}")
    ax[0].axvline(r.median(), color=PALETTE[2], ls=":", label=f"median {r.median():.1f}")
    ax[0].legend()
    ax[0].set(title="Rating distribution (rated books)", xlabel="rating", ylabel="books")
    k = KNOWN(b[b["is_rated"]])
    order = k.groupby("primary_genre")["rating"].median().sort_values().index
    sns.boxplot(data=k, y="primary_genre", x="rating", order=order, ax=ax[1], color=PALETTE[1], fliersize=1.5)
    ax[1].set(title="Rating by genre", xlabel="rating", ylabel="")
    ax[1].set_xlim(2.5, 5.05)
    fig.tight_layout()
    return save(fig, "q3_ratings")


def chart_ratings_vs_reviews(b):
    t, s = q5_ratings_vs_reviews(b)
    r = b[b["is_rated"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].scatter(r["num_reviews"], r["rating"] + np.random.RandomState(0).uniform(-0.04, 0.04, len(r)),
                  s=6, alpha=0.3, color=PALETTE[0])
    ax[0].set_xscale("log")
    ax[0].set(title=f"Rating vs number of reviews (Spearman ρ = {s['spearman_rho']:.2f})",
              xlabel="reviews (log scale)", ylabel="rating")
    ax[1].plot(t.index.astype(str), t["avg_rating"], marker="o", color=PALETTE[0], label="average rating")
    ax2 = ax[1].twinx()
    ax2.plot(t.index.astype(str), t["share_below_4"] * 100, marker="s", color=PALETTE[4], label="% rated below 4")
    ax2.grid(False)
    ax[1].set(title="Average rating and % below 4.0 by review count", xlabel="reviews", ylabel="average rating")
    ax2.set_ylabel("% below 4.0")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.75, -0.02))
    fig.tight_layout()
    return save(fig, "q5_ratings_vs_reviews")


def chart_correlations(b):
    c = correlation_table(b)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": .8})
    ax.set_title("Spearman correlations (rated books)")
    fig.tight_layout()
    return save(fig, "correlations")


def chart_popularity_curve(b):
    """Stand-in for the publication-trend line chart: how rating and reviews change along the Audible
    bestseller rank (the only 'ordering' the data has)."""
    r = b[b["is_rated"] & b["audible_rank"].notna()].copy()
    bins = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, np.inf]
    labels = ["1-100", "101-500", "501-1k", "1k-2.5k", "2.5k-5k", "5k-10k", "10k-25k", "25k+"]
    r["rank_band"] = pd.cut(r["audible_rank"], bins, labels=labels)
    t = r.groupby("rank_band", observed=True).agg(avg_rating=("rating", "mean"),
                                                  median_reviews=("num_reviews", "median"), books=("title", "size"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t.index.astype(str), t["avg_rating"], marker="o", color=PALETTE[0], label="average rating")
    ax2 = ax.twinx()
    ax2.plot(t.index.astype(str), t["median_reviews"], marker="s", color=PALETTE[3], label="median reviews")
    ax2.set_yscale("log")
    ax2.grid(False)
    ax.set(title="Rating and reviews along the Audible bestseller rank", xlabel="overall Audible rank",
           ylabel="average rating")
    ax2.set_ylabel("median reviews (log)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout()
    save(fig, "popularity_curve")
    return t


def chart_author_popularity(b):
    t, s = q8_author_popularity(b)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t.index.astype(str), t["avg_rating"], marker="o", color=PALETTE[5])
    for x, (y, n) in enumerate(zip(t["avg_rating"], t["books"])):
        ax.annotate(f"{y:.2f}\n(n={n})", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set(title=f"Rating vs author popularity (reviews of the author's other books), ρ = {s['spearman_rho']:.2f}",
           xlabel="author's reviews on other books", ylabel="average rating")
    ax.set_ylim(t["avg_rating"].min() - 0.15, t["avg_rating"].max() + 0.15)
    fig.tight_layout()
    return save(fig, "q8_author_popularity")


def chart_clusters(b, summary):
    m = b[b["map_x"].notna()]
    fig, ax = plt.subplots(figsize=(11, 8))
    names = {int(k): v["name"].split(" · ")[0] for k, v in summary.items()}
    cmap = plt.get_cmap("tab20")
    for c in sorted(m["cluster"].unique()):
        d = m[m["cluster"] == c]
        ax.scatter(d["map_x"], d["map_y"], s=7, alpha=0.7, color=cmap(c % 20), label=f"{c}: {names[c]}")
        ax.text(d["map_x"].median(), d["map_y"].median(), str(c), fontsize=11, weight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0))
    ax.set(title="Book clusters (t-SNE of the text + genre topic space, 2,059 books with listed genres)",
           xticks=[], yticks=[])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=3, frameon=False)
    fig.tight_layout()
    return save(fig, "clusters_map")


def chart_silhouette(metrics):
    sil = metrics["clustering"]["silhouette_by_k"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = [int(k) for k in sil]
    ax.plot(ks, list(sil.values()), marker="o", color=PALETTE[0])
    best = metrics["clustering"]["k"]
    ax.axvline(best, ls="--", color=PALETTE[4])
    ax.annotate(f"chosen k = {best}", (best, sil[str(best)]), xytext=(8, -18), textcoords="offset points")
    ax.set(title="KMeans: silhouette score by number of clusters", xlabel="k", ylabel="silhouette")
    fig.tight_layout()
    return save(fig, "silhouette")


def chart_model_comparison(metrics):
    t = pd.DataFrame(metrics["model_comparison"])
    t = t[~t["uses_fine_genres"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    k = metrics["eval_settings"]["k"]
    t.set_index("model")[[f"precision@{k}", f"ndcg@{k}", f"recall@{k}"]].plot.barh(ax=ax[0], color=PALETTE[:3])
    ax[0].invert_yaxis()
    ax[0].set(title="Genre-blind comparison (relevant = shares a fine genre)", xlabel="score", ylabel="")
    a = pd.DataFrame(metrics["feature_ablation"])
    colors = [PALETTE[4] if f else PALETTE[0] for f in a["uses_fine_genres"]]
    ax[1].barh(a["features"], a[f"precision@{k}"], color=colors)
    ax[1].invert_yaxis()
    for i, v in enumerate(a[f"precision@{k}"]):
        ax[1].text(v + 0.01, i, f"{v:.2f}", va="center")
    ax[1].set(title=f"Feature combinations: precision@{k} (red = uses the label source)", xlabel=f"precision@{k}")
    ax[1].set_xlim(0, 1)
    fig.tight_layout()
    return save(fig, "model_comparison")


def chart_price_length(b):
    k = KNOWN(b)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    sns.histplot(b["price"], bins=60, ax=ax[0], color=PALETTE[3])
    ax[0].set(title="Price distribution (₹, after the decimal-shift fix)", xlabel="price (₹)", ylabel="books")
    order = k.groupby("primary_genre")["listening_minutes"].median().sort_values().index
    sns.boxplot(data=k.assign(hours=k["listening_minutes"] / 60), y="primary_genre", x="hours", order=order,
                ax=ax[1], color=PALETTE[6], fliersize=1.5)
    ax[1].set(title="Listening time by genre (hours)", xlabel="hours", ylabel="")
    ax[1].set_xlim(0, 40)
    fig.tight_layout()
    return save(fig, "price_length")


def load_metrics():
    return json.loads(config.MODEL_METRICS.read_text()), json.loads(config.CLUSTER_SUMMARY.read_text())
