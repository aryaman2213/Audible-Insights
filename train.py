"""Build the recommender: features, clusters, hybrid weights, evaluation. Writes models/ and the enriched book table.

    python train.py          (~2 min)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

import config
from src import evaluation as E
from src import recommenders as R


def table(df: pd.DataFrame) -> list:
    return json.loads(df.round(4).to_json(orient="records"))


def main():
    t0 = time.time()
    config.MODELS_DIR.mkdir(exist_ok=True)
    books = pd.read_csv(config.BOOKS_CLEAN)

    print("1/5 features + clusters")
    blocks, vec = R.build_blocks(books)
    clusters = R.fit_clusters(books, blocks, vec)
    quality = R.quality_score(books)
    print(f"    k={clusters.k}, silhouette={clusters.silhouette_by_k[clusters.k]:.3f}")

    print("2/5 tuning hybrid weights (validation queries)")
    hybrid_weights, tuning = E.tune_hybrid(books, blocks, clusters.labels, quality)
    print("    best:", hybrid_weights)

    print("3/5 comparing models (test queries)")
    rankers = E.build_rankers(books, blocks, quality, clusters.labels, clusters.text_labels, hybrid_weights)
    per_query = {}
    comparison = E.evaluate_rankers(books, {n: r for n, (r, _) in rankers.items()}, per_query=per_query)
    comparison["uses_fine_genres"] = [rankers[n][1] for n in comparison["model"]]
    print(comparison.round(3).to_string(index=False))
    sig = E.significance(per_query, [("Hybrid (genre-blind)", "Content-based (genre-blind)"),
                                     ("Hybrid (genre-blind)", "Clustering-based (genre-blind)"),
                                     ("Content-based (genre-blind)", "Random")])
    print(pd.DataFrame(sig).round(4).to_string(index=False))

    print("4/5 feature ablation + rating prediction")
    ablation = E.feature_ablation(books, blocks)
    print(ablation.round(3).to_string(index=False))
    rmse = E.rating_prediction(books, blocks)
    print(rmse.round(4).to_string(index=False))

    print("5/5 saving")
    art = R.build_artifacts(books, hybrid_weights=hybrid_weights, blocks=(blocks, vec), clusters=clusters)
    joblib.dump(art, config.RECOMMENDER_ARTIFACTS, compress=3)

    summary = {int(c): {"name": clusters.names[c], "top_terms": clusters.top_terms[c], "size": clusters.sizes[c],
                        "fitted_books": int((clusters.fit_mask & (clusters.labels == c)).sum()),
                        "genres": books.loc[clusters.labels == c, "primary_genre"].value_counts().head(5).to_dict(),
                        "avg_rating": round(float(books.loc[clusters.labels == c, "rating"].mean()), 3),
                        "examples": books.loc[(clusters.labels == c) & clusters.fit_mask]
                                         .nlargest(5, "num_reviews")["title"].tolist()}
               for c in range(clusters.k)}
    config.CLUSTER_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    metrics = {
        "clustering": {
            "k": clusters.k, "silhouette_by_k": {int(k): round(v, 4) for k, v in clusters.silhouette_by_k.items()},
            "fitted_on_books": int(clusters.fit_mask.sum()),
            "assigned_by": pd.Series(clusters.assigned_by).value_counts().to_dict(),
            "dbscan": clusters.dbscan,
        },
        "hybrid_weights": hybrid_weights,
        "hybrid_tuning": table(tuning),
        "model_comparison": table(comparison),
        "significance": sig,
        "feature_ablation": table(ablation),
        "rating_prediction": table(rmse),
        "eval_settings": {"k": config.EVAL_K, "queries": config.EVAL_QUERIES,
                          "relevant": "shares >= 1 fine Audible genre with the query",
                          "strictly_relevant": "shares >= 2 fine genres",
                          "pool": "books with listed genres", "seed": config.SEED},
    }
    config.MODEL_METRICS.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
