"""The app's pages. Each function draws one page."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import data
import ui
from src import eda


# ---------------------------------------------------------------- home
def home():
    b = data.books()
    m = data.metrics()
    ui.hero("Audible Insights", "Find your next great listen",
            "A book recommendation system built on 6,000+ Audible India titles: NLP on descriptions, genre-aware "
            "clustering and a hybrid model that balances similarity with quality.",
            pills=["Content-based", "Clustering-based", "Hybrid", f"{m['clustering']['k']} clusters",
                   "TF-IDF + SVD", "KMeans · DBSCAN"])
    rated = b[b["is_rated"]]
    ui.kpi_row([
        ("Books", f"{len(b):,}", "Unique titles after merging and de-duplicating the two files"),
        ("Average rating", f"{rated['rating'].mean():.2f} ★", f"Across {len(rated):,} rated books"),
        ("Total reviews", f"{b['num_reviews'].sum() / 1e6:.2f} M", "Sum of review counts"),
        ("Genres", f"{b.loc[b['primary_genre'] != 'Unknown', 'primary_genre'].nunique()} · 876",
         "17 broad genres, mapped from 876 fine Audible genres"),
        ("Hidden gems", f"{int(b['hidden_gem'].sum()):,}", "Rated 4.5+ by 5–65 reviewers"),
    ])
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1, ui.card("go-similar"):
        st.markdown('<p class="card-title">Loved a book?</p><p class="card-sub">Pick it and get books like it, '
                    'with the reason for each pick.</p>', unsafe_allow_html=True)
        st.page_link(st.session_state["pages"]["similar"], label="Find similar books", icon=":material/auto_stories:")
    with c2, ui.card("go-foryou"):
        st.markdown('<p class="card-title">Tell us your taste</p><p class="card-sub">Genres, sub-genres, authors or '
                    'favourite books. No history needed.</p>', unsafe_allow_html=True)
        st.page_link(st.session_state["pages"]["for_you"], label="Personalised picks", icon=":material/person_search:")
    with c3, ui.card("go-explore"):
        st.markdown('<p class="card-title">Explore the catalogue</p><p class="card-sub">Genres, ratings, authors, '
                    'clusters and how the models compare.</p>', unsafe_allow_html=True)
        st.page_link(st.session_state["pages"]["explore"], label="Explore the data", icon=":material/insights:")

    st.subheader("Most loved right now")
    st.caption("Ranked by weighted rating: a 4.8 from 30,000 listeners beats a 5.0 from three.")
    tab_all, tab_gems, tab_reviewed = st.tabs(["Top rated", "Hidden gems", "Most reviewed"])
    complete = b[b["has_description"] & (b["genre_source"] == "listed")]     # showcase only well-described books
    with tab_all:
        ui.book_cards(complete.sort_values("weighted_rating", ascending=False).head(8))
    with tab_gems:
        ui.book_cards(complete[complete["hidden_gem"]].sort_values("weighted_rating", ascending=False).head(8))
    with tab_reviewed:
        ui.book_cards(complete.sort_values("num_reviews", ascending=False).head(8))


# ---------------------------------------------------------------- similar books
def similar():
    b = data.books()
    rec = data.recommender()
    labels, _, _, _ = data.options()
    ui.hero("Recommendations", "Books like the one you loved",
            "Choose a book. The model compares its description, genres and author with every other title and "
            "explains each pick.")

    c1, c2 = st.columns([2.6, 1.4])
    st.session_state.setdefault("similar-book", None)
    st.session_state.setdefault("similar-method", "Hybrid")
    choice = c1.selectbox("Book", range(len(labels)), format_func=lambda i: labels[i],
                          placeholder="Start typing a title or author…", key="similar-book")
    method_label = c2.segmented_control("Model", list(data.METHODS), help=data.METHOD_HELP,
                                        key="similar-method") or "Hybrid"
    if choice is None:
        st.caption("Try: *The Martian*, *Atomic Habits*, *Murder on the Orient Express*, *Rich Dad Poor Dad*")
        examples = ["The Martian", "Atomic Habits", "Murder on the Orient Express (Dramatised)",
                    "Sapiens", "Harry Potter and the Philosopher's Stone, Book 1", "Ikigai"]
        cols = st.columns(len(examples))
        for col, t in zip(cols, examples):
            hit = b.index[b["title"].str.startswith(t)]
            if len(hit):
                col.button(t.split(":")[0].split(" (")[0].replace(" and the Philosopher's Stone, Book 1", "")[:28],
                           key=f"ex-{t}", width="stretch", on_click=st.session_state.__setitem__,
                           args=("similar-book", int(hit[0])))
        return

    q = b.iloc[choice]
    with ui.card("query-book"):
        st.markdown(f'<p class="card-title">{q["title"]}</p><p class="card-sub">by {q["author"]} · '
                    f'{q["primary_genre"] if q["primary_genre"] != "Unknown" else "genre not listed"} · '
                    f'{"★ %.1f (%s reviews)" % (q["rating"], f"{q.num_reviews:,}") if q["is_rated"] else "not rated"}'
                    f' · {ui.inr(q["price"])} · cluster {q["cluster"]}: {q["cluster_name"]}</p>',
                    unsafe_allow_html=True)
        if isinstance(q["description"], str):
            st.write(q["description"])
        else:
            st.caption("No description was captured for this book, so recommendations rely on its title, author "
                       "and genres, and are less precise.")
        if isinstance(q["genres"], str):
            st.caption("Audible genres: " + q["genres"].replace(" | ", " · "))

    c1, c2, c3 = st.columns([1, 1, 2])
    st.session_state.setdefault("similar-n", 10)
    n = c1.slider("How many", 5, 30, step=5, key="similar-n")
    other_authors = c2.toggle("Other authors only", key="similar-other",
                              help="Hide more books by the same author (useful for long series).")
    f = data.filter_controls("similar")
    if other_authors:
        f = data.Filters(**{**f.__dict__, "exclude_authors": (q["author"],)})
    mask = data.mask(f)

    tab_recs, tab_compare = st.tabs(["Recommendations", "Compare the three models"])
    with tab_recs:
        recs = rec.recommend([choice], data.METHODS[method_label], n=n, mask=mask)
        ui.book_cards(recs, why=data.explain(recs, [choice]))
    with tab_compare:
        st.caption("Top 6 from each model for the same book and filters.")
        cols = st.columns(3)
        for col, (name, method) in zip(cols, data.METHODS.items()):
            with col:
                st.markdown(f"**{name}**")
                r = rec.recommend([choice], method, n=6, mask=mask)
                for i, (_, row) in enumerate(r.iterrows(), 1):
                    star = f"★ {row['rating']:.1f}" if row["is_rated"] else "unrated"
                    st.markdown(f"{i}. **{row['title'][:60]}**  \n<small>{row['author']} · {star} · "
                                f"{row['primary_genre']}</small>", unsafe_allow_html=True)


# ---------------------------------------------------------------- personalised
SCENARIOS = {
    "New listener who likes sci-fi": {"fy-sub": ["Science Fiction", "Space Opera Science Fiction",
                                                 "Hard Science Fiction", "Dystopian Science Fiction",
                                                 "Adventure Science Fiction"], "fy-genres": [], "fy-authors": [],
                                      "fy-liked": []},
    "Thriller fan": {"fy-genres": ["Mystery, Thriller & Horror"], "fy-sub": ["Psychological Thrillers",
                                                                             "Crime Thrillers"],
                     "fy-authors": ["Lee Child"], "fy-liked": []},
    "Self-improvement, short listens": {"fy-genres": ["Personal Development"], "fy-sub": [], "fy-authors": [],
                                        "fy-liked": [], "fy-filter-hours": 6},
}


def _apply_scenario(preset: dict):
    for k, v in preset.items():
        st.session_state[k] = v


def for_you():
    b = data.books()
    rec = data.recommender()
    labels, genres, fine, authors = data.options()
    ui.hero("Personalised picks", "Recommendations from your taste",
            "Works with no reading history: pick any mix of genres, sub-genres, authors and books you have liked.")
    st.caption("Quick start:")
    cols = st.columns(len(SCENARIOS) + 1)
    for col, (name, preset) in zip(cols, SCENARIOS.items()):
        col.button(name, width="stretch", key=f"sc-{name}", on_click=_apply_scenario, args=(preset,))
    cols[-1].button("Clear", width="stretch", icon=":material/restart_alt:", on_click=_apply_scenario,
                    args=({"fy-genres": [], "fy-sub": [], "fy-authors": [], "fy-liked": [], "fy-filter-hours": "Any"},))

    for k in ["fy-genres", "fy-sub", "fy-authors", "fy-liked"]:
        st.session_state.setdefault(k, [])
    st.session_state.setdefault("fy-n", 12)
    c1, c2 = st.columns(2)
    g = c1.multiselect("Genres you enjoy", genres, key="fy-genres")
    sub = c2.multiselect("Sub-genres (876 Audible categories)", fine, key="fy-sub",
                         placeholder="e.g. Space Opera Science Fiction, Legal Thrillers…")
    c3, c4 = st.columns(2)
    au = c3.multiselect("Authors you like", authors, key="fy-authors")
    liked = c4.multiselect("Books you have enjoyed", range(len(labels)), format_func=lambda i: labels[i],
                           key="fy-liked", placeholder="Search titles…")
    n = st.slider("How many", 5, 30, step=1, key="fy-n")
    f = data.filter_controls("fy-filter")

    if not (g or sub or au or liked):
        st.info("Choose at least one genre, sub-genre, author or book, or use a quick start above. Until then, "
                "here are the catalogue's best-rated titles that match your filters.", icon=":material/lightbulb:")
        ui.book_cards(b[data.mask(f)].sort_values("weighted_rating", ascending=False).head(n))
        return
    recs = rec.recommend_for_profile(liked_ids=liked, genres=g, authors=au, sub_genres=sub, n=n, mask=data.mask(f))
    ui.book_cards(recs, why=data.explain_profile(recs, set(g), set(au), liked))


# ---------------------------------------------------------------- explore
def explore():
    b = data.books()
    ui.hero("Exploratory analysis", "What the catalogue tells us",
            "Interactive answers to the project's EDA questions. The full write-up is in notebooks/eda.ipynb.")
    t1, t2, t3, t4, t5 = st.tabs(["Genres", "Ratings & reviews", "Authors", "Price & length", "Data quality"])
    known = b[b["primary_genre"] != "Unknown"]

    with t1:
        g = eda.q1_popular_genres(b).reset_index()
        gi = g.set_index("primary_genre")
        sf, biz, kids = gi.loc["Science Fiction & Fantasy"], gi.loc["Business & Careers"], gi.loc["Children & Teens"]
        ui.insight_cards([
            (f"{g.iloc[0]['books']}", f"titles in <b>{g.iloc[0]['primary_genre']}</b>, the largest shelf. "
                                      f"{g.iloc[1]['primary_genre']} ({g.iloc[1]['books']}) and "
                                      f"{g.iloc[2]['primary_genre']} ({g.iloc[2]['books']}) follow."),
            (f"{sf['median_reviews']:,.0f}", f"median reviews per <b>Sci-Fi & Fantasy</b> title, "
                                             f"{sf['median_reviews'] / biz['median_reviews']:.0f}× Business: few "
                                             "titles, huge audiences."),
            (f"{kids['share_of_reviews']:.0%}", "of all reviews go to <b>Children & Teens</b>, from only "
                                                f"{kids['books']} titles."),
        ])
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g.sort_values("books"), x="books", y="primary_genre", orientation="h",
                         labels={"primary_genre": "", "books": "books"}, color_discrete_sequence=[ui.ACCENT])
            ui.chart_card(fig, "Books per genre", "Books with a listed or confidently inferred genre", height=480)
        with c2:
            fig = px.bar(g.sort_values("median_reviews"), x="median_reviews", y="primary_genre", orientation="h",
                         labels={"primary_genre": "", "median_reviews": "median reviews per book"},
                         color_discrete_sequence=["#0ea5e9"])
            ui.chart_card(fig, "Engagement per title", "Median number of reviews per book", height=480)
        fine = eda.q1_fine_genres(b, 20).reset_index()
        fine.columns = ["genre", "books"]
        fig = px.bar(fine.sort_values("books"), x="books", y="genre", orientation="h", labels={"genre": ""},
                     color_discrete_sequence=["#10b981"])
        ui.chart_card(fig, "Most common fine Audible genres", height=520)

    with t2:
        d = eda.q3_rating_distribution(b)
        t, s = eda.q5_ratings_vs_reviews(b)
        ui.insight_cards([
            (f"{d['mean']:.2f} ★", f"average rating; median {d['median']:.1f}. Ratings are compressed at the top."),
            (f"{d['share_4_5_plus']:.0%}", "of rated books score 4.5 or more; only "
                                           f"{d['share_below_4']:.1%} are below 4.0."),
            ("33%", "of books with fewer than 10 reviews are perfect 5.0s. A raw 5.0 is mostly a small sample."),
        ])
        st.write("")
        c1, c2 = st.columns(2)
        rated = b[b["is_rated"]]
        with c1:
            fig = px.histogram(rated, x="rating", nbins=40, labels={"rating": "rating"},
                               color_discrete_sequence=[ui.ACCENT])
            fig.update_layout(yaxis_title="books")
            ui.chart_card(fig, "Rating distribution", f"{len(rated):,} rated books")
        with c2:
            tt = t.reset_index()
            fig = go.Figure()
            fig.add_scatter(x=tt["review_band"].astype(str), y=tt["avg_rating"], mode="lines+markers",
                            name="average rating", line=dict(color=ui.ACCENT, width=3))
            fig.add_scatter(x=tt["review_band"].astype(str), y=tt["share_below_4"] * 100, mode="lines+markers",
                            name="% below 4.0", yaxis="y2", line=dict(color=ui.BAD, width=3))
            fig.update_layout(yaxis=dict(title="average rating"),
                              yaxis2=dict(title="% below 4.0", overlaying="y", side="right", showgrid=False),
                              xaxis_title="number of reviews")
            ui.chart_card(fig, "Ratings by review count", f"Spearman ρ = {s['spearman_rho']:.2f}: weak but real")
        order = known[known["is_rated"]].groupby("primary_genre")["rating"].median().sort_values().index
        fig = px.box(known[known["is_rated"]], x="rating", y="primary_genre", category_orders={"primary_genre":
                     list(order)[::-1]}, points=False, labels={"primary_genre": ""},
                     color_discrete_sequence=["#0ea5e9"])
        fig.update_xaxes(range=[2.5, 5.05])
        ui.chart_card(fig, "Rating by genre", height=520)
        c = eda.correlation_table(b)
        fig = px.imshow(c.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        ui.chart_card(fig, "Correlations (Spearman)", "Price follows listening time (0.66); neither moves ratings",
                      height=460)

    with t3:
        a = eda.q2_top_authors(b).head(20).reset_index()
        tpop, spop = eda.q8_author_popularity(b)
        ui.insight_cards([
            ("4.87 ★", "<b>Tracey West</b> tops authors with 3+ well-reviewed books; Tui T. Sutherland is 4.86 "
                       "across 9."),
            ("13 books", "by <b>Julia Donaldson</b>, all averaging 4.77 with 30k reviews: top-rated at scale."),
            (f"ρ = {spop['spearman_rho']:.2f}", "between author popularity and rating: an established author is a "
                                                 "safer bet, not a guarantee."),
        ])
        st.write("")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.bar(a.iloc[::-1], x="avg_rating", y="author", orientation="h", hover_data=["books",
                         "total_reviews"], labels={"author": "", "avg_rating": "average rating"},
                         color_discrete_sequence=["#8b5cf6"])
            fig.update_xaxes(range=[4.4, 4.95])
            ui.chart_card(fig, "Highest-rated authors", "At least 3 rated books, each with 20+ reviews", height=560)
        with c2:
            tp = tpop.reset_index()
            fig = px.line(tp, x="author_popularity", y="avg_rating", markers=True, hover_data=["books"],
                          labels={"author_popularity": "reviews on the author's other books",
                                  "avg_rating": "average rating"})
            fig.update_traces(line=dict(color="#8b5cf6", width=3))
            ui.chart_card(fig, "Author popularity vs rating", "Each book compared with its author's other books",
                          height=560, tickangle=-20)

    with t4:
        pl = eda.price_and_length(b).reset_index()
        rep_price = data.cleaning_report()["final_coverage"]["price_decimal_shift_corrected"]
        ui.insight_cards([
            ("₹1,003", "median price of <b>History</b>, the longest genre (16.7 h). Audiobooks are priced by "
                       "length."),
            ("25%", "of <b>Language & Education</b> titles are free taster lessons."),
            (f"₹{b['price'].median():,.0f}", "median price across the catalogue."),
        ])
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(b, x="price", nbins=60, labels={"price": "price (₹)"},
                               color_discrete_sequence=["#f59e0b"])
            fig.update_layout(yaxis_title="books")
            ui.chart_card(fig, "Price distribution", f"{rep_price} prices that were 10× too high (dropped decimal) "
                          "were corrected")
        with c2:
            fig = px.scatter(pl, x="median_hours", y="median_price", text="primary_genre",
                             labels={"median_hours": "median listening time (h)", "median_price": "median price (₹)"})
            fig.update_traces(textposition="top center", marker=dict(size=11, color=ui.ACCENT), textfont_size=10)
            ui.chart_card(fig, "Price vs length by genre")

    with t5:
        rep = data.cleaning_report()
        ui.insight_cards([
            (f"{rep['missing_raw']['advanced_description_junk']:,}", "descriptions were scraping failures (CAPTCHA "
             "pages, error pages, checkout text, award one-liners) and were treated as missing."),
            (f"{rep['dedup_catalog']['exact_duplicates']:,}", "exact duplicate rows in the basic catalog, plus "
             f"{rep['dedup_advanced']['same_book_duplicates'] + rep['dedup_catalog']['same_book_duplicates']:,} "
             "re-scraped copies of the same book."),
            (f"{rep['merge']['books']:,}", f"unique books after merging: {rep['merge']['in_both_files']:,} in both "
             f"files, {rep['merge']['only_catalog']:,} only in the basic one."),
            (f"{rep['final_coverage']['price_decimal_shift_corrected']}", "prices were 10× too high (a dropped "
             "decimal: same length as other books, 14× the price per hour) and were divided by 10."),
        ])
        st.write("")
        cov = pd.DataFrame({"field": ["Rating", "Description", "Genre (listed)", "Genre (inferred by NLP)",
                                      "Listening time"],
                            "books": [rep["final_coverage"]["rated"], rep["final_coverage"]["with_description"],
                                      rep["final_coverage"]["with_listed_genre"], rep["genre_inference"]["inferred"],
                                      rep["final_coverage"]["with_listening_time"]]})
        cov["share"] = cov["books"] / rep["merge"]["books"]
        fig = px.bar(cov, x="share", y="field", orientation="h", text=cov["books"].map("{:,}".format),
                     labels={"field": "", "share": "share of books"}, color_discrete_sequence=[ui.ACCENT])
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        ui.chart_card(fig, "How complete is each field?", "After cleaning; the rest is honestly missing")
        st.info("Publication year is not in either file (only 17 descriptions state one), so publication trends "
                "cannot be analysed. Joining dates in from an external books API is the next step.",
                icon=":material/info:")


# ---------------------------------------------------------------- clusters
def clusters():
    b = data.books()
    m = data.metrics()["clustering"]
    summary = data.cluster_summary()
    best_db = max(m["dbscan"], key=lambda d: d["agreement_with_kmeans_ari"])
    ui.hero("NLP & clustering", f"{m['k']} clusters of similar books",
            "TF-IDF on titles and descriptions plus genre signals, reduced to 150 SVD topics and grouped with "
            f"KMeans. DBSCAN finds the same structure (agreement {best_db['agreement_with_kmeans_ari']:.2f}).")
    ui.kpi_row([
        ("Clusters (KMeans)", str(m["k"]), "k chosen by the best silhouette score between 8 and 30"),
        ("Silhouette", f"{m['silhouette_by_k'][str(m['k'])]:.3f}", "Higher = tighter, better-separated clusters"),
        ("Fitted on", f"{m['fitted_on_books']:,} books", "Books with listed genres; the rest join the nearest "
                                                       "cluster by their text"),
        ("DBSCAN agreement", f"{best_db['agreement_with_kmeans_ari']:.2f} ARI", f"At eps = {best_db['eps']}: "
         f"{best_db['clusters']} clusters, {best_db['noise_share']:.1%} noise"),
    ])
    st.write("")
    mp = b[b["map_x"].notna()].copy()
    mp["cluster_label"] = mp["cluster"].astype(str) + ". " + mp["cluster_name"].str.split(" · ").str[0]
    fig = px.scatter(mp, x="map_x", y="map_y", color="cluster_label", hover_name="title",
                     hover_data={"author": True, "primary_genre": True, "map_x": False, "map_y": False,
                                 "cluster_label": False},
                     color_discrete_sequence=px.colors.qualitative.Dark24, labels={"cluster_label": ""})
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, yanchor="top"))
    ui.chart_card(fig, "Cluster map", "t-SNE projection of the 2,059 books the clusters were learned on. Hover for "
                  "titles; click legend items to hide clusters.", height=620)

    names = {int(k): f"{k}. {v['name']}" for k, v in summary.items()}
    c = st.selectbox("Look inside a cluster", list(names), format_func=names.get, key="cluster-pick")
    info = summary[str(c)]
    c1, c2 = st.columns([1, 1.4])
    with c1, ui.card(f"cluster-info-{c}"):
        st.markdown(f'<p class="card-title">{info["name"]}</p><p class="card-sub">{info["size"]:,} books '
                    f'({info["fitted_books"]} fitted) · average rating {info["avg_rating"]:.2f}</p>',
                    unsafe_allow_html=True)
        st.markdown("**Characteristic terms:** " + ", ".join(info["top_terms"][:10]))
        gm = pd.Series(info["genres"]).drop("Unknown", errors="ignore")
        fig = px.bar(gm.iloc[::-1], orientation="h", labels={"value": "books", "index": ""},
                     color_discrete_sequence=[ui.ACCENT])
        fig.update_layout(showlegend=False, height=230, margin=dict(l=4, r=8, t=8, b=4), yaxis_title="")
        st.plotly_chart(fig, width="stretch", theme=None, config=ui.PLOTLY_CONFIG, key=f"gm-{c}")
    with c2:
        top = b[(b["cluster"] == c) & (b["cluster_assigned_by"] != "title")].sort_values("weighted_rating",
                                                                                       ascending=False).head(6)
        ui.book_cards(top, desc_chars=140)

    with st.expander("KMeans vs DBSCAN, and how k was chosen", icon=":material/science:"):
        c1, c2 = st.columns(2)
        with c1:
            sil = pd.DataFrame({"k": [int(k) for k in m["silhouette_by_k"]], "silhouette": list(m["silhouette_by_k"].values())})
            fig = px.line(sil, x="k", y="silhouette", markers=True)
            fig.add_vline(x=m["k"], line_dash="dash", line_color=ui.BAD)
            ui.chart_card(fig, "Silhouette by k", f"Peak at k = {m['k']}", height=300)
        with c2:
            db = pd.DataFrame(m["dbscan"])
            db.columns = ["eps", "clusters", "noise", "largest cluster", "silhouette", "agreement (ARI)"]
            st.dataframe(db.style.format({"noise": "{:.1%}", "largest cluster": "{:.1%}", "silhouette": "{:.3f}",
                                          "agreement (ARI)": "{:.2f}"}, na_rep="–"), hide_index=True, width="stretch")
            st.caption("DBSCAN depends heavily on eps: from 97% noise to one giant cluster. At eps 0.8–0.9 it "
                       f"reproduces the KMeans clusters (ARI {best_db['agreement_with_kmeans_ari']:.2f}). KMeans is kept because every book gets a cluster "
                       "and its centroids can place the books that have no genre.")


# ---------------------------------------------------------------- model performance
def performance():
    m = data.metrics()
    rep = data.cleaning_report()
    k = m["eval_settings"]["k"]
    ui.hero("Evaluation", "How good are the recommendations?",
            f"No per-user ratings exist, so a recommendation counts as relevant when it shares at least one of the "
            f"query book's 876 fine Audible genres. Scores are averaged over {m['model_comparison'][0]['queries']} "
            f"test books, top {k} recommendations each.")
    cmp = pd.DataFrame(m["model_comparison"])
    blind = cmp[~cmp["uses_fine_genres"]].set_index("model")
    hyb, cont, rnd = blind.loc["Hybrid (genre-blind)"], blind.loc["Content-based (genre-blind)"], blind.loc["Random"]
    rm = pd.DataFrame(m["rating_prediction"]).set_index("model")
    ui.kpi_row([
        (f"Hybrid precision@{k}", f"{hyb[f'precision@{k}']:.3f}", "Genre-blind setting (text only)",
         f"{hyb[f'precision@{k}'] / rnd[f'precision@{k}']:.1f}× random", "normal"),
        (f"Hybrid nDCG@{k}", f"{hyb[f'ndcg@{k}']:.3f}", "Rewards relevant books near the top",
         f"+{(hyb[f'ndcg@{k}'] / cont[f'ndcg@{k}'] - 1):.0%} vs content", "normal"),
        ("Best fair features", f"{pd.DataFrame(m['feature_ablation']).query('~uses_fine_genres')[f'precision@{k}'].max():.3f}",
         "Text + broad genre + author, precision@10"),
        ("Rating RMSE", f"{rm.loc['Hybrid (k-NN + author mean)', 'rmse']:.3f}", "Predicting held-out ratings",
         f"{rm.loc['Hybrid (k-NN + author mean)', 'rmse'] / rm.loc['Global mean (baseline)', 'rmse'] - 1:.0%} vs mean",
         "inverse"),
    ])
    st.write("")
    fmt = {c: "{:.3f}" for c in cmp.columns if "@" in c}
    fmt.update({"avg_rating_of_recs": "{:.2f}", "catalogue_coverage": "{:.0%}"})
    st.markdown("#### Model comparison")
    st.caption("**Genre-blind** models use only titles and descriptions, so they cannot see the labels being "
               "scored. That makes them the fair comparison. **Full-feature** models (production) also use the fine "
               "genres, which define relevance, so their scores are an optimistic upper bound.")
    show = cmp.drop(columns=["queries"]).rename(columns={"uses_fine_genres": "uses fine genres"})
    st.dataframe(show.style.format(fmt), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        bl = blind.reset_index()
        fig = px.bar(bl, x=[f"precision@{k}", f"ndcg@{k}", f"recall@{k}"], y="model", orientation="h",
                     barmode="group", labels={"value": "score", "model": "", "variable": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=list(bl["model"])[::-1])
        ui.chart_card(fig, "Genre-blind comparison", "Hybrid > content > clustering > baselines", height=380)
    with c2:
        ab = pd.DataFrame(m["feature_ablation"])
        fig = px.bar(ab, x=f"precision@{k}", y="features", orientation="h",
                     color=ab["uses_fine_genres"].map({True: "uses the label source", False: "fair"}),
                     color_discrete_map={"fair": ui.ACCENT, "uses the label source": ui.BAD},
                     text=ab[f"precision@{k}"].map("{:.2f}".format), labels={"features": "", "color": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=list(ab["features"])[::-1])
        ui.chart_card(fig, "Which features matter?", "Content-based model, one feature block added at a time",
                      height=380)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Rating prediction (RMSE)")
        st.caption(f"Predict the rating of {int(rm['test_books'].iloc[0]):,} held-out books (5+ reviews) from "
                   "similar books. Ratings barely vary (std 0.35), so gains over the mean are modest.")
        st.dataframe(rm[["rmse", "mae"]].style.format("{:.4f}"), width="stretch")
    with c2:
        st.markdown("#### Hybrid weights")
        w = m["hybrid_weights"]
        st.caption("Grid-searched on separate validation books: nDCG@10 plus a small reward for well-rated picks.")
        st.markdown(f"**score = {w['content']:.2f} × similarity + {w['cluster']:.2f} × same cluster + "
                    f"{w['quality']:.2f} × weighted rating**")
        tune = pd.DataFrame(m["hybrid_tuning"]).sort_values("objective", ascending=False).head(6)
        fmt = {c: "{:.2f}" if c in ("content", "cluster", "quality", "avg_rating_of_recs") else "{:.3f}"
               for c in tune.columns}
        st.dataframe(tune.style.format(fmt), hide_index=True, width="stretch")
    gi = rep["genre_inference"]
    with st.expander("Genre inference model (NLP)", icon=":material/psychology:"):
        st.markdown(
            f"Logistic regression on TF-IDF of title + description, trained on {gi['trained_on']:,} books with "
            f"listed genres, 17 classes. 5-fold cross-validation: accuracy **{gi['cv_accuracy']:.1%}**, macro-F1 "
            f"{gi['cv_macro_f1']:.3f}, top-3 accuracy {gi['cv_top3_accuracy']:.1%}. Only predictions with "
            f"probability ≥ {gi['min_probability']:.2f} are kept (right {gi['cv_accuracy_when_confident']:.0%} of "
            f"the time on CV), which filled in the genre for **{gi['inferred']}** of {gi['candidates']} books that "
            f"had a description but no genre.")
        pg = pd.DataFrame(gi["per_genre"]).T.sort_values("f1-score", ascending=False)
        st.dataframe(pg.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}",
                                      "support": "{:.0f}"}), width="stretch")


# ---------------------------------------------------------------- about
def about():
    rep = data.cleaning_report()
    ui.hero("About", "Audible Insights: Intelligent Book Recommendations",
            "GUVI × HCL capstone: data cleaning, EDA, NLP, clustering, three recommendation models and a Streamlit "
            "app, deployed on Streamlit Community Cloud and AWS.")
    st.markdown(f"""
#### Pipeline
1. **Merge & clean** (`pipeline.py`). `Audible_Catlog.csv` ({rep['raw']['catalog']['rows']:,} rows) and
   `Audible_Catlog_Advanced_Features.csv` ({rep['raw']['advanced']['rows']:,} rows) are standardised and
   de-duplicated, then outer-joined on normalised title + author, giving {rep['merge']['books']:,} books. Steps:
   - `-1` ratings become *unrated*.
   - {rep['missing_raw']['advanced_description_junk']:,} junk descriptions are removed.
   - Listening time is parsed to minutes.
   - The rank string is split into an overall Audible rank and 876 fine genres, which are mapped to 17 broad genres.
2. **NLP** (`src/nlp.py`). TF-IDF (1–2-grams, domain stop words) on title + description. A logistic-regression
   classifier infers the genre where it is missing.
3. **Clustering** (`src/recommenders.py`). SVD to 150 topics, then KMeans (k by silhouette), with DBSCAN for
   comparison. Clusters are learned on books with genres; other books join the nearest cluster by text.
4. **Recommenders.** Content-based (cosine similarity over text, genre and author blocks), clustering-based
   (best-rated in cluster) and hybrid (tuned blend of similarity, cluster and weighted rating).
5. **Evaluation** (`train.py`). Precision, recall and nDCG@10 against genre overlap; RMSE for rating
   prediction; feature ablation.

#### Honest limitations
- **No per-user data.** Collaborative filtering isn't possible, and "relevance" is approximated by shared genres.
- **Missing scraped data.** {rep['merge']['books'] - rep['final_coverage']['with_description']:,} books have no
  usable description and 62% have no genre, so their recommendations rely on title and author and are less precise.
- **No publication dates.** Publication-year trends cannot be analysed.

#### Tech
Python · pandas · scikit-learn · SciPy · Plotly · Streamlit · Docker · AWS EC2
""")
