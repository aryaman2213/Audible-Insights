"""Build the project report (HTML -> PDF).

Numbers come from models/model_metrics.json, models/cluster_summary.json and data/processed/cleaning_report.json;
written answers come from notebooks/eda.ipynb; charts from reports/figures/ and reports/screenshots/. So the report
always matches the analysis. The PDF is printed by headless Google Chrome through Playwright.

    python reports/build_report.py --live-url https://<your-app>.streamlit.app --aws-url http://<ec2-ip>
"""
import argparse
import base64
import html
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import markdown
import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REPORTS = ROOT / "reports"
PDF_PATH = REPORTS / "Audible_Insights_Report.pdf"
AUTHOR = "Aryaman Bhatnagar"
GITHUB_URL = "https://github.com/aryaman2213/audible-insights"


def md(text: str) -> str:
    text = re.sub(r"([^\n])\n(?=(\d+\.|-) )", r"\1\n\n", text)
    return markdown.markdown(text, extensions=["tables"])


def img(path: Path, caption: str, max_width=1400, quality=82) -> str:
    """Embed an image as a compressed JPEG data URI."""
    im = Image.open(path).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/jpeg;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def fig(name: str, caption: str) -> str:
    return img(config.FIGURES_DIR / f"{name}.png", caption, max_width=1100)


def shot(name: str, caption: str) -> str:
    return img(REPORTS / "screenshots" / f"{name}.png", caption)


def notebook_answers() -> dict:
    """{'Q1': (question, answer html), 'S1': ...} from the executed notebook."""
    nb = nbformat.read(ROOT / "notebooks" / "eda.ipynb", as_version=4)
    answers, current = {}, None
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        heading = re.search(r"### ([QS]\d+)\. (.+)", cell.source)
        if heading:
            current = heading.groups()
        elif cell.source.startswith("**Answer") and current:
            text = re.sub(r"^\*\*Answer[^*]*\*\*\s*", "", cell.source)
            answers[current[0]] = (current[1], md(text))
            current = None
    return answers


def table(rows: list[dict], cols: list[tuple[str, str, str]], highlight=None) -> str:
    head = "".join(f"<th>{label}</th>" for _, label, _ in cols)
    body = []
    for r in rows:
        cls = ' class="champ"' if highlight and highlight(r) else ""
        cells = "".join(f"<td>{format(r[k], f) if isinstance(r.get(k), (int, float)) and f else html.escape(str(r.get(k, '–')))}</td>"
                        for k, _, f in cols)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: Inter, -apple-system, 'Segoe UI', sans-serif; color: #0f172a; font-size: 10.2pt; line-height: 1.5; }
h1 { font-size: 26pt; margin: 0 0 6pt 0; letter-spacing: -0.02em; }
h2 { font-size: 15.5pt; margin: 20pt 0 6pt 0; padding-bottom: 4pt; border-bottom: 2px solid #e0e7ff; color: #312e81; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 3pt 0; break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #e2e8f0; padding: 4pt 6pt; text-align: left; }
th { background: #eef2ff; }
tr.champ td { background: #ecfdf5; font-weight: 600; }
code { background: #f1f5f9; padding: 0 3pt; border-radius: 3pt; font-size: 8.8pt; }
figure { margin: 6pt 0 10pt 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 225mm; border: 1px solid #e2e8f0; border-radius: 6px; }
figcaption { color: #64748b; font-size: 8.5pt; margin-top: 3pt; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.eyebrow { color: #4f46e5; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; font-size: 9pt; }
.sub { color: #475569; font-size: 12.5pt; max-width: 150mm; }
.meta td { border: none; padding: 2pt 10pt 2pt 0; font-size: 10pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin: 8pt 0; break-inside: avoid; }
.kpi { border: 1px solid #e2e8f0; border-radius: 8px; padding: 6pt 8pt; }
.kpi b { display: block; font-size: 14pt; } .kpi span { color: #64748b; font-size: 8.5pt; }
.qa { break-inside: avoid; margin-bottom: 6pt; } .qa .q { font-weight: 650; }
.box { background: #f8fafc; border-left: 4px solid #4f46e5; padding: 6pt 10pt; margin: 8pt 0; break-inside: avoid; }
.page { break-before: page; }
"""


def build_html(live_url: str | None, aws_url: str | None) -> str:
    m = json.loads(config.MODEL_METRICS.read_text())
    rep = json.loads(config.CLEANING_REPORT.read_text())
    summary = json.loads(config.CLUSTER_SUMMARY.read_text())
    answers = notebook_answers()
    k = m["eval_settings"]["k"]
    cl = m["clustering"]
    best_db = max(cl["dbscan"], key=lambda d: d["agreement_with_kmeans_ari"])
    cmp = {r["model"]: r for r in m["model_comparison"]}
    hyb, rnd = cmp["Hybrid (genre-blind)"], cmp["Random"]
    rmse = {r["model"]: r for r in m["rating_prediction"]}
    gi = rep["genre_inference"]
    fc, mr = rep["final_coverage"], rep["missing_raw"]
    w = m["hybrid_weights"]

    def qa(ids):
        return "".join(f'<div class="qa"><p class="q">{q}. {html.escape(answers[q][0])}</p>{answers[q][1]}</div>'
                       for q in ids if q in answers)

    links = f'<tr><td>GitHub</td><td><a href="{GITHUB_URL}">{GITHUB_URL}</a></td></tr>'
    if live_url:
        links += f'<tr><td>Streamlit app</td><td><a href="{live_url}">{live_url}</a></td></tr>'
    if aws_url:
        links += f'<tr><td>AWS app</td><td><a href="{aws_url}">{aws_url}</a></td></tr>'

    metric_cols = [("model", "Model", ""), (f"precision@{k}", f"Precision@{k}", ".3f"), (f"recall@{k}", f"Recall@{k}", ".3f"),
                   (f"ndcg@{k}", f"nDCG@{k}", ".3f"), (f"strict_precision@{k}", "Strict P@10", ".3f"),
                   ("avg_rating_of_recs", "Avg rating of recs", ".2f"), ("catalogue_coverage", "Coverage", ".0%")]
    clusters_rows = sorted(({"c": int(c), "name": v["name"], "size": v["size"], "fitted": v["fitted_books"],
                             "ex": " · ".join(v["examples"][:2])} for c, v in summary.items()), key=lambda r: -r["size"])

    body = f"""
<div class="cover">
  <div class="eyebrow">GUVI × HCL capstone · Recommendation systems</div>
  <h1>Audible Insights: Intelligent Book Recommendations</h1>
  <p class="sub">A book recommendation system for {rep['merge']['books']:,} Audible India titles: data cleaning and merging,
  NLP on descriptions, clustering, and content-based, clustering-based and hybrid recommenders, served through a
  Streamlit app on Streamlit Community Cloud and AWS.</p>
  <table class="meta">
    <tr><td>Author</td><td>{AUTHOR}</td></tr>
    <tr><td>Date</td><td>{date.today():%d %B %Y}</td></tr>
    {links}
    <tr><td>Stack</td><td>Python, pandas, scikit-learn, SciPy, Plotly, Streamlit, Docker, AWS EC2</td></tr>
  </table>
  <div class="kpis">
    <div class="kpi"><b>{rep['merge']['books']:,}</b><span>unique books from {rep['raw']['catalog']['rows'] + rep['raw']['advanced']['rows']:,} raw rows</span></div>
    <div class="kpi"><b>{cl['k']} clusters</b><span>silhouette {cl['silhouette_by_k'][str(cl['k'])]:.3f}; DBSCAN agrees (ARI {best_db['agreement_with_kmeans_ari']:.2f})</span></div>
    <div class="kpi"><b>{hyb[f'precision@{k}']:.3f}</b><span>hybrid precision@{k}, genre-blind ({hyb[f'precision@{k}'] / rnd[f'precision@{k}']:.1f}× random)</span></div>
    <div class="kpi"><b>{rmse['Hybrid (k-NN + author mean)']['rmse']:.3f}</b><span>rating RMSE vs {rmse['Global mean (baseline)']['rmse']:.3f} for the mean</span></div>
  </div>
</div>

<h2>1. Problem statement and business use cases</h2>
<p>Design a book recommendation system that cleans and merges two Audible datasets, applies NLP and clustering,
builds several recommendation models, compares them with evaluation metrics, and serves recommendations through a
user-friendly Streamlit interface deployed on the cloud.</p>
<ul>
<li><b>Personalised reading.</b> "Similar books" explains every pick; "For you" works from genres, sub-genres,
authors or liked books, with no history needed.</li>
<li><b>Libraries and bookstores.</b> Top-rated, most-reviewed and hidden-gem shelves to drive borrowing and sales.</li>
<li><b>Authors and publishers.</b> Which genres have the most titles versus the most listeners, how ratings behave,
and how pricing tracks length.</li>
<li><b>Reader engagement.</b> Trending and highly rated books across 17 genres.</li>
</ul>

<h2>2. Data preparation and cleaning</h2>
<p><code>Audible_Catlog.csv</code> ({rep['raw']['catalog']['rows']:,} rows: name, author, rating, reviews, price) and
<code>Audible_Catlog_Advanced_Features.csv</code> ({rep['raw']['advanced']['rows']:,} rows: the same plus description, listening
time, ranks and genre). Note that the brief describes these the other way round: the "advanced" file is the one
with descriptions and genres.</p>
<table><thead><tr><th>Problem found</th><th>Count</th><th>Action</th></tr></thead><tbody>
<tr><td>Exact duplicate rows</td><td>{rep['dedup_catalog']['exact_duplicates']:,} + {rep['dedup_advanced']['exact_duplicates']:,}</td><td>Dropped</td></tr>
<tr><td>Same book scraped twice with different values</td><td>{rep['dedup_catalog']['same_book_duplicates'] + rep['dedup_advanced']['same_book_duplicates']:,}</td><td>Kept the most complete copy, then the most reviewed</td></tr>
<tr><td>Author spelled two ways for the same title</td><td>{rep['author_spellings_unified']}</td><td>Fuzzy match (≥ {config.AUTHOR_FUZZY_MATCH}) before merging</td></tr>
<tr><td>Rating −1 (always with missing reviews)</td><td>{mr['catalog_unrated_(rating_-1)']:,} + {mr['advanced_unrated_(rating_-1)']:,}</td><td>Marked unrated, never averaged in</td></tr>
<tr><td>Descriptions that are scraping failures: CAPTCHA pages, Amazon error pages, checkout text, publisher stubs, award one-liners</td><td>{mr['advanced_description_junk']:,}</td><td>Treated as missing</td></tr>
<tr><td>Ranks &amp; Genre / listening time missing</td><td>{mr['advanced_ranks_and_genre_missing']:,} / {mr['advanced_listening_time_missing']:,}</td><td>Left missing; genre inferred by NLP where confident</td></tr>
<tr><td>Prices 10× too high (dropped decimal)</td><td>{fc['price_decimal_shift_corrected']}</td><td>Divided by 10 and flagged (same length as other books, 14× the price per hour)</td></tr>
<tr><td>876 fine genres in one rank string</td><td>n/a</td><td>Parsed (split on <code>,#</code>), mapped to 17 broad genres by ordered keyword rules</td></tr>
</tbody></table>
<p><b>Merge:</b> outer join on normalised title + author. The advanced (later) scrape wins where both files have a
value. Result: {rep['merge']['books']:,} books ({rep['merge']['in_both_files']:,} in both files, {rep['merge']['only_catalog']:,}
only in the basic one, {rep['merge']['only_advanced']:,} only in the advanced one). Coverage after cleaning:
{fc['rated']:,} rated, {fc['with_description']:,} with a real description, {fc['with_listed_genre']:,} with a listed genre.</p>
<p><b>Derived:</b> Bayesian weighted rating (m = {config.BAYES_MIN_REVIEWS} reviews), popularity (log reviews),
hidden-gem flag (rating ≥ {config.HIDDEN_GEM_MIN_RATING}, 5–65 reviews), listening minutes, overall Audible rank.</p>

<h2>3. Exploratory data analysis</h2>
{qa(['Q1'])}
{fig('q1_genres', 'Q1. Books and total reviews per genre')}
{qa(['Q2'])}
{fig('q2_top_authors', 'Q2. Highest-rated authors')}
{qa(['Q3'])}
{fig('q3_ratings', 'Q3. Rating distribution and rating by genre')}
{qa(['Q4'])}
{fig('popularity_curve', 'Q4 substitute: rating and reviews along the Audible bestseller rank (no publication dates exist)')}
{qa(['Q5'])}
{fig('q5_ratings_vs_reviews', 'Q5. Ratings vs number of reviews')}
{fig('correlations', 'Correlations between numeric features')}

<h2 class="page">4. NLP and clustering</h2>
<h3>Text features</h3>
<p>Title (twice) + cleaned description → TF-IDF (1–2-grams, sublinear tf; English, audiobook and publisher stop
words such as "Penguin presents" and "Audie award").</p>
<div class="box"><b>Genre inference.</b> Logistic regression on TF-IDF, trained on {gi['trained_on']:,} books with known genres
(17 classes). 5-fold CV accuracy {gi['cv_accuracy']:.1%}, macro-F1 {gi['cv_macro_f1']:.3f}, top-3 accuracy {gi['cv_top3_accuracy']:.1%}.
The confidence cut-off ({gi['min_probability']:.2f}) is chosen on CV so the predictions kept are right
{gi['cv_accuracy_when_confident']:.0%} of the time. That filled in {gi['inferred']} of {gi['candidates']} books that had a description but no
genre; the rest stay "Unknown".</div>
<h3>Clustering</h3>
<p>The text and genre blocks are reduced with TruncatedSVD to {config.SVD_COMPONENTS} topics. Clusters are learned on the
{cl['fitted_on_books']:,} books with listed genres (KMeans, k chosen by silhouette over 8–30). Every other book then
joins the nearest cluster centroid in a text-only space. Clustering everything together left the genre-less books
in one leftover cluster.</p>
{qa(['Q6'])}
{fig('silhouette', 'Silhouette score by k')}
{fig('clusters_map', 't-SNE map of the 16 clusters')}
{table(clusters_rows, [('c', '#', ''), ('name', 'Cluster', ''), ('size', 'Books', ''), ('fitted', 'Fitted', ''), ('ex', 'Examples', '')])}
<h3>DBSCAN comparison</h3>
{table(cl['dbscan'], [('eps', 'eps', ''), ('clusters', 'Clusters', ''), ('noise_share', 'Noise', '.1%'),
                      ('largest_cluster_share', 'Largest cluster', '.1%'), ('silhouette_clustered_only', 'Silhouette', '.3f'),
                      ('agreement_with_kmeans_ari', 'Agreement with KMeans (ARI)', '.2f')],
       highlight=lambda r: r is best_db)}

<h2 class="page">5. Recommendation models and evaluation</h2>
<table><thead><tr><th>Model</th><th>How it scores a candidate</th></tr></thead><tbody>
<tr><td>Content-based</td><td>Cosine similarity over weighted blocks: text {config.TEXT_WEIGHT}, broad genre {config.GENRE_WEIGHT}, fine genres {config.GENRE_WEIGHT}, author {config.AUTHOR_WEIGHT}</td></tr>
<tr><td>Clustering-based</td><td>Books in the query's cluster, ranked by weighted rating (title-only cluster members down-weighted)</td></tr>
<tr><td>Hybrid</td><td>{w['content']:.2f} × similarity + {w['cluster']:.2f} × same cluster + {w['quality']:.2f} × weighted rating; weights grid-searched on validation queries</td></tr>
</tbody></table>
<div class="box"><b>Evaluation design.</b> The data has no per-user ratings, so a recommendation counts as <b>relevant</b> when it
shares at least one of the query book's 876 fine Audible genres (<b>strictly relevant</b>: two). Test set:
{hyb['queries']} books with listed genres, top {k} recommendations each, seed {m['eval_settings']['seed']} (tuning used a different seed).
<b>Genre-blind</b> models use only titles and descriptions, so they cannot see the labels; they are the fair comparison.
Full-feature (production) models also use the fine genres and are an optimistic upper bound.</div>
{table(m['model_comparison'], metric_cols, highlight=lambda r: r['model'] == 'Hybrid (genre-blind)')}
{fig('model_comparison', 'Genre-blind comparison and feature ablation')}
{qa(['Q7'])}
<h3>Which features matter? Rating prediction (RMSE)</h3>
{table(m['feature_ablation'], [('features', 'Features', ''), (f'precision@{k}', f'Precision@{k}', '.3f'), (f'ndcg@{k}', f'nDCG@{k}', '.3f'),
                               (f'strict_precision@{k}', 'Strict P@10', '.3f')], highlight=lambda r: r['features'] == 'text + broad genre + author')}
{table(m['rating_prediction'], [('model', 'Rating predictor', ''), ('rmse', 'RMSE', '.4f'), ('mae', 'MAE', '.4f')],
       highlight=lambda r: r['model'].startswith('Hybrid'))}
{qa(['Q9', 'Q8'])}
{fig('q8_author_popularity', 'Q8. Rating vs author popularity')}

<h2>6. Scenario questions</h2>
{qa(['S1', 'S2', 'S3'])}

<h2 class="page">7. Streamlit application</h2>
<p>Seven pages:</p>
<ul>
<li><b>Discover:</b> KPIs plus top-rated, hidden-gem and most-reviewed shelves.</li>
<li><b>Similar books:</b> pick a book and a model, get explained recommendations, and compare the three models side by side.</li>
<li><b>For you:</b> genres, sub-genres, authors and liked books, with quick-start scenarios.</li>
<li><b>Explore:</b> the EDA, interactive.</li>
<li><b>Clusters:</b> map, cluster browser, KMeans vs DBSCAN.</li>
<li><b>Model performance.</b></li>
<li><b>About.</b></li>
</ul>
<p>Every recommendation page has filters: genre, minimum rating and reviews, price, length, unrated books and
hidden gems.</p>
{shot('01_discover', 'Discover: KPIs and shelves')}
{shot('02_similar_books', 'Similar books: The Martian, hybrid model, with reasons')}
{shot('03_compare_models', 'The three models side by side for the same book')}
{shot('04_for_you_thriller', 'For you: the thriller-fan quick start')}
{shot('07_clusters', 'Clusters page')}
{shot('08_model_performance', 'Model performance page')}

<h2>8. Deployment</h2>
<ul>
<li><b>Streamlit Community Cloud.</b> The app runs from the GitHub repo (<code>app/streamlit_app.py</code>, pinned
<code>app/requirements.txt</code>).{f' Live at <a href="{live_url}">{live_url}</a>.' if live_url else ''}</li>
<li><b>AWS EC2 + Docker</b> (<code>Dockerfile</code>, <code>deploy/</code>):
  <ul>
  <li>An Amazon Linux 2023 t3.micro runs a start-up script that installs Docker, builds the image and serves the app on
  port 80.{f' Live at <a href="{aws_url}">{aws_url}</a>.' if aws_url else ''}</li>
  <li>The container restarts on its own, and a health check watches Streamlit.</li>
  <li>The image is stateless (the model is baked in), so it scales horizontally: an Auto Scaling group behind an
  Application Load Balancer, or Elastic Beanstalk's Docker platform, as documented in
  <code>deploy/AWS_DEPLOY.md</code>.</li>
  </ul></li>
</ul>

<h2>9. Insights, limitations and next steps</h2>
<ol>
<li><b>Data quality was the main problem.</b> About 2,000 descriptions were scrape failures, and most books have
no genre. Treating these honestly as missing, and inferring genre only where confident, mattered more than the choice of
model.</li>
<li><b>Ratings are compressed</b> (mean 4.46, 61% at 4.5+), so weighted rating is essential. A raw 5.0 is usually a
small sample.</li>
<li><b>Popularity is not quality.</b> That makes hidden-gem recommendations a real opportunity for libraries and
publishers.</li>
<li><b>Genres are the strongest signal.</b> Adding the broad genre to description text nearly doubles precision,
and the hybrid's quality term keeps recommendations well rated.</li>
</ol>
<ul>
<li><b>No user data:</b> collaborative filtering and true personalisation metrics need per-user ratings.</li>
<li><b>No publication dates:</b> join them in from Open Library / Google Books to analyse trends.</li>
<li><b>Next:</b> sentence embeddings (e.g. MiniLM) instead of TF-IDF for short descriptions; re-scrape the blocked pages.</li>
</ul>

<h2>10. Deliverables</h2>
<table><tbody>
<tr><td>Merged and cleaned dataset</td><td><code>data/processed/books_clean.csv</code>, <code>cleaning_report.json</code></td></tr>
<tr><td>Scripts</td><td><code>pipeline.py</code>, <code>train.py</code>, <code>src/</code> (cleaning, NLP, recommenders, evaluation, EDA)</td></tr>
<tr><td>EDA notebook</td><td><code>notebooks/eda.ipynb</code>: all 12 questions plus my own analysis</td></tr>
<tr><td>Models and metrics</td><td><code>models/</code>: recommender, genre classifier, metrics, cluster summary</td></tr>
<tr><td>Streamlit app</td><td><code>app/</code>{f' · <a href="{live_url}">{live_url}</a>' if live_url else ''}</td></tr>
<tr><td>AWS deployment</td><td><code>Dockerfile</code>, <code>deploy/ec2_user_data.sh</code>, <code>deploy/AWS_DEPLOY.md</code>{f' · <a href="{aws_url}">{aws_url}</a>' if aws_url else ''}</td></tr>
<tr><td>Documentation</td><td><code>README.md</code>, this report, <code>reports/screenshots/</code></td></tr>
</tbody></table>
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-url", default=None, help="Streamlit Cloud URL, once deployed")
    parser.add_argument("--aws-url", default=None, help="AWS URL (http://<ec2-public-ip>), once deployed")
    args = parser.parse_args()
    page_html = build_html(args.live_url, args.aws_url)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_content(page_html, wait_until="load")
        page.pdf(path=str(PDF_PATH), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;color:#94a3b8;width:100%;text-align:center;">'
                                 'Audible Insights · <span class="pageNumber"></span>/<span class="totalPages"></span></div>')
        browser.close()
    print(f"Saved {PDF_PATH.relative_to(ROOT)} ({PDF_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
