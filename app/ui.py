"""Look and feel: CSS, the Plotly template, book cards and small layout helpers shared by every page."""
import html
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
INK, MUTED, GRID = "#0f172a", "#64748b", "#eef0f4"
ACCENT = "#4f46e5"
GOOD, BAD, WARN = "#059669", "#e11d48", "#d97706"
CHART_COLORS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48", "#8b5cf6", "#64748b"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, [data-testid="stMetric"], [data-testid="stWidgetLabel"], input, textarea,
[data-baseweb="select"], [data-testid="stSidebar"] p { font-family: %(font)s; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; color: %(ink)s; }

.hero { padding: 0.25rem 0 1.1rem 0; }
.hero .eyebrow { color: %(accent)s; font-weight: 600; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
.hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0.15rem 0 0.35rem 0; padding: 0; }
.hero p { color: %(muted)s; font-size: 0.98rem; margin: 0; max-width: 62rem; }
.pills { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.8rem; }
.pill { background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 0.22rem 0.7rem; font-size: 0.78rem; font-weight: 500; }
.pill.grey { background: #f1f5f9; color: #334155; }

[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e6e8ef; border-radius: 14px;
    padding: 0.9rem 1rem 0.8rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
[data-testid="stMetricLabel"] p { color: %(muted)s; font-size: 0.8rem; font-weight: 500; white-space: normal; }
[data-testid="stMetricLabel"] > div { overflow: visible; }
[data-testid="stMetricValue"] { color: %(ink)s; font-size: clamp(1.05rem, 1.55vw, 1.55rem); font-weight: 700; }
[data-testid="stMetricValue"] > div { white-space: nowrap; overflow: visible; }

[class*="st-key-card-"] { background: #ffffff; border-radius: 14px !important; border-color: #e6e8ef !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); padding: 1rem 1.1rem 0.6rem 1.1rem !important; }
.card-title { font-weight: 650; font-size: 0.98rem; color: %(ink)s; margin: 0 0 0.1rem 0; }
.card-sub { color: %(muted)s; font-size: 0.8rem; margin: 0 0 0.2rem 0; }

.insight { background: #ffffff; border: 1px solid #e6e8ef; border-left: 4px solid %(accent)s; border-radius: 12px;
    padding: 0.85rem 1rem; height: 100%%; }
.insight .num { font-size: 1.35rem; font-weight: 750; color: %(ink)s; }
.insight .txt { color: #334155; font-size: 0.88rem; line-height: 1.45; margin-top: 0.15rem; }


/* book cards */
.books { display: grid; grid-template-columns: repeat(auto-fill, minmax(19rem, 1fr)); gap: 0.85rem; margin: 0.3rem 0 1rem 0; }
.book { background: #ffffff; border: 1px solid #e6e8ef; border-radius: 14px; padding: 0.85rem 0.95rem 0.8rem 0.85rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); display: flex; gap: 0.85rem;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.book:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08); border-color: #d5d9e5; }
.book .cover { flex: 0 0 3.6rem; height: 5.1rem; border-radius: 6px 9px 9px 6px; color: #fff; display: flex;
    flex-direction: column; justify-content: space-between; padding: 0.35rem 0.4rem; position: relative;
    box-shadow: inset 4px 0 0 rgba(0,0,0,0.18), 0 2px 6px rgba(15,23,42,0.15); overflow: hidden; }
.book .cover .ini { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; line-height: 1; }
.book .cover .gl { font-size: 0.52rem; font-weight: 600; opacity: 0.85; line-height: 1.15; text-transform: uppercase;
    letter-spacing: 0.04em; }
.book .body { display: flex; flex-direction: column; gap: 0.26rem; min-width: 0; flex: 1; }
.book .top { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
.stars { position: relative; display: inline-block; color: #e2e8f0; letter-spacing: 1px; font-size: 0.82rem; line-height: 1; }
.stars span { position: absolute; left: 0; top: 0; overflow: hidden; white-space: nowrap; color: #f59e0b; }

/* home hero band */
.band { background: linear-gradient(120deg, #312e81 0%%, #4f46e5 55%%, #0ea5e9 120%%); border-radius: 20px;
    padding: 1.7rem 1.8rem 1.5rem 1.8rem; color: #fff; margin-bottom: 0.4rem; }
.band .eyebrow { color: #c7d2fe; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }
.band h1 { color: #fff; font-size: 2.3rem; font-weight: 800; margin: 0.2rem 0 0.4rem 0; padding: 0; letter-spacing: -0.03em; }
.band p { color: #e0e7ff; font-size: 1rem; margin: 0; max-width: 56rem; }
.band .pills { margin-top: 0.9rem; }
.band .pill { background: rgba(255,255,255,0.16); color: #fff; }
.st-key-home-search { margin-top: -0.2rem; }
.st-key-dl-similar, .st-key-dl-for-you { display: flex; justify-content: flex-end; }
.results-head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; margin: 0; }
.results-head b { font-size: 1.02rem; color: %(ink)s; }
.results-head span { color: %(muted)s; font-size: 0.85rem; }
.footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e6e8ef; color: %(muted)s; font-size: 0.8rem;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
.footer a { color: %(accent)s; text-decoration: none; }
.chip { background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 0.12rem 0.55rem; font-size: 0.72rem;
    font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 75%%; }
.chip.unknown { background: #f1f5f9; color: #64748b; }
.chip.gem { background: #ecfdf5; color: #047857; }
.book .rank { color: %(muted)s; font-size: 0.75rem; font-weight: 600; }
.book .title { font-weight: 650; font-size: 0.96rem; color: %(ink)s; line-height: 1.3; }
.book .author { color: %(muted)s; font-size: 0.82rem; }
.book .meta { color: #334155; font-size: 0.8rem; }
.book .meta b { color: #b45309; }
.book .desc { color: #475569; font-size: 0.8rem; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden; }
.book .why { color: %(accent)s; font-size: 0.76rem; font-weight: 500; border-top: 1px dashed #e2e8f0; padding-top: 0.35rem;
    margin-top: auto; }
.empty { background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; border-radius: 12px; padding: 0.8rem 1rem; font-size: 0.9rem; }

section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e8ef; }
@media (min-width: 992px) {
  section[data-testid="stSidebar"][aria-expanded="true"] { width: 310px !important; min-width: 310px !important; }
}
@media (max-width: 1180px) {
  [data-testid="stMetric"] { padding: 0.75rem 0.7rem 0.65rem 0.7rem; }
  [data-testid="stMetricValue"] { font-size: clamp(0.98rem, 1.7vw, 1.3rem); }
}
@media (max-width: 640px) {
  .block-container { padding-top: 3.6rem; padding-left: 1rem; padding-right: 1rem; }
  .hero h1 { font-size: 1.6rem; }
  .band { padding: 1.2rem 1.1rem; border-radius: 16px; }
  .band h1 { font-size: 1.7rem; }
  .books { grid-template-columns: 1fr; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) { flex-wrap: wrap; gap: 0.6rem; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
      min-width: calc(50%% - 0.3rem) !important; flex: 1 1 calc(50%% - 0.3rem) !important; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
}
section[data-testid="stSidebar"] h2 { font-size: 1.05rem; }
.sidebar-note { color: %(muted)s; font-size: 0.78rem; line-height: 1.45; }
</style>
""" % {"font": FONT, "ink": INK, "muted": MUTED, "accent": ACCENT}


# Hover tooltips and drag-to-zoom stay on; the floating toolbar is hidden because it covers legends on phones.
PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}


def install_theme():
    st.markdown(CSS, unsafe_allow_html=True)
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=12.5),
        colorway=CHART_COLORS,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#d7dbe3", ticks=""),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#d7dbe3", ticks=""),
        hoverlabel=dict(bgcolor="white", bordercolor="#d7dbe3", font=dict(family=FONT, color=INK)),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11.5)),
        bargap=0.25,
    )
    pio.templates["audible"] = template
    pio.templates.default = "plotly_white+audible"
    px.defaults.template = "plotly_white+audible"


def hero(eyebrow, title, subtitle, pills=()):
    pill_html = "".join(f'<span class="pill{" grey" if i else ""}">{p}</span>' for i, p in enumerate(pills))
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p>'
        f'{f"<div class=pills>{pill_html}</div>" if pills else ""}</div>',
        unsafe_allow_html=True,
    )


def band(eyebrow, title, subtitle, pills=()):
    pill_html = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="band"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p>'
                f'{f"<div class=pills>{pill_html}</div>" if pills else ""}</div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: (label, value, help[, delta[, delta_color]]) tuples rendered as cards."""
    for col, item in zip(st.columns(len(items)), items):
        label, value, help_text, *rest = item
        delta = rest[0] if rest else None
        delta_color = rest[1] if len(rest) > 1 else "normal"
        col.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def card(title):
    return st.container(border=True, key="card-" + slug(title))


def chart_card(fig, title, subtitle=None, height=360, tickangle=0):
    with card(title):
        st.markdown(f'<p class="card-title">{title}</p>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<p class="card-sub">{subtitle}</p>', unsafe_allow_html=True)
        fig.update_layout(template="plotly_white+audible", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=height,
                          margin=dict(l=4, r=8, t=34, b=4), title=None, legend_title_text="",
                          uniformtext_minsize=10, uniformtext_mode="hide")
        fig.update_xaxes(tickangle=tickangle, automargin=True)
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key="chart-" + slug(title))


def insight_cards(items):
    """items: (headline number, sentence) pairs."""
    for col, (number, text) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="insight"><div class="num">{number}</div><div class="txt">{text}</div></div>',
                     unsafe_allow_html=True)


def inr(value):
    return "–" if value is None or value != value else ("Free" if value == 0 else f"₹{value:,.0f}")


def hours(minutes):
    if minutes is None or minutes != minutes:
        return None
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m:02d} m" if h else f"{m} m"


def compact(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{int(n)}"


# One colour pair per broad genre, used for the book "covers".
GENRE_COLORS = {
    "Business & Careers": ("#1e3a8a", "#3b82f6"), "Children & Teens": ("#c2410c", "#fb923c"),
    "Literature & Fiction": ("#7c2d12", "#d97706"), "Personal Development": ("#047857", "#34d399"),
    "Religion & Spirituality": ("#6b21a8", "#c084fc"), "Psychology, Philosophy & Society": ("#334155", "#94a3b8"),
    "Science & Technology": ("#0e7490", "#22d3ee"), "Mystery, Thriller & Horror": ("#111827", "#4b5563"),
    "Money & Finance": ("#14532d", "#4ade80"), "Health & Wellness": ("#0f766e", "#5eead4"),
    "Biographies & Memoirs": ("#78350f", "#b45309"), "History": ("#713f12", "#ca8a04"),
    "Science Fiction & Fantasy": ("#312e81", "#818cf8"), "Romance": ("#9d174d", "#f472b6"),
    "Language & Education": ("#1d4ed8", "#93c5fd"), "Arts, Sports & Leisure": ("#b91c1c", "#f87171"),
    "Comedy & Humour": ("#a16207", "#facc15"), "Unknown": ("#475569", "#94a3b8"),
}


GENRE_SHORT = {
    "Business & Careers": "Business", "Children & Teens": "Children", "Literature & Fiction": "Fiction",
    "Personal Development": "Self-help", "Religion & Spirituality": "Spiritual",
    "Psychology, Philosophy & Society": "Society", "Science & Technology": "Science",
    "Mystery, Thriller & Horror": "Thriller", "Money & Finance": "Finance", "Health & Wellness": "Health",
    "Biographies & Memoirs": "Memoir", "History": "History", "Science Fiction & Fantasy": "Sci-fi",
    "Romance": "Romance", "Language & Education": "Learning", "Arts, Sports & Leisure": "Leisure",
    "Comedy & Humour": "Comedy", "Unknown": "Audible",
}


def unique_genres(s) -> list:
    """'A | B | A' -> ['A', 'B'] (the source lists some genres twice, under different ranks)."""
    return list(dict.fromkeys(g.strip() for g in s.split("|") if g.strip())) if isinstance(s, str) else []


def initials(title: str) -> str:
    words = [w for w in re.split(r"[\s:,\-–—]+", re.sub(r"\(.*?\)|\[.*?\]", "", title)) if w and w[0].isalnum()]
    skip = {"the", "a", "an", "of", "and", "to", "in", "on", "for"}
    main = [w for w in words if w.lower() not in skip] or words
    return "".join(w[0].upper() for w in main[:2]) or "?"


def stars(rating: float) -> str:
    pct = max(0.0, min(1.0, rating / 5)) * 100
    return f'<span class="stars" aria-hidden="true">★★★★★<span style="width:{pct:.0f}%">★★★★★</span></span>'


def footer():
    st.markdown(
        '<div class="footer"><span>Audible Insights · GUVI × HCL capstone by Aryaman Bhatnagar</span>'
        '<span>Data: Audible India catalogue (6,056 books) · '
        '<a href="https://github.com/aryaman2213/audible-insights" target="_blank">Source on GitHub</a></span></div>',
        unsafe_allow_html=True)


def results_header(n: int, text: str):
    st.markdown(f'<div class="results-head"><b>{n} recommendation{"s" if n != 1 else ""}</b>'
                f'<span>{html.escape(text)}</span></div>', unsafe_allow_html=True)


def csv_download(df, file_name: str, key: str):
    cols = ["title", "author", "primary_genre", "rating", "num_reviews", "price", "listening_minutes"]
    out = df[[c for c in cols if c in df]].rename(columns={"primary_genre": "genre", "num_reviews": "reviews",
                                                           "price": "price_inr"})
    st.download_button("Download list (CSV)", out.to_csv(index=False).encode("utf-8"), file_name=file_name,
                       mime="text/csv", icon=":material/download:", key=key)


def book_cards(df, why=None, ranked=True, desc_chars=240):
    """Render books as a responsive card grid. why: optional {book index: explanation} shown at the foot of a card."""
    if df.empty:
        st.markdown('<div class="empty">No books match these settings. Loosen a filter and try again.</div>',
                    unsafe_allow_html=True)
        return
    cards = []
    for rank, (idx, b) in enumerate(df.iterrows(), start=1):
        genre = b["primary_genre"]
        chip_cls = "chip unknown" if genre == "Unknown" else "chip"
        chip = f'<span class="{chip_cls}" title="{html.escape(" · ".join(unique_genres(b.get("genres"))) or genre)}">{html.escape(genre if genre != "Unknown" else "Genre not listed")}</span>'
        if b.get("hidden_gem"):
            chip += ' <span class="chip gem">hidden gem</span>'
        meta = []
        if b["is_rated"]:
            meta.append(f'{stars(b["rating"])} <b>{b["rating"]:.1f}</b> · {compact(b["num_reviews"])} reviews')
        else:
            meta.append("Not rated yet")
        meta.append(inr(b["price"]))
        if hours(b["listening_minutes"]):
            meta.append(hours(b["listening_minutes"]))
        desc = b["description"] if isinstance(b["description"], str) else ""
        if len(desc) > desc_chars:
            desc = desc[:desc_chars].rsplit(" ", 1)[0] + "…"
        why_html = f'<div class="why">{html.escape(why[idx])}</div>' if why and why.get(idx) else ""
        c1, c2 = GENRE_COLORS.get(genre, GENRE_COLORS["Unknown"])
        cover = (f'<div class="cover" aria-hidden="true" style="background:linear-gradient(160deg,{c1},{c2})">'
                 f'<span class="ini">{html.escape(initials(b["title"]))}</span>'
                 f'<span class="gl">{GENRE_SHORT.get(genre, "Audible")}</span></div>')
        cards.append(
            f'<div class="book">{cover}<div class="body"><div class="top"><div>{chip}</div>'
            f'{f"<span class=rank>#{rank}</span>" if ranked else ""}</div>'
            f'<div class="title">{html.escape(b["title"])}</div><div class="author">by {html.escape(b["author"])}</div>'
            f'<div class="meta">{" · ".join(meta)}</div>'
            f'{f"<div class=desc>{html.escape(desc)}</div>" if desc else ""}{why_html}</div></div>')
    st.markdown(f'<div class="books">{"".join(cards)}</div>', unsafe_allow_html=True)
