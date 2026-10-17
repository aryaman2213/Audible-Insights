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
.books { display: grid; grid-template-columns: repeat(auto-fill, minmax(17.5rem, 1fr)); gap: 0.8rem; margin: 0.3rem 0 1rem 0; }
.book { background: #ffffff; border: 1px solid #e6e8ef; border-radius: 14px; padding: 0.85rem 1rem 0.8rem 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); display: flex; flex-direction: column; gap: 0.28rem; }
.book .top { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
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
        chip = f'<span class="{chip_cls}" title="{html.escape(str(b.get("genres") or genre))}">{html.escape(genre if genre != "Unknown" else "Genre not listed")}</span>'
        if b.get("hidden_gem"):
            chip += ' <span class="chip gem">hidden gem</span>'
        meta = []
        if b["is_rated"]:
            meta.append(f'<b>★ {b["rating"]:.1f}</b> · {compact(b["num_reviews"])} reviews')
        else:
            meta.append("Not rated yet")
        meta.append(inr(b["price"]))
        if hours(b["listening_minutes"]):
            meta.append(hours(b["listening_minutes"]))
        desc = b["description"] if isinstance(b["description"], str) else ""
        if len(desc) > desc_chars:
            desc = desc[:desc_chars].rsplit(" ", 1)[0] + "…"
        why_html = f'<div class="why">{html.escape(why[idx])}</div>' if why and why.get(idx) else ""
        cards.append(
            f'<div class="book"><div class="top"><div>{chip}</div>'
            f'{f"<span class=rank>#{rank}</span>" if ranked else ""}</div>'
            f'<div class="title">{html.escape(b["title"])}</div><div class="author">by {html.escape(b["author"])}</div>'
            f'<div class="meta">{" · ".join(meta)}</div>'
            f'{f"<div class=desc>{html.escape(desc)}</div>" if desc else ""}{why_html}</div>')
    st.markdown(f'<div class="books">{"".join(cards)}</div>', unsafe_allow_html=True)
