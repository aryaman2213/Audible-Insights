"""Audible Insights (entry point).

Run from the project root:
    streamlit run app/streamlit_app.py

Sets up the theme and navigation. Pages live in views.py; data and model access in data.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

import streamlit as st

import config
import src.eda
import src.recommenders

# Streamlit only hot-reloads modules inside app/. config.py and src/ live at the project root, so after a
# redeploy (Streamlit Cloud pulls new code without restarting) they could be stale; reload them each run.
importlib.reload(config)
importlib.reload(src.recommenders)
importlib.reload(src.eda)

import data  # noqa: E402
import ui  # noqa: E402
import views  # noqa: E402

st.set_page_config(page_title="Audible Insights", page_icon=":material/headphones:", layout="wide")
ui.install_theme()

missing = [p for p in data.ARTIFACTS if not p.exists()]
if missing:
    st.error("Data or models not found. From the project root run `python pipeline.py` and then `python train.py`.")
    st.stop()

pages = {
    "home": st.Page(views.home, title="Discover", icon=":material/headphones:", default=True),
    "similar": st.Page(views.similar, title="Similar books", icon=":material/auto_stories:", url_path="similar"),
    "for_you": st.Page(views.for_you, title="For you", icon=":material/person_search:", url_path="for-you"),
    "explore": st.Page(views.explore, title="Explore", icon=":material/insights:", url_path="explore"),
    "clusters": st.Page(views.clusters, title="Clusters", icon=":material/bubble_chart:", url_path="clusters"),
    "performance": st.Page(views.performance, title="Model performance", icon=":material/model_training:",
                           url_path="models"),
    "about": st.Page(views.about, title="About", icon=":material/info:", url_path="about"),
}
st.session_state["pages"] = pages
st.navigation(list(pages.values()), position="top").run()
