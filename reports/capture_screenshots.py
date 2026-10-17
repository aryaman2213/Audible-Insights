"""Capture full-page screenshots of the running Streamlit app into reports/screenshots/.

    streamlit run app/streamlit_app.py --server.port 8521        # in another terminal
    python reports/capture_screenshots.py --url http://localhost:8521

Uses Playwright with the locally installed Google Chrome (pip install playwright; no browser download).
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"
WIDTH = 1440


def settle(page, ms=3500):
    """Wait for Streamlit to finish running and Plotly to draw."""
    page.wait_for_selector('[data-testid="stMain"]', timeout=60_000)
    page.wait_for_function('!document.querySelector(\'[data-testid="stStatusWidget"] [data-testid="stSpinner"]\')',
                           timeout=120_000)
    page.wait_for_timeout(ms)


def full_page(page, name, max_height=None):
    """Streamlit scrolls inside its own container, so grow the viewport to the content height first."""
    height = page.evaluate("""() => Math.max(...['[data-testid="stMain"]', '[data-testid="stMainBlockContainer"]']
        .map(s => document.querySelector(s)).filter(Boolean).map(e => e.scrollHeight))""")
    if max_height:
        height = min(height, max_height)
    page.set_viewport_size({"width": WIDTH, "height": int(height) + 40})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{name}.png"))
    page.set_viewport_size({"width": WIDTH, "height": 900})
    print("saved", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8521")
    base = parser.parse_args().url.rstrip("/")
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": WIDTH, "height": 900}, device_scale_factor=1)

        page.goto(base)
        settle(page, 6000)
        full_page(page, "01_discover")

        page.goto(f"{base}/similar")
        settle(page, 3000)
        page.get_by_role("button", name="The Martian").click()
        settle(page, 4000)
        full_page(page, "02_similar_books", max_height=2200)
        page.get_by_role("tab", name="Compare the three models").click()
        settle(page, 2000)
        full_page(page, "03_compare_models")

        page.goto(f"{base}/for-you")
        settle(page, 3000)
        page.get_by_role("button", name="Thriller fan").click()
        settle(page, 4000)
        full_page(page, "04_for_you_thriller", max_height=2200)

        page.goto(f"{base}/explore")
        settle(page, 5000)
        full_page(page, "05_explore_genres")
        page.get_by_role("tab", name="Ratings & reviews").click()
        settle(page, 3000)
        full_page(page, "06_explore_ratings")

        page.goto(f"{base}/clusters")
        settle(page, 6000)
        full_page(page, "07_clusters")

        page.goto(f"{base}/models")
        settle(page, 5000)
        full_page(page, "08_model_performance")

        page.goto(f"{base}/about")
        settle(page, 2500)
        full_page(page, "09_about")
        browser.close()


if __name__ == "__main__":
    main()
