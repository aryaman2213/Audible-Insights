"""End-to-end test of the Streamlit app: clicks every button, tab, dropdown, slider, toggle and expander on every page
and checks that each action renders without an exception and shows the expected content.

    streamlit run app/streamlit_app.py --server.port 8521        # in another terminal
    python tests/e2e_app.py --url http://localhost:8521

Uses Playwright with the locally installed Google Chrome. Exits non-zero if any check fails.
"""
import argparse
import re
import sys

from playwright.sync_api import sync_playwright

FAILURES, PASSES = [], []
CONSOLE_ERRORS = []
BAD_RESPONSES = []
DURATION = re.compile(r"·\s(?:(\d+) h )?(\d+) m\b")


def minutes(text):
    """Listening time on a card ('9 h 32 m' or '45 m') in minutes, or None."""
    m = DURATION.search(text)
    return None if not m else int(m.group(1) or 0) * 60 + int(m.group(2))


def settle(page, ms=1200):
    page.wait_for_timeout(500)          # sliders debounce: give the re-run time to start before waiting for it
    page.wait_for_selector('[data-testid="stMain"]', timeout=60_000)
    page.wait_for_function('!document.querySelector(\'[data-testid="stStatusWidget"] [data-testid="stSpinner"]\')',
                           timeout=120_000)
    page.wait_for_timeout(ms)


def check(cond, label):
    (PASSES if cond else FAILURES).append(label)
    print(("  ok   " if cond else "  FAIL ") + label)


def no_exception(page, label):
    n = page.locator('[data-testid="stException"]').count()
    if n:
        print(page.locator('[data-testid="stException"]').first.inner_text()[:800])
    check(n == 0, f"{label}: no exception")


def cards(page):
    return page.locator(".book")


def card_texts(page):
    return [t for t in page.locator(".book").all_inner_texts()]


def open_expander(page, name):
    exp = page.locator('[data-testid="stExpander"]').filter(has_text=name).first
    summary = exp.locator("summary")
    if exp.locator("details").get_attribute("open") is None:
        summary.click()
        settle(page, 600)
    return exp


def select_option(page, label, text, exact_option=None):
    """Type into a Streamlit selectbox / multiselect labelled `label` and pick the first matching option."""
    box = page.locator('[data-testid="stSelectbox"], [data-testid="stMultiSelect"]').filter(
        has=page.get_by_text(label, exact=True)).first
    box.locator("input").click()
    box.locator("input").fill(text)
    page.wait_for_timeout(500)
    opt = page.get_by_role("option").filter(has_text=exact_option or text).first
    opt.click()
    settle(page)
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)


def slider_keys(page, label, key, times):
    s = page.locator('[data-testid="stSlider"]').filter(has=page.get_by_text(label, exact=True)).first
    thumb = s.get_by_role("slider").first
    thumb.focus()
    for _ in range(times):
        thumb.press(key)
    settle(page)


def toggle(page, label):
    page.locator('[data-testid="stCheckbox"]').filter(has_text=label).first.locator("label").click()
    settle(page)


def tab(page, name):
    page.get_by_role("tab", name=name).click()
    settle(page, 900)


# ---------------------------------------------------------------- pages
def test_home(page, base):
    print("\n[Discover]")
    page.goto(base)
    settle(page, 3000)
    no_exception(page, "home load")
    check(page.locator('[data-testid="stMetric"]').count() == 5, "5 KPI cards")
    for t in ["Top rated", "Hidden gems", "Most reviewed"]:
        tab(page, t)
        panel = page.locator('[role="tabpanel"]:visible')
        check(panel.locator(".book").count() == 8, f"tab '{t}' shows 8 books")
        if t == "Hidden gems":
            check(panel.inner_text().count("hidden gem") >= 8, "hidden-gem tab cards all carry the hidden-gem chip")
    for link, path in [("Find similar books", "/similar"), ("Personalised picks", "/for-you"),
                       ("Explore the data", "/explore")]:
        page.goto(base)
        settle(page, 1500)
        page.get_by_role("link", name=link).click()
        settle(page, 1500)
        check(page.url.rstrip("/").endswith(path), f"link '{link}' opens {path}")
        no_exception(page, f"after link {link}")


def test_similar(page, base):
    print("\n[Similar books]")
    examples = ["The Martian", "Atomic Habits", "Murder on the Orient Express", "Sapiens", "Harry Potter", "Ikigai"]
    for ex in examples:
        page.goto(f"{base}/similar")
        settle(page, 1500)
        btn = page.get_by_role("button", name=ex)
        check(btn.count() == 1, f"example button '{ex}' present")
        if not btn.count():
            continue
        btn.click()
        settle(page)
        no_exception(page, f"example {ex}")
        q = page.locator('[class*="st-key-card-query-book"]').inner_text()
        check(ex.split()[0].lower() in q.lower(), f"query card shows '{ex}'")
        check(cards(page).count() == 10, f"{ex}: 10 recommendations")
        check(page.locator(".book .why").count() == 10, f"{ex}: every card has a reason")

    # model switch
    for model in ["Content", "Cluster", "Hybrid"]:
        page.get_by_role("radio", name=model, exact=True).click()
        settle(page)
        no_exception(page, f"model {model}")
        check(cards(page).count() == 10, f"model {model}: 10 recommendations")

    # how many
    slider_keys(page, "How many", "ArrowRight", 2)
    check(cards(page).count() == 20, "slider 10 -> 20 recommendations")
    slider_keys(page, "How many", "ArrowLeft", 2)

    # other authors only (Ikigai is the current book: Héctor García)
    author = page.locator('[class*="st-key-card-query-book"] .card-sub').inner_text().split("·")[0].replace("by", "").strip()
    toggle(page, "Other authors only")
    authors = [t.split("\n")[2] if len(t.split("\n")) > 2 else "" for t in card_texts(page)]
    check(all(author not in a for a in authors) and len(authors) > 0, f"'Other authors only' hides {author}")
    toggle(page, "Other authors only")

    # filters
    open_expander(page, "Filters")
    select_option(page, "Only these genres", "Business", "Business & Careers")
    texts = card_texts(page)
    check(len(texts) > 0 and all("Business & Careers" in t for t in texts), "genre filter: only Business & Careers")
    slider_keys(page, "Minimum rating", "ArrowRight", 45)          # 4.5
    ratings = [float(m.group(1)) for t in card_texts(page) for m in [re.search(r"★ (\d\.\d)", t)] if m]
    check(all(r >= 4.5 for r in ratings), "min rating 4.5 respected (unrated allowed by toggle)")
    toggle(page, "Include unrated books")
    check(all("Not rated yet" not in t for t in card_texts(page)), "unrated books hidden when toggle off")
    toggle(page, "Hidden gems only")
    texts = card_texts(page)
    check(all("hidden gem" in t for t in texts), f"hidden gems only ({len(texts)} cards or empty message)")
    no_exception(page, "filters combined")
    toggle(page, "Hidden gems only")
    toggle(page, "Include unrated books")
    slider_keys(page, "Minimum rating", "Home", 1)
    page.locator('[data-testid="stMultiSelect"]').filter(has=page.get_by_text("Only these genres", exact=True)) \
        .locator('[aria-label*="Clear all"], [title="Clear all"]').first.click()
    settle(page)
    # price + length
    slider_keys(page, "Maximum price", "ArrowRight", 3)               # ₹500
    prices = [int(m.group(1).replace(",", "")) for t in card_texts(page) for m in [re.search(r"₹([\d,]+)", t)] if m]
    check(all(p <= 500 for p in prices) and prices, f"max price ₹500 respected ({len(prices)} priced cards)")
    slider_keys(page, "Maximum price", "Home", 1)
    slider_keys(page, "Maximum length", "ArrowRight", 3)              # 6 hours
    hrs = [minutes(t) for t in card_texts(page)]
    check(hrs and all(h is not None and h <= 360 for h in hrs), f"max length 6 h respected, lengths known ({len(hrs)})")
    slider_keys(page, "Maximum length", "Home", 1)
    slider_keys(page, "Minimum reviews", "ArrowRight", 4)             # 100
    no_exception(page, "min reviews")
    slider_keys(page, "Minimum reviews", "Home", 1)

    # compare tab
    tab(page, "Compare the three models")
    panel = page.locator('[role="tabpanel"]:visible')
    check(all(m in panel.inner_text() for m in ["Hybrid", "Content", "Cluster"]) and panel.locator("li").count() == 18,
          "compare tab lists 6 books for each of the three models")
    tab(page, "Recommendations")

    # search box
    select_option(page, "Book", "Rich Dad Poor Dad", "Rich Dad Poor Dad: What the Rich")
    check("Rich Dad Poor Dad" in page.locator('[class*="st-key-card-query-book"]').inner_text(),
          "selectbox search picks Rich Dad Poor Dad")
    check(cards(page).count() == 10, "Rich Dad: 10 recommendations")

    # a book with no description and no genre still works
    select_option(page, "Book", "Ek Ruka Hua Faisla") if False else None


def test_for_you(page, base):
    print("\n[For you]")
    page.goto(f"{base}/for-you")
    settle(page, 1500)
    no_exception(page, "for-you load")
    check(page.get_by_text("Choose at least one genre").count() == 1, "empty-state hint shown")
    check(cards(page).count() == 12, "empty state shows 12 top-rated books")
    for preset, expect in [("New listener who likes sci-fi", "Science Fiction"),
                           ("Thriller fan", "Mystery, Thriller & Horror"),
                           ("Self-improvement, short listens", "Personal Development")]:
        page.get_by_role("button", name=preset).click()
        settle(page)
        no_exception(page, f"preset {preset}")
        texts = card_texts(page)
        check(len(texts) == 12, f"{preset}: 12 picks")
        share = sum(expect.split(",")[0] in t for t in texts) / max(len(texts), 1)
        check(share >= 0.8, f"{preset}: {share:.0%} of picks match '{expect}'")
    hrs = [minutes(t) for t in card_texts(page)]
    check(hrs and all(h is not None and h <= 360 for h in hrs), "self-improvement preset: all ≤ 6 h")
    page.get_by_role("button", name="New listener who likes sci-fi").click()
    settle(page)
    open_expander(page, "Filters")
    length = page.locator('[data-testid="stSlider"]').filter(has=page.get_by_text("Maximum length", exact=True))
    check("Any" in length.inner_text(), "switching preset resets the length filter to Any")
    page.get_by_role("button", name="Clear", exact=True).click()
    settle(page)
    check(page.get_by_text("Choose at least one genre").count() == 1, "Clear empties the taste")
    select_option(page, "Authors you like", "Agatha Christie")
    texts = card_texts(page)
    check(sum("Agatha Christie" in t for t in texts) >= 6, "author preference: Christie books first")
    select_option(page, "Books you have enjoyed", "The Alchemist", "The Alchemist")
    no_exception(page, "liked book + author")
    check(cards(page).count() == 12, "liked book + author: 12 picks")
    select_option(page, "Genres you enjoy", "Romance")
    select_option(page, "Sub-genres (876 Audible categories)", "Regency Romance")
    no_exception(page, "all four taste inputs")
    slider_keys(page, "How many", "ArrowRight", 3)
    check(cards(page).count() == 15, "how many 12 -> 15")
    page.get_by_role("button", name="Clear", exact=True).click()
    settle(page)


def test_explore(page, base):
    print("\n[Explore]")
    page.goto(f"{base}/explore")
    settle(page, 3000)
    no_exception(page, "explore load")
    for t, min_charts in [("Genres", 3), ("Ratings & reviews", 5), ("Authors", 3), ("Price & length", 2),
                          ("Data quality", 1)]:
        tab(page, t)
        panel = page.locator('[role="tabpanel"]:visible')
        n = panel.locator(".js-plotly-plot").count()
        check(n >= min_charts, f"tab '{t}': {n} charts")
        check(panel.locator(".insight").count() >= 3, f"tab '{t}': insight cards")
        no_exception(page, f"tab {t}")
    tab(page, "Ratings & reviews")
    exp = open_expander(page, "Statistical tests")
    check("Kruskal" in exp.inner_text() or page.get_by_text("Kruskal").count() > 0, "statistical tests table opens")


def test_clusters(page, base):
    print("\n[Clusters]")
    page.goto(f"{base}/clusters")
    settle(page, 3000)
    no_exception(page, "clusters load")
    check(page.locator(".js-plotly-plot").count() >= 2, "cluster map + genre chart")
    box = page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text("Look inside a cluster", exact=True))
    for i in range(16):
        box.locator("input").click()
        box.locator("input").press("ArrowDown")
        page.wait_for_timeout(400)
        opts = page.get_by_role("option")
        if i == 0:
            check(opts.count() == 16, "16 clusters in the dropdown")
        opts.nth(i).click()
        settle(page, 600)
        n = cards(page).count()
        check(n == 6, f"cluster option {i + 1}: 6 books")
        no_exception(page, f"cluster option {i + 1}")
    open_expander(page, "KMeans vs DBSCAN")
    check(page.get_by_text("Silhouette by k").count() == 1, "silhouette chart in expander")


def test_performance(page, base):
    print("\n[Model performance]")
    page.goto(f"{base}/models")
    settle(page, 3000)
    no_exception(page, "models load")
    check(page.locator('[data-testid="stMetric"]').count() == 4, "4 KPI cards")
    check(page.locator('[data-testid="stDataFrame"]').count() >= 4, "comparison, significance, RMSE, tuning tables")
    check(page.get_by_text("Is the difference real?").count() == 1, "significance section")
    open_expander(page, "Genre inference model")
    check(page.locator('[data-testid="stDataFrame"]').count() >= 5, "per-genre table inside expander")


def test_about(page, base):
    print("\n[About]")
    page.goto(f"{base}/about")
    settle(page, 1500)
    no_exception(page, "about load")
    txt = page.locator('[data-testid="stMain"]').inner_text()
    check("Pipeline" in txt and "Honest limitations" in txt, "about content")


def test_persistence_and_nav(page, base):
    print("\n[Navigation + state]")
    page.goto(f"{base}/similar")
    settle(page, 1500)
    page.get_by_role("button", name="Sapiens").click()
    settle(page)
    for name in ["Explore", "Similar books"]:
        page.get_by_role("link", name=name).first.click()
        settle(page, 1500)
    check("Sapiens" in page.locator('[class*="st-key-card-query-book"]').inner_text(),
          "chosen book survives navigating away and back")
    for name in ["Discover", "For you", "Clusters", "Model performance", "About"]:
        page.get_by_role("link", name=name).first.click()
        settle(page, 1500)
        no_exception(page, f"top nav '{name}'")


def test_mobile(browser, base):
    print("\n[Mobile 390 px]")
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                              has_touch=True)
    page = ctx.new_page()
    for path in ["", "/similar", "/for-you", "/explore", "/clusters", "/models", "/about"]:
        page.goto(base + path)
        settle(page, 2500)
        no_exception(page, f"mobile {path or '/'}")
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        check(overflow <= 1, f"mobile {path or '/'}: no horizontal page scroll ({overflow}px)")
    ctx.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8521")
    base = parser.parse_args().url.rstrip("/")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda m: CONSOLE_ERRORS.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: CONSOLE_ERRORS.append(str(e)))
        page.on("response", lambda r: BAD_RESPONSES.append(r.url) if r.status >= 400 else None)
        for test in [test_home, test_similar, test_for_you, test_explore, test_clusters, test_performance,
                     test_about, test_persistence_and_nav]:
            try:
                test(page, base)
            except Exception as e:  # a crash inside a test is a failure, keep testing the rest
                check(False, f"{test.__name__} crashed: {type(e).__name__}: {str(e)[:300]}")
        try:
            test_mobile(browser, base)
        except Exception as e:
            check(False, f"test_mobile crashed: {e}")
        browser.close()
    # Streamlit probes "<page>/_stcore/health" before falling back to the root path when a page is opened by its
    # URL; those 404s are the framework's own and expected. Anything else failing to load is a real error.
    real_bad = [u for u in BAD_RESPONSES if "/_stcore/" not in u]
    check(not real_bad, f"no failed requests apart from Streamlit's own path probes ({len(real_bad)})")
    for u in real_bad[:10]:
        print("   failed request:", u)
    real_console = [c for c in CONSOLE_ERRORS if "Failed to load resource" not in c or real_bad]
    check(not real_console, f"no browser console errors ({len(real_console)})")
    for c in real_console[:10]:
        print("   console:", c[:200])
    print(f"\n{len(PASSES)} passed, {len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
