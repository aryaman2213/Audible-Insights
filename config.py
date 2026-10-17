"""Every path, cleaning rule and model setting for Audible Insights, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

CATALOG_CSV = RAW_DIR / "Audible_Catlog.csv"
ADVANCED_CSV = RAW_DIR / "Audible_Catlog_Advanced_Features.csv"

BOOKS_CLEAN = PROCESSED_DIR / "books_clean.csv"          # one row per book, used by EDA, models and the app
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.json"  # every count the cleaning step produced

RECOMMENDER_ARTIFACTS = MODELS_DIR / "recommender.joblib"  # vectoriser, matrices, clusters
GENRE_CLASSIFIER = MODELS_DIR / "genre_classifier.joblib"
MODEL_METRICS = MODELS_DIR / "model_metrics.json"
CLUSTER_SUMMARY = MODELS_DIR / "cluster_summary.json"

SEED = 42

# ---------------------------------------------------------------- cleaning
# The raw files mark "no value" with these sentinels.
MISSING_SENTINELS = ["-1", "Not yet known", ""]

# Descriptions that are scraping failures, not book text (CAPTCHA pages, error pages, checkout boilerplate,
# publisher stubs). Matched case-insensitively; a match means the description is treated as missing.
JUNK_DESCRIPTION_PATTERNS = [
    r"make sure you're not a robot",
    r"rush hour and traffic is piling up",
    r"by completing your purchase you agree",
    r"^brought to you by \w+\.?$",
    r"^exclusively from audible$",
    r"^this audiobook contains the following works:?$",
]
MIN_DESCRIPTION_CHARS = 20          # shorter text carries no usable signal

# Short descriptions that only announce an award or bestseller status ("Pulitzer Prize, Fiction, 2000",
# "A New York Times Best Seller") describe nothing about the book. One is dropped when, after removing these words and
# English stop words, fewer than BLURB_MIN_CONTENT_WORDS content words remain.
BLURB_TRIGGER = r"prize|award|best ?seller|nominee|finalist|short-?listed|longlisted|\bwinner\b"
BLURB_MAX_CHARS = 200
BLURB_MIN_CONTENT_WORDS = 4
BLURB_WORDS = set("""
    prize prizes award awards awarded winner winners won win nominee nominated nomination finalist shortlisted
    longlisted short listed long best seller sellers bestseller bestselling selling new york times sunday usa today
    wall street journal publishers weekly international national number one instant top audie audies booker man
    pulitzer fiction nonfiction non general book books year author authors audiobook audiobooks critics circle
    women's women baileys costa nobel literature british specsavers category categories classic solo narration male
    female judges grammy spoken word album audiofile earphones listen audible's audible amazon charts phenomenal
    indian updated expanded edition original work first novel debut business educational teens fantasy romance
    personal development science technology goodreads five star ratings copies sold over several changed millions
    lives brought penguin""".split())

# Two titles are the same book when their normalised titles match and the authors are this similar
# (catches "Herman Hesse" / "Hermann Hesse", "The Arbinger Institute" / "Arbinger Institute").
AUTHOR_FUZZY_MATCH = 0.85

OVERALL_RANK_CATEGORY = "Audible Audiobooks & Originals"

# ~130 prices are 10x too high (a dropped decimal point: "A Game of Thrones" at ₹18,149). Evidence: these books are
# as long as the rest (median 8.4 h vs 8.2 h) but cost 14x more per hour; divided by 10 they cost ₹128/h, in line
# with the catalogue (IQR ₹66-124/h). Nothing is priced between ₹3,000 and ₹5,000 apart from a handful.
PRICE_DECIMAL_SHIFT_ABOVE = 5000

# ---------------------------------------------------------------- genre taxonomy
# The raw data has 876 fine genres. Each is mapped to one broad genre by the first rule with a keyword that starts a
# word in it. Order matters: "Children's Science Fiction" is a children's title, "Romantic Comedy" is romance,
# "Stress Management" is personal development rather than business. The first four rules are exceptions that must
# win before the broad keywords ("history", "science", "children", "romance") claim them.
GENRE_RULES = [
    ("Literature & Fiction", ["literary history", "literary theory", "literary criticism", "military fiction",
                              "military action fiction", "theory & criticism"]),
    ("Arts, Sports & Leisure", ["film history", "art history"]),
    ("Psychology, Philosophy & Society", ["social science", "political science"]),
    # Audible's "Love & Romance" sits under Relationships (non-fiction), not romance novels.
    ("Personal Development", ["love & romance", "parent & adult children", "parenting"]),
    ("Children & Teens", ["children", "for teens", "kids", "early learning", "fairy tale", "nursery",
                          "teen & young adult"]),
    ("Science Fiction & Fantasy", ["science fiction", "sci-fi", "fantasy", "dystopian", "space opera",
                                   "post-apocalyptic", "cyberpunk", "steampunk", "paranormal", "supernatural",
                                   "dragon", "time travel"]),
    ("Mystery, Thriller & Horror", ["thriller", "technothriller", "mystery", "mysteries", "crime", "suspense",
                                    "detective", "espionage", "horror", "noir", "heist", "ghost"]),
    ("Romance", ["romance", "romantic", "erotica", "erotic"]),
    ("Biographies & Memoirs", ["biograph", "memoir", "autobiograph", "diaries", "letters"]),
    ("Health & Wellness", ["health", "diet", "nutrition", "fitness", "medicine", "medical", "wellness",
                           "weight loss", "exercise", "sleep", "disease", "addiction", "aging", "anatomy",
                           "sexuality", "beauty", "anxiety", "depression", "grief", "naturopathy", "diabetes",
                           "smoking", "compulsive", "counselling", "psychotherapy", "pregnancy", "bodybuilding",
                           "pain management", "cancer", "ptsd", "eating disorders", "physician", "sexual abuse",
                           "codependency", "attention deficit", "personality disorders", "infertility", "injuries",
                           "nursing", "pharmacology", "vitamins", "vision & eye", "elder care", "twelve-step"]),
    ("Personal Development", ["personal success", "self-help", "personal development", "self-esteem",
                              "motivation", "productivity", "time management", "success", "happiness",
                              "communication & social", "social skills", "stress management", "anger management",
                              "creativity", "habit", "relationships", "parenting", "self-improvement",
                              "personal transformation", "emotions", "public speaking", "memory improvement",
                              "marriage", "motherhood", "fatherhood", "self-hypnosis", "cleaning & organising",
                              "lifestyle", "mentoring", "coaching", "conflict resolution", "sex instruction", "neuro-linguistic", "divorce",
                              "etiquette"]),
    ("Money & Finance", ["finance", "investing", "money", "stocks", "bank", "economic", "microeconomics",
                         "macroeconomics", "trading", "wealth", "real estate", "accounting", "taxation", "budget",
                         "retirement", "econometrics", "mortgage", "capitalism", "inflation", "interest", "insurance",
                         "mutual funds", "international trades"]),
    ("Business & Careers", ["business", "leadership", "management", "marketing", "advertising", "sales",
                            "entrepreneur", "career", "strategy", "strategic", "organisational", "organizational",
                            "workplace", "e-commerce", "industries", "negotiat", "startup", "corporate",
                            "job hunting", "customer service", "commerce", "consumer", "meetings", "presentations",
                            "non-profits", "free enterprise", "labor", "labour", "public relations"]),
    ("Religion & Spirituality", ["spirit", "religio", "hindu", "buddhis", "christian", "christology", "meditation",
                                 "mystic", "new thought", "bible", "islam", "judaism", "occult", "astrology", "yoga",
                                 "mind,", "theology", "sikh", "tantra", "gods", "worship", "prayer", "zen",
                                 "new age", "esoteric", "divination", "mythology", "atheism", "agnosticism",
                                 "church", "quran", "consciousness", "metaphysics", "taoism", "reincarnation",
                                 "parapsychology", "witchcraft", "wicca", "catholicism", "chakras", "esp",
                                 "out-of-body"]),
    ("History", ["history", "histories", "world war", "ancient", "medieval", "civilization", "military",
                 "archaeology", "rome", "holocaust"]),
    ("Science & Technology", ["science", "physics", "astronomy", "cosmology", "biology", "chemistry",
                              "technology", "biotechnology", "computer", "programming", "mathematic", "engineering",
                              "neuroscience", "nature", "ecology", "environment", "evolution", "genetics", "data",
                              "software", "internet", "artificial intelligence", "astrophysics", "zoology",
                              "animals", "machine theory", "robotics", "network", "botany", "climate", "geology",
                              "future studies", "pets", "dogs", "cryptography", "cybernetics", "palaeontology"]),
    ("Psychology, Philosophy & Society", ["psycholog", "freudian", "philosoph", "sociolog", "politic", "geopolitics",
                                          "social", "anthropolog", "women's studies", "lgbt studies", "culture",
                                          "law", "judicial", "current affairs", "government", "ethics", "essays",
                                          "journalism", "media studies", "gender", "public policy", "society",
                                          "mind", "logic", "true crime", "war", "terrorism", "activism",
                                          "democracy", "fascism", "racism", "colonialism", "imperialism",
                                          "diplomacy", "international relations", "globalisation", "civil rights",
                                          "human rights", "criminology", "demographic", "geography",
                                          "true accounts", "customs & traditions", "privacy", "immigration",
                                          "conservatism", "philanthropy", "sustainable", "epistemology", "constitutions",
                                          "pornography studies"]),
    ("Language & Education", ["language", "grammar", "education", "textbook", "study guide", "words", "teaching",
                              "test prep", "reference", "dictionar", "writing", "vocabulary", "school",
                              "linguistics", "speech", "study skills", "authorship"]),
    ("Comedy & Humour", ["humour", "humor", "comedy", "satire", "funny", "jokes", "parody", "comic"]),
    ("Arts, Sports & Leisure", ["music", "art", "film", "sport", "travel", "cook", "hobb", "games", "cricket",
                                "football", "photography", "theatre", "theater", "dance", "design", "craft",
                                "outdoor", "fashion", "television", "entertainment", "chess", "golf", "food",
                                "cinema", "painting", "tennis", "baseball", "marathon", "running", "home",
                                "architecture", "gastronomy", "acting", "playwriting", "triathlon", "expeditions",
                                "soccer", "cycling", "climbing", "gardening", "celebrity chefs", "screenwriting",
                                "transportation"]),
    ("Literature & Fiction", ["fiction", "literature", "classic", "short stor", "poetry", "drama", "antholog",
                              "saga", "novel", "plays", "literary", "fables", "myths", "folk", "storytelling",
                              "action & adventure"]),
]
OTHER_GENRE = "Other"
UNKNOWN_GENRE = "Unknown"

# A genre inferred from the description is only kept when the classifier is confident enough. The probability cut-off
# is chosen on cross-validation as the lowest one at which inferred genres are still right this often.
GENRE_INFERENCE_TARGET_ACCURACY = 0.70

# ---------------------------------------------------------------- quality score
# Bayesian (IMDb-style) weighted rating: a 5.0 from 2 reviews should not outrank a 4.7 from 20,000.
BAYES_MIN_REVIEWS = 50              # m: the review count at which a book's own rating gets half the weight

# "Hidden gem": rated highly by the people who did review it, but few have found it yet.
HIDDEN_GEM_MIN_RATING = 4.5
HIDDEN_GEM_MIN_REVIEWS = 5          # enough reviews that the rating is not a single fan
HIDDEN_GEM_MAX_REVIEW_QUANTILE = 0.25

# ---------------------------------------------------------------- NLP + clustering
TFIDF_PARAMS = dict(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True,
                    max_features=30_000)
SVD_COMPONENTS = 150                # dense "topic" space used by clustering and the hybrid model
KMEANS_K_RANGE = range(8, 31, 2)    # k is picked by silhouette score inside this range
DBSCAN_EPS_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
DBSCAN_MIN_SAMPLES = 5

# Clusters are learned on books with a listed genre from text + genres (text alone clusters poorly: silhouette 0.05),
# then the other books join the nearest cluster by text (see recommenders.fit_clusters).
CLUSTER_WEIGHTS = {"text": 1.0, "broad": 0.3, "fine": 0.3}

# Feature-block weights for the content vector (title/description text, genres, author).
TEXT_WEIGHT = 1.0
GENRE_WEIGHT = 0.6
AUTHOR_WEIGHT = 0.3

# ---------------------------------------------------------------- evaluation
EVAL_K = 10                          # precision@k / recall@k
EVAL_QUERIES = 600                   # query books sampled for the offline evaluation
RATING_KNN_NEIGHBOURS = 20           # neighbours used to predict a book's rating (RMSE task)
TEST_SIZE = 0.2
