"""Raw CSVs -> data/processed/books_clean.csv (+ cleaning_report.json).

    python pipeline.py
"""
import json

import config
from src import cleaning, nlp


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    books, report = cleaning.run()
    books, report["genre_inference"] = nlp.infer_missing_genres(books)
    books.drop(columns="fine_genres").to_csv(config.BOOKS_CLEAN, index=False)
    config.CLEANING_REPORT.write_text(json.dumps(report, indent=2, default=int))
    print(json.dumps(report, indent=2, default=int))
    print(f"\n{len(books):,} books -> {config.BOOKS_CLEAN.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
