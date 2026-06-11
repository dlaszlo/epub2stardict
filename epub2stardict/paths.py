"""Per-book working directory resolution.

The numbered pipeline scripts (100–550) operate inside a single book's
directory. The directory is selected via the BOOK_DIR environment variable:

    BOOK_DIR=data/books/asimov_i_robot python 100_epub_to_text.py

The global steps (600, 650, 700) do NOT use BOOK_DIR — they work on the
union of all books and on data/650_dictionary.jsonl.
"""

import os
from pathlib import Path


def book_dir() -> Path:
    d = os.environ.get("BOOK_DIR")
    if not d:
        raise SystemExit(
            "BOOK_DIR is not set. Example:\n"
            "  BOOK_DIR=data/books/asimov_i_robot python <script>.py"
        )
    return Path(d)
