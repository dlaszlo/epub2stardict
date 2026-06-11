import os
import re

import phunspell

from epub2stardict.io_jsonl import read_jsonl, write_jsonl, write_jsonl_atomic
from epub2stardict.paths import book_dir

BOOK_DIR = book_dir()
INPUT_PATH = BOOK_DIR / "500_book_senses.jsonl"
OUTPUT_PATH = BOOK_DIR / "550_word_senses_bad.jsonl"

# FIX=1: remove the failing rows from 500_book_senses.jsonl. The 500 reuse
# only keeps rows present in its output, so the next 500 run regenerates them.
FIX = os.environ.get("FIX") == "1"

HU_LETTERS_RE = re.compile(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+")


def meaning_is_ok(meaning: str, speller: phunspell.Phunspell) -> bool:
    tokens = HU_LETTERS_RE.findall(meaning)
    if not tokens:
        return False
    return all(speller.lookup(tok.lower()) for tok in tokens)


def main():
    speller = phunspell.Phunspell("hu_HU")
    records = list(read_jsonl(INPUT_PATH))
    print(f"Loaded {len(records)} records from {INPUT_PATH}")

    good, bad = [], []
    for rec in records:
        # ok=false rows are known failures; 500 regenerates them on its own.
        if rec.get("ok") and not meaning_is_ok(rec.get("meaning_hu") or "", speller):
            print(f"{rec.get('word', '')} -> {rec.get('meaning_hu', '')}")
            bad.append(rec)
        else:
            good.append(rec)

    write_jsonl(OUTPUT_PATH, bad)
    print(f"Processed {len(records)} records, wrote {len(bad)} bad records to {OUTPUT_PATH}")

    if FIX and bad:
        write_jsonl_atomic(INPUT_PATH, good)
        print(f"FIX=1: removed {len(bad)} rows from {INPUT_PATH}; re-run 500 to regenerate them.")
    elif FIX:
        print("FIX=1: nothing to remove.")


if __name__ == "__main__":
    main()
