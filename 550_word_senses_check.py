import re

import phunspell

from epub2stardict.io_jsonl import read_jsonl, write_jsonl

INPUT_PATH = "data/500_word_senses.jsonl"
OUTPUT_PATH = "data/550_word_senses_bad.jsonl"

HU_LETTERS_RE = re.compile(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+")


def meaning_is_ok(meaning: str, speller: phunspell.Phunspell) -> bool:
    tokens = HU_LETTERS_RE.findall(meaning)
    if not tokens:
        return False
    return all(speller.lookup(tok.lower()) for tok in tokens)


def iter_bad_records(records, speller):
    for rec in records:
        meaning = rec.get("meaning_hu") or ""
        if not meaning_is_ok(meaning, speller):
            print(f"{rec.get('word', '')} -> {meaning}")
            yield rec


def main():
    speller = phunspell.Phunspell("hu_HU")
    records = list(read_jsonl(INPUT_PATH))
    print(f"Loaded {len(records)} records from {INPUT_PATH}")

    bad = write_jsonl(OUTPUT_PATH, iter_bad_records(records, speller))
    print(f"Processed {len(records)} records, wrote {bad} bad records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
