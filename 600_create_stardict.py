"""Write the per-book StarDict dictionary and PocketBook .dic with PyGlossary.

Input: $BOOK_DIR/500_book_senses.jsonl. Output: $BOOK_DIR/dict/<slug>-eng-hun.*
— one dictionary per book, named after the book in the filename and in the
glossary metadata. The dictionary title comes from $BOOK_DIR/book_title.txt
if present, otherwise from the prettified slug; the description comes from
$BOOK_DIR/book_info.txt.

The per-row `model` field is provenance only and is not displayed.
"""

from collections import defaultdict

from pyglossary.glossary_v2 import Glossary

from epub2stardict.io_jsonl import read_jsonl
from epub2stardict.paths import book_dir

BOOK_DIR = book_dir()
INPUT_PATH = BOOK_DIR / "500_book_senses.jsonl"
OUTPUT_DIR = BOOK_DIR / "dict"
SLUG = BOOK_DIR.name
DICT_BASENAME = f"{SLUG}-eng-hun"


def load_title() -> str:
    title_path = BOOK_DIR / "book_title.txt"
    if title_path.is_file():
        return title_path.read_text(encoding="utf-8").strip()
    return SLUG.replace("_", " ").title()


def load_description(title: str) -> str:
    info_path = BOOK_DIR / "book_info.txt"
    if info_path.is_file():
        return info_path.read_text(encoding="utf-8").strip()
    return f"English-Hungarian dictionary built from the {title} word list."


def build_definition_block(rec: dict, seen_examples: set[str]) -> str:
    meaning = (rec.get("meaning_hu") or "").strip()
    pos_hu = (rec.get("pos_hu") or "").strip()
    ex_surface = (rec.get("example_surface_en") or "").strip()
    ex_lemma = (rec.get("example_lemma_en") or "").strip()

    headline_parts: list[str] = []
    if meaning:
        headline_parts.append(meaning)
    else:
        headline_parts.append((rec.get("word") or rec.get("lemma") or "<?>").strip())
    if pos_hu:
        headline_parts.append(f"({pos_hu})")

    lines = [" ".join(headline_parts)]
    for example in (ex_surface, ex_lemma):
        if example and example not in seen_examples:
            lines.append(example)
            seen_examples.add(example)
    return "\n".join(lines)


def collect_entries() -> list[tuple[str, str]]:
    word_to_blocks: dict[str, list[str]] = defaultdict(list)
    word_to_seen_examples: dict[str, set[str]] = defaultdict(set)

    for rec in read_jsonl(INPUT_PATH):
        if not rec.get("ok"):
            continue
        word = (rec.get("word") or rec.get("lemma") or "").strip()
        if not word:
            continue
        block = build_definition_block(rec, word_to_seen_examples[word])
        if block:
            word_to_blocks[word].append(block)

    return [
        (word, "\n\n".join(blocks))
        for word, blocks in word_to_blocks.items()
    ]


def new_glossary(title: str, description: str) -> Glossary:
    glos = Glossary()
    glos.setInfo("title", title)
    glos.setInfo("name", title)
    glos.setInfo("description", description)
    glos.sourceLangName = "English"
    glos.targetLangName = "Hungarian"
    return glos


def main():
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    title = f"Eng-Hun: {load_title()}"
    description = load_description(load_title())
    print(f"Dictionary title: {title}")

    entries = collect_entries()
    print(f"Entries: {len(entries)}")

    Glossary.init()

    glos = new_glossary(title, description)
    for word, definition in entries:
        glos.addEntry(glos.newEntry(word, definition, defiFormat="m"))

    out_basename = str(OUTPUT_DIR / DICT_BASENAME)
    print(f"PyGlossary write: {out_basename}.* (Stardict, dictzip=True)")
    glos.write(out_basename, formatName="Stardict", dictzip=True)

    # PocketBook's built-in reader (SDIC .dic) drops literal newlines but
    # renders HTML, so the same entries go in with <br/> line breaks.
    pb_glos = new_glossary(title, description)
    for word, definition in entries:
        html = definition.replace("\n", "<br/>")
        pb_glos.addEntry(pb_glos.newEntry(word, html, defiFormat="h"))

    pb_path = str(OUTPUT_DIR / f"{DICT_BASENAME}.dic")
    print(f"PyGlossary write: {pb_path} (PocketBookSdic)")
    pb_glos.write(pb_path, formatName="PocketBookSdic")

    print("Done:")
    for ext in (".ifo", ".idx", ".dict", ".dict.dz", ".syn", ".dic"):
        p = OUTPUT_DIR / f"{DICT_BASENAME}{ext}"
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
