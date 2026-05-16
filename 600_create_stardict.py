"""Write a StarDict dictionary with PyGlossary.

Input: data/500_word_senses.jsonl (single source, the output of step 500).
Every record has a `model` field — it goes into the definition headline.
"""

from collections import defaultdict
from pathlib import Path

from pyglossary.glossary_v2 import Glossary

from epub2stardict.io_jsonl import read_jsonl

INPUT_PATH = Path("data/500_word_senses.jsonl")
OUTPUT_DIR = Path("data/eng-hun-dict")
DICT_BASENAME = "eng-hun"

BOOKNAME = "English-Hungarian dictionary"
DESCRIPTION = "English-Hungarian dictionary built from Isaac Asimov - I, Robot word list."


def short_model_name(model: str) -> str:
    """Display-friendly model name. Strips the provider prefix and a
    'claude-' family prefix so the headline stays compact:
      'anthropic/claude-sonnet-4.6' -> 'sonnet-4.6'
      'openai/gpt-4o' -> 'gpt-4o'
      'google/gemma-3-27b-it' -> 'gemma-3-27b-it'
    """
    name = model.rsplit("/", 1)[-1]
    if name.startswith("claude-"):
        name = name[len("claude-"):]
    return name


def build_definition_block(rec: dict, seen_examples: set[str]) -> str:
    meaning = (rec.get("meaning_hu") or "").strip()
    pos_hu = (rec.get("pos_hu") or "").strip()
    model = short_model_name((rec.get("model") or "").strip())
    ex_surface = (rec.get("example_surface_en") or "").strip()
    ex_lemma = (rec.get("example_lemma_en") or "").strip()

    headline_parts: list[str] = []
    if meaning:
        headline_parts.append(meaning)
    else:
        headline_parts.append((rec.get("word") or rec.get("lemma") or "<?>").strip())
    if pos_hu:
        headline_parts.append(f"({pos_hu})")
    if model:
        headline_parts.append(f"({model})")

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


def main():
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = collect_entries()
    print(f"Entries: {len(entries)}")

    Glossary.init()
    glos = Glossary()
    glos.setInfo("title", BOOKNAME)
    glos.setInfo("name", BOOKNAME)
    glos.setInfo("description", DESCRIPTION)
    glos.sourceLangName = "English"
    glos.targetLangName = "Hungarian"

    for word, definition in entries:
        glos.addEntry(glos.newEntry(word, definition, defiFormat="m"))

    out_basename = str(OUTPUT_DIR / DICT_BASENAME)
    print(f"PyGlossary write: {out_basename}.* (Stardict, dictzip=True)")
    glos.write(out_basename, format="Stardict", dictzip=True)

    print("Done:")
    for ext in (".ifo", ".idx", ".dict", ".dict.dz", ".syn"):
        p = OUTPUT_DIR / f"{DICT_BASENAME}{ext}"
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
