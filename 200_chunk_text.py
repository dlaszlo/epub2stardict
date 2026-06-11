import re

import spacy

from epub2stardict.io_jsonl import write_jsonl
from epub2stardict.paths import book_dir

BOOK_DIR = book_dir()
INPUT_PATH = BOOK_DIR / "100_book.txt"
OUTPUT_PATH = BOOK_DIR / "200_chunks.jsonl"

# Note: step 200 processes the whole book in a single NLP(text) call,
# peak ~4-5 GiB VRAM. Fits an 8GB GPU, but only if no other process
# (browser, video, Slack) holds 2-3 GiB. On OOM, swap `spacy.prefer_gpu()`
# for `spacy.require_cpu()`.
_GPU = spacy.prefer_gpu()
print(f"spaCy GPU: {'on' if _GPU else 'off (CPU fallback)'}", flush=True)

NLP = spacy.load("en_core_web_trf")
NLP.add_pipe("sentencizer")


def split_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        text = f.read()

    sentences = split_to_sentences(text)
    records = ({"id": i, "sentence": s} for i, s in enumerate(sentences))
    n = write_jsonl(OUTPUT_PATH, records)
    print(f"Written {n} sentences to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
