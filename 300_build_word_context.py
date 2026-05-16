from collections import defaultdict

import spacy

from epub2stardict.io_jsonl import read_jsonl, write_jsonl
from epub2stardict.text import accept_word_form, normalize_for_match

CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/300_word_contexts.jsonl"

# GPU needs cupy + an NVIDIA driver. Falls back to CPU on detection failure.
_GPU = spacy.prefer_gpu()
print(f"spaCy GPU: {'on' if _GPU else 'off (CPU fallback)'}", flush=True)

NLP = spacy.load("en_core_web_trf")


def main():
    word_contexts: dict[str, set[int]] = defaultdict(set)

    records = list(read_jsonl(CHUNKS_PATH))
    sentences = [r["sentence"] for r in records]
    print(f"Processing {len(sentences)} sentences with nlp.pipe(batch_size=64)...", flush=True)

    # parser+ner not used here — disabled for faster token-level processing.
    for rec, doc in zip(records, NLP.pipe(sentences, batch_size=64, disable=["parser", "ner"])):
        cid = rec["id"]
        for token in doc:
            word = normalize_for_match(token.text)
            if accept_word_form(word):
                word_contexts[word].add(cid)

    out_records = (
        {"word": w, "contexts": sorted(ids)}
        for w, ids in word_contexts.items()
    )
    n = write_jsonl(OUTPUT_PATH, out_records)
    print(f"Written {n} word records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
