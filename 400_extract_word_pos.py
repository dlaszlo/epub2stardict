import os
from collections import defaultdict

import spacy

from epub2stardict.io_jsonl import read_jsonl, write_jsonl
from epub2stardict.text import BAD_POS, accept_word_form, normalize_for_match, sanitize_lemma

WORD_CONTEXTS_PATH = "data/300_word_contexts.jsonl"
CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/400_word_pos.jsonl"

# POS-noise heuristic: if a (word, POS) appears in this many contexts or fewer,
# and another POS of the same word has at least this many times more — drop the
# minority (typical tagger compound-modifier mistakes: 'robot' ADJ × 3 vs NOUN × 320).
POS_NOISE_MAX_CTX = 3
POS_NOISE_DOMINANCE = 10

# GPU needs cupy + an NVIDIA driver. Falls back to CPU on detection failure.
_GPU = spacy.prefer_gpu()
print(f"spaCy GPU: {'on' if _GPU else 'off (CPU fallback)'}", flush=True)

NLP = spacy.load("en_core_web_trf")


def load_chunks() -> dict[int, str]:
    return {rec["id"]: rec["sentence"] for rec in read_jsonl(CHUNKS_PATH)}


def collect_lemma_pos(word, contexts, chunks, nlp_cache, stats):
    """(lemma, pos) -> set of context IDs where it occurs. sanitize_lemma
    folds the dash-leakage cases ('robot-' -> 'robot') into the clean entry
    automatically via the dict-key match."""
    lemma_pos_to_contexts: dict[tuple[str, str], set[int]] = defaultdict(set)

    for cid in contexts:
        sentence = chunks.get(cid)
        if not sentence:
            continue
        doc = nlp_cache.get(cid)
        if doc is None:
            doc = NLP(sentence)
            nlp_cache[cid] = doc

        for token in doc:
            if normalize_for_match(token.text) == word:
                raw_lemma = token.lemma_.lower()
                clean_lemma = sanitize_lemma(raw_lemma)
                if clean_lemma is None:
                    stats["lemma_dropped"] += 1
                    continue
                if clean_lemma != raw_lemma:
                    stats["lemma_cleaned"] += 1
                lemma_pos_to_contexts[(clean_lemma, token.pos_)].add(cid)

    return lemma_pos_to_contexts


def filter_pos_noise(lemma_pos_map):
    """For one word, drop (lemma, POS) entries whose total contexts is
    <= POS_NOISE_MAX_CTX while another POS of the same word has at least
    POS_NOISE_DOMINANCE× more."""
    if len(lemma_pos_map) < 2:
        return lemma_pos_map
    pos_total: dict[str, int] = defaultdict(int)
    for (_lemma, pos), ctxs in lemma_pos_map.items():
        pos_total[pos] += len(ctxs)
    if len(pos_total) < 2:
        return lemma_pos_map
    max_count = max(pos_total.values())
    return {
        (lemma, pos): ctxs
        for (lemma, pos), ctxs in lemma_pos_map.items()
        if not (
            pos_total[pos] <= POS_NOISE_MAX_CTX
            and max_count >= POS_NOISE_DOMINANCE * pos_total[pos]
        )
    }


def precompute_docs(word_recs, chunks):
    """One batched nlp.pipe() pass over only the sentences we actually need.
    With LIMIT_WORDS the set shrinks automatically because we only collect
    context IDs from the (already sliced) word_recs."""
    needed_cids = set()
    for rec in word_recs:
        needed_cids.update(rec["contexts"])
    cid_list = sorted(c for c in needed_cids if c in chunks)
    if not cid_list:
        return {}
    sentences = [chunks[c] for c in cid_list]
    print(
        f"Pre-computing {len(cid_list)} spaCy docs in batches (nlp.pipe, batch_size=64)...",
        flush=True,
    )
    # parser+ner not used in step 400 — only tagger+lemmatizer are needed.
    docs = NLP.pipe(sentences, batch_size=64, disable=["parser", "ner"])
    return dict(zip(cid_list, docs))


def iter_word_pos_records(word_recs, chunks):
    nlp_cache = precompute_docs(word_recs, chunks)
    print(f"Cache ready ({len(nlp_cache)} docs). Aggregating per-word POS records...", flush=True)
    next_id = 1
    stats = {
        "skipped_form": 0,
        "skipped_empty": 0,
        "lemma_cleaned": 0,
        "lemma_dropped": 0,
        "pos_noise_dropped": 0,
    }

    for rec in word_recs:
        word = rec["word"]
        if not accept_word_form(word):
            stats["skipped_form"] += 1
            continue

        lemma_pos_map = collect_lemma_pos(word, rec["contexts"], chunks, nlp_cache, stats)
        # BAD_POS filter
        lemma_pos_map = {k: v for k, v in lemma_pos_map.items() if k[1] not in BAD_POS}
        # POS-noise filter (minority POS dominated by another POS of the same word)
        before = len(lemma_pos_map)
        lemma_pos_map = filter_pos_noise(lemma_pos_map)
        stats["pos_noise_dropped"] += before - len(lemma_pos_map)

        if not lemma_pos_map:
            stats["skipped_empty"] += 1
            continue

        for (lemma, pos) in sorted(lemma_pos_map.keys()):
            yield {
                "id": next_id,
                "word": word,
                "lemma": lemma,
                "pos": pos,
                "contexts": sorted(lemma_pos_map[(lemma, pos)]),
            }
            next_id += 1

    print(
        f"Skipped (form): {stats['skipped_form']}, "
        f"skipped (no usable POS): {stats['skipped_empty']}"
    )
    print(
        f"Lemma cleaned (dash-strip): {stats['lemma_cleaned']}, "
        f"lemma records dropped (mid-word junk): {stats['lemma_dropped']}"
    )
    print(f"POS-noise records dropped: {stats['pos_noise_dropped']}")


def main():
    chunks = load_chunks()
    word_recs = list(read_jsonl(WORD_CONTEXTS_PATH))
    total_in = len(word_recs)

    limit = int(os.environ.get("LIMIT_WORDS") or 0)
    if limit > 0:
        word_recs = word_recs[:limit]
        print(
            f"LIMIT_WORDS={limit} — only the first {limit} of {total_in} unique words "
            f"will be considered. All downstream scripts will inherit this limit via "
            f"{OUTPUT_PATH}."
        )

    n = write_jsonl(OUTPUT_PATH, iter_word_pos_records(word_recs, chunks))
    print(f"Total words in: {total_in}, used: {len(word_recs)}, records written: {n} → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
