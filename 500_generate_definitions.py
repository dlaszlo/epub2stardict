import hashlib
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import APIStatusError

from epub2stardict.io_jsonl import append_jsonl, read_jsonl, write_jsonl
from epub2stardict.llm import GlossInput, LLMClient

CHUNKS_PATH = "data/200_chunks.jsonl"
WORDS_PATH = "data/400_word_pos.jsonl"
OUTPUT_PATH = "data/500_word_senses.jsonl"
PREV_OUTPUT_PATH = "data/500_word_senses_prev.jsonl"
ERROR_PATH = "data/500_word_senses.errors.jsonl"

MAX_EXAMPLES_PER_WORD = 2
BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE") or 5)
NUM_WORKERS = int(os.environ.get("LLM_NUM_WORKERS") or 8)


def format_eta(seconds: int) -> str:
    seconds = max(0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_input_key(word_rec: dict) -> str:
    """Fingerprint over (word, lemma, pos, contexts). Model is NOT included:
    to regenerate against a new model, delete the output file."""
    data = {k: word_rec[k] for k in ("word", "lemma", "pos", "contexts")}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def build_batches(word_recs: list[dict]) -> list[tuple[int, list[dict]]]:
    return [
        (i // BATCH_SIZE + 1, word_recs[i:i + BATCH_SIZE])
        for i in range(0, len(word_recs), BATCH_SIZE)
    ]


def process_batch(
    batch_no: int,
    batch_recs: list[dict],
    chunks: dict[int, str],
    total_words: int,
) -> tuple[int, int, int, int, list[dict], list[dict]]:
    """Process one batch. Returns: processed, failed, in_tok, out_tok, results, errors."""

    client = LLMClient(max_tokens=2000)
    print(f"[worker] Starting batch #{batch_no} with {len(batch_recs)} words", flush=True)

    pending_inputs: list[GlossInput] = []
    pending_meta: list[dict] = []

    for rec in batch_recs:
        valid_ids = [cid for cid in rec["contexts"] if cid in chunks]
        if not valid_ids:
            print(f"[batch {batch_no}] id={rec['id']} '{rec['word']}' SKIP: no valid contexts", flush=True)
            continue

        if len(valid_ids) > MAX_EXAMPLES_PER_WORD:
            valid_ids = random.sample(valid_ids, MAX_EXAMPLES_PER_WORD)

        example_sentences = [chunks[cid] for cid in valid_ids]
        batch_index = len(pending_inputs) + 1

        pending_inputs.append(GlossInput(
            lemma=rec.get("lemma") or rec["word"],
            word=rec["word"],
            pos=rec.get("pos"),
            example_sentences=example_sentences,
            index=batch_index,
        ))
        pending_meta.append({
            "id": rec["id"],
            "word": rec["word"],
            "lemma": rec.get("lemma") or rec["word"],
            "pos": rec.get("pos"),
            "input_key": rec["input_key"],
        })

        print(
            f"[batch {batch_no}] id={rec['id']}/{total_words}: '{rec['word']}' "
            f"(lemma='{pending_meta[-1]['lemma']}', pos='{rec.get('pos')}', "
            f"{len(valid_ids)}/{len(rec['contexts'])} examples)",
            flush=True,
        )

    if not pending_inputs:
        return 0, 0, 0, 0, [], []

    batch_error: Optional[str] = None
    batch_status_code: Optional[int] = None
    batch_request_id: Optional[str] = None
    try:
        result = client.generate(pending_inputs)
        outputs = result.outputs
        in_tok = result.in_tokens
        out_tok = result.out_tokens
        model_name = result.model
    except APIStatusError as e:
        # HTTP 4xx/5xx — log with structured fields so errors can be
        # filtered later with jq (e.g. all 429s, all 403s).
        batch_status_code = getattr(e, "status_code", None)
        batch_request_id = getattr(e, "request_id", None)
        print(
            f"[batch {batch_no}] HTTP {batch_status_code} error "
            f"(request_id={batch_request_id}): {e}",
            flush=True,
        )
        outputs = []
        in_tok = out_tok = 0
        batch_error = f"HTTP {batch_status_code}: {e}"
        model_name = client.model
    except Exception as e:
        print(f"[batch {batch_no}] ERROR during model call: {e}", flush=True)
        outputs = []
        in_tok = out_tok = 0
        batch_error = str(e)
        model_name = client.model

    results: list[dict] = []
    errors: list[dict] = []
    processed = failed = 0

    if outputs:
        for meta, out in zip(pending_meta, outputs):
            ok = out.ok and batch_error is None
            results.append({
                "id": meta["id"],
                "word": meta["word"],
                "lemma": meta["lemma"],
                "pos": meta["pos"],
                "pos_hu": out.pos_hu if ok else "",
                "meaning_hu": out.meaning_hu if ok else "",
                "example_surface_en": out.example_surface_en if ok else "",
                "example_lemma_en": out.example_lemma_en if ok else "",
                "ok": ok,
                "model": model_name,
                "input_key": meta["input_key"],
            })
            processed += 1
            if not ok:
                failed += 1
                errors.append({
                    "batch": batch_no,
                    "id": meta["id"],
                    "word": meta["word"],
                    "lemma": meta["lemma"],
                    "pos": meta["pos"],
                    "error": batch_error or out.error,
                    "status_code": batch_status_code,
                    "request_id": batch_request_id,
                    "raw_hu": out.raw_hu,
                    "raw_example_surface_en": out.raw_example_surface_en,
                    "raw_example_lemma_en": out.raw_example_lemma_en,
                    "raw_batch": out.raw_batch,
                    "input_key": meta["input_key"],
                })

            status = "OK" if ok else f"ERROR: {batch_error or out.error}"
            print(f"[batch {batch_no}] '{meta['word']}' (id={meta['id']}) -> {status}", flush=True)
    else:
        for meta in pending_meta:
            results.append({
                "id": meta["id"],
                "word": meta["word"],
                "lemma": meta["lemma"],
                "pos": meta["pos"],
                "pos_hu": "",
                "meaning_hu": "",
                "example_surface_en": "",
                "example_lemma_en": "",
                "ok": False,
                "model": model_name,
                "input_key": meta["input_key"],
            })
            errors.append({
                "batch": batch_no,
                "id": meta["id"],
                "word": meta["word"],
                "lemma": meta["lemma"],
                "pos": meta["pos"],
                "error": batch_error or "unknown batch error",
                "status_code": batch_status_code,
                "request_id": batch_request_id,
                "raw_batch": None,
                "input_key": meta["input_key"],
            })
            processed += 1
            failed += 1
            print(f"[batch {batch_no}] '{meta['word']}' (id={meta['id']}) -> ERROR: {batch_error}", flush=True)

    print(f"[worker] Finished batch #{batch_no}: processed={processed}, failed={failed}", flush=True)
    return processed, failed, in_tok, out_tok, results, errors


def load_reusable_records(word_recs_by_id: dict[int, dict]) -> dict[int, dict]:
    """Backup OUTPUT_PATH → PREV_OUTPUT_PATH and return the OK records that can be reused."""
    if not Path(OUTPUT_PATH).exists():
        print("No previous output found; full run required.")
        return {}

    print(f"Backing up {OUTPUT_PATH} → {PREV_OUTPUT_PATH}")
    shutil.copyfile(OUTPUT_PATH, PREV_OUTPUT_PATH)

    reusable: dict[int, dict] = {}
    for rec in read_jsonl(PREV_OUTPUT_PATH):
        wid = rec.get("id")
        if wid not in word_recs_by_id or not rec.get("ok"):
            continue
        if rec.get("input_key") != word_recs_by_id[wid]["input_key"]:
            continue
        reusable[wid] = rec
    print(f"Reusing {len(reusable)} previous OK records with matching input_key.")
    return reusable


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    chunks = {rec["id"]: rec["sentence"] for rec in read_jsonl(CHUNKS_PATH)}
    print(f"Loaded {len(chunks)} sentences from {CHUNKS_PATH}")

    word_recs = list(read_jsonl(WORDS_PATH))
    for rec in word_recs:
        rec["input_key"] = make_input_key(rec)
    word_recs_by_id = {rec["id"]: rec for rec in word_recs}

    print(f"Loaded {len(word_recs)} word entries from {WORDS_PATH}")
    print(f"MAX_EXAMPLES_PER_WORD={MAX_EXAMPLES_PER_WORD}, BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}")
    print(f"LLM_MODEL={os.environ.get('LLM_MODEL', '<unset>')}")

    reusable = load_reusable_records(word_recs_by_id)
    to_process = [rec for rec in word_recs if rec["id"] not in reusable]
    total_to_process = len(to_process)
    print(f"Words to process this run: {total_to_process}")

    write_jsonl(OUTPUT_PATH, reusable.values())
    Path(ERROR_PATH).write_text("", encoding="utf-8")

    if total_to_process == 0:
        print("\n========== SUMMARY ==========")
        print(f"Total words:               {len(word_recs)}")
        print(f"Reused OK from previous:   {len(reusable)}")
        print("Nothing to process. Output ready.")
        return

    batches = build_batches(to_process)
    print(f"Total batches: {len(batches)}")

    start = time.time()
    total_processed = total_failed = total_in = total_out = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, no, recs, chunks, total_to_process): (no, len(recs))
            for no, recs in batches
        }

        for future in as_completed(futures):
            batch_no, batch_size = futures[future]
            try:
                processed, failed, in_tok, out_tok, results, errors = future.result()
            except Exception as e:
                print(f"[main] Batch #{batch_no} raised exception: {e}", flush=True)
                processed = failed = batch_size
                in_tok = out_tok = 0
                results = errors = []

            total_processed += processed
            total_failed += failed
            total_in += in_tok
            total_out += out_tok

            if results:
                append_jsonl(OUTPUT_PATH, results)
            if errors:
                append_jsonl(ERROR_PATH, errors)

            elapsed = time.time() - start
            avg = elapsed / total_processed if total_processed else 0
            eta = format_eta(int((total_to_process - total_processed) * avg))
            print(
                f"[main] Batch #{batch_no} done. {total_processed}/{total_to_process} processed, "
                f"failed: {total_failed}. Elapsed: {format_eta(int(elapsed))}, ETA: {eta}",
                flush=True,
            )

    print("\n========== SUMMARY ==========")
    print(f"Total words:               {len(word_recs)}")
    print(f"Reused OK from previous:   {len(reusable)}")
    print(f"Processed this run:        {total_processed}/{total_to_process}")
    print(f"Failed this run:           {total_failed}")
    print(f"Elapsed:                   {format_eta(int(time.time() - start))}")
    print(f"Tokens in/out/total:       {total_in}/{total_out}/{total_in + total_out}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Errors: {ERROR_PATH}")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
