# epub2stardict

Builds one English-to-Hungarian StarDict dictionary **per EPUB book**.
Collects the book's headwords, generates Hungarian glosses and example
sentences with an LLM (using the book as context), writes the StarDict
files with PyGlossary. The glosses are book-specific on purpose — that is
what makes these dictionaries better than a generic one while reading that
book.

## What this is

- **Input:** one or more book directories under `data/books/<slug>/`, each
  containing a `book.epub` and a 1-2 sentence `book_info.txt` used as prompt
  context. An optional `book_title.txt` (one line) gives the display title;
  without it the slug is prettified (`asimov_i_robot` → `Asimov I Robot`).
- **Output:** per book, a StarDict dictionary at
  `$BOOK_DIR/dict/<slug>-eng-hun.{ifo,idx,dict.dz}` plus a PocketBook SDIC
  dictionary at `$BOOK_DIR/dict/<slug>-eng-hun.dic` (for the built-in
  reader; copy it to `system/dictionaries` on the device). The book title
  goes into the filename and the glossary metadata, so the dictionaries are
  easy to tell apart on the device.
- **Intermediate data:** per-book JSONL files under `data/books/<slug>/`.
  Books never share state; each book's pipeline is fully independent.

## Pipeline (script numbers = run order)

Every step is per-book and `BOOK_DIR`-driven. Set `BOOK_DIR` first:

```
export BOOK_DIR=data/books/asimov_i_robot
```

1. `100_epub_to_text.py` — `$BOOK_DIR/book.epub` → `$BOOK_DIR/100_book.txt`
   (ASCII-normalized).
2. `200_chunk_text.py` — text → `$BOOK_DIR/200_chunks.jsonl` (one sentence
   per line, segmented with spaCy).
3. `300_build_word_context.py` — sentences → `$BOOK_DIR/300_word_contexts.jsonl`
   (one row per unique surface word + the sentence IDs it appears in).
   Only `[a-z]{3,}` forms are kept.
4. `400_extract_word_pos.py` — `$BOOK_DIR/400_word_pos.jsonl`: one row per
   `(word, lemma, POS)` combination with its context sentence IDs.
   `BAD_POS` tags (`PROPN`, `SYM`, `PUNCT`, `X`, `SPACE`) are dropped.
   Lemmas are sanitized (`robot-` → `robot`, with mid-word junk like
   `phd → ph.d.` discarded); minority POS readings that look like tagger
   noise (e.g. `robot` ADJ × 3 vs NOUN × 320) are filtered.
5. `500_generate_definitions.py` — calls the LLM. Writes
   `$BOOK_DIR/500_book_senses.jsonl` with the Hungarian gloss, Hungarian POS
   label, two example sentences, and the active `model` name. Incremental:
   backs up the previous output to `500_book_senses_prev.jsonl` and reuses
   any row whose `input_key` (hash of `word|lemma|pos|contexts`) still matches.
6. `550_word_senses_check.py` — runs phunspell over the Hungarian glosses
   of the `ok=true` rows, writes failing rows to
   `$BOOK_DIR/550_word_senses_bad.jsonl`. By default diagnostic only; with
   `FIX=1` it also removes the failing rows from `500_book_senses.jsonl`,
   so the next 500 run regenerates exactly those words (the 500 reuse only
   keeps rows present in its output). Typical QA loop:
   `550 → FIX=1 550 → 500 → 550` until clean (or the rest is genuinely
   un-spellcheckable).

7. `600_create_stardict.py` — turns the `ok=true` rows of
   `$BOOK_DIR/500_book_senses.jsonl` into the book's StarDict dictionary
   and PocketBook SDIC `.dic` under `$BOOK_DIR/dict/` via PyGlossary. The
   SDIC format strips literal newlines but renders HTML, so the PocketBook
   entries use `<br/>` line breaks.

### Adding a new book

```
mkdir -p data/books/<slug>
cp newbook.epub data/books/<slug>/book.epub
echo "1-2 sentence description" > data/books/<slug>/book_info.txt
echo "Author – Title" > data/books/<slug>/book_title.txt  # optional display title
BOOK_DIR=data/books/<slug> python 100_epub_to_text.py
BOOK_DIR=data/books/<slug> python 200_chunk_text.py
BOOK_DIR=data/books/<slug> python 300_build_word_context.py
BOOK_DIR=data/books/<slug> python 400_extract_word_pos.py
BOOK_DIR=data/books/<slug> python 500_generate_definitions.py
BOOK_DIR=data/books/<slug> FIX=1 python 550_word_senses_check.py  # optional QA: drop bad glosses, then re-run 500
BOOK_DIR=data/books/<slug> python 600_create_stardict.py
```

Steps 200/300/400 are deterministic and cache their output on disk per
book. Step 500 reuses good rows within the same book by `input_key`, so a
re-run only hits the API for new or previously failed words in that book.
Step 600 is cheap to re-run; it reads the 500 output as its source of
truth. Existing books are untouched by all of this — a new book never
forces work on an old one.

## Architecture

```
epub2stardict/                  # shared library
  io_jsonl.py                   # JSONL read/write/append + atomic write
  text.py                       # word filtering, lemma sanitization, BAD_POS
  llm.py                        # LLMClient + prompt + JSON parser
  paths.py                      # BOOK_DIR resolution
100_…py … 600_…py               # pipeline scripts, all per-book (BOOK_DIR-driven)
data/
  books/<slug>/                 # one directory per book
    book.epub, book_info.txt    # inputs (+ optional book_title.txt)
    100_… 200_… … 550_…         # intermediates
    dict/<slug>-eng-hun.*       # the book's StarDict + PocketBook output
```

The numbered scripts are deliberately **utility-style**: each has its own
`main()`, runs standalone, and reads its config from environment variables
(`BOOK_DIR` for per-book scripts, `.env` for LLM settings). Anything shared
lives under `epub2stardict/`.

## LLM provider

A single OpenAI-compatible Chat Completions endpoint via the official
`openai` Python SDK. Switching provider is purely an env change:

```
LLM_API_KEY=...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-3-flash-preview
```

OpenAI, OpenRouter, and Ollama's OpenAI-compatible endpoint all work with
the same three variables — see `.env.example` and `README.md` for examples.

The request always sends `tool_choice="none"` so Llama-style deployments
don't slip into tool-calling mode. The strict-JSON response format is
toggled via `LLM_JSON_RESPONSE_FORMAT` (default on; turn off for weak open
models that return empty output otherwise). Transient errors are handled
by the SDK's built-in retry (`max_retries=5`).

The active model name is written into every record in
`$BOOK_DIR/500_book_senses.jsonl` — provenance only, step 600 does not
display it. The headline shows the gloss and the POS label; the book is
identified by the dictionary itself, not by the entries:

```
süt (ige)
Mom baked bread this morning.
I want to bake a pie.
```

The 500-step `input_key` deliberately does **not** include the model name.
After a model change, already-good rows in the same book are still reused
(and keep their original model label). To regenerate one book against a new
model, delete its `500_book_senses.jsonl` before re-running step 500.

## GPU

`spacy.prefer_gpu()` is called in steps 200/300/400. With `cupy-cuda13x`
(in `requirements.txt`) and an NVIDIA driver present, the spaCy transformer
runs on GPU; otherwise it falls back to CPU automatically. The 200 step
processes the whole book in a single call — peak VRAM ~4-5 GiB. If
something else is holding GPU memory, replace `spacy.prefer_gpu()` with
`spacy.require_cpu()` in 200.

## Test runs

`LIMIT_WORDS=50` (env) makes the 400 step take only the first 50 unique
words from the 300 output. Every downstream per-book script inherits the
narrowed input via the 400 file. Steps 100/200/300 still run over the
whole book. The limit is per-book — set it independently for each book
you want to truncate.

## Concurrency

Step 500 batches the words (`LLM_BATCH_SIZE` per request, default 5) and
runs the requests in a thread pool (`LLM_NUM_WORKERS` parallel calls,
default 8). For local Ollama set `LLM_NUM_WORKERS=1`.

## Design principles

- **KISS first, DRY second.** Extract a helper only when there is real
  duplication (3+ call sites). One LLM provider, one client class, no
  abstract `Service` base.
- **YAGNI.** Two providers, a real DB, parallel multi-book runs — we'll
  add them when needed. Not now.
- **Env-driven config.** No CLI argparse on the numbered scripts, no
  `config.py`. `.env` + `BOOK_DIR` and that's it.
- **JSONL between every step.** One record per line, easy to `jq`. Atomic
  write (`os.replace` over a same-directory tmp) where a crash mid-write
  would lose expensive data (e.g. the 550 FIX rewrite of the 500 output).
- **Numbered scripts.** Run order is in the filename; no separate runner.
- **Books are fully independent.** Everything lives under
  `data/books/<slug>/` and never touches other books. One book = one
  pipeline run = one dictionary; reproducible and cheap to redo per book.

## What NOT to add

- Another provider abstraction next to `LLMClient`. If a real second
  provider needs different code, fold it into `LLMClient` as a branch,
  not a new class hierarchy.
- A homegrown HTTP client or retry loop. The `openai` SDK handles both.
- A custom StarDict writer. PyGlossary handles it.
- argparse on the numbered scripts. Env variables only.
- A logging framework. `print(..., flush=True)` is enough.
- Postgres / SQLite. Per-book JSONL files cover single-user, single-machine,
  sequential use comfortably.
- A cumulative merged dictionary across books. We built one (600/650/700
  with an LLM merge step) and removed it: merging to the "most general
  sense" erases exactly the book-specific glosses that make this product
  worth having. One book = one dictionary; pick the dictionary that matches
  the book on the device.
