# epub2stardict

Builds a book-specific English-to-Hungarian StarDict dictionary from an EPUB.
Collects headwords from the text, generates Hungarian glosses and example
sentences with an LLM, writes the StarDict files with PyGlossary.

## What this is

- **Input:** an EPUB file (`data/book.epub`) plus a 1-2 sentence book
  description (`data/book_info.txt`) used as prompt context.
- **Output:** a StarDict dictionary at `data/eng-hun-dict/eng-hun.{ifo,idx,dict.dz}`.
- **Intermediate data:** numbered JSONL files under `data/` — one input/output
  per pipeline step.

## Pipeline (script numbers = run order)

1. `100_epub_to_text.py` — EPUB → `data/100_book.txt` (ASCII-normalized).
2. `200_chunk_text.py` — text → `data/200_chunks.jsonl` (one sentence per
   line, segmented with spaCy).
3. `300_build_word_context.py` — sentences → `data/300_word_contexts.jsonl`
   (one row per unique surface word + the sentence IDs it appears in).
   Only `[a-z]{3,}` forms are kept.
4. `400_extract_word_pos.py` — `data/400_word_pos.jsonl`: one row per
   `(word, lemma, POS)` combination with its context sentence IDs.
   `BAD_POS` tags (`PROPN`, `SYM`, `PUNCT`, `X`, `SPACE`) are dropped.
   Lemmas are sanitized (`robot-` → `robot`, with mid-word junk like
   `phd → ph.d.` discarded); minority POS readings that look like tagger
   noise (e.g. `robot` ADJ × 3 vs NOUN × 320) are filtered.
5. `500_generate_definitions.py` — calls the LLM. Writes
   `data/500_word_senses.jsonl` with the Hungarian gloss, Hungarian POS
   label, two example sentences, and the active `model` name. Incremental:
   backs up the previous output to `500_word_senses_prev.jsonl` and reuses
   any row whose `input_key` (hash of `word|lemma|pos|contexts`) still matches.
6. `550_word_senses_check.py` — runs phunspell over the Hungarian glosses,
   writes failing rows to `data/550_word_senses_bad.jsonl`. Diagnostic only;
   the 600 step doesn't consult it.
7. `600_create_stardict.py` — turns `data/500_word_senses.jsonl` into a
   StarDict dictionary via PyGlossary.

Step 100 runs once per book. Steps 200/300/400 are deterministic and cache
their output on disk. Step 500 reuses good rows by `input_key`, so a re-run
only hits the API for new or previously failed words.

## Architecture

```
epub2stardict/                  # shared library
  io_jsonl.py                   # JSONL read/write/append
  text.py                       # word filtering, lemma sanitization, BAD_POS
  llm.py                        # LLMClient + prompt + JSON parser
100_…py … 600_…py               # numbered pipeline scripts
data/                           # input + JSONL intermediates + final dictionary
```

The numbered scripts are deliberately **utility-style**: each has its own
`main()`, runs standalone, and reads its config from environment variables.
Anything shared lives under `epub2stardict/`.

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
`data/500_word_senses.jsonl`. Step 600 shortens it for the headline
(strips the `provider/` prefix and a `claude-` family prefix):

```
süt (ige) (gemini-3-flash-preview)
Mom baked bread this morning.
I want to bake a pie.
```

The `input_key` deliberately does **not** include the model name. After a
model change, already-good rows are still reused (and keep their original
model label). To regenerate everything against a new model, delete
`data/500_word_senses.jsonl` before running step 500.

## GPU

`spacy.prefer_gpu()` is called in steps 200/300/400. With `cupy-cuda13x`
(in `requirements.txt`) and an NVIDIA driver present, the spaCy transformer
runs on GPU; otherwise it falls back to CPU automatically. The 200 step
processes the whole book in a single call — peak VRAM ~4-5 GiB. If
something else is holding GPU memory, replace `spacy.prefer_gpu()` with
`spacy.require_cpu()` in 200.

## Test runs

`LIMIT_WORDS=50` (env) makes the 400 step take only the first 50 unique
words from the 300 output. Every downstream script inherits the narrowed
input via the 400 file. Steps 100/200/300 still run over the whole book.

## Concurrency

Step 500 batches the words (`LLM_BATCH_SIZE` per request, default 5) and
runs the requests in a thread pool (`LLM_NUM_WORKERS` parallel calls,
default 8). For local Ollama set `LLM_NUM_WORKERS=1`.

## Design principles

- **KISS first, DRY second.** Extract a helper only when there is real
  duplication (3+ call sites). One LLM provider, one client class, no
  abstract `Service` base.
- **YAGNI.** Two providers or multi-source merging — we'll add them when
  needed. Not now.
- **Env-driven config.** No CLI argparse on the numbered scripts, no
  `config.py`. `.env` and that's it.
- **JSONL between every step.** One record per line, easy to `jq`.
- **Numbered scripts.** Run order is in the filename; no separate runner.

## What NOT to add

- Another provider abstraction next to `LLMClient`. If a real second
  provider needs different code, fold it into `LLMClient` as a branch,
  not a new class hierarchy.
- A homegrown HTTP client or retry loop. The `openai` SDK handles both.
- A custom StarDict writer. PyGlossary handles it.
- argparse on the numbered scripts. Env variables only.
- A logging framework. `print(..., flush=True)` is enough.
- Multi-source merging in step 600. Single source is the only mode.
