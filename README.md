# epub2stardict

Builds a book-specific English-to-Hungarian StarDict dictionary from an EPUB.
Collects the book's vocabulary, generates Hungarian glosses and example
sentences with an LLM, writes the StarDict files with PyGlossary.

Pipeline overview: see `CLAUDE.md`.

## Prerequisites

- **Python 3.10+**
- **`dictzip`** (Debian/Ubuntu: `sudo apt install dictzip`; macOS: `brew install dictd`)
- **C compiler + Python dev headers** for `phunspell`
  (Debian/Ubuntu: `sudo apt install build-essential python3-dev`)
- An **OpenAI-compatible LLM endpoint** — OpenRouter, OpenAI, or a local Ollama

The Hungarian `hu_HU` Hunspell dictionary ships inside the `phunspell` package.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf   # ~500 MB, one-time
cp .env.example .env
# fill in LLM_API_KEY and LLM_MODEL
```

GPU works automatically with an NVIDIA card (`cupy-cuda13x` is in
`requirements.txt`). The scripts print `spaCy GPU: on` or `off (CPU fallback)`
on startup.

## Configure `.env`

```dotenv
LLM_API_KEY=sk-or-v1-...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-3-flash-preview
```

Other providers:

- **OpenAI:** `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini`
- **Ollama:** `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.1:8b`,
  `LLM_API_KEY=ollama`, `LLM_NUM_WORKERS=1`

For weaker open models that return empty output to the strict-JSON request,
set `LLM_JSON_RESPONSE_FORMAT=0`.

## Run

Put the input in `data/`:

```
data/book.epub
data/book_info.txt     # 1-2 sentences: title, author, genre — prompt context
```

Run the numbered scripts in order:

```bash
python 100_epub_to_text.py
python 200_chunk_text.py
python 300_build_word_context.py
python 400_extract_word_pos.py
python 500_generate_definitions.py    # the long one (LLM calls)
python 550_word_senses_check.py       # optional gloss spellcheck
python 600_create_stardict.py
```

Output: `data/eng-hun-dict/eng-hun.{ifo,idx,dict.dz}`. Drop into KOReader,
GoldenDict, sdcv, or any StarDict reader.

Step 500 is incremental — re-running only calls the LLM for new or previously
failed rows. Earlier steps cache their JSONL output.

## Test runs

`LIMIT_WORDS=50` in `.env` makes step 400 take only the first 50 unique words,
and every downstream step inherits the narrowed set via the 400 output.

## Tuning step 500

```dotenv
LLM_BATCH_SIZE=5        # words per request
LLM_NUM_WORKERS=8       # parallel requests
```

For Ollama, use `LLM_NUM_WORKERS=1`. On HTTP 429 (rate limit), lower workers
first, then batch size.

## Troubleshooting

- **`spaCy GPU: off (CPU fallback)`** — no NVIDIA driver or `cupy-cuda13x`
  not installed. CPU still works, just slower.
- **`empty response from model` (Llama variants)** — try `LLM_JSON_RESPONSE_FORMAT=0`.
- **`JSON parse error`** — model isn't following the JSON contract. Try a
  stronger model. Raw responses are in `data/500_word_senses.errors.jsonl`.
- **PyGlossary StarDict writer fails** — `dictzip` not installed system-wide.
