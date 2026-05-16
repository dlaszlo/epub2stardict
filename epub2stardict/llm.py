"""LLM client for generating dictionary definitions.

Talks to an OpenAI-compatible Chat Completions endpoint (OpenAI, OpenRouter,
Ollama OpenAI-compat mode all work). The provider isn't known here: config
comes from env vars; the client only needs `base_url` + `api_key` + `model`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Enable the structured-JSON response_format request. Strong models (Claude,
# GPT-4 family, Gemini) support it; many Llama deployments on OpenRouter don't
# and **silently return empty output**. Set to 0/false/no to disable.
_USE_JSON_FORMAT = os.environ.get("LLM_JSON_RESPONSE_FORMAT", "1").lower() not in (
    "0", "false", "no", "off", "",
)


# ---------------------------------------------------------------------------
# POS map-ek
# ---------------------------------------------------------------------------

POS_MAP_HU: dict[str, str] = {
    "NOUN": "főnév",
    "VERB": "ige",
    "AUX": "segédige",
    "ADJ": "melléknév",
    "ADV": "határozószó",
    "PRON": "névmás",
    "ADP": "elöljárószó",
    "DET": "névelő",
    "NUM": "számnév",
    "CCONJ": "kötőszó",
    "SCONJ": "kötőszó",
    "PART": "partikula",
    "INTJ": "indulatszó",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Template: $book_info and $entries_block — uses string.Template because
# str.format would choke on the {} in the JSON examples.
PROMPT_TEMPLATE = Template("""You are building an English-to-Hungarian dictionary for a specific book.

Book context:
$book_info

For each entry, return:
- "hu": Hungarian headword (1-3 words)
- "example_surface_en": example sentence using the surface form
- "example_lemma_en": example sentence using the lemma

Hungarian headword conventions (Akadémiai dictionary style):
- Verbs: 3rd person singular indicative ("fut", "süt", "érkezik"). NOT infinitive ("futni").
- Nouns: nominative singular ("könyv", not "könyvet" or "könyvek").
- Adjectives: positive form ("magas", not "magasabb").
- No article, no personal endings, no parentheses, no alternatives.
- Never reply with "Sajnálom", "Nem tudom", "Sorry" or similar meta-responses.

Special cases for AUX (auxiliary and modal verbs) — these MUST reflect the surface
form's specific meaning, NOT just the bare lemma:
- "be" surface forms (lemma=be):
    is/am/are → "van" (present);  was/were → "volt" (past, NOT "van");  been → "volt".
- "have" AUX surface forms (lemma=have):
    has → "van" (present possession);  had → "volt" (past possession or past perfect AUX).
- "do" AUX surface forms (lemma=do):
    does → "csinál" (content) or no direct translation as pure AUX;
    did → "csinált" (content) or no direct translation as pure AUX.
- Modal verbs translate to their Hungarian modal equivalent:
    can → "tud";  could → "tudna";  may/might → "lehet";  must → "kell";
    should → "kellene";  will/shall → "fog";  would → "-na/-ne" or "szeretne";
    ought → "kellene".

English example sentence rules (the two sentences you generate):
- 6-12 words each.
- Common everyday vocabulary, CEFR A2-B1 level.
- One main clause: subject + verb + object/complement.
- Present tense, active voice (unless the meaning requires otherwise).
- No rare or literary words, no idioms, no passive voice, no subordinate clauses.
- The first sentence uses the surface form exactly as given; the second uses the lemma.
- Use the given POS. Do NOT quote or adapt the book's sentences.
- The target word should stand out — keep the rest of the sentence simpler than the word itself.

Examples of the right output:

INPUT:
ENTRY 100 | lemma: bake | surface: baked | POS: VERB
Examples from the book:
1. She had baked a cake earlier that day.

OUTPUT for entry 100:
{"index": 100, "hu": "süt", "example_surface_en": "Mom baked bread this morning.", "example_lemma_en": "I want to bake a pie."}

INPUT:
ENTRY 101 | lemma: bridge | surface: bridges | POS: NOUN
Examples from the book:
1. The old bridges of the city were beautiful.

OUTPUT for entry 101:
{"index": 101, "hu": "híd", "example_surface_en": "We crossed two bridges this morning.", "example_lemma_en": "There is a long bridge over the river."}

Now process these entries:

$entries_block

Return a JSON object of the form {"results": [...]} containing one object per input entry, in the same order, with the original "index" values. Keys per entry: index, hu, example_surface_en, example_lemma_en.
""")


# ---------------------------------------------------------------------------
# DTO-k
# ---------------------------------------------------------------------------


@dataclass
class GlossInput:
    lemma: str
    word: str
    pos: Optional[str]
    example_sentences: list[str]
    index: Optional[int] = None


@dataclass
class GlossOutput:
    index: int
    lemma: str
    word: str
    pos: Optional[str]
    pos_hu: str
    meaning_hu: str
    example_surface_en: str
    example_lemma_en: str
    ok: bool
    error: Optional[str] = None
    raw_hu: Optional[str] = None
    raw_example_surface_en: Optional[str] = None
    raw_example_lemma_en: Optional[str] = None
    raw_batch: Optional[str] = None


@dataclass
class GenerateResult:
    outputs: list[GlossOutput]
    in_tokens: int
    out_tokens: int
    model: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BAD_GLOSS_TOKENS = (
    "sajnálom", "nem tudom", "sorry", "i am sorry", "i'm sorry",
    "unknown", "nincs", "nem ismert",
)


def is_bad_gloss(gloss: str) -> bool:
    if not gloss or not gloss.strip():
        return True
    cleaned = re.sub(r"^[^0-9A-Za-záéíóöőúüűÁÉÍÓÖŐÚÜŰ]+", "", gloss.strip())
    if not cleaned:
        return True
    first_token = cleaned.split()[0]
    lower = first_token.lower()
    if any(bad in lower for bad in _BAD_GLOSS_TOKENS):
        return True
    if not re.match(r"^[a-záéíóöőúüű]", lower):
        return True
    if len(first_token) > 30:
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt build
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
_BOOK_INFO_PATH = _BASE_DIR / "data" / "book_info.txt"


def _load_book_info() -> str:
    try:
        return _BOOK_INFO_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


_BOOK_INFO = _load_book_info()


def _build_entries_block(items: list[GlossInput]) -> str:
    chunks: list[str] = []
    for seq_idx, item in enumerate(items, start=1):
        idx = item.index if item.index is not None else seq_idx
        pos_tag = item.pos or "unknown"
        examples = "".join(f"{j}. {s}\n" for j, s in enumerate(item.example_sentences, start=1))
        chunks.append(
            f"ENTRY {idx} | lemma: {item.lemma} | surface: {item.word} | POS: {pos_tag}\n"
            f"Examples from the book:\n"
            f"{examples}"
        )
    return "\n".join(chunks)


def build_prompt(items: list[GlossInput]) -> str:
    return PROMPT_TEMPLATE.substitute(
        book_info=_BOOK_INFO,
        entries_block=_build_entries_block(items),
    )


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _strip_code_fences(raw: str) -> str:
    """Some models wrap the JSON in a code fence; strip it if present."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _empty_output(item: GlossInput, idx: int, error: str, raw: str) -> GlossOutput:
    return GlossOutput(
        index=idx,
        lemma=item.lemma,
        word=item.word,
        pos=item.pos,
        pos_hu="",
        meaning_hu="",
        example_surface_en="",
        example_lemma_en="",
        ok=False,
        error=error,
        raw_batch=raw,
    )


def parse_response(raw: str, inputs: list[GlossInput]) -> list[GlossOutput]:
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[llm] JSON parse error: {e}")
        print(f"[llm] RAW RESPONSE:\n{raw}")
        return [
            _empty_output(item, item.index or i, f"json parse error: {e}", raw)
            for i, item in enumerate(inputs, start=1)
        ]

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        items_list = data["results"]
    elif isinstance(data, list):
        # Some models omit the wrapper object — accept that shape too.
        items_list = data
    else:
        print(f"[llm] JSON root has unexpected shape. RAW:\n{raw}")
        return [
            _empty_output(item, item.index or i, "json root has unexpected shape", raw)
            for i, item in enumerate(inputs, start=1)
        ]

    by_index: dict[int, dict] = {}
    for obj in items_list:
        if isinstance(obj, dict) and isinstance(obj.get("index"), int):
            by_index[obj["index"]] = obj

    outputs: list[GlossOutput] = []
    for seq_idx, item in enumerate(inputs, start=1):
        idx = item.index if item.index is not None else seq_idx
        obj = by_index.get(idx)
        pos_hu = POS_MAP_HU.get(item.pos, "") if item.pos else ""

        if obj is None:
            outputs.append(_empty_output(item, idx, f"missing entry for index {idx}", raw))
            continue

        hu = str(obj.get("hu") or "").strip()
        ex_surface = str(obj.get("example_surface_en") or "").strip()
        ex_lemma = str(obj.get("example_lemma_en") or "").strip()

        if is_bad_gloss(hu):
            outputs.append(GlossOutput(
                index=idx, lemma=item.lemma, word=item.word, pos=item.pos,
                pos_hu="", meaning_hu="", example_surface_en="", example_lemma_en="",
                ok=False, error=f"bad gloss: {hu!r}",
                raw_hu=hu, raw_example_surface_en=ex_surface, raw_example_lemma_en=ex_lemma,
                raw_batch=raw,
            ))
            continue

        outputs.append(GlossOutput(
            index=idx, lemma=item.lemma, word=item.word, pos=item.pos,
            pos_hu=pos_hu,
            meaning_hu=hu,
            example_surface_en=ex_surface,
            example_lemma_en=ex_lemma,
            ok=True,
            raw_hu=hu, raw_example_surface_en=ex_surface, raw_example_lemma_en=ex_lemma,
            raw_batch=raw,
        ))

    return outputs


# ---------------------------------------------------------------------------
# Kliens
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """OpenAI-compatible Chat Completions client.

    Env vars:
      LLM_API_KEY   — required
      LLM_BASE_URL  — required (e.g. https://openrouter.ai/api/v1)
      LLM_MODEL     — required (e.g. anthropic/claude-sonnet-4.5)

    Transient errors are handled by the SDK's own retry logic (max_retries).
    """

    model: str = field(default="")
    temperature: float = 0.1
    max_tokens: int = 2000
    max_retries: int = 5
    extra_body: dict = field(default_factory=dict)

    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self):
        api_key = os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set (.env or env).")
        if not base_url:
            raise RuntimeError("LLM_BASE_URL is not set (.env or env).")

        if not self.model:
            self.model = os.environ.get("LLM_MODEL", "")
        if not self.model:
            raise RuntimeError("LLM_MODEL is not set (.env or env).")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=self.max_retries,
        )

    def generate(self, items: list[GlossInput]) -> GenerateResult:
        if not items:
            return GenerateResult(outputs=[], in_tokens=0, out_tokens=0, model=self.model)

        prompt = build_prompt(items)

        create_kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Explicit no-tools: some Llama deployments on OpenRouter slip
            # into tool-calling mode and return empty content with
            # finish_reason="tool_calls", even when no tools list was sent.
            # This forces a plain-text response.
            "tool_choice": "none",
        }
        if _USE_JSON_FORMAT:
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        resp = self._client.chat.completions.create(**create_kwargs)

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        refusal = getattr(choice.message, "refusal", None)

        if not content:
            # 200 OK but empty content — finish_reason explains why.
            # Common cases:
            #   "stop"           — model refused on its own (often a prompt-format issue)
            #   "length"         — max_tokens ran out before any real output
            #   "content_filter" — moderation filter blocked the response
            raise ValueError(
                f"empty response from model "
                f"(finish_reason={finish_reason!r}, refusal={refusal!r})"
            )

        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        outputs = parse_response(content, items)
        return GenerateResult(outputs=outputs, in_tokens=in_tok, out_tokens=out_tok, model=self.model)
