"""Prompt, output schema and response parsing.

The system prompt, the JSON user payload, the output JSON Schema and the tolerant
response parser are the same contract every provider uses. Providers differ only in
how they enforce the schema (their native structured-output mechanism).
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from .backend import TranslationRequest
from .errors import TranslationError

SNIPPET_LIMIT = 2000


def _reject_constant(name: str) -> Any:
    # serde_json does not accept NaN or Infinity.
    raise ValueError(f"invalid number {name}")


def prompts(request: TranslationRequest) -> tuple[str, str]:
    """``(system, user)``: the system prompt and the serialized input payload."""
    payload = {
        "source_language": request.source_language.tag if request.source_language else None,
        "target_language": request.target_language.tag,
        "context": [{"source": c.source, "translation": c.translation} for c in request.context],
        "segments": [{"id": index, "text": text} for index, text in enumerate(request.segments)],
    }
    # serde_json::to_string: compact, non-ASCII kept as-is.
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return translation_system_prompt(request), user


def translations(provider: str, text: str, source_segments: list[str]) -> list[str]:
    """Parses the model's JSON reply and restores input order.

    Fails closed on malformed JSON or a wrong shape. Also tolerates:
    ids given as numeric strings, ids out of range (ignored), duplicate ids (first wins),
    and missing ids (the source text is kept for that segment).
    """
    try:
        output = _parse_output(text)
    except ValueError as error:
        raise TranslationError(
            f"{provider} returned invalid translation JSON for {len(source_segments)} segments; "
            f"response was: {snippet(text)}: {error}"
        ) from error
    result = list(source_segments)
    filled = [False] * len(source_segments)
    for segment_id, segment_text in output:
        if segment_id < len(result) and not filled[segment_id]:
            result[segment_id] = segment_text
            filled[segment_id] = True
    return result


def _parse_output(text: str) -> list[tuple[int, str]]:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{error.msg} at line {error.lineno} column {error.colno}") from error
    if not isinstance(value, dict):
        raise ValueError("invalid type: expected struct TranslationOutput")
    if "translations" not in value:
        raise ValueError("missing field `translations`")
    entries = value["translations"]
    if not isinstance(entries, list):
        raise ValueError("invalid type: expected a sequence")
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid type: expected struct TranslationOutputSegment")
        if "id" not in entry:
            raise ValueError("missing field `id`")
        if "text" not in entry:
            raise ValueError("missing field `text`")
        parsed.append((_segment_id(entry["id"]), _segment_text(entry["text"])))
    return parsed


def _segment_id(value: Any) -> int:
    # serde untagged: an unsigned integer, or a string that parses as one after trimming.
    if isinstance(value, bool):
        raise ValueError("invalid type: boolean, expected usize")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("invalid value: integer, expected usize")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isascii() and stripped.isdigit():
            return int(stripped)
        raise ValueError(f"invalid digit found in string: {value!r}")
    raise ValueError("invalid type: expected usize or string")


def _segment_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid type: expected a string")
    return value


def snippet(text: str) -> str:
    """One-line, bounded rendering of a model response for error messages."""
    trimmed = text.strip()
    elided = len(trimmed) > SNIPPET_LIMIT
    visible = trimmed[:SNIPPET_LIMIT] if elided else trimmed
    pieces = []
    for character in visible:
        if character == "\n":
            pieces.append("\\n")
        elif character == "\r":
            pieces.append("\\r")
        elif character == "\t":
            pieces.append("\\t")
        elif unicodedata.category(character) == "Cc":
            pieces.append(f"\\u{{{ord(character):04x}}}")
        else:
            pieces.append(character)
    output = "".join(pieces)
    if elided:
        output += f"… ({len(trimmed.encode('utf-8'))} bytes total)"
    return output


def output_schema(expected: int) -> dict[str, Any]:
    """The JSON Schema every provider enforces: exactly one ``{id, text}`` per input segment."""
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(expected - 1, 0),
                            "description": "The ID copied from the corresponding input segment.",
                        },
                        "text": {
                            "type": "string",
                            "description": "The translation of the input segment with this ID.",
                        },
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def translation_system_prompt(request: TranslationRequest) -> str:
    source = request.source_language.display_name if request.source_language else "the detected source language"
    target = request.target_language.display_name
    prompt = (
        "You are a professional manga translator.\n"
        "\n"
        "Translation requirements:\n"
        f"- Translate every input segment from {source} into natural {target}.\n"
        "- Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects.\n"
        "- Localize idioms and sound effects naturally while keeping wording concise enough for speech bubbles.\n"
        "- Use surrounding segments only for disambiguation and continuity; never merge or split segments.\n"
        f"- Write every translated `text` value only in {target}; do not include source text, notes, explanations, or alternatives.\n"
        f"- Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {target} conventions.\n"
        "\n"
        "Output requirements:\n"
        "- Each input segment has a numeric `id`.\n"
        "- Return only a JSON object whose `translations` array contains one object with `id` and translated `text` for every input segment.\n"
        "- Copy every input ID exactly once; order does not matter.\n"
        "- Never merge, split, omit, duplicate, or add segments."
    )
    if request.context:
        prompt += (
            "\n\nContext requirements:\n"
            "Use the supplied context only to preserve terminology, character voice, and dialogue continuity.\n"
            "Do not translate or return the context entries."
        )
    if request.image is not None:
        prompt += (
            "\n\nImage requirements:\n"
            "Use the attached original page image as visual context for speaker identity, tone, layout, and ambiguous OCR.\n"
            "Translate only the supplied segments; do not add text seen in the image that is absent from the input segments."
        )
    instructions = (request.instructions or "").strip()
    if instructions:
        prompt += "\n\nAdditional instructions:\n" + instructions
    return prompt
