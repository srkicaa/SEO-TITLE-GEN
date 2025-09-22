"""Wrapper around OpenAI API calls for title generation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


SMALL_WORDS: tuple[str, ...] = (
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "vs",
    "via",
)

WORD_SPLIT_PATTERN = re.compile(r'([\s\-–—/]+)')

from typing import Any, Dict, List, Optional, Sequence

from .utils import (
    extract_response_json,
    extract_structured_from_response,
    load_env,
    normalize_structured_value,
    save_generated_titles_snapshot,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency missing at runtime
    OpenAI = None  # type: ignore


MODEL_OPTIONS: Sequence[Dict[str, Any]] = (
    {
        "id": "gpt-5",
        "label": "GPT-5 (high quality)",
        "prompt_cost_per_1k": 0.02,
        "completion_cost_per_1k": 0.06,
    },
    {
        "id": "gpt-5-mini",
        "label": "GPT-5 Mini",
        "prompt_cost_per_1k": 0.01,
        "completion_cost_per_1k": 0.03,
    },
    {
        "id": "gpt-5-nano",
        "label": "GPT-5 Nano (cost saver)",
        "prompt_cost_per_1k": 0.0005,
        "completion_cost_per_1k": 0.0015,
    },
    {
        "id": "gpt-4.1-mini",
        "label": "GPT-4.1 Mini",
        "prompt_cost_per_1k": 0.003,
        "completion_cost_per_1k": 0.006,
    },
)
MODEL_LOOKUP = {option["id"]: option for option in MODEL_OPTIONS}
DEFAULT_MODEL_ID = MODEL_OPTIONS[0]["id"]


@dataclass
class GenerationResult:
    """Container for generated title options."""

    titles: List[Dict[str, Any]]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Optional[Any] = None
    saved_path: Optional[str] = None


class TitleGenerationError(RuntimeError):
    """Raised when the OpenAI workflow fails."""


def generate_titles(
    *,
    titles_context: Sequence[str],
    order_payload: Dict[str, Any],
    prompt_text: str,
    model: str,
    client: Any = None,
    max_output_tokens: int = 900,
    expected_titles: Optional[int] = None,
) -> GenerationResult:
    """Invoke the OpenAI Responses API to create titles."""

    if not prompt_text.strip():
        raise TitleGenerationError("Prompt text is empty; cannot generate titles")

    if client is None:
        if OpenAI is None:
            raise TitleGenerationError(
                "openai package is not installed; install it to enable generation"
            )
        load_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise TitleGenerationError("OPENAI_API_KEY is not set; update .env or export the key before running.")
        client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are an SEO editor generating link placement titles. "
        "Always answer in strict JSON matching the requested schema, keep every suggested title brand-free (no casino/operator names or anchor keywords), "
        "and localize titles whenever regional cues appear in the prompt or are known to you."
    )

    user_prompt = prompt_text
    if titles_context and "{serp_titles" not in prompt_text:
        joined = "\n".join(f"- {title}" for title in titles_context)
        user_prompt = f"{prompt_text}\n\nReference titles:\n{joined}"

    num_expected = expected_titles
    if num_expected is None:
        try:
            num_expected = int(order_payload.get("num_titles_requested", 0))
        except (TypeError, ValueError):
            num_expected = None

    text_config: Dict[str, Any] = {
        "verbosity": "medium"
    }

    titles_schema = {
        "type": "object",
        "properties": {
            "titles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["title", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["titles"],
        "additionalProperties": False,
    }

    text_config["format"] = {
        "type": "json_schema",
        "name": "title_array",
        "schema": titles_schema,
    }

    effective_max_tokens = max_output_tokens
    if num_expected:
        calculated = 600 + num_expected * 260
        effective_max_tokens = max(max_output_tokens, min(calculated, 4096))

    request_payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": effective_max_tokens,
        "text": text_config,
    }

    if model in {"gpt-5", "gpt-5-mini"}:
        request_payload["reasoning"] = {"effort": "low"}

    response = client.responses.create(  # type: ignore[attr-defined]
        **request_payload
    )

    prompt_tokens, completion_tokens = _extract_usage(response)
    raw_json_text = _extract_response_json_v2(response)
    if raw_json_text.strip().strip('"') == "title_array":
        structured = extract_structured_from_response(response)
        if structured is not None:
            if isinstance(structured, str):
                raw_json_text = structured
            else:
                try:
                    raw_json_text = json.dumps(structured)
                except TypeError:
                    raw_json_text = ""
        if raw_json_text.strip().strip('"') == "title_array":
            retry_response = client.responses.create(  # type: ignore[attr-defined]
                **request_payload
            )
            response = retry_response
            prompt_tokens, completion_tokens = _extract_usage(response)
            raw_json_text = _extract_response_json_v2(response)

    try:
        parsed = json.loads(raw_json_text)
    except json.JSONDecodeError as exc:
        repaired = _repair_generation_json(raw_json_text)
        if repaired is None:
            debug_snapshot = _debug_dump_response(response)
            raise TitleGenerationError(
                f"Model returned invalid JSON: {exc}. Raw: {raw_json_text}. Debug: {debug_snapshot}"
            ) from exc
        parsed = json.loads(repaired)

    if isinstance(parsed, dict) and "titles" in parsed:
        parsed_items = parsed.get("titles")
    else:
        parsed_items = parsed

    if not isinstance(parsed_items, list):
        raise TitleGenerationError(
            "Expected 'titles' array in the model response"
        )

    titles: List[Dict[str, Any]] = []
    for item in parsed_items:
        if isinstance(item, dict) and "title" in item:
            normalized = dict(item)
            title_value = normalized.get("title")
            if isinstance(title_value, str):
                normalized["title"] = _enforce_title_case(title_value)
            titles.append(normalized)

    if not titles:
        debug_snapshot = _debug_dump_response(response)
        raise TitleGenerationError(
            "Model response contained no valid title objects. "
            f"Raw: {raw_json_text}. Debug: {debug_snapshot}"
        )

    cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

    saved_snapshot = save_generated_titles_snapshot(
        order_payload=dict(order_payload),
        titles=titles,
        model=model,
        prompt_text=prompt_text,
        titles_context=titles_context,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        cost_usd=cost,
    )

    return GenerationResult(
        titles=titles,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        raw_response=response,
        saved_path=str(saved_snapshot) if saved_snapshot else None,
    )


def _extract_response_json_v2(response: Any) -> str:
    output_items = getattr(response, "output", None)
    if output_items:
        for item in output_items:
            parsed_value = getattr(item, "parsed", None) or (
                item.get("parsed") if isinstance(item, dict) else None
            )
            if parsed_value:
                if hasattr(parsed_value, "model_dump"):
                    parsed_value = parsed_value.model_dump()  # type: ignore[attr-defined]
                try:
                    return json.dumps(parsed_value)
                except TypeError:
                    pass
            content = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            if isinstance(content, list):
                for block in content:
                    block_parsed = getattr(block, "parsed", None) or (
                        block.get("parsed") if isinstance(block, dict) else None
                    )
                    if block_parsed:
                        try:
                            return json.dumps(block_parsed)
                        except TypeError:
                            pass
    fallback = ""
    try:
        fallback = extract_response_json(response)
    except RuntimeError:
        fallback = ""
    cleaned = fallback.strip()
    if cleaned and cleaned.strip('"') != "title_array":
        return cleaned

    structured = extract_structured_from_response(response)
    if structured is not None:
        if isinstance(structured, str):
            text_value = structured.strip()
            if text_value and text_value.strip('"') != "title_array":
                return text_value
        else:
            try:
                return json.dumps(structured)
            except TypeError:
                pass
    return cleaned


def _enforce_title_case(text: str) -> str:
    if not text:
        return text

    tokens = WORD_SPLIT_PATTERN.split(text)
    result: List[str] = []
    word_position = 0

    for token in tokens:
        if token == "":
            continue
        if WORD_SPLIT_PATTERN.fullmatch(token):
            result.append(token)
            continue

        prefix_match = re.match(r"^[^A-Za-z0-9]*", token)
        prefix = prefix_match.group(0) if prefix_match else ""
        core = token[len(prefix):]
        suffix_match = re.search(r"[^A-Za-z0-9]*$", core)
        suffix = suffix_match.group(0) if suffix_match else ""
        word = core[: len(core) - len(suffix)] if suffix else core

        if not word:
            result.append(token)
            continue

        lower_word = word.lower()
        if word.isupper() and len(word) > 1:
            processed = word
        elif word_position != 0 and lower_word in SMALL_WORDS:
            processed = lower_word
        else:
            processed = lower_word[:1].upper() + lower_word[1:]

        word_position += 1
        rebuilt = f"{prefix}{processed}{suffix}"
        result.append(rebuilt)

    return "".join(result)


def _debug_dump_response(response: Any, *, limit: int = 12000) -> str:
    """Best-effort serialization of an OpenAI response for debugging failures."""

    snapshots: List[str] = []
    for attr in ("model_dump_json", "model_dump", "to_dict"):
        if hasattr(response, attr):
            try:
                value = getattr(response, attr)()
                if isinstance(value, str):
                    snapshots.append(value)
                else:
                    snapshots.append(json.dumps(value, default=str))
            except Exception:
                continue
    text_attr = getattr(response, "text", None)
    if text_attr is not None:
        try:
            if isinstance(text_attr, str):
                snapshots.append(text_attr)
            elif hasattr(text_attr, "model_dump_json"):
                snapshots.append(text_attr.model_dump_json())  # type: ignore[attr-defined]
            elif hasattr(text_attr, "model_dump"):
                snapshots.append(json.dumps(text_attr.model_dump(), default=str))  # type: ignore[attr-defined]
            elif hasattr(text_attr, "to_dict"):
                snapshots.append(json.dumps(text_attr.to_dict(), default=str))  # type: ignore[attr-defined]
            elif isinstance(text_attr, (dict, list)):
                snapshots.append(json.dumps(text_attr, default=str))
            else:
                snapshots.append(str(text_attr))
        except Exception:
            pass
    if hasattr(response, "output"):
        try:
            snapshots.append(json.dumps(normalize_structured_value(response.output), default=str))
        except Exception:
            pass
    response_format_attr = getattr(response, "response_format", None)
    if response_format_attr is not None:
        try:
            snapshots.append(json.dumps(response_format_attr, default=str))
        except Exception:
            snapshots.append(str(response_format_attr))
    if not snapshots:
        snapshots.append(repr(response))

    merged = " | ".join(snapshot for snapshot in snapshots if snapshot)
    merged = merged[:limit]
    return merged


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    prompt_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "output_tokens", None) or getattr(
        usage, "completion_tokens", 0
    )
    return int(prompt_tokens or 0), int(completion_tokens or 0)



def _repair_generation_json(raw: str) -> Optional[str]:
    """Attempt to salvage simple JSON array formatting issues."""

    start = raw.find('[')
    if start == -1:
        return None
    candidate = raw[start:]

    closing_bracket = candidate.rfind(']')
    if closing_bracket != -1:
        candidate = candidate[: closing_bracket + 1]
    else:
        closing_brace = candidate.rfind('}')
        if closing_brace == -1:
            return None
        candidate = candidate[: closing_brace + 1] + ']'

    open_braces = candidate.count('{')
    close_braces = candidate.count('}')
    if close_braces < open_braces:
        candidate += '}' * (open_braces - close_braces)

    open_brackets = candidate.count('[')
    close_brackets = candidate.count(']')
    if close_brackets < open_brackets:
        candidate += ']' * (open_brackets - close_brackets)

    candidate = re.sub(r',\s*\]$', ']', candidate, flags=re.S)

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None

def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    option = MODEL_LOOKUP.get(model)
    if not option:
        return 0.0
    prompt_price = option.get("prompt_cost_per_1k") or 0.0
    completion_price = option.get("completion_cost_per_1k") or 0.0
    cost = (prompt_tokens / 1000.0) * prompt_price + (completion_tokens / 1000.0) * completion_price
    return round(cost, 6)


__all__ = [
    "DEFAULT_MODEL_ID",
    "GenerationResult",
    "MODEL_OPTIONS",
    "TitleGenerationError",
    "generate_titles",
]
