"""Utility helpers for Streamlit orchestration of DataForSEO SERP fetches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Add parent directory to path so we can import dfs_serp_fetcher
sys.path.insert(0, str(Path(__file__).parent.parent))

from dfs_serp_fetcher import SerpFetcherError, run_fetcher  # type: ignore

try:  # optional dependency for loading credentials from .env
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled gracefully at runtime
    load_dotenv = None  # type: ignore

try:  # optional dependency used when invoking GPT-5-nano to parse orders
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
GENERATED_DIR = ASSETS_DIR / "generated"
LOG_DIR = ASSETS_DIR / "logs"
LOG_PATH = LOG_DIR / "app.log"
DEFAULT_SERP_RESULT_LIMIT = 50
DATAFORSEO_FETCH_LIMIT = 100

HEADER_ALIASES: Dict[str, Iterable[str]] = {
    "domain": ("domain", "target domain", "site", "target", "target_domain"),
    "anchor_keyword": (
        "anchor",
        "anchor keyword",
        "anchor text",
        "anchor_keyword",
    ),
    "anchor_country": (
        "anchor country",
        "brand country",
        "anchor region",
        "brand region",
        "anchor_country",
    ),
    "target_url": (
        "target url",
        "url",
        "anchor url",
        "destination url",
        "target_url",
        "anchor_url",
    ),
    "serp_keyword": (
        "serp keyword",
        "keyword",
        "search keyword",
        "query",
        "serp_keyword",
    ),
}
RECOGNIZED_COLUMNS = tuple(HEADER_ALIASES.keys())
AI_PARSE_MODEL = "gpt-5-nano"
MODELS_WITHOUT_TEMPERATURE = {"gpt-5-nano"}

_HEURISTIC_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("crypto", "bitcoin", "ethereum", "blockchain"), "cryptocurrency"),
    (("cashback", "credit card", "credit-card", "credit"), "credit cards"),
    (("poker", "casino", "slots", "roulette", "blackjack", "plinko"), "online casino"),
    (("trading", "broker", "forex", "stocks"), "online trading"),
    (("loan", "mortgage", "debt", "finance"), "personal finance"),
    (("insurance",), "insurance comparison"),
    (("travel", "holiday", "tour"), "travel guide"),
    (("betting", "sportsbook"), "sports betting"),
    (("seo", "marketing"), "digital marketing"),
]

DOMAIN_PATTERN = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$")


def _get_attr(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def create_openai_client(*, api_key: Optional[str] = None) -> Any:
    """Return an OpenAI client while working around old httpx proxies args."""

    if OpenAI is None:
        raise RuntimeError("openai package is required but is not installed")

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    try:
        http_client = None
        import httpx  # type: ignore

        if "http_client" in inspect.signature(OpenAI.__init__).parameters:
            try:
                http_client = httpx.Client()
            except TypeError:
                class _PatchedHttpxClient(httpx.Client):
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        kwargs.pop("proxies", None)
                        super().__init__(*args, **kwargs)

                http_client = _PatchedHttpxClient()

        if http_client is not None:
            client_kwargs["http_client"] = http_client
    except Exception:  # pragma: no cover - guard against missing httpx
        LOGGER.debug("Unable to initialize custom http client for OpenAI", exc_info=True)

    return OpenAI(**client_kwargs)


def _schema_hint(text_config: Optional[Dict[str, Any]]) -> str:
    if not text_config:
        return "Respond with valid JSON."
    format_block = text_config.get("format") if isinstance(text_config, dict) else None
    if isinstance(format_block, dict) and format_block.get("type") == "json_schema":
        schema = format_block.get("schema")
        try:
            schema_snippet = json.dumps(schema, indent=2)
        except TypeError:
            schema_snippet = str(schema)
        return (
            "Respond with valid JSON matching this schema strictly:\n"
            f"{schema_snippet}"
        )
    return "Respond with valid JSON that matches the requested schema."


def invoke_openai_response(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    text_config: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Any:
    """Call OpenAI responses API, falling back to chat.completions when unavailable."""

    responses_client = getattr(client, "responses", None)
    responses_create = getattr(responses_client, "create", None) if responses_client else None

    if callable(responses_create):
        payload: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }
        if text_config is not None:
            payload["text"] = text_config
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if (
            temperature is not None
            and model not in MODELS_WITHOUT_TEMPERATURE
        ):
            try:
                if "temperature" in inspect.signature(responses_create).parameters:
                    payload["temperature"] = temperature
            except (TypeError, ValueError):
                pass
        return responses_create(**payload)

    chat_client = getattr(client, "chat", None)
    completions = getattr(chat_client, "completions", None) if chat_client else None
    completions_create = getattr(completions, "create", None) if completions else None

    if callable(completions_create):
        schema_instruction = _schema_hint(text_config)
        augmented_user_prompt = f"{user_prompt}\n\n{schema_instruction}" if user_prompt else schema_instruction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_user_prompt},
        ]

        response_format_payload: Optional[Dict[str, Any]] = None
        if isinstance(text_config, dict):
            format_block = text_config.get("format")
            if (
                isinstance(format_block, dict)
                and format_block.get("type") == "json_schema"
                and isinstance(format_block.get("schema"), dict)
            ):
                response_format_payload = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": format_block.get("name", "structured_response"),
                        "schema": format_block["schema"],
                        "strict": True,
                    },
                }

        base_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if (
            temperature is not None
            and model not in MODELS_WITHOUT_TEMPERATURE
        ):
            base_payload["temperature"] = max(min(temperature, 2.0), 0.0)
        response_format_supported = True
        if response_format_payload is not None:
            base_payload["response_format"] = response_format_payload

        last_error: Optional[Exception] = None
        for token_field in ("max_completion_tokens", "max_tokens"):
            payload = dict(base_payload)
            payload[token_field] = max_output_tokens
            try:
                if reasoning is not None:
                    payload["reasoning"] = reasoning
                return completions_create(**payload)
            except TypeError as exc:
                lower_error = str(exc).lower()
                if f"unexpected keyword argument '{token_field}'" in lower_error:
                    last_error = exc
                    continue
                if "reasoning" in lower_error and "unexpected" in lower_error:
                    payload.pop("reasoning", None)
                    try:
                        return completions_create(**payload)
                    except Exception as exc_inner:
                        last_error = exc_inner
                        continue
                if response_format_supported and "response_format" in lower_error and "unexpected" in lower_error:
                    response_format_supported = False
                    base_payload.pop("response_format", None)
                    payload.pop("response_format", None)
                    try:
                        if reasoning is not None and "reasoning" not in lower_error:
                            payload["reasoning"] = reasoning
                        return completions_create(**payload)
                    except Exception as exc_inner:
                        last_error = exc_inner
                        continue
                raise
            except Exception as exc:  # pragma: no cover - fallback for API errors
                message = str(exc).lower()
                normalized_field = token_field.replace("_", "")
                if "unsupported parameter" in message and normalized_field in message.replace("_", ""):
                    last_error = exc
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to invoke chat.completions with available token parameters")

    raise RuntimeError(
        "OpenAI client does not provide responses or chat.completions endpoints; upgrade the openai package."
    )


def extract_finish_reason(response: Any) -> Optional[str]:
    output_items = _get_attr(response, "output")
    if isinstance(output_items, list) and output_items:
        first = output_items[0]
        finish_reason = _get_attr(first, "finish_reason")
        if finish_reason:
            return str(finish_reason)

    choices = _get_attr(response, "choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        finish_reason = _get_attr(first, "finish_reason")
        if finish_reason:
            return str(finish_reason)

    return None


def extract_primary_message_text(response: Any) -> Optional[str]:
    choices = _get_attr(response, "choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = _get_attr(first, "message")
        if message is not None:
            parsed_payload = _get_attr(message, "parsed")
            if parsed_payload:
                if isinstance(parsed_payload, str):
                    return parsed_payload.strip()
                try:
                    return json.dumps(parsed_payload)
                except TypeError:
                    pass
            content = _get_attr(message, "content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                collected: List[str] = []
                for block in content:
                    text_value = _get_attr(block, "text")
                    if isinstance(text_value, str):
                        collected.append(text_value)
                if collected:
                    return "".join(collected).strip()

    output_items = _get_attr(response, "output")
    if isinstance(output_items, list) and output_items:
        first = output_items[0]
        parsed_payload = _get_attr(first, "parsed")
        if parsed_payload:
            if isinstance(parsed_payload, str):
                return parsed_payload.strip()
            try:
                return json.dumps(parsed_payload)
            except TypeError:
                pass
        content = _get_attr(first, "content")
        if isinstance(content, list):
            collected = []
            for block in content:
                text_value = _get_attr(block, "text")
                if isinstance(text_value, str):
                    collected.append(text_value)
            if collected:
                return "".join(collected).strip()

    return None


@dataclass
class OrderInput:
    """Represents a single link building order."""

    domain: str
    anchor_keyword: str
    target_url: str
    anchor_country: str = ""
    serp_keyword: str = ""
    extra_context: str = ""

    def sanitized_domain(self) -> str:
        return sanitize_domain(self.domain)

    def to_dict(self) -> Dict[str, str]:
        return {
            "domain": self.domain,
            "anchor_keyword": self.anchor_keyword,
            "anchor_country": self.anchor_country,
            "target_url": self.target_url,
            "serp_keyword": self.serp_keyword,
            "extra_context": self.extra_context,
        }


@dataclass
class FetchSettings:
    """Configurable knobs for the DataForSEO fetch workflow."""

    language_name: str = "English"
    location_name: str = "United States"
    device: str = "desktop"
    os: str = "windows"
    depth: int = 100
    poll_interval: float = 5.0
    max_attempts: int = 720  # 5s * 720 ~= 60 minutes total wait
    live: bool = False
    use_site_operator: bool = True
    default_serp_keyword: str = ""
    tag: Optional[str] = None
    request_timeout_ms: int = 120_000


@dataclass
class SerpFetchResult:
    """Container for the SERP payload used by the UI."""

    query: str
    titles: List[Dict[str, Any]]
    raw: Dict[str, Any]
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)


def ensure_dirs() -> None:
    """Ensure asset directories exist before caching."""

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Load .env if python-dotenv is available."""

    if load_dotenv:
        load_dotenv()


def sanitize_domain(domain: str) -> str:
    """Strip protocol and trailing slashes for DataForSEO queries."""

    cleaned = domain.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("/")
    return cleaned


def enable_debug_logging(enabled: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if enabled else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler_found = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(LOG_PATH):
            handler.setLevel(level)
            handler_found = True
    if not handler_found:
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    LOGGER.log(logging.INFO if enabled else logging.DEBUG, "Debug logging %s", "enabled" if enabled else "disabled")


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value.strip(), flags=re.IGNORECASE))


def _looks_like_domain(value: str) -> bool:
    cleaned = value.strip().lower()
    if not cleaned or " " in cleaned:
        return False
    cleaned = cleaned.strip("/")
    return bool(DOMAIN_PATTERN.match(cleaned))


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc or parsed.path
    return host.strip("/")


def compute_generation_hash(
    order: Dict[str, Any], *, serp_keyword: str, model: str, prompt: str
) -> str:
    components = [
        str(order.get("domain", "")).strip().lower(),
        str(order.get("target_url", "")).strip().lower(),
        str(order.get("anchor_keyword", "")).strip().lower(),
        str(order.get("anchor_country", "")).strip().lower(),
        serp_keyword.strip().lower(),
        model.strip().lower(),
        prompt.strip(),
    ]
    base = "|".join(components)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def normalize_structured_value(value: Any) -> Optional[Any]:
    """Normalize nested responses into JSON-friendly primitives."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        placeholder = stripped.strip('"')
        if placeholder in {"title_array", "order_batch"}:
            return None
        if _looks_like_json_fragment(stripped):
            return stripped
        brace_positions = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos != -1]
        if brace_positions:
            start = min(brace_positions)
            candidate = stripped[start:].strip()
            if _looks_like_json_fragment(candidate):
                return candidate
        return None
    if isinstance(value, dict):
        if not value:
            return None
        for placeholder_key in ("title_array", "order_batch"):
            if placeholder_key in value:
                nested = normalize_structured_value(value.get(placeholder_key))
                if nested is not None:
                    return nested
        if "value" in value:
            nested = normalize_structured_value(value.get("value"))
            if nested is not None:
                return nested
        for key in ("output", "outputs", "data", "content"):
            if key in value:
                nested = normalize_structured_value(value.get(key))
                if nested is not None:
                    return nested
        if "titles" in value and isinstance(value["titles"], list):
            return value
        if "orders" in value and isinstance(value["orders"], list):
            return value
        for candidate in value.values():
            nested = normalize_structured_value(candidate)
            if nested is not None:
                if isinstance(nested, str) and not _looks_like_json_fragment(nested):
                    continue
                return nested
        return None
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, dict) for item in value):
            for item in value:
                nested = normalize_structured_value(item)
                if nested is not None:
                    if isinstance(nested, str) and not _looks_like_json_fragment(nested):
                        continue
                    return nested
            if _looks_like_order_collection(value) or _looks_like_titles_collection(value):
                return value
            return None
        if all(isinstance(item, str) for item in value):
            for item in value:
                nested = normalize_structured_value(item)
                if nested is not None:
                    if isinstance(nested, str) and not _looks_like_json_fragment(nested):
                        continue
                    return nested
        for item in value:
            nested = normalize_structured_value(item)
            if nested is not None:
                if isinstance(nested, str) and not _looks_like_json_fragment(nested):
                    continue
                return nested
        return None
    return None


def _looks_like_order_collection(value: Sequence[Any]) -> bool:
    required = {"domain", "anchor_keyword", "target_url"}
    for item in value:
        if not isinstance(item, dict):
            return False
        keys = {str(key).lower() for key in item.keys()}
        if not required.issubset(keys):
            return False
    return True


def _looks_like_titles_collection(value: Sequence[Any]) -> bool:
    candidate_keys = {"title", "url", "snippet"}
    matches = 0
    for item in value:
        if not isinstance(item, dict):
            return False
        keys = {str(key).lower() for key in item.keys()}
        if keys & candidate_keys:
            matches += 1
    return matches == len(value) and matches > 0


def _looks_like_json_fragment(value: str) -> bool:
    stripped = value.lstrip()
    if not stripped:
        return False
    return stripped[0] in "{["


def _find_first_key(payload: Any, keys: set[str]) -> Optional[Any]:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key]:
                value = payload[key]
                if value:
                    return value
        for value in payload.values():
            found = _find_first_key(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_key(item, keys)
            if found is not None:
                return found
    return None


def extract_structured_from_response(response: Any) -> Optional[Any]:
    """Attempt to locate structured payloads emitted by the Responses API."""

    candidates: List[Any] = []

    def append_candidate(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, dict, list)):
            candidates.append(value)
            return
        for attr in ("model_dump", "to_dict"):
            if hasattr(value, attr):
                try:
                    extracted = getattr(value, attr)()
                    if isinstance(extracted, (str, dict, list)):
                        append_candidate(extracted)
                        return
                except Exception:
                    continue
        if hasattr(value, "model_dump_json"):
            try:
                extracted = json.loads(value.model_dump_json())  # type: ignore[attr-defined]
                append_candidate(extracted)
                return
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            snapshot = {
                key: getattr(value, key)
                for key in value.__dict__.keys()
                if not key.startswith("_")
            }
            if snapshot:
                candidates.append(snapshot)
                return
        try:
            iterator = list(value)  # type: ignore[arg-type]
            if iterator:
                candidates.extend(iterator)
        except TypeError:
            pass

    append_candidate(response)
    for attr in ("output", "text", "content", "data", "output_text"):
        append_candidate(getattr(response, attr, None))

    for candidate in list(candidates):
        if isinstance(candidate, dict):
            for key in ("text", "data", "output", "outputs", "messages", "choices"):
                append_candidate(candidate.get(key))
        elif isinstance(candidate, list):
            for item in candidate:
                append_candidate(item)

    for candidate in candidates:
        cleaned = normalize_structured_value(candidate)
        if cleaned is not None:
            return cleaned
        parsed = _find_first_key(
            candidate,
            {"parsed", "json", "data", "value", "output", "outputs", "content", "text"},
        )
        cleaned = normalize_structured_value(parsed)
        if cleaned is not None:
            return cleaned
    return None


def _slugify_filename(value: str, *, fallback: str = "titles", max_length: int = 48) -> str:
    """Collapse arbitrary text into a filesystem-safe slug."""

    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower())
    normalized = normalized.strip("-")
    if not normalized:
        normalized = fallback
    return normalized[:max_length]


def get_cache_path(order: OrderInput, *, serp_keyword: str, limit: int) -> Path:
    """Deterministically derive a cache filename for an order."""

    key = f"{order.sanitized_domain()}|{serp_keyword}|{limit}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    filename = f"serp_{digest}_limit{limit}.json"
    return SAMPLES_DIR / filename


def load_cached_serp(order: OrderInput, *, serp_keyword: str, limit: int) -> Optional[SerpFetchResult]:
    """Return a cached SERP payload when available."""

    ensure_dirs()
    cache_path = get_cache_path(order, serp_keyword=serp_keyword, limit=limit)
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            titles = raw.get("extracted") or []
            return SerpFetchResult(
                query=raw.get("query", ""),
                titles=titles,
                raw=raw,
                source="cache",
                meta={"cache_path": str(cache_path)},
            )
        except json.JSONDecodeError as exc:  # corrupted cache
            LOGGER.warning("Failed to parse cached SERP payload %s: %s", cache_path, exc)
            return None
    return None


def cache_serp_result(order: OrderInput, *, serp_keyword: str, limit: int, payload: Dict[str, Any]) -> Optional[Path]:
    """Persist the raw SERP response for offline usage."""

    ensure_dirs()
    cache_path = get_cache_path(order, serp_keyword=serp_keyword, limit=limit)
    try:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to write SERP cache %s: %s", cache_path, exc)
        return None
    return cache_path


def save_generated_titles_snapshot(
    order_payload: Dict[str, Any],
    *,
    titles: Sequence[Dict[str, Any]],
    model: str,
    prompt_text: str,
    titles_context: Sequence[str],
    usage: Dict[str, Any],
    cost_usd: float,
) -> Optional[Path]:
    """Persist generated titles for later analysis without altering runtime flow."""

    ensure_dirs()
    domain = sanitize_domain(str(order_payload.get("domain", ""))) or "unknown-domain"
    target_dir = GENERATED_DIR / domain
    target_dir.mkdir(parents=True, exist_ok=True)

    slug_source = str(
        order_payload.get("anchor_keyword")
        or order_payload.get("serp_keyword")
        or order_payload.get("target_url")
        or order_payload.get("domain")
        or "titles"
    )
    slug = _slugify_filename(slug_source)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    unique_suffix = uuid.uuid4().hex[:6]
    filename = f"{timestamp}_{slug}_{unique_suffix}.json"
    file_path = target_dir / filename

    snapshot = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": model,
        "order": order_payload,
        "titles": list(titles),
        "titles_context": list(titles_context),
        "prompt_text": prompt_text,
        "usage": usage,
        "cost_usd": cost_usd,
    }

    try:
        file_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to write generation snapshot %s: %s", file_path, exc)
        return None

    return file_path


def fetch_serp(
    order: OrderInput,
    *,
    settings: FetchSettings,
    limit: int = DEFAULT_SERP_RESULT_LIMIT,
    use_cache: bool = False,
    cache_only: bool = False,
    debug: bool = False,
) -> SerpFetchResult:
    """Fetch SERP titles for an order, honoring caching options."""

    load_env()
    serp_keyword = order.serp_keyword or settings.default_serp_keyword
    if not serp_keyword:
        raise ValueError("A SERP keyword is required for this order")

    if use_cache or cache_only:
        cached = load_cached_serp(order, serp_keyword=serp_keyword, limit=limit)
        if cached:
            return cached
        if cache_only:
            raise FileNotFoundError("No cached SERP payload available for this order")

    args = argparse.Namespace(
        dfs_login=None,
        dfs_password=None,
        live=settings.live,
        query=None,
        domain=order.sanitized_domain(),
        serp_keyword=serp_keyword,
        no_site_operator=not settings.use_site_operator,
        language_name=settings.language_name,
        location_name=settings.location_name,
        device=settings.device,
        os=settings.os,
        depth=settings.depth,
        num_titles=limit,
        tag=settings.tag,
        postback_url=None,
        poll_interval=settings.poll_interval,
        max_attempts=settings.max_attempts,
        debug=debug,
        timeout_ms=settings.request_timeout_ms,
    )

    try:
        payload = run_fetcher(args)
    except SerpFetcherError:
        raise
    except Exception as exc:
        raise SerpFetcherError(f"Unexpected failure invoking run_fetcher: {exc}") from exc

    titles = payload.get("extracted") or []
    result = SerpFetchResult(
        query=payload.get("query", ""),
        titles=titles,
        raw=payload,
        source="api",
        meta={"limit": limit},
    )

    cache_serp_result(order, serp_keyword=serp_keyword, limit=limit, payload=payload)
    return result


def list_cached_payloads() -> List[Path]:
    """Return all cached SERP payload paths."""

    ensure_dirs()
    return sorted(SAMPLES_DIR.glob("serp_*.json"))


def parse_orders_text(text: str) -> Tuple[List[OrderInput], List[str], bool]:
    """Parse pasted orders from text, returning orders, errors, and AI fallback flag."""

    stripped = text.strip()
    if not stripped:
        return [], [], False

    rows, headers = _try_parse_csv(stripped)
    if rows:
        orders, errors = _rows_to_orders(rows, headers=headers)
        return orders, errors, False

    heuristic_orders, heuristic_errors = _heuristic_parse(stripped)
    needs_ai = not heuristic_orders
    return heuristic_orders, heuristic_errors, needs_ai


def _try_parse_csv(text: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    """Attempt to parse text as CSV/TSV and return rows with optional headers."""

    sample = "\n".join(line for line in text.splitlines() if line.strip())
    if not sample:
        return [], None

    delimiter = "\t" if "\t" in sample else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        pass

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: List[List[str]] = []
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        rows.append([cell.strip() for cell in row])

    if not rows:
        return [], None

    headers: Optional[List[str]] = None
    if _looks_like_header(rows[0]):
        headers = [cell.strip() for cell in rows.pop(0)]
    return rows, headers


def _looks_like_header(cells: Sequence[str]) -> bool:
    sample = "".join(cells).lower()
    return any(alias in sample for values in HEADER_ALIASES.values() for alias in values)


def _rows_to_orders(rows: List[List[str]], *, headers: Optional[List[str]]) -> Tuple[List[OrderInput], List[str]]:
    orders: List[OrderInput] = []
    errors: List[str] = []

    if headers:
        header_map = _build_header_map(headers)
    else:
        header_map = None

    for index, row in enumerate(rows, start=1):
        mapped = _map_row(row, header_map=header_map)
        if isinstance(mapped, OrderInput):
            orders.append(mapped)
        else:
            errors.append(f"Row {index}: {mapped}")
    return orders, errors


def _build_header_map(headers: Sequence[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for idx, header in enumerate(headers):
        key = header.strip().lower()
        for canonical, aliases in HEADER_ALIASES.items():
            if key == canonical or key in aliases:
                mapping[idx] = canonical
                break
    return mapping


def _map_row(row: Sequence[str], *, header_map: Optional[Dict[int, str]]) -> OrderInput | str:
    values: Dict[str, str] = {name: "" for name in RECOGNIZED_COLUMNS}

    if header_map:
        for idx, cell in enumerate(row):
            column = header_map.get(idx)
            if column:
                values[column] = cell.strip()
        if not _looks_like_url(values.get("target_url", "")):
            for cell in row:
                if _looks_like_url(cell):
                    values["target_url"] = cell.strip()
                    break
        if not values.get("domain"):
            for cell in row:
                if _looks_like_domain(cell):
                    values["domain"] = cell.strip()
                    break
    else:
        if len(row) < 3:
            return "expected at least three columns (domain, anchor_keyword, target_url)"
        cleaned = [cell.strip() for cell in row if cell.strip()]

        target_index: Optional[int] = None
        for idx, cell in enumerate(cleaned):
            if _looks_like_url(cell):
                target_index = idx
                break

        if target_index is None:
            return "target URL must be provided in column 3"

        values["target_url"] = cleaned[target_index]

        remaining = cleaned[:target_index] + cleaned[target_index + 1 :]

        target_host = _domain_from_url(values["target_url"]).lower().lstrip("www.")

        domain_value = None
        for cell in remaining:
            if _looks_like_domain(cell):
                cell_norm = cell.lower().lstrip("www.")
                if cell_norm != target_host:
                    domain_value = cell
                    break
        if domain_value is None:
            for cell in remaining:
                if _looks_like_domain(cell):
                    domain_value = cell
                    break
        if domain_value is None and values["target_url"]:
            domain_value = _domain_from_url(values["target_url"])
        values["domain"] = domain_value or ""

        anchor_value = None
        for cell in remaining:
            if not cell or cell == domain_value or _looks_like_url(cell):
                continue
            if " " in cell or not _looks_like_domain(cell):
                anchor_value = cell
                break
        if anchor_value is None:
            for cell in remaining:
                if cell and cell != domain_value and not _looks_like_url(cell):
                    anchor_value = cell
                    break
        if anchor_value:
            values["anchor_keyword"] = anchor_value

        residue = [cell for cell in remaining if cell not in {domain_value, anchor_value}]
        if residue:
            values["anchor_country"] = residue.pop(0)
        if residue:
            values["serp_keyword"] = residue.pop(0)

        if not values["anchor_keyword"] and residue:
            values["anchor_keyword"] = residue.pop(0)

    if not values["domain"] or not values["anchor_keyword"] or not values["target_url"]:
        return "domain, anchor keyword, and target URL are required"

    if not _looks_like_url(values["target_url"]):
        return "target URL must be a fully-qualified http(s) URL"

    return OrderInput(
        domain=values["domain"],
        anchor_keyword=values["anchor_keyword"],
        target_url=values["target_url"],
        anchor_country=values.get("anchor_country", ""),
        serp_keyword=values.get("serp_keyword", ""),
        extra_context=values.get("extra_context", ""),
    )


def _heuristic_parse(text: str) -> Tuple[List[OrderInput], List[str]]:
    """Fallback parser for lines lacking delimiters (e.g., pasted without separators)."""

    orders: List[OrderInput] = []
    errors: List[str] = []
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.lower().startswith("target domain"):
            continue
        match = re.search(r"https?://", line)
        if not match:
            errors.append(f"Line {index}: unable to locate target URL; use AI parse")
            continue
        target_url = line[match.start():].strip()
        prefix = line[: match.start()].strip()
        if not prefix:
            errors.append(f"Line {index}: missing domain/anchor content")
            continue
        parts = prefix.split()
        domain = parts[0]
        anchor = prefix[len(domain) :].strip()
        if not anchor:
            errors.append(f"Line {index}: missing anchor keyword")
            continue
        orders.append(
            OrderInput(
                domain=domain,
                anchor_keyword=anchor,
                target_url=target_url,
            )
        )
    return orders, errors


def heuristic_serp_keyword(anchor_keyword: str, target_url: str) -> str:
    anchor_lower = (anchor_keyword or "").lower()
    target_lower = (target_url or "").lower()

    def matches(triggers: Tuple[str, ...]) -> bool:
        return any(word in anchor_lower or word in target_lower for word in triggers)

    for triggers, keyword in _HEURISTIC_RULES:
        if matches(triggers):
            return keyword

    anchor_tokens = [token for token in re.split(r"[^a-z0-9]+", anchor_lower) if token]
    if anchor_tokens:
        primary = " ".join(anchor_tokens[:2])
        return primary.strip()
    return ""


def apply_heuristic_serp_keywords(orders: List[Dict[str, str]]) -> List[str]:
    suggestions: List[str] = []
    for order in orders:
        suggestion = heuristic_serp_keyword(order.get("anchor_keyword", ""), order.get("target_url", ""))
        order["serp_keyword"] = suggestion
        suggestions.append(suggestion)
    return suggestions


def ai_parse_orders(
    text: str,
    *,
    client: Optional[Any] = None,
    model: str = AI_PARSE_MODEL,
    temperature: float = 0.1,
    max_output_tokens: int = 800,
) -> Tuple[List[OrderInput], str]:
    """Use GPT-5-nano to convert pasted text into structured orders."""

    if not text.strip():
        return [], ""

    if client is None:
        load_env()
        if OpenAI is None:
            raise RuntimeError("openai package is required for AI parsing but is not installed")
        client = create_openai_client()

    lines = [line for line in text.splitlines() if line.strip()]
    header_hint = ""
    if lines and any(delim in lines[0] for delim in ("\t", ",")):
        header_hint = lines[0]

    prompt = (
        "You convert clipboard tables into normalized JSON orders. Each order must include the keys: "
        "domain (placement site), anchor_keyword (requested anchor text), anchor_country (empty string if unknown), "
        "target_url (destination we are linking to), and serp_keyword (broad topical keyword). Preserve text apart from trimming. "
        "If the input header lists columns like 'Target URL, Anchor, Domain', map them accordingly (domain is NOT the target URL). "
        "Respond ONLY with a JSON object exactly matching the schema. For example: {\"orders\": [{\"domain\": \"example.com\", \"anchor_keyword\": \"anchor\", \"anchor_country\": \"\", \"target_url\": \"https://example.com\", \"serp_keyword\": \"keyword\"}]}" 
        "If any field is missing, emit an empty string for it. Do not include explanations, prefix text, or reasoning."
    )

    text_config: Dict[str, Any] = {
        "verbosity": "medium",
        "format": {
            "type": "json_schema",
            "name": "order_batch",
            "schema": {
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {"type": "string"},
                                "anchor_keyword": {"type": "string"},
                                "anchor_country": {"type": "string"},
                                "target_url": {"type": "string"},
                                "serp_keyword": {"type": "string"},
                            },
                            "required": [
                                "domain",
                                "anchor_keyword",
                                "anchor_country",
                                "target_url",
                                "serp_keyword",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["orders"],
                "additionalProperties": False,
            },
        },
    }

    kwargs = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": (f"Headers: {header_hint}\n" if header_hint else "") + text,
            },
        ],
        "max_output_tokens": max_output_tokens,
        "text": text_config,
    }
    if temperature is not None and model not in MODELS_WITHOUT_TEMPERATURE:
        kwargs["temperature"] = temperature

    LOGGER.debug(
        "AI parse request prepared: header=%r rows=%d model=%s max_tokens=%s",
        header_hint,
        len(lines),
        model,
        max_output_tokens,
    )

    def _call_openai(use_schema: bool, attempt: int) -> tuple[Any, str]:
        local_text = {"verbosity": "medium"}
        if use_schema:
            local_text["format"] = kwargs["text"]["format"]

        payload = {
            "model": model,
            "input": kwargs["input"],
            "max_output_tokens": kwargs["max_output_tokens"],
            "text": local_text,
        }
        if temperature is not None and model not in MODELS_WITHOUT_TEMPERATURE:
            payload["temperature"] = temperature

        LOGGER.debug("AI parse HTTP call (attempt %d, schema=%s)", attempt, use_schema)
        response_local = invoke_openai_response(
            client,
            model=model,
            system_prompt=kwargs["input"][0]["content"],
            user_prompt=kwargs["input"][1]["content"],
            max_output_tokens=kwargs["max_output_tokens"],
            text_config=payload["text"],
            temperature=kwargs.get("temperature"),
        )

        structured_local = extract_structured_from_response(response_local)
        raw_local = ""
        if structured_local is not None:
            if isinstance(structured_local, str):
                raw_local = structured_local.strip()
            else:
                raw_local = json.dumps(structured_local)

        if not raw_local:
            try:
                raw_local = extract_response_json(response_local)
            except RuntimeError as exc:
                LOGGER.debug("extract_response_json failed: %s", exc)
                if hasattr(response_local, "output") and response_local.output:
                    raw_local = json.dumps(response_local.output)
                elif hasattr(response_local, "choices") and response_local.choices:
                    raw_local = json.dumps(
                        [getattr(choice, "message", "") for choice in response_local.choices]
                    )
                else:
                    raw_local = ""

        LOGGER.debug(
            "AI parse raw content (truncated to 400 chars): %s",
            (raw_local or "")[:400],
        )
        return response_local, raw_local.strip()

    response, raw_content = _call_openai(use_schema=True, attempt=1)

    if not raw_content or raw_content.strip('"') == "order_batch":
        LOGGER.debug("AI parse retry without schema due to placeholder output")
        response, raw_content = _call_openai(use_schema=False, attempt=2)

    if not raw_content and hasattr(response, "model_dump"):
        try:
            candidate = _find_text_in_payload(response.model_dump())
            raw_content = candidate.strip()
        except Exception:
            pass
    if not raw_content:
        raise RuntimeError("AI parse returned empty content")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        snippet = raw_content[:200]
        raise RuntimeError(
            f"Model returned invalid JSON: {exc}. Raw: {snippet}"
        ) from exc

    if isinstance(parsed, list):
        orders_payload = parsed
    elif isinstance(parsed, dict):
        orders_payload = parsed.get("orders", [])
    else:
        orders_payload = []

    orders: List[OrderInput] = []
    for item in orders_payload:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "")).strip()
        anchor_keyword = str(item.get("anchor_keyword", "")).strip()
        target_url = str(item.get("target_url", "")).strip()
        anchor_country = str(item.get("anchor_country", "")).strip()
        serp_keyword = str(item.get("serp_keyword", "")).strip()
        if domain and anchor_keyword and target_url:
            orders.append(
                OrderInput(
                    domain=domain,
                    anchor_keyword=anchor_keyword,
                    anchor_country=anchor_country,
                    target_url=target_url,
                    serp_keyword=serp_keyword,
                )
            )
    return orders, raw_content


def suggest_serp_keywords_ai(
    orders: List[Dict[str, str]],
    *,
    client: Optional[Any] = None,
    model: str = AI_PARSE_MODEL,
    max_output_tokens: int = 400,
) -> List[str]:
    """Use GPT-5-nano to suggest thematic SERP keywords for orders."""

    if not orders:
        return []

    if client is None:
        load_env()
        if OpenAI is None:
            raise RuntimeError("openai package is required for AI suggestions but is not installed")
        client = create_openai_client()

    prompt = (
        "You suggest broad, thematic SERP keywords for SEO research. Each suggestion must be a short phrase "
        "(2-4 words) capturing the main topic of the placement site and anchor. Avoid brands or overly specific modifiers. "
        "Return only JSON shaped like {\"suggestions\": [{\"serp_keyword\": str}]}. "
        "Use empty strings if unsure."
    )

    order_lines = []
    for idx, order in enumerate(orders, start=1):
        order_lines.append(
            f"{idx}. domain={order.get('domain','')}, anchor={order.get('anchor_keyword','')}, target={order.get('target_url','')}"
        )
    user_content = "\n".join(order_lines)

    response = invoke_openai_response(
        client,
        model=model,
        system_prompt=prompt,
        user_prompt=user_content,
        max_output_tokens=max_output_tokens,
    )

    raw_json = extract_response_json(response)
    raw_json = raw_json.strip()
    if not raw_json:
        return ["" for _ in orders]

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return ["" for _ in orders]

    suggestions_payload = payload.get("suggestions")
    results: List[str] = []
    if isinstance(suggestions_payload, list):
        for item in suggestions_payload:
            if isinstance(item, dict):
                results.append(str(item.get("serp_keyword", "") or ""))
            elif isinstance(item, str):
                results.append(item)
    if len(results) < len(orders):
        results.extend(["" for _ in range(len(orders) - len(results))])
    return results[: len(orders)]


def extract_response_json(response: Any) -> str:
    """Extract the JSON string from an OpenAI responses API payload."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        if isinstance(output_text, list):
            flattened = '\n'.join(str(item) for item in output_text if item)
            if flattened.strip():
                return flattened.strip()
        elif isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

    sections: List[Any] = []
    if hasattr(response, "output") and response.output is not None:
        sections.extend(response.output)  # type: ignore[arg-type]
    if hasattr(response, "content") and response.content is not None:
        sections.extend(response.content)  # type: ignore[arg-type]

    for item in sections:
        item_dict = item if isinstance(item, dict) else getattr(item, "__dict__", {})
        item_type = item_dict.get("type") or getattr(item, "type", None)
        text_value = item_dict.get("text") or getattr(item, "text", None)
        if text_value and isinstance(text_value, str):
            return text_value.strip()
        parsed_value = item_dict.get("parsed") or getattr(item, "parsed", None)
        if parsed_value:
            try:
                return json.dumps(parsed_value)
            except TypeError:
                return json.dumps(json.loads(str(parsed_value)))
        content = item_dict.get("content") or getattr(item, "content", None)
        if isinstance(content, list):
            for block in content:
                block_dict = block if isinstance(block, dict) else getattr(block, "__dict__", {})
                text_value = block_dict.get("text") or getattr(block, "text", None)
                if text_value and isinstance(text_value, str):
                    return text_value.strip()
                parsed_value = block_dict.get("parsed") or getattr(block, "parsed", None)
                if parsed_value:
                    return json.dumps(parsed_value)
        elif isinstance(content, str) and content.strip():
            return content.strip()

    if hasattr(response, "choices"):
        for choice in response.choices:  # type: ignore[assignment]
            message = getattr(choice, "message", None)
            if message:
                parsed_attr = getattr(message, "parsed", None)
                if parsed_attr:
                    try:
                        return json.dumps(parsed_attr)
                    except TypeError:
                        if isinstance(parsed_attr, str):
                            return parsed_attr.strip()
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    for block in content:
                        block_dict = block if isinstance(block, dict) else getattr(block, "__dict__", {})
                        text_value = block_dict.get("text") or getattr(block, "text", None)
                        if text_value and isinstance(text_value, str):
                            return text_value.strip()
                        parsed_value = block_dict.get("parsed") or getattr(block, "parsed", None)
                        if parsed_value:
                            return json.dumps(parsed_value)
            content = getattr(choice, "text", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            parsed_value = getattr(choice, "parsed", None)
            if parsed_value:
                try:
                    return json.dumps(parsed_value)
                except TypeError:
                    if isinstance(parsed_value, str):
                        return parsed_value.strip()

    data_attr = getattr(response, "data", None)
    if data_attr:
        structured = normalize_structured_value(data_attr)
        if structured is not None:
            if isinstance(structured, str):
                return structured.strip()
            return json.dumps(structured)

    structured = extract_structured_from_response(response)
    if structured is not None:
        if isinstance(structured, str):
            return structured.strip()
        return json.dumps(structured)

    if data_attr is not None:
        try:
            return json.dumps(data_attr)
        except TypeError:
            return str(data_attr)

    if hasattr(response, "model_dump_json"):
        try:
            payload = json.loads(response.model_dump_json())
            extracted = _find_text_in_payload(payload)
            if extracted:
                return extracted
        except Exception:
            pass

    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
            extracted = _find_text_in_payload(payload)
            if extracted:
                return extracted
        except Exception:
            pass

    raise RuntimeError("Unable to extract JSON from OpenAI response payload")



def _find_text_in_payload(node: Any) -> str:
    """Recursively search for a useful text field within an OpenAI response dump."""

    if isinstance(node, str) and node.strip():
        return node.strip()
    if isinstance(node, dict):
        parsed_value = node.get("parsed")
        if parsed_value:
            if isinstance(parsed_value, (dict, list)):
                try:
                    return json.dumps(parsed_value)
                except TypeError:
                    return ""
            if isinstance(parsed_value, str) and parsed_value.strip():
                return parsed_value.strip()
        priority_keys = [
            "output_text",
            "text",
            "raw_output",
            "content",
            "output",
            "data",
        ]
        for key in priority_keys:
            if key in node:
                result = _find_text_in_payload(node[key])
                if result:
                    return result
        for value in node.values():
            result = _find_text_in_payload(value)
            if result:
                return result
    if isinstance(node, (list, tuple)):
        for item in node:
            result = _find_text_in_payload(item)
            if result:
                return result
    return ""

__all__ = [
    "create_openai_client",
    "invoke_openai_response",
    "extract_finish_reason",
    "extract_primary_message_text",
    "AI_PARSE_MODEL",
    "DEFAULT_SERP_RESULT_LIMIT",
    "FetchSettings",
    "OrderInput",
    "SerpFetchResult",
    "ai_parse_orders",
    "apply_heuristic_serp_keywords",
    "cache_serp_result",
    "extract_response_json",
    "extract_structured_from_response",
    "fetch_serp",
    "get_cache_path",
    "heuristic_serp_keyword",
    "list_cached_payloads",
    "load_cached_serp",
    "parse_orders_text",
    "suggest_serp_keywords_ai",
    "sanitize_domain",
    "normalize_structured_value",
    "compute_generation_hash",
    "enable_debug_logging",
    "LOG_PATH",
]
