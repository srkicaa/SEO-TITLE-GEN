"""Streamlit entrypoint for DataForSEO-powered title generation workflow."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from streamlit_app.db import (
    init_db,
    list_generation_domains,
    list_generation_models,
    query_generation_history,
    store_generation,
    store_serp_snapshot,
)
from streamlit_app.openai_writer import (
    DEFAULT_MODEL_ID,
    MODEL_OPTIONS,
    TitleGenerationError,
    generate_titles,
)
from streamlit_app.prompt_manager import PLACEHOLDER_DESCRIPTIONS, PromptTemplate, get_default_templates
from streamlit_app.utils import (
    DEFAULT_SERP_RESULT_LIMIT,
    DATAFORSEO_FETCH_LIMIT,
    FetchSettings,
    OrderInput,
    SerpFetchResult,
    LOG_PATH,
    compute_generation_hash,
    enable_debug_logging,
    ai_parse_orders,
    apply_heuristic_serp_keywords,
    fetch_serp,
    suggest_serp_keywords_ai,
    parse_orders_text,
)

ORDER_COLUMNS = [
    "domain",
    "anchor_keyword",
    "anchor_country",
    "target_url",
    "serp_keyword",
    "extra_context",
]
SERP_RESULTS_KEY = "serp_results"
GENERATION_RESULTS_KEY = "generation_results"
PROMPT_OVERRIDES_KEY = "prompt_overrides"
CUSTOM_PROMPT_KEY = "custom_prompt_text"
PASTE_PREVIEW_KEY = "paste_preview"
PASTE_LOG_KEY = "paste_log"
ORDERS_KEY = "orders"
SELECTED_ORDER_KEY = "selected_order_index"

BATCH_SELECTION_KEY = "batch_selection"
GENERATION_BATCH_SELECTION_KEY = "generation_batch_selection"

class _SafeDict(dict):
    """Dict that preserves unknown placeholders during str.format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"



def main() -> None:
    st.set_page_config(page_title="DataForSEO Title Generator", layout="wide")
    st.title("Link-Building Title Generator")
    st.caption(
        "Queue link building orders, pull SERP titles from DataForSEO, and craft optimized link placement titles via OpenAI."
    )

    templates = get_default_templates()
    template_lookup = {template.key: template for template in templates}

    _init_session_state(template_lookup)
    init_db()

    sidebar_state = _render_sidebar_controls(template_lookup)

    orders_tab, serp_tab, generation_tab, history_tab = st.tabs([
        "1. Orders",
        "2. SERP Review",
        "3. Generate Titles",
        "4. History",
    ])

    with orders_tab:
        _render_orders_tab(template_lookup, sidebar_state)
    with serp_tab:
        _render_serp_tab(sidebar_state)
    with generation_tab:
        _render_generation_tab(template_lookup, sidebar_state)
    with history_tab:
        _render_history_tab()


def _init_session_state(template_lookup: Dict[str, PromptTemplate]) -> None:
    if ORDERS_KEY not in st.session_state:
        st.session_state[ORDERS_KEY] = []
    if SERP_RESULTS_KEY not in st.session_state:
        st.session_state[SERP_RESULTS_KEY] = {}
    if GENERATION_RESULTS_KEY not in st.session_state:
        st.session_state[GENERATION_RESULTS_KEY] = {}
    if PROMPT_OVERRIDES_KEY not in st.session_state:
        st.session_state[PROMPT_OVERRIDES_KEY] = {
            key: template.template for key, template in template_lookup.items()
        }
    if CUSTOM_PROMPT_KEY not in st.session_state:
        st.session_state[CUSTOM_PROMPT_KEY] = ""
    if PASTE_PREVIEW_KEY not in st.session_state:
        st.session_state[PASTE_PREVIEW_KEY] = []
    if PASTE_LOG_KEY not in st.session_state:
        st.session_state[PASTE_LOG_KEY] = None
    if SELECTED_ORDER_KEY not in st.session_state:
        st.session_state[SELECTED_ORDER_KEY] = 0
    if BATCH_SELECTION_KEY not in st.session_state:
        st.session_state[BATCH_SELECTION_KEY] = []
    if GENERATION_BATCH_SELECTION_KEY not in st.session_state:
        st.session_state[GENERATION_BATCH_SELECTION_KEY] = []


def _render_sidebar_controls(template_lookup: Dict[str, PromptTemplate]) -> Dict[str, Any]:
    st.sidebar.header("Settings")

    default_limit = st.sidebar.slider(
        "SERP titles to fetch",
        min_value=5,
        max_value=100,
        value=st.session_state.get("serp_limit", DEFAULT_SERP_RESULT_LIMIT),
        step=5,
    )
    st.session_state["serp_limit"] = default_limit

    default_keyword = st.sidebar.text_input(
        "Fallback SERP keyword",
        value=st.session_state.get("default_serp_keyword", ""),
        placeholder="e.g. best online casino bonuses",
    )
    st.session_state["default_serp_keyword"] = default_keyword

    use_cache = st.sidebar.checkbox(
        "Prefer cached SERP payloads",
        value=st.session_state.get("serp_use_cache", False),
    )
    st.session_state["serp_use_cache"] = use_cache

    cache_only = st.sidebar.checkbox(
        "Offline mode (cache only)",
        value=st.session_state.get("serp_cache_only", False),
        help="When enabled, the app only reads cached payloads and never calls the API.",
    )
    st.session_state["serp_cache_only"] = cache_only

    dfs_mode_options = ["Live (immediate)", "Standard queue"]
    default_mode_index = 0 if st.session_state.get("dfs_live", True) else 1
    dfs_mode = st.sidebar.radio(
        "DataForSEO mode",
        options=dfs_mode_options,
        index=default_mode_index,
        help="Choose between live or queued SERP endpoints.",
    )
    use_live_endpoint = dfs_mode == "Live (immediate)"
    st.session_state["dfs_live"] = use_live_endpoint

    st.sidebar.markdown("---")
    model_labels = {option["id"]: option["label"] for option in MODEL_OPTIONS}
    model_ids = list(model_labels.keys())
    default_model = st.session_state.get("model_id", DEFAULT_MODEL_ID)
    if default_model not in model_ids:
        default_model = DEFAULT_MODEL_ID
    chosen_index = model_ids.index(default_model)
    model_id = st.sidebar.selectbox(
        "OpenAI model",
        options=model_ids,
        index=chosen_index,
        format_func=lambda model: model_labels.get(model, model),
    )
    st.session_state["model_id"] = model_id

    titles_requested = st.sidebar.slider(
        "Titles to request per order",
        min_value=1,
        max_value=10,
        value=st.session_state.get("num_titles_requested", 5),
    )
    st.session_state["num_titles_requested"] = titles_requested

    st.sidebar.markdown("---")
    keep_orders_on_append = st.sidebar.checkbox(
        "Keep existing orders when appending preview",
        value=False,
        help="When enabled, new paste uploads append to current orders. Otherwise they replace the list.",
    )
    st.sidebar.caption("Prompts can reference these fields: " + ", ".join(PLACEHOLDER_DESCRIPTIONS))

    debug_logging = st.sidebar.checkbox(
        "Enable debug logging",
        value=st.session_state.get("debug_logging", False),
        help="Writes detailed AI parse/generation events to assets/logs/app.log",
    )
    if debug_logging != st.session_state.get("debug_logging", False):
        enable_debug_logging(debug_logging)
        st.session_state["debug_logging"] = debug_logging

    return {
        "limit": default_limit,
        "default_serp_keyword": default_keyword,
        "use_cache": use_cache,
        "cache_only": cache_only,
        "use_live_endpoint": use_live_endpoint,
        "model_id": model_id,
        "titles_requested": titles_requested,
        "template_lookup": template_lookup,
        "num_titles_requested": titles_requested,
        "keep_orders_on_append": keep_orders_on_append,
    }


def _render_orders_tab(template_lookup: Dict[str, PromptTemplate], sidebar_state: Dict[str, Any]) -> None:
    st.subheader("Manage orders")

    _render_pasteingress(sidebar_state)
    _render_csv_upload()

    st.markdown("### Current orders")
    orders_df = _get_orders_dataframe()
    edited_df = st.data_editor(
        orders_df,
        num_rows="dynamic",
        width="stretch",
        key="orders_editor",
    )

    st.session_state[ORDERS_KEY] = _coerce_orders_from_df(edited_df)
    current_orders = st.session_state[ORDERS_KEY]
    generation_results = st.session_state.get(GENERATION_RESULTS_KEY, {})
    if generation_results:
        pruned = {
            idx: result
            for idx, result in generation_results.items()
            if idx < len(current_orders)
        }
        if len(pruned) != len(generation_results):
            st.session_state[GENERATION_RESULTS_KEY] = pruned

    if st.session_state[ORDERS_KEY]:
        st.success(f"{len(st.session_state[ORDERS_KEY])} orders ready for processing.")
    else:
        st.info("Add orders via the table, CSV upload, or paste helper above.")


def _render_pasteingress(sidebar_state: Dict[str, Any]) -> None:
    with st.expander("Paste orders from Sheets", expanded=False):
        with st.form("paste_form"):
            paste_text = st.text_area(
                "Paste rows with Domain, Anchor, Target URL, optional SERP keyword",
                key="paste_text_area",
                height=160,
            )
            col_parse, col_ai = st.columns(2)
            quick_parse = col_parse.form_submit_button("Quick parse")
            ai_parse = col_ai.form_submit_button("AI parse (GPT-5 nano)")

        if quick_parse:
            orders, errors, needs_ai = parse_orders_text(paste_text)
            st.session_state[PASTE_PREVIEW_KEY] = [order.to_dict() for order in orders]
            feedback = _format_parse_feedback(errors, needs_ai)
            st.session_state[PASTE_LOG_KEY] = {
                "variant": "info" if feedback else "success",
                "message": feedback or f"Parsed {len(orders)} order(s) with quick parser.",
                "detail": None,
            }
        elif ai_parse:
            try:
                orders, raw_json = ai_parse_orders(paste_text)
                st.session_state[PASTE_PREVIEW_KEY] = [order.to_dict() for order in orders]
                st.session_state[PASTE_LOG_KEY] = {
                    "variant": "success",
                    "message": f"AI parsed {len(orders)} order(s).",
                    "detail": raw_json,
                }
            except Exception as exc:  # broad for runtime dependencies
                st.session_state[PASTE_PREVIEW_KEY] = []
                st.session_state[PASTE_LOG_KEY] = {
                    "variant": "error",
                    "message": f"AI parse failed: {exc}",
                    "detail": None,
                }

        preview = st.session_state.get(PASTE_PREVIEW_KEY, [])
        if preview:
            st.markdown("#### Preview")
            st.dataframe(pd.DataFrame(preview), width="stretch")

            sugg_col1, sugg_col2, sugg_col3 = st.columns(3)
            if sugg_col1.button("Heuristic SERP keywords", key="serp_heuristic_btn"):
                apply_heuristic_serp_keywords(preview)
                st.session_state[PASTE_PREVIEW_KEY] = [dict(order) for order in preview]
                st.success("Applied heuristic SERP keywords.")
            if sugg_col2.button("AI SERP keywords (GPT-5 nano)", key="serp_ai_btn"):
                try:
                    suggestions = suggest_serp_keywords_ai(preview)
                    for order, suggestion in zip(preview, suggestions):
                        order["serp_keyword"] = suggestion
                    st.session_state[PASTE_PREVIEW_KEY] = [dict(order) for order in preview]
                    st.success("Applied AI SERP keyword suggestions.")
                except Exception as exc:
                    st.error(f"AI SERP suggestion failed: {exc}")
            if sugg_col3.button("Clear SERP keywords", key="serp_clear_btn"):
                for order in preview:
                    order["serp_keyword"] = ""
                st.session_state[PASTE_PREVIEW_KEY] = [dict(order) for order in preview]
                st.info("Cleared SERP keywords in preview.")

            if st.button("Append preview to orders"):
                new_orders = [dict(order) for order in preview]
                if sidebar_state.get("keep_orders_on_append"):
                    st.session_state[ORDERS_KEY].extend(new_orders)
                else:
                    st.session_state[ORDERS_KEY] = new_orders
                    st.session_state[SERP_RESULTS_KEY] = {}
                    st.session_state[GENERATION_RESULTS_KEY] = {}
                st.session_state[PASTE_PREVIEW_KEY] = []
                st.session_state[PASTE_LOG_KEY] = None
        log = st.session_state.get(PASTE_LOG_KEY)
        if log:
            variant = log.get("variant") if isinstance(log, dict) else "info"
            message = log.get("message") if isinstance(log, dict) else str(log)
            detail = log.get("detail") if isinstance(log, dict) else None
            if variant == "success":
                st.success(message)
            elif variant == "warning":
                st.warning(message)
            elif variant == "error":
                st.error(message)
            else:
                st.info(message)
            if detail:
                with st.expander("Show parsed JSON", expanded=False):
                    st.code(detail)


def _render_csv_upload() -> None:
    with st.expander("Upload CSV/TSV", expanded=False):
        uploaded = st.file_uploader("Upload orders", type=["csv", "tsv"], accept_multiple_files=False)
        if uploaded is not None:
            raw_text = uploaded.getvalue().decode("utf-8", errors="ignore")
            orders, errors, _ = parse_orders_text(raw_text)
            if orders:
                st.session_state[ORDERS_KEY].extend([order.to_dict() for order in orders])  # type: ignore[arg-type]
                st.success(f"Loaded {len(orders)} orders from file.")
            if errors:
                st.warning("\n".join(errors))


def _get_orders_dataframe() -> pd.DataFrame:
    orders = st.session_state.get(ORDERS_KEY, [])
    if not orders:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    return pd.DataFrame(orders, columns=ORDER_COLUMNS)


def _coerce_orders_from_df(df: pd.DataFrame) -> List[Dict[str, str]]:
    coerced: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        record = {col: _serialize_cell(row.get(col, "")) for col in ORDER_COLUMNS}
        if any(record.values()):
            coerced.append(record)
    return coerced


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _format_parse_feedback(errors: List[str], needs_ai: bool) -> str:
    lines: List[str] = []
    if errors:
        lines.append("\n".join(errors))
    if needs_ai:
        lines.append("Structure unclear. Try the AI parse option.")
    return "\n".join(lines)


def _render_serp_tab(sidebar_state: Dict[str, Any]) -> None:
    st.subheader("Fetch SERP titles")

    orders = st.session_state.get(ORDERS_KEY, [])
    if not orders:
        st.info("Add orders first to fetch SERP data.")
        return

    options = [f"#{idx + 1}: {order.get('domain', '')} → {order.get('target_url', '')}" for idx, order in enumerate(orders)]
    index_options = list(range(len(options)))

    selected_index = st.selectbox(
        "Select order",
        options=index_options,
        format_func=lambda idx: options[idx],
        index=min(st.session_state.get(SELECTED_ORDER_KEY, 0), len(options) - 1),
    )
    st.session_state[SELECTED_ORDER_KEY] = selected_index

    default_batch = st.session_state.get(BATCH_SELECTION_KEY, index_options)
    default_batch = [idx for idx in default_batch if idx in index_options]
    if not default_batch:
        default_batch = [selected_index]
    batch_selection = st.multiselect(
        "Orders for batch fetch",
        options=index_options,
        format_func=lambda idx: options[idx],
        default=default_batch,
        key="serp_batch_selection",
    )
    st.session_state[BATCH_SELECTION_KEY] = batch_selection

    current_order = orders[selected_index]
    serp_keyword = current_order.get("serp_keyword") or sidebar_state["default_serp_keyword"]
    st.write(f"Using SERP keyword: `{serp_keyword}`")

    batch_col1, batch_col2 = st.columns(2)
    fetch_selected_batch = batch_col1.button("Fetch selected orders", key="fetch_selected_batch")
    fetch_all_batch = batch_col2.button("Fetch all orders", key="fetch_all_batch")

    fetch_col1, fetch_col2, fetch_col3 = st.columns(3)
    fetch_button = fetch_col1.button("Fetch titles", key="fetch_single", width="stretch")
    refresh_cache = fetch_col2.button("Reload from cache", key="reload_cache", width="stretch")
    clear_selection = fetch_col3.button("Reset selection", key="reset_serp_selection", width="stretch")

    limit = sidebar_state["limit"]
    if fetch_button or refresh_cache:
        with st.spinner("Fetching SERP titles..."):
            try:
                cache_only = refresh_cache or sidebar_state.get("cache_only", False)
                use_cache = sidebar_state.get("use_cache", False) if not refresh_cache else True
                result = _perform_fetch(
                    selected_index,
                    current_order,
                    sidebar_state,
                    limit,
                    use_cache=use_cache,
                    cache_only=cache_only,
                )
                st.success(f"Loaded {len(result.titles)} titles ({result.source}).")
                st.caption(f"Query: {result.query}")
            except Exception as exc:
                st.error(f"SERP fetch failed: {exc}")

    if fetch_selected_batch or fetch_all_batch:
        targets = batch_selection if fetch_selected_batch else index_options
        if not targets:
            st.warning("Select at least one order for batch fetch.")
        else:
            errors: List[str] = []
            successes = 0
            progress = st.progress(0.0)
            with st.spinner("Fetching SERP titles in batch..."):
                total = len(targets)
                for position, idx in enumerate(targets, start=1):
                    order_data = orders[idx]
                    try:
                        _perform_fetch(
                            idx,
                            order_data,
                            sidebar_state,
                            limit,
                            use_cache=sidebar_state.get("use_cache", False),
                            cache_only=sidebar_state.get("cache_only", False),
                        )
                        successes += 1
                    except Exception as exc:
                        errors.append(f"#{idx + 1} {order_data.get('domain', '')}: {exc}")
                    progress.progress(position / total)
            progress.empty()
            if successes:
                st.success(f"Fetched titles for {successes} order(s).")
            if errors:
                st.error("Batch fetch issues:\n" + "\n".join(errors))

    if clear_selection and selected_index in st.session_state[SERP_RESULTS_KEY]:
        del st.session_state[SERP_RESULTS_KEY][selected_index]

    result = st.session_state[SERP_RESULTS_KEY].get(selected_index)
    if not result:
        st.info("Fetch titles to review them here.")
        return

    titles = result["titles"]
    if not titles:
        st.warning("No titles returned.")
        return

    st.markdown("### Titles (unaltered)")
    df = pd.DataFrame(titles)
    st.dataframe(df, width="stretch")

    default_selection_count = min(sidebar_state["titles_requested"] * 5, len(titles))
    slider_key = f"serp_slider_{selected_index}"
    use_count = st.slider(
        "Titles to send to prompt",
        min_value=1,
        max_value=len(titles),
        value=min(result.get("selection_count", default_selection_count), len(titles)),
        key=slider_key,
    )
    selected_indices = list(range(use_count))

    st.session_state[SERP_RESULTS_KEY][selected_index]["selection_count"] = use_count
    st.session_state[SERP_RESULTS_KEY][selected_index]["selected_titles"] = selected_indices

    if st.checkbox("Show raw JSON", key=f"serp_raw_json_{selected_index}"):
        st.code(json.dumps(result["raw"], indent=2))


def _store_serp_result(index: int, result: SerpFetchResult) -> None:
    st.session_state[SERP_RESULTS_KEY][index] = {
        "titles": result.titles,
        "raw": result.raw,
        "query": result.query,
        "source": result.source,
        "selection_count": min(len(result.titles), st.session_state.get("num_titles_requested", 3) * 5),
        "selected_titles": list(range(min(len(result.titles), st.session_state.get("num_titles_requested", 3) * 5))),
    }


def _perform_fetch(
    index: int,
    order_data: Dict[str, str],
    sidebar_state: Dict[str, Any],
    limit: int,
    *,
    use_cache: bool,
    cache_only: bool,
) -> SerpFetchResult:
    serp_keyword = order_data.get("serp_keyword") or sidebar_state.get("default_serp_keyword", "")
    if not serp_keyword:
        raise ValueError(f"Order #{index + 1} missing SERP keyword and no fallback provided.")

    order_obj = OrderInput(
        domain=order_data.get("domain", ""),
        anchor_keyword=order_data.get("anchor_keyword", ""),
        target_url=order_data.get("target_url", ""),
        serp_keyword=serp_keyword,
    )

    settings = FetchSettings(
        default_serp_keyword=sidebar_state.get("default_serp_keyword", ""),
        live=sidebar_state.get("use_live_endpoint", True),
    )

    result = fetch_serp(
        order_obj,
        settings=settings,
        limit=DATAFORSEO_FETCH_LIMIT,
        use_cache=use_cache,
        cache_only=cache_only,
    )
    _store_serp_result(index, result)
    try:
        query_hash = compute_generation_hash(
            order_data,
            serp_keyword=serp_keyword,
            model=f"serp-limit-{limit}",
            prompt="serp_snapshot",
        )
        store_serp_snapshot(
            query_hash=query_hash,
            order_data=order_data,
            serp_keyword=serp_keyword,
            limit=DATAFORSEO_FETCH_LIMIT,
            result=result,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to persist SERP snapshot for %s: %s",
            order_data.get("domain", ""),
            exc,
        )
    return result


def _ensure_serp_for_order(
    index: int,
    order_data: Dict[str, str],
    sidebar_state: Dict[str, Any],
    limit: int,
) -> Dict[str, Any]:
    serp_results = st.session_state.get(SERP_RESULTS_KEY, {})
    if index in serp_results:
        return serp_results[index]

    _perform_fetch(
        index,
        order_data,
        sidebar_state,
        limit,
        use_cache=sidebar_state.get("use_cache", False),
        cache_only=sidebar_state.get("cache_only", False),
    )
    return st.session_state[SERP_RESULTS_KEY][index]


def _get_titles_for_generation(serp_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    titles = serp_entry.get("titles", [])
    selected_indices = serp_entry.get("selected_titles")
    if selected_indices:
        return [titles[i] for i in selected_indices if 0 <= i < len(titles)]
    selection_count = serp_entry.get("selection_count", len(titles))
    return titles[:selection_count]


def _prepare_export_dataframe(orders: List[Dict[str, str]], generation_results: Dict[int, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    serp_results = st.session_state.get(SERP_RESULTS_KEY, {})
    for index, generation in generation_results.items():
        if index >= len(orders):
            continue
        order = orders[index]
        serp_entry = serp_results.get(index, {})
        query = serp_entry.get("query", "")
        for item in generation.titles:
            rows.append(
                {
                    "order_index": index + 1,
                    "domain": order.get("domain", ""),
                    "anchor_keyword": order.get("anchor_keyword", ""),
                    "anchor_country": order.get("anchor_country", ""),
                    "target_url": order.get("target_url", ""),
                    "serp_keyword": order.get("serp_keyword", ""),
                    "generated_title": item.get("title"),
                    "rationale": item.get("rationale"),
                    "model": generation.model,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "estimated_cost_usd": generation.cost_usd,
                    "serp_query": query,
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_generation_tab(template_lookup: Dict[str, PromptTemplate], sidebar_state: Dict[str, Any]) -> None:
    st.subheader("Generate titles with OpenAI")

    orders = st.session_state.get(ORDERS_KEY, [])
    if not orders:
        st.info("Add orders first to enable generation.")
        return

    options = [f"#{idx + 1}: {order.get('domain', '')}" for idx, order in enumerate(orders)]
    index_options = list(range(len(options)))

    if not index_options:
        st.info("Add orders first to enable generation.")
        return

    default_batch = st.session_state.get(GENERATION_BATCH_SELECTION_KEY, index_options)
    default_batch = [idx for idx in default_batch if idx in index_options]
    if not default_batch:
        default_batch = index_options
    generation_selection = st.multiselect(
        "Orders for batch generation",
        options=index_options,
        format_func=lambda idx: options[idx],
        default=default_batch,
        key="generation_batch_selection",
    )
    if GENERATION_BATCH_SELECTION_KEY not in st.session_state:
        st.session_state[GENERATION_BATCH_SELECTION_KEY] = generation_selection

    selected_index = st.selectbox(
        "Select order",
        options=index_options,
        format_func=lambda idx: options[idx],
        index=min(st.session_state.get(SELECTED_ORDER_KEY, 0), len(options) - 1),
        key="generation_order_selector",
    )
    st.session_state[SELECTED_ORDER_KEY] = selected_index

    batch_generate_cols = st.columns(2)
    generate_selected_batch = batch_generate_cols[0].button(
        "Generate batch (from list above)", key="generate_selected_batch"
    )
    generate_all_batch = batch_generate_cols[1].button(
        "Generate batch (all orders)", key="generate_all_batch"
    )
    st.caption(
        "Use the multiselect to curate a batch. "
        "`Generate batch (from list above)` runs only those entries, while `Generate batch (all orders)` ignores the multiselect and processes every order. "
        "The single `Generate titles` button further down only affects the currently selected order with the prompt shown."
    )

    try:
        result_entry = _ensure_serp_for_order(selected_index, orders[selected_index], sidebar_state, sidebar_state["limit"])
    except Exception as exc:
        st.error(f"Unable to fetch SERP titles for order #{selected_index + 1}: {exc}")
        return

    order = orders[selected_index]
    available_titles = result_entry.get("titles", [])

    st.markdown("### Titles (unaltered)")
    if available_titles:
        st.dataframe(pd.DataFrame(available_titles), width="stretch")
    else:
        st.warning("No titles returned.")

    default_selection_count = min(sidebar_state.get("titles_requested", 3) * 5, len(available_titles)) if available_titles else 0
    if available_titles:
        slider_key = f"generation_slider_{selected_index}"
        use_count = st.slider(
            "Titles to send to prompt",
            min_value=1,
            max_value=len(available_titles),
            value=min(result_entry.get("selection_count", default_selection_count), len(available_titles)),
            key=slider_key,
        )
    else:
        use_count = 0

    selected_indices = list(range(use_count)) if use_count else []
    st.session_state[SERP_RESULTS_KEY][selected_index]["selection_count"] = use_count or result_entry.get("selection_count", 0)
    st.session_state[SERP_RESULTS_KEY][selected_index]["selected_titles"] = selected_indices

    titles = [available_titles[i] for i in selected_indices if 0 <= i < len(available_titles)] if selected_indices else available_titles[: result_entry.get("selection_count", 0)]

    if st.checkbox("Show raw JSON", key=f"generation_raw_json_{selected_index}") and result_entry.get("raw") is not None:
        st.code(json.dumps(result_entry["raw"], indent=2))

    if not titles:
        st.warning("No SERP titles available for this order. Adjust the selection or refetch.")
        return

    template_options = list(template_lookup.keys()) + ["custom"]
    format_mapping = {**{key: template_lookup[key].name for key in template_lookup}, "custom": "Custom"}

    default_choice = st.session_state.get("prompt_choice", template_options[0])
    if default_choice not in template_options:
        default_choice = template_options[0]
    chosen_template_key = st.selectbox(
        "Prompt preset",
        options=template_options,
        format_func=lambda key: format_mapping.get(key, key),
        index=template_options.index(default_choice),
        key="prompt_preset_selector",
    )
    st.session_state["prompt_choice"] = chosen_template_key

    if chosen_template_key == "custom":
        prompt_text = st.text_area(
            "Custom prompt",
            value=st.session_state.get(CUSTOM_PROMPT_KEY, ""),
            height=320,
        )
        st.session_state[CUSTOM_PROMPT_KEY] = prompt_text
    else:
        prompt_text = st.text_area(
            "Edit prompt",
            value=st.session_state[PROMPT_OVERRIDES_KEY].get(
                chosen_template_key, template_lookup[chosen_template_key].template
            ),
            height=320,
            key=f"prompt_editor_{chosen_template_key}",
        )
        st.session_state[PROMPT_OVERRIDES_KEY][chosen_template_key] = prompt_text

    st.markdown("#### Placeholder preview")
    context = _build_prompt_context(order, titles, sidebar_state)
    rendered_preview = prompt_text.format_map(_SafeDict(context))
    st.text_area("Rendered prompt", value=rendered_preview, height=240, key="rendered_preview_area", disabled=True)

    if generate_selected_batch or generate_all_batch:
        targets = generation_selection if generate_selected_batch else index_options
        if not targets:
            st.warning("Select at least one order for batch generation.")
        else:
            errors: List[str] = []
            successes = 0
            progress = st.progress(0.0)
            with st.spinner("Generating titles in batch..."):
                total = len(targets)
                for position, idx in enumerate(targets, start=1):
                    order_data = orders[idx]
                    try:
                        serp_entry = _ensure_serp_for_order(idx, order_data, sidebar_state, sidebar_state["limit"])
                        titles_for_order = _get_titles_for_generation(serp_entry)
                        if not titles_for_order:
                            raise ValueError("No SERP titles available for this order")
                        context_for_order = _build_prompt_context(order_data, titles_for_order, sidebar_state)
                        rendered_prompt = prompt_text.format_map(_SafeDict(context_for_order))
                        generation = generate_titles(
                            titles_context=[title.get("title", "") for title in titles_for_order],
                            order_payload=order_data,
                            prompt_text=rendered_prompt,
                            model=sidebar_state["model_id"],
                            expected_titles=sidebar_state["num_titles_requested"],
                        )
                        st.session_state[GENERATION_RESULTS_KEY][idx] = generation
                        _persist_generation(
                            dict(order_data),
                            context_for_order.get("serp_keyword", ""),
                            prompt_text,
                            rendered_prompt,
                            generation,
                        )
                        successes += 1
                    except Exception as exc:
                        errors.append(f"#{idx + 1} {order_data.get('domain', '')}: {exc}")
                    progress.progress(position / total)
            progress.empty()
            if successes:
                st.success(f"Generated titles for {successes} order(s).")
            if errors:
                st.error("Batch generation issues:\n" + "\n".join(errors))

    generate_button = st.button("Generate titles", type="primary")

    if generate_button:
        try:
            generation = generate_titles(
                titles_context=[title.get("title", "") for title in titles],
                order_payload=order,
                prompt_text=rendered_preview,
                model=sidebar_state["model_id"],
                expected_titles=sidebar_state["num_titles_requested"],
            )
            st.session_state[GENERATION_RESULTS_KEY][selected_index] = generation
            st.success(
                f"Generated {len(generation.titles)} options using {generation.model} (cost ≈ ${generation.cost_usd})."
            )
            _persist_generation(
                dict(order),
                context.get("serp_keyword", ""),
                prompt_text,
                rendered_preview,
                generation,
            )
        except TitleGenerationError as exc:
            st.error(f"Generation failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error calling OpenAI: {exc}")

    existing = st.session_state.get(GENERATION_RESULTS_KEY, {}).get(selected_index)
    if existing:
        st.markdown("### Generated titles")
        for idx, item in enumerate(existing.titles, start=1):
            title = item.get("title")
            rationale = item.get("rationale")
            st.markdown(f"**{idx}. {title}**\n\n{rationale}")
        meta_lines = [
            f"Model: {existing.model}",
            f"Prompt tokens: {existing.prompt_tokens}",
            f"Completion tokens: {existing.completion_tokens}",
            f"Estimated cost: ${existing.cost_usd}",
        ]
        st.caption(" | ".join(meta_lines))
        if st.checkbox("Show raw response", key=f"raw_generation_{selected_index}"):
            raw_payload: Dict[str, Any] = {}
            if existing.raw_response is not None:
                raw = existing.raw_response
                if hasattr(raw, "to_dict"):
                    raw_payload = raw.to_dict()
                elif hasattr(raw, "model_dump"):
                    raw_payload = raw.model_dump()
                elif hasattr(raw, "model_dump_json"):
                    raw_payload = json.loads(raw.model_dump_json())
                else:
                    raw_payload = json.loads(json.dumps(raw, default=str))
            st.code(json.dumps(raw_payload, indent=2))

    generation_results = st.session_state.get(GENERATION_RESULTS_KEY, {})
    if generation_results:
        export_df = _prepare_export_dataframe(orders, generation_results)
        if not export_df.empty:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")
            export_col1, export_col2 = st.columns(2)
            export_col1.download_button("Download CSV", csv_bytes, file_name="title_generation.csv", mime="text/csv")
            export_col2.download_button("Download JSON", json_bytes, file_name="title_generation.json", mime="application/json")
            st.markdown("#### Batch results table")
            table_view = export_df[
                [
                    "order_index",
                    "domain",
                    "anchor_country",
                    "generated_title",
                    "rationale",
                    "model",
                ]
            ]
            st.dataframe(table_view, width="stretch")

            st.markdown("#### Browse results by order")
            grouped = export_df.groupby("order_index", sort=True)
            for order_index, group in grouped:
                domain_name = group["domain"].iloc[0] if not group.empty else ""
                header = f"Order #{order_index}: {domain_name}" if domain_name else f"Order #{order_index}"
                with st.expander(header, expanded=False):
                    for _, row in group.iterrows():
                        st.markdown(f"**{row['generated_title']}**\n\n{row['rationale']}")


def _persist_generation(
    order: Dict[str, Any],
    serp_keyword: str,
    prompt_text: str,
    rendered_prompt: str,
    generation: Any,
) -> None:
    logger = logging.getLogger(__name__)
    try:
        generation_record = {
            "model": generation.model,
            "prompt": prompt_text,
            "rendered_prompt": rendered_prompt,
            "output_json": json.dumps(generation.titles, ensure_ascii=False),
            "cost_usd": generation.cost_usd,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
        }
        query_hash = compute_generation_hash(
            order,
            serp_keyword=serp_keyword,
            model=generation.model,
            prompt=prompt_text,
        )
        logger.debug(
            "Persisting generation: domain=%s target=%s model=%s hash=%s titles=%d",
            order.get("domain", ""),
            order.get("target_url", ""),
            generation.model,
            query_hash,
            len(generation.titles),
        )
        store_generation(
            query_hash=query_hash,
            order_data=order,
            serp_keyword=serp_keyword,
            generation=generation_record,
        )
    except Exception as exc:
        logger.exception("Failed to persist generation for %s", order.get("domain", ""))
        st.warning(f"Failed to record history for {order.get('domain', '')}: {exc}", icon="⚠️")


def _render_history_tab() -> None:
    st.subheader("Generation history")

    domains = list_generation_domains()
    models = list_generation_models()

    if st.session_state.get("debug_logging"):
        if LOG_PATH.exists():
            with st.expander("Debug log", expanded=False):
                try:
                    log_text = LOG_PATH.read_text(encoding="utf-8")
                except Exception as exc:
                    st.warning(f"Unable to read log file: {exc}")
                else:
                    tail_lines = "\n".join(log_text.splitlines()[-400:])
                    st.code(tail_lines or "(log is empty)")
                    st.download_button(
                        "Download full log",
                        data=log_text.encode("utf-8"),
                        file_name="app.log",
                        mime="text/plain",
                    )
        else:
            st.info("Debug logging is enabled but no entries have been written yet.")

    filter_col1, filter_col2 = st.columns(2)
    domain_filter = filter_col1.multiselect("Filter by domain", options=domains)
    model_filter = filter_col2.multiselect("Filter by model", options=models)

    search_text = st.text_input("Search titles, rationales, or prompts", placeholder="e.g. bonus, sweepstakes")

    use_date_filter = st.checkbox("Filter by date range", value=False)
    start_iso = end_iso = None
    if use_date_filter:
        date_col1, date_col2 = st.columns(2)
        start_date = date_col1.date_input("Start date")
        end_date = date_col2.date_input("End date")
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()

    limit = st.slider("Max records to load", min_value=25, max_value=500, value=200, step=25)

    records = query_generation_history(
        domains=domain_filter or None,
        models=model_filter or None,
        search=search_text.strip() or None,
        start_date=start_iso,
        end_date=end_iso,
        limit=limit,
    )

    if not records:
        st.info("No generation history stored yet.")
        return

    rows: List[Dict[str, Any]] = []
    detail_blocks: List[Dict[str, Any]] = []
    search_lower = search_text.strip().lower() if search_text else ""

    for record in records:
        outputs = []
        try:
            outputs = json.loads(record.get("output_json") or "[]")
        except Exception:
            outputs = []

        if search_lower:
            base_text = " ".join(
                filter(
                    None,
                    [
                        record.get("order_domain", ""),
                        record.get("anchor_keyword", ""),
                        record.get("anchor_country", ""),
                        record.get("serp_keyword", ""),
                        record.get("target_url", ""),
                        record.get("rendered_prompt", ""),
                        record.get("prompt", ""),
                    ],
                )
            ).lower()
            output_match = any(
                search_lower in (item.get("title", "") + " " + item.get("rationale", "")).lower()
                for item in outputs
            )
            if search_lower not in base_text and not output_match:
                continue

        for idx, item in enumerate(outputs, start=1):
            rows.append(
                {
                    "Created": record.get("created_at"),
                    "Domain": record.get("order_domain"),
                    "Anchor": record.get("anchor_keyword"),
                    "Country": record.get("anchor_country", ""),
                    "SERP keyword": record.get("serp_keyword"),
                    "Title #": idx,
                    "Title": item.get("title", ""),
                    "Rationale": item.get("rationale", ""),
                    "Model": record.get("model"),
                    "Target URL": record.get("target_url"),
                    "Cost (USD)": record.get("cost_usd"),
                }
            )
        detail_blocks.append({"record": record, "outputs": outputs})

    if not rows:
        st.info("No results match the current filters.")
        return

    history_df = pd.DataFrame(rows)
    st.dataframe(history_df, use_container_width=True)
    csv_bytes = history_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history CSV", csv_bytes, file_name="generation_history.csv", mime="text/csv")

    st.markdown("#### Detailed records")
    for entry in detail_blocks:
        record = entry["record"]
        outputs = entry["outputs"]
        header = f"{record.get('created_at', '')} — {record.get('order_domain', '')} ({record.get('model', '')})"
        with st.expander(header, expanded=False):
            st.markdown(
                " | ".join(
                    filter(
                        None,
                        [
                            f"Anchor keyword: {record.get('anchor_keyword', '')}",
                            f"Anchor country: {record.get('anchor_country', '')}",
                            f"SERP keyword: {record.get('serp_keyword', '')}",
                            f"Cost: ${record.get('cost_usd', 0):.4f}",
                            f"Tokens: {record.get('prompt_tokens', 0)}→{record.get('completion_tokens', 0)}",
                        ],
                    )
                )
            )
            st.markdown(f"**Target URL:** {record.get('target_url', '')}")
            st.markdown("**Rendered prompt**")
            st.code(record.get("rendered_prompt", ""), language="")
            if record.get("prompt") and record.get("prompt") != record.get("rendered_prompt"):
                st.markdown("**Source prompt template/custom text**")
                st.code(record.get("prompt", ""), language="")
            st.markdown("**Titles**")
            for idx, item in enumerate(outputs, start=1):
                st.markdown(f"**{idx}. {item.get('title', '')}**\n\n{item.get('rationale', '')}")


def _build_prompt_context(order: Dict[str, str], titles: List[Dict[str, Any]], sidebar_state: Dict[str, Any]) -> Dict[str, str]:
    bullet_titles = "\n".join(f"- {item.get('title', '')}" for item in titles)
    additional_context = (order.get("extra_context") or "").strip()
    anchor_country_raw = str(order.get("anchor_country", "") or "").strip()
    anchor_country = anchor_country_raw if anchor_country_raw else "Unspecified"

    return {
        "domain": order.get("domain", ""),
        "target_url": order.get("target_url", ""),
        "anchor_keyword": order.get("anchor_keyword", ""),
        "anchor_country": anchor_country,
        "serp_keyword": order.get("serp_keyword") or sidebar_state["default_serp_keyword"],
        "num_requested": str(sidebar_state["num_titles_requested"]),
        "serp_titles": bullet_titles,
        "additional_context": additional_context,
    }


if __name__ == "__main__":
    main()
