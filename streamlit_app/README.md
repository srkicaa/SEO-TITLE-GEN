# Streamlit App Overview

This package (`streamlit_app/`) delivers the UI workflow for managing link-building orders, fetching SERP titles from DataForSEO, and generating editorially-aligned replacements with OpenAI.

## Modules
- `app.py` – Streamlit entrypoint handling tabs, session state, sidebar controls, and layout.
- `utils.py` – shared helpers for DataForSEO caching, order parsing (CSV, clipboard, GPT‑5‑nano), and `OrderInput` models.
- `prompt_manager.py` – default prompt presets and placeholder metadata surfaced by the UI.
- `openai_writer.py` – OpenAI Responses API wrapper (model catalogue, cost estimation, result validation).
- `__init__.py` – package marker.

## Session State Keys
| Key | Description |
| --- | --- |
| `orders` | List of order dictionaries edited via the table UI. |
| `serp_results` | Cached SERP responses per order index (titles, raw JSON, selected count). |
| `generation_results` | OpenAI outputs keyed by order index. |
| `prompt_overrides` | Mutable prompt preset text, allowing on-the-fly edits without touching source files. |
| `custom_prompt_text` | The last custom prompt entered in the UI. |
| `paste_preview`, `paste_log` | Temporary structures used by the paste/AI parse helper. |
| `serp_limit`, `serp_use_cache`, `serp_cache_only`, `default_serp_keyword` | Cached sidebar options. |
| `model_id`, `num_titles_requested` | Preferred generation settings. |

## Orders Tab
- **Manual table editing**: use the data editor to tweak `domain`, `anchor_keyword`, `target_url`, `serp_keyword`, and optional `extra_context`.
- **Paste helper**: supports Quick Parse (CSV/TSV heuristics) and **AI Parse** using GPT‑5‑nano with a JSON schema enforced response.
- **CSV upload**: accepts comma- or tab-delimited files; header aliases are mapped automatically.
- Blank orders can be appended for manual entry; the "Clear orders" button resets all downstream results.

## SERP Review Tab
- Batch-select orders and fetch SERPs with one click using the multiselect + `Fetch selected` controls.
- Each call runs `dfs_serp_fetcher.py` via `utils.fetch_serp` and stores the untouched titles for downstream use.
- Queries default to `site:{domain} {serp_keyword}` unless the site operator is disabled in code (currently enabled).
- Sidebar toggles:
  - **DataForSEO mode** – defaults to *Live (immediate)*, with an option to fall back to the standard queue.
  - **SERP titles to fetch** (default 25).
  - **Prefer cached SERP payloads** – load from `assets/samples/` before hitting the API.
  - **Offline mode** – require cache hits; no API calls issued.
- Adjust how many top titles feed the prompt via the per-order slider; selections persist.
- Raw JSON payloads remain available for debugging.

## Generate Titles Tab
- Configure prompts/models once, then generate titles for a single order or an entire selection via the batch controls.
- Prompt presets originate from `prompt_manager.get_default_templates()` and remain session-editable; a custom textarea is available for bespoke prompts.
- Available models (`openai_writer.MODEL_OPTIONS`): `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1-mini`; requests auto-set verbosity and reasoning (when supported).
- Placeholder previews show rendered prompts (with safe substitution) before any API call.
- `generate_titles` validates JSON output, captures token usage, and estimates per-run cost.
- Batch runs automatically export results to the session state, with on-demand CSV/JSON downloads summarising all generated titles and rationales.
- Successful runs display titles with rationales plus metadata (tokens, cost); raw responses remain available for debugging.

## Caching & Assets
- Each SERP result is cached to `assets/samples/serp_<sha1>_limitX.json`. Files can be checked in for offline demos (avoid sensitive payloads).
- `utils.list_cached_payloads()` exposes cache inventory for future tooling/tests.

## Environment Variables
Relies on `.env` (loaded with `python-dotenv`) containing:
- `DFS_LOGIN` / `DATAFORSEO_LOGIN`
- `DFS_PASSWORD` / `DATAFORSEO_PASSWORD`
- `OPENAI_API_KEY`

The UI defaults to the DataForSEO **Live** endpoint; switch to the standard queue via the sidebar mode selector if needed.

## Extensibility Notes
- Add more prompt presets by extending `get_default_templates()`.
- Plug in additional model IDs by updating `MODEL_OPTIONS` in `openai_writer.py`.
- For batch processing, iterate through `serp_results` to call `generate_titles` on multiple orders at once.
- Consider adding pytest suites that stub `run_fetcher` and OpenAI to validate parsing, caching, and prompt rendering logic.

### OpenAI Generation
- Supported model IDs: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1-mini`.
- The app sends `max_output_tokens=900` and `text={"verbosity": "medium"}` for every call.
- `reasoning={"effort": "medium"}` is added automatically when using GPT-5/GPT-5-mini.
- Temperature is no longer exposed or sent; GPT-5 models will reject it via the Responses API.

- Default generation request count is 6 titles; prompts enforce six brand-free options to avoid casino review phrasing.

## Persistence
- SQLite database stored at `assets/data/app.sqlite` keeps SERP snapshots (keyed by domain + keyword + limit) and every generation output.
- When a matching snapshot exists, the app skips the DataForSEO call and hydrates results directly from the database.
- Generation logs include model, prompt, rendered prompt, JSON output, and token/cost metrics for later analysis.
