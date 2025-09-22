# DataForSEO Title Generation Toolkit

This workspace combines the legacy DataForSEO fetcher scripts with a new Streamlit UI that orchestrates SERP collection and OpenAI-powered title ideation for link-building orders.

## Repository Layout
- `dfs_serp_fetcher.py` / `dfs_serp_fetcher.js` – existing CLI utilities for posting DataForSEO SERP tasks and deduping titles.
- `streamlit_app/` – Streamlit package wrapping order intake, DataForSEO orchestration, prompt management, and OpenAI generation.
- `assets/samples/` – storage for cached or mock SERP payloads used in offline mode.
- `instructions.txt`, `CONTENT_RESEARCH_WORKFLOW.md`, `live_regular.md`, `llms.txt` – reference material and prompts supplied with the project.

## Environment & Dependencies
1. Copy `.env` into the project root with the following keys populated:
   - `DFS_LOGIN`, `DFS_PASSWORD` (or `DATAFORSEO_LOGIN`, `DATAFORSEO_PASSWORD`)
   - `OPENAI_API_KEY`
2. Install dependencies:
   ```bash
   npm install axios        # only if you use the legacy Node fetcher
   pip install -r requirements.txt  # optional if you curate one
   pip install streamlit requests python-dotenv openai pandas
   ```
3. (Optional) Capture SERP responses locally by dropping JSON into `assets/samples/` for offline testing.

## Running the Streamlit Workflow
```bash
streamlit run streamlit_app/app.py
```
The app loads environment variables via `python-dotenv`, so the credentials in `.env` are used automatically.

### High-Level Flow
1. **Orders tab** – paste, upload, or manually edit orders with columns: `domain`, `anchor_keyword`, `target_url`, `serp_keyword`, `extra_context`. A GPT‑5‑nano helper can parse messy clipboard dumps into structured rows.
2. **SERP Review tab** – batch-select orders and fetch SERPs in one click. The default mode is **DataForSEO Live**, with an option to switch to the standard queue. Cached payloads in `assets/samples/` can still be reused or forced via the sidebar toggles.
3. **Generate Titles tab** – configure prompts/models once and generate for all selected orders in bulk. Each run surfaces the raw SERP titles used, the rendered prompt, cost estimates, and downloadable CSV/JSON exports.

## Legacy Fetchers
You can still run the original scripts from the command line:
```bash
node dfs_serp_fetcher.js --domain example.com --serp-keyword "seed"
python dfs_serp_fetcher.py --domain example.com --serp-keyword "seed"
```
These respect the same environment variables and remain unmodified.

## Assets & Caching
- Every SERP call stores the raw payload under `assets/samples/serp_<digest>_limitX.json` for reproducible runs.
- The Streamlit sidebar exposes "Prefer cached" and "Offline mode" switches to control when caches are used.

## Prompt Management
- Default presets live in `streamlit_app/prompt_manager.py` and can be edited in the UI per session.
- All prompts support placeholders: `{domain}`, `{target_url}`, `{anchor_keyword}`, `{serp_keyword}`, `{num_requested}`, `{serp_titles}`, `{additional_context}`.
- Custom prompts are fully supported via a textarea in the generation tab.

### OpenAI Model Settings
- UI now lists official GPT-5 aliases: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, plus `gpt-4.1-mini` as a legacy fallback.
- Requests always use the Responses API with `max_output_tokens=900`, `text={"verbosity": "medium"}`, and auto-add `reasoning={"effort": "medium"}` for GPT-5 / GPT-5-mini.
- Temperature sliders were removed to avoid unsupported-parameter errors on GPT-5 models.
- Outputs are sanity-checked; malformed arrays are auto-repaired when possible before parsing.

## Batch Processing & Exports
- Use the batch multiselect controls on the SERP and Generation tabs to fetch or generate against every order in one shot.
- The sidebar defaults to the DataForSEO **Live** endpoint; switch to the standard queue via the mode selector if needed.
- Batch generation automatically reuses your current prompt/model settings and surfaces a downloadable CSV/JSON containing every generated title, rationale, and cost snippet.

## Testing Ideas
- Add pytest coverage for `streamlit_app/utils.py` (parsers, caching) using fixtures from `assets/samples/`.
- Record Streamlit session outputs or SERP JSON to confirm offline mode.

Refer to `streamlit_app/README.md` for a deeper dive into the UI workflow and module-level responsibilities.
