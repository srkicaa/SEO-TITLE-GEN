# Progress Log

## 2025-09-18
- Streamlit workflow now runs purely on official GPT-5 model aliases (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, plus `gpt-4.1-mini`).
- Removed the temperature slider and all temperature arguments because GPT-5 reasoning models reject that parameter via the Responses API.
- OpenAI payload is simplified to the supported knobs: `max_output_tokens=900`, `text={"verbosity": "medium"}`, and `reasoning={"effort": "medium"}` only when using GPT-5 or GPT-5-mini.
- Updated prompt presets to forbid brand/trademark usage in generated titles.
- Batched SERP fetch/generation remains available; new exports deliver combined CSV/JSON output.
- Documentation refreshed (`README.md`, `streamlit_app/README.md`) with the current GPT-5 settings and workflow expectations.

To continue work: restart Streamlit via `streamlit run streamlit_app/app.py` and use the updated model picker. 
- Improved OpenAI response parsing with new fallback for `output_text`/message blocks so GPT-5 Responses payloads are handled reliably.
- Removed all temperature references from generation calls; payload now uses only supported GPT-5 parameters (verbosity + optional reasoning).
- Added JSON repair fallback so partial arrays from GPT-5 are parsed when possible.
- Default title request count set to 5 and prompt templates now demand five brand-free options.

## 2025-09-19
- Added automatic archiving of every OpenAI generation run under `assets/generated/`, capturing the order payload, prompt, model, usage, and resulting titles for later analysis.
- Normalised generated headlines to Title Case on the server side so UI output is consistent even when models vary in casing.
- Added three new prompt presets (`News Pulse`, `Trend Report`, `Thought Leadership`) for topical, trend-driven, and forward-looking headline angles without touching the original defaults.
- Hardened the AI order parser with JSON schema responses, structured fallbacks, and clearer error messages.
- Improved quick CSV parsing to keep anchor countries separate and require fully-qualified target URLs.
- Consolidated topical prompts into a single `Topical Pulse` preset that emphasises anchor country context.
- AI order parser now tolerates list payloads from GPT and the consolidated topical prompt explicitly pulls in recent {serp_keyword} developments.
- Added SQLite-backed history with searchable Streamlit tab for previously generated titles.
- Adjusted AI parser to use JSON schema via text.format (no response_format) after OpenAI API rejection.
- Bumped default SERP fetch limit to 35 and wired full logging/history instrumentation.
- Persisted every SERP fetch (limit 35 by default) to assets/data/app.sqlite alongside full logging hooks.
- Reverted AI parser to create()+manual fallback extraction with retry (no responses.parse).
- SERP fetch now pulls 100 titles per domain (preview still uses 35) and archives every payload in SQLite.

## 2025-09-20
- Defaulted all new Streamlit runs to the DataForSEO Standard queue and added an explicit sidebar toggle when the live endpoint is required.
- Implemented two-tier SERP caching: first reuse exact task hashes, then fall back to the most recent snapshot for the requested domain before calling the API.
- Normalised stored domain names so history lookups ignore protocol/`www` prefixes and return consistent payloads regardless of input formatting.
- Added a “Force new DataForSEO fetch” sidebar option that bypasses history while still respecting cache-only mode, giving manual control over fresh pulls.

## 2025-10-30
- Audited both SERP fetchers and corrected all Google Organic endpoints to `/serp/google/organic/task_post` and `/serp/google/organic/task_get/regular/{id}` for Standard queue workflows; Live requests stay on `/live/regular`.
- Increased default Standard-queue polling window (5 s interval × 240 attempts ≈ 20 min) and added explicit response validation so POST calls fail fast unless the task reports `status_code` 20100.
- Captured a debug run for `primedope.com` using `python3 dfs_serp_fetcher.py --domain primedope.com --serp-keyword casino --debug`, which revealed DataForSEO returned `task_status_code=40203` (“daily money limit exceeded”), explaining the prior 40401 polls.
- Next actions: raise/reset the daily cost ceiling in the DataForSEO dashboard, re-run the debug command to harvest POST/GET payloads after the limit resets, and share those JSON snippets with DataForSEO support if tasks still fail.
