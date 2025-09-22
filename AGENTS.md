# Repository Guidelines

## Project Structure & Module Organization
- Root contains dfs_serp_fetcher.js (original Node implementation), dfs_serp_fetcher.py (Python port), .env for credentials, and instructions.txt for product context.
- Keep new Streamlit or orchestration code inside a streamlit_app/ folder with submodules (openai_writer.py, utils.py, pp.py) to separate UI from fetchers.
- Store shared assets (mock SERP payloads, prompt samples) under ssets/ to aid local testing and documentation.

## Build, Test, and Development Commands
- 
pm install axios — install the Node dependency required by dfs_serp_fetcher.js.
- 
ode dfs_serp_fetcher.js — run the legacy fetcher with environment variables set (DFS_LOGIN, DFS_PASSWORD).
- pip install requests python-dotenv streamlit — minimal Python stack for the new workflow.
- python dfs_serp_fetcher.py --domain example.com --serp-keyword "seed" — fetch domain-specific SERP titles from DataForSEO.
- streamlit run streamlit_app/app.py — launch the planned UI once implemented.

## Coding Style & Naming Conventions
- JavaScript: follow existing two-space indentation, prefer const/let, and descriptive function names (postSerpTask).
- Python: use 4-space indentation, snake_case for functions (un_fetcher) and uppercase constants (DFS_BASE).
- Config and secrets live in .env; never commit real credentials.

## Testing Guidelines
- Use captured JSON responses in ssets/samples/ for offline validation.
- For Python, add pytest cases that exercise polling, dedupe, and CLI argument parsing.
- Manual smoke tests: run both fetchers against the sandbox API before pointing to production credentials.

## Security & Configuration Tips
- Rotate API keys if .env is shared; vault production secrets outside the repo.
- Expose command-line flags (--no-site-operator, --num-titles) rather than hard-coding campaign parameters.
- Log sensitive API responses only when --debug is passed.

## Commit & Pull Request Guidelines
- Write commits in imperative tense (e.g., "Add Streamlit wrapper"), grouping related changes.
- Pull requests should describe the DataForSEO/OpenAI touchpoints, list test evidence, and flag any billing-impacting defaults.
- Include screenshots or CLI output when UI or prompt behavior changes.
