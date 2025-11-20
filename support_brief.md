# DataForSEO Support Brief

## Question for Support
- Could you review our workflow and suggest the best DataForSEO pricing/optimization plan, given that the app batches link-building orders, fetches Google Organic SERPs (default Standard queue, optional Live switch) using `site:<domain> <keyword>`, depth 100, desktop/English/US targeting, and polls every 5 s for up to an hour, then dedupes the top ~50 titles (title, URL, snippet, rank, type) while caching results to avoid repeat API calls, and the OpenAI titles are downstream from those cached SERPs?

## Current Workflow Snapshot
- Streamlit UI manages orders → SERP fetch → OpenAI title generation; legacy Python/Node scripts mirror the fetcher.
- Standard queue POSTs to `serp/google/organic/task_post`; Live endpoint is a toggle.
- Default depth 100, desktop, English/United States; queries usually include `site:` to scope to the target domain.
- Extraction captures title, URL, snippet, rank data, dedupes to ~50 results, and caches raw payloads (disk + SQLite).

## Cost Observations and Math Check
- Standard base price first 10 results: $0.0006; next 9 blocks (depth=100) at 75%: 9 × $0.00045 = $0.00405 → $0.00465 before multipliers.
- `site:` operator triggers ×5 multiplier → $0.00465 × 5 ≈ $0.02325 per domain (matches observed ~$0.023).
- Live endpoint base pricing: $0.002 + 9 × $0.0015 = $0.0155 → with `site:` multiplier ≈ $0.0775 per domain (higher than our anecdotal ~$0.05; task.cost field should be treated as source of truth).
- Polling every 5 s up to 720 attempts causes heavy GET traffic (5 s for ~60 min).

## Optimization Plan (Priority Order)
1. **Adopt DataForSEO Labs for domain-focused pulls**  
   - Use `dataforseo_labs/google/ranked_keywords` or `keywords_for_site` with `target=<domain>`, filters, and limits.  
   - Avoids `site:` multiplier, still returns titles/URLs/ranks relevant to the domain.
2. **Drop `site:` in SERP API and post-filter results**  
   - Run broader keyword queries; filter extracted items by domain + casino keywords in-app.  
   - Keeps Standard queue pricing without the ×5 multiplier.
3. **Lower depth and expand adaptively**  
   - Default to depth 10–20, increase only when insufficient matches are found.  
   - Directly reduces billed result blocks.
4. **Stick to Standard queue; limit Live usage**  
   - Live mode only when immediate results are essential; log per-task cost from responses.
5. **Relax polling cadence / use callbacks**  
   - Switch to webhooks (`postback_url`) or exponential backoff to cut GET call volume.
6. **Strengthen cache policy**  
   - Keep existing cache but enforce TTL (e.g., 24–72 h) and surface snapshot age in UI.
7. **Expose `task.cost` metrics**  
   - Capture/display cost per task to validate billing and inform tuning.

## Casino-Content Filtering Strategy
- **Labs with topical filters**: Request ranked keywords for the domain with `include_keywords` such as `["casino","slots","roulette","blackjack","poker"]`. Labs responses already include SERP title and landing URL so only casino-centric rows are returned.
- **SERP API post-filtering**: If using Google Organic without `site:`, filter extracted items by domain match plus casino keywords in title/snippet before dedupe; increase depth only when casino matches are sparse.
- **Hybrid funnel**: Use Labs to identify casino URLs quickly, then fetch SERP snapshots or scrape those URLs directly for meta titles if needed.

## Next Steps
- Prototype Labs-backed fetch flow (flagged) alongside current SERP path; compare `task.cost` across at least one batch.
- Add configuration to toggle `site:` usage and set initial depth (with on-demand refetch for deeper pages).
- Update polling logic to use callbacks/backoff and log per-task costs for visibility.
- Share results of Labs vs SERP post-filter tests with the team before finalizing pricing renegotiation.
