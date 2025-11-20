// dfs_serp_fetcher.js
/**
 * SERP Title Analyzer (Node.js)
 *
 * - Uses DataForSEO SERP API (production, Standard queue) to POST a Google Organic task,
 *   poll for results, extract and dedupe title-like items, and return the top N titles.
 *
 * - Default NUM_TITLES set to 50 (you can override via env var NUM_TITLES).
 *
 * - Important: Google Organic billing is per 100 results (minimum billed unit = 100).
 *   Even if you only analyze top 10 or 50, you will be charged for the 100-result block.
 *
 * Requirements:
 *   - Node 14+
 *   - npm install axios
 *
 * Environment variables:
 *   - DFS_LOGIN     (DataForSEO login, from https://app.dataforseo.com/api-access)
 *   - DFS_PASSWORD  (DataForSEO API password)
 *   - NUM_TITLES    (optional; default 50)
 *
 * Endpoints & docs:
 *   - POST task_post: https://docs.dataforseo.com/v3/serp/google/organic/task_post/?bash
 *   - GET task_get:  https://docs.dataforseo.com/v3/serp/google/organic/task_get/regular/?bash
 *   - Pricing / depth/billing: https://dataforseo.com/pricing/serp/serp-api
 *   - How many results & filter=0: https://dataforseo.com/help-center/how-many-results-scraped
 *   - Pingbacks/postbacks: https://dataforseo.com/help-center/pingbacks-postbacks-with-dataforseo-api
 *
 * Usage:
 *   - Set DFS_LOGIN and DFS_PASSWORD environment variables.
 *   - Optionally set NUM_TITLES (default 50).
 *   - Run: node dfs_serp_fetcher.js
 *
 * Outputs:
 *   - Logs raw POST and GET responses (for debugging).
 *   - Logs extracted top N title-like items.
 *   - Outputs JSON object with task_id, query, depth, extracted.
 *
 * Notes on postback/pingback:
 *   - To receive results pushed to your server, include postback_url in the payload and handle gzip-compressed POSTs.
 *   - Whitelist DataForSEO IPs (see docs) if using postback/pingback.
 */

const axios = require('axios');

const DFS_BASE = 'https://api.dataforseo.com/v3';
const TASK_POST_URL = `${DFS_BASE}/serp/google/organic/task_post`;
const TASK_GET_URL = `${DFS_BASE}/serp/google/organic/task_get/regular`;

// ENV & defaults
const DFS_LOGIN = process.env.DFS_LOGIN || '';
const DFS_PASSWORD = process.env.DFS_PASSWORD || '';
const NUM_TITLES = parseInt(process.env.NUM_TITLES || '50', 10); // DEFAULT = 50 as requested

if (!DFS_LOGIN || !DFS_PASSWORD) {
  console.error('ERROR: Set DFS_LOGIN and DFS_PASSWORD environment variables (get from https://app.dataforseo.com/api-access).');
  process.exit(1);
}

// Basic Auth header
function createAuthHeader(login, password) {
  const cred = Buffer.from(`${login}:${password}`).toString('base64');
  return `Basic ${cred}`;
}

const HEADERS = {
  'Authorization': createAuthHeader(DFS_LOGIN, DFS_PASSWORD),
  'Content-Type': 'application/json'
};

function ensureTaskStatus(respJson, { allowedParent, allowedTask, context }) {
  const parentStatus = respJson && respJson.status_code;
  if (!allowedParent.includes(parentStatus)) {
    const message = (respJson && respJson.status_message) || 'unknown';
    throw new Error(`${context} failed: status_code=${parentStatus} message=${message}`);
  }
  const tasks = respJson && respJson.tasks;
  if (!Array.isArray(tasks) || tasks.length === 0) {
    throw new Error(`${context} failed: tasks array missing`);
  }
  const taskStatus = tasks[0].status_code;
  if (!allowedTask.includes(taskStatus)) {
    const message = tasks[0].status_message || 'unknown';
    throw new Error(`${context} failed: task_status_code=${taskStatus} message=${message}`);
  }
}

// Sleep utility
function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

// 1) POST a Standard Google Organic task_post
async function postSerpTask({
  keyword,
  language_name = 'English',
  location_name = 'United States',
  device = 'desktop',
  os = 'windows',
  depth = 100, // default 100 results (billing per 100)
  tag = null,
  postback_url = null
} = {}) {
  const payload = [{
    keyword,
    language_name,
    location_name,
    device,
    os,
    depth
  }];
  if (tag) payload[0].tag = tag;
  if (postback_url) {
    payload[0].postback_url = postback_url;
    payload[0].postback_data = 'regular'; // 'regular' | 'advanced' | 'html'
  }

  const resp = await axios.post(TASK_POST_URL, payload, { headers: HEADERS, timeout: 60000 });
  const json = resp.data;
  ensureTaskStatus(json, {
    allowedParent: [20000],
    allowedTask: [20100],
    context: 'Task POST'
  });
  return json;
}

// 2) Poll Task GET for result readiness (Standard queue can take several minutes)
async function pollTaskGet(taskId, { intervalMs = 5000, maxAttempts = 720 } = {}) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const resp = await axios.get(`${TASK_GET_URL}/${taskId}`, { headers: HEADERS, timeout: 60000 });
      const json = resp.data;
      if (json && Array.isArray(json.tasks) && json.tasks.length > 0) {
        const task = json.tasks[0];
        // status_code 20000 means OK; result present indicates ready
        if (task.status_code === 20000 && Array.isArray(task.result) && task.result.length > 0) {
          return json;
        }
        // If final attempt, surface status info
        if (attempt === maxAttempts) {
          throw new Error(`Task not ready. Last status_code=${task.status_code} message=${task.status_message}`);
        }
      }
    } catch (err) {
      if (attempt === maxAttempts) throw err;
    }
    await sleep(intervalMs);
  }
  throw new Error('Polling timed out.');
}

// 3) Extract title-like items from Task GET response
function extractSerpItems(respJson, topN = 50) {
  const items = [];

  function collect(node, resultType = null) {
    if (!node) return;
    if (Array.isArray(node)) {
      for (const el of node) collect(el, resultType);
      return;
    }
    if (typeof node === 'object') {
      // If object contains title-like or snippet-like keys, capture
      if (node.title || node.title_full || node.headline || node.name || node.snippet || node.description) {
        const title = node.title || node.title_full || node.headline || node.name || null;
        const url = node.url || node.link || node.displayed_url || null;
        const snippet = node.snippet || node.description || node.passage || null;
        const rank_absolute = node.rank_absolute || node.rank || null;
        const rank_group = node.rank_group || null;

        items.push({
          title: title ? String(title).trim() : null,
          url: url ? String(url).trim() : null,
          snippet: snippet ? String(snippet).trim() : null,
          rank_absolute,
          rank_group,
          result_type: resultType || node.type || null
        });
      }
      // Recurse into nested fields; detect block names to pass as result_type
      for (const [k, v] of Object.entries(node)) {
        let newType = resultType;
        if (['organic', 'items', 'featured_snippet', 'knowledge_graph', 'local_pack', 'people_also_ask'].includes(k)) {
          newType = k;
        }
        collect(v, newType);
      }
    }
  }

  // Start at tasks -> result if present
  if (respJson && typeof respJson === 'object' && respJson.tasks) {
    collect(respJson.tasks);
  } else {
    collect(respJson);
  }

  // Deduplicate preserving order by (title || url)
  const seen = new Set();
  const dedup = [];
  for (const it of items) {
    const t = it.title || '';
    const u = it.url || '';
    const key = `${t}||${u}`;
    if (t && !seen.has(key)) {
      dedup.push(it);
      seen.add(key);
    }
    if (dedup.length >= topN) break;
  }

  return dedup;
}

// 4) Parse task ID from POST response (robust)
function parseTaskIdFromPostResponse(postResp) {
  if (!postResp) return null;
  if (Array.isArray(postResp.tasks) && postResp.tasks.length > 0) {
    const t = postResp.tasks[0];
    return t.id || t.task_id || t.taskId || null;
  }
  if (Array.isArray(postResp) && postResp.length > 0) {
    const t = postResp[0];
    return t.id || t.task_id || null;
  }
  return null;
}

// Main flow
(async () => {
  try {
    // Config - adjust as needed
    const domain = 'example.com';
    const seedKeyword = 'best seo practices';
    const query = `site:${domain} ${seedKeyword}`; // remove 'site:' if unnecessary
    const depth = 100; // default 100 results (billing per 100)
    const tag = 'serp_title_analysis_job';

    console.log('Posting DataForSEO task_post (Standard queue, production)...');
    const postResp = await postSerpTask({
      keyword: query,
      language_name: 'English',
      location_name: 'United States',
      device: 'desktop',
      os: 'windows',
      depth,
      tag
    });

    console.log('POST response (raw):', JSON.stringify(postResp, null, 2));
    if (postResp.cost !== undefined) console.log('POST indicated cost:', postResp.cost);

    const taskId = parseTaskIdFromPostResponse(postResp);
    if (!taskId) {
      console.error('ERROR: Could not parse task_id from POST response. Inspect the POST response above.');
      process.exit(2);
    }
    console.log('Task ID:', taskId);

    console.log('Polling Task GET for results (may take seconds to minutes)...');
    const got = await pollTaskGet(taskId, { intervalMs: 5000, maxAttempts: 720 });
    console.log('Task GET response (raw):', JSON.stringify(got, null, 2));
    if (got.cost !== undefined) console.log('Task cost:', got.cost);
    if (got.time) console.log('API reported time:', got.time);

    const topN = NUM_TITLES;
    const extracted = extractSerpItems(got, topN);

    console.log(`Extracted top ${topN} title-like items:`);
    extracted.forEach((it, idx) => {
      console.log(`#${idx + 1}: "${it.title}"  (url: ${it.url || 'N/A'})  rank: ${it.rank_absolute || 'N/A'}  type: ${it.result_type || 'N/A'}`);
    });

    // Final output object
    const output = {
      task_id: taskId,
      query,
      depth,
      extracted
    };
    console.log('OUTPUT JSON:\n', JSON.stringify(output, null, 2));

    process.exit(0);
  } catch (err) {
    console.error('ERROR during execution:', (err && err.message) || err);
    if (err.response && err.response.data) {
      console.error('Response data:', JSON.stringify(err.response.data, null, 2));
    }
    process.exit(3);
  }
})();
