#!/usr/bin/env python3
"""
Python port of the DataForSEO SERP Title Analyzer.

- Posts a Google Organic task (Standard queue) via DataForSEO API
- Polls for completion, extracts and deduplicates title-like entries
- Returns the top N items as JSON (mirrors dfs_serp_fetcher.js output structure)

Usage examples:
  python dfs_serp_fetcher.py --domain example.com --serp-keyword "best seo practices"
  python dfs_serp_fetcher.py --query "site:example.com best seo practices" --num-titles 25

Environment variables (first non-empty value is used):
  DFS_LOGIN | DATAFORSEO_LOGIN
  DFS_PASSWORD | DATAFORSEO_PASSWORD
  NUM_TITLES (optional default)

Dependencies:
  pip install requests
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

DFS_BASE = "https://api.dataforseo.com/v3"
TASK_POST_URL = f"{DFS_BASE}/serp/google/organic/task_post"
TASK_GET_URL = f"{DFS_BASE}/serp/google/organic/task_get/regular"
TASK_LIVE_URL = f"{DFS_BASE}/serp/google/organic/live/regular"

DEFAULT_LANGUAGE = "English"
DEFAULT_LOCATION = "United States"
DEFAULT_DEVICE = "desktop"
DEFAULT_OS = "windows"
DEFAULT_DEPTH = 100
DEFAULT_POLL_INTERVAL = 5.0  # seconds; standard queue often needs several minutes
DEFAULT_MAX_ATTEMPTS = 720  # 60 minutes total wait time by default


class SerpFetcherError(RuntimeError):
    """Raised for DataForSEO workflow errors."""


def _get_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def create_auth_header(login: str, password: str) -> Dict[str, str]:
    credentials = f"{login}:{password}".encode("utf-8")
    token = base64.b64encode(credentials).decode("utf-8")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


def _ensure_task_status(
    response_json: Dict[str, Any],
    *,
    allowed_parent: Iterable[int],
    allowed_task: Iterable[int],
    context: str,
) -> None:
    parent_status = response_json.get("status_code")
    if parent_status not in allowed_parent:
        status_message = response_json.get("status_message", "unknown")
        raise SerpFetcherError(
            f"{context} failed: status_code={parent_status} message={status_message}"
        )
    tasks = response_json.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SerpFetcherError(f"{context} failed: tasks array missing")
    task_status = tasks[0].get("status_code")
    if task_status not in allowed_task:
        task_message = tasks[0].get("status_message", "unknown")
        raise SerpFetcherError(
            f"{context} failed: task_status_code={task_status} message={task_message}"
        )


def post_serp_live(
    *,
    keyword: str,
    language_name: str = DEFAULT_LANGUAGE,
    location_name: str = DEFAULT_LOCATION,
    device: str = DEFAULT_DEVICE,
    os_name: str = DEFAULT_OS,
    depth: int = DEFAULT_DEPTH,
    tag: Optional[str] = None,
    headers: Dict[str, str],
    timeout: int = 60000,
) -> Dict[str, Any]:
    payload: List[Dict[str, Any]] = [
        {
            "keyword": keyword,
            "language_name": language_name,
            "location_name": location_name,
            "device": device,
            "os": os_name,
            "depth": depth,
        }
    ]
    if tag:
        payload[0]["tag"] = tag

    response = requests.post(
        TASK_LIVE_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout / 1000.0,
    )
    response.raise_for_status()
    payload_json = response.json()
    _ensure_task_status(
        payload_json,
        allowed_parent={20000},
        allowed_task={20000},
        context="Live SERP request",
    )
    return payload_json


def post_serp_task(
    *,
    keyword: str,
    language_name: str = DEFAULT_LANGUAGE,
    location_name: str = DEFAULT_LOCATION,
    device: str = DEFAULT_DEVICE,
    os_name: str = DEFAULT_OS,
    depth: int = DEFAULT_DEPTH,
    tag: Optional[str] = None,
    postback_url: Optional[str] = None,
    headers: Dict[str, str],
    timeout: int = 60000,
) -> Dict[str, Any]:
    payload: List[Dict[str, Any]] = [
        {
            "keyword": keyword,
            "language_name": language_name,
            "location_name": location_name,
            "device": device,
            "os": os_name,
            "depth": depth,
        }
    ]
    if tag:
        payload[0]["tag"] = tag
    if postback_url:
        payload[0]["postback_url"] = postback_url
        payload[0]["postback_data"] = "regular"

    response = requests.post(
        TASK_POST_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout / 1000.0,
    )
    response.raise_for_status()
    payload_json = response.json()
    _ensure_task_status(
        payload_json,
        allowed_parent={20000},
        allowed_task={20100},
        context="Task POST",
    )
    return payload_json


def poll_task_get(
    task_id: str,
    *,
    headers: Dict[str, str],
    interval: float = DEFAULT_POLL_INTERVAL,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: int = 60000,
) -> Dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                f"{TASK_GET_URL}/{task_id}",
                headers=headers,
                timeout=timeout / 1000.0,
            )
            response.raise_for_status()
            payload = response.json()
            tasks = payload.get("tasks") or []
            if tasks:
                task = tasks[0]
                status_code = task.get("status_code")
                results = task.get("result")
                if status_code == 20000 and isinstance(results, list) and results:
                    return payload
                if attempt == max_attempts:
                    message = task.get("status_message", "unknown status")
                    elapsed_seconds = attempt * interval if interval > 0 else 0.0
                    if elapsed_seconds > 0:
                        wait_details = f"after waiting {elapsed_seconds:.0f}s (~{elapsed_seconds / 60.0:.1f}m)"
                    else:
                        wait_details = f"after {max_attempts} attempts"
                    raise SerpFetcherError(
                        f"Task not ready {wait_details}. "
                        f"status_code={status_code} message={message}"
                    )
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise SerpFetcherError(f"Task polling failed: {exc}") from exc
        time.sleep(interval)
    raise SerpFetcherError("Polling timed out")


TITLE_KEYS = {
    "title",
    "title_full",
    "headline",
    "name",
}
SNIPPET_KEYS = {
    "snippet",
    "description",
    "passage",
}
RESULT_TYPE_HINTS = {
    "organic",
    "items",
    "featured_snippet",
    "knowledge_graph",
    "local_pack",
    "people_also_ask",
}


def extract_serp_items(resp_json: Dict[str, Any], top_n: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def collect(node: Any, result_type: Optional[str] = None) -> None:
        if node is None:
            return
        if isinstance(node, list):
            for child in node:
                collect(child, result_type)
            return
        if isinstance(node, dict):
            has_title = any(k in node and node[k] for k in TITLE_KEYS)
            has_snippet = any(k in node and node[k] for k in SNIPPET_KEYS)
            if has_title or has_snippet:
                title = _first_non_empty(node, TITLE_KEYS)
                url = _first_non_empty(node, {"url", "link", "displayed_url"})
                snippet = _first_non_empty(node, SNIPPET_KEYS)
                rank_absolute = node.get("rank_absolute") or node.get("rank")
                rank_group = node.get("rank_group")
                items.append(
                    {
                        "title": title.strip() if isinstance(title, str) else None,
                        "url": url.strip() if isinstance(url, str) else None,
                        "snippet": snippet.strip() if isinstance(snippet, str) else None,
                        "rank_absolute": rank_absolute,
                        "rank_group": rank_group,
                        "result_type": result_type or node.get("type"),
                    }
                )
            for key, value in node.items():
                next_type = result_type
                if key in RESULT_TYPE_HINTS:
                    next_type = key
                collect(value, next_type)

    tasks = resp_json.get("tasks")
    if tasks is not None:
        collect(tasks)
    else:
        collect(resp_json)

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        title = item.get("title") or ""
        url = item.get("url") or ""
        if not title:
            continue
        key = f"{title}||{url}"
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
        if len(deduped) >= top_n:
            break
    return deduped


def _first_non_empty(node: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = node.get(key)
        if value:
            return value
    return None


def parse_task_id(post_response: Dict[str, Any]) -> Optional[str]:
    tasks = post_response.get("tasks")
    if isinstance(tasks, list) and tasks:
        task = tasks[0]
        return task.get("id") or task.get("task_id") or task.get("taskId")
    if isinstance(post_response, list) and post_response:
        task = post_response[0]
        if isinstance(task, dict):
            return task.get("id") or task.get("task_id")
    return None


def build_query(domain: Optional[str], keyword: Optional[str], use_site: bool) -> str:
    if domain and keyword:
        return f"site:{domain} {keyword}" if use_site else f"{domain} {keyword}"
    if keyword:
        return keyword
    raise SerpFetcherError("A SERP keyword is required unless --query is provided")


def run_fetcher(args: argparse.Namespace) -> Dict[str, Any]:
    login = args.dfs_login or _get_env("DFS_LOGIN", "DATAFORSEO_LOGIN")
    password = args.dfs_password or _get_env("DFS_PASSWORD", "DATAFORSEO_PASSWORD")
    if not login or not password:
        raise SerpFetcherError(
            "Provide DataForSEO credentials via --dfs-login/--dfs-password or environment variables."
        )

    timeout_ms = getattr(args, "timeout_ms", 60000)

    num_titles_env = os.getenv("NUM_TITLES")
    default_num = int(num_titles_env) if num_titles_env and num_titles_env.isdigit() else 50
    top_n = args.num_titles or default_num
    if top_n <= 0:
        raise SerpFetcherError("--num-titles must be greater than 0")

    query = args.query or build_query(args.domain, args.serp_keyword, not args.no_site_operator)

    headers = create_auth_header(login, password)

    if args.debug:
        mode = "live" if args.live else "standard"
        print(f"Posting {mode} task for query: {query}", file=sys.stderr)

    if args.live:
        post_response = post_serp_live(
            keyword=query,
            language_name=args.language_name,
            location_name=args.location_name,
            device=args.device,
            os_name=args.os,
            depth=args.depth,
            tag=args.tag,
            headers=headers,
            timeout=timeout_ms,
        )
        if args.debug:
            print(json.dumps(post_response, indent=2), file=sys.stderr)
        extracted = extract_serp_items(post_response, top_n)
        task_id = parse_task_id(post_response)
        return {
            "mode": "live",
            "task_id": task_id,
            "query": query,
            "depth": args.depth,
            "extracted": extracted,
        }

    post_response = post_serp_task(
        keyword=query,
        language_name=args.language_name,
        location_name=args.location_name,
        device=args.device,
        os_name=args.os,
        depth=args.depth,
        tag=args.tag,
        postback_url=args.postback_url,
        headers=headers,
        timeout=timeout_ms,
    )

    if args.debug:
        print(json.dumps(post_response, indent=2), file=sys.stderr)

    task_id = parse_task_id(post_response)
    if not task_id:
        raise SerpFetcherError("Unable to parse task_id from POST response")

    if args.debug:
        print(f"Polling task {task_id} ...", file=sys.stderr)

    poll_response = poll_task_get(
        task_id,
        headers=headers,
        interval=args.poll_interval,
        max_attempts=args.max_attempts,
        timeout=timeout_ms,
    )

    if args.debug:
        print(json.dumps(poll_response, indent=2), file=sys.stderr)

    extracted = extract_serp_items(poll_response, top_n)

    return {
        "mode": "standard",
        "task_id": task_id,
        "query": query,
        "depth": args.depth,
        "extracted": extracted,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SERP titles via DataForSEO (Python port)")
    parser.add_argument("--dfs-login", help="DataForSEO login (overrides env)")
    parser.add_argument("--dfs-password", help="DataForSEO password (overrides env)")
    parser.add_argument("--live", action="store_true", help="Use the Live endpoint (immediate results)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", help="Full query to send to DataForSEO")
    group.add_argument("--domain", help="Target domain to combine with --serp-keyword")
    parser.add_argument("--serp-keyword", help="Keyword for the SERP query (used with --domain)")
    parser.add_argument("--no-site-operator", action="store_true", help="Disable automatic site: prefix when using --domain")
    parser.add_argument("--language-name", default=DEFAULT_LANGUAGE)
    parser.add_argument("--location-name", default=DEFAULT_LOCATION)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--os", dest="os", default=DEFAULT_OS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--num-titles", type=int, help="Number of titles to return (overrides NUM_TITLES env)")
    parser.add_argument("--tag", help="Optional tag for the DataForSEO task")
    parser.add_argument("--postback-url", help="Optional postback URL")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--debug", action="store_true", help="Print verbose logs to stderr")
    args = parser.parse_args(argv)

    if not args.query and not args.domain:
        parser.error("Either --query or --domain must be provided")
    if not args.query and not args.serp_keyword:
        parser.error("--serp-keyword is required when using --domain")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        result = run_fetcher(args)
        print(json.dumps(result, indent=2))
        return 0
    except SerpFetcherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # unexpected
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
