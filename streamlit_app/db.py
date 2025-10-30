"""Lightweight SQLite persistence for SERP snapshots and generated titles."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import SerpFetchResult, sanitize_domain

DB_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"
DB_PATH = DB_DIR / "app.sqlite"


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise database tables if they do not yet exist."""

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS serp_snapshots (
                id INTEGER PRIMARY KEY,
                query_hash TEXT UNIQUE,
                domain TEXT,
                serp_keyword TEXT,
                anchor_keyword TEXT,
                target_url TEXT,
                result_limit INTEGER,
                raw_json TEXT,
                extracted_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_titles (
                id INTEGER PRIMARY KEY,
                query_hash TEXT,
                model TEXT,
                prompt TEXT,
                rendered_prompt TEXT,
                output_json TEXT,
                cost_usd REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                order_domain TEXT,
                anchor_country TEXT,
                serp_keyword TEXT,
                anchor_keyword TEXT,
                target_url TEXT,
                created_at TEXT
            )
            """
        )
        try:
            conn.execute("ALTER TABLE generated_titles ADD COLUMN anchor_country TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def store_serp_snapshot(
    *,
    query_hash: str,
    order_data: Dict[str, str],
    serp_keyword: str,
    limit: int,
    result: SerpFetchResult,
) -> None:
    """Persist a SERP fetch result."""

    payload = json.dumps(result.raw)
    extracted = json.dumps(result.titles)
    now = datetime.utcnow().isoformat()
    raw_domain = order_data.get("domain", "")
    normalized_domain = sanitize_domain(raw_domain) or raw_domain.strip()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO serp_snapshots (
                query_hash,
                domain,
                serp_keyword,
                anchor_keyword,
                target_url,
                result_limit,
                raw_json,
                extracted_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query_hash) DO UPDATE SET
                raw_json=excluded.raw_json,
                extracted_json=excluded.extracted_json,
                created_at=excluded.created_at
            """,
            (
                query_hash,
                normalized_domain,
                serp_keyword,
                order_data.get("anchor_keyword", ""),
                order_data.get("target_url", ""),
                limit,
                payload,
                extracted,
                now,
            ),
        )
        conn.commit()


def get_serp_snapshot(query_hash: str) -> Optional[Dict[str, Any]]:
    """Return a cached SERP snapshot if available."""

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM serp_snapshots WHERE query_hash = ?", (query_hash,)
        ).fetchone()

    if row is None:
        return None

    raw_payload = json.loads(row["raw_json"])
    extracted = json.loads(row["extracted_json"])
    result = SerpFetchResult(
        query=raw_payload.get("query", ""),
        titles=extracted,
        raw=raw_payload,
        source="db",
        meta={"limit": row["result_limit"], "stored_at": row["created_at"]},
    )
    return {
        "result": result,
        "order": {
            "domain": row["domain"],
            "serp_keyword": row["serp_keyword"],
            "anchor_keyword": row["anchor_keyword"],
            "target_url": row["target_url"],
        },
        "limit": row["result_limit"],
    }


def get_latest_serp_snapshot_for_domain(
    domain: str, *, limit: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Return the most recent snapshot stored for a domain, optionally filtered by limit."""

    normalized = sanitize_domain(domain).lower()
    if not normalized:
        return None

    variants = [
        normalized,
        f"http://{normalized}",
        f"https://{normalized}",
        f"www.{normalized}",
        f"http://www.{normalized}",
        f"https://www.{normalized}",
    ]

    placeholders = ",".join("?" for _ in variants)

    def _fetch_rows(conn: sqlite3.Connection, *, include_limit: bool) -> list[sqlite3.Row]:
        domain_query = f"""
            SELECT *
            FROM serp_snapshots
            WHERE LOWER(domain) IN ({placeholders})
        """
        domain_params: list[Any] = list(variants)
        if include_limit and limit is not None:
            domain_query += " AND result_limit = ?"
            domain_params.append(limit)
        domain_query += " ORDER BY datetime(created_at) DESC LIMIT 5"
        rows = conn.execute(domain_query, domain_params).fetchall()
        if rows:
            return rows

        wildcard_query = """
            SELECT *
            FROM serp_snapshots
            WHERE LOWER(domain) LIKE ?
        """
        wildcard_params: list[Any] = [f"%{normalized}"]
        if include_limit and limit is not None:
            wildcard_query += " AND result_limit = ?"
            wildcard_params.append(limit)
        wildcard_query += " ORDER BY datetime(created_at) DESC LIMIT 5"
        return conn.execute(wildcard_query, wildcard_params).fetchall()

    def _select_best(rows: list[sqlite3.Row]) -> Optional[sqlite3.Row]:
        best_row: Optional[sqlite3.Row] = None
        for row in rows:
            extracted_raw = row["extracted_json"]
            try:
                extracted = json.loads(extracted_raw)
            except json.JSONDecodeError:
                extracted = None
            if extracted:
                return row
            if best_row is None:
                best_row = row
        return best_row

    with _connect() as conn:
        rows = _fetch_rows(conn, include_limit=True)
        row = _select_best(rows)
        if row is None:
            rows = _fetch_rows(conn, include_limit=False)
            row = _select_best(rows)

    if row is None:
        return None

    raw_payload = json.loads(row["raw_json"])
    extracted = json.loads(row["extracted_json"])
    result = SerpFetchResult(
        query=raw_payload.get("query", ""),
        titles=extracted,
        raw=raw_payload,
        source="db",
        meta={"limit": row["result_limit"], "stored_at": row["created_at"]},
    )

    return {
        "result": result,
        "order": {
            "domain": row["domain"],
            "serp_keyword": row["serp_keyword"],
            "anchor_keyword": row["anchor_keyword"],
            "target_url": row["target_url"],
        },
        "limit": row["result_limit"],
    }


def store_generation(
    *,
    query_hash: str,
    order_data: Dict[str, str],
    serp_keyword: str,
    generation: Dict[str, Any],
) -> None:
    """Persist model output for later analysis."""

    now = datetime.utcnow().isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO generated_titles (
                query_hash,
                model,
                prompt,
                rendered_prompt,
                output_json,
                cost_usd,
                prompt_tokens,
                completion_tokens,
                order_domain,
                anchor_country,
                serp_keyword,
                anchor_keyword,
                target_url,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_hash,
                generation["model"],
                generation["prompt"],
                generation["rendered_prompt"],
                generation["output_json"],
                generation["cost_usd"],
                generation["prompt_tokens"],
                generation["completion_tokens"],
                order_data.get("domain", ""),
                order_data.get("anchor_country", ""),
                serp_keyword,
                order_data.get("anchor_keyword", ""),
                order_data.get("target_url", ""),
                now,
            ),
        )
        conn.commit()


def list_generation_domains() -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT order_domain FROM generated_titles WHERE order_domain <> '' ORDER BY order_domain"
        ).fetchall()
    return [row[0] for row in rows]


def list_generation_models() -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model FROM generated_titles WHERE model <> '' ORDER BY model"
        ).fetchall()
    return [row[0] for row in rows]


def query_generation_history(
    *,
    domains: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    search: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM generated_titles"
    conditions: List[str] = []
    params: List[Any] = []

    if domains:
        placeholders = ",".join(["?"] * len(domains))
        conditions.append(f"order_domain IN ({placeholders})")
        params.extend(domains)
    if models:
        placeholders = ",".join(["?"] * len(models))
        conditions.append(f"model IN ({placeholders})")
        params.extend(models)
    if start_date:
        conditions.append("datetime(created_at) >= datetime(?)")
        params.append(start_date)
    if end_date:
        conditions.append("datetime(created_at) <= datetime(?)")
        params.append(end_date)
    if search:
        like = f"%{search.lower()}%"
        conditions.append(
            "(LOWER(output_json) LIKE ? OR LOWER(rendered_prompt) LIKE ? OR LOWER(prompt) LIKE ? OR LOWER(anchor_keyword) LIKE ? OR LOWER(order_domain) LIKE ? OR LOWER(target_url) LIKE ?)"
        )
        params.extend([like] * 6)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY datetime(created_at) DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_all_serp_snapshots_for_domain(domain: str) -> list[Dict[str, Any]]:
    """Return all snapshots for a domain, ordered by most recent first."""
    normalized = sanitize_domain(domain).lower()
    if not normalized:
        return []

    variants = [
        normalized,
        f"http://{normalized}",
        f"https://{normalized}",
        f"www.{normalized}",
        f"http://www.{normalized}",
        f"https://www.{normalized}",
    ]

    placeholders = ",".join("?" for _ in variants)
    query = f"""
        SELECT
            id,
            query_hash,
            domain,
            serp_keyword,
            anchor_keyword,
            target_url,
            result_limit,
            raw_json,
            extracted_json,
            created_at,
            LENGTH(extracted_json) as json_length
        FROM serp_snapshots
        WHERE LOWER(domain) IN ({placeholders})
        ORDER BY datetime(created_at) DESC
    """

    with _connect() as conn:
        rows = conn.execute(query, variants).fetchall()

    results = []
    for row in rows:
        try:
            extracted = json.loads(row["extracted_json"])
            title_count = len(extracted) if isinstance(extracted, list) else 0
        except (json.JSONDecodeError, TypeError):
            title_count = 0

        results.append({
            "id": row["id"],
            "query_hash": row["query_hash"],
            "domain": row["domain"],
            "serp_keyword": row["serp_keyword"],
            "anchor_keyword": row["anchor_keyword"],
            "target_url": row["target_url"],
            "result_limit": row["result_limit"],
            "created_at": row["created_at"],
            "title_count": title_count,
            "raw_json": row["raw_json"],
            "extracted_json": row["extracted_json"],
        })

    return results


__all__ = [
    "DB_PATH",
    "init_db",
    "get_serp_snapshot",
    "get_latest_serp_snapshot_for_domain",
    "get_all_serp_snapshots_for_domain",
    "store_serp_snapshot",
    "store_generation",
    "list_generation_domains",
    "list_generation_models",
    "query_generation_history",
]
