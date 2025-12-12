"""SQLite-backed historical evaluation memory."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EvaluationResult

DB_PATH = Path(os.getenv("EVAL_HISTORY_DB", "evaluation_history.db"))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    overall_score REAL NOT NULL,
    relevance REAL NOT NULL,
    hallucination REAL NOT NULL,
    completeness REAL NOT NULL,
    latency_ms REAL NOT NULL,
    passed INTEGER NOT NULL,
    failure_reason TEXT,
    source TEXT,
    raw_result_json TEXT
);
"""


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db() -> None:
    """Ensure the SQLite table exists."""
    conn = _get_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def _infer_failure_reason(result: EvaluationResult) -> Optional[str]:
    """Rough categorical reason for failure based on result flags."""

    if result.passed:
        return None

    if getattr(result.hallucination, "is_hallucinated", False):
        return "hallucination"
    if not getattr(result.completeness, "is_complete", True):
        return "completeness"
    if not getattr(result.relevance, "is_relevant", True):
        return "relevance"
    return "overall_score"


def log_evaluation(result: EvaluationResult, source: str = "normal") -> None:
    """Persist a single evaluation into SQLite."""

    init_db()

    data = result.to_dict()
    ts = data.get("timestamp") or datetime.utcnow().isoformat()
    failure_reason = _infer_failure_reason(result)

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO evaluation_runs (
                timestamp,
                overall_score,
                relevance,
                hallucination,
                completeness,
                latency_ms,
                passed,
                failure_reason,
                source,
                raw_result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                float(data["overall_score"]),
                float(data["relevance"]["score"]),
                float(data["hallucination"]["score"]),
                float(data["completeness"]["score"]),
                float(data["latency"]["total_ms"]),
                int(bool(data["passed"])),
                failure_reason,
                source,
                json.dumps(data),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    # nearest-rank style index
    k = int(round((p / 100.0) * (len(values) - 1)))
    return float(values[k])


def get_stats(limit: int = 20) -> Dict[str, Any]:
    """Return aggregate stats for the last `limit` evaluation runs."""

    import statistics

    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                overall_score,
                relevance,
                hallucination,
                completeness,
                latency_ms,
                passed,
                failure_reason
            FROM evaluation_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"count": 0}

    overall_scores = [r[0] for r in rows]
    relevances = [r[1] for r in rows]
    hallucinations = [r[2] for r in rows]
    completeness = [r[3] for r in rows]
    latencies = [r[4] for r in rows]
    passed_flags = [r[5] for r in rows]
    failure_reasons = [r[6] for r in rows if r[6]]

    most_common_failure: Optional[str] = None
    if failure_reasons:
        most_common_failure = Counter(failure_reasons).most_common(1)[0][0]

    return {
        "count": len(rows),
        "avg_overall": statistics.mean(overall_scores),
        "avg_relevance": statistics.mean(relevances),
        "avg_hallucination": statistics.mean(hallucinations),
        "avg_completeness": statistics.mean(completeness),
        "p95_latency": _percentile(latencies, 95),
        "pass_rate": sum(passed_flags) / len(passed_flags),
        "most_common_failure": most_common_failure,
    }
