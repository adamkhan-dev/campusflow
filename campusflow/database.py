"""SQLite persistence and lightweight Python TF-IDF/keyword retrieval for CAMPUSFLOW."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "campusflow.db"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "ka",
    "ke",
    "ki",
    "ko",
    "mein",
    "of",
    "on",
    "or",
    "se",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve explicit db_path, CAMPUSFLOW_DB_PATH env override, or default SQLite file."""
    if db_path is not None:
        return Path(db_path)
    env_path = os.getenv("CAMPUSFLOW_DB_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_DB_PATH


@contextmanager
def _get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(resolve_db_path(db_path)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Create the CAMPUSFLOW SQLite schema if it does not already exist."""
    db_file = resolve_db_path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection(db_file) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lectures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                lecture_title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_reference TEXT NOT NULL,
                is_uncertain INTEGER NOT NULL DEFAULT 0,
                uncertainty_reason TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS revision_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id INTEGER NOT NULL,
                item_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS lecture_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                feedback_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS student_schedule_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_key TEXT NOT NULL DEFAULT 'local-demo',
                course_code TEXT NOT NULL,
                course_name TEXT NOT NULL DEFAULT '',
                section TEXT NOT NULL,
                day TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                room TEXT NOT NULL DEFAULT '',
                instructor TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS timetable_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                course_name TEXT NOT NULL DEFAULT '',
                section TEXT NOT NULL DEFAULT 'BCS-5A',
                day TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                room TEXT NOT NULL DEFAULT '',
                instructor TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS uploaded_timetable_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_key TEXT NOT NULL DEFAULT 'local-demo',
                file_name TEXT NOT NULL DEFAULT '',
                course_code TEXT NOT NULL,
                course_name TEXT NOT NULL DEFAULT '',
                section TEXT NOT NULL,
                day TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                room TEXT NOT NULL DEFAULT '',
                instructor TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS student_planner_state (
                student_key TEXT PRIMARY KEY,
                program TEXT NOT NULL DEFAULT 'BCS',
                section_code TEXT NOT NULL DEFAULT '1A',
                pdf_filename TEXT NOT NULL DEFAULT '',
                state_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            """
        )
        # Lightweight forward migrations for existing SQLite files
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(knowledge_chunks);").fetchall()
        }
        if "uncertainty_reason" not in cols:
            conn.execute(
                "ALTER TABLE knowledge_chunks ADD COLUMN uncertainty_reason TEXT NOT NULL DEFAULT '';"
            )
        tt_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(timetable_slots);").fetchall()
        }
        if "instructor" not in tt_cols:
            conn.execute(
                "ALTER TABLE timetable_slots ADD COLUMN instructor TEXT NOT NULL DEFAULT '';"
            )
        conn.commit()


def save_lecture(
    course_code: str,
    lecture_title: str,
    summary_data: dict[str, Any],
    chunks: list[dict[str, Any]],
    db_path: Path | str | None = None,
) -> int:
    """Persist a lecture (replacing prior captures for the same course) and its chunks."""
    init_db(db_path)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_json = json.dumps(summary_data, ensure_ascii=False)
    clean_code = course_code.strip() or "GENERAL"

    with _get_connection(db_path) as conn:
        prior_rows = conn.execute(
            "SELECT id FROM lectures WHERE UPPER(course_code) = ?",
            (clean_code.upper(),),
        ).fetchall()
        prior_ids = [int(r["id"]) for r in prior_rows]
        if prior_ids:
            placeholders = ",".join("?" for _ in prior_ids)
            conn.execute(
                f"DELETE FROM revision_items WHERE lecture_id IN ({placeholders})",
                prior_ids,
            )
            conn.execute(
                f"DELETE FROM knowledge_chunks WHERE lecture_id IN ({placeholders})",
                prior_ids,
            )
            conn.execute(
                f"DELETE FROM lectures WHERE id IN ({placeholders})",
                prior_ids,
            )

        cursor = conn.execute(
            """
            INSERT INTO lectures (course_code, lecture_title, created_at, summary_json)
            VALUES (?, ?, ?, ?)
            """,
            (clean_code, lecture_title.strip(), created_at, summary_json),
        )
        lecture_id = int(cursor.lastrowid)

        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO knowledge_chunks (
                    lecture_id,
                    topic,
                    content,
                    source_type,
                    source_reference,
                    is_uncertain,
                    uncertainty_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lecture_id,
                    str(chunk.get("topic", "General")).strip() or "General",
                    str(chunk.get("content", "")).strip(),
                    str(chunk.get("source_type", "SYNTHESIZED")).strip() or "SYNTHESIZED",
                    str(chunk.get("source_reference", "Unmarked in source")).strip()
                    or "Unmarked in source",
                    1 if chunk.get("is_uncertain", False) else 0,
                    str(chunk.get("uncertainty_reason", "")).strip(),
                ),
            )
        conn.commit()
        return lecture_id


def list_lectures(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all stored lectures ordered newest first, with chunk and uncertainty counts."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                l.id,
                l.course_code,
                l.lecture_title,
                l.created_at,
                l.summary_json,
                COUNT(k.chunk_id) AS chunk_count,
                COALESCE(SUM(k.is_uncertain), 0) AS uncertain_count
            FROM lectures l
            LEFT JOIN knowledge_chunks k ON l.id = k.lecture_id
            GROUP BY l.id
            ORDER BY l.id DESC
            """
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["summary"] = json.loads(item["summary_json"])
        results.append(item)
    return results


def get_lecture(
    lecture_id: int, db_path: Path | str | None = None
) -> dict[str, Any] | None:
    """Fetch a single lecture record by ID."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, course_code, lecture_title, created_at, summary_json
            FROM lectures
            WHERE id = ?
            """,
            (lecture_id,),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["summary"] = json.loads(data["summary_json"])
    return data


def get_lecture_chunks(
    lecture_id: int,
    only_uncertain: bool = False,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Fetch all knowledge chunks belonging to a lecture."""
    init_db(db_path)
    query = """
        SELECT chunk_id, lecture_id, topic, content, source_type, source_reference,
               is_uncertain, uncertainty_reason
        FROM knowledge_chunks
        WHERE lecture_id = ?
    """
    params: list[Any] = [lecture_id]
    if only_uncertain:
        query += " AND is_uncertain = 1"
    query += " ORDER BY chunk_id ASC"

    with _get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {**dict(row), "is_uncertain": bool(row["is_uncertain"])}
        for row in rows
    ]


def verify_uncertain_chunk(
    chunk_id: int,
    corrected_content: str,
    db_path: Path | str | None = None,
) -> bool:
    """Allow human review to confirm or correct an uncertain chunk and clear the warning."""
    init_db(db_path)
    cleaned = corrected_content.strip()
    if not cleaned:
        return False
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE knowledge_chunks
            SET content = ?, is_uncertain = 0, uncertainty_reason = ''
            WHERE chunk_id = ?
            """,
            (cleaned, chunk_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def save_revision_items(
    lecture_id: int,
    item_type: str,
    items: list[dict[str, Any]],
    replace_existing: bool = True,
    db_path: Path | str | None = None,
) -> None:
    """Store generated MCQs or flashcards in the revision_items table."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        if replace_existing:
            conn.execute(
                "DELETE FROM revision_items WHERE lecture_id = ? AND item_type = ?",
                (lecture_id, item_type),
            )
        for item in items:
            conn.execute(
                """
                INSERT INTO revision_items (lecture_id, item_type, payload_json)
                VALUES (?, ?, ?)
                """,
                (lecture_id, item_type, json.dumps(item, ensure_ascii=False)),
            )
        conn.commit()


def get_revision_items(
    lecture_id: int,
    item_type: str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load saved revision items (MCQs or flashcards) for a lecture."""
    init_db(db_path)
    query = """
        SELECT id, lecture_id, item_type, payload_json
        FROM revision_items
        WHERE lecture_id = ?
    """
    params: list[Any] = [lecture_id]
    if item_type is not None:
        query += " AND item_type = ?"
        params.append(item_type)
    query += " ORDER BY id ASC"

    with _get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "id": row["id"],
            "lecture_id": row["lecture_id"],
            "item_type": row["item_type"],
            "payload": json.loads(row["payload_json"]),
        }
        for row in rows
    ]


def delete_lecture(lecture_id: int, db_path: Path | str | None = None) -> bool:
    """Delete a lecture and cascade-delete its chunks, revision items, and feedback."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM lectures WHERE id = ?", (lecture_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_lecture_notes(
    lecture_id: int,
    summary_data: dict[str, Any],
    chunks: list[dict[str, Any]] | None = None,
    db_path: Path | str | None = None,
) -> bool:
    """Update stored summary_json (and optionally replace chunks) for an existing lecture ID."""
    init_db(db_path)
    summary_json = json.dumps(summary_data, ensure_ascii=False)
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE lectures SET summary_json = ? WHERE id = ?",
            (summary_json, lecture_id),
        )
        if cursor.rowcount == 0:
            return False
        if chunks is not None:
            conn.execute("DELETE FROM knowledge_chunks WHERE lecture_id = ?", (lecture_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        lecture_id,
                        topic,
                        content,
                        source_type,
                        source_reference,
                        is_uncertain,
                        uncertainty_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lecture_id,
                        str(chunk.get("topic", "General")).strip() or "General",
                        str(chunk.get("content", "")).strip(),
                        str(chunk.get("source_type", "SYNTHESIZED")).strip() or "SYNTHESIZED",
                        str(chunk.get("source_reference", "Unmarked in source")).strip()
                        or "Unmarked in source",
                        1 if chunk.get("is_uncertain", False) else 0,
                        str(chunk.get("uncertainty_reason", "")).strip(),
                    ),
                )
        conn.commit()
        return True


def save_lecture_feedback(
    lecture_id: int,
    rating: str,
    feedback_text: str = "",
    db_path: Path | str | None = None,
) -> int:
    """Persist positive ('good') or negative ('bad') user feedback for a lecture's notes."""
    init_db(db_path)
    clean_rating = rating.strip().lower()
    if clean_rating not in {"good", "bad"}:
        raise ValueError("Feedback rating must be 'good' or 'bad'.")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO lecture_feedback (lecture_id, rating, feedback_text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (lecture_id, clean_rating, feedback_text.strip(), created_at),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_latest_lecture_feedback(
    lecture_id: int,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return the most recent feedback record for a lecture, if any."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, lecture_id, rating, feedback_text, created_at
            FROM lecture_feedback
            WHERE lecture_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (lecture_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized alphanumeric terms, dropping common stopwords."""
    raw_tokens = re.findall(r"[a-zA-Z0-9_+\-/]+", text.lower())
    return [tok for tok in raw_tokens if len(tok) > 1 and tok not in _STOPWORDS]


def retrieve_relevant_chunks(
    query: str,
    lecture_id: int | None = None,
    top_k: int = 5,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Lightweight pure-Python TF-IDF + keyword retrieval over stored SQLite chunks.

    Returns up to `top_k` matching chunks sorted by relevance score without any
    external vector database. Only chunks with positive overlap are returned.
    """
    init_db(db_path)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    sql = """
        SELECT
            k.chunk_id,
            k.lecture_id,
            l.course_code,
            l.lecture_title,
            k.topic,
            k.content,
            k.source_type,
            k.source_reference,
            k.is_uncertain,
            k.uncertainty_reason
        FROM knowledge_chunks k
        JOIN lectures l ON k.lecture_id = l.id
    """
    params: list[Any] = []
    if lecture_id is not None:
        sql += " WHERE k.lecture_id = ?"
        params.append(lecture_id)
    sql += " ORDER BY k.chunk_id ASC"

    with _get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    docs: list[dict[str, Any]] = []
    doc_freq: Counter[str] = Counter()
    for row in rows:
        combined_text = f"{row['topic']} {row['topic']} {row['content']}"
        tokens = _tokenize(combined_text)
        term_counts = Counter(tokens)
        for term in term_counts:
            doc_freq[term] += 1
        docs.append(
            {
                **dict(row),
                "is_uncertain": bool(row["is_uncertain"]),
                "_tokens": tokens,
                "_counts": term_counts,
            }
        )

    num_docs = len(docs)
    query_counts = Counter(query_tokens)

    def _idf(term: str) -> float:
        return math.log((num_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1.0

    # Build normalized TF-IDF vector weights for the query.
    query_vec: dict[str, float] = {}
    for term, tf in query_counts.items():
        if term in doc_freq:
            query_vec[term] = (1.0 + math.log(tf)) * _idf(term)

    query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
    normalized_query = query.strip().lower()

    scored: list[dict[str, Any]] = []
    for doc in docs:
        counts: Counter[str] = doc["_counts"]
        if not counts:
            continue

        # Cosine similarity over TF-IDF weights.
        dot = 0.0
        doc_sq = 0.0
        for term, tf in counts.items():
            weight = (1.0 + math.log(tf)) * _idf(term)
            doc_sq += weight * weight
            if term in query_vec:
                dot += weight * query_vec[term]

        doc_norm = math.sqrt(doc_sq)
        cosine = (dot / (query_norm * doc_norm)) if (query_norm > 0 and doc_norm > 0) else 0.0

        # Small topic match & exact phrase bonus for crisp short-query precision.
        topic_tokens = set(_tokenize(doc["topic"]))
        topic_overlap = sum(1 for t in query_counts if t in topic_tokens)
        phrase_bonus = 0.15 if normalized_query and normalized_query in doc["content"].lower() else 0.0
        score = cosine + (0.12 * topic_overlap) + phrase_bonus

        if score > 0:
            clean_doc = {k: v for k, v in doc.items() if not k.startswith("_")}
            clean_doc["score"] = round(score, 4)
            clean_doc["source_badge"] = f"[{clean_doc['source_type']} @ {clean_doc['source_reference']}]"
            scored.append(clean_doc)

    scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Timetable SQLite Persistence & Startup Seeding
# ---------------------------------------------------------------------------

DEFAULT_TIMETABLE_SEED: list[tuple[str, str, str, str, str, str, str, str]] = [
    ("CSC322", "Operating Systems", "BCS-5A", "Mon", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
    ("CSC322", "Operating Systems", "BCS-5A", "Wed", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
    ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Mon", "08:30", "10:00", "106 AB-2", "Ms. Ayesha Hussain"),
    ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Thu", "11:30", "13:00", "106 AB-2", "Ms. Ayesha Hussain"),
    ("CSC325", "Computer Organization & Assembly", "BCS-5A", "Tue", "10:00", "11:30", "CL-6", "Mr. Khurram Iqbal"),
    ("CSC325", "Computer Organization & Assembly", "BCS-5A", "Thu", "10:00", "11:30", "CL-6", "Mr. Khurram Iqbal"),
    ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Wed", "11:30", "13:00", "107 AB-2", "Mr. Rizwan Rashid"),
    ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Fri", "10:00", "11:30", "107 AB-2", "Mr. Rizwan Rashid"),
]

_LEGACY_SEED_TITLES = (
    "Master Theorem & Recurrence Relations",
    "x86 Addressing Modes & Registers",
)


def list_timetable_slots(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all saved class slots ordered by weekday and start time."""
    init_db(db_path)
    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5}
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, course_code, course_name, section, day, start_time, end_time, room, instructor
            FROM timetable_slots
            ORDER BY id ASC
            """
        ).fetchall()
    items = [dict(row) for row in rows]
    items.sort(key=lambda r: (day_order.get(r["day"], 99), r["start_time"], r["course_code"]))
    return items


def add_timetable_slot(
    course_code: str,
    course_name: str,
    day: str,
    start_time: str,
    end_time: str,
    room: str,
    section: str = "BCS-5A",
    instructor: str = "",
    db_path: Path | str | None = None,
) -> int:
    """Insert a class slot into timetable_slots and return its ID."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO timetable_slots (course_code, course_name, section, day, start_time, end_time, room, instructor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                course_code.strip().upper(),
                course_name.strip(),
                section.strip() or "BCS-5A",
                day.strip(),
                start_time.strip(),
                end_time.strip(),
                room.strip() or "TBA",
                instructor.strip() or "TBA",
            ),
        )
        return int(cursor.lastrowid)


def replace_course_slots(
    course_code: str,
    replacement_slots: list[tuple[Any, ...]],
    db_path: Path | str | None = None,
) -> None:
    """Atomically swap all saved slots for a course_code with its alternative section slots."""
    init_db(db_path)
    target_code = course_code.strip().upper()
    normalized_rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for row in replacement_slots:
        if len(row) == 7:
            normalized_rows.append((*row, ""))
        else:
            normalized_rows.append(tuple(str(x) for x in row[:8]))  # type: ignore[arg-type]

    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM timetable_slots WHERE UPPER(course_code) = ?", (target_code,))
        for row in normalized_rows:
            conn.execute(
                """
                INSERT INTO timetable_slots (course_code, course_name, section, day, start_time, end_time, room, instructor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )


def delete_timetable_slot(slot_id: int, db_path: Path | str | None = None) -> bool:
    """Remove a class slot by ID."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM timetable_slots WHERE id = ?", (slot_id,))
        return cursor.rowcount > 0


def reset_timetable_slots(db_path: Path | str | None = None) -> None:
    """Restore the default Fall 2026 COMSATS timetable seed."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM timetable_slots")
        for row in DEFAULT_TIMETABLE_SEED:
            conn.execute(
                """
                INSERT INTO timetable_slots (course_code, course_name, section, day, start_time, end_time, room, instructor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )


def set_timetable_slots(
    slots_rows: list[tuple[Any, ...]],
    db_path: Path | str | None = None,
) -> None:
    """Atomically replace all rows in timetable_slots with a finalized personalized schedule."""
    init_db(db_path)
    normalized_rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for row in slots_rows:
        if len(row) == 7:
            normalized_rows.append((*tuple(str(x) for x in row), "TBA"))  # type: ignore[arg-type]
        else:
            normalized_rows.append(tuple(str(x) for x in row[:8]))  # type: ignore[arg-type]
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM timetable_slots")
        for row in normalized_rows:
            conn.execute(
                """
                INSERT INTO timetable_slots (course_code, course_name, section, day, start_time, end_time, room, instructor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )


def clear_all_lectures_and_schedule(db_path: Path | str | None = None) -> None:
    """Clear all rows via SQL DELETE (safe under Windows file locks) without dropping schema."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM lecture_feedback")
        conn.execute("DELETE FROM revision_items")
        conn.execute("DELETE FROM knowledge_chunks")
        conn.execute("DELETE FROM lectures")
        conn.execute("DELETE FROM timetable_slots")


def reset_demo_database(db_path: Path | str | None = None) -> None:
    """Wipe existing rows via SQL DELETE and re-seed all 4 CS subjects and schedule."""
    clear_all_lectures_and_schedule(db_path=db_path)
    seed_initial_data(db_path=db_path)


def _seed_csc301_binary_search(db_path: Path | str | None = None) -> int:
    """Seed CSC301 — Design & Analysis of Algorithms (Binary Search & Recurrence Complexity)."""
    summary = {
        "generation_source": "demo_kit",
        "executive_summary": (
            "Binary Search locates a target inside a sorted array by checking the middle item and cutting "
            "the remaining search interval in half at every step. Using the overflow-safe midpoint equation "
            "avoids 32-bit integer overflow on huge arrays, achieving O(log n) worst-case runtime and "
            "needing only O(1) extra memory iteratively."
        ),
        "structured_notes": [
            {
                "heading": "Sorted Precondition & Master Theorem Recurrence",
                "source_type": "SLIDES_PDF",
                "formal_definition": (
                    "Binary Search is a divide-and-conquer search algorithm over a sorted ascending array A[0..n−1] "
                    "that inspects the midpoint element A[mid] and halves the candidate interval [low..high] on every step."
                ),
                "detailed_explanation": (
                    "Because ascending order ensures every element left of mid is ≤ A[mid] and every element right is ≥ A[mid], "
                    "one scalar comparison conclusively rules out half of the remaining items.\n\n"
                    "Recurrence T(n) = T(n ÷ 2) + constant satisfies Master Theorem Case 2 with a = 1, b = 2, n^(log₂ 1) = 1, "
                    "resolving tightly to logarithmic Θ(log n) worst-case runtime."
                ),
                "formula_and_breakdown": (
                    "T(n) = T(n ÷ 2) + c  →  Θ(log n) time\n\n"
                    "• a = 1 — Only one recursive half is explored per comparison.\n"
                    "• b = 2 — Candidate interval shrinks by factor 2 each step.\n"
                    "• Invariant — If target exists, its index always stays inside [low..high]."
                ),
                "worked_example": (
                    "Comparing scale: For n = 1,024 sorted keys, T(1024) halves 10 times (1024→512→…→1), "
                    "finishing in at most ⌊log₂ 1024⌋ + 1 = 11 comparisons instead of 1,024 linear checks."
                ),
                "exam_takeaways": [
                    "Binary Search requires sorted ascending order before probing midpoints.",
                    "Unsuccessful searches terminate cleanly once low crosses above high.",
                ],
                "takeaway": "Recurrence Bound: T(n) = T(n ÷ 2) + constant → Θ(log n) time",
            },
            {
                "heading": "Worked Probe Trace & Overflow-Safe Midpoint",
                "source_type": "WHITEBOARD_IMAGE",
                "formal_definition": (
                    "Overflow-safe midpoint arithmetic computes mid = low + ⌊(high − low) ÷ 2⌋ so intermediate addition "
                    "never exceeds upper index bound `high` inside signed 32-bit registers."
                ),
                "detailed_explanation": (
                    "Tracing on concrete sorted arrays verifies loop boundary updates (mid + 1 vs mid − 1) so no infinite "
                    "loop occurs on 1- or 2-element windows.\n\n"
                    "Subtracting (high − low) first ensures intermediate arithmetic never overflows signed 32-bit integers past 2³¹ − 1."
                ),
                "formula_and_breakdown": (
                    "midpoint = low + (high − low) ÷ 2\n"
                    "A = [3, 8, 14, 21, 29, 37, 45], Target = 37\n\n"
                    "• Update Left — If A[mid] < target: set low = mid + 1.\n"
                    "• Update Right — If A[mid] > target: set high = mid − 1."
                ),
                "worked_example": (
                    "Step-by-Step Board Trace (Target = 37):\n"
                    "1. Probe 1: low = 0, high = 6 → mid = 0 + (6 − 0) ÷ 2 = 3. A[3] = 21 < 37 → low = 4.\n"
                    "2. Probe 2: low = 4, high = 6 → mid = 4 + (6 − 4) ÷ 2 = 5. A[5] = 37 == Target → return index 5."
                ),
                "exam_takeaways": [
                    "Found target 37 at index 5 in 2 probes vs 6 under sequential linear scan.",
                    "Always shift low = mid + 1 or high = mid − 1 to guarantee interval shrinkage.",
                ],
                "takeaway": "Safe Midpoint Rule: midpoint = low + (high − low) ÷ 2",
            },
            {
                "heading": "Worst-Case Comparison Bound & O(1) Iterative Space",
                "source_type": "AUDIO",
                "formal_definition": (
                    "Iterative Binary Search bounds worst-case probe count at ⌊log₂ n⌋ + 1 comparisons while using strictly "
                    "constant O(1) auxiliary memory for scalar index pointers."
                ),
                "detailed_explanation": (
                    "Doubling the sorted array size adds at most one single extra midpoint check, giving logarithmic efficiency "
                    "even across billions of database keys.\n\n"
                    "Writing the algorithm with an iterative while loop eliminates the O(log n) function call-stack overhead "
                    "incurred by naive recursive implementations."
                ),
                "formula_and_breakdown": (
                    "Max Probes = ⌊log₂ n⌋ + 1  |  Iterative Space = O(1)  |  Recursive Space = O(log n)\n\n"
                    "• ⌊log₂ n⌋ + 1 — Worst-case unsuccessful comparison count before low > high.\n"
                    "• Iterative Pointers — Uses only low, high, and mid scalar registers."
                ),
                "worked_example": (
                    "With INT_MAX = 2,147,483,647, low = 1,500,000,000, high = 2,000,000,000:\n"
                    "• Naive (low + high) = 3.5 billion → Signed 32-bit overflow!\n"
                    "• Safe low + (high − low) ÷ 2 = 1.5B + 250M = 1,750,000,000 (zero overflow)."
                ),
                "exam_takeaways": [
                    "1,024 sorted elements finish in at most 11 comparisons instead of 1,024 linear checks.",
                    "Prefer iterative loops in production systems code for strict O(1) stack footprint.",
                ],
                "takeaway": "Key Insight: Iterative implementation keeps space strictly O(1) and worst-case probes ≤ ⌊log₂ n⌋ + 1.",
            },
        ],
        "extracted_pseudocode": (
            "def binary_search(A, target):\n"
            "    low, high = 0, len(A) - 1\n"
            "    while low <= high:\n"
            "        mid = low + (high - low) // 2   # prevents integer overflow\n"
            "        if A[mid] == target:\n"
            "            return mid\n"
            "        elif A[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n\n"
            "Complexity: T(n) = T(n ÷ 2) + c  →  Θ(log n) time, O(1) space"
        ),
    }
    chunks = [
        {
            "topic": "Binary Search Sorted Precondition",
            "content": "Binary Search requires the input array A to be sorted in ascending order before probing midpoints.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 1",
            "is_uncertain": False,
        },
        {
            "topic": "Overflow-Safe Midpoint Formula",
            "content": "Overflow-safe midpoint equation smudged — inferred as midpoint = low + (high − low) ÷ 2",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": True,
            "uncertainty_reason": (
                "Right half of midpoint formula (high − low) ÷ 2 was smeared on the whiteboard; "
                "confirmed via audio explanation preventing 32-bit overflow."
            ),
        },
        {
            "topic": "Integer Overflow Prevention",
            "content": "Adding low + high directly can overflow 32-bit signed integers past 2³¹ − 1, whereas low + (high − low) ÷ 2 stays safely within bounds.",
            "source_type": "AUDIO",
            "source_reference": "00:34",
            "is_uncertain": False,
        },
        {
            "topic": "Logarithmic Recurrence T(n)",
            "content": "Each probe discards half of the array in constant time, solving by Master Theorem Case 2 to Θ(log n) comparisons.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 2",
            "is_uncertain": False,
        },
        {
            "topic": "Worked Trace on Sample Array",
            "content": "Tracing [3, 8, 14, 21, 29, 37, 45] for target 37: Probe 1 checks index 3 (21 < 37 → low = 4); Probe 2 checks index 5 (37 matches target, found).",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": False,
        },
        {
            "topic": "Worst-Case Comparison Bound",
            "content": "Binary search performs at most ⌊log₂ n⌋ + 1 comparisons, requiring O(1) auxiliary space when written iteratively.",
            "source_type": "AUDIO",
            "source_reference": "00:55",
            "is_uncertain": False,
        },
    ]
    lec_id = save_lecture("CSC301", "Binary Search & Recurrence Complexity", summary, chunks, db_path=db_path)
    mcqs = [
        {"question": "What precondition must hold on array A before running Binary Search?", "options": ["Elements must be unique primes", "Elements must be sorted", "Length must be a power of 2", "Array must be linked-list backed"], "correct_index": 1, "explanation": "Binary Search eliminates half of the search range relying on sorted order.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "Why prefer low + (high − low) ÷ 2 over (low + high) ÷ 2?", "options": ["It avoids signed integer overflow on large indices", "It lowers time complexity from O(log n) to O(1)", "It works on unsorted arrays", "It eliminates the while loop"], "correct_index": 0, "explanation": "Adding low + high can exceed INT_MAX; subtracting first keeps intermediate arithmetic ≤ high.", "source_badge": "[AUDIO @ 00:34]"},
        {"question": "Which recurrence describes Binary Search runtime?", "options": ["T(n) = 2·T(n ÷ 2) + n", "T(n) = T(n − 1) + c", "T(n) = T(n ÷ 2) + c", "T(n) = 4·T(n ÷ 2) + c"], "correct_index": 2, "explanation": "One half-sized recursive subproblem plus O(1) comparison gives T(n) = T(n ÷ 2) + c.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "Which Master Theorem case solves T(n) = T(n ÷ 2) + Θ(1) to Θ(log n)?", "options": ["Case 1", "Case 2 (balanced work with a = 1, b = 2)", "Case 3", "Does not fit any case"], "correct_index": 1, "explanation": "Here n^(log₂ 1) = n⁰ = 1 matches constant work Θ(1), fitting Case 2.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "Tracing A = [3, 8, 14, 21, 29, 37, 45] for target 37, which index is probed FIRST?", "options": ["mid = 2 (14)", "mid = 3 (21)", "mid = 4 (29)", "mid = 5 (37)"], "correct_index": 1, "explanation": "Initially low = 0, high = 6, so mid = 0 + (6 − 0) ÷ 2 = 3 (A[3] = 21).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "After probing mid = 3 (21 < 37) on [3, 8, 14, 21, 29, 37, 45], what is the updated search window?", "options": ["low = 0, high = 2", "low = 3, high = 6", "low = 4, high = 6", "low = 5, high = 6"], "correct_index": 2, "explanation": "Since A[mid] < target, we update low = mid + 1 = 4 while high stays 6.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "How many probes does Binary Search take to find 37 in [3, 8, 14, 21, 29, 37, 45]?", "options": ["1 probe", "2 probes", "4 probes", "6 probes"], "correct_index": 1, "explanation": "Step 1 checks index 3 (21); Step 2 checks index 5 (37 == 37).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "What is the auxiliary space complexity of iterative Binary Search?", "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "correct_index": 0, "explanation": "Iterative Binary Search uses only constant scalar variables (low, high, mid).", "source_badge": "[AUDIO @ 00:55]"},
        {"question": "What is the maximum number of comparisons for n = 1,024 sorted items?", "options": ["5", "11", "512", "1,024"], "correct_index": 1, "explanation": "Worst-case probes = ⌊log₂ 1024⌋ + 1 = 10 + 1 = 11.", "source_badge": "[AUDIO @ 00:55]"},
        {"question": "When searching terminates unsuccessfully (low > high), what happened to the interval?", "options": ["Target appeared twice", "The remaining candidate interval became empty", "low overflowed", "Array became unsorted"], "correct_index": 1, "explanation": "Crossing low > high proves zero candidate indices remain containing target.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
    ]
    flashcards = [
        {"front": "Required precondition for Binary Search?", "back": "Input array A must already be sorted in ascending order.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "Write the overflow-safe midpoint equation.", "back": "midpoint = low + (high − low) ÷ 2.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"front": "Why avoid (low + high) ÷ 2 on large arrays?", "back": "Adding low + high can exceed 32-bit signed INT_MAX and wrap negative.", "source_badge": "[AUDIO @ 00:34]"},
        {"front": "State the recurrence and Big-Theta bound of Binary Search.", "back": "T(n) = T(n ÷ 2) + c, solving to Θ(log n) by Master Theorem Case 2.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"front": "Compare space complexity: iterative vs recursive Binary Search.", "back": "Iterative uses O(1) space; recursive uses O(log n) call stack frames.", "source_badge": "[AUDIO @ 00:55]"},
        {"front": "Step trace for target 37 in [3, 8, 14, 21, 29, 37, 45]?", "back": "Probe 1: mid = 3 (21 < 37 → low = 4). Probe 2: mid = 5 (37 matches target).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
    ]
    save_revision_items(lec_id, "mcq", mcqs, db_path=db_path)
    save_revision_items(lec_id, "flashcard", flashcards, db_path=db_path)
    return lec_id


def _seed_aic354_naive_bayes(db_path: Path | str | None = None) -> int:
    """Seed AIC354 — Machine Learning Fundamentals (Naive Bayes & Bayesian Classification)."""
    summary = {
        "generation_source": "demo_kit",
        "executive_summary": (
            "Naive Bayes classifies documents or data points using Bayes' Theorem under the simplifying "
            "assumption that features act independently once the class label is known. Predictions add log "
            "probabilities instead of multiplying tiny decimals to prevent floating-point underflow, while "
            "Laplace (Add-1) Smoothing ensures unseen words never zero out an entire class score."
        ),
        "structured_notes": [
            {
                "heading": "Conditional Independence & Zero-Frequency Collapse",
                "source_type": "SLIDES_PDF",
                "formal_definition": (
                    "The Naive Bayes classifier applies Bayes' Theorem under the conditional independence assumption "
                    "that every observed feature word xᵢ occurs independently given the known class label y."
                ),
                "detailed_explanation": (
                    "Estimating the full joint distribution over V vocabulary words simultaneously requires exponential "
                    "training samples. Assuming conditional independence collapses that exponential table into a compact "
                    "product of single-word empirical frequencies computed in O(n × C) linear time.\n\n"
                    "However, if an unseen test token appears with training frequency 0, multiplying by zero collapses the "
                    "entire class product to 0 regardless of how strongly every other word supports that class."
                ),
                "formula_and_breakdown": (
                    "P(y | x₁,…,xₙ) ∝ P(y) × Πᵢ P(xᵢ | y)\n\n"
                    "• P(y) — Class Prior: Training documents in class y ÷ Total training documents.\n"
                    "• P(xᵢ | y) — Conditional Likelihood: Estimated probability of feature word xᵢ inside class y.\n"
                    "• Zero-Frequency Failure: If any P(xᵢ | y) = 0, the entire product vanishes to 0."
                ),
                "worked_example": (
                    "Suppose P(Spam) = 0.4, P('offer'|Spam) = 0.05, P('winner'|Spam) = 0.04, unseen P('crypto'|Spam) = 0:\n"
                    "Unsmoothed product = 0.4 × 0.05 × 0.04 × 0 = 0 (positive evidence erased cleanly by one word)."
                ),
                "exam_takeaways": [
                    "Marginal evidence P(X) cancels out when comparing candidate classes.",
                    "Unsmoothed likelihood products collapse completely whenever a test word never appeared in class y.",
                ],
                "takeaway": "Core Assumption: Given class y, word likelihoods multiply cleanly in O(n × C) time.",
            },
            {
                "heading": "Log-Space Decision & Vocabulary Size Normalization",
                "source_type": "AUDIO",
                "formal_definition": (
                    "Log-space Naive Bayes classification selects the Maximum A Posteriori class by maximizing "
                    "the sum of log class prior plus summed log feature likelihoods, stabilized via Add-1 smoothing."
                ),
                "detailed_explanation": (
                    "Multiplying dozens of small decimals like 0.001 × 0.002 quickly underflows standard 64-bit IEEE 754 "
                    "precision toward zero. Because logarithm increases monotonically, summing logs preserves exact class rankings.\n\n"
                    "Adding +1 to each of the V unique word counts injects V extra total mass into the numerator, so "
                    "adding Vocabulary Size V to the denominator ensures every smoothed word probability sums cleanly to 1."
                ),
                "formula_and_breakdown": (
                    "ŷ = argmax_y [ log P(y) + Σᵢ log P(featureᵢ | y) ]\n"
                    "P(word | class) = (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)\n\n"
                    "• log P(y) — Log Prior avoiding floating-point underflow.\n"
                    "• +1 — Add-1 Laplace pseudocount eliminating zero probabilities.\n"
                    "• + V — Vocabulary Size denominator normalization keeping probabilities summing to 1."
                ),
                "worked_example": (
                    "Why + V in denominator: Suppose V = 3 unique words in a class totaling N = 10 occurrences (5, 3, 2):\n"
                    "Sum of smoothed weights = (5+1)/(10+3) + (3+1)/(10+3) + (2+1)/(10+3) = 13/13 = 1.0."
                ),
                "exam_takeaways": [
                    "Adding log probabilities turns tiny products into numerically stable sums.",
                    "Vocabulary Size V counts unique distinct tokens across the entire training corpus.",
                ],
                "takeaway": "Laplace Rule: (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)",
            },
            {
                "heading": "Worked Spam Filter Trace & Laplace Denominator",
                "source_type": "WHITEBOARD_IMAGE",
                "formal_definition": (
                    "Bayes' Decision Rule with Add-1 Laplace smoothing assigns strictly non-zero conditional likelihoods "
                    "(Count + 1) ÷ (Class Words + V) to both seen and unseen vocabulary tokens."
                ),
                "detailed_explanation": (
                    "On the whiteboard spam-filter problem, V = 1,000 unique vocabulary tokens and Spam training "
                    "emails contain 200 total word positions. Without smoothing, an unseen token receives 0 ÷ 200 = 0.\n\n"
                    "Applying Laplace Add-1 raises the denominator to 200 + 1,000 = 1,200, giving seen keywords proportional "
                    "boosts while preserving a safe positive residual for brand-new test words."
                ),
                "formula_and_breakdown": (
                    "P(word | Spam) = (Word Count + 1) ÷ (Total Spam Words + Vocabulary Size V)\n\n"
                    "• Total Spam Words = 200  ·  Vocabulary Size V = 1,000  ·  Smoothed Denominator = 1,200.\n"
                    "• P('prize' | Spam): Seen count = 5 → (5 + 1) ÷ 1,200.\n"
                    "• P('urgent' | Spam): Unseen count = 0 → (0 + 1) ÷ 1,200."
                ),
                "worked_example": (
                    "Step-by-Step Board Calculation (V = 1,000, Spam Words = 200):\n"
                    "1. Seen word 'prize' (count = 5): (5 + 1) ÷ (200 + 1,000) = 6 ÷ 1,200 = 0.005.\n"
                    "2. Unseen word 'urgent' (count = 0): (0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200 ≈ 0.00083."
                ),
                "exam_takeaways": [
                    "Denominator MUST equal (Total Words in Class + Vocabulary Size V), never just Total Words + 1.",
                    "Unseen words always receive 1 ÷ (Total Words in Class + V) under Add-1 smoothing.",
                ],
                "takeaway": "Board Result: Smoothed denominator (Total Words + V = 1,200) prevents unseen tokens from collapsing predictions.",
            },
        ],
        "extracted_pseudocode": (
            "Log-Space Prediction Rule:\n"
            "  Pick class y maximizing:  log P(y) + Σ log P(featureᵢ | y)\n\n"
            "Laplace (Add-1) Smoothing Formula:\n"
            "  P(word | class) = (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)\n\n"
            "Spam Filter Verification (V = 1,000, Total Spam Words = 200):\n"
            "  • P('prize'  | Spam) = (5 + 1) ÷ (200 + 1,000) = 6 ÷ 1,200 = 0.005\n"
            "  • P('urgent' | Spam) = (0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200 ≈ 0.00083"
        ),
    }
    chunks = [
        {
            "topic": "Conditional Independence Assumption",
            "content": "Naive Bayes assumes all input features are conditionally independent given class y, reducing joint likelihood into a product of single-feature probabilities.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 1",
            "is_uncertain": False,
        },
        {
            "topic": "Log-Space Prediction Rule",
            "content": "Multiplying many small word probabilities causes floating-point underflow, so Naive Bayes picks class y with highest [ log P(y) + Σ log P(featureᵢ | y) ].",
            "source_type": "AUDIO",
            "source_reference": "00:22",
            "is_uncertain": False,
        },
        {
            "topic": "Zero-Frequency Collapse Problem",
            "content": "If any test word never appeared in class y during training (Word Count = 0), its raw likelihood is 0 and multiplies the entire class posterior down to zero.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 2",
            "is_uncertain": False,
        },
        {
            "topic": "Laplace Smoothing Denominator",
            "content": "Laplace smoothing denominator smudged — inferred as (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": True,
            "uncertainty_reason": (
                "Whiteboard smudge obscured '(Total Words in Class + Vocabulary Size V)' in the denominator; "
                "confirmed Vocabulary Size V keeps smoothed probabilities summing to 1."
            ),
        },
        {
            "topic": "Worked Spam Filter Calculation",
            "content": "With Vocabulary Size V = 1,000 and 200 total Spam words, seen word 'prize' (count = 5) gives (5 + 1) ÷ (200 + 1,000) = 0.005 and unseen 'urgent' (count = 0) gets 1 ÷ 1,200.",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": False,
        },
        {
            "topic": "Vocabulary Size Normalization",
            "content": "Adding Vocabulary Size V to the denominator ensures that summing all V smoothed word probabilities within a class still equals 1.",
            "source_type": "AUDIO",
            "source_reference": "00:46",
            "is_uncertain": False,
        },
    ]
    lec_id = save_lecture("AIC354", "Naive Bayes & Bayesian Belief Classification", summary, chunks, db_path=db_path)
    mcqs = [
        {"question": "Why is the Naive Bayes classifier called 'naive'?", "options": ["It ignores prior probabilities", "It assumes features are conditionally independent given class y", "It cannot classify binary targets", "It only uses linear regression weights"], "correct_index": 1, "explanation": "Conditional independence given class y simplifies joint feature likelihood into a product.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "Why convert products of feature probabilities into sums of logarithms during inference?", "options": ["To speed up sorting", "To prevent floating-point underflow from tiny decimals", "To enforce L1 regularization", "To remove class priors"], "correct_index": 1, "explanation": "Multiplying dozens of small decimals approaches zero in float precision; summing logs avoids underflow.", "source_badge": "[AUDIO @ 00:22]"},
        {"question": "What happens in unsmoothed Naive Bayes when a test email contains one unseen Spam word?", "options": ["That word is auto-deleted", "Likelihood 0 multiplies the entire Spam posterior to zero", "The prior P(Spam) doubles", "Accuracy increases automatically"], "correct_index": 1, "explanation": "A single zero term in the probability product collapses the entire posterior score for that class.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "Which formula computes Add-1 (Laplace) smoothing for a word in a class?", "options": ["(Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)", "Word Count ÷ (Total Words in Class + 1)", "(Word Count − 1) ÷ (Total Words − V)", "(Total Words + V) ÷ (Word Count + 1)"], "correct_index": 0, "explanation": "Adding 1 to each word count requires adding Vocabulary Size V to the denominator so probabilities sum to 1.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "Why add Vocabulary Size V in the Laplace smoothing denominator?", "options": ["To penalize rare classes", "So smoothed probabilities across all V words still sum to 1", "To maximize log loss", "To drop stop words"], "correct_index": 1, "explanation": "Adding 1 to each of the V numerator counts adds V total mass to the numerator.", "source_badge": "[AUDIO @ 00:46]"},
        {"question": "Given V = 1,000 and 200 Spam words, what is Laplace-smoothed P('prize' | Spam) when count = 5?", "options": ["5 ÷ 200", "6 ÷ 1,000", "6 ÷ 1,200 = 0.005", "5 ÷ 1,200"], "correct_index": 2, "explanation": "(5 + 1) ÷ (200 + 1,000) = 6 ÷ 1,200 = 0.005.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "Given V = 1,000 and 200 Spam words, what is Laplace-smoothed P('urgent' | Spam) if count = 0?", "options": ["0", "1 ÷ 200", "1 ÷ 1,200", "1 ÷ 1,000"], "correct_index": 2, "explanation": "(0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200, preventing zero-frequency collapse.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "In Bayes' Theorem for classification, which term cancels out when picking the top class y?", "options": ["Class prior P(y)", "Marginal evidence P(X)", "Feature likelihood P(feature | y)", "Vocabulary size V"], "correct_index": 1, "explanation": "Marginal evidence P(X) is identical across candidate classes during comparison.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "How is the empirical class prior P(Spam) computed from training documents?", "options": ["Total words ÷ V", "Spam document count ÷ Total training documents", "1 ÷ V", "Sum of feature weights"], "correct_index": 1, "explanation": "Prior probability is the fraction of training documents belonging to that class.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "What runtime complexity characterizes Naive Bayes prediction on n tokens across C classes?", "options": ["O(n² × C)", "O(n × C)", "O(2ⁿ)", "O(V³)"], "correct_index": 1, "explanation": "We add precomputed log likelihoods once per token per class, taking O(n × C) time.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
    ]
    flashcards = [
        {"front": "State Naive Bayes' conditional independence assumption.", "back": "Features are assumed mutually independent given class y, turning joint likelihood into a product.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "Why sum log P(y) + Σ log P(featureᵢ | y) instead of multiplying decimals?", "back": "Prevents floating-point underflow when multiplying many small probabilities together.", "source_badge": "[AUDIO @ 00:22]"},
        {"front": "What is the Zero-Frequency Problem?", "back": "One unseen word (count 0) multiplies the entire posterior score to zero without smoothing.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"front": "State the Add-1 (Laplace) smoothing rule in plain English.", "back": "(Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"front": "Why add Vocabulary Size V in the Laplace denominator?", "back": "Keeps the smoothed distribution normalized so all V probabilities sum to 1.", "source_badge": "[AUDIO @ 00:46]"},
        {"front": "Laplace example: unseen word (count = 0) with 200 Spam words and V = 1,000?", "back": "(0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200 (safe non-zero weight).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
    ]
    save_revision_items(lec_id, "mcq", mcqs, db_path=db_path)
    save_revision_items(lec_id, "flashcard", flashcards, db_path=db_path)
    return lec_id


def _seed_csc325_ieee754(db_path: Path | str | None = None) -> int:
    """Seed CSC325 — Computer Organization & Assembly (IEEE 754 Single-Precision Floating Point)."""
    summary = {
        "generation_source": "demo_kit",
        "executive_summary": (
            "IEEE 754 single-precision numbers pack real values into 32 bits: 1 sign bit, an 8-bit biased "
            "exponent (bias = 127), and a 23-bit mantissa fraction after an implicit hidden leading 1. "
            "Converting −12.625 sets sign S = 1, shifts binary 1100.101₂ to 1.100101 × 2³, adds bias to get "
            "exponent 130 (10000010₂), and packs the word into hexadecimal 0xC14A0000."
        ),
        "structured_notes": [
            {
                "heading": "32-Bit Field Layout & Hidden Leading 1",
                "source_type": "SLIDES_PDF",
                "formal_definition": (
                    "IEEE 754 binary32 single precision packs real numbers into three adjacent bit fields totaling 32 bits: "
                    "1 Most-Significant Sign bit (S), an 8-bit unsigned Biased Exponent (E' = E + 127), and a 23-bit "
                    "normalized Mantissa Fraction omitting the implicit leading 1."
                ),
                "detailed_explanation": (
                    "Storing real numbers inside fixed 32-bit hardware registers requires balanced dynamic range and "
                    "precision. Placing Sign, Exponent, and Mantissa strictly in descending significance order lets hardware "
                    "compare non-negative floating-point magnitudes almost like ordinary unsigned integers.\n\n"
                    "Because every non-zero binary scientific number normalizes to 1.xxxxx × 2ᴱ, the leading '1' before "
                    "the binary point is always known implicitly, awarding 24 effective significand bits inside just 23 stored bits."
                ),
                "formula_and_breakdown": (
                    "Value = (−1)ˢ × (1.Fraction)₂ × 2^(E' − 127)\n\n"
                    "• S (Bit 31, 1 bit) — Sign Bit: 0 denotes positive (+), 1 denotes negative (−).\n"
                    "• E' (Bits 30–23, 8 bits) — Biased Exponent: E + 127 (valid normal range [1 .. 254]).\n"
                    "• Fraction (Bits 22–0, 23 bits) — Mantissa Bits: Fractional tail after implicit leading 1."
                ),
                "worked_example": (
                    "Decode IEEE 754 normal bit fields for E' = 129, S = 0, Fraction = 10000...:\n"
                    "1. Unbiased exponent E = 129 − 127 = 2.\n"
                    "2. Restoring hidden 1 gives (1.1)₂ = 1.5.\n"
                    "3. Value = (+1) × 1.5 × 2² = 6.0."
                ),
                "exam_takeaways": [
                    "Bias equals 2^(k−1) − 1 = 127 for 8-bit single precision (1023 for double precision).",
                    "Never store the leading 1 inside the 23 mantissa bits for normal floats.",
                ],
                "takeaway": "Value Formula: (−1)ˢ × (1.Fraction)₂ × 2^(Exponent Field − 127)",
            },
            {
                "heading": "Worked Conversion: −12.625 to Hexadecimal",
                "source_type": "WHITEBOARD_IMAGE",
                "formal_definition": (
                    "IEEE 754 hexadecimal packing concatenates the 1-bit Sign, 8-bit Biased Exponent, and 23-bit Mantissa "
                    "into 32 bits and encodes each consecutive 4-bit nibble as a hexadecimal digit."
                ),
                "detailed_explanation": (
                    "On the whiteboard trace, aligning bit boundaries cleanly prevents off-by-one nibble mistakes: the "
                    "Sign bit sits at bit 31 followed immediately by 8 exponent bits, meaning the top hex nibble merges "
                    "Sign plus the highest 3 bits of Exponent.\n\n"
                    "Since IEEE 754 stores negative floats using sign-magnitude rather than two's complement, toggling "
                    "Sign bit 31 converts +12.625 (0x414A0000) directly into −12.625 (0xC14A0000)."
                ),
                "formula_and_breakdown": (
                    "[1 | 10000010 | 10010100000000000000000] = 0xC14A0000\n\n"
                    "• Nibbles 1100 (C) 0001 (1): Combines Sign = 1 + Exponent 10000010₂ (130 = 127 + 3).\n"
                    "• Nibbles 0100 (4) 1010 (A): Combines Exponent LSB (0) + top 7 Mantissa bits (1001010).\n"
                    "• Remaining 16 zero bits: Hex digits 0000."
                ),
                "worked_example": (
                    "Group the 32 bits of −12.625 into 8 hex nibbles:\n"
                    "1. Bit string: 1 10000010 10010100000000000000000\n"
                    "2. 4-bit groups: 1100 | 0001 | 0100 | 1010 | 0000 | 0000 | 0000 | 0000\n"
                    "3. Hex digits:      C  |   1  |   4  |   A  |   0  |   0  |   0  |   0  → 0xC14A0000."
                ),
                "exam_takeaways": [
                    "Negating any IEEE 754 float flips ONLY bit 31 (two's complement is NOT used).",
                    "Always zero-pad the mantissa to the RIGHT (trailing bits), never to the left.",
                ],
                "takeaway": "32-Bit Result: [1 | 10000010 | 10010100000000000000000] → Hex 0xC14A0000",
            },
            {
                "heading": "Reserved Exponents (Zero, Infinity & NaN)",
                "source_type": "AUDIO",
                "formal_definition": (
                    "All-zero (0) and all-one (255) exponent encodings break normal implicit-1 semantics to represent exact zero, "
                    "gradual underflow subnormals, positive/negative infinities, and invalid-operation NaN flags."
                ),
                "detailed_explanation": (
                    "Binary fractions near zero lose normalized headroom once unbiased exponent falls below −126. Reserving "
                    "E' = 0 enables gradual subnormal scaling (leading 0.Fraction × 2⁻¹²⁶) and ±0 representation.\n\n"
                    "Reserving E' = 255 distinguishes overflow infinities (fraction == 0) from non-numeric results like 0 ÷ 0 (NaN, fraction ≠ 0)."
                ),
                "formula_and_breakdown": (
                    "E' = 0 (00000000₂) = Zero or Subnormal  |  E' = 255 (11111111₂) = Infinity or NaN\n\n"
                    "• E'=0, Fraction=0 → Signed ±0.\n"
                    "• E'=255, Fraction=0 → Signed ±Infinity.\n"
                    "• E'=255, Fraction≠0 → NaN (Not a Number)."
                ),
                "worked_example": (
                    "Classify 32-bit word 0x7F800000:\n"
                    "1. Sign = 0, Exponent = 11111111₂ (255), Fraction = 000…000.\n"
                    "2. Result = +Infinity."
                ),
                "exam_takeaways": [
                    "Exponent 0 encodes ±0 (frac=0) or subnormals (frac≠0).",
                    "Exponent 255 encodes ±Infinity (frac=0) or NaN (frac≠0).",
                ],
                "takeaway": "Special Patterns: Exponent 0 = Zero / Subnormals  |  Exponent 255 = Infinity or NaN",
            },
        ],
        "extracted_pseudocode": (
            "Value Formula: (−1)ˢ × (1.Fraction)₂ × 2^(Exponent Field − 127)\n\n"
            "Worked Encoding for −12.625:\n"
            "  • Sign (1 bit)       : 1   (negative)\n"
            "  • Binary Magnitude   : 12.625 = 1100.101₂ = 1.100101 × 2³\n"
            "  • Biased Exp (8 bit) : 127 + 3 = 130 = 10000010₂\n"
            "  • Mantissa (23 bit)  : 10010100000000000000000\n\n"
            "32-Bit Word: 1 10000010 10010100000000000000000\n"
            "Hex Nibbles: 1100 0001 0100 1010 0000 0000 0000 0000 = 0xC14A0000"
        ),
    }
    chunks = [
        {
            "topic": "IEEE 754 Single-Precision Bit Widths",
            "content": "A 32-bit IEEE 754 single-precision float contains 1 Sign bit (S), 8 Biased Exponent bits (E'), and 23 Mantissa fraction bits.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 1",
            "is_uncertain": False,
        },
        {
            "topic": "Normalized Form & Hidden Bit",
            "content": "Normalized values equal (−1)ˢ × (1.Fraction)₂ × 2^(E' − 127); the leading 1 before the binary point is implicit and omitted from the 23 stored bits.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 1",
            "is_uncertain": False,
        },
        {
            "topic": "Binary Conversion of −12.625",
            "content": "Converting −12.625 gives Sign S = 1, integer 12 = 1100₂, fraction 0.625 = .101₂, yielding 1100.101₂ = 1.100101 × 2³ (unbiased exponent E = 3).",
            "source_type": "AUDIO",
            "source_reference": "00:20",
            "is_uncertain": False,
        },
        {
            "topic": "8-Bit Biased Exponent Field",
            "content": "8-bit biased exponent field smudged for −12.625 — inferred as 127 + 3 = 130 (10000010₂)",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": True,
            "uncertainty_reason": (
                "Exponent box in the 32-bit diagram had a dry-erase smudge; calculated Bias (127) + E (3) = 130 = 10000010₂."
            ),
        },
        {
            "topic": "Hexadecimal Packing (0xC14A0000)",
            "content": "Packing S = 1, exponent = 10000010₂, and mantissa = 100101... groups into nibbles 1100 0001 0100 1010 0000 0000 0000 0000 = 0xC14A0000.",
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 2",
            "is_uncertain": False,
        },
        {
            "topic": "Special Exponent Values (0 and 255)",
            "content": "In 8-bit IEEE 754 exponents, E' = 0 encodes zero or subnormals, while E' = 255 encodes ±Infinity (fraction = 0) or NaN (fraction ≠ 0).",
            "source_type": "AUDIO",
            "source_reference": "00:48",
            "is_uncertain": False,
        },
    ]
    lec_id = save_lecture("CSC325", "IEEE 754 Single-Precision Floating Point", summary, chunks, db_path=db_path)
    mcqs = [
        {"question": "How are the 32 bits partitioned in IEEE 754 single-precision floating point?", "options": ["1 Sign, 8 Exponent, 23 Mantissa", "1 Sign, 11 Exponent, 20 Mantissa", "2 Sign, 7 Exponent, 23 Mantissa", "1 Sign, 10 Exponent, 21 Mantissa"], "correct_index": 0, "explanation": "Single precision allocates 1 sign bit, 8 exponent bits, and 23 fraction bits.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "What exponent bias is used in IEEE 754 single-precision (8-bit exponent) format?", "options": ["63", "127", "128", "1023"], "correct_index": 1, "explanation": "Single precision uses bias 2⁷ − 1 = 127; 1023 is for 64-bit double precision.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "What is the unsigned binary equivalent of fractional decimal 0.625?", "options": [".100₂", ".101₂", ".110₂", ".011₂"], "correct_index": 1, "explanation": "0.625 = 0.5 (2⁻¹) + 0.125 (2⁻³) = .101 in binary.", "source_badge": "[AUDIO @ 00:20]"},
        {"question": "Normalizing −1100.101₂ yields −1.100101 × 2ᴱ. What is the unbiased exponent E?", "options": ["2", "3", "4", "−3"], "correct_index": 1, "explanation": "Shifting the binary point 3 positions left gives 1.100101 × 2³, so E = 3.", "source_badge": "[AUDIO @ 00:20]"},
        {"question": "For unbiased exponent E = 3, what is the 8-bit biased exponent E' stored in IEEE 754 single precision?", "options": ["00000011 (3)", "10000001 (129)", "10000010 (130)", "10000100 (132)"], "correct_index": 2, "explanation": "Biased exponent E' = 127 + 3 = 130 = 128 + 2 = 10000010 in binary.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "What happens to the leading '1' of normalized 1.100101 inside the 23-bit mantissa field?", "options": ["It is stored as bit 22", "It is XORed into the sign bit", "It is implicit (hidden) and omitted from storage", "It is truncated"], "correct_index": 2, "explanation": "Normalized binary numbers always start with 1., yielding 24 bits of precision in 23 stored bits.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "What are the top 6 non-zero bits of the 23-bit mantissa for −12.625?", "options": ["110010", "100101", "010100", "100001"], "correct_index": 1, "explanation": "After dropping leading 1., the fractional bits are 100101 followed by 17 zeros.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "What is the complete 32-bit hexadecimal representation of −12.625?", "options": ["0x414A0000", "0xC14A0000", "0xC12A0000", "0xC0CA0000"], "correct_index": 1, "explanation": "Grouping 1 10000010 10010100... into nibbles gives 1100(C) 0001(1) 0100(4) 1010(A) 0000 = 0xC14A0000.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "What does an 8-bit exponent field of 255 (11111111₂) with a non-zero fraction represent?", "options": ["Positive Zero", "Smallest subnormal", "NaN (Not a Number)", "Overflow to −1"], "correct_index": 2, "explanation": "All-1s exponent with fraction = 0 is ±Infinity; with fraction ≠ 0 it encodes NaN.", "source_badge": "[AUDIO @ 00:48]"},
        {"question": "What single bit change flips +12.625 (0x414A0000) into −12.625?", "options": ["Changing LSB bit 0", "Setting MSB sign bit 31 from 0 to 1", "Inverting all exponent bits", "Adding 127 to mantissa"], "correct_index": 1, "explanation": "IEEE 754 uses sign-magnitude format, so toggling only sign bit 31 negates the value.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
    ]
    flashcards = [
        {"front": "State the IEEE 754 32-bit field widths.", "back": "1 Sign bit (S) · 8 Biased Exponent bits (E') · 23 Mantissa/Fraction bits.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "Why does IEEE 754 add Bias 127 to the exponent?", "back": "Stores exponents as unsigned integers (1–254) so floats sort cleanly in hardware.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "What is the hidden bit in normalized IEEE 754 numbers?", "back": "The implicit leading 1 before the binary point, granting 24-bit precision in 23 stored bits.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "Compute the smudged 8-bit exponent for −12.625.", "back": "−1100.101₂ = −1.100101 × 2³ → Biased Exponent = 127 + 3 = 130 (10000010₂).", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"front": "Write the 32-bit hex encoding of −12.625.", "back": "0xC14A0000 (Sign = 1, Exponent = 10000010₂, Fraction = 100101...).", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"front": "What do Exponent = 0 and Exponent = 255 encode in IEEE 754?", "back": "0 encodes Zero or Subnormals; 255 encodes ±Infinity (fraction = 0) or NaN (fraction ≠ 0).", "source_badge": "[AUDIO @ 00:48]"},
    ]
    save_revision_items(lec_id, "mcq", mcqs, db_path=db_path)
    save_revision_items(lec_id, "flashcard", flashcards, db_path=db_path)
    return lec_id


def _seed_csc322_fcfs(db_path: Path | str | None = None) -> int:
    """Seed CSC322 — Operating Systems (First-Come, First-Served FCFS CPU Scheduling)."""
    fcfs_summary = {
        "generation_source": "demo_kit",
        "executive_summary": (
            "First-Come, First-Served (FCFS) is a non-preemptive CPU scheduler that dispatches tasks strictly "
            "in order of arrival using a FIFO ready queue. Across our worked three-process schedule, P1–P3 "
            "achieve an Average Turnaround Time of 5.67 ms and Average Waiting Time of 2.67 ms. While simple "
            "and starvation-free, FCFS suffers from the Convoy Effect when short jobs stall behind one long CPU burst."
        ),
        "structured_notes": [
            {
                "heading": "FIFO Dispatch Policy & Non-Preemptive Execution",
                "source_type": "SLIDES_PDF",
                "formal_definition": (
                    "First-Come, First-Served (FCFS) is a non-preemptive CPU scheduling algorithm in which processes "
                    "are inserted at the tail of a First-In, First-Out (FIFO) ready queue upon arrival and hold the "
                    "CPU uninterrupted until their CPU burst terminates or blocks on I/O."
                ),
                "detailed_explanation": (
                    "Because scheduling decisions happen strictly when a running process finishes or enters I/O wait, "
                    "FCFS demands almost zero dispatcher complexity or kernel timer bookkeeping. Once dispatched from "
                    "the head of the ready queue, no newly arriving higher-priority or shorter task can evict the active job.\n\n"
                    "Over long batch runs, every finite burst ahead of a process eventually completes, which mathematically "
                    "eliminates indefinite postponement (starvation) while trading off interactive responsiveness."
                ),
                "formula_and_breakdown": (
                    "Dispatch Order = Sort ascending by Arrival Time (AT), tie-broken by PID\n\n"
                    "• Ready Queue — FIFO buffer holding runnable jobs awaiting CPU allocation.\n"
                    "• Non-Preemptive Mode — Active job releases CPU solely upon burst completion or I/O wait.\n"
                    "• Context Switch Count — Strictly N transitions for N non-blocking bursts."
                ),
                "worked_example": (
                    "Suppose Jobs P1 (AT = 0 ms, BT = 4 ms) and P2 (AT = 1 ms, BT = 3 ms) enter the queue:\n"
                    "1. At t = 0 ms, only P1 is present → P1 starts on CPU immediately.\n"
                    "2. At t = 1 ms, P2 arrives and waits behind P1 without preempting it.\n"
                    "3. At t = 4 ms, P1 completes cleanly and P2 immediately begins execution."
                ),
                "exam_takeaways": [
                    "FCFS is strictly non-preemptive and backed by a standard FIFO queue.",
                    "Zero starvation: every arriving process eventually advances to the front of the queue.",
                ],
                "takeaway": "Scheduling Policy: Earliest arrival runs first and keeps CPU until its burst finishes.",
            },
            {
                "heading": "Turnaround Time & Waiting Time Metrics",
                "source_type": "SLIDES_PDF",
                "formal_definition": (
                    "Turnaround Time (TAT) measures the total interval from process arrival in the ready queue until "
                    "final completion, whereas Waiting Time (WT) isolates only the idle time spent queued ready for the CPU."
                ),
                "detailed_explanation": (
                    "Evaluating scheduler performance requires separating productive CPU execution from idle queueing "
                    "delay. Completion Time alone depends on absolute timeline offsets, so subtracting Arrival Time "
                    "normalizes total lifetime from the moment the job became ready.\n\n"
                    "Subtracting the required CPU Burst Time from Turnaround Time exposes exact Waiting Time — making "
                    "average waiting time the canonical benchmark for comparing FCFS against SJF or Round Robin."
                ),
                "formula_and_breakdown": (
                    "TAT = CT − AT   |   WT = TAT − BT   |   Avg WT = (Σ WTᵢ) ÷ N\n\n"
                    "• CT — Completion Time: Timeline clock instant when a process exits its final burst.\n"
                    "• AT — Arrival Time: Instant the process enters the ready queue.\n"
                    "• BT — Burst Time: Total CPU execution duration needed by the process."
                ),
                "worked_example": (
                    "Verify metrics for P2 (AT = 1 ms, BT = 3 ms, CT = 7 ms):\n"
                    "1. Turnaround Time TAT = CT − AT = 7 − 1 = 6 ms.\n"
                    "2. Waiting Time WT = TAT − BT = 6 − 3 = 3 ms."
                ),
                "exam_takeaways": [
                    "Always compute Completion Time from the Gantt timeline BEFORE Turnaround and Waiting times.",
                    "Idle CPU gaps occur if the ready queue empties before the next arrival timestamp.",
                ],
                "takeaway": "Formulas: Turnaround Time = Completion Time − Arrival Time  |  Waiting Time = Turnaround Time − Burst Time",
            },
            {
                "heading": "Worked Gantt Chart & Average Waiting Time",
                "source_type": "WHITEBOARD_IMAGE",
                "formal_definition": (
                    "A Gantt chart visually partitions continuous CPU timeline intervals [start, end] across active "
                    "processes, proving execution order, burst durations, and exact completion instants."
                ),
                "detailed_explanation": (
                    "Translating an arrival-burst table onto a horizontal timeline makes queue overlaps immediately "
                    "visible. Whenever a preceding job ends after the next job's arrival, the successor runs at zero gap.\n\n"
                    "Reading each block width (End − Start) verifies execution burst directly from diagram geometry "
                    "even when table entries are partially obscured."
                ),
                "formula_and_breakdown": (
                    "Gantt Block Width = End Time − Start Time = Burst Time\n"
                    "Avg TAT = (Σ TATᵢ) ÷ N   |   Avg WT = (Σ WTᵢ) ÷ N\n\n"
                    "• P1 [0–4] → BT = 4 ms, CT = 4 ms, TAT = 4 ms, WT = 0 ms.\n"
                    "• P2 [4–7] → BT = 7 − 4 = 3 ms, CT = 7 ms, TAT = 6 ms, WT = 3 ms.\n"
                    "• P3 [7–9] → BT = 2 ms, CT = 9 ms, TAT = 7 ms, WT = 5 ms."
                ),
                "worked_example": (
                    "Complete 3-Process Board Trace:\n"
                    "1. Draw Gantt timeline: 0 ── P1 ── 4 ── P2 ── 7 ── P3 ── 9.\n"
                    "2. Deduce P2 burst = 7 − 4 = 3 ms from slot [4–7].\n"
                    "3. Avg Turnaround Time = (4 + 6 + 7) ÷ 3 = 17 ÷ 3 = 5.67 ms.\n"
                    "4. Avg Waiting Time = (0 + 3 + 5) ÷ 3 = 8 ÷ 3 = 2.67 ms."
                ),
                "exam_takeaways": [
                    "Verify Σ(Gantt widths) == Σ(Burst Times) when zero idle gaps occur.",
                    "P2's burst duration equals End(7) − Start(4) = 3 ms.",
                ],
                "takeaway": "Worked Averages: Avg Turnaround = 17 ÷ 3 = 5.67 ms  |  Avg Waiting = (0 + 3 + 5) ÷ 3 = 2.67 ms",
            },
            {
                "heading": "The Convoy Effect & Performance Trade-Off",
                "source_type": "AUDIO",
                "formal_definition": (
                    "The Convoy Effect is a scheduling degradation phenomenon where many short CPU- or I/O-bound processes "
                    "stall behind one long CPU-intensive job at the front of a non-preemptive FCFS queue."
                ),
                "detailed_explanation": (
                    "When a single compute-heavy task holding a 25 ms burst leads ten 1 ms interactive jobs, every brief "
                    "task idles for 25+ ms before finishing in 1 ms bursts. Once all short tasks flush I/O requests "
                    "together, disks jam while the CPU sits idle, depressing overall utilization.\n\n"
                    "Swapping the execution order so short jobs finish first slashes Average Waiting Time dramatically, "
                    "motivating Shortest-Job-First (SJF) and Round-Robin algorithms."
                ),
                "formula_and_breakdown": (
                    "Convoy Penalty ≈ Long Burst Duration × (Number of Trailing Short Processes)\n\n"
                    "• Long Burst — Compute-heavy leading process blocking the FIFO queue.\n"
                    "• Device Idleness — I/O peripherals sit underutilized while short jobs queue for CPU."
                ),
                "worked_example": (
                    "Compare 1 long burst (24 ms) + 2 short bursts (3 ms, 3 ms):\n"
                    "• Long first: Avg WT = (0 + 24 + 27) ÷ 3 = 17.0 ms (Convoy Effect).\n"
                    "• Short first: Avg WT = (0 + 3 + 6) ÷ 3 = 3.0 ms (nearly 6× lower wait)."
                ),
                "exam_takeaways": [
                    "Convoy Effect happens specifically when a long CPU burst enters FIFO ahead of short bursts.",
                    "FCFS optimizes fairness by arrival time, NOT minimal average waiting time.",
                ],
                "takeaway": "Key Trade-Off: Starvation-free and fair by arrival, but vulnerable to high wait times behind long bursts.",
            },
        ],
        "extracted_pseudocode": (
            "Formulas:\n"
            "  Turnaround Time (TAT) = Completion Time − Arrival Time\n"
            "  Waiting Time (WT)     = Turnaround Time − Burst Time\n\n"
            "Gantt Chart Timeline (ms):\n"
            "  0 ──── P1 (4 ms) ──── 4 ─── P2 (3 ms) ─── 7 ── P3 (2 ms) ── 9\n\n"
            "Process | Arrival | Burst | Completion | Turnaround | Waiting\n"
            "--------+---------+-------+------------+------------+--------\n"
            "P1      |   0 ms  |  4 ms |    4 ms    |    4 ms    |   0 ms\n"
            "P2      |   1 ms  |  3 ms*|    7 ms    |    6 ms    |   3 ms\n"
            "P3      |   2 ms  |  2 ms |    9 ms    |    7 ms    |   5 ms\n\n"
            "* P2 Burst Time read from board slot [4–7] → 7 − 4 = 3 ms\n"
            "Avg Turnaround = 17 ÷ 3 = 5.67 ms  ·  Avg Waiting = 8 ÷ 3 = 2.67 ms"
        ),
    }

    fcfs_chunks = [
        {
            "topic": "FCFS Scheduling Policy (FIFO)",
            "content": (
                "First-Come, First-Served (FCFS) is a non-preemptive CPU scheduling algorithm backed by a "
                "First-In, First-Out (FIFO) ready queue. Each process holds the CPU until its burst completes."
            ),
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 1",
            "is_uncertain": False,
        },
        {
            "topic": "Turnaround & Waiting Time Formulas",
            "content": (
                "Turnaround Time equals Completion Time minus Arrival Time (TAT = CT − AT). "
                "Waiting Time equals Turnaround Time minus Burst Time (WT = TAT − BT)."
            ),
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 2",
            "is_uncertain": False,
        },
        {
            "topic": "Process Arrivals & Known Bursts",
            "content": (
                "Three processes arrive on the ready queue: P1 arrives at 0 ms with Burst 4 ms; "
                "P2 arrives at 1 ms; P3 arrives at 2 ms with Burst 2 ms."
            ),
            "source_type": "AUDIO",
            "source_reference": "00:24",
            "is_uncertain": False,
        },
        {
            "topic": "P2 Burst Time Board Entry",
            "content": "P2 Burst Time smudged on whiteboard — inferred as 3 ms from Gantt chart slot [4–7]",
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": True,
            "uncertainty_reason": (
                "Dry-erase smudge over P2's Burst Time cell on the whiteboard; duration 7 − 4 = 3 ms "
                "was deduced from the Gantt chart [4–7] and requires student confirmation."
            ),
        },
        {
            "topic": "Worked Gantt Chart & Average Metrics",
            "content": (
                "The FCFS Gantt chart runs P1 on [0–4], P2 on [4–7], and P3 on [7–9]. "
                "Completion times are P1 = 4, P2 = 7, P3 = 9; Turnaround times are 4, 6, 7 (Avg = 5.67 ms); "
                "Waiting times are 0, 3, 5 (Avg = 8 ÷ 3 = 2.67 ms)."
            ),
            "source_type": "WHITEBOARD_IMAGE",
            "source_reference": "Whiteboard #1",
            "is_uncertain": False,
        },
        {
            "topic": "The Convoy Effect",
            "content": (
                "The Convoy Effect occurs when one long CPU-bound process runs ahead of many shorter processes "
                "in an FCFS queue, causing short jobs to pile up waiting behind it and raising Average Waiting Time."
            ),
            "source_type": "AUDIO",
            "source_reference": "00:52",
            "is_uncertain": False,
        },
        {
            "topic": "Starvation Freedom & Trade-offs",
            "content": (
                "FCFS never causes starvation because every arriving process eventually advances to the front "
                "of the FIFO queue once prior bursts finish, though it does not minimize Average Waiting Time."
            ),
            "source_type": "SLIDES_PDF",
            "source_reference": "Slide 2",
            "is_uncertain": False,
        },
    ]

    lec_id = save_lecture(
        course_code="CSC322",
        lecture_title="First-Come, First-Served (FCFS) CPU Scheduling",
        summary_data=fcfs_summary,
        chunks=fcfs_chunks,
        db_path=db_path,
    )

    mcqs = [
        {"question": "Which data structure backs the ready queue in First-Come, First-Served (FCFS) CPU scheduling?", "options": ["First-In, First-Out (FIFO) queue", "Min-heap priority queue keyed by burst time", "Last-In, First-Out (LIFO) stack", "Circular multilevel feedback ring with 4 ms quantum"], "correct_index": 0, "explanation": "FCFS dispatches processes strictly in order of arrival using a standard FIFO queue.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "How is standard FCFS CPU scheduling classified regarding preemption?", "options": ["Always preemptive every 2 milliseconds", "Non-preemptive until the CPU burst finishes or blocks", "Preemptive whenever a shorter burst arrives", "Adaptive between preemptive and cooperative modes"], "correct_index": 1, "explanation": "Once a process holds the CPU under FCFS, it keeps it until completion or voluntary wait.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"question": "Which formula correctly computes Turnaround Time (TAT) for a process?", "options": ["TAT = Burst Time − Arrival Time", "TAT = Completion Time + Waiting Time", "TAT = Completion Time − Arrival Time", "TAT = Start Time − Completion Time"], "correct_index": 2, "explanation": "Turnaround Time measures the total elapsed interval from arrival to completion (CT − AT).", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "Given Turnaround Time (TAT) and Burst Time (BT), how is Waiting Time (WT) calculated?", "options": ["WT = TAT + BT", "WT = TAT − BT", "WT = CT − BT", "WT = BT − AT"], "correct_index": 1, "explanation": "Waiting Time isolates time spent queued rather than executing: WT = TAT − BT.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"question": "In the lecture's whiteboard example, how was P2's smudged Burst Time determined?", "options": ["Assumed to be 5 ms by averaging P1 and P3", "Read as 3 ms from the Gantt chart interval [4–7] (7 − 4)", "Set to 1 ms equal to P2's arrival time", "Discarded from the schedule due to ambiguity"], "correct_index": 1, "explanation": "On the Gantt chart, P2 executes from 4 ms to 7 ms, giving a burst duration of 3 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "With P1 = 4 ms (arrives 0), P2 = 3 ms (arrives 1), and P3 = 2 ms (arrives 2), what is P3's Completion Time?", "options": ["7 ms", "8 ms", "9 ms", "11 ms"], "correct_index": 2, "explanation": "P1 finishes at 4 ms, P2 finishes at 7 ms, and P3 (burst 2 ms) finishes at 7 + 2 = 9 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "For process P2 (Arrival = 1 ms, Burst = 3 ms, Completion = 7 ms), what is P2's Waiting Time?", "options": ["1 ms", "3 ms", "4 ms", "6 ms"], "correct_index": 1, "explanation": "Turnaround(P2) = 7 − 1 = 6 ms, so Waiting(P2) = 6 − 3 = 3 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "What is the Average Waiting Time across P1 (0 ms), P2 (3 ms), and P3 (5 ms)?", "options": ["2.00 ms", "2.67 ms", "3.00 ms", "5.67 ms"], "correct_index": 1, "explanation": "Average WT = (0 + 3 + 5) ÷ 3 = 8 ÷ 3 = 2.67 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"question": "What causes the Convoy Effect in FCFS scheduling?", "options": ["Too many equal-length bursts arriving at time 0", "Frequent context switches from a tiny time quantum", "Multiple shorter processes piling up behind one long CPU-bound process", "Aging priorities pushing I/O jobs to the back forever"], "correct_index": 2, "explanation": "When a long CPU-bound process runs first, shorter jobs wait behind it and inflate average waiting time.", "source_badge": "[AUDIO @ 00:52]"},
        {"question": "Why does FCFS guarantee freedom from starvation?", "options": ["It dynamically boosts priority every 10 ms", "Every process in the FIFO queue advances as earlier bursts finish", "It runs the shortest job ahead of long bursts", "It preempts any burst longer than 4 ms"], "correct_index": 1, "explanation": "Because processes execute in strict arrival order and finite bursts end, no job waits forever.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
    ]

    flashcards = [
        {"front": "What scheduling rule does FCFS follow and is it preemptive?", "back": "Non-preemptive execution in strict arrival order using a FIFO ready queue.", "source_badge": "[SLIDES_PDF @ Slide 1]"},
        {"front": "State the formula for Turnaround Time (TAT).", "back": "Turnaround Time = Completion Time − Arrival Time.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"front": "State the formula for Waiting Time (WT).", "back": "Waiting Time = Turnaround Time − Burst Time.", "source_badge": "[SLIDES_PDF @ Slide 2]"},
        {"front": "How was P2's smudged Burst Time resolved in the board example?", "back": "P2 occupied [4–7] on the Gantt chart, giving Burst Time = 7 − 4 = 3 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"front": "What are Average WT and Average TAT for P1(0,4), P2(1,3), P3(2,2)?", "back": "Avg WT = (0 + 3 + 5) ÷ 3 = 2.67 ms; Avg TAT = (4 + 6 + 7) ÷ 3 = 5.67 ms.", "source_badge": "[WHITEBOARD_IMAGE @ Whiteboard #1]"},
        {"front": "Define the Convoy Effect in FCFS CPU Scheduling.", "back": "Short processes wait behind one long CPU-bound process, increasing Average Waiting Time and lowering utilization.", "source_badge": "[AUDIO @ 00:52]"},
    ]

    save_revision_items(lec_id, "mcq", mcqs, db_path=db_path)
    save_revision_items(lec_id, "flashcard", flashcards, db_path=db_path)
    return lec_id


def _seed_all_four_subjects(db_path: Path | str | None = None) -> None:
    """Insert all 4 CS course lectures (ordered so CSC322 gets the newest ID and appears first)."""
    _seed_csc301_binary_search(db_path=db_path)
    _seed_aic354_naive_bayes(db_path=db_path)
    _seed_csc325_ieee754(db_path=db_path)
    _seed_csc322_fcfs(db_path=db_path)


def seed_initial_data(db_path: Path | str | None = None) -> None:
    """
    Ensure all 4 COMSATS CS subjects (CSC322, CSC325, AIC354, CSC301) and the
    Fall 2026 schedule are seeded cleanly and deduplicated per course.
    """
    init_db(db_path)
    canonical_titles = {
        "First-Come, First-Served (FCFS) CPU Scheduling",
        "IEEE 754 Single-Precision Floating Point",
        "Naive Bayes & Bayesian Belief Classification",
        "Binary Search & Recurrence Complexity",
    }
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, course_code, lecture_title, summary_json FROM lectures ORDER BY id DESC"
        ).fetchall()
        titles = {row["lecture_title"] for row in rows}
        codes = {row["course_code"] for row in rows}
        lec_count = len(rows)
        slot_count = conn.execute("SELECT COUNT(*) FROM timetable_slots").fetchone()[0]
        old_section_slots = conn.execute(
            "SELECT COUNT(*) FROM timetable_slots WHERE section LIKE 'FA26-%' OR instructor = ''"
        ).fetchone()[0]

    has_legacy_lectures = any(t in titles for t in _LEGACY_SEED_TITLES)
    is_old_single_seed = (
        lec_count == 1
        and codes == {"CSC322"}
        and titles == {"First-Come, First-Served (FCFS) CPU Scheduling"}
    )
    has_outdated_canonical = False
    for row in rows:
        if row["lecture_title"] in canonical_titles:
            try:
                parsed = json.loads(row["summary_json"])
                notes = parsed.get("structured_notes") or []
                if (
                    not notes
                    or "takeaway" not in notes[0]
                    or "source_type" not in notes[0]
                    or "formal_definition" not in notes[0]
                    or "formula_and_breakdown" not in notes[0]
                    or "generation_source" not in parsed
                ):
                    has_outdated_canonical = True
                    break
            except Exception:
                has_outdated_canonical = True
                break

    if has_legacy_lectures or is_old_single_seed or has_outdated_canonical:
        with _get_connection(db_path) as conn:
            conn.execute("DELETE FROM lecture_feedback")
            conn.execute("DELETE FROM revision_items")
            conn.execute("DELETE FROM knowledge_chunks")
            conn.execute("DELETE FROM lectures")
        lec_count = 0
    elif lec_count > len({c.strip().upper() for c in codes}):
        # Deduplicate any stray repeated captures for the same course, keeping newest ID
        seen_codes: set[str] = set()
        drop_ids: list[int] = []
        for row in rows:
            cu = row["course_code"].strip().upper()
            if cu in seen_codes:
                drop_ids.append(int(row["id"]))
            else:
                seen_codes.add(cu)
        if drop_ids:
            placeholders = ",".join("?" for _ in drop_ids)
            with _get_connection(db_path) as conn:
                conn.execute(f"DELETE FROM revision_items WHERE lecture_id IN ({placeholders})", drop_ids)
                conn.execute(f"DELETE FROM knowledge_chunks WHERE lecture_id IN ({placeholders})", drop_ids)
                conn.execute(f"DELETE FROM lectures WHERE id IN ({placeholders})", drop_ids)

    if old_section_slots > 0 and slot_count <= 8:
        slot_count = 0

    if lec_count == 0:
        _seed_all_four_subjects(db_path=db_path)

    if slot_count == 0:
        reset_timetable_slots(db_path=db_path)


# Student-specific planner persistence (separate from the shared demo timetable).
def load_student_schedule(student_key: str = "local-demo", db_path: Path | str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, course_code, course_name, section, day, start_time, end_time, room, instructor "
            "FROM student_schedule_slots WHERE student_key = ? ORDER BY day, start_time, course_code",
            (student_key,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_student_schedule(
    slots: list[dict[str, Any]], student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> None:
    """Atomically replace one student's complete planned schedule."""
    init_db(db_path)
    normalized = []
    required = ("course_code", "course_name", "section", "day", "start_time", "end_time", "room", "instructor")
    for slot in slots:
        missing = [key for key in required if key not in slot]
        if missing:
            raise ValueError("Schedule row missing: " + ", ".join(missing))
        normalized.append(tuple(str(slot[key] or "") for key in required))
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM student_schedule_slots WHERE student_key = ?", (student_key,))
        conn.executemany(
            "INSERT INTO student_schedule_slots "
            "(student_key, course_code, course_name, section, day, start_time, end_time, room, instructor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(student_key, *row) for row in normalized],
        )


def save_uploaded_pdf_catalog(
    slots: list[dict[str, Any]],
    file_name: str = "",
    student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> None:
    """Persist rows parsed from an uploaded Timetable PDF for zero-dependency reloads."""
    init_db(db_path)
    required = ("course_code", "course_name", "section", "day", "start_time", "end_time", "room", "instructor")
    normalized = [
        tuple(str(slot.get(key) or "") for key in required)
        for slot in slots
        if str(slot.get("course_code") or "").strip()
    ]
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM uploaded_timetable_slots WHERE student_key = ?", (student_key,))
        if normalized:
            conn.executemany(
                "INSERT INTO uploaded_timetable_slots "
                "(student_key, file_name, course_code, course_name, section, day, start_time, end_time, room, instructor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(student_key, str(file_name or ""), *row) for row in normalized],
            )


def load_uploaded_pdf_catalog(
    student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return `(slots, uploaded_pdf_filename)` previously parsed for this student."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT file_name, course_code, course_name, section, day, start_time, end_time, room, instructor "
            "FROM uploaded_timetable_slots WHERE student_key = ? ORDER BY id ASC",
            (student_key,),
        ).fetchall()
    if not rows:
        return [], ""
    fname = str(rows[0]["file_name"] or "")
    slots = [
        {
            "course_code": r["course_code"],
            "course_name": r["course_name"],
            "section": r["section"],
            "day": r["day"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "room": r["room"],
            "instructor": r["instructor"],
        }
        for r in rows
    ]
    return slots, fname


def clear_uploaded_pdf_catalog(
    student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> None:
    """Clear stored uploaded Timetable PDF catalog rows."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM uploaded_timetable_slots WHERE student_key = ?", (student_key,))


def save_planner_state(
    state_dict: dict[str, Any],
    student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> None:
    """Persist Tab 4 Class/Section, failed courses, passed prerequisites, and section overrides."""
    init_db(db_path)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    prog = str(state_dict.get("program") or "BCS").upper()
    sec = str(state_dict.get("section_code") or "1A").upper()
    pdf_name = str(state_dict.get("pdf_filename") or "")
    payload = json.dumps(state_dict, ensure_ascii=False)
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO student_planner_state (student_key, program, section_code, pdf_filename, state_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_key) DO UPDATE SET
                program = excluded.program,
                section_code = excluded.section_code,
                pdf_filename = excluded.pdf_filename,
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (student_key, prog, sec, pdf_name, payload, now_str),
        )


def load_planner_state(
    student_key: str = "local-demo",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load persisted Tab 4 planner state dictionary (returns empty dict if none saved)."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT program, section_code, pdf_filename, state_json FROM student_planner_state WHERE student_key = ?",
            (student_key,),
        ).fetchone()
    if row is None:
        return {}
    try:
        data = json.loads(row["state_json"] or "{}")
    except Exception:
        data = {}
    if isinstance(data, dict):
        data.setdefault("program", row["program"])
        data.setdefault("section_code", row["section_code"])
        data.setdefault("pdf_filename", row["pdf_filename"])
        return data
    return {}
