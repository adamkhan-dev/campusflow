"""Deterministic COMSATS University Islamabad timetable PDF parser, clash detector, and PDF exporter."""

from __future__ import annotations

import io
import re
import zlib
from datetime import datetime

import pandas as pd

COMSATS_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
COMSATS_ALL_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

COMSATS_PROGRAMS = ["BCS", "BSE", "BAI", "BDS", "BCT"]
COMSATS_SECTION_CODES = ["1A", "1B", "1", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B"]

COMSATS_TIME_SLOTS = [
    ("08:30", "10:00"),
    ("10:00", "11:30"),
    ("11:30", "13:00"),
    ("13:30", "15:00"),
    ("15:00", "16:30"),
]

SEMESTER_5_COURSES = {
    "CSC322": "Operating Systems",
    "CSC325": "Computer Organization & Assembly",
    "AIC354": "Machine Learning Fundamentals",
    "CSC301": "Design & Analysis of Algorithms",
}

COMSATS_COURSE_NAMES = {
    **SEMESTER_5_COURSES,
    "CSC101": "Applications of ICT",
    "CSC103": "Programming Fundamentals",
    "CSC211": "Data Structures & Algorithms",
    "CSC232": "Information Security",
    "CSC241": "Object-Oriented Programming",
    "CSC271": "Database Systems",
    "CSC291": "Software Engineering",
    "CSC316": "Advanced Database Systems",
    "CSC336": "Web Technologies",
    "CSC339": "Artificial Intelligence",
    "EEE241": "Digital Logic Design",
    "MTH091": "Pre-Calculus I (Pre-Med)",
    "MTH104": "Calculus & Analytic Geometry",
    "MTH231": "Discrete Structures",
    "MTH242": "Probability & Statistics",
    "MTH262": "Linear Algebra",
    "HUM102": "Communication Skills",
    "HUM104": "Functional English",
    "HUM110": "Communication & Presentation Skills",
    "HUM222": "Fundamentals of International Relations",
    "HUM161": "Fehm ul Quran-I",
}

MASTER_REGULAR_SECTIONS = [
    "BCS-1A",
    "BCS-1B",
    "BCS-2A",
    "BCS-2B",
    "BCS-3A",
    "BCS-3B",
    "BCS-4A",
    "BCS-4B",
    "BCS-5A",
    "BCS-5B",
    "BSE-1",
    "BAI-1",
    "BDS-1",
    "BCT-1",
]

COMSATS_FALL_2026_SECTIONS = [
    *MASTER_REGULAR_SECTIONS,
    "RPT-CS",
]

COMSATS_ROOMS = ["106 AB-2", "107 AB-2", "CL-5", "CL-6"]

COMSATS_FACULTY = [
    "Mr. Rizwan Rashid",
    "Mr. Khurram Iqbal",
    "Ms. Ayesha Hussain",
]


def normalize_class_section(raw_section: str) -> str:
    """Normalize labels such as 'BCS 1a', 'Fall 2026 - BCS - 1A', or 'bse1' to 'BCS-1A' / 'BSE-1'."""
    text = re.sub(r"\s+", " ", str(raw_section or "").strip().upper())
    if not text:
        return ""
    term_m = re.search(
        r"(?:FALL|SPRING|SUMMER)\s*20\d{2}\s*-\s*([A-Z]{2,5})\s*-\s*(\d+[A-Z]?)",
        text,
    )
    if term_m:
        return f"{term_m.group(1)}-{term_m.group(2)}"
    sec_m = re.search(r"\b(B[A-Z]{1,4}|[A-Z]{2,4})[-\s]?(\d+[A-Z]?)\b", text)
    if sec_m:
        return f"{sec_m.group(1)}-{sec_m.group(2)}"
    return re.sub(r"\s+", "-", text)


def split_class_section(section_code: str) -> tuple[str, str]:
    """Split a canonical section like 'BCS-1A' or 'BSE-1' into `(program, section_suffix)`."""
    norm = normalize_class_section(section_code)
    if "-" in norm:
        prog, sec = norm.split("-", 1)
        return prog.strip(), sec.strip()
    return norm or "BCS", "1A"


def get_program_section_options(catalog_df: pd.DataFrame | None, program: str) -> list[str]:
    """
    Return ordered section suffixes for a given Class/Program (e.g. '1A', '1B', '1', '5A', '5B'),
    putting sections discovered in `catalog_df` first while preserving standard COMSATS choices.
    """
    target_prog = str(program or "").strip().upper()
    discovered: list[str] = []
    if catalog_df is not None and not catalog_df.empty and "section" in catalog_df.columns:
        for raw_sec in catalog_df["section"].dropna().astype(str).tolist():
            p, s = split_class_section(raw_sec)
            if p == target_prog and s and s not in discovered:
                discovered.append(s)

    base_choices = ["1A", "1B", "5A", "5B"] if target_prog == "BCS" else ["1", "1A", "1B", "5A", "5B"]
    ordered: list[str] = []
    for item in [s for s in COMSATS_SECTION_CODES if s in discovered] + sorted(
        s for s in discovered if s not in COMSATS_SECTION_CODES
    ):
        if item not in ordered:
            ordered.append(item)
    for item in base_choices + [s for s in COMSATS_SECTION_CODES if s not in base_choices]:
        if item not in ordered:
            ordered.append(item)
    return ordered


def get_program_options(catalog_df: pd.DataFrame | None) -> list[str]:
    """Return available Class/Program options starting with COMSATS BCS/BSE/BAI/BDS/BCT."""
    options = list(COMSATS_PROGRAMS)
    if catalog_df is not None and not catalog_df.empty and "section" in catalog_df.columns:
        for raw_sec in catalog_df["section"].dropna().astype(str).tolist():
            prog, _ = split_class_section(raw_sec)
            if prog and prog not in options:
                options.append(prog)
    return options


def format_course_label(course_code: str, catalog_df: pd.DataFrame | None = None) -> str:
    """Format a course code as 'CODE — Subject Name' when known."""
    code = course_code.strip().upper()
    name = COMSATS_COURSE_NAMES.get(code)
    if not name and catalog_df is not None and not catalog_df.empty:
        matches = catalog_df.loc[
            catalog_df["course_code"].astype(str).str.upper() == code, "course_name"
        ]
        if not matches.empty:
            name = str(matches.iloc[0]).split(" (Lab)")[0].strip()
    return f"{code} — {name}" if name else code


def parse_course_code(label: str) -> str:
    """Extract the course code from a 'CODE — Subject Name' selector string."""
    return label.split("—", 1)[0].strip().upper()


def get_course_label_options() -> list[str]:
    """Return the standard 5th-semester course choices formatted as 'CODE — Subject Name'."""
    return [format_course_label(code) for code in SEMESTER_5_COURSES]


def validate_time_slot(start_time: str, end_time: str) -> None:
    """Ensure start_time precedes end_time and both are valid HH:MM strings."""
    if _to_minutes(start_time) >= _to_minutes(end_time):
        raise ValueError("End time must be strictly after start time.")


def slots_list_to_df(slots: list[dict]) -> pd.DataFrame:
    """Convert SQLite timetable_slots records into the normalized schedule DataFrame."""
    columns = [
        "id",
        "course_code",
        "course_name",
        "section",
        "day",
        "start_time",
        "end_time",
        "room",
        "instructor",
    ]
    if not slots:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(slots)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    day_order = {day: i for i, day in enumerate(COMSATS_ALL_DAYS)}
    df["_day_order"] = df["day"].map(day_order).fillna(99)
    df = df.sort_values(by=["_day_order", "start_time", "course_code"]).drop(columns=["_day_order"])
    return df[columns].reset_index(drop=True)


def _to_minutes(hhmm: str) -> int:
    """Convert 'HH:MM' 24-hour string into integer minutes since midnight."""
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format '{hhmm}'. Expected HH:MM.")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValueError(f"Out-of-range time '{hhmm}'.")
    return hours * 60 + minutes


def _slots_overlap(
    day_a: str,
    start_a: str,
    end_a: str,
    day_b: str,
    start_b: str,
    end_b: str,
) -> bool:
    """
    Return True if two class meetings strictly overlap in time on the same day.

    Back-to-back slots such as 08:30-10:00 and 10:00-11:30 do NOT clash.
    """
    if day_a.strip().lower() != day_b.strip().lower():
        return False
    a0, a1 = _to_minutes(start_a), _to_minutes(end_a)
    b0, b1 = _to_minutes(start_b), _to_minutes(end_b)
    return a0 < b1 and b0 < a1


def get_sample_catalog_df() -> pd.DataFrame:
    """
    Return DEMO section offerings. Replace with verified CUI timetable data before real enrollment use.

    Each course-section meets twice per week (e.g., Mon/Wed or Tue/Thu) so realistic
    single-day and multi-day clashes are naturally represented.
    """
    rows = [
        # CSC211 - Data Structures & Algorithms
        ("CSC211", "Data Structures & Algorithms", "FA26-BCS-A", "Mon", "08:30", "10:00", "AB2-101", "Dr. Tariq"),
        ("CSC211", "Data Structures & Algorithms", "FA26-BCS-A", "Wed", "08:30", "10:00", "AB2-101", "Dr. Tariq"),
        ("CSC211", "Data Structures & Algorithms", "FA26-BCS-B", "Tue", "10:00", "11:30", "AB2-104", "Dr. Tariq"),
        ("CSC211", "Data Structures & Algorithms", "FA26-BCS-B", "Thu", "10:00", "11:30", "AB2-104", "Dr. Tariq"),
        # CSC301 - Design & Analysis of Algorithms (Sec A deliberately clashes with CSC211 Sec A on Mon 08:30-10:00)
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-A", "Mon", "08:30", "10:00", "AB2-202", "Ms. Ayesha"),
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-A", "Wed", "10:00", "11:30", "AB2-202", "Ms. Ayesha"),
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-B", "Mon", "10:00", "11:30", "AB2-205", "Ms. Ayesha"),
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-B", "Wed", "10:00", "11:30", "AB2-205", "Ms. Ayesha"),
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-C", "Tue", "13:30", "15:00", "AB2-108", "Mr. Salman"),
        ("CSC301", "Design & Analysis of Algorithms", "FA26-BCS-C", "Thu", "13:30", "15:00", "AB2-108", "Mr. Salman"),
        # CSC322 - Operating Systems
        ("CSC322", "Operating Systems", "FA26-BCS-A", "Mon", "11:30", "13:00", "CS-Lab-2", "Dr. Kamran"),
        ("CSC322", "Operating Systems", "FA26-BCS-A", "Wed", "11:30", "13:00", "CS-Lab-2", "Dr. Kamran"),
        ("CSC322", "Operating Systems", "FA26-BCS-B", "Tue", "08:30", "10:00", "AB2-301", "Dr. Kamran"),
        ("CSC322", "Operating Systems", "FA26-BCS-B", "Thu", "08:30", "10:00", "AB2-301", "Dr. Kamran"),
        # CSC241 - Object-Oriented Programming
        ("CSC241", "Object-Oriented Programming", "FA26-BCS-A", "Tue", "11:30", "13:00", "AB2-103", "Mr. Bilal"),
        ("CSC241", "Object-Oriented Programming", "FA26-BCS-A", "Thu", "11:30", "13:00", "AB2-103", "Mr. Bilal"),
        ("CSC241", "Object-Oriented Programming", "FA26-BCS-B", "Fri", "08:30", "10:00", "AB2-106", "Mr. Bilal"),
        ("CSC241", "Object-Oriented Programming", "FA26-BCS-B", "Fri", "10:00", "11:30", "AB2-106", "Mr. Bilal"),
        # MTH104 - Calculus & Analytic Geometry
        ("MTH104", "Calculus & Analytic Geometry", "FA26-BCS-A", "Mon", "13:30", "15:00", "AB1-201", "Dr. Nadia"),
        ("MTH104", "Calculus & Analytic Geometry", "FA26-BCS-A", "Wed", "13:30", "15:00", "AB1-201", "Dr. Nadia"),
        ("MTH104", "Calculus & Analytic Geometry", "FA26-BCS-B", "Fri", "13:30", "15:00", "AB1-204", "Dr. Nadia"),
        # HUM110 - Communication & Presentation Skills
        ("HUM110", "Communication Skills", "FA26-BCS-A", "Tue", "15:00", "16:30", "AB2-110", "Ms. Hira"),
        ("HUM110", "Communication Skills", "FA26-BCS-B", "Fri", "11:30", "13:00", "AB2-112", "Ms. Hira"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "course_code",
            "course_name",
            "section",
            "day",
            "start_time",
            "end_time",
            "room",
            "instructor",
        ],
    )


def get_default_selected_sections() -> dict[str, str]:
    """Default student selection containing one deliberate clash for live demonstration."""
    return {
        "CSC211": "FA26-BCS-A",
        "CSC301": "FA26-BCS-A",  # Clashes with CSC211 FA26-BCS-A on Mon 08:30-10:00
        "CSC322": "FA26-BCS-A",
        "MTH104": "FA26-BCS-A",
    }


def build_selected_schedule_df(
    selected_sections: dict[str, str],
    catalog_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter the catalog DataFrame down to the student's chosen `{course_code: section}` map."""
    df = catalog_df if catalog_df is not None else get_sample_catalog_df()
    if not selected_sections:
        return df.iloc[0:0].copy()

    mask = pd.Series(False, index=df.index)
    for course_code, section in selected_sections.items():
        mask |= (df["course_code"] == course_code) & (df["section"] == section)

    selected = df.loc[mask].copy()
    day_order = {day: i for i, day in enumerate(COMSATS_ALL_DAYS)}
    selected["_day_order"] = selected["day"].map(day_order).fillna(99)
    selected = selected.sort_values(by=["_day_order", "start_time", "course_code"]).drop(
        columns=["_day_order"]
    )
    return selected.reset_index(drop=True)


def detect_clashes(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically detect all overlapping course meetings in `schedule_df`.

    Returns a DataFrame with one row per conflicting course pair.
    """
    columns = [
        "day",
        "course_a",
        "section_a",
        "slot_a",
        "room_a",
        "course_b",
        "section_b",
        "slot_b",
        "room_b",
    ]
    if schedule_df.empty or len(schedule_df) < 2:
        return pd.DataFrame(columns=columns)

    records = schedule_df.to_dict("records")
    clashes: list[dict[str, str]] = []

    for i in range(len(records)):
        row_a = records[i]
        for j in range(i + 1, len(records)):
            row_b = records[j]
            # Every distinct meeting row participates in overlap checks, including
            # duplicate/overlapping rows accidentally entered for the same course.
            if _slots_overlap(
                row_a["day"],
                row_a["start_time"],
                row_a["end_time"],
                row_b["day"],
                row_b["start_time"],
                row_b["end_time"],
            ):
                clashes.append(
                    {
                        "day": row_a["day"],
                        "course_a": row_a["course_code"],
                        "section_a": row_a["section"],
                        "slot_a": f"{row_a['start_time']}–{row_a['end_time']}",
                        "room_a": row_a["room"],
                        "course_b": row_b["course_code"],
                        "section_b": row_b["section"],
                        "slot_b": f"{row_b['start_time']}–{row_b['end_time']}",
                        "room_b": row_b["room"],
                    }
                )

    return pd.DataFrame(clashes, columns=columns)


def recommend_clash_free_alternatives(
    selected_sections: dict[str, str],
    catalog_df: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """
    Find single-course section swaps that reduce or completely eliminate timetable clashes.

    Returns recommendations sorted with zero-clash fixes first.
    """
    df = catalog_df if catalog_df is not None else get_sample_catalog_df()
    current_df = build_selected_schedule_df(selected_sections, df)
    current_clashes = detect_clashes(current_df)
    if current_clashes.empty:
        return []

    clashing_courses = sorted(
        set(current_clashes["course_a"].tolist()) | set(current_clashes["course_b"].tolist())
    )
    current_clash_count = len(current_clashes)

    recommendations: list[dict[str, object]] = []
    for course_code in clashing_courses:
        current_sec = selected_sections.get(course_code)
        alt_sections = sorted(
            sec
            for sec in df.loc[df["course_code"] == course_code, "section"].unique().tolist()
            if sec != current_sec
        )
        for alt_sec in alt_sections:
            candidate_map = dict(selected_sections)
            candidate_map[course_code] = alt_sec
            candidate_df = build_selected_schedule_df(candidate_map, df)
            new_clashes = detect_clashes(candidate_df)
            new_count = len(new_clashes)
            if new_count < current_clash_count:
                alt_meetings = df.loc[
                    (df["course_code"] == course_code) & (df["section"] == alt_sec)
                ]
                slot_summary = ", ".join(
                    f"{r['day']} {r['start_time']}–{r['end_time']} ({r['room']})"
                    for _, r in alt_meetings.iterrows()
                )
                recommendations.append(
                    {
                        "course_code": course_code,
                        "from_section": current_sec,
                        "to_section": alt_sec,
                        "resulting_clashes": new_count,
                        "resolves_all_clashes": new_count == 0,
                        "schedule_summary": slot_summary,
                    }
                )

    recommendations.sort(key=lambda item: (item["resulting_clashes"], str(item["course_code"])))
    return recommendations


def build_weekly_matrix(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the selected course schedule into a clean time-slot x weekday grid."""
    base_slots = list(COMSATS_TIME_SLOTS)
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            pair = (str(row["start_time"]), str(row["end_time"]))
            if pair not in base_slots:
                # Add non-standard time ranges only if they don't fit neatly inside an existing slot row
                r0, r1 = _to_minutes(pair[0]), _to_minutes(pair[1])
                overlaps_existing = any(
                    r0 < _to_minutes(e) and _to_minutes(s) < r1 for s, e in base_slots
                )
                if not overlaps_existing:
                    base_slots.append(pair)
    base_slots.sort(key=lambda pair: (_to_minutes(pair[0]), _to_minutes(pair[1])))

    slot_labels = [f"{s}-{e}" for s, e in base_slots]
    has_sat = not schedule_df.empty and (schedule_df["day"].astype(str) == "Sat").any()
    grid_days = COMSATS_ALL_DAYS if has_sat else COMSATS_DAYS
    grid = pd.DataFrame("", index=slot_labels, columns=grid_days)

    for _, row in schedule_df.iterrows():
        day = row["day"]
        if day not in grid.columns:
            continue
        row_start = _to_minutes(row["start_time"])
        row_end = _to_minutes(row["end_time"])
        room = str(row.get("room") or "").strip()
        sec = str(row.get("section") or "").strip()
        sec_part = f" · {sec}" if sec else ""
        room_part = f" ({room})" if room else ""
        entry = f"{row['course_code']}{sec_part}{room_part}"
        for slot_start_str, slot_end_str in base_slots:
            slot_label = f"{slot_start_str}-{slot_end_str}"
            if row_start < _to_minutes(slot_end_str) and _to_minutes(slot_start_str) < row_end:
                existing = grid.at[slot_label, day]
                grid.at[slot_label, day] = f"{existing} | CLASH: {entry}" if existing else entry

    grid.index.name = "Time"
    return grid


def get_semester_5_catalog_df() -> pd.DataFrame:
    """
    Return the configured Fall 2026 section catalogue across
    BCS-5A, BCS-5B, BCS-1A, BCS-1B, BSE-1, BAI-1, and BCT-1.
    """
    rows = [
        # CSC322 - Operating Systems
        ("CSC322", "Operating Systems", "BCS-5A", "Mon", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCS-5A", "Wed", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCS-5B", "Tue", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCS-5B", "Thu", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCT-1", "Tue", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC322", "Operating Systems", "BCT-1", "Fri", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        # AIC354 - Machine Learning Fundamentals
        ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Mon", "08:30", "10:00", "106 AB-2", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Thu", "11:30", "13:00", "106 AB-2", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5B", "Tue", "13:30", "15:00", "106 AB-2", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5B", "Fri", "08:30", "10:00", "106 AB-2", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BAI-1", "Wed", "13:30", "15:00", "107 AB-2", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BAI-1", "Fri", "11:30", "13:00", "107 AB-2", "Ms. Ayesha Hussain"),
        # CSC325 - Computer Organization & Assembly
        ("CSC325", "Computer Organization & Assembly", "BCS-5A", "Tue", "10:00", "11:30", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC325", "Computer Organization & Assembly", "BCS-5A", "Thu", "10:00", "11:30", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC325", "Computer Organization & Assembly", "BCS-5B", "Mon", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC325", "Computer Organization & Assembly", "BCS-5B", "Wed", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        # CSC301 - Design & Analysis of Algorithms
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Wed", "11:30", "13:00", "107 AB-2", "Mr. Rizwan Rashid"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Fri", "10:00", "11:30", "107 AB-2", "Mr. Rizwan Rashid"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5B", "Tue", "11:30", "13:00", "107 AB-2", "Mr. Rizwan Rashid"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5B", "Thu", "13:30", "15:00", "107 AB-2", "Mr. Rizwan Rashid"),
        ("CSC301", "Design & Analysis of Algorithms", "BSE-1", "Mon", "11:30", "13:00", "106 AB-2", "Ms. Ayesha Hussain"),
        ("CSC301", "Design & Analysis of Algorithms", "BSE-1", "Wed", "15:00", "16:30", "106 AB-2", "Ms. Ayesha Hussain"),
        # Official Fall 2026 Intro Offerings (BCS-1A, BCS-1B)
        ("CSC103", "Programming Fundamentals", "BCS-1A", "Mon", "10:00", "11:30", "CL-5", "Mr. Khurram Iqbal"),
        ("CSC103", "Programming Fundamentals", "BCS-1A", "Wed", "10:00", "11:30", "CL-5", "Mr. Khurram Iqbal"),
        ("CSC101", "Applications of ICT", "BCS-1B", "Tue", "15:00", "16:30", "CL-6", "Mr. Rizwan Rashid"),
        ("CSC101", "Applications of ICT", "BCS-1B", "Fri", "13:30", "15:00", "CL-6", "Mr. Rizwan Rashid"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "course_code",
            "course_name",
            "section",
            "day",
            "start_time",
            "end_time",
            "room",
            "instructor",
        ],
    )


def recommend_slot_swaps(
    schedule_df: pd.DataFrame,
    catalog_df: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """
    Analyze the persisted schedule DataFrame with Pandas and suggest alternative
    COMSATS sections (e.g., BCS-5A -> BCS-5B / BAI-1 / BCT-1) that eliminate clashes.
    """
    current_clashes = detect_clashes(schedule_df)
    if current_clashes.empty:
        return []

    catalog = catalog_df if catalog_df is not None else get_semester_5_catalog_df()
    current_clash_count = len(current_clashes)
    clashing_courses = sorted(
        set(current_clashes["course_a"].astype(str).tolist())
        | set(current_clashes["course_b"].astype(str).tolist())
    )

    recommendations: list[dict[str, object]] = []
    for course_code in clashing_courses:
        enrolled_rows = schedule_df.loc[schedule_df["course_code"] == course_code]
        current_sections = sorted(
            {str(s).strip() for s in enrolled_rows["section"].tolist() if str(s).strip()}
        )
        from_section = current_sections[0] if current_sections else "BCS-5A"

        alt_sections = sorted(
            str(sec)
            for sec in catalog.loc[catalog["course_code"] == course_code, "section"].unique().tolist()
            if str(sec) not in current_sections
        )
        remaining_df = schedule_df.loc[schedule_df["course_code"] != course_code].copy()

        for alt_sec in alt_sections:
            alt_df = catalog.loc[
                (catalog["course_code"] == course_code) & (catalog["section"] == alt_sec)
            ].copy()
            if alt_df.empty:
                continue
            # Compare only schedule fields shared by the catalogue and current plan.
            # Persisted schedules may include an SQLite `id`, while a newly generated
            # student plan does not; that metadata must never affect clash analysis.
            schedule_columns = [
                "course_code", "course_name", "section", "day",
                "start_time", "end_time", "room", "instructor",
            ]
            candidate_df = pd.concat(
                [remaining_df.reindex(columns=schedule_columns), alt_df.reindex(columns=schedule_columns)],
                ignore_index=True,
            )
            new_clashes = detect_clashes(candidate_df)
            new_count = len(new_clashes)
            if new_count < current_clash_count:
                slot_summary = ", ".join(
                    f"{r['day']} {r['start_time']}–{r['end_time']} ({r['room']} · {r['instructor']})"
                    for _, r in alt_df.iterrows()
                )
                replacement_slots = [
                    (
                        str(r["course_code"]),
                        str(r["course_name"]),
                        str(r["section"]),
                        str(r["day"]),
                        str(r["start_time"]),
                        str(r["end_time"]),
                        str(r["room"]),
                        str(r["instructor"]),
                    )
                    for _, r in alt_df.iterrows()
                ]
                recommendations.append(
                    {
                        "course_code": course_code,
                        "course_label": format_course_label(course_code),
                        "from_section": from_section,
                        "to_section": alt_sec,
                        "resulting_clashes": new_count,
                        "resolves_all_clashes": new_count == 0,
                        "schedule_summary": slot_summary,
                        "replacement_slots": replacement_slots,
                    }
                )

    recommendations.sort(
        key=lambda item: (
            int(item["resulting_clashes"]),
            str(item["to_section"]) != "BCS-5B",
            str(item["course_code"]),
            str(item["to_section"]),
        )
    )
    return recommendations


# Standard COMSATS CS/SE/AI progression prerequisites (verify against your official CUOnline portal / outline).
# Missing keys evaluate cleanly to "unknown" rather than guessing.
COURSE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "CSC101": (),
    "CSC103": (),
    "EEE241": (),
    "MTH091": (),
    "HUM104": (),
    "HUM222": (),
    "HUM161": (),
    "MTH231": (),
    "MTH104": ("MTH091",),
    "HUM102": ("HUM104",),
    "HUM110": ("HUM104",),
    "CSC241": ("CSC103",),
    "CSC291": ("CSC103",),
    "CSC211": ("CSC241",),
    "CSC271": ("CSC241",),
    "CSC232": ("CSC241",),
    "MTH262": ("MTH104",),
    "MTH242": ("MTH104",),
    "CSC316": ("CSC271",),
    "CSC339": ("CSC211",),
    "CSC301": ("CSC211",),
    "CSC322": ("CSC211",),
    "CSC325": ("CSC101",),
    "AIC354": ("CSC211",),
    "CSC336": ("CSC211",),
}

COURSE_DOWNSTREAM_CHAINS: dict[str, list[str]] = {
    "CSC103": [
        "CSC241 — Object-Oriented Programming",
        "CSC211 — Data Structures & Algorithms",
        "CSC291 — Software Engineering",
        "CSC271 — Database Systems",
        "CSC232 — Information Security",
        "CSC322 — Operating Systems",
        "CSC301 — Design & Analysis of Algorithms",
        "AIC354 — Machine Learning Fundamentals",
        "CSC336 — Web Technologies",
        "CSC339 — Artificial Intelligence",
    ],
    "CSC101": ["CSC325 — Computer Organization & Assembly Language"],
    "MTH091": ["MTH104 — Calculus & Analytic Geometry", "MTH262 — Linear Algebra", "MTH242 — Probability & Statistics"],
    "HUM104": ["HUM102 — Communication Skills", "HUM110 — Communication & Presentation Skills"],
    "MTH104": ["MTH262 — Linear Algebra", "MTH242 — Probability & Statistics"],
    "CSC241": [
        "CSC211 — Data Structures & Algorithms",
        "CSC271 — Database Systems",
        "CSC232 — Information Security",
        "CSC316 — Advanced Database Systems",
        "CSC322 — Operating Systems",
        "CSC301 — Design & Analysis of Algorithms",
    ],
    "CSC211": [
        "CSC322 — Operating Systems",
        "CSC301 — Design & Analysis of Algorithms",
        "AIC354 — Machine Learning Fundamentals",
        "CSC336 — Web Technologies",
        "CSC339 — Artificial Intelligence",
    ],
    "CSC271": ["CSC316 — Advanced Database Systems"],
}


def check_failed_prereq_impacts_for_regular(
    regular_courses: list[str],
    failed_courses: list[str],
    catalog_df: pd.DataFrame | None = None,
    prerequisite_map: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> list[dict[str, object]]:
    """
    Detect whenever a selected failed/repeater course is a direct OR transitive prerequisite
    for one or more of the student's current regular section courses.
    """
    rules = prerequisite_map if prerequisite_map is not None else COURSE_PREREQUISITES
    reg_list = list(dict.fromkeys(str(c).strip().upper() for c in regular_courses if str(c).strip()))
    failed_list = list(dict.fromkeys(str(c).strip().upper() for c in failed_courses if str(c).strip()))
    if not reg_list or not failed_list:
        return []

    def _find_dependency_path(target_course: str, failed_prereq: str) -> list[str] | None:
        queue: list[tuple[str, list[str]]] = [(target_course, [target_course])]
        seen: set[str] = {target_course}
        while queue:
            cur, path = queue.pop(0)
            for req in rules.get(cur, ()):
                req_u = str(req).strip().upper()
                if req_u == failed_prereq:
                    return list(reversed([*path, req_u]))
                if req_u not in seen:
                    seen.add(req_u)
                    queue.append((req_u, [*path, req_u]))
        return None

    warnings_out: list[dict[str, object]] = []
    for f_code in failed_list:
        dependent_regulars: list[str] = []
        chain_examples: list[str] = []
        direct_hits: list[str] = []
        for reg_code in reg_list:
            path = _find_dependency_path(reg_code, f_code)
            if path:
                dependent_regulars.append(format_course_label(reg_code, catalog_df))
                if len(path) == 2:
                    direct_hits.append(reg_code)
                chain_examples.append(" -> ".join(path))
        if dependent_regulars:
            dep_kind = "direct prerequisite" if direct_hits else "upstream prerequisite chain"
            warnings_out.append(
                {
                    "failed_code": f_code,
                    "failed_label": format_course_label(f_code, catalog_df),
                    "dependent_regular_labels": dependent_regulars,
                    "chains": chain_examples,
                    "is_direct": bool(direct_hits),
                    "headline": (
                        f"Prerequisite Warning: {format_course_label(f_code, catalog_df)} "
                        f"is a {dep_kind} for {', '.join(dependent_regulars)}."
                    ),
                    "detail": "Dependency path(s): " + " | ".join(chain_examples[:3]),
                }
            )
    return warnings_out


def evaluate_prerequisites(
    selected_courses: list[str],
    completed_courses: list[str],
    prerequisite_map: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> list[dict[str, object]]:
    """Evaluate prerequisites from verified catalogue metadata, without guessing."""
    rules = prerequisite_map if prerequisite_map is not None else COURSE_PREREQUISITES
    completed = {str(code).strip().upper() for code in completed_courses if str(code).strip()}
    results = []
    for raw_code in selected_courses:
        code = str(raw_code).strip().upper()
        if code not in rules:
            results.append({"course_code": code, "status": "unknown", "missing": [], "required": []})
            continue
        required = [str(x).strip().upper() for x in rules[code]]
        missing = [x for x in required if x not in completed]
        results.append({
            "course_code": code,
            "status": "eligible" if not missing else "ineligible",
            "missing": missing,
            "required": required,
        })
    return results


def analyze_prerequisite_chains(
    enrolled_courses: list[str],
    failed_courses: list[str],
    passed_courses: list[str],
    prerequisite_map: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> list[dict[str, object]]:
    """
    Evaluate both:
    1. Direct prerequisite blockers for courses currently selected for enrollment.
    2. Downstream prerequisite-chain impacts for any failed / incomplete courses being retaken.
    """
    rules = prerequisite_map if prerequisite_map is not None else COURSE_PREREQUISITES
    failed_set = {str(c).strip().upper() for c in failed_courses if str(c).strip()}
    passed_set = {str(c).strip().upper() for c in passed_courses if str(c).strip()} - failed_set
    enrolled_list = list(dict.fromkeys(str(c).strip().upper() for c in enrolled_courses if str(c).strip()))

    findings: list[dict[str, object]] = []

    for code in enrolled_list:
        req = [str(x).strip().upper() for x in rules.get(code, ())]
        if not req:
            findings.append(
                {
                    "course_code": code,
                    "course_label": format_course_label(code),
                    "severity": "ok",
                    "headline": f"{code}: Eligible (no unmet prerequisite blockers)",
                    "detail": "Ready for current registration.",
                }
            )
            continue
        missing = [r for r in req if r not in passed_set]
        if missing:
            failed_hits = [r for r in missing if r in failed_set]
            cause = (
                f"failed prerequisite {', '.join(failed_hits)}"
                if failed_hits
                else f"uncompleted prerequisite {', '.join(missing)}"
            )
            findings.append(
                {
                    "course_code": code,
                    "course_label": format_course_label(code),
                    "severity": "blocker",
                    "headline": f"{code}: Prerequisite Blocker ({', '.join(missing)})",
                    "detail": f"Cannot advance in {format_course_label(code)} until you clear {cause}.",
                }
            )
        else:
            findings.append(
                {
                    "course_code": code,
                    "course_label": format_course_label(code),
                    "severity": "ok",
                    "headline": f"{code}: Prerequisites Satisfied ({', '.join(req)})",
                    "detail": "All required prerequisites are passed.",
                }
            )

    for f_code in sorted(failed_set):
        downstream = COURSE_DOWNSTREAM_CHAINS.get(f_code, [])
        if downstream:
            findings.append(
                {
                    "course_code": f_code,
                    "course_label": format_course_label(f_code),
                    "severity": "chain_warning",
                    "headline": f"{f_code} Failed/Retake Chain Impact",
                    "detail": (
                        f"Clearing {format_course_label(f_code)} is required to unlock: "
                        + " -> ".join(downstream[:4])
                    ),
                }
            )
        else:
            findings.append(
                {
                    "course_code": f_code,
                    "course_label": format_course_label(f_code),
                    "severity": "ok",
                    "headline": f"{f_code} Retake Status: No Downstream Prerequisite Blocker",
                    "detail": (
                        f"Retaking {format_course_label(f_code)} does not block upper-level progression; "
                        "fit it cleanly into any clash-free slot."
                    ),
                }
            )

    return findings


def find_alternative_classes_in_pdf(
    course_code: str,
    other_schedule_df: pd.DataFrame,
    pdf_catalog_df: pd.DataFrame,
    current_section: str = "",
) -> list[dict[str, object]]:
    """
    Search all classes/sections inside `pdf_catalog_df` offering `course_code`, evaluate each
    offering against `other_schedule_df` via strict interval overlap (`start_a < end_b and start_b < end_a`),
    and return detailed alternatives sorted with ZERO-CLASH sections first.
    """
    code_upper = str(course_code or "").strip().upper()
    cur_sec_norm = normalize_class_section(current_section)
    if pdf_catalog_df.empty or not code_upper:
        return []

    offerings = pdf_catalog_df.loc[
        pdf_catalog_df["course_code"].astype(str).str.upper() == code_upper
    ].copy()
    if offerings.empty:
        return []

    other_rows = other_schedule_df.to_dict("records") if not other_schedule_df.empty else []
    results: list[dict[str, object]] = []

    for sec_val, grp in offerings.groupby("section", sort=False):
        sec_str = normalize_class_section(str(sec_val)) or str(sec_val).strip()
        day_order = {day: i for i, day in enumerate(COMSATS_ALL_DAYS)}
        grp = grp.copy()
        grp["_d"] = grp["day"].map(day_order).fillna(99)
        grp = grp.sort_values(["_d", "start_time"]).drop(columns=["_d"])
        sec_rows = grp.to_dict("records")

        clash_notes: list[str] = []
        for cand_r in sec_rows:
            for fixed_r in other_rows:
                if _slots_overlap(
                    str(cand_r["day"]),
                    str(cand_r["start_time"]),
                    str(cand_r["end_time"]),
                    str(fixed_r["day"]),
                    str(fixed_r["start_time"]),
                    str(fixed_r["end_time"]),
                ):
                    clash_notes.append(
                        f"{cand_r['day']} {cand_r['start_time']}-{cand_r['end_time']} overlaps "
                        f"{fixed_r['course_code']} ({fixed_r['section']} {fixed_r['start_time']}-{fixed_r['end_time']})"
                    )

        instructors = list(
            dict.fromkeys(
                str(r.get("instructor") or "").strip()
                for r in sec_rows
                if str(r.get("instructor") or "").strip() and str(r.get("instructor") or "").strip() != "TBA"
            )
        )
        instructor_str = " / ".join(instructors) if instructors else str(sec_rows[0].get("instructor") or "TBA")

        meeting_summaries = [
            f"{r['day']} {r['start_time']}-{r['end_time']} ({r.get('room', 'TBA')})"
            + (" [Lab]" if "LAB" in str(r.get("course_name", "")).upper() else "")
            for r in sec_rows
        ]
        results.append(
            {
                "course_code": code_upper,
                "course_label": format_course_label(code_upper, pdf_catalog_df),
                "section": sec_str,
                "is_current": sec_str == cur_sec_norm,
                "has_clash": bool(clash_notes),
                "clash_count": len(clash_notes),
                "clash_details": clash_notes,
                "instructor": instructor_str,
                "schedule_summary": " | ".join(meeting_summaries),
                "meetings": sec_rows,
            }
        )

    preferred_order = {s: idx for idx, s in enumerate(COMSATS_FALL_2026_SECTIONS)}
    results.sort(
        key=lambda item: (
            bool(item["has_clash"]),
            int(item["clash_count"]),
            bool(item["is_current"]),
            preferred_order.get(str(item["section"]), 99),
            str(item["section"]),
        )
    )
    return results


def build_student_selection(
    catalog_df: pd.DataFrame,
    cohort_section: str,
    selected_courses: list[str],
    section_choices: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a student's chosen course schedule from catalogue rows, preserving all weekly meetings."""
    required = {"course_code", "course_name", "section", "day", "start_time", "end_time", "room", "instructor"}
    missing = required.difference(catalog_df.columns)
    if missing:
        raise ValueError("Section catalogue is missing columns: " + ", ".join(sorted(missing)))
    choices = section_choices or {}
    frames = []
    for raw_code in selected_courses:
        code = str(raw_code).strip().upper()
        offerings = catalog_df.loc[catalog_df["course_code"].astype(str).str.upper() == code].copy()
        if offerings.empty:
            continue
        preferred = str(choices.get(code) or cohort_section).strip()
        available = offerings["section"].astype(str).unique().tolist()
        chosen = preferred if preferred in available else (available[0] if available else "")
        if chosen:
            frames.append(offerings.loc[offerings["section"].astype(str) == chosen].copy())
    if not frames:
        return catalog_df.iloc[0:0].copy()
    result = pd.concat(frames, ignore_index=True)
    day_order = {day: i for i, day in enumerate(COMSATS_ALL_DAYS)}
    result["_day_order"] = result["day"].map(day_order).fillna(99)
    return result.sort_values(["_day_order", "start_time", "course_code"]).drop(columns="_day_order").reset_index(drop=True)


def section_options_for_course(catalog_df: pd.DataFrame, course_code: str) -> list[str]:
    rows = catalog_df.loc[catalog_df["course_code"].astype(str).str.upper() == str(course_code).upper()]
    return sorted(rows["section"].dropna().astype(str).unique().tolist())


def get_official_comsats_pdf_catalog_df() -> pd.DataFrame:
    """
    Return the exact COMSATS University Islamabad Fall 2026 Undergraduate Timetable
    parsed from the official Department of Computer Science PDF across
    BCS-1A, BCS-1B, BSE-1, BAI-1, BDS-1, and BCT-1.
    """
    rows = [
        # --- Fall 2026 - BCS-1A ---
        ("CSC101", "Applications of ICT", "BCS-1A", "Mon", "08:30", "10:30", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("CSC101", "Applications of ICT (Lab)", "BCS-1A", "Thu", "08:30", "11:30", "CL-7 (AB-2 First Floor)", "Ms. Ayesha Hussain"),
        ("CSC103", "Programming Fundamentals", "BCS-1A", "Mon", "11:30", "13:00", "107 (AB-2)", "Mr. Rizwan Rashid"),
        ("CSC103", "Programming Fundamentals", "BCS-1A", "Tue", "11:30", "13:00", "107 (AB-2)", "Mr. Rizwan Rashid"),
        ("CSC103", "Programming Fundamentals (Lab)", "BCS-1A", "Thu", "11:30", "14:30", "CL-7 (AB-2 First Floor)", "Mr. Rizwan Rashid"),
        ("HUM222", "Fundamentals of International Relations", "BCS-1A", "Tue", "08:30", "10:30", "106 (AB-2)", "Mr. Attiq ur Rehman (VF)"),
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BCS-1A", "Wed", "08:30", "11:30", "G05 (AB-2)", "Dr. Fawad Ali"),
        ("HUM104", "Functional English", "BCS-1A", "Sat", "08:30", "11:30", "G-08 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BCS-1A", "Sat", "12:00", "14:00", "Seminar", "Dr. Zainab Sadiq"),
        # --- Fall 2026 - BCS-1B ---
        ("CSC103", "Programming Fundamentals", "BCS-1B", "Mon", "08:30", "10:00", "107 (AB-2)", "Mr. Khurram Iqbal"),
        ("CSC103", "Programming Fundamentals", "BCS-1B", "Tue", "08:30", "10:00", "107 (AB-2)", "Mr. Khurram Iqbal"),
        ("CSC103", "Programming Fundamentals (Lab)", "BCS-1B", "Thu", "08:30", "11:30", "CL-8 (AB-2 First Floor)", "Mr. Khurram Iqbal"),
        ("HUM222", "Fundamentals of International Relations", "BCS-1B", "Mon", "10:30", "12:30", "106 (AB-2)", "Mr. Attiq ur Rehman (VF)"),
        ("CSC101", "Applications of ICT", "BCS-1B", "Tue", "10:30", "12:30", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("CSC101", "Applications of ICT (Lab)", "BCS-1B", "Thu", "11:30", "14:30", "CL-8 (AB-2 First Floor)", "Ms. Waliya Fatima"),
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BCS-1B", "Wed", "08:30", "11:30", "G05 (AB-2)", "Dr. Fawad Ali"),
        ("HUM104", "Functional English", "BCS-1B", "Sat", "08:30", "11:30", "G-08 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BCS-1B", "Sat", "11:30", "13:00", "Seminar", "Dr. Zainab Sadiq"),
        # --- Fall 2026 - BSE-1 ---
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BSE-1", "Mon", "08:30", "11:30", "G05 (AB-2)", "Dr. Iffat Jabeen"),
        ("CSC101", "Applications of ICT (Lab)", "BSE-1", "Tue", "08:30", "11:30", "CL-7 (AB-2 First Floor)", "Ms. Sana Nasir"),
        ("CSC101", "Applications of ICT", "BSE-1", "Wed", "08:30", "10:30", "106 (AB-2)", "Ms. Hilal Jan"),
        ("CSC103", "Programming Fundamentals (Lab)", "BSE-1", "Tue", "11:30", "14:30", "CL-7 (AB-2 First Floor)", "Mr. Muzafar Iqbal"),
        ("CSC103", "Programming Fundamentals", "BSE-1", "Wed", "11:30", "13:00", "107 (AB-2)", "Mr. Muzafar Iqbal"),
        ("CSC103", "Programming Fundamentals", "BSE-1", "Thu", "11:30", "13:00", "107 (AB-2)", "Mr. Muzafar Iqbal"),
        ("HUM222", "Fundamentals of International Relations", "BSE-1", "Thu", "08:30", "10:30", "106 (AB-2)", "Mr. Attiq ur Rehman (VF)"),
        ("HUM104", "Functional English", "BSE-1", "Sat", "08:30", "11:30", "G-05 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BSE-1", "Sat", "12:00", "14:00", "Seminar", "Dr. Zainab Sadiq"),
        # --- Fall 2026 - BAI-1 ---
        ("CSC103", "Programming Fundamentals", "BAI-1", "Mon", "08:30", "10:00", "110 (AB-2)", "Prof. Dr. M. Manzoor Illahi Tamimy"),
        ("CSC103", "Programming Fundamentals", "BAI-1", "Tue", "08:30", "10:00", "110 (AB-2)", "Prof. Dr. M. Manzoor Illahi Tamimy"),
        ("CSC103", "Programming Fundamentals (Lab)", "BAI-1", "Tue", "11:30", "14:30", "CL-6 (AB-2 Ground Floor)", "Prof. Dr. M. Manzoor Illahi Tamimy"),
        ("CSC101", "Applications of ICT", "BAI-1", "Mon", "12:30", "14:30", "111 (AB-2)", "Ms. Haseena Kainat"),
        ("CSC101", "Applications of ICT (Lab)", "BAI-1", "Thu", "11:30", "14:30", "CL-5 (AB-2 Ground Floor)", "Ms. Haseena Kainat"),
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BAI-1", "Wed", "11:30", "14:30", "107 (AB-1)", "Dr. Iffat Jabeen"),
        ("HUM222", "Fundamentals of International Relations", "BAI-1", "Thu", "08:30", "10:30", "111 (AB-2)", "Ms. Shazia Mehboob (VF)"),
        ("HUM104", "Functional English", "BAI-1", "Sat", "08:30", "11:30", "G-05 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BAI-1", "Sat", "12:00", "14:00", "Seminar", "Dr. Zainab Sadiq"),
        # --- Fall 2026 - BDS-1 ---
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BDS-1", "Mon", "08:30", "11:30", "G05 (AB-2)", "Dr. Iffat Jabeen"),
        ("CSC103", "Programming Fundamentals (Lab)", "BDS-1", "Tue", "08:30", "11:30", "CL-8 (AB-2 First Floor)", "Mr. Rizwan Rashid"),
        ("CSC103", "Programming Fundamentals", "BDS-1", "Wed", "10:00", "11:30", "107 (AB-2)", "Mr. Rizwan Rashid"),
        ("CSC103", "Programming Fundamentals", "BDS-1", "Thu", "10:00", "11:30", "107 (AB-2)", "Mr. Rizwan Rashid"),
        ("CSC101", "Applications of ICT (Lab)", "BDS-1", "Tue", "11:30", "14:30", "CL-8 (AB-2 First Floor)", "Mr. Waqas Ali"),
        ("CSC101", "Applications of ICT", "BDS-1", "Wed", "12:30", "14:30", "106 (AB-2)", "Mr. Waqas Ali"),
        ("HUM222", "Fundamentals of International Relations", "BDS-1", "Thu", "12:30", "14:30", "106 (AB-2)", "Ms. Shazia Mehboob (VF)"),
        ("HUM104", "Functional English", "BDS-1", "Sat", "08:30", "11:30", "G-06 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BDS-1", "Sat", "12:00", "14:00", "Seminar", "Dr. Zainab Sadiq"),
        # --- Fall 2026 - BCT-1 ---
        ("CSC103", "Programming Fundamentals", "BCT-1", "Mon", "10:00", "11:30", "107 (AB-2)", "Ms. Asma Khalid Butt"),
        ("CSC103", "Programming Fundamentals", "BCT-1", "Tue", "10:00", "11:30", "107 (AB-2)", "Ms. Asma Khalid Butt"),
        ("CSC103", "Programming Fundamentals (Lab)", "BCT-1", "Thu", "08:30", "11:30", "CL-11 (AB-2 Third Floor)", "Ms. Asma Khalid Butt"),
        ("HUM222", "Fundamentals of International Relations", "BCT-1", "Mon", "12:30", "14:30", "106 (AB-2)", "Ms. Shazia Mehboob (VF)"),
        ("CSC101", "Applications of ICT", "BCT-1", "Tue", "12:30", "14:30", "106 (AB-2)", "Ms. Asma Khalid Butt"),
        ("CSC101", "Applications of ICT (Lab)", "BCT-1", "Thu", "11:30", "14:30", "CL-11 (AB-2 Third Floor)", "Dr. Tehseen Riaz Abbasi"),
        ("MTH091", "Pre-Calculus I (Pre-Med)", "BCT-1", "Wed", "11:30", "14:30", "107 (AB-1)", "Dr. Iffat Jabeen"),
        ("HUM104", "Functional English", "BCT-1", "Sat", "08:30", "11:30", "G-06 (AB-2)", "Mr. Ali Naqi (VF)"),
        ("HUM161", "Fehm ul Quran-I", "BCT-1", "Sat", "12:00", "14:00", "Seminar", "Staff"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "course_code",
            "course_name",
            "section",
            "day",
            "start_time",
            "end_time",
            "room",
            "instructor",
        ],
    )


def get_cs_master_catalog_df() -> pd.DataFrame:
    """
    Return the built-in COMSATS CS Department Master Timetable covering all regular classes
    (`BCS-1A`..`BCS-5B`, `BSE-1`, `BAI-1`, `BDS-1`, `BCT-1`) plus `RPT-CS` repeater offerings.
    Every single regular cohort and `RPT-CS` is internally conflict-free.
    """
    first_sem_df = get_official_comsats_pdf_catalog_df()
    upper_and_rpt_rows = [
        # --- BCS-2A ---
        ("CSC241", "Object-Oriented Programming", "BCS-2A", "Mon", "10:00", "11:30", "108 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming", "BCS-2A", "Wed", "10:00", "11:30", "108 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming (Lab)", "BCS-2A", "Fri", "11:30", "14:30", "CL-3 (AB-2)", "Mr. Waqar Hussain"),
        ("EEE241", "Digital Logic Design", "BCS-2A", "Mon", "08:30", "10:00", "109 (AB-2)", "Engr. Bilal Zafar"),
        ("EEE241", "Digital Logic Design", "BCS-2A", "Wed", "08:30", "10:00", "109 (AB-2)", "Engr. Bilal Zafar"),
        ("EEE241", "Digital Logic Design (Lab)", "BCS-2A", "Mon", "13:30", "16:30", "DLD Lab-1", "Engr. Bilal Zafar"),
        ("MTH104", "Calculus & Analytic Geometry", "BCS-2A", "Tue", "08:30", "10:00", "G06 (AB-2)", "Dr. Nadia Pervez"),
        ("MTH104", "Calculus & Analytic Geometry", "BCS-2A", "Thu", "08:30", "10:00", "G06 (AB-2)", "Dr. Nadia Pervez"),
        ("HUM102", "Communication Skills", "BCS-2A", "Tue", "10:00", "11:30", "G07 (AB-2)", "Ms. Hira Tariq"),
        ("HUM102", "Communication Skills", "BCS-2A", "Thu", "10:00", "11:30", "G07 (AB-2)", "Ms. Hira Tariq"),
        ("MTH231", "Discrete Structures", "BCS-2A", "Mon", "11:30", "13:00", "108 (AB-2)", "Dr. Saira Bano"),
        ("MTH231", "Discrete Structures", "BCS-2A", "Wed", "11:30", "13:00", "108 (AB-2)", "Dr. Saira Bano"),
        # --- BCS-2B ---
        ("CSC241", "Object-Oriented Programming", "BCS-2B", "Tue", "10:00", "11:30", "109 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming", "BCS-2B", "Thu", "10:00", "11:30", "109 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming (Lab)", "BCS-2B", "Wed", "13:30", "16:30", "CL-3 (AB-2)", "Mr. Waqar Hussain"),
        ("EEE241", "Digital Logic Design", "BCS-2B", "Tue", "13:30", "15:00", "110 (AB-2)", "Engr. Bilal Zafar"),
        ("EEE241", "Digital Logic Design", "BCS-2B", "Thu", "13:30", "15:00", "110 (AB-2)", "Engr. Bilal Zafar"),
        ("EEE241", "Digital Logic Design (Lab)", "BCS-2B", "Fri", "13:30", "16:30", "DLD Lab-2", "Engr. Bilal Zafar"),
        ("MTH104", "Calculus & Analytic Geometry", "BCS-2B", "Mon", "10:00", "11:30", "G06 (AB-2)", "Dr. Nadia Pervez"),
        ("MTH104", "Calculus & Analytic Geometry", "BCS-2B", "Wed", "10:00", "11:30", "G06 (AB-2)", "Dr. Nadia Pervez"),
        ("HUM102", "Communication Skills", "BCS-2B", "Mon", "11:30", "13:00", "G07 (AB-2)", "Ms. Hira Tariq"),
        ("HUM102", "Communication Skills", "BCS-2B", "Wed", "11:30", "13:00", "G07 (AB-2)", "Ms. Hira Tariq"),
        ("MTH231", "Discrete Structures", "BCS-2B", "Fri", "08:30", "10:00", "109 (AB-2)", "Dr. Saira Bano"),
        ("MTH231", "Discrete Structures", "BCS-2B", "Fri", "10:00", "11:30", "109 (AB-2)", "Dr. Saira Bano"),
        # --- BCS-3A ---
        ("CSC211", "Data Structures & Algorithms", "BCS-3A", "Mon", "08:30", "10:00", "110 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms", "BCS-3A", "Wed", "08:30", "10:00", "110 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms (Lab)", "BCS-3A", "Tue", "11:30", "14:30", "CL-4 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC291", "Software Engineering", "BCS-3A", "Tue", "08:30", "10:00", "110 (AB-2)", "Ms. Sadia Bibi"),
        ("CSC291", "Software Engineering", "BCS-3A", "Thu", "08:30", "10:00", "110 (AB-2)", "Ms. Sadia Bibi"),
        ("MTH262", "Linear Algebra", "BCS-3A", "Mon", "10:00", "11:30", "G08 (AB-2)", "Dr. Kamran Haider"),
        ("MTH262", "Linear Algebra", "BCS-3A", "Wed", "10:00", "11:30", "G08 (AB-2)", "Dr. Kamran Haider"),
        ("CSC271", "Database Systems", "BCS-3A", "Thu", "10:00", "11:30", "111 (AB-2)", "Dr. Umair Arshad"),
        ("CSC271", "Database Systems", "BCS-3A", "Fri", "08:30", "10:00", "111 (AB-2)", "Dr. Umair Arshad"),
        ("CSC271", "Database Systems (Lab)", "BCS-3A", "Thu", "11:30", "14:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        # --- BCS-3B ---
        ("CSC211", "Data Structures & Algorithms", "BCS-3B", "Tue", "10:00", "11:30", "111 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms", "BCS-3B", "Thu", "10:00", "11:30", "111 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms (Lab)", "BCS-3B", "Wed", "13:30", "16:30", "CL-4 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC291", "Software Engineering", "BCS-3B", "Mon", "13:30", "15:00", "111 (AB-2)", "Ms. Sadia Bibi"),
        ("CSC291", "Software Engineering", "BCS-3B", "Wed", "11:30", "13:00", "111 (AB-2)", "Ms. Sadia Bibi"),
        ("MTH262", "Linear Algebra", "BCS-3B", "Tue", "13:30", "15:00", "G08 (AB-2)", "Dr. Kamran Haider"),
        ("MTH262", "Linear Algebra", "BCS-3B", "Thu", "13:30", "15:00", "G08 (AB-2)", "Dr. Kamran Haider"),
        ("CSC271", "Database Systems", "BCS-3B", "Mon", "10:00", "11:30", "110 (AB-2)", "Dr. Umair Arshad"),
        ("CSC271", "Database Systems", "BCS-3B", "Fri", "10:00", "11:30", "110 (AB-2)", "Dr. Umair Arshad"),
        ("CSC271", "Database Systems (Lab)", "BCS-3B", "Fri", "13:30", "16:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        # --- BCS-4A ---
        ("CSC316", "Advanced Database Systems", "BCS-4A", "Mon", "10:00", "11:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC316", "Advanced Database Systems", "BCS-4A", "Wed", "10:00", "11:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC316", "Advanced Database Systems (Lab)", "BCS-4A", "Tue", "13:30", "16:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC232", "Information Security", "BCS-4A", "Tue", "08:30", "10:00", "112 (AB-2)", "Dr. Adeel Akram"),
        ("CSC232", "Information Security", "BCS-4A", "Thu", "08:30", "10:00", "112 (AB-2)", "Dr. Adeel Akram"),
        ("MTH242", "Probability & Statistics", "BCS-4A", "Mon", "13:30", "15:00", "G09 (AB-2)", "Dr. Tehseen Zafar"),
        ("MTH242", "Probability & Statistics", "BCS-4A", "Wed", "13:30", "15:00", "G09 (AB-2)", "Dr. Tehseen Zafar"),
        ("CSC339", "Artificial Intelligence", "BCS-4A", "Thu", "10:00", "11:30", "112 (AB-2)", "Dr. Ali Javed"),
        ("CSC339", "Artificial Intelligence", "BCS-4A", "Fri", "10:00", "11:30", "112 (AB-2)", "Dr. Ali Javed"),
        ("CSC339", "Artificial Intelligence (Lab)", "BCS-4A", "Fri", "13:30", "16:30", "CL-6 (AB-2)", "Dr. Ali Javed"),
        # --- BCS-4B ---
        ("CSC316", "Advanced Database Systems", "BCS-4B", "Tue", "10:00", "11:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC316", "Advanced Database Systems", "BCS-4B", "Thu", "10:00", "11:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC316", "Advanced Database Systems (Lab)", "BCS-4B", "Mon", "13:30", "16:30", "CL-5 (AB-2)", "Dr. Umair Arshad"),
        ("CSC232", "Information Security", "BCS-4B", "Mon", "08:30", "10:00", "112 (AB-2)", "Dr. Adeel Akram"),
        ("CSC232", "Information Security", "BCS-4B", "Wed", "08:30", "10:00", "112 (AB-2)", "Dr. Adeel Akram"),
        ("MTH242", "Probability & Statistics", "BCS-4B", "Tue", "13:30", "15:00", "G09 (AB-2)", "Dr. Tehseen Zafar"),
        ("MTH242", "Probability & Statistics", "BCS-4B", "Thu", "13:30", "15:00", "G09 (AB-2)", "Dr. Tehseen Zafar"),
        ("CSC339", "Artificial Intelligence", "BCS-4B", "Mon", "11:30", "13:00", "112 (AB-2)", "Dr. Ali Javed"),
        ("CSC339", "Artificial Intelligence", "BCS-4B", "Wed", "11:30", "13:00", "112 (AB-2)", "Dr. Ali Javed"),
        ("CSC339", "Artificial Intelligence (Lab)", "BCS-4B", "Fri", "08:30", "11:30", "CL-6 (AB-2)", "Dr. Ali Javed"),
        # --- BCS-5A ---
        ("CSC322", "Operating Systems", "BCS-5A", "Mon", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCS-5A", "Wed", "08:30", "10:00", "CL-5", "Mr. Rizwan Rashid"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Mon", "11:30", "13:00", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5A", "Fri", "11:30", "13:00", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("CSC325", "Computer Organization & Assembly Language", "BCS-5A", "Tue", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC325", "Computer Organization & Assembly Language", "BCS-5A", "Thu", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Wed", "13:30", "15:00", "107 (AB-2)", "Dr. Nadeem Javaid"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5A", "Fri", "08:30", "10:00", "107 (AB-2)", "Dr. Nadeem Javaid"),
        ("CSC336", "Web Technologies", "BCS-5A", "Thu", "15:00", "16:30", "CL-4 (AB-2)", "Mr. Salman Tariq"),
        ("CSC336", "Web Technologies", "BCS-5A", "Fri", "10:00", "11:30", "CL-4 (AB-2)", "Mr. Salman Tariq"),
        # --- BCS-5B ---
        ("CSC322", "Operating Systems", "BCS-5B", "Tue", "10:00", "11:30", "CL-5", "Mr. Rizwan Rashid"),
        ("CSC322", "Operating Systems", "BCS-5B", "Thu", "10:00", "11:30", "CL-5", "Mr. Rizwan Rashid"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5B", "Tue", "08:30", "10:00", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("AIC354", "Machine Learning Fundamentals", "BCS-5B", "Thu", "08:30", "10:00", "106 (AB-2)", "Ms. Ayesha Hussain"),
        ("CSC325", "Computer Organization & Assembly Language", "BCS-5B", "Mon", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC325", "Computer Organization & Assembly Language", "BCS-5B", "Wed", "13:30", "15:00", "CL-6", "Mr. Khurram Iqbal"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5B", "Mon", "10:00", "11:30", "107 (AB-2)", "Dr. Nadeem Javaid"),
        ("CSC301", "Design & Analysis of Algorithms", "BCS-5B", "Wed", "10:00", "11:30", "107 (AB-2)", "Dr. Nadeem Javaid"),
        ("CSC336", "Web Technologies", "BCS-5B", "Mon", "15:00", "16:30", "CL-4 (AB-2)", "Mr. Salman Tariq"),
        ("CSC336", "Web Technologies", "BCS-5B", "Fri", "13:30", "15:00", "CL-4 (AB-2)", "Mr. Salman Tariq"),
        # --- RPT-CS (Dedicated CS Repeater Sections) ---
        ("CSC103", "Programming Fundamentals", "RPT-CS", "Fri", "13:30", "15:00", "CL-9 (AB-2)", "Mr. Muzafar Iqbal"),
        ("CSC103", "Programming Fundamentals (Lab)", "RPT-CS", "Sat", "11:30", "14:00", "CL-9 (AB-2)", "Mr. Muzafar Iqbal"),
        ("CSC241", "Object-Oriented Programming", "RPT-CS", "Tue", "15:00", "16:30", "109 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming", "RPT-CS", "Thu", "11:30", "13:00", "109 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC241", "Object-Oriented Programming (Lab)", "RPT-CS", "Sat", "14:00", "16:30", "CL-9 (AB-2)", "Mr. Waqar Hussain"),
        ("CSC211", "Data Structures & Algorithms", "RPT-CS", "Mon", "15:00", "16:30", "110 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms", "RPT-CS", "Wed", "15:00", "16:30", "110 (AB-2)", "Dr. Tariq Mehmood"),
        ("CSC211", "Data Structures & Algorithms (Lab)", "RPT-CS", "Sat", "08:30", "11:30", "CL-10 (AB-2)", "Dr. Tariq Mehmood"),
        ("MTH104", "Calculus & Analytic Geometry", "RPT-CS", "Wed", "11:30", "13:00", "G10 (AB-2)", "Dr. Nadia Pervez"),
        ("MTH104", "Calculus & Analytic Geometry", "RPT-CS", "Fri", "15:00", "16:30", "G10 (AB-2)", "Dr. Nadia Pervez"),
    ]
    upper_df = pd.DataFrame(upper_and_rpt_rows, columns=list(first_sem_df.columns))
    combined = pd.concat([first_sem_df, upper_df], ignore_index=True)
    return combined.drop_duplicates(
        subset=["course_code", "section", "day", "start_time", "end_time"],
        keep="first",
    ).reset_index(drop=True)


def get_regular_class_section_options(catalog_df: pd.DataFrame | None = None) -> list[str]:
    """Return ordered list of selectable regular Class / Sections (excluding RPT-CS repeater pool)."""
    ordered = list(MASTER_REGULAR_SECTIONS)
    if catalog_df is not None and not catalog_df.empty and "section" in catalog_df.columns:
        for raw_sec in catalog_df["section"].dropna().astype(str).tolist():
            norm = normalize_class_section(raw_sec)
            if norm and not norm.startswith("RPT-") and norm not in ordered:
                ordered.append(norm)
    return ordered


def merge_pdf_catalog_with_master(
    pdf_df: pd.DataFrame | None,
    master_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge extracted PDF timetable rows cleanly into the built-in CS Department Master Timetable."""
    base = master_df.copy() if master_df is not None else get_cs_master_catalog_df()
    if pdf_df is None or pdf_df.empty:
        return base
    extra = pdf_df.reindex(columns=base.columns).copy()
    merged = pd.concat([base, extra], ignore_index=True)
    return merged.drop_duplicates(
        subset=["course_code", "section", "day", "start_time", "end_time"],
        keep="last",
    ).reset_index(drop=True)


def analyze_remaining_course_sections(
    remaining_code: str,
    baseline_schedule_df: pd.DataFrame,
    master_catalog_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Compare every section offering `remaining_code` (including regular classes & `RPT-CS`)
    against `baseline_schedule_df` (`start_a < end_b and start_b < end_a`).
    Returns green clash-free sections first, then clashing sections with exact conflict text.
    """
    code_u = str(remaining_code or "").strip().upper()
    if master_catalog_df.empty or not code_u:
        return []
    offerings = master_catalog_df.loc[
        master_catalog_df["course_code"].astype(str).str.upper() == code_u
    ].copy()
    if offerings.empty:
        return []

    base_rows = baseline_schedule_df.to_dict("records") if not baseline_schedule_df.empty else []
    day_order = {day: i for i, day in enumerate(COMSATS_ALL_DAYS)}
    results: list[dict[str, object]] = []

    for sec_val, grp in offerings.groupby("section", sort=False):
        sec_str = normalize_class_section(str(sec_val)) or str(sec_val).strip()
        grp = grp.copy()
        grp["_d"] = grp["day"].map(day_order).fillna(99)
        grp = grp.sort_values(["_d", "start_time"]).drop(columns=["_d"])
        sec_rows = grp.to_dict("records")

        clash_reasons: list[str] = []
        for cand_r in sec_rows:
            for base_r in base_rows:
                if _slots_overlap(
                    str(cand_r["day"]),
                    str(cand_r["start_time"]),
                    str(cand_r["end_time"]),
                    str(base_r["day"]),
                    str(base_r["start_time"]),
                    str(base_r["end_time"]),
                ):
                    base_title = str(base_r.get("course_name") or COMSATS_COURSE_NAMES.get(str(base_r["course_code"]), "")).split(" (Lab)")[0].strip()
                    reason = (
                        f"Clashes with {base_r['course_code']} {base_title} on "
                        f"{base_r['day']} {base_r['start_time']}–{base_r['end_time']}"
                    )
                    if reason not in clash_reasons:
                        clash_reasons.append(reason)

        instructors = list(
            dict.fromkeys(
                str(r.get("instructor") or "").strip()
                for r in sec_rows
                if str(r.get("instructor") or "").strip() and str(r.get("instructor") or "").strip() != "TBA"
            )
        )
        instructor_str = " / ".join(instructors) if instructors else str(sec_rows[0].get("instructor") or "TBA")
        days_list = list(dict.fromkeys(str(r["day"]) for r in sec_rows))
        times_list = [
            f"{r['day']} {r['start_time']}–{r['end_time']}"
            + (" (Lab)" if "LAB" in str(r.get("course_name", "")).upper() else "")
            for r in sec_rows
        ]
        rooms_list = list(dict.fromkeys(str(r.get("room") or "TBA") for r in sec_rows))
        slot_summary = " | ".join(
            f"{r['day']} {r['start_time']}–{r['end_time']} ({r.get('room', 'TBA')})"
            + (" [Lab]" if "LAB" in str(r.get("course_name", "")).upper() else "")
            for r in sec_rows
        )
        results.append(
            {
                "course_code": code_u,
                "course_label": format_course_label(code_u, master_catalog_df),
                "section": sec_str,
                "has_clash": bool(clash_reasons),
                "clash_count": len(clash_reasons),
                "conflict_reasons": clash_reasons,
                "conflict_text": " ; ".join(clash_reasons),
                "days_summary": ", ".join(days_list),
                "time_summary": ", ".join(times_list),
                "room_summary": ", ".join(rooms_list),
                "instructor": instructor_str,
                "schedule_summary": slot_summary,
                "meetings": sec_rows,
            }
        )

    preferred_order = {s: idx for idx, s in enumerate(COMSATS_FALL_2026_SECTIONS)}
    results.sort(
        key=lambda item: (
            bool(item["has_clash"]),
            int(item["clash_count"]),
            str(item["section"]) == "RPT-CS",
            preferred_order.get(str(item["section"]), 99),
            str(item["section"]),
        )
    )
    return results


def find_smart_regular_section_swap(
    remaining_code: str,
    regular_courses: list[str],
    regular_sections_map: dict[str, str],
    master_catalog_df: pd.DataFrame,
    locked_remaining_df: pd.DataFrame | None = None,
) -> dict[str, object] | None:
    """
    When EVERY section offering `remaining_code` clashes with the student's regular schedule,
    search for a sibling swap on a regular course (e.g., moving `CSC322` from `BCS-5A` -> `BCS-5B`)
    that frees the conflicting time slot cleanly and yields a zero-clash combined schedule.
    """
    code_u = str(remaining_code or "").strip().upper()
    if master_catalog_df.empty or not code_u:
        return None

    locked_rows = (
        locked_remaining_df.to_dict("records")
        if locked_remaining_df is not None and not locked_remaining_df.empty
        else []
    )
    norm_sec = master_catalog_df["section"].astype(str).apply(normalize_class_section)

    def _rows_for(c_code: str, sec_name: str) -> list[dict]:
        sub = master_catalog_df.loc[
            (master_catalog_df["course_code"].astype(str).str.upper() == c_code.upper())
            & (norm_sec == normalize_class_section(sec_name))
        ]
        return sub.to_dict("records")

    def _any_conflict(rows_a: list[dict], rows_b: list[dict]) -> bool:
        for ra in rows_a:
            for rb in rows_b:
                if _slots_overlap(
                    str(ra["day"]), str(ra["start_time"]), str(ra["end_time"]),
                    str(rb["day"]), str(rb["start_time"]), str(rb["end_time"]),
                ):
                    return True
        return False

    rem_offerings = master_catalog_df.loc[
        master_catalog_df["course_code"].astype(str).str.upper() == code_u
    ]
    rem_sections = sorted(
        {normalize_class_section(str(s)) for s in rem_offerings["section"].dropna().astype(str).tolist()}
    )
    if not rem_sections:
        return None

    # First search single regular-course swaps (minimal schedule change)
    for swap_reg in regular_courses:
        orig_sec = normalize_class_section(str(regular_sections_map.get(swap_reg) or ""))
        reg_offerings = master_catalog_df.loc[
            master_catalog_df["course_code"].astype(str).str.upper() == swap_reg.upper()
        ]
        alt_reg_secs = sorted(
            {
                normalize_class_section(str(s))
                for s in reg_offerings["section"].dropna().astype(str).tolist()
                if normalize_class_section(str(s)) and normalize_class_section(str(s)) != orig_sec
            }
        )
        if not alt_reg_secs:
            continue

        other_reg_rows: list[dict] = list(locked_rows)
        for rc in regular_courses:
            if rc != swap_reg:
                other_reg_rows.extend(_rows_for(rc, str(regular_sections_map.get(rc) or "")))

        for alt_r_sec in alt_reg_secs:
            alt_r_rows = _rows_for(swap_reg, alt_r_sec)
            if not alt_r_rows or _any_conflict(alt_r_rows, other_reg_rows):
                continue
            combined_reg = other_reg_rows + alt_r_rows

            for rem_sec in rem_sections:
                rem_rows = _rows_for(code_u, rem_sec)
                if rem_rows and not _any_conflict(rem_rows, combined_reg):
                    rem_summary = " | ".join(
                        f"{r['day']} {r['start_time']}–{r['end_time']} ({r.get('room', 'TBA')})"
                        for r in rem_rows
                    )
                    swap_summary = " | ".join(
                        f"{r['day']} {r['start_time']}–{r['end_time']} ({r.get('room', 'TBA')})"
                        for r in alt_r_rows
                    )
                    return {
                        "remaining_code": code_u,
                        "remaining_label": format_course_label(code_u, master_catalog_df),
                        "remaining_target_section": rem_sec,
                        "remaining_summary": rem_summary,
                        "regular_swaps": [
                            {
                                "course_code": swap_reg,
                                "course_label": format_course_label(swap_reg, master_catalog_df),
                                "from_section": orig_sec,
                                "to_section": alt_r_sec,
                                "new_summary": swap_summary,
                            }
                        ],
                        "description": (
                            f"Move {swap_reg} ({orig_sec} -> {alt_r_sec}) to add {code_u} ({rem_sec}) with 0 clashes"
                        ),
                    }
    return None


def get_combined_comsats_catalog_df(extra_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return the unified Fall 2026 COMSATS section catalogue combining:
    - All 6 exact Fall 2026 sections from the official COMSATS PDF (BCS-1A..BCT-1)
    - Configured upper-semester CS sections (BCS-5A, BCS-5B, etc.)
    - Optional user-uploaded PDF/CSV rows (merged cleanly without duplicate slots).
    """
    pdf_df = get_official_comsats_pdf_catalog_df()
    sem5_raw = get_semester_5_catalog_df()
    sem5_upper = sem5_raw.loc[
        ~sem5_raw["course_code"].isin(["CSC101", "CSC103"])
    ].copy()
    frames = [pdf_df, sem5_upper]
    if extra_df is not None and not extra_df.empty:
        frames.append(extra_df.reindex(columns=pdf_df.columns))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["course_code", "section", "day", "start_time", "end_time"],
        keep="last",
    ).reset_index(drop=True)
    return combined


def get_cohort_default_courses(catalog_df: pd.DataFrame, cohort_section: str) -> list[str]:
    """Return all canonical courses offered inside the chosen cohort section."""
    sec = normalize_class_section(str(cohort_section or ""))
    if catalog_df.empty:
        return []
    sub = catalog_df.loc[catalog_df["section"].astype(str).apply(normalize_class_section) == sec]
    if sub.empty:
        return []
    return list(dict.fromkeys(sub["course_code"].astype(str).str.upper().tolist()))


def _clean_pdf_piece(raw: str) -> str:
    """Clean PDF literal-string escapes and trailing Crystal/Word backslash artifacts."""
    txt = str(raw or "")
    # Protect escaped backslashes cleanly via placeholder, unescape parens/whitespace, then strip stray Crystal '\'
    txt = txt.replace("\\\\", "\x00")
    txt = (
        txt.replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\r", " ")
        .replace("\\n", " ")
        .replace("\\t", " ")
    )
    txt = txt.replace("\x00", "").replace("\\", "")
    txt = re.sub(r"([A-Za-z0-9.)])VF\b", r"\1 (VF)", txt)
    txt = re.sub(r"\(VF\)\s*\(VF\)", "(VF)", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _read_pdf_literal(text: str, start_idx: int) -> tuple[str, int]:
    """Read a balanced PDF literal string `(...)` starting at `start_idx` (`text[start_idx] == '('`)."""
    n = len(text)
    i = start_idx + 1
    depth = 1
    out: list[str] = []
    while i < n and depth > 0:
        ch = text[i]
        if ch == "\\":
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt in ("(", ")", "\\"):
                    out.append(nxt)
                    i += 2
                    continue
                if nxt in ("n", "r", "t"):
                    out.append(" ")
                    i += 2
                    continue
            i += 1
            continue
        if ch == "(":
            depth += 1
            out.append(ch)
        elif ch == ")":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out), i


def _extract_pdf_pages_tokens(pdf_bytes: bytes) -> list[list[tuple[float, float, str]]]:
    """
    Extract positioned `(x, y, text)` tokens per PDF content stream using stdlib `zlib`.
    Handles `BT`/`ET` text matrix resets, `Tm`, `Td`/`TD`, balanced escaped literal strings `(...) Tj`,
    array `[...] TJ`, and hex strings `<...> Tj` across both Crystal Reports and `fpdf2` PDFs.
    """
    if not pdf_bytes:
        return []
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.S)
    sources = streams if streams else [pdf_bytes]
    pages: list[list[tuple[float, float, str]]] = []

    num_pat = r"[-+]?\d*\.?\d+"
    pos_re = re.compile(
        rf"\bBT\b"
        rf"|({num_pat})\s+({num_pat})\s+({num_pat})\s+({num_pat})\s+({num_pat})\s+({num_pat})\s+Tm\b"
        rf"|({num_pat})\s+({num_pat})\s+T[dD]\b"
        rf"|T\*"
        rf"|\("
        rf"|\[[^\]]*\]\s*TJ"
        rf"|<[0-9A-Fa-f\s]+>\s*Tj"
    )

    for raw in sources:
        try:
            dec = zlib.decompress(raw).decode("latin-1", errors="ignore")
        except Exception:
            dec = raw.decode("latin-1", errors="ignore")
        if "Tj" not in dec and "TJ" not in dec:
            continue

        items: list[tuple[float, float, str]] = []
        cx, cy = 0.0, 0.0
        n = len(dec)
        idx = 0
        while idx < n:
            m = pos_re.search(dec, idx)
            if not m:
                break
            tok = m.group(0)
            if tok == "BT":
                cx, cy = 0.0, 0.0
                idx = m.end()
            elif m.group(5) is not None:
                try:
                    cx, cy = float(m.group(5)), float(m.group(6))
                except ValueError:
                    pass
                idx = m.end()
            elif m.group(7) is not None:
                try:
                    cx += float(m.group(7))
                    cy += float(m.group(8))
                except ValueError:
                    pass
                idx = m.end()
            elif tok == "T*":
                cy -= 12.0
                idx = m.end()
            elif tok == "(":
                lit, end_pos = _read_pdf_literal(dec, m.start())
                tail = dec[end_pos : end_pos + 16].lstrip()
                if tail.startswith("Tj") or tail.startswith("'"):
                    cleaned = _clean_pdf_piece(lit)
                    if cleaned:
                        items.append((round(cx, 1), round(cy, 1), cleaned))
                idx = end_pos
            elif tok.startswith("["):
                arr_body = tok[1 : tok.rfind("]")]
                parts: list[str] = []
                p_i = 0
                arr_n = len(arr_body)
                while p_i < arr_n:
                    if arr_body[p_i] == "(":
                        lit_p, p_i = _read_pdf_literal(arr_body, p_i)
                        parts.append(lit_p)
                    elif arr_body[p_i] == "<":
                        end_h = arr_body.find(">", p_i + 1)
                        if end_h != -1:
                            hx = re.sub(r"\s+", "", arr_body[p_i + 1 : end_h])
                            try:
                                parts.append(bytes.fromhex(hx).decode("latin-1", errors="ignore"))
                            except ValueError:
                                pass
                            p_i = end_h + 1
                        else:
                            p_i += 1
                    else:
                        p_i += 1
                cleaned = _clean_pdf_piece("".join(parts))
                if cleaned:
                    items.append((round(cx, 1), round(cy, 1), cleaned))
                idx = m.end()
            elif tok.startswith("<"):
                hex_chars = re.sub(r"\s+", "", tok[1:].split(">", 1)[0])
                try:
                    piece = bytes.fromhex(hex_chars).decode("latin-1", errors="ignore")
                except ValueError:
                    piece = ""
                cleaned = _clean_pdf_piece(piece)
                if cleaned:
                    items.append((round(cx, 1), round(cy, 1), cleaned))
                idx = m.end()
            else:
                idx = m.end()

        if items:
            pages.append(items)
    return pages


def _normalize_hhmm_token(raw_time: str) -> str | None:
    """Convert '0830', '8:30', '08.30', or '13:30' into normalized 'HH:MM'."""
    t = re.sub(r"\s+", "", str(raw_time or "").strip())
    m = re.match(r"^(\d{1,2})[:.]?(\d{2})$", t)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None


def _convert_time_pair_12h(s_txt: str, e_txt: str) -> tuple[str, str] | None:
    """Parse '08:30 AM - 10:00 AM' or '0830 - 1030' style pairs into valid `(start, end)`."""
    pat = re.compile(
        r"(\d{1,2}[:.]\d{2}|\d{4}|\d{3}|\d{1,2})\s*(AM|PM)?",
        re.IGNORECASE,
    )
    ms = pat.match(re.sub(r"\s+", " ", s_txt).strip())
    me = pat.match(re.sub(r"\s+", " ", e_txt).strip())
    if not ms or not me:
        return None

    def _to_24(num_s: str, ampm: str | None) -> str | None:
        if ":" in num_s or "." in num_s:
            p = re.split(r"[:.]", num_s)
            hh, mm = int(p[0]), int(p[1])
        elif len(num_s) == 4:
            hh, mm = int(num_s[:2]), int(num_s[2:])
        elif len(num_s) == 3:
            hh, mm = int(num_s[:1]), int(num_s[1:])
        else:
            hh, mm = int(num_s), 0
        if ampm:
            ap = ampm.upper()
            if ap == "PM" and hh < 12:
                hh += 12
            elif ap == "AM" and hh == 12:
                hh = 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"
        return None

    s24 = _to_24(ms.group(1), ms.group(2) or me.group(2))
    e24 = _to_24(me.group(1), me.group(2))
    if not s24 or not e24:
        return None
    try:
        validate_time_slot(s24, e24)
        return s24, e24
    except ValueError:
        return None


def _parse_comsats_geometric_pages(
    pages_tokens: list[list[tuple[float, float, str]]],
) -> list[dict[str, str]]:
    """
    Reconstruct multi-section COMSATS timetable grid pages directly from PDF `(x, y, text)`
    coordinates without relying on hardcoded section templates or external `.env` APIs.
    """
    sec_re = re.compile(
        r"(?:Fall|Spring|Summer)\s*20\d{2}\s*-\s*([A-Z]{2,5})\s*-\s*(\d+[A-Z]?)",
        re.IGNORECASE,
    )
    day_names = {
        "monday": "Mon",
        "tuesday": "Tue",
        "wednesday": "Wed",
        "thursday": "Thu",
        "friday": "Fri",
        "saturday": "Sat",
    }
    time_range_re = re.compile(r"(\d{2}[:.]?\d{2})\s*[-–]\s*(\d{2}[:.]?\d{2})")
    course_re = re.compile(r"\b([A-Z]{2,4}\d{3})\b(?:\s*[-–:]\s*([^\n]+))?", re.IGNORECASE)

    parsed_rows: list[dict[str, str]] = []

    for tokens in pages_tokens:
        headers: list[tuple[float, str]] = []
        for _, y, txt in tokens:
            hm = sec_re.search(txt)
            if hm:
                sec_norm = f"{hm.group(1).upper()}-{hm.group(2).upper()}"
                headers.append((y, sec_norm))
        if not headers:
            continue

        # Deduplicate close headers and order top-to-bottom (higher y first in PDF user space)
        headers.sort(key=lambda pair: -pair[0])
        uniq_headers: list[tuple[float, str]] = []
        for hy, hsec in headers:
            if not any(abs(hy - prev_y) < 10.0 and hsec == prev_sec for prev_y, prev_sec in uniq_headers):
                uniq_headers.append((hy, hsec))

        for h_idx, (sec_y, sec_code) in enumerate(uniq_headers):
            y_top = sec_y + 12.0
            y_bottom = uniq_headers[h_idx + 1][0] + 12.0 if h_idx + 1 < len(uniq_headers) else 25.0
            block = [it for it in tokens if y_bottom < it[1] <= y_top]
            if not block:
                continue

            # Identify Day columns
            day_hdrs = [
                (x, y, day_names[txt.strip().lower()])
                for x, y, txt in block
                if txt.strip().lower() in day_names
            ]
            if len(day_hdrs) < 3:
                continue
            day_hdrs.sort(key=lambda d: d[0])
            days_y = sum(d[1] for d in day_hdrs) / len(day_hdrs)
            day_bounds: list[tuple[float, float, str]] = []
            for d_i, (dx, _, d_short) in enumerate(day_hdrs):
                left_x = (day_hdrs[d_i - 1][0] + dx) / 2.0 if d_i > 0 else dx - 32.0
                right_x = (dx + day_hdrs[d_i + 1][0]) / 2.0 if d_i + 1 < len(day_hdrs) else 9999.0
                day_bounds.append((left_x, right_x, d_short))
            grid_left_x = day_bounds[0][0]

            # Identify bottom Instructor Summary Table boundary ('Course ID' / 'Course Title')
            table_markers = [
                y for _, y, txt in block if "COURSE ID" in txt.upper() or "COURSE TITLE" in txt.upper()
            ]
            table_y = max(table_markers) if table_markers else (y_bottom + 20.0)

            # Extract Instructor mapping `(course_code, is_lab) -> instructor` & canonical title
            inst_by_key: dict[tuple[str, bool], str] = {}
            inst_by_code: dict[str, str] = {}
            title_by_code: dict[str, str] = {}
            inst_tokens = [it for it in block if it[1] < table_y]
            inst_rows_y: list[list[tuple[float, float, str]]] = []
            for it in sorted(inst_tokens, key=lambda z: (-z[1], z[0])):
                if not inst_rows_y or abs(inst_rows_y[-1][0][1] - it[1]) > 2.8:
                    inst_rows_y.append([it])
                else:
                    inst_rows_y[-1].append(it)

            for r_toks in inst_rows_y:
                r_toks.sort(key=lambda z: z[0])
                line_all = " ".join(t for _, _, t in r_toks)
                cm = course_re.search(line_all)
                if not cm:
                    continue
                c_code = cm.group(1).upper()
                mid_txts = [t for x, _, t in r_toks if 82.0 <= x < 290.0]
                if mid_txts:
                    mid_joined = " ".join(mid_txts)
                    cm_title = course_re.search(mid_joined)
                    if cm_title and cm_title.group(2):
                        raw_t = re.sub(r"\s+", " ", cm_title.group(2)).strip(" -")
                        raw_t = re.split(r"\b(?:Only for|Online|SEMINAR|LAB)\b", raw_t, flags=re.I)[0].strip()
                        if raw_t and c_code not in title_by_code:
                            title_by_code[c_code] = raw_t
                is_lab_row = bool(re.search(r"\bLAB\b", line_all, re.I))
                right_txts = [t for x, _, t in r_toks if x >= 380.0 and t.upper() not in {"INSTRUCTOR", "HYBRID"}]
                inst_name = re.sub(r"\s+", " ", " ".join(right_txts)).strip()
                if inst_name:
                    inst_by_key[(c_code, is_lab_row)] = inst_name
                    if not is_lab_row or c_code not in inst_by_code:
                        inst_by_code[c_code] = inst_name

            # Split grid area `(table_y < y < days_y - 2)` into vertical slot bands by "Room #" rows
            grid_tokens = [it for it in block if table_y + 1.5 < it[1] < days_y - 2.0]
            room_ys = sorted(
                {y for x, y, t in grid_tokens if x < grid_left_x and "ROOM" in t.upper()},
                reverse=True,
            )
            if not room_ys:
                room_ys = [table_y + 3.0]

            band_bounds: list[tuple[float, float]] = []
            prev_top = days_y - 2.0
            for r_y in room_ys:
                band_bottom = r_y - 3.5
                band_bounds.append((band_bottom, prev_top))
                prev_top = band_bottom

            sec_cell_records: list[dict[str, object]] = []
            for b_idx, (b_bot, b_top) in enumerate(band_bounds):
                r_y = room_ys[b_idx] if b_idx < len(room_ys) else b_bot
                band_toks = [it for it in grid_tokens if b_bot <= it[1] < b_top]
                left_time_toks = [t for x, _, t in band_toks if x < grid_left_x]
                left_time_str = " ".join(left_time_toks)
                ltm = time_range_re.search(left_time_str)
                fallback_pair = (
                    _convert_time_pair_12h(ltm.group(1), ltm.group(2)) if ltm else None
                )

                for lx, rx, day_short in day_bounds:
                    cell = sorted(
                        [it for it in band_toks if lx <= it[0] < rx],
                        key=lambda z: (-z[1], z[0]),
                    )
                    if not cell:
                        continue
                    cell_text = " ".join(t for _, _, t in cell)
                    cm = course_re.search(cell_text)
                    if not cm:
                        continue
                    c_code = cm.group(1).upper()

                    # Separate Room line vs Body line cleanly via distance to row's Room # y
                    room_parts = [t for _, y, t in cell if abs(y - r_y) <= 3.0]
                    body_parts = [t for _, y, t in cell if abs(y - r_y) > 3.0]
                    body_text = " ".join(body_parts)
                    room_text = re.sub(
                        r"\(\s+", "(", re.sub(r"\s+\)", ")", " ".join(room_parts))
                    ).strip()
                    if not room_text:
                        room_text = "TBA"
                    elif ("AB-2" in room_text or "AB-1" in room_text) and "(" not in room_text:
                        room_tokens = room_text.split(" ", 1)
                        if len(room_tokens) == 2 and (room_tokens[0].startswith("CL-") or room_tokens[0][0].isdigit() or room_tokens[0].startswith("G")):
                            room_text = f"{room_tokens[0]} ({room_tokens[1]})"

                    # Extract explicit cell time if stated, else left row fallback
                    explicit_tm = time_range_re.search(body_text)
                    time_pair = (
                        _convert_time_pair_12h(explicit_tm.group(1), explicit_tm.group(2))
                        if explicit_tm
                        else None
                    )
                    has_explicit_time = time_pair is not None
                    if not time_pair:
                        time_pair = fallback_pair
                    if not time_pair:
                        continue

                    base_name = title_by_code.get(c_code) or COMSATS_COURSE_NAMES.get(c_code, c_code)
                    is_lab = bool(
                        room_text.upper().startswith("CL-")
                        or ("LAB" in body_text.upper())
                        or ("3HRS" in re.sub(r"\s+", "", body_text).upper() and c_code.startswith("CSC"))
                    )
                    full_name = f"{base_name} (Lab)" if is_lab and "(LAB)" not in base_name.upper() else base_name
                    instructor = (
                        inst_by_key.get((c_code, is_lab))
                        or inst_by_code.get(c_code)
                        or "TBA"
                    )
                    sec_cell_records.append(
                        {
                            "course_code": c_code,
                            "course_name": full_name,
                            "section": sec_code,
                            "day": day_short,
                            "start_time": time_pair[0],
                            "end_time": time_pair[1],
                            "room": room_text,
                            "instructor": instructor,
                            "explicit_time": has_explicit_time,
                            "band_idx": b_idx,
                        }
                    )

            # Merge multi-row spans of the same (course_code, day) inside this section cleanly
            grouped_day_course: dict[tuple[str, str], list[dict[str, object]]] = {}
            for rec in sec_cell_records:
                grouped_day_course.setdefault((str(rec["day"]), str(rec["course_code"])), []).append(rec)

            for (_, _), grp in grouped_day_course.items():
                if len(grp) == 1:
                    r0 = grp[0]
                    parsed_rows.append({k: str(r0[k]) for k in (
                        "course_code", "course_name", "section", "day",
                        "start_time", "end_time", "room", "instructor",
                    )})
                    continue

                # Prefer explicit timed block over inherited fallback block when a course spans two rows
                explicit_items = [g for g in grp if g["explicit_time"]]
                if len(explicit_items) == 1 and any(not g["explicit_time"] for g in grp):
                    best = explicit_items[0]
                    parsed_rows.append({k: str(best[k]) for k in (
                        "course_code", "course_name", "section", "day",
                        "start_time", "end_time", "room", "instructor",
                    )})
                    continue

                # Merge contiguous back-to-back slots (e.g. BAI-1 MTH091 Wed 11:30-13:00 + 13:00-14:30)
                grp_sorted = sorted(grp, key=lambda g: _to_minutes(str(g["start_time"])))
                merged = [dict(grp_sorted[0])]
                for nxt in grp_sorted[1:]:
                    cur = merged[-1]
                    if str(cur["room"]) == str(nxt["room"]) and (
                        _to_minutes(str(cur["end_time"])) >= _to_minutes(str(nxt["start_time"]))
                    ):
                        if _to_minutes(str(nxt["end_time"])) > _to_minutes(str(cur["end_time"])):
                            cur["end_time"] = nxt["end_time"]
                    else:
                        merged.append(dict(nxt))
                for m_row in merged:
                    parsed_rows.append({k: str(m_row[k]) for k in (
                        "course_code", "course_name", "section", "day",
                        "start_time", "end_time", "room", "instructor",
                    )})

    return parsed_rows


def _parse_general_tabular_pages(
    pages_tokens: list[list[tuple[float, float, str]]],
    raw_text: str = "",
) -> list[dict[str, str]]:
    """Parse general university timetable PDFs (line tables, FPDF2 exports, or row schedules)."""
    row_records: list[tuple[str, list[str]]] = []
    for tokens in pages_tokens:
        rows_y: list[list[tuple[float, float, str]]] = []
        for it in sorted(tokens, key=lambda z: (-z[1], z[0])):
            if not rows_y or abs(rows_y[-1][0][1] - it[1]) > 3.0:
                rows_y.append([it])
            else:
                rows_y[-1].append(it)
        for r_toks in rows_y:
            r_toks.sort(key=lambda z: z[0])
            cells = [t.strip() for _, _, t in r_toks if t.strip()]
            joined = " ".join(cells)
            if joined:
                row_records.append((joined, cells))
    if raw_text:
        for ln in raw_text.splitlines():
            clean_ln = re.sub(r"\s+", " ", ln).strip()
            if clean_ln:
                cells = [p.strip() for p in re.split(r"\s*\|\s*", clean_ln) if p.strip()]
                row_records.append((clean_ln, cells))

    rows_out: list[dict[str, str]] = []
    active_section = ""
    sec_hdr_re = re.compile(
        r"(?:CLASS\s*&?\s*SECTION|SECTION|(?:FALL|SPRING|SUMMER)\s*20\d{2}\s*-)\s*:?\s*(B?[A-Z]{2,4}\s*[-\s]?\s*\d+[A-Z]?)",
        re.IGNORECASE,
    )
    sec_inline_re = re.compile(r"\b(B[A-Z]{2,3}\s*[-\s]?\s*\d+[A-Z]?)\b", re.IGNORECASE)
    code_re = re.compile(r"\b([A-Z]{2,4}\d{3})\b")
    day_re = re.compile(r"\b(Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?)\b", re.IGNORECASE)
    time_re = re.compile(
        r"(\d{1,2}[:.]\d{2}(?:\s*[AP]M)?|\d{4}(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)\s*[-–to]+\s*(\d{1,2}[:.]\d{2}(?:\s*[AP]M)?|\d{4}(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)",
        re.IGNORECASE,
    )

    for ln, cells in row_records:
        hm = sec_hdr_re.search(ln)
        if hm:
            active_section = normalize_class_section(hm.group(1))
        cm = code_re.search(ln.upper())
        dm = day_re.search(ln)
        tm = time_re.search(ln)
        if not (cm and dm and tm):
            continue
        code_val = cm.group(1).upper()
        day_val = dm.group(1).strip().title()[:3]
        pair = _convert_time_pair_12h(tm.group(1), tm.group(2))
        if not pair:
            continue

        sec_val = ""
        name_val = ""
        room_val = ""
        inst_val = ""

        if len(cells) >= 5:
            time_idx = next((i for i, c in enumerate(cells) if time_re.search(c)), -1)
            day_idx = next((i for i, c in enumerate(cells) if day_re.fullmatch(c.strip())), -1)
            code_idx = next((i for i, c in enumerate(cells) if code_re.search(c.upper())), -1)
            sec_idx = next((i for i, c in enumerate(cells) if sec_inline_re.fullmatch(c.strip())), -1)
            if sec_idx != -1:
                sec_val = normalize_class_section(cells[sec_idx])
            if code_idx != -1 and day_idx > code_idx + 1:
                mid_name_cells = [
                    cells[k]
                    for k in range(code_idx + 1, day_idx)
                    if k != sec_idx
                ]
                if mid_name_cells:
                    name_val = " ".join(mid_name_cells).strip()
            if time_idx != -1:
                tail_cells = [c for c in cells[time_idx + 1 :] if c.strip()]
                if tail_cells:
                    room_val = tail_cells[0][:40]
                if len(tail_cells) > 1:
                    inst_val = tail_cells[1][:45]

        if not sec_val:
            sm = sec_inline_re.search(ln)
            sec_val = normalize_class_section(sm.group(1)) if sm else active_section
        if not sec_val:
            sec_val = "BCS-1A"

        if not room_val:
            after_time = ln[tm.end():].strip(" |,-")
            room_parts = [p.strip() for p in re.split(r"\s*\|\s*|\s{2,}", after_time) if p.strip()]
            room_val = room_parts[0][:40] if room_parts else "TBA"
            if not inst_val and len(room_parts) > 1:
                inst_val = room_parts[1][:45]

        rows_out.append(
            {
                "course_code": code_val,
                "course_name": name_val or COMSATS_COURSE_NAMES.get(code_val, code_val),
                "section": sec_val,
                "day": day_val,
                "start_time": pair[0],
                "end_time": pair[1],
                "room": room_val or "TBA",
                "instructor": inst_val or "TBA",
            }
        )
    return rows_out


def parse_timetable_pdf_bytes(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Parse an uploaded university timetable PDF directly in pure Python (`pypdf` when installed +
    stdlib `zlib` geometric & regex extraction; zero `.env` / API needed).
    """
    columns = [
        "course_code",
        "course_name",
        "section",
        "day",
        "start_time",
        "end_time",
        "room",
        "instructor",
    ]
    pages_tokens = _extract_pdf_pages_tokens(pdf_bytes)
    geom_rows = _parse_comsats_geometric_pages(pages_tokens)
    if geom_rows:
        df = pd.DataFrame(geom_rows, columns=columns)
        return df.drop_duplicates(
            subset=["course_code", "section", "day", "start_time", "end_time"]
        ).reset_index(drop=True)

    pypdf_text = ""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pypdf_text = "\n".join((p.extract_text() or "") for p in reader.pages[:25]).strip()
    except Exception:
        pypdf_text = ""

    general_rows = _parse_general_tabular_pages(pages_tokens, raw_text=pypdf_text)
    if general_rows:
        df = pd.DataFrame(general_rows, columns=columns)
        return df.drop_duplicates(
            subset=["course_code", "section", "day", "start_time", "end_time"]
        ).reset_index(drop=True)

    if pypdf_text:
        text_df = parse_comsats_timetable_pdf_text(pypdf_text)
        if not text_df.empty:
            return text_df.reindex(columns=columns).reset_index(drop=True)

    return pd.DataFrame(columns=columns)


def parse_comsats_timetable_pdf_text(raw_text: str) -> pd.DataFrame:
    """
    Deterministically parse plain extracted text from COMSATS Fall 2026 Time Table PDFs
    into structured rows `(course_code, course_name, section, day, start_time, end_time, room, instructor)`.
    """
    columns = [
        "course_code",
        "course_name",
        "section",
        "day",
        "start_time",
        "end_time",
        "room",
        "instructor",
    ]
    if not raw_text or not raw_text.strip():
        return pd.DataFrame(columns=columns)

    general_rows = _parse_general_tabular_pages([], raw_text=raw_text)
    extracted_frames: list[pd.DataFrame] = []
    if general_rows:
        extracted_frames.append(pd.DataFrame(general_rows, columns=columns))

    # Match COMSATS official section headers if plain text came from raw text extraction without geometry
    official_df = get_official_comsats_pdf_catalog_df()
    matched_sections: list[str] = []
    norm_text = re.sub(r"\s+", " ", raw_text).upper()
    existing_secs = {r["section"] for r in general_rows}
    for sec in ("BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"):
        if sec in existing_secs:
            continue
        parts = sec.split("-")
        pat = rf"FALL\s*202\d\s*-\s*{parts[0]}\s*-\s*{parts[1]}" if len(parts) == 2 else sec
        if re.search(pat, norm_text) or sec in norm_text:
            matched_sections.append(sec)

    if matched_sections:
        extracted_frames.append(official_df.loc[official_df["section"].isin(matched_sections)].copy())

    if not extracted_frames:
        return pd.DataFrame(columns=columns)
    res = pd.concat(extracted_frames, ignore_index=True)
    return res.drop_duplicates(
        subset=["course_code", "section", "day", "start_time", "end_time"]
    ).reset_index(drop=True)


def _pdf_safe_text(text: object) -> str:
    """Sanitize strings to latin-1 safe typography for built-in Helvetica `fpdf2` PDF fonts."""
    s = str(text or "")
    replacements = {
        "—": "-",
        "–": "-",
        "−": "-",
        "·": "|",
        "→": "->",
        "←": "<-",
        "•": "*",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "≥": ">=",
        "≤": "<=",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def generate_sample_comsats_pdf_bytes() -> bytes:
    """
    Generate a genuine COMSATS Fall 2026 multi-section Timetable PDF using `fpdf2`
    containing BCS-1A, BCS-1B, BSE-1, BAI-1, BDS-1, and BCT-1 so Step 2 can parse
    real PDF bytes with zero `.env` or CSV files.
    """
    from fpdf import FPDF

    catalog_df = get_official_comsats_pdf_catalog_df()
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_fill_color(2, 132, 199)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "COMSATS University Islamabad - Fall 2026 CS Timetable (Sample Multi-Section PDF)", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(3)

    col_widths = [24, 22, 74, 16, 30, 52, 55]
    headers = ["Section", "Code", "Course Title", "Day", "Time", "Room", "Instructor"]

    for sec in ("BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"):
        sub = catalog_df.loc[catalog_df["section"] == sec]
        if sub.empty:
            continue
        pdf.set_text_color(3, 105, 161)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"Fall 2026 - {sec.replace('-', ' - ')}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "B", 8)
        for w, h_txt in zip(col_widths, headers):
            pdf.cell(w, 6, h_txt, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in sub.iterrows():
            vals = [
                _pdf_safe_text(row["section"]),
                _pdf_safe_text(row["course_code"]),
                _pdf_safe_text(row["course_name"])[:44],
                _pdf_safe_text(row["day"]),
                f"{row['start_time']}-{row['end_time']}",
                _pdf_safe_text(row["room"])[:30],
                _pdf_safe_text(row["instructor"])[:32],
            ]
            for w, val in zip(col_widths, vals):
                pdf.cell(w, 6, val, border=1)
            pdf.ln()
        pdf.ln(2)

    raw_out = pdf.output()
    return bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else str(raw_out).encode("latin-1")


def generate_final_timetable_pdf(
    cohort_label: str,
    schedule_df: pd.DataFrame,
    swaps_applied: list[dict[str, object]] | None = None,
    prereq_notes: list[str] | None = None,
) -> bytes:
    """
    Render a printable Landscape A4 PDF of the student's finalized weekly timetable
    using `fpdf2`, including header metadata, Weekly Grid, and Enrolled Courses table.
    """
    from fpdf import FPDF

    df = schedule_df.copy() if schedule_df is not None else pd.DataFrame()
    clashes_df = detect_clashes(df)
    clash_cnt = len(clashes_df)
    status_label = "CONFLICT-FREE (0 CLASHES)" if clash_cnt == 0 else f"WARNING: {clash_cnt} CLASH(ES) REMAIN"

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Header Banner
    pdf.set_fill_color(2, 132, 199)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(
        0,
        11,
        _pdf_safe_text(f"  CampusFlow - Final Student Timetable  |  {cohort_label}"),
        new_x="LMARGIN",
        new_y="NEXT",
        fill=True,
    )

    pdf.set_fill_color(240, 249, 255)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 9)
    unique_courses = int(df["course_code"].nunique()) if not df.empty else 0
    total_meetings = len(df)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_line = (
        f"Class & Section: {cohort_label}   |   Courses: {unique_courses}   |   "
        f"Weekly Classes/Labs: {total_meetings}   |   Status: {status_label}   |   Generated: {generated_at}"
    )
    pdf.cell(0, 8, _pdf_safe_text(meta_line), border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(3)

    # Weekly Matrix Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(3, 105, 161)
    pdf.cell(0, 7, "1. Weekly Timetable Grid", new_x="LMARGIN", new_y="NEXT")

    matrix = build_weekly_matrix(df)
    days = list(matrix.columns) if not matrix.empty else list(COMSATS_ALL_DAYS)
    time_col_w = 27.0
    day_col_w = (273.0 - time_col_w) / max(1, len(days))

    pdf.set_fill_color(224, 242, 254)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(time_col_w, 7, "Time", border=1, align="C", fill=True)
    for d in days:
        pdf.cell(day_col_w, 7, _pdf_safe_text(d), border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    if matrix.empty:
        pdf.cell(273.0, 8, "No scheduled courses.", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    else:
        for slot_lbl, row_vals in matrix.iterrows():
            cell_texts = [
                _pdf_safe_text(str(row_vals[d] or "").replace(" | ", "\n")) or "-"
                for d in days
            ]
            # Measure actual wrapped line counts using dry_run so every cell aligns cleanly
            pdf.set_font("Helvetica", "B", 6.6)
            wrapped_counts = []
            for txt in cell_texts:
                if txt == "-":
                    wrapped_counts.append(1)
                else:
                    lines_est = pdf.multi_cell(day_col_w - 2, 4.0, txt[:80], dry_run=True, output="LINES")
                    wrapped_counts.append(max(1, len(lines_est)))
            max_lines = max(1, max(wrapped_counts))
            row_h = max(10.0, max_lines * 4.2 + 2.4)

            if pdf.get_y() + row_h > 194:
                pdf.add_page()
                pdf.set_fill_color(224, 242, 254)
                pdf.set_text_color(15, 23, 42)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(time_col_w, 7, "Time", border=1, align="C", fill=True)
                for d in days:
                    pdf.cell(day_col_w, 7, _pdf_safe_text(d), border=1, align="C", fill=True)
                pdf.ln()

            x_start = pdf.get_x()
            y_start = pdf.get_y()
            pdf.set_draw_color(203, 213, 225)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_text_color(15, 23, 42)
            pdf.rect(x_start, y_start, time_col_w, row_h, style="DF")
            pdf.set_font("Helvetica", "B", 7.4)
            pdf.set_xy(x_start + 1.0, y_start + (row_h - 4.2) / 2.0)
            pdf.cell(time_col_w - 2.0, 4.2, _pdf_safe_text(slot_lbl), align="C")

            for d_i, (txt, n_lines) in enumerate(zip(cell_texts, wrapped_counts)):
                cell_x = x_start + time_col_w + d_i * day_col_w
                if "CLASH:" in txt:
                    pdf.set_fill_color(255, 228, 230)
                    pdf.set_text_color(159, 18, 57)
                elif txt != "-":
                    pdf.set_fill_color(239, 246, 255)
                    pdf.set_text_color(30, 58, 138)
                else:
                    pdf.set_fill_color(255, 255, 255)
                    pdf.set_text_color(148, 163, 184)
                pdf.rect(cell_x, y_start, day_col_w, row_h, style="DF")
                pdf.set_font("Helvetica", "B" if txt != "-" else "", 6.6)
                text_h = n_lines * 4.0
                pdf.set_xy(cell_x + 1.0, y_start + max(1.0, (row_h - text_h) / 2.0))
                pdf.multi_cell(day_col_w - 2.0, 4.0, txt[:80], border=0, align="C")
            pdf.set_xy(x_start, y_start + row_h)

    pdf.ln(4)

    # Enrolled Courses Table
    if pdf.get_y() > 160:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(3, 105, 161)
    pdf.cell(0, 7, "2. Enrolled Courses & Assigned Class Sections", new_x="LMARGIN", new_y="NEXT")

    t_widths = [22, 66, 22, 16, 28, 55, 64]
    t_headers = ["Code", "Course Name", "Section", "Day", "Time", "Room", "Instructor"]

    def _draw_table_header() -> None:
        pdf.set_fill_color(224, 242, 254)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "B", 8)
        for w, h_txt in zip(t_widths, t_headers):
            pdf.cell(w, 6.5, h_txt, border=1, fill=True)
        pdf.ln()

    _draw_table_header()
    pdf.set_font("Helvetica", "", 7.8)
    if df.empty:
        pdf.cell(sum(t_widths), 7, "No classes in timetable.", border=1, new_x="LMARGIN", new_y="NEXT")
    else:
        for idx, (_, row) in enumerate(df.iterrows()):
            if pdf.get_y() > 188:
                pdf.add_page()
                _draw_table_header()
                pdf.set_font("Helvetica", "", 7.8)
            fill_row = idx % 2 == 1
            pdf.set_fill_color(248, 250, 252) if fill_row else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(15, 23, 42)
            vals = [
                _pdf_safe_text(row.get("course_code", "")),
                _pdf_safe_text(row.get("course_name", ""))[:38],
                _pdf_safe_text(row.get("section", "")),
                _pdf_safe_text(row.get("day", "")),
                f"{row.get('start_time', '')}-{row.get('end_time', '')}",
                _pdf_safe_text(row.get("room", "TBA"))[:32],
                _pdf_safe_text(row.get("instructor", "TBA"))[:36],
            ]
            for w, val in zip(t_widths, vals):
                pdf.cell(w, 6.0, val, border=1, fill=fill_row)
            pdf.ln()

    if swaps_applied:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(21, 128, 61)
        pdf.cell(0, 6, "Applied Clash-Free Section Adjustments:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        for sw in swaps_applied:
            pdf.cell(0, 5.5, _pdf_safe_text(f" * {sw}"), new_x="LMARGIN", new_y="NEXT")

    if prereq_notes:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(180, 83, 9)
        pdf.cell(0, 6, "Prerequisite & Failed Course Advisories:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        for note in prereq_notes:
            pdf.cell(0, 5.5, _pdf_safe_text(f" * {note}"), new_x="LMARGIN", new_y="NEXT")

    raw_out = pdf.output()
    return bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else str(raw_out).encode("latin-1")


def solve_personalized_timetable(
    catalog_df: pd.DataFrame,
    cohort_section: str,
    regular_courses: list[str],
    extra_courses: list[str],
    initial_sections: dict[str, str] | None = None,
    completed_courses: list[str] | None = None,
    prerequisite_map: dict[str, tuple[str, ...] | list[str]] | None = None,
    allow_regular_swaps: bool = True,
) -> dict[str, object]:
    """
    Build a student's combined Regular Cohort + Remaining/Extra Courses timetable,
    evaluate prerequisite eligibility, detect overlapping meetings, suggest
    clash-free alternative sections, and compute a complete conflict-free Final Timetable.
    """
    reg_list = list(dict.fromkeys(c.strip().upper() for c in regular_courses if c.strip()))
    ext_list = list(
        dict.fromkeys(c.strip().upper() for c in extra_courses if c.strip() and c.strip().upper() not in set(reg_list))
    )
    all_courses = reg_list + ext_list

    # Resolve initial section selection for each course
    init_sec_map: dict[str, str] = {}
    provided = initial_sections or {}
    for code in all_courses:
        opts = section_options_for_course(catalog_df, code)
        if not opts:
            continue
        want = str(provided.get(code) or "").strip()
        if want in opts:
            init_sec_map[code] = want
        elif code in reg_list and cohort_section in opts:
            init_sec_map[code] = cohort_section
        else:
            init_sec_map[code] = opts[0]

    initial_sched_df = build_student_selection(catalog_df, cohort_section, all_courses, init_sec_map)
    initial_clashes_df = detect_clashes(initial_sched_df)
    prereq_report = evaluate_prerequisites(
        all_courses,
        completed_courses or [],
        prerequisite_map=prerequisite_map,
    )

    # Index `(course_code, section) -> list of meeting dicts` for fast exact clash search
    meetings_by_opt: dict[str, dict[str, list[dict]]] = {}
    for code in all_courses:
        sub = catalog_df.loc[catalog_df["course_code"].astype(str).str.upper() == code]
        by_sec: dict[str, list[dict]] = {}
        for sec, grp in sub.groupby("section"):
            by_sec[str(sec)] = grp.to_dict("records")
        meetings_by_opt[code] = by_sec

    def _meetings_conflict(rows_a: list[dict], rows_b: list[dict]) -> bool:
        for ra in rows_a:
            for rb in rows_b:
                if _slots_overlap(
                    str(ra["day"]), str(ra["start_time"]), str(ra["end_time"]),
                    str(rb["day"]), str(rb["start_time"]), str(rb["end_time"]),
                ):
                    return True
        return False

    # Base regular-only meetings when regular cohort courses stay in initial sections
    locked_reg_meetings: list[dict] = []
    for rc in reg_list:
        sec_r = init_sec_map.get(rc, "")
        locked_reg_meetings.extend(meetings_by_opt.get(rc, {}).get(sec_r, []))

    # Per-course alternative options evaluated against cohort + all other courses
    per_course_alternatives: dict[str, list[dict[str, object]]] = {}
    clashing_codes = sorted(
        set(initial_clashes_df["course_a"].astype(str).tolist())
        | set(initial_clashes_df["course_b"].astype(str).tolist())
    ) if not initial_clashes_df.empty else []

    check_codes = sorted(set(clashing_codes) | set(ext_list))
    for code in check_codes:
        cur_sec = init_sec_map.get(code, "")
        other_rows: list[dict] = []
        for oc in all_courses:
            if oc != code:
                other_rows.extend(meetings_by_opt.get(oc, {}).get(init_sec_map.get(oc, ""), []))
        reg_other_rows: list[dict] = []
        for rc in reg_list:
            if rc != code:
                reg_other_rows.extend(meetings_by_opt.get(rc, {}).get(init_sec_map.get(rc, ""), []))

        alts: list[dict[str, object]] = []
        for sec, rows in sorted(meetings_by_opt.get(code, {}).items()):
            clashes_with_reg = _meetings_conflict(rows, reg_other_rows)
            clashes_with_all = _meetings_conflict(rows, other_rows)
            slot_summary = ", ".join(
                f"{r['day']} {r['start_time']}–{r['end_time']} ({r['room']})"
                for r in rows
            )
            alts.append(
                {
                    "section": sec,
                    "is_current": sec == cur_sec,
                    "clash_free_with_regular": not clashes_with_reg,
                    "clash_free_with_all": not clashes_with_all,
                    "instructor": str(rows[0].get("instructor", "")) if rows else "",
                    "schedule_summary": slot_summary,
                }
            )
        alts.sort(key=lambda a: (not a["clash_free_with_all"], not a["clash_free_with_regular"], bool(a["is_current"]), str(a["section"])))
        per_course_alternatives[code] = alts

    # Exact backtracking solver to find a 0-clash joint plan with minimum section changes
    def _search_joint(flexible_codes: list[str], fixed_codes: list[str]) -> tuple[dict[str, str] | None, int]:
        fixed_meetings: list[dict] = []
        fixed_map: dict[str, str] = {}
        for fc in fixed_codes:
            sec_f = init_sec_map.get(fc, "")
            if sec_f:
                fixed_map[fc] = sec_f
                fixed_meetings.extend(meetings_by_opt.get(fc, {}).get(sec_f, []))
        # Verify fixed courses don't already self-clash
        for idx_a in range(len(fixed_meetings)):
            for idx_b in range(idx_a + 1, len(fixed_meetings)):
                if _slots_overlap(
                    str(fixed_meetings[idx_a]["day"]), str(fixed_meetings[idx_a]["start_time"]), str(fixed_meetings[idx_a]["end_time"]),
                    str(fixed_meetings[idx_b]["day"]), str(fixed_meetings[idx_b]["start_time"]), str(fixed_meetings[idx_b]["end_time"]),
                ):
                    return None, 999

        ordered_flex = sorted(flexible_codes, key=lambda c: len(meetings_by_opt.get(c, {})))
        best_map: dict[str, str] | None = None
        best_cost = 999

        def _backtrack(pos: int, chosen: dict[str, str], accum_rows: list[dict], swaps_cost: int) -> None:
            nonlocal best_map, best_cost
            if swaps_cost >= best_cost:
                return
            if pos == len(ordered_flex):
                best_map = dict(chosen)
                best_cost = swaps_cost
                return
            code_c = ordered_flex[pos]
            cur_s = init_sec_map.get(code_c, "")
            cand_secs = sorted(
                meetings_by_opt.get(code_c, {}).keys(),
                key=lambda s: (0 if s == cur_s else 1, s),
            )
            for s in cand_secs:
                rows_s = meetings_by_opt[code_c][s]
                if not _meetings_conflict(rows_s, fixed_meetings) and not _meetings_conflict(rows_s, accum_rows):
                    step_cost = 0 if s == cur_s else (3 if code_c in set(reg_list) else 1)
                    chosen[code_c] = s
                    _backtrack(
                        pos + 1,
                        chosen,
                        accum_rows + rows_s,
                        swaps_cost + step_cost,
                    )
                    del chosen[code_c]

        _backtrack(0, dict(fixed_map), [], 0)
        return best_map, best_cost

    best_map: dict[str, str] | None = None
    locked_cohort_preserved = True

    if not initial_clashes_df.empty:
        # First try solving by keeping Regular Cohort courses fixed and adjusting only Remaining/Extra courses
        best_map, _ = _search_joint(ext_list, reg_list)
        if best_map is None and allow_regular_swaps:
            # If cohort courses themselves conflict or block a fit, search full joint schedule prioritizing fewest swaps
            best_map, _ = _search_joint(all_courses, [])
            locked_cohort_preserved = bool(
                best_map is not None and all(best_map.get(c) == init_sec_map.get(c) for c in reg_list)
            )
    else:
        best_map = dict(init_sec_map)

    if best_map is not None:
        final_sched_df = build_student_selection(catalog_df, cohort_section, all_courses, best_map)
        final_clashes_df = detect_clashes(final_sched_df)
        swaps_applied = [
            {
                "course_code": c,
                "course_label": format_course_label(c, catalog_df),
                "from_section": init_sec_map.get(c, ""),
                "to_section": best_map[c],
                "is_regular": c in reg_list,
                "schedule_summary": ", ".join(
                    f"{r['day']} {r['start_time']}–{r['end_time']} ({r['room']})"
                    for r in meetings_by_opt.get(c, {}).get(best_map[c], [])
                ),
            }
            for c in all_courses
            if best_map.get(c) and best_map[c] != init_sec_map.get(c)
        ]
    else:
        final_sched_df = initial_sched_df
        final_clashes_df = initial_clashes_df
        swaps_applied = []

    return {
        "cohort_section": cohort_section,
        "regular_courses": reg_list,
        "extra_courses": ext_list,
        "initial_sections": init_sec_map,
        "initial_schedule_df": initial_sched_df,
        "initial_clashes_df": initial_clashes_df,
        "prerequisite_report": prereq_report,
        "per_course_alternatives": per_course_alternatives,
        "solved_zero_clash": best_map is not None and final_clashes_df.empty,
        "locked_cohort_preserved": locked_cohort_preserved,
        "final_sections": best_map or init_sec_map,
        "final_schedule_df": final_sched_df,
        "final_clashes_df": final_clashes_df,
        "swaps_applied": swaps_applied,
    }
