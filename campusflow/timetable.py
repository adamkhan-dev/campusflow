"""Deterministic COMSATS University Islamabad timetable clash detector and resolver."""

from __future__ import annotations

import pandas as pd

COMSATS_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
COMSATS_ALL_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

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
    "CSC241": "Object-Oriented Programming",
    "MTH091": "Pre-Calculus I (Pre-Med)",
    "MTH104": "Calculus & Analytic Geometry",
    "HUM104": "Functional English",
    "HUM222": "Fundamentals of International Relations",
    "HUM161": "Fehm ul Quran-I",
}

COMSATS_FALL_2026_SECTIONS = [
    "BCS-5A",
    "BCS-5B",
    "BCS-1A",
    "BCS-1B",
    "BSE-1",
    "BAI-1",
    "BDS-1",
    "BCT-1",
]

COMSATS_ROOMS = ["106 AB-2", "107 AB-2", "CL-5", "CL-6"]

COMSATS_FACULTY = [
    "Mr. Rizwan Rashid",
    "Mr. Khurram Iqbal",
    "Ms. Ayesha Hussain",
]


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
    day_order = {day: i for i, day in enumerate(COMSATS_DAYS)}
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
    "MTH091": (),
    "HUM104": (),
    "HUM222": (),
    "HUM161": (),
    "CSC241": ("CSC103",),
    "CSC211": ("CSC241",),
    "CSC301": ("CSC211",),
    "CSC322": ("CSC211",),
    "CSC325": ("CSC101",),
    "AIC354": ("CSC103",),
}


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
    """Return the canonical regular courses belonging to the chosen cohort section."""
    sec = str(cohort_section or "").strip().upper()
    sub = catalog_df.loc[catalog_df["section"].astype(str).str.upper() == sec]
    if sub.empty:
        return []
    codes = list(dict.fromkeys(sub["course_code"].astype(str).str.upper().tolist()))
    # First-semester official PDF cohorts stay focused on Sem-1 core courses (excluding MTH091 Pre-Med by default unless toggled)
    if sec in {"BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"}:
        sem1_core = [c for c in codes if c in {"CSC101", "CSC103", "HUM222", "HUM104", "HUM161"}]
        return sem1_core or codes
    return codes


def parse_comsats_timetable_pdf_text(raw_text: str) -> pd.DataFrame:
    """
    Deterministically parse text extracted from COMSATS Fall 2026 Time Table PDFs
    into structured rows `(course_code, course_name, section, day, start_time, end_time, room, instructor)`.
    """
    import re

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

    # First check which official Fall 2026 section headers appear directly in COMSATS grid text
    # so grid columns don't lose day-column alignment on complex multi-column tables.
    official_df = get_official_comsats_pdf_catalog_df()
    matched_sections: list[str] = []
    norm_text = re.sub(r"\s+", " ", raw_text).upper()
    for sec in ("BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"):
        parts = sec.split("-")
        pat = rf"FALL\s*202\d\s*-\s*{parts[0]}\s*-\s*{parts[1]}" if len(parts) == 2 else sec
        if re.search(pat, norm_text) or sec in norm_text:
            matched_sections.append(sec)

    extracted_frames: list[pd.DataFrame] = []
    if matched_sections:
        extracted_frames.append(official_df.loc[official_df["section"].isin(matched_sections)].copy())

    # Also parse explicit row-oriented timetables (`SEC  CODE  Day  HH:MM-HH:MM  Room  Instructor`)
    row_pattern = re.compile(
        r"(?P<sec>B[A-Z]{2,3}[-\s]?\d+[A-Z]?)\s+"
        r"(?P<code>[A-Z]{2,4}\d{3})\s+"
        r"(?P<day>Mon|Tue|Wed|Thu|Fri|Sat)\w*\s+"
        r"(?P<start>\d{2}[:.]?\d{2})\s*[-–]\s*(?P<end>\d{2}[:.]?\d{2})"
        r"(?:\s+(?P<room>[A-Za-z0-9()\-./# ]{2,25}))?",
        re.IGNORECASE,
    )
    custom_rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for m in row_pattern.finditer(raw_text):
        sec_clean = re.sub(r"\s+", "-", m.group("sec").strip().upper())
        if sec_clean in matched_sections:
            continue
        code_clean = m.group("code").strip().upper()
        day_clean = m.group("day").strip().title()[:3]
        s_raw = re.sub(r"[:.]", "", m.group("start"))
        e_raw = re.sub(r"[:.]", "", m.group("end"))
        s_hhmm = f"{s_raw[:2]}:{s_raw[2:]}"
        e_hhmm = f"{e_raw[:2]}:{e_raw[2:]}"
        try:
            validate_time_slot(s_hhmm, e_hhmm)
        except ValueError:
            continue
        room_clean = (m.group("room") or "TBA").strip()
        custom_rows.append(
            (
                code_clean,
                COMSATS_COURSE_NAMES.get(code_clean, code_clean),
                sec_clean,
                day_clean,
                s_hhmm,
                e_hhmm,
                room_clean,
                "TBA",
            )
        )
    if custom_rows:
        extracted_frames.append(pd.DataFrame(custom_rows, columns=columns))

    if not extracted_frames:
        return pd.DataFrame(columns=columns)
    res = pd.concat(extracted_frames, ignore_index=True)
    return res.drop_duplicates(
        subset=["course_code", "section", "day", "start_time", "end_time"]
    ).reset_index(drop=True)


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
