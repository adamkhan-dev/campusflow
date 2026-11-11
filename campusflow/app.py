"""CampusFlow — Dynamic 4-Subject Full-Width Academic Workspace."""

from __future__ import annotations

import html
import sys
import textwrap
from pathlib import Path

import pandas as pd

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR.parent / "demo_files"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ai_engine  # noqa: E402
import database  # noqa: E402
import timetable  # noqa: E402

st.set_page_config(
    page_title="CampusFlow",
    layout="wide",
    initial_sidebar_state="collapsed",
)

database.seed_initial_data()


# ---------------------------------------------------------------------------
# Cached Data Loaders — partitioned by DB path and selectively invalidated
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_list_lectures_for(db_path_str: str) -> list[dict]:
    """Return all saved lectures for the resolved DB path."""
    return database.list_lectures(db_path=db_path_str)


def _cached_list_lectures() -> list[dict]:
    """Return cached saved lectures for the active DB path."""
    return _cached_list_lectures_for(str(database.resolve_db_path()))


@st.cache_data(show_spinner=False)
def _cached_list_timetable_slots_for(db_path_str: str) -> list[dict]:
    """Return saved timetable slots for the resolved DB path."""
    return database.list_timetable_slots(db_path=db_path_str)


def _cached_list_timetable_slots() -> list[dict]:
    """Return cached timetable slots for the active DB path."""
    return _cached_list_timetable_slots_for(str(database.resolve_db_path()))


@st.cache_data(show_spinner=False)
def _cached_semester5_catalog():
    """Return the static COMSATS Fall 2026 section catalog DataFrame."""
    return timetable.get_semester_5_catalog_df()


@st.cache_data(show_spinner=False)
def _cached_combined_comsats_catalog():
    """Return the unified COMSATS Fall 2026 PDF + Sem-5 section catalog DataFrame."""
    return timetable.get_combined_comsats_catalog_df()


@st.cache_data(show_spinner=False)
def _cached_course_label_options() -> list[str]:
    """Return ordered list of course label strings for dropdowns."""
    return timetable.get_course_label_options()


@st.cache_data(show_spinner=False)
def _cached_extract_timetable_pdf(pdf_bytes: bytes, file_name: str) -> tuple[list[dict], bool]:
    """Extract all classes and slots from an uploaded university timetable PDF (cached cleanly)."""
    _ = file_name
    return ai_engine.parse_timetable_pdf_with_fallback(pdf_bytes)


MATTE_CSS = """
<style>
:root { color-scheme: light; --cf-bg:#f8fafc; --cf-card:#ffffff; --cf-border:#e2e8f0; --cf-text:#0f172a; --cf-muted:#64748b; --cf-blue:#0284c7; --cf-blue-soft:#e0f2fe; --cf-green:#16a34a; --cf-amber:#d97706; --primary-color:#0284c7; --background-color:#f8fafc; --secondary-background-color:#ffffff; --text-color:#0f172a; }
header, header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none!important; height:0!important; min-height:0!important;}
html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"] {background:var(--cf-bg)!important; color:var(--cf-text)!important;}
.block-container {max-width:1440px!important; padding:1.25rem 2rem 3rem!important; background:var(--cf-bg)!important;}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {color:var(--cf-text)!important;}
.stMarkdown p, .stMarkdown li {line-height:1.68;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stCaption {color:var(--cf-muted)!important;}
h1,h2,h3,h4 {letter-spacing:-.025em!important; color:#0f172a!important;}
[data-testid="stTabs"] [role="tablist"] {gap:1.35rem; border-bottom:1px solid var(--cf-border);}
button[data-baseweb="tab"] {color:#64748b!important; font-weight:600!important; padding:.7rem .15rem!important; background:transparent!important;}
button[data-baseweb="tab"]:hover, button[data-baseweb="tab"][aria-selected="true"] {color:var(--cf-blue)!important;}
button[data-baseweb="tab"] p {color:inherit!important;}
[data-baseweb="tab-highlight"] {background:var(--cf-blue)!important; height:2px!important;}
[data-baseweb="tab-border"] {background:var(--cf-border)!important;}
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {background:var(--cf-blue)!important; border:1px solid var(--cf-blue)!important; color:#ffffff!important; border-radius:9px!important; font-weight:650!important; box-shadow:0 1px 2px rgba(15,23,42,.08)!important; transition:background .15s ease,transform .15s ease!important;}
button[kind="primary"] *, [data-testid="stBaseButton-primary"] *, [data-testid="stFormSubmitButton"] button[kind="primary"] * {color:#ffffff!important;}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {background:#0369a1!important; border-color:#0369a1!important; transform:translateY(-1px);}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {background:#ffffff!important; border:1px solid var(--cf-border)!important; color:#334155!important; border-radius:9px!important;}
button[kind="secondary"] *, [data-testid="stBaseButton-secondary"] * {color:#334155!important;}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {border-color:#7dd3fc!important; color:#0369a1!important; background:#f0f9ff!important;}
button[kind="secondary"]:hover * {color:#0369a1!important;}
/* Explicit Light-Theme Interactive Widget Overrides */
[data-testid="stRadio"] label, [data-testid="stRadio"] label *, [data-testid="stRadio"] p, [data-testid="stRadio"] span, [data-testid="stRadio"] [data-testid="stMarkdownContainer"] *, [data-baseweb="radio"] div, [data-baseweb="radio"] p, [data-baseweb="radio"] span {color:#0f172a!important;}
[data-testid="stRadio"] [role="radiogroup"] {gap:0.45rem;}
[data-testid="stRadio"] label[data-baseweb="radio"] {background:#f8fafc!important; border:1px solid #e2e8f0!important; border-radius:9px!important; padding:0.45rem 0.75rem!important; transition:border-color .15s ease, background .15s ease!important;}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover {border-color:#38bdf8!important; background:#f0f9ff!important;}
[data-testid="stSegmentedControl"], [data-baseweb="button-group"] {background:#f1f5f9!important; border:1px solid #cbd5e1!important; border-radius:10px!important; padding:2px!important;}
[data-testid="stSegmentedControl"] button, [data-baseweb="button-group"] button, button[data-testid="stBaseButton-segmented_control"] {background:#ffffff!important; color:#334155!important; border:1px solid #e2e8f0!important; font-weight:600!important;}
[data-testid="stSegmentedControl"] button *, [data-baseweb="button-group"] button *, button[data-testid="stBaseButton-segmented_control"] * {color:#334155!important;}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"], [data-baseweb="button-group"] button[aria-checked="true"], [data-baseweb="button-group"] button[aria-pressed="true"], button[data-testid="stBaseButton-segmented_controlActive"] {background:#e0f2fe!important; border-color:#0284c7!important; color:#0369a1!important; font-weight:750!important;}
[data-testid="stSegmentedControl"] button[aria-checked="true"] *, [data-testid="stSegmentedControl"] button[aria-pressed="true"] *, [data-baseweb="button-group"] button[aria-checked="true"] *, [data-baseweb="button-group"] button[aria-pressed="true"] *, button[data-testid="stBaseButton-segmented_controlActive"] * {color:#0369a1!important;}
[data-testid="stProgress"] > div > div > div {background-color:#e2e8f0!important;}
[data-testid="stProgress"] > div > div > div > div {background-color:#0284c7!important;}
input, textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {background:#ffffff!important; color:#0f172a!important; border-color:#cbd5e1!important; border-radius:8px!important;}
[data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"], [role="option"] * {background-color:#ffffff!important; color:#0f172a!important;}
[role="option"]:hover, [aria-selected="true"][role="option"], [aria-selected="true"][role="option"] * {background-color:#f0f9ff!important; color:#0369a1!important;}
[data-baseweb="tag"] {background-color:#e0f2fe!important; border:1px solid #bae6fd!important; color:#0369a1!important;}
[data-baseweb="tag"] span {color:#0369a1!important;}
[data-testid="stFileUploader"] section {background:#f8fbff!important; border:1px dashed #93c5fd!important; border-radius:12px!important;}
[data-testid="stFileUploader"] section:hover {border-color:var(--cf-blue)!important; background:#f0f9ff!important;}
[data-testid="stFileUploader"] section *, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploaderFileName"] {color:#334155!important;}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {background:#ffffff!important; color:#0f172a!important; border-color:#cbd5e1!important;}
[data-testid="stChatMessage"] {background:#f8fafc!important; border:1px solid #e2e8f0!important; border-radius:10px!important;}
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {background:#f1f5f9!important; color:#0f172a!important; border:1px solid #e2e8f0!important; border-radius:10px!important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color:#0f172a!important;}
[data-testid="stAlert"] {border-radius:10px!important;}
[data-testid="stDataFrame"], [data-testid="stTable"] {border:1px solid var(--cf-border)!important; border-radius:10px!important; overflow:hidden!important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff!important; border:1px solid var(--cf-border)!important; border-radius:14px!important; box-shadow:0 4px 6px -1px rgba(0,0,0,.035)!important;}
[data-testid="stExpander"] {border:1px solid var(--cf-border)!important; border-radius:10px!important; background:#fff!important;}
[data-testid="stHtml"]:has([data-cf-js-hook="true"]) {display:none!important; margin:0!important; padding:0!important; height:0!important;}
/* CampusFlow component system */
.cf-card,.cf-study-card,.cf-eval-card,.cf-stat-card,.cf-flashcard,.cf-definition-box,.cf-formula-box,.cf-worked-box,.cf-exam-takeaway-box,.cf-takeaway-box,.cf-study-explanation {background:#fff!important; color:#334155!important; border:1px solid #e2e8f0!important; border-radius:12px!important; box-shadow:0 4px 6px -1px rgba(15,23,42,.035)!important;}
.cf-card,.cf-study-card,.cf-eval-card {padding:1rem 1.1rem!important; margin-bottom:.8rem!important;}
.cf-takeaway-box {background:#eff6ff!important;border-left:3px solid #3b82f6!important;padding:.85rem 1rem!important;margin:.65rem 0!important;line-height:1.7;}
.cf-takeaway-label {display:block;color:#0369a1!important;font-size:.72rem;font-weight:750;letter-spacing:.055em;text-transform:uppercase;margin-bottom:.35rem;}
.cf-study-list {margin:.3rem 0 0 1.1rem;padding:0;color:#334155!important;}
.cf-study-list li {margin:.25rem 0;color:#334155!important;}
.cf-stat-card {padding:1rem 1.1rem!important;min-height:78px;}
.cf-stat-label,.cf-muted {color:#64748b!important;}
.cf-stat-label {font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;}
.cf-stat-value {font-size:1.08rem;font-weight:700;color:#0f172a!important;margin-top:.25rem;}
.cf-src-badge,.cf-source-pill,.cf-pill-active,.cf-badge-ok,.cf-badge-bad {display:inline-flex;align-items:center;gap:.25rem;padding:.22rem .55rem;border-radius:999px;font-size:.7rem;font-weight:650;border:1px solid #e2e8f0;background:#f8fafc;color:#475569!important;}
.cf-badge-ok {background:#f0fdf4!important;border-color:#bbf7d0!important;color:#15803d!important;}
.cf-badge-bad {background:#fff7ed!important;border-color:#fed7aa!important;color:#c2410c!important;}
.cf-pill-active {background:#eff6ff!important;border-color:#bfdbfe!important;color:#1d4ed8!important;}
.cf-flashcard {padding:2rem!important;min-height:220px;display:flex;flex-direction:column;justify-content:center;}
.cf-flashcard-side {color:#0284c7!important;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem;}
.cf-flashcard-body {color:#0f172a!important;font-size:1.25rem;line-height:1.6;font-weight:600;}
.cf-quiz-step {color:#0284c7!important;font-size:.75rem;font-weight:750;text-transform:uppercase;}
.cf-quiz-header {display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;}
.cf-quiz-question {font-size:1.12rem;font-weight:700;color:#0f172a!important;margin:.65rem 0 .8rem 0;line-height:1.5;}
.cf-source-pill {margin-top:.3rem;}
.cf-topbar {display:flex;align-items:baseline;gap:.65rem;margin-bottom:.75rem;padding-bottom:.45rem;border-bottom:1px solid var(--cf-border);}
.cf-brand {font-size:1.55rem;font-weight:800;letter-spacing:-.03em;color:#0284c7!important;}
.cf-version {font-size:.74rem;font-weight:650;color:#64748b!important;background:#e2e8f0;padding:.15rem .5rem;border-radius:999px;}
.cf-uploader-head {display:flex;justify-content:space-between;align-items:center;gap:.4rem;margin:.35rem 0 .25rem 0;}
.cf-uploader-title {font-size:.82rem;font-weight:700;color:#0f172a!important;}
.cf-ext-badge {font-size:.66rem;font-weight:700;color:#0369a1!important;background:#e0f2fe;border:1px solid #bae6fd;padding:.1rem .45rem;border-radius:999px;}
.cf-demo-row {display:flex;justify-content:space-between;align-items:center;gap:.6rem;padding:.45rem 0;border-bottom:1px solid #f1f5f9;font-size:.83rem;color:#334155!important;}
.cf-demo-row:last-of-type {border-bottom:none;}
.cf-mini-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem;margin-top:.75rem;}
.cf-mini-stat {background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:.55rem .7rem;}
.cf-mini-num {font-size:1.05rem;font-weight:800;color:#0f172a!important;line-height:1.2;}
.cf-mini-cap {font-size:.69rem;font-weight:650;color:#64748b!important;text-transform:uppercase;letter-spacing:.04em;margin-top:.12rem;}
.cf-prov-badge-live,.cf-prov-badge-demo,.cf-prov-badge-local {display:inline-flex;align-items:center;gap:.25rem;padding:.22rem .6rem;border-radius:999px;font-size:.71rem;font-weight:700;}
.cf-prov-badge-live {background:#ecfeff;border:1px solid #a5f3fc;color:#0369a1!important;}
.cf-prov-badge-demo {background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8!important;}
.cf-prov-badge-local {background:#f8fafc;border:1px solid #cbd5e1;color:#475569!important;}
.cf-src-audio {background:#eff6ff!important;border-color:#93c5fd!important;color:#1d4ed8!important;}
.cf-src-board {background:#f0fdf4!important;border-color:#86efac!important;color:#15803d!important;}
.cf-src-slides {background:#f0f9ff!important;border-color:#7dd3fc!important;color:#0369a1!important;}
.cf-src-notes {background:#fefce8!important;border-color:#fde047!important;color:#a16207!important;}
/* Hierarchical NOTES document styling (Module 2 exclusive) */
.cf-notes-doc-header {display:flex;justify-content:space-between;align-items:center;gap:.75rem;flex-wrap:wrap;padding:.72rem 1rem;margin:.65rem 0 .7rem 0;background:linear-gradient(90deg,#f0f9ff 0%,#ffffff 100%);border:1px solid #bae6fd;border-radius:12px;box-shadow:0 2px 4px rgba(15,23,42,.03);}
.cf-notes-doc-title-row {display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;}
.cf-notes-badge {display:inline-flex;align-items:center;padding:.24rem .72rem;border-radius:8px;background:#0284c7;color:#ffffff!important;font-size:.76rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;box-shadow:0 1px 2px rgba(2,132,199,.25);}
.cf-notes-doc-title {font-size:1.05rem;font-weight:800;color:#0f172a!important;letter-spacing:-.015em;}
.cf-notes-meta {font-size:.78rem;font-weight:600;color:#475569!important;}
.cf-notes-outline {background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:.85rem 1.05rem;margin-bottom:.95rem;box-shadow:0 2px 4px rgba(15,23,42,.025);}
.cf-notes-outline-label {font-size:.72rem;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:#0284c7!important;margin-bottom:.5rem;}
.cf-notes-toc-item {padding:.38rem 0;border-bottom:1px solid #f1f5f9;}
.cf-notes-toc-item:last-child {border-bottom:none;padding-bottom:0;}
.cf-notes-toc-main {font-size:.88rem;font-weight:700;color:#0f172a!important;}
.cf-notes-toc-subs {display:flex;flex-wrap:wrap;gap:.38rem;margin:.3rem 0 .1rem 1.25rem;}
.cf-notes-toc-chip {display:inline-flex;align-items:center;gap:.3rem;font-size:.75rem;color:#334155!important;background:#f8fafc;border:1px solid #e2e8f0;border-radius:999px;padding:.15rem .58rem;}
.cf-notes-toc-chip b {color:#0284c7!important;font-weight:750;}
.cf-notes-section {background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #0284c7;border-radius:12px;padding:1.05rem 1.2rem;margin-bottom:1rem;box-shadow:0 4px 6px -1px rgba(15,23,42,.035);}
.cf-notes-h1-row {display:flex;justify-content:space-between;align-items:center;gap:.7rem;flex-wrap:wrap;padding-bottom:.65rem;margin-bottom:.85rem;border-bottom:1px solid #f1f5f9;}
.cf-notes-h1-left {display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;}
.cf-notes-sec-num {display:inline-flex;align-items:center;justify-content:center;min-width:1.75rem;height:1.75rem;padding:0 .4rem;border-radius:8px;background:#e0f2fe;border:1px solid #bae6fd;color:#0369a1!important;font-size:.86rem;font-weight:800;}
.cf-notes-h1 {font-size:1.12rem;font-weight:800;color:#0f172a!important;line-height:1.3;}
.cf-notes-subblock {margin:.72rem 0 .82rem .35rem;padding:.7rem .95rem;border-left:3px solid #bae6fd;background:#f8fafc;border-radius:0 10px 10px 0;}
.cf-notes-h2 {display:flex;align-items:baseline;gap:.45rem;font-size:.99rem;font-weight:750;color:#0f172a!important;margin-bottom:.42rem;line-height:1.35;}
.cf-notes-sub-num {color:#0284c7!important;font-weight:800;font-size:.94rem;flex-shrink:0;}
.cf-notes-prose p {color:#334155!important;font-size:.92rem;line-height:1.68;margin:.32rem 0!important;}
.cf-notes-formula {margin:.5rem 0;padding:.58rem .85rem;background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;color:#0369a1!important;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88rem;font-weight:650;line-height:1.55;white-space:pre-wrap;}
.cf-notes-bullets {margin:.38rem 0 .1rem 1.15rem!important;}
.cf-notes-bullets li {font-size:.9rem;line-height:1.6;color:#334155!important;margin:.22rem 0;}
.cf-advisor-box {background:linear-gradient(135deg,#f0f9ff 0%,#ffffff 100%);border:1px solid #bae6fd;border-left:4px solid #0284c7;border-radius:12px;padding:.95rem 1.1rem;margin:.6rem 0 .9rem 0;color:#1e293b!important;line-height:1.65;}
.cf-advisor-label {font-size:.72rem;font-weight:800;letter-spacing:.065em;text-transform:uppercase;color:#0284c7!important;margin-bottom:.35rem;display:block;}
.cf-calendar-wrap {width:100%;overflow-x:auto;border:1px solid #e2e8f0;border-radius:12px;background:#fff;box-shadow:0 4px 6px -1px rgba(15,23,42,.035);}
.cf-calendar {width:100%;border-collapse:separate;border-spacing:0;min-width:760px;}
.cf-calendar th,.cf-calendar td {padding:.65rem .55rem;border-right:1px solid #eef2f7;border-bottom:1px solid #eef2f7;vertical-align:top;text-align:left;}
.cf-calendar thead th {background:#f8fafc;color:#475569;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;font-weight:750;}
.cf-calendar-time {background:#f8fafc;color:#64748b;font-size:.72rem;white-space:nowrap;min-width:120px;}
.cf-calendar-free {color:#94a3b8;font-size:.75rem;}
.cf-calendar-class {background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:.4rem .45rem;margin:.12rem 0;color:#1e3a8a;}
.cf-calendar-class strong {display:block;font-size:.76rem;color:#1e40af;}
.cf-calendar-class small {display:block;font-size:.66rem;color:#475569;line-height:1.4;margin-top:.12rem;}
.cf-calendar-clash {background:#fff1f2;border-color:#fecdd3;color:#9f1239;}
.cf-calendar-clash strong {color:#be123c;}
.cf-step-head {display:flex;justify-content:space-between;align-items:center;gap:.65rem;flex-wrap:wrap;margin-bottom:.45rem;}
.cf-step-title-group {display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;}
.cf-step-pill {display:inline-flex;align-items:center;padding:.2rem .65rem;border-radius:999px;background:#0284c7;color:#ffffff!important;font-size:.7rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}
.cf-step-title {font-size:1.04rem;font-weight:750;color:#0f172a!important;letter-spacing:-.015em;}
.cf-alt-row {display:flex;justify-content:space-between;align-items:center;gap:.7rem;flex-wrap:wrap;padding:.58rem .8rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:.35rem 0;}
.cf-alt-row-zero {background:#f0fdf4!important;border-color:#bbf7d0!important;}
.cf-alt-row-clash {background:#fff7ed!important;border-color:#fed7aa!important;}
[data-testid="stDownloadButton"] button[kind="primary"] {background:var(--cf-blue)!important;border:1px solid var(--cf-blue)!important;color:#ffffff!important;border-radius:9px!important;font-weight:650!important;}
[data-testid="stDownloadButton"] button[kind="primary"] * {color:#ffffff!important;}
@media (max-width: 900px) {.block-container{padding-left:1rem!important;padding-right:1rem!important;}.cf-study-card{padding:.85rem!important;}}
</style>
"""
st.html(MATTE_CSS)


def _render_weekly_timetable(matrix_df) -> None:
    """Render a compact, readable weekly timetable using the clash-checked matrix."""
    headers = "".join(f"<th>{html.escape(str(day))}</th>" for day in matrix_df.columns)
    rows = []
    for slot, values in matrix_df.iterrows():
        cells = []
        for day in matrix_df.columns:
            raw = str(values[day] or "").strip()
            if not raw:
                cell = '<td><span class="cf-calendar-free">Free</span></td>'
            else:
                entries = [part.strip() for part in raw.split(" | ") if part.strip()]
                blocks = []
                for entry in entries:
                    is_clash = "CLASH:" in entry
                    label = entry.replace("CLASH:", "").strip()
                    blocks.append(
                        '<div class="cf-calendar-class cf-calendar-clash"' if is_clash
                        else '<div class="cf-calendar-class"'
                    )
                    blocks[-1] += f"><strong>{html.escape(label.split(' · ')[0].split(' (')[0])}</strong><small>{html.escape(label)}</small></div>"
                cell = f"<td>{''.join(blocks)}</td>"
            cells.append(cell)
        rows.append(f"<tr><th class=\"cf-calendar-time\">{html.escape(str(slot))}</th>{''.join(cells)}</tr>")
    markup = f'<div class="cf-calendar-wrap"><table class="cf-calendar"><thead><tr><th>Time</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    _render_html(markup)


def _render_html(markup: str, target=None) -> None:
    """Render HTML via st.markdown after stripping leading indentation to avoid code blocks."""
    cleaned = "\n".join(
        line.lstrip() for line in textwrap.dedent(markup).splitlines() if line.strip()
    )
    (target or st).markdown(cleaned, unsafe_allow_html=True)

COURSE_CATALOG_CONFIG: dict[str, dict] = {
    "CSC322": {
        "code": "CSC322",
        "subject": "Operating Systems",
        "label": timetable.format_course_label("CSC322"),
        "topic": "First-Come, First-Served (FCFS) CPU Scheduling",
        "short_topic": "FCFS CPU Scheduling",
        "audio_file": "csc322_os_audio.wav",
        "audio_desc": "Spoken lecture on FIFO ready queue & Convoy Effect",
        "image_file": "csc322_os_whiteboard.png",
        "image_desc": "Gantt chart [0–9] & smudged P2 burst time cell",
        "pdf_file": "csc322_os_slides.pdf",
        "pdf_desc": "Non-preemptive rules & turnaround formulas",
        "sample_notes": (
            "Lecture on First-Come, First-Served (FCFS) non-preemptive CPU scheduling using a FIFO queue. "
            "Board trace: P1 (arrival 0 ms, burst 4 ms), P2 (arrival 1 ms, Gantt slot [4–7] → burst 3 ms), "
            "P3 (arrival 2 ms, burst 2 ms). Average Waiting Time = 2.67 ms, Average Turnaround = 5.67 ms. "
            "Convoy Effect happens when one long CPU-bound process stalls short jobs behind it."
        ),
        "quick_questions": [
            "How was P2's missing burst time deduced from the Gantt chart?",
            "What are the Turnaround Time and Waiting Time formulas?",
            "What causes the Convoy Effect in FCFS scheduling?",
        ],
    },
    "CSC325": {
        "code": "CSC325",
        "subject": "Computer Organization & Assembly",
        "label": timetable.format_course_label("CSC325"),
        "topic": "IEEE 754 Single-Precision Floating Point",
        "short_topic": "IEEE 754 Floating Point",
        "audio_file": "csc325_coal_audio.wav",
        "audio_desc": "Walkthrough of −12.625 binary conversion & bias 127",
        "image_file": "csc325_coal_whiteboard.png",
        "image_desc": "32-bit layout & smudged 8-bit biased exponent field",
        "pdf_file": "csc325_coal_slides.pdf",
        "pdf_desc": "Field widths, hidden leading 1, & special values",
        "sample_notes": (
            "Lecture on IEEE 754 32-bit single-precision floating point: 1 Sign bit, 8 Biased Exponent bits (bias 127), "
            "and 23 Mantissa fraction bits with an implicit hidden leading 1. "
            "Worked board conversion for −12.625: Sign S = 1, 12.625 = 1100.101₂ = 1.100101 × 2³. "
            "Biased exponent = 127 + 3 = 130 (10000010₂), hexadecimal 0xC14A0000."
        ),
        "quick_questions": [
            "How is the 8-bit biased exponent for -12.625 calculated?",
            "What are the Sign, Exponent, and Mantissa bit widths in 32-bit IEEE 754?",
            "How does -12.625 pack into hexadecimal 0xC14A0000?",
        ],
    },
    "AIC354": {
        "code": "AIC354",
        "subject": "Machine Learning Fundamentals",
        "label": timetable.format_course_label("AIC354"),
        "topic": "Naive Bayes & Bayesian Belief Classification",
        "short_topic": "Naive Bayes Classification",
        "audio_file": "aic354_ml_audio.wav",
        "audio_desc": "Conditional independence & log-space spam filter example",
        "image_file": "aic354_ml_whiteboard.png",
        "image_desc": "Bayes rule & smudged Laplace smoothing denominator",
        "pdf_file": "aic354_ml_slides.pdf",
        "pdf_desc": "Zero-frequency problem & vocabulary size smoothing",
        "sample_notes": (
            "Lecture on Naive Bayes classification using conditional independence and log-space prediction: "
            "pick class y with highest [ log P(y) + Σ log P(featureᵢ | y) ]. "
            "Laplace (Add-1) smoothing = (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V) prevents unseen words from zeroing scores. "
            "Spam example (V = 1,000, Total Spam Words = 200): 'prize' count 5 → 6 ÷ 1,200; unseen 'urgent' count 0 → 1 ÷ 1,200."
        ),
        "quick_questions": [
            "Why do we add Vocabulary Size V in Laplace smoothing?",
            "What conditional independence assumption does Naive Bayes make?",
            "Why sum log probabilities instead of multiplying decimals?",
        ],
    },
    "CSC301": {
        "code": "CSC301",
        "subject": "Design & Analysis of Algorithms",
        "label": timetable.format_course_label("CSC301"),
        "topic": "Binary Search & Recurrence Complexity",
        "short_topic": "Binary Search & Recurrence",
        "audio_file": "csc301_daa_audio.wav",
        "audio_desc": "Divide-and-conquer probe trace & Master Theorem proof",
        "image_file": "csc301_daa_whiteboard.png",
        "image_desc": "Trace on [3..45] & smudged overflow-safe midpoint equation",
        "pdf_file": "csc301_daa_slides.pdf",
        "pdf_desc": "Iterative algorithm, invariants, & logarithmic bound",
        "sample_notes": (
            "Lecture on Binary Search over sorted ascending arrays. "
            "Overflow-safe midpoint formula = low + (high − low) ÷ 2 prevents signed 32-bit overflow. "
            "Recurrence T(n) = T(n ÷ 2) + constant solves by Master Theorem Case 2 to Θ(log n) runtime and O(1) iterative space. "
            "Trace on [3, 8, 14, 21, 29, 37, 45] for target 37: Probe 1 checks index 3 (21 < 37 → low = 4), Probe 2 checks index 5 (37 found)."
        ),
        "quick_questions": [
            "Why is low + (high − low) ÷ 2 safer than averaging (low + high) ÷ 2?",
            "How does T(n) = T(n ÷ 2) + c solve to Θ(log n) time?",
            "Trace Binary Search on [3, 8, 14, 21, 29, 37, 45] for target 37.",
        ],
    },
}

CUSTOM_COURSE_OPTION = "Custom Course / Any Subject..."
COURSE_OPTIONS = [
    COURSE_CATALOG_CONFIG[c]["label"] for c in ("CSC322", "CSC325", "AIC354", "CSC301")
] + [CUSTOM_COURSE_OPTION]


def _parse_custom_course_input(raw_text: str) -> tuple[str, str]:
    """Parse free-form custom course input into `(course_code, display_label)`."""
    cleaned = " ".join((raw_text or "").strip().split())
    if not cleaned:
        return "GENERAL", "GENERAL — Custom Subject"
    for sep in ("—", "–", " - ", ": "):
        if sep in cleaned:
            head, tail = cleaned.split(sep, 1)
            head_clean = head.strip()
            tail_clean = tail.strip()
            code_val = head_clean.upper() if head_clean else "CUSTOM"
            full_label = f"{code_val} — {tail_clean}" if tail_clean else code_val
            return code_val, full_label
    return cleaned.upper(), cleaned


def _display_course_label(lecture: dict | None, fallback_code: str = "CSC322") -> str:
    """Return full course display label, respecting custom names stored in summary."""
    if lecture:
        summary = lecture.get("summary") or {}
        custom_lbl = str(summary.get("course_label") or "").strip()
        if custom_lbl:
            return custom_lbl
        return timetable.format_course_label(lecture["course_code"])
    return timetable.format_course_label(fallback_code)


def _provenance_badge_info(summary: dict | None, chunk_count: int = 0) -> tuple[str, str]:
    """Return `(badge_text, css_class)` identifying Live Gemini vs Demo Kit vs Local Fallback."""
    summary = summary or {}
    source = str(summary.get("generation_source") or "").strip().lower()
    model_id = str(summary.get("generation_model") or "").strip()
    if source == "gemini_live":
        label = (
            f"✨ Generated Live via Gemini Flash API ({model_id})"
            if model_id
            else "✨ Generated Live via Gemini Flash API"
        )
        return label, "cf-prov-badge-live"
    if source == "local_fallback":
        return "Offline Local Synthesis Fallback", "cf-prov-badge-local"
    if source == "demo_kit" or chunk_count >= 0:
        return "⚡ Loaded from Demo Kit", "cf-prov-badge-demo"
    return "⚡ Loaded from Demo Kit", "cf-prov-badge-demo"


def _uploaded_modality_badge(summary: dict | None) -> str:
    """Build a crisp upload-provenance badge string from stored `uploaded_labels`."""
    raw_labels = list((summary or {}).get("uploaded_labels") or [])
    if not raw_labels:
        return ""
    pretty_map = {
        "Audio": "Lecture Audio",
        "Board Photo": "Whiteboard Photo",
        "Slides PDF": "Slides PDF",
        "Classroom Notes": "Classroom Notes",
    }
    names = [pretty_map.get(str(lbl).strip(), str(lbl).strip()) for lbl in raw_labels if str(lbl).strip()]
    if not names:
        return ""
    if len(names) == 1:
        joined = names[0]
    elif len(names) == 2:
        joined = f"{names[0]} & {names[1]}"
    else:
        joined = ", ".join(names[:-1]) + f" & {names[-1]}"
    return f"✨ Processed from Uploaded {joined}"


def _short_topic_label(title: str, course_code: str = "") -> str:
    """Return a clean concise topic string for headers and selectors."""
    cleaned = title.strip()
    cfg = COURSE_CATALOG_CONFIG.get(course_code.strip().upper())
    if cfg and cleaned == cfg["topic"]:
        return cfg["short_topic"]
    return cleaned or (cfg["short_topic"] if cfg else "Lecture Topic")


def _format_file_kb(path: Path) -> str:
    if not path.exists():
        return "Missing"
    size_kb = path.stat().st_size / 1024.0
    if size_kb >= 1024:
        return f"{size_kb / 1024.0:.1f} MB"
    return f"{size_kb:.0f} KB"


def _build_practice_lec_map(lectures: list[dict]) -> dict[str, int]:
    """Build deterministic Practice selector labels -> lecture IDs, disambiguating duplicates if added."""
    base_labels = [
        f"{_display_course_label(l)} · {_short_topic_label(l['lecture_title'], l['course_code'])}"
        for l in lectures
    ]
    counts: dict[str, int] = {}
    for lbl in base_labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    mapping: dict[str, int] = {}
    for lec, lbl in zip(lectures, base_labels):
        final_lbl = f"{lbl} (#{lec['id']})" if counts[lbl] > 1 else lbl
        mapping[final_lbl] = int(lec["id"])
    return mapping


def _find_practice_label_for_id(lec_map: dict[str, int], lec_id: int | None) -> str | None:
    if lec_id is None:
        return next(iter(lec_map.keys()), None)
    for lbl, lid in lec_map.items():
        if lid == lec_id:
            return lbl
    return next(iter(lec_map.keys()), None)


def _latest_lecture_for_course(lectures: list[dict], course_code: str) -> dict | None:
    target = course_code.strip().upper()
    return next((l for l in lectures if l["course_code"].strip().upper() == target), None)


def _apply_course_sync(
    course_code: str,
    lecture_id: int | None,
    lectures: list[dict],
) -> None:
    """Synchronize shared course/lecture session keys BEFORE any widgets instantiate."""
    is_custom_opt = course_code in (CUSTOM_COURSE_OPTION, "Custom")
    code_upper = CUSTOM_COURSE_OPTION if is_custom_opt else course_code.strip().upper()
    cfg = COURSE_CATALOG_CONFIG.get(code_upper)
    lec_map = _build_practice_lec_map(lectures)

    if is_custom_opt:
        st.session_state["active_course_code"] = CUSTOM_COURSE_OPTION
        st.session_state["cap_course_choice"] = CUSTOM_COURSE_OPTION
        st.session_state["cap_custom_course"] = ""
        st.session_state["cap_title"] = ""
        return

    st.session_state["active_course_code"] = code_upper
    target_lec = None
    if lecture_id is not None:
        target_lec = next((l for l in lectures if l["id"] == lecture_id), None)
    if target_lec is None:
        target_lec = _latest_lecture_for_course(lectures, code_upper)
    if target_lec is None and lectures:
        target_lec = lectures[0]

    if cfg:
        st.session_state["cap_course_choice"] = cfg["label"]
        st.session_state["cap_title"] = (
            target_lec["lecture_title"] if target_lec else cfg["topic"]
        )
    else:
        st.session_state["cap_course_choice"] = CUSTOM_COURSE_OPTION
        st.session_state["cap_custom_course"] = _display_course_label(target_lec, code_upper)
        st.session_state["cap_title"] = target_lec["lecture_title"] if target_lec else ""

    if target_lec is not None:
        st.session_state["active_lecture_id"] = target_lec["id"]
        st.session_state["library_selected_id"] = target_lec["id"]
        prac_label = _find_practice_label_for_id(lec_map, target_lec["id"])
        if prac_label:
            st.session_state["practice_lecture_select"] = prac_label


def _on_cap_course_change() -> None:
    """React immediately when Course selector changes in ScribeAI Studio."""
    chosen = st.session_state.get("cap_course_choice", COURSE_OPTIONS[0])
    lectures = _cached_list_lectures()
    if chosen == CUSTOM_COURSE_OPTION:
        _apply_course_sync(CUSTOM_COURSE_OPTION, None, lectures)
    else:
        code = timetable.parse_course_code(chosen)
        lec = _latest_lecture_for_course(lectures, code)
        _apply_course_sync(code, lec["id"] if lec else None, lectures)


def _on_practice_lec_change() -> None:
    """React immediately when Lecture selector changes in Revision Intelligence."""
    chosen_label = st.session_state.get("practice_lecture_select")
    lectures = _cached_list_lectures()
    lec_map = _build_practice_lec_map(lectures)
    lec_id = lec_map.get(chosen_label or "")
    if lec_id is None:
        return
    lec = next((l for l in lectures if l["id"] == lec_id), None)
    if lec is None:
        return
    _apply_course_sync(lec["course_code"], lec["id"], lectures)


def _render_feedback_bar(
    lecture: dict,
    chunks: list[dict],
) -> None:
    """Render persistent Good / Not Good note quality feedback & Gemini regeneration bar."""
    lecture_id = int(lecture["id"])
    summary = lecture.get("summary") or {}

    fb_notice_key = f"fb_regen_notice_{lecture_id}"
    if fb_notice_key in st.session_state:
        st.success(st.session_state.pop(fb_notice_key))

    improve_open_key = f"fb_improve_open_{lecture_id}"
    improve_open = bool(st.session_state.get(improve_open_key, False))
    latest_fb = database.get_latest_lecture_feedback(lecture_id)
    is_verified_good = (not improve_open) and (latest_fb is not None) and (latest_fb["rating"] == "good")

    with st.container(border=True):
        if is_verified_good:
            status_right = '<span class="cf-badge-ok">Feedback recorded — Notes marked as verified!</span>'
        elif summary.get("regenerated_with_feedback"):
            status_right = '<span class="cf-source-pill">Regenerated from feedback</span>'
        else:
            status_right = '<span class="cf-muted" style="font-size:0.77rem;">Rate note depth &amp; clarity</span>'

        _render_html(
            f"""
            <div data-cf-feedback-bar="true" style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.25rem;">
                <span style="color:#0f172a;font-size:0.86rem;font-weight:700;">Note Quality Feedback</span>
                {status_right}
            </div>
            """
        )

        fb_c1, fb_c2 = st.columns(2, gap="small")
        if fb_c1.button("👍 Good Notes", key=f"fb_good_{lecture_id}", width="stretch"):
            database.save_lecture_feedback(lecture_id, "good", "")
            st.session_state[improve_open_key] = False
            st.rerun()

        if fb_c2.button("👎 Not Good — Improve & Regenerate", key=f"fb_bad_{lecture_id}", width="stretch"):
            database.save_lecture_feedback(lecture_id, "bad", "")
            st.session_state[improve_open_key] = True
            st.rerun()

        if improve_open:
            feedback_val = st.text_input(
                "What should Gemini improve? (e.g., Make explanations deeper, add worked examples, simplify formulas)",
                placeholder="e.g., Make explanations deeper, add worked examples, simplify formulas",
                key=f"fb_text_{lecture_id}",
            )
            regen_c1, regen_c2 = st.columns([2.4, 0.9], gap="small")
            if regen_c1.button(
                "✨ Regenerate Notes with Gemini",
                key=f"fb_regen_{lecture_id}",
                type="primary",
                width="stretch",
            ):
                final_fb = (feedback_val or "").strip() or (
                    "Make explanations deeper, add worked examples, simplify formulas"
                )
                database.save_lecture_feedback(lecture_id, "bad", final_fb)
                with st.spinner("Regenerating improved lecture notes with Gemini..."):
                    regen_result, used_fb = ai_engine.regenerate_lecture_notes_with_timeout(
                        course_code=lecture["course_code"],
                        lecture_title=lecture["lecture_title"],
                        current_summary=summary,
                        chunks=chunks,
                        feedback_text=final_fb,
                        timeout_seconds=ai_engine.SYNTHESIS_TIMEOUT_SECONDS,
                    )
                used_model = getattr(regen_result, "_model_used", "") or ai_engine.load_api_environment()[1]
                regen_chunks = [c.model_dump() for c in regen_result.knowledge_chunks]
                updated_summary = {
                    "course_label": summary.get("course_label") or _display_course_label(lecture),
                    "detected_topic": summary.get("detected_topic") or regen_result.detected_topic or lecture["lecture_title"],
                    "uploaded_labels": list(summary.get("uploaded_labels") or []),
                    "generation_source": "local_fallback" if used_fb else "gemini_live",
                    "generation_model": "" if used_fb else used_model,
                    "regenerated_with_feedback": final_fb,
                    "executive_summary": regen_result.executive_summary,
                    "structured_notes": [s.model_dump() for s in regen_result.structured_notes],
                    "extracted_pseudocode": regen_result.extracted_pseudocode,
                }
                database.update_lecture_notes(
                    lecture_id,
                    updated_summary,
                    chunks=regen_chunks or None,
                )
                refreshed_chunks = database.get_lecture_chunks(lecture_id)
                if refreshed_chunks:
                    try:
                        pack, _ = ai_engine.generate_revision_pack_with_timeout(
                            lecture_title=f"{_display_course_label(lecture)} {lecture['lecture_title']}",
                            chunks=refreshed_chunks,
                            use_cache=False,
                            timeout_seconds=ai_engine.SYNTHESIS_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        pack = ai_engine.build_local_revision_pack(
                            lecture_title=f"{_display_course_label(lecture)} {lecture['lecture_title']}",
                            chunks=refreshed_chunks,
                        )
                    database.save_revision_items(lecture_id, "mcq", [m.model_dump() for m in pack.mcqs])
                    database.save_revision_items(lecture_id, "flashcard", [f.model_dump() for f in pack.flashcards])
                    st.session_state[f"quiz_idx_{lecture_id}"] = 0
                    st.session_state[f"quiz_submitted_{lecture_id}"] = False
                    st.session_state[f"quiz_round_{lecture_id}"] = (
                        st.session_state.get(f"quiz_round_{lecture_id}", 0) + 1
                    )
                    if "quiz_answers_by_lec" in st.session_state:
                        st.session_state["quiz_answers_by_lec"].pop(lecture_id, None)
                    st.session_state["quiz_answers"] = {}
                    st.session_state[f"fc_idx_{lecture_id}"] = 0
                    st.session_state[f"fc_flip_{lecture_id}"] = False

                _cached_list_lectures_for.clear()
                st.session_state[improve_open_key] = False
                st.session_state["pending_sync"] = {
                    "course_code": lecture["course_code"],
                    "lecture_id": lecture_id,
                }
                st.session_state[fb_notice_key] = (
                    "Notes enhanced and 10 MCQs / 6 flashcards refreshed in your Lecture Vault."
                    if used_fb
                    else f"✨ Notes regenerated live via Gemini ({used_model}) and synced with 10 MCQs & 6 flashcards!"
                )
                st.rerun()
            if regen_c2.button("Cancel", key=f"fb_cancel_{lecture_id}", width="stretch"):
                st.session_state[improve_open_key] = False
                st.rerun()


def _resolve_section_badge(sec: dict, chunks: list[dict]) -> tuple[str, str]:
    """Return `(badge_label, css_class)` using strict canonical modality tags."""
    raw_tag = str(sec.get("source_type") or sec.get("source_tag") or "").strip().upper()
    if not raw_tag:
        if chunks:
            raw_tag = str(chunks[0].get("source_type") or "SLIDES_PDF").upper()
        else:
            raw_tag = "SLIDES_PDF"

    if raw_tag in ("AUDIO", "LECTURE_AUDIO"):
        return "AUDIO", "cf-src-audio"
    if raw_tag in ("WHITEBOARD", "WHITEBOARD_IMAGE", "BOARD", "IMAGE"):
        return "WHITEBOARD_IMAGE", "cf-src-board"
    if raw_tag in ("NOTES",):
        return "NOTES", "cf-src-notes"
    return "SLIDES_PDF", "cf-src-slides"


def _render_lecture_notes(
    lecture: dict,
    chunks: list[dict],
    key_prefix: str,
) -> None:
    """Render provenance header, feedback bar, and clean hierarchical NOTES (Module 2 exclusive)."""
    summary = lecture.get("summary", {})
    course_display = _display_course_label(lecture)
    prov_label, prov_cls = _provenance_badge_info(summary, len(chunks))
    up_badge = _uploaded_modality_badge(summary)
    up_badge_html = (
        f'<span class="cf-pill-active">{html.escape(up_badge)}</span>' if up_badge else ""
    )

    _render_html(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.2rem;">
            <span style="color:#0f172a;font-size:1.38rem;font-weight:700;line-height:1.25;">{html.escape(course_display)} — {html.escape(lecture['lecture_title'])}</span>
            <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;">
                {up_badge_html}
                <span class="{prov_cls}">{html.escape(prov_label)}</span>
            </div>
        </div>
        """
    )
    st.caption(f"{lecture['created_at']} · {len(chunks)} captured knowledge chunks")

    _render_feedback_bar(lecture, chunks)

    exec_sum = summary.get("executive_summary", "")
    cached_images = st.session_state.get(f"uploaded_images_{lecture['id']}") or []
    renderable_images: list[tuple[str, bytes]] = []
    if cached_images:
        import io as _io
        from PIL import Image as _PILImage

        for img_name, img_data in cached_images:
            try:
                with _PILImage.open(_io.BytesIO(img_data)) as _chk:
                    _chk.verify()
                renderable_images.append((img_name, img_data))
            except Exception:
                pass

    overview_html = (
        f"""
        <div class="cf-card">
            <div style="color:#0284c7;font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-bottom:0.4rem;">Overview</div>
            <div style="color:#334155;line-height:1.6;">{html.escape(exec_sum)}</div>
        </div>
        """
        if exec_sum
        else ""
    )

    if renderable_images and exec_sum:
        ov_col, img_col = st.columns([1.45, 1], gap="medium")
        with ov_col:
            _render_html(overview_html)
        with img_col:
            for img_name, img_data in renderable_images:
                st.image(img_data, caption=f"Uploaded Whiteboard Photo: {img_name}", width="stretch")
    elif exec_sum:
        _render_html(overview_html)
    elif renderable_images:
        for img_name, img_data in renderable_images:
            st.image(img_data, caption=f"Uploaded Whiteboard Photo: {img_name}", width="stretch")

    sections = summary.get("structured_notes", [])
    if sections:
        normalized_sections: list[tuple[str, str, str, list[dict], str]] = []
        total_subs = 0
        for sec in sections:
            heading = str(sec.get("heading") or "Study Section").strip()
            badge_label, badge_cls = _resolve_section_badge(sec, chunks)
            sub_list = ai_engine.normalize_note_subsections(sec)
            takeaway = str(sec.get("takeaway") or "").strip()
            total_subs += len(sub_list)
            normalized_sections.append((heading, badge_label, badge_cls, sub_list, takeaway))

        _render_html(
            f"""
            <div class="cf-notes-doc-header" data-cf-notes="{html.escape(key_prefix)}">
                <div class="cf-notes-doc-title-row">
                    <span class="cf-notes-badge">NOTES</span>
                    <span class="cf-notes-doc-title">Structured Lecture Study Notes</span>
                </div>
                <span class="cf-notes-meta">{len(normalized_sections)} Sections · {total_subs} Subheadings</span>
            </div>
            """
        )

        toc_items: list[str] = []
        for s_idx, (heading, _, _, sub_list, _) in enumerate(normalized_sections, start=1):
            sub_chips = "".join(
                f'<span class="cf-notes-toc-chip"><b>{s_idx}.{sub_idx}</b> {html.escape(sub["subheading"])}</span>'
                for sub_idx, sub in enumerate(sub_list, start=1)
            )
            subs_wrap = f'<div class="cf-notes-toc-subs">{sub_chips}</div>' if sub_chips else ""
            toc_items.append(
                f'<div class="cf-notes-toc-item"><div class="cf-notes-toc-main">{s_idx}. {html.escape(heading)}</div>{subs_wrap}</div>'
            )

        _render_html(
            f"""
            <div class="cf-notes-outline">
                <div class="cf-notes-outline-label">Notes Outline — Headings &amp; Subheadings</div>
                {"".join(toc_items)}
            </div>
            """
        )

        for s_idx, (heading, badge_label, badge_cls, sub_list, takeaway) in enumerate(
            normalized_sections, start=1
        ):
            sub_blocks: list[str] = []
            for sub_idx, sub in enumerate(sub_list, start=1):
                body_txt = str(sub.get("body") or "").strip()
                paras = [p.strip() for p in body_txt.split("\n\n") if p.strip()]
                prose_html = (
                    '<div class="cf-notes-prose">' + "".join(f"<p>{html.escape(p)}</p>" for p in paras) + "</div>"
                    if paras
                    else ""
                )
                formula_txt = str(sub.get("formula") or "").strip()
                formula_html = (
                    f'<div class="cf-notes-formula">{html.escape(formula_txt)}</div>'
                    if formula_txt
                    else ""
                )
                bullets = [str(b).strip() for b in (sub.get("bullets") or []) if str(b).strip()]
                bullets_html = (
                    '<ul class="cf-study-list cf-notes-bullets">'
                    + "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
                    + "</ul>"
                    if bullets
                    else ""
                )
                sub_blocks.append(
                    f'<div class="cf-notes-subblock">'
                    f'<div class="cf-notes-h2"><span class="cf-notes-sub-num">{s_idx}.{sub_idx}</span>'
                    f'<span>{html.escape(str(sub.get("subheading") or "Topic"))}</span></div>'
                    f'{prose_html}{formula_html}{bullets_html}'
                    f'</div>'
                )

            takeaway_callout_html = ""
            if takeaway:
                takeaway_callout_html = (
                    f'<div class="cf-takeaway-box">'
                    f'<span class="cf-takeaway-label">Key Takeaway</span>'
                    f'{html.escape(takeaway)}'
                    f'</div>'
                )

            _render_html(
                f"""
                <div class="cf-notes-section">
                    <div class="cf-notes-h1-row">
                        <div class="cf-notes-h1-left">
                            <span class="cf-notes-sec-num">{s_idx}.</span>
                            <span class="cf-notes-h1">{html.escape(heading)}</span>
                        </div>
                        <span class="cf-src-badge {badge_cls}">{html.escape(badge_label)}</span>
                    </div>
                    {"".join(sub_blocks)}
                    {takeaway_callout_html}
                </div>
                """
            )

    pseudocode = (summary.get("extracted_pseudocode") or "").strip()
    if pseudocode:
        st.markdown("#### Formulas, Code & Worked Example")
        st.code(pseudocode, language="text")


def _load_canonical_sample_kit(course_code: str) -> int:
    """Load pre-built canonical chunks, 10 MCQs, and 6 flashcards instantly with zero network calls."""
    code = course_code.strip().upper()
    cfg = COURSE_CATALOG_CONFIG.get(code, COURSE_CATALOG_CONFIG["CSC322"])
    target_code = cfg["code"]
    target_topic = cfg["topic"]

    existing_lectures = database.list_lectures()
    for lec in existing_lectures:
        if lec["course_code"].strip().upper() == target_code and lec["lecture_title"].strip() == target_topic:
            chunks = database.get_lecture_chunks(lec["id"])
            mcqs = database.get_revision_items(lec["id"], "mcq")
            cards = database.get_revision_items(lec["id"], "flashcard")
            if chunks and len(mcqs) >= 10 and len(cards) >= 6:
                return int(lec["id"])

    seeders = {
        "CSC322": database._seed_csc322_fcfs,
        "CSC325": database._seed_csc325_ieee754,
        "AIC354": database._seed_aic354_naive_bayes,
        "CSC301": database._seed_csc301_binary_search,
    }
    seeder = seeders.get(target_code, database._seed_csc322_fcfs)
    new_id = int(seeder())
    _cached_list_lectures_for.clear()
    return new_id


def _run_synthesis_and_save(
    course_code: str,
    course_label: str,
    lecture_title: str,
    audio_payload: list[tuple[str, bytes, str]],
    image_payload: list[tuple[str, bytes, str]],
    pdf_payload: list[tuple[str, bytes, str]],
    notes_text: str,
    status_cb=None,
) -> tuple[int, int, bool, str]:
    """Run live Gemini lecture synthesis (with 35s timeout), persist SQLite chunks, and build revision pack."""
    mod_info = ai_engine.describe_provided_modalities(
        audio_files=audio_payload,
        image_files=image_payload,
        pdf_files=pdf_payload,
        extra_notes=notes_text,
    )
    active_labels_str = ", ".join(mod_info["labels"]) or "Lecture Inputs"
    if status_cb:
        status_cb(
            f"Reading {active_labels_str} with Gemini Flash API "
            f"(up to {int(ai_engine.SYNTHESIS_TIMEOUT_SECONDS)}s for multimodal comprehension)..."
        )

    synthesis, syn_fallback = ai_engine.synthesize_lecture_with_timeout(
        course_code=course_code,
        lecture_title=lecture_title,
        audio_files=audio_payload,
        image_files=image_payload,
        pdf_files=pdf_payload,
        extra_notes=notes_text,
        timeout_seconds=ai_engine.SYNTHESIS_TIMEOUT_SECONDS,
    )
    used_model = getattr(synthesis, "_model_used", "") or ai_engine.load_api_environment()[1]

    cfg_course = COURSE_CATALOG_CONFIG.get((course_code or "").strip().upper())
    title_clean = (lecture_title or "").strip()
    is_default_course_topic = bool(cfg_course and title_clean == cfg_course["topic"])
    detected = (getattr(synthesis, "detected_topic", "") or "").strip()
    uploaded_names_low = " ".join(
        fname for fname, _, _ in (audio_payload + image_payload + pdf_payload)
    ).lower()
    code_low = (course_code or "").strip().lower()

    if detected and (not title_clean or title_clean == "Untitled Lecture" or (is_default_course_topic and (not code_low or code_low not in uploaded_names_low))):
        resolved_title = detected
    elif title_clean and title_clean != "Untitled Lecture":
        resolved_title = title_clean
    else:
        all_names = [fname for fname, _, _ in (audio_payload + image_payload + pdf_payload)]
        resolved_title = (
            Path(all_names[0]).stem.replace("_", " ").replace("-", " ").title()
            if all_names
            else "Untitled Lecture"
        )

    summary_dict = {
        "course_label": course_label or timetable.format_course_label(course_code),
        "detected_topic": detected or resolved_title,
        "uploaded_labels": list(mod_info["labels"]),
        "generation_source": "local_fallback" if syn_fallback else "gemini_live",
        "generation_model": "" if syn_fallback else used_model,
        "executive_summary": synthesis.executive_summary,
        "structured_notes": [s.model_dump() for s in synthesis.structured_notes],
        "extracted_pseudocode": synthesis.extracted_pseudocode,
    }
    chunk_dicts = [c.model_dump() for c in synthesis.knowledge_chunks]
    new_id = database.save_lecture(
        course_code=course_code or "GENERAL",
        lecture_title=resolved_title,
        summary_data=summary_dict,
        chunks=chunk_dicts,
    )
    _cached_list_lectures_for.clear()
    stored_chunks = database.get_lecture_chunks(new_id)
    if stored_chunks:
        try:
            if status_cb:
                status_cb("Generating 10 MCQs and 6 flashcards from your uploaded lecture...")
            pack, _rev_fb = ai_engine.generate_revision_pack_with_timeout(
                lecture_title=f"{course_label or course_code} {resolved_title}",
                chunks=stored_chunks,
                use_cache=False,
                timeout_seconds=ai_engine.SYNTHESIS_TIMEOUT_SECONDS,
            )
        except Exception:
            pack = ai_engine.build_local_revision_pack(
                lecture_title=f"{course_label or course_code} {resolved_title}",
                chunks=stored_chunks,
            )
        database.save_revision_items(new_id, "mcq", [m.model_dump() for m in pack.mcqs])
        database.save_revision_items(new_id, "flashcard", [f.model_dump() for f in pack.flashcards])
    return new_id, int(mod_info["inputs_active"]), syn_fallback, ("" if syn_fallback else used_model)


def _get_quick_questions(lecture: dict | None, chunks: list[dict]) -> list[str]:
    """Return 3 course-relevant quick-question prompts for the active lecture."""
    if not lecture:
        return list(COURSE_CATALOG_CONFIG["CSC322"]["quick_questions"])
    code = lecture["course_code"].strip().upper()
    cfg = COURSE_CATALOG_CONFIG.get(code)
    title = lecture.get("lecture_title", "").strip()
    if cfg and (title == cfg["topic"] or cfg["short_topic"].lower() in title.lower()):
        return list(cfg["quick_questions"])
    topics = [str(c.get("topic") or "").strip() for c in chunks[:3] if str(c.get("topic") or "").strip()]
    prompts = [f"Explain {t} from this lecture." for t in topics]
    while len(prompts) < 3:
        prompts.append("Summarize the key rules and worked examples covered.")
    return prompts[:3]


def _submit_library_question(prompt_text: str, lecture_id: int, history: list[dict]) -> None:
    """Run grounded retrieval + answer synthesis and append turn to session chat history."""
    history.append({"role": "user", "content": prompt_text})
    matched = database.retrieve_relevant_chunks(
        query=prompt_text,
        lecture_id=lecture_id,
        top_k=4,
    )
    if not matched:
        history.append(
            {
                "role": "assistant",
                "content": "No matching notes were found in this lecture for that question.",
                "badges": [],
            }
        )
        return

    with st.spinner("Searching lecture notes..."):
        rag_reply, _ = ai_engine.answer_grounded_question_with_timeout(
            question=prompt_text,
            retrieved_chunks=matched,
            timeout_seconds=ai_engine.SYNTHESIS_TIMEOUT_SECONDS,
        )
    history.append(
        {
            "role": "assistant",
            "content": rag_reply.answer,
            "badges": rag_reply.cited_badges,
        }
    )


# ---------------------------------------------------------------------------
# Load State & Synchronize Shared Reactive Selections Before Instantiating Widgets
# ---------------------------------------------------------------------------
all_lectures = _cached_list_lectures()
saved_slots = _cached_list_timetable_slots()
schedule_df = timetable.slots_list_to_df(saved_slots)
student_saved_slots = database.load_student_schedule()
status_schedule_df = timetable.slots_list_to_df(student_saved_slots) if student_saved_slots else schedule_df
clashes_df = timetable.detect_clashes(status_schedule_df)
lec_map = _build_practice_lec_map(all_lectures)

if "pending_sync" in st.session_state:
    sync_req = st.session_state.pop("pending_sync")
    _apply_course_sync(
        course_code=sync_req["course_code"],
        lecture_id=sync_req.get("lecture_id"),
        lectures=all_lectures,
    )

default_lec = all_lectures[0] if all_lectures else None
if "active_course_code" not in st.session_state:
    default_code = default_lec["course_code"] if default_lec else "CSC322"
    _apply_course_sync(default_code, default_lec["id"] if default_lec else None, all_lectures)

valid_lec_ids = {l["id"] for l in all_lectures}
if (
    "active_lecture_id" not in st.session_state
    or st.session_state["active_lecture_id"] not in valid_lec_ids
):
    if default_lec:
        st.session_state["active_lecture_id"] = default_lec["id"]

if "cap_course_choice" not in st.session_state or st.session_state["cap_course_choice"] not in COURSE_OPTIONS:
    st.session_state["cap_course_choice"] = COURSE_OPTIONS[0]
if "cap_title" not in st.session_state:
    init_code = st.session_state.get("active_course_code", "CSC322")
    st.session_state["cap_title"] = COURSE_CATALOG_CONFIG.get(init_code, {}).get("topic", "")
if "library_selected_id" not in st.session_state or st.session_state["library_selected_id"] not in valid_lec_ids:
    if default_lec:
        st.session_state["library_selected_id"] = default_lec["id"]
if lec_map and (
    "practice_lecture_select" not in st.session_state
    or st.session_state["practice_lecture_select"] not in lec_map
):
    st.session_state["practice_lecture_select"] = _find_practice_label_for_id(
        lec_map, st.session_state.get("active_lecture_id")
    )

# Resolve Top Status Bar Values Dynamically From Active Session State
active_code_state = st.session_state.get("active_course_code", "CSC322")
active_lec = next((l for l in all_lectures if l["id"] == st.session_state.get("active_lecture_id")), None)
if active_code_state == CUSTOM_COURSE_OPTION:
    raw_custom_val = (st.session_state.get("cap_custom_course") or "").strip()
    _, custom_lbl_val = _parse_custom_course_input(raw_custom_val)
    active_course_str = custom_lbl_val if raw_custom_val else "Custom Subject"
    active_topic_str = (st.session_state.get("cap_title") or "").strip() or "New Lecture Capture"
else:
    if active_lec and active_lec["course_code"].strip().upper() == active_code_state.strip().upper():
        active_course_str = _display_course_label(active_lec)
        active_topic_str = _short_topic_label(active_lec["lecture_title"], active_lec["course_code"])
    else:
        cfg_top = COURSE_CATALOG_CONFIG.get(active_code_state)
        active_course_str = cfg_top["label"] if cfg_top else timetable.format_course_label("CSC322")
        active_topic_str = cfg_top["short_topic"] if cfg_top else "FCFS CPU Scheduling"

clash_count = len(clashes_df)
if clash_count == 0:
    alert_text = "All Clear"
    alert_color = "#16a34a"
elif clash_count == 1:
    alert_text = "1 Time Clash"
    alert_color = "#d97706"
else:
    alert_text = f"{clash_count} Time Clashes"
    alert_color = "#d97706"

_render_html(
    """
    <div class="cf-topbar">
        <span class="cf-brand">CampusFlow</span>
        <span class="cf-version">v1.0</span>
    </div>
    """
)

sb_c1, sb_c2, sb_c3, sb_c4 = st.columns(4, gap="medium")
_render_html(
    """
    <div class="cf-stat-card">
        <div class="cf-stat-label">Semester</div>
        <div class="cf-stat-value">Fall 2026</div>
    </div>
    """,
    target=sb_c1,
)
_render_html(
    f"""
    <div class="cf-stat-card">
        <div class="cf-stat-label">Active Course</div>
        <div class="cf-stat-value">{html.escape(active_course_str)}</div>
    </div>
    """,
    target=sb_c2,
)
_render_html(
    f"""
    <div class="cf-stat-card">
        <div class="cf-stat-label">Current Topic</div>
        <div class="cf-stat-value">{html.escape(active_topic_str)}</div>
    </div>
    """,
    target=sb_c3,
)
_render_html(
    f"""
    <div class="cf-stat-card">
        <div class="cf-stat-label">Schedule Alerts</div>
        <div class="cf-stat-value" style="color:{alert_color};">{html.escape(alert_text)}</div>
    </div>
    """,
    target=sb_c4,
)

tab_capture, tab_library, tab_practice, tab_schedule = st.tabs(
    ["Capture", "Vault & Q&A", "Quiz & Flashcards", "Timetable & Clash"]
)


# ===========================================================================
# TAB 1: 1. ScribeAI Studio (Lecture Capture)
# ===========================================================================
with tab_capture:
    cap_left, cap_right = st.columns([3, 2], gap="large")

    with cap_left:
        with st.container(border=True):
            _render_html(
                """
                <div data-cf-studio="true" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.45rem;">
                    <span style="color:#0f172a;font-size:0.98rem;font-weight:700;">Capture Studio</span>
                    <span class="cf-source-pill">1 - 3 Multimodal Inputs</span>
                </div>
                """
            )
            meta_c1, meta_c2 = st.columns([1.15, 1.35])
            with meta_c1:
                selected_course_opt = st.selectbox(
                    "Course",
                    options=COURSE_OPTIONS,
                    key="cap_course_choice",
                    on_change=_on_cap_course_change,
                )
                if selected_course_opt == CUSTOM_COURSE_OPTION:
                    custom_raw_course = st.text_input(
                        "Custom Course Code & Name",
                        placeholder="e.g. CSC336 — Web Technologies",
                        key="cap_custom_course",
                    )
                    resolved_course, resolved_course_label = _parse_custom_course_input(custom_raw_course)
                    active_kit_cfg = None
                else:
                    resolved_course = timetable.parse_course_code(selected_course_opt)
                    active_kit_cfg = COURSE_CATALOG_CONFIG.get(resolved_course)
                    resolved_course_label = active_kit_cfg["label"] if active_kit_cfg else resolved_course
                    custom_raw_course = ""
            with meta_c2:
                topic_placeholder = (
                    active_kit_cfg["topic"] if active_kit_cfg else "Enter lecture topic (any discipline)"
                )
                lecture_title_input = st.text_input(
                    "Lecture Topic",
                    placeholder=topic_placeholder,
                    key="cap_title",
                )

            up_c1, up_c2, up_c3 = st.columns(3, gap="small")
            with up_c1:
                _render_html(
                    """
                    <div class="cf-uploader-head">
                        <span class="cf-uploader-title">Lecture Audio</span>
                        <span class="cf-ext-badge">MP3 / WAV</span>
                    </div>
                    """
                )
                audio_files = st.file_uploader(
                    "Lecture Audio",
                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                    accept_multiple_files=True,
                    key="cap_audio_files",
                    label_visibility="collapsed",
                )
            with up_c2:
                _render_html(
                    """
                    <div class="cf-uploader-head">
                        <span class="cf-uploader-title">Whiteboard Photo</span>
                        <span class="cf-ext-badge">PNG / JPG</span>
                    </div>
                    """
                )
                image_files = st.file_uploader(
                    "Whiteboard Photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True,
                    key="cap_image_files",
                    label_visibility="collapsed",
                )
            with up_c3:
                _render_html(
                    """
                    <div class="cf-uploader-head">
                        <span class="cf-uploader-title">Slides PDF</span>
                        <span class="cf-ext-badge">PDF</span>
                    </div>
                    """
                )
                pdf_files = st.file_uploader(
                    "Slides PDF",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="cap_pdf_files",
                    label_visibility="collapsed",
                )

            _render_html("<div style='height:0.35rem;'></div>")
            if active_kit_cfg:
                sample_btn_label = f"Load {active_kit_cfg['code']} Sample Kit"
                act_c1, act_c2 = st.columns(2, gap="small")
                process_clicked = act_c1.button("Process Lecture", type="primary", width="stretch")
                sample_clicked = act_c2.button(sample_btn_label, width="stretch")
            else:
                sample_btn_label = ""
                process_clicked = st.button("Process Lecture", type="primary", width="stretch")
                sample_clicked = False

        if "cap_action_notice" in st.session_state:
            st.success(st.session_state.pop("cap_action_notice"))

        if sample_clicked and active_kit_cfg:
            new_id = _load_canonical_sample_kit(active_kit_cfg["code"])
            st.session_state.pop(f"uploaded_images_{new_id}", None)
            st.session_state.setdefault("cap_inputs_active_by_lec", {})[new_id] = 3
            st.session_state["pending_sync"] = {
                "course_code": active_kit_cfg["code"],
                "lecture_id": new_id,
            }
            st.session_state["cap_action_notice"] = (
                f"⚡ Loaded {active_kit_cfg['code']} canonical sample kit (10 MCQs, 6 flashcards)."
            )
            st.rerun()

        if process_clicked:
            if not (audio_files or image_files or pdf_files):
                st.warning("Upload at least one file (Lecture Audio, Whiteboard Photo, or Slides PDF) first.")
            else:
                final_code = (resolved_course or "GENERAL").strip().upper()
                final_course_label = resolved_course_label or final_code
                final_title = (lecture_title_input or "Untitled Lecture").strip()
                audio_payload = [
                    (f.name, f.getvalue(), ai_engine.infer_mime_type(f.name, "audio/mpeg"))
                    for f in (audio_files or [])
                ]
                image_payload = [
                    (f.name, f.getvalue(), ai_engine.infer_mime_type(f.name, "image/png"))
                    for f in (image_files or [])
                ]
                pdf_payload = [
                    (f.name, f.getvalue(), "application/pdf")
                    for f in (pdf_files or [])
                ]
                mod_info = ai_engine.describe_provided_modalities(
                    audio_files=audio_payload,
                    image_files=image_payload,
                    pdf_files=pdf_payload,
                    extra_notes="",
                )
                active_label_text = ", ".join(mod_info["labels"]) or "Lecture Inputs"

                with st.status(
                    f"Processing {final_course_label} ({active_label_text})...",
                    expanded=True,
                ) as _proc_status:
                    st.write(f"Preparing {active_label_text}...")
                    try:
                        new_id, inputs_active_count, used_fallback, used_model = _run_synthesis_and_save(
                            course_code=final_code,
                            course_label=final_course_label,
                            lecture_title=final_title,
                            audio_payload=audio_payload,
                            image_payload=image_payload,
                            pdf_payload=pdf_payload,
                            notes_text="",
                            status_cb=st.write,
                        )
                        if image_payload:
                            st.session_state[f"uploaded_images_{new_id}"] = [
                                (fname, raw_bytes) for fname, raw_bytes, _ in image_payload
                            ]
                        else:
                            st.session_state.pop(f"uploaded_images_{new_id}", None)
                        mode_note = (
                            " (Offline Local Fallback)"
                            if used_fallback
                            else f" (✨ Generated Live via Gemini Flash API: {used_model})"
                        )
                        _proc_status.update(
                            label=f"{final_code} lecture saved from {active_label_text}{mode_note}.",
                            state="complete",
                            expanded=False,
                        )
                        st.session_state.setdefault("cap_inputs_active_by_lec", {})[new_id] = inputs_active_count
                        st.session_state["pending_sync"] = {
                            "course_code": final_code,
                            "lecture_id": new_id,
                        }
                        st.session_state["cap_action_notice"] = (
                            f"Processed {final_course_label} from {active_label_text}{mode_note} — view structured NOTES in the Vault & Q&A tab."
                        )
                        st.rerun()
                    except Exception as exc:
                        _proc_status.update(label="Processing failed.", state="error", expanded=False)
                        st.error(f"Could not process lecture: {exc}")

    with cap_right:
        _render_html('<div data-cf-right-stack="true"></div>')
        # 1. Dynamic Multimodal Capture Kit & Pipeline Status Card
        audio_cnt = len(audio_files or [])
        image_cnt = len(image_files or [])
        pdf_cnt = len(pdf_files or [])
        if active_kit_cfg:
            audio_size = _format_file_kb(DEMO_DIR / active_kit_cfg["audio_file"])
            board_size = _format_file_kb(DEMO_DIR / active_kit_cfg["image_file"])
            slides_size = _format_file_kb(DEMO_DIR / active_kit_cfg["pdf_file"])
            up_audio_status = f"{audio_cnt} staged" if audio_cnt else audio_size
            up_img_status = f"{image_cnt} staged" if image_cnt else board_size
            up_pdf_status = f"{pdf_cnt} staged" if pdf_cnt else slides_size
            audio_cls = "cf-pill-active" if audio_cnt else "cf-source-pill"
            board_cls = "cf-pill-active" if image_cnt else "cf-source-pill"
            slides_cls = "cf-pill-active" if pdf_cnt else "cf-source-pill"
            audio_pill = f"Audio ({audio_cnt} staged)" if audio_cnt else "Audio Ready"
            board_pill = f"Board Vision ({image_cnt} staged)" if image_cnt else "Board Vision Ready"
            slides_pill = f"Slides Parser ({pdf_cnt} staged)" if pdf_cnt else "Slides Parser Ready"

            _render_html(
                f"""
                <div class="cf-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                        <span style="color:#0f172a;font-size:0.98rem;font-weight:700;">Multimodal Capture Kit</span>
                        <span class="cf-source-pill">{html.escape(active_kit_cfg['code'])}</span>
                    </div>
                    <div class="cf-muted" style="margin-bottom:0.55rem;">
                        {html.escape(active_kit_cfg['subject'])} · Ready in <code style="color:#0369a1;background:#f0f9ff;border:1px solid #bae6fd;padding:1px 5px;border-radius:4px;">demo_files/</code> or click <strong style="color:#0f172a;">{html.escape(sample_btn_label)}</strong>:
                    </div>
                    <div class="cf-demo-row">
                        <span style="color:#475569;"><strong style="color:#0f172a;">{html.escape(active_kit_cfg['audio_file'])}</strong> — {html.escape(active_kit_cfg['audio_desc'])}</span>
                        <span class="{audio_cls}">{html.escape(up_audio_status)}</span>
                    </div>
                    <div class="cf-demo-row">
                        <span style="color:#475569;"><strong style="color:#0f172a;">{html.escape(active_kit_cfg['image_file'])}</strong> — {html.escape(active_kit_cfg['image_desc'])}</span>
                        <span class="{board_cls}">{html.escape(up_img_status)}</span>
                    </div>
                    <div class="cf-demo-row">
                        <span style="color:#475569;"><strong style="color:#0f172a;">{html.escape(active_kit_cfg['pdf_file'])}</strong> — {html.escape(active_kit_cfg['pdf_desc'])}</span>
                        <span class="{slides_cls}">{html.escape(up_pdf_status)}</span>
                    </div>
                    <div style="margin-top:0.65rem;">
                        <span class="{audio_cls}">{html.escape(audio_pill)}</span>
                        <span class="{board_cls}">{html.escape(board_pill)}</span>
                        <span class="{slides_cls}">{html.escape(slides_pill)}</span>
                    </div>
                </div>
                """
            )
        else:
            staged_count = audio_cnt + image_cnt + pdf_cnt
            audio_cls = "cf-pill-active" if audio_cnt else "cf-source-pill"
            board_cls = "cf-pill-active" if image_cnt else "cf-source-pill"
            slides_cls = "cf-pill-active" if pdf_cnt else "cf-source-pill"
            audio_pill = f"Audio ({audio_cnt} staged)" if audio_cnt else "Audio Ready"
            board_pill = f"Board Vision ({image_cnt} staged)" if image_cnt else "Board Vision Ready"
            slides_pill = f"Slides Parser ({pdf_cnt} staged)" if pdf_cnt else "Slides Parser Ready"
            custom_disp_preview = resolved_course_label if custom_raw_course.strip() else "any university subject"
            _render_html(
                f"""
                <div class="cf-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                        <span style="color:#0f172a;font-size:0.98rem;font-weight:700;">Custom Subject Live Pipeline</span>
                        <span class="cf-source-pill">{html.escape(resolved_course if custom_raw_course.strip() else 'ANY SUBJECT')}</span>
                    </div>
                    <div class="cf-muted" style="margin-bottom:0.6rem;">
                        Works for <strong style="color:#0f172a;">{html.escape(custom_disp_preview)}</strong> or any university discipline. Stage 1, 2, or 3 files and click <strong style="color:#0f172a;">Process Lecture</strong> ({staged_count} file(s) staged).
                    </div>
                    <div>
                        <span class="{audio_cls}">{html.escape(audio_pill)}</span>
                        <span class="{board_cls}">{html.escape(board_pill)}</span>
                        <span class="{slides_cls}">{html.escape(slides_pill)}</span>
                    </div>
                </div>
                """
            )

        # 2. Capture Summary Card
        cap_lec = active_lec
        if cap_lec:
            cap_chunks = database.get_lecture_chunks(cap_lec["id"])
            cap_mcqs = database.get_revision_items(cap_lec["id"], "mcq")
            cap_cards = database.get_revision_items(cap_lec["id"], "flashcard")

            present_file_mods = {
                c.get("source_type")
                for c in cap_chunks
                if c.get("source_type") in {"AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF"}
            }
            stored_inputs_active = st.session_state.get("cap_inputs_active_by_lec", {}).get(cap_lec["id"])
            if stored_inputs_active is not None:
                inputs_active_count = int(stored_inputs_active)
            elif present_file_mods:
                inputs_active_count = len(present_file_mods)
            elif cap_chunks:
                inputs_active_count = 1
            else:
                inputs_active_count = 0

            cap_prov_label, cap_prov_cls = _provenance_badge_info(cap_lec.get("summary"), len(cap_chunks))
            cap_up_badge = _uploaded_modality_badge(cap_lec.get("summary"))
            cap_up_html = (
                f'<div style="margin-top:0.35rem;"><span class="cf-pill-active">{html.escape(cap_up_badge)}</span></div>'
                if cap_up_badge
                else ""
            )
            cap_course_disp = _display_course_label(cap_lec)
            _render_html(
                f"""
                <div class="cf-card" style="margin-bottom:0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.35rem;">
                        <span style="color:#0f172a;font-size:0.98rem;font-weight:700;">Capture Summary</span>
                        <span class="{cap_prov_cls}">{html.escape(cap_prov_label)}</span>
                    </div>
                    <div class="cf-muted">
                        {html.escape(cap_course_disp)} · {html.escape(cap_lec['lecture_title'])}
                    </div>
                    {cap_up_html}
                    <div class="cf-mini-grid">
                        <div class="cf-mini-stat">
                            <div class="cf-mini-num">{len(cap_chunks)}</div>
                            <div class="cf-mini-cap">Note Chunks</div>
                        </div>
                        <div class="cf-mini-stat">
                            <div class="cf-mini-num">{len(cap_mcqs)}</div>
                            <div class="cf-mini-cap">Quiz MCQs</div>
                        </div>
                        <div class="cf-mini-stat">
                            <div class="cf-mini-num">{len(cap_cards)}</div>
                            <div class="cf-mini-cap">Flashcards</div>
                        </div>
                        <div class="cf-mini-stat">
                            <div class="cf-mini-num">{inputs_active_count} / 3</div>
                            <div class="cf-mini-cap">Inputs Active</div>
                        </div>
                    </div>
                </div>
                """
            )


# ===========================================================================
# TAB 2: Vault & Q&A
# ===========================================================================
with tab_library:
    if not all_lectures:
        st.info("No lectures saved yet. Process one in the Capture tab first.")
    else:
        resolved_lec = active_lec
        if resolved_lec is None and active_code_state != CUSTOM_COURSE_OPTION:
            resolved_lec = _latest_lecture_for_course(all_lectures, active_code_state)
        if resolved_lec is None:
            resolved_lec = next(
                (l for l in all_lectures if l["id"] == st.session_state.get("library_selected_id")),
                all_lectures[0],
            )
        selected_lib_id = int(resolved_lec["id"])
        st.session_state["library_selected_id"] = selected_lib_id
        selected_lec = database.get_lecture(selected_lib_id) or resolved_lec
        selected_chunks = database.get_lecture_chunks(selected_lib_id)

        lib_col, chat_col = st.columns([1.65, 1], gap="large")

        with lib_col:
            _render_lecture_notes(selected_lec, selected_chunks, key_prefix="lib")

        with chat_col:
            with st.container(border=True):
                course_pill_text = _display_course_label(selected_lec)
                topic_sub = _short_topic_label(selected_lec["lecture_title"], selected_lec["course_code"])
                _render_html(
                    f"""
                    <div data-cf-qa-panel="true" style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:0.15rem;">
                        <span style="color:#0f172a;font-size:0.98rem;font-weight:700;">Ask About This Lecture</span>
                        <span class="cf-source-pill">{html.escape(selected_lec['course_code'])}</span>
                    </div>
                    <div class="cf-muted" style="margin-bottom:0.35rem;">{html.escape(course_pill_text)} · {html.escape(topic_sub)}</div>
                    """
                )

                if "vault_chat" not in st.session_state:
                    st.session_state["vault_chat"] = {}
                history: list[dict] = st.session_state["vault_chat"].setdefault(selected_lib_id, [])

                _render_html(
                    '<div style="color:#0284c7;font-size:0.72rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin:0.15rem 0 0.25rem 0;">Quick Questions</div>'
                )
                quick_prompts = _get_quick_questions(selected_lec, selected_chunks)
                for q_i, quick_q in enumerate(quick_prompts):
                    if st.button(quick_q, key=f"lib_quick_{selected_lib_id}_{q_i}", width="stretch"):
                        _submit_library_question(quick_q, selected_lib_id, history)
                        st.rerun()

                if history:
                    with st.container(border=True):
                        _render_html('<div data-cf-chat-scroll="true"></div>')
                        for msg in history:
                            with st.chat_message(msg["role"]):
                                st.markdown(msg["content"])
                                badges = msg.get("badges") or []
                                if badges:
                                    pills = " ".join(
                                        f'<span class="cf-source-pill">{html.escape(b)}</span>'
                                        for b in badges
                                    )
                                    _render_html(pills)
                else:
                    _render_html(
                        '<div class="cf-muted" style="margin:0.25rem 0 0.2rem 0;font-size:0.81rem;">Click a quick question above or ask anything below.</div>'
                    )

                user_prompt = st.chat_input(
                    "Ask a question about these notes...",
                    key=f"lib_chat_input_{selected_lib_id}",
                )
                if user_prompt:
                    _submit_library_question(user_prompt, selected_lib_id, history)
                    st.rerun()

                if history and st.button("Clear Conversation", key=f"clear_chat_{selected_lib_id}"):
                    st.session_state["vault_chat"][selected_lib_id] = []
                    st.rerun()


# ===========================================================================
# TAB 3: 3. Revision Intelligence (Quiz & Cards)
# ===========================================================================
with tab_practice:
    if not all_lectures:
        st.info("No lectures available yet. Capture a lecture first.")
    else:
        p_col1, p_col2, p_col3 = st.columns([2.1, 1.5, 1.4], gap="medium")
        chosen_practice_label = p_col1.selectbox(
            "Select Lecture",
            options=list(lec_map.keys()),
            key="practice_lecture_select",
            on_change=_on_practice_lec_change,
        )
        practice_lec_id = lec_map[chosen_practice_label]
        practice_lec = database.get_lecture(practice_lec_id)
        practice_chunks = database.get_lecture_chunks(practice_lec_id)

        _render_html("<div style='height:1.72rem;'></div>", target=p_col2)
        practice_mode = (
            p_col2.segmented_control(
                "Mode",
                options=["Interactive Quiz (10 Qs)", "Flashcard Deck"],
                default="Interactive Quiz (10 Qs)",
                key="practice_mode_control",
                label_visibility="collapsed",
                width="stretch",
            )
            or "Interactive Quiz (10 Qs)"
        )

        _render_html("<div style='height:1.72rem;'></div>", target=p_col3)
        if p_col3.button("Generate Fresh 10-Question Set", type="primary", width="stretch"):
            if not ai_engine.is_api_configured():
                st.error("API key missing in .env file.")
            else:
                with st.spinner("Generating 10 MCQs and 6 flashcards..."):
                    try:
                        pack = ai_engine.generate_revision_pack(
                            lecture_title=(
                                f"{_display_course_label(practice_lec)} "
                                f"{practice_lec['lecture_title']}"
                            ),
                            chunks=practice_chunks,
                        )
                        database.save_revision_items(
                            practice_lec_id, "mcq", [m.model_dump() for m in pack.mcqs]
                        )
                        database.save_revision_items(
                            practice_lec_id, "flashcard", [f.model_dump() for f in pack.flashcards]
                        )
                        st.session_state[f"quiz_idx_{practice_lec_id}"] = 0
                        st.session_state[f"quiz_submitted_{practice_lec_id}"] = False
                        st.session_state[f"quiz_round_{practice_lec_id}"] = (
                            st.session_state.get(f"quiz_round_{practice_lec_id}", 0) + 1
                        )
                        if "quiz_answers_by_lec" in st.session_state:
                            st.session_state["quiz_answers_by_lec"].pop(practice_lec_id, None)
                        st.session_state["quiz_answers"] = {}
                        st.session_state[f"fc_idx_{practice_lec_id}"] = 0
                        st.session_state[f"fc_flip_{practice_lec_id}"] = False
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate revision set: {exc}")

        st.markdown(f"#### {chosen_practice_label}")
        mcqs = database.get_revision_items(practice_lec_id, "mcq")
        cards = database.get_revision_items(practice_lec_id, "flashcard")

        if practice_mode == "Interactive Quiz (10 Qs)":
            if not mcqs:
                st.caption("Click Generate Fresh 10-Question Set to create questions.")
            else:
                if "quiz_answers_by_lec" not in st.session_state:
                    st.session_state["quiz_answers_by_lec"] = {}
                lec_answers: dict[int, int] = st.session_state["quiz_answers_by_lec"].setdefault(
                    practice_lec_id, {}
                )
                st.session_state["quiz_answers"] = lec_answers

                idx_state_key = f"quiz_idx_{practice_lec_id}"
                sub_state_key = f"quiz_submitted_{practice_lec_id}"
                round_state_key = f"quiz_round_{practice_lec_id}"

                if idx_state_key not in st.session_state:
                    st.session_state[idx_state_key] = 0
                if sub_state_key not in st.session_state:
                    st.session_state[sub_state_key] = False
                if round_state_key not in st.session_state:
                    st.session_state[round_state_key] = 0

                total_q = len(mcqs)
                is_submitted = bool(st.session_state[sub_state_key])
                round_no = int(st.session_state[round_state_key])

                if not is_submitted:
                    cur_idx = max(0, min(int(st.session_state[idx_state_key]), total_q - 1))
                    item = mcqs[cur_idx]
                    q = item["payload"]
                    answered_count = len(lec_answers)

                    with st.container(border=True):
                        _render_html(
                            f"""
                            <div class="cf-quiz-header">
                                <span class="cf-quiz-step">Question {cur_idx + 1} of {total_q}</span>
                                <span class="cf-muted">Answered: {answered_count} / {total_q}</span>
                            </div>
                            """
                        )
                        st.progress((cur_idx + 1) / total_q)
                        _render_html(
                            f'<div class="cf-quiz-question">{html.escape(q["question"])}</div>'
                        )

                        saved_choice = lec_answers.get(cur_idx)
                        valid_saved = (
                            saved_choice
                            if (saved_choice is not None and 0 <= saved_choice < len(q["options"]))
                            else None
                        )
                        chosen_opt = st.radio(
                            f"Question {cur_idx + 1} Options",
                            options=list(range(len(q["options"]))),
                            index=valid_saved,
                            format_func=lambda opt_i, opts=q["options"]: opts[opt_i],
                            key=f"quiz_radio_{practice_lec_id}_{cur_idx}_{round_no}",
                            label_visibility="collapsed",
                        )
                        if chosen_opt is not None:
                            lec_answers[cur_idx] = chosen_opt

                        _render_html("<div style='height:0.65rem;'></div>")
                        nav_left, _, nav_right = st.columns([1.2, 2.6, 1.4])
                        if cur_idx > 0:
                            if nav_left.button(
                                "Previous",
                                key=f"quiz_prev_{practice_lec_id}_{cur_idx}",
                                width="stretch",
                            ):
                                st.session_state[idx_state_key] = cur_idx - 1
                                st.rerun()

                        if cur_idx < total_q - 1:
                            if nav_right.button(
                                "Next Question",
                                key=f"quiz_next_{practice_lec_id}_{cur_idx}",
                                type="primary",
                                width="stretch",
                            ):
                                st.session_state[idx_state_key] = cur_idx + 1
                                st.rerun()
                        else:
                            if nav_right.button(
                                "Finish & Evaluate Quiz",
                                key=f"quiz_finish_{practice_lec_id}",
                                type="primary",
                                width="stretch",
                            ):
                                st.session_state[sub_state_key] = True
                                st.rerun()
                else:
                    correct_count = sum(
                        1
                        for q_i, item in enumerate(mcqs)
                        if lec_answers.get(q_i) == item["payload"]["correct_index"]
                    )
                    accuracy = round((correct_count / total_q) * 100) if total_q else 0
                    passed = accuracy >= 60
                    status_label = "Passed" if passed else "Needs Review"
                    status_color = "#16a34a" if passed else "#d97706"

                    st.markdown("#### Quiz Results & Evaluation")
                    m1, m2, m3 = st.columns(3, gap="medium")
                    _render_html(
                        f"""
                        <div class="cf-stat-card">
                            <div class="cf-stat-label">Final Score</div>
                            <div class="cf-stat-value" style="font-size:1.35rem;">{correct_count} / {total_q}</div>
                        </div>
                        """,
                        target=m1,
                    )
                    _render_html(
                        f"""
                        <div class="cf-stat-card">
                            <div class="cf-stat-label">Accuracy</div>
                            <div class="cf-stat-value" style="font-size:1.35rem;">{accuracy}%</div>
                        </div>
                        """,
                        target=m2,
                    )
                    _render_html(
                        f"""
                        <div class="cf-stat-card">
                            <div class="cf-stat-label">Status</div>
                            <div class="cf-stat-value" style="font-size:1.35rem;color:{status_color};">{status_label}</div>
                        </div>
                        """,
                        target=m3,
                    )

                    eval_cols = st.columns(2, gap="medium")
                    for q_i, item in enumerate(mcqs):
                        q = item["payload"]
                        user_idx = lec_answers.get(q_i)
                        right_idx = int(q["correct_index"])
                        is_correct = user_idx == right_idx
                        if user_idx is None:
                            user_answer_str = "Not answered"
                            badge_html = '<span class="cf-badge-bad">Unanswered</span>'
                        elif is_correct:
                            user_answer_str = q["options"][user_idx]
                            badge_html = '<span class="cf-badge-ok">Correct</span>'
                        else:
                            user_answer_str = q["options"][user_idx]
                            badge_html = '<span class="cf-badge-bad">Incorrect</span>'

                        correct_str = q["options"][right_idx]
                        source_pill = html.escape(q.get("source_badge", ""))

                        with eval_cols[q_i % 2]:
                            _render_html(
                                f"""
                                <div class="cf-eval-card">
                                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.45rem;">
                                        <span class="cf-quiz-step">Question {q_i + 1}</span>
                                        {badge_html}
                                    </div>
                                    <div style="color:#0f172a;font-weight:600;margin-bottom:0.55rem;">{html.escape(q['question'])}</div>
                                    <div class="cf-muted" style="margin-bottom:0.25rem;"><strong>Your answer:</strong> {html.escape(user_answer_str)}</div>
                                    <div class="cf-muted" style="margin-bottom:0.45rem;"><strong>Correct answer:</strong> {html.escape(correct_str)}</div>
                                    <div class="cf-muted" style="margin-bottom:0.45rem;">{html.escape(q['explanation'])}</div>
                                    <div><span class="cf-source-pill">{source_pill}</span></div>
                                </div>
                                """
                            )

                    if st.button(
                        "Retake Quiz",
                        key=f"quiz_retake_{practice_lec_id}",
                        type="primary",
                    ):
                        st.session_state["quiz_answers_by_lec"][practice_lec_id] = {}
                        st.session_state["quiz_answers"] = {}
                        st.session_state[idx_state_key] = 0
                        st.session_state[sub_state_key] = False
                        st.session_state[round_state_key] = round_no + 1
                        st.rerun()

        else:
            if not cards:
                st.caption("Click Generate Fresh 10-Question Set to create flashcards.")
            else:
                fc_idx_key = f"fc_idx_{practice_lec_id}"
                fc_flip_key = f"fc_flip_{practice_lec_id}"
                if fc_idx_key not in st.session_state:
                    st.session_state[fc_idx_key] = 0
                if fc_flip_key not in st.session_state:
                    st.session_state[fc_flip_key] = False

                cur = st.session_state[fc_idx_key] % len(cards)
                card = cards[cur]["payload"]
                flipped = st.session_state[fc_flip_key]

                side_label = (
                    f"Card {cur + 1} of {len(cards)} · Answer"
                    if flipped
                    else f"Card {cur + 1} of {len(cards)} · Question"
                )
                body_text = card["back"] if flipped else card["front"]

                _render_html(
                    f"""
                    <div class="cf-flashcard">
                        <div class="cf-flashcard-side">{html.escape(side_label)}</div>
                        <div class="cf-flashcard-body">{html.escape(body_text)}</div>
                    </div>
                    """
                )
                if card.get("source_badge"):
                    st.caption(card["source_badge"])

                nav_1, nav_2, nav_3 = st.columns(3)
                if nav_1.button("Previous", key=f"fc_prev_{practice_lec_id}", width="stretch"):
                    st.session_state[fc_idx_key] = (cur - 1) % len(cards)
                    st.session_state[fc_flip_key] = False
                    st.rerun()
                if nav_2.button("Flip Card", key=f"fc_flip_btn_{practice_lec_id}", type="primary", width="stretch"):
                    st.session_state[fc_flip_key] = not flipped
                    st.rerun()
                if nav_3.button("Next", key=f"fc_next_{practice_lec_id}", width="stretch"):
                    st.session_state[fc_idx_key] = (cur + 1) % len(cards)
                    st.session_state[fc_flip_key] = False
                    st.rerun()


# ===========================================================================
# TAB 4: Full CS Master Timetable, Remaining Course Clash Solver & PDF Export
# ===========================================================================
with tab_schedule:
    if not st.session_state.get("t4_hydrated"):
        db_pdf_rows, db_pdf_name = database.load_uploaded_pdf_catalog()
        if db_pdf_rows:
            st.session_state["t4_pdf_rows"] = db_pdf_rows
            st.session_state["t4_pdf_name"] = db_pdf_name or "Uploaded_Timetable.pdf"
            st.session_state["t4_pdf_source_note"] = "Merged locally from saved Timetable PDF"
        saved_plan_meta = database.load_planner_state()
        if saved_plan_meta:
            reg_sec_saved = str(
                saved_plan_meta.get("regular_section")
                or (
                    f"{saved_plan_meta.get('program', '')}-{saved_plan_meta.get('section_code', '')}"
                    if saved_plan_meta.get("program") and saved_plan_meta.get("section_code")
                    else "BCS-5A"
                )
            )
            st.session_state["t4_regular_section"] = reg_sec_saved
            st.session_state["t4_prev_regular_section"] = reg_sec_saved
            if isinstance(saved_plan_meta.get("remaining_courses"), list):
                st.session_state["t4_remaining_courses"] = list(saved_plan_meta["remaining_courses"])
            if isinstance(saved_plan_meta.get("remaining_sections"), dict):
                st.session_state["t4_remaining_sections"] = {
                    str(k): str(v) for k, v in saved_plan_meta["remaining_sections"].items()
                }
            if isinstance(saved_plan_meta.get("regular_swaps"), dict):
                st.session_state["t4_regular_swaps"] = {
                    str(k): str(v) for k, v in saved_plan_meta["regular_swaps"].items()
                }
        st.session_state["t4_hydrated"] = True

    if "t4_pending_updates" in st.session_state:
        pending_u = st.session_state.pop("t4_pending_updates")
        for pk, pv in pending_u.items():
            st.session_state[pk] = pv

    # Build unified CS Department Master Timetable (built-in catalog + any uploaded PDF)
    builtin_master_df = timetable.get_cs_master_catalog_df()
    loaded_pdf_rows = list(st.session_state.get("t4_pdf_rows") or [])
    loaded_pdf_name = str(st.session_state.get("t4_pdf_name") or "")
    uploaded_pdf_df = (
        timetable.slots_list_to_df(loaded_pdf_rows).drop(columns=["id"], errors="ignore")
        if loaded_pdf_rows
        else builtin_master_df.iloc[0:0].copy()
    )
    master_catalog_df = timetable.merge_pdf_catalog_with_master(uploaded_pdf_df, builtin_master_df)
    all_sections_count = int(master_catalog_df["section"].nunique()) if not master_catalog_df.empty else 0
    all_courses_count = int(master_catalog_df["course_code"].nunique()) if not master_catalog_df.empty else 0

    catalog_pill = (
        f"CS Master Timetable + PDF ({loaded_pdf_name})"
        if loaded_pdf_name and loaded_pdf_rows
        else f"Built-in CS Master ({all_sections_count} Sections · {all_courses_count} Courses)"
    )
    _render_html(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.25rem;">
            <span style="color:#0f172a;font-size:1.25rem;font-weight:750;">CS Department Timetable Clash &amp; Repeater Solver</span>
            <span class="cf-pill-active">{html.escape(catalog_pill)}</span>
        </div>
        """
    )
    st.caption(
        "Select your regular class/section to load all regular courses automatically, pick any remaining or failed courses, "
        "inspect clash-free sections and smart swaps across all CS classes & RPT-CS, and download your final timetable PDF."
    )

    # Optional Full CS Timetable PDF Uploader (auto-merges into built-in catalog)
    with st.container(border=True):
        up_hdr_l, up_hdr_r = st.columns([2.6, 1.15], gap="medium")
        with up_hdr_l:
            _render_html(
                """
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
                    <span class="cf-source-pill">Optional PDF Merge</span>
                    <strong style="color:#0f172a;font-size:0.93rem;">Upload Full CS Timetable PDF</strong>
                </div>
                """
            )
            uploaded_pdf = st.file_uploader(
                "Upload Full CS Timetable PDF",
                type=["pdf"],
                key="t4_pdf_uploader",
                label_visibility="collapsed",
                help="Extracts all sections and courses from any uploaded CS timetable PDF and merges them into the master catalog automatically.",
            )
        with up_hdr_r:
            _render_html("<div style='height:1.55rem;'></div>")
            clear_pdf_clicked = st.button(
                "Reset to Built-in Master",
                key="t4_clear_pdf",
                width="stretch",
                disabled=not bool(loaded_pdf_rows),
            )

        if clear_pdf_clicked:
            database.clear_uploaded_pdf_catalog()
            st.session_state["t4_dismissed_pdf_sig"] = str(st.session_state.get("t4_pdf_sig") or "")
            for k_clear in ("t4_pdf_rows", "t4_pdf_name", "t4_pdf_sig", "t4_pdf_source_note"):
                st.session_state.pop(k_clear, None)
            st.rerun()

        if uploaded_pdf is None:
            st.session_state.pop("t4_dismissed_pdf_sig", None)
        else:
            raw_pdf_bytes = uploaded_pdf.getvalue()
            pdf_sig = f"{uploaded_pdf.name}:{len(raw_pdf_bytes)}"
            if (
                st.session_state.get("t4_pdf_sig") != pdf_sig
                and st.session_state.get("t4_dismissed_pdf_sig") != pdf_sig
            ):
                st.session_state.pop("t4_dismissed_pdf_sig", None)
                with st.spinner(f"Extracting sections & courses from '{uploaded_pdf.name}'..."):
                    parsed_rows, used_ai = _cached_extract_timetable_pdf(raw_pdf_bytes, uploaded_pdf.name)
                st.session_state["t4_pdf_sig"] = pdf_sig
                if parsed_rows:
                    st.session_state["t4_pdf_rows"] = parsed_rows
                    st.session_state["t4_pdf_name"] = uploaded_pdf.name
                    st.session_state["t4_pdf_source_note"] = (
                        "Merged via Gemini fallback" if used_ai else "Extracted locally via pure-Python PDF parser"
                    )
                    database.save_uploaded_pdf_catalog(parsed_rows, uploaded_pdf.name)
                    st.rerun()
                else:
                    st.warning(
                        f"Could not find tabular timetable rows in '{uploaded_pdf.name}'. Using built-in CS Master Timetable."
                    )

        if loaded_pdf_rows:
            pdf_sec_cnt = len({timetable.normalize_class_section(str(r.get("section") or "")) for r in loaded_pdf_rows})
            st.success(
                f"Merged {len(loaded_pdf_rows)} PDF slots across {pdf_sec_cnt} sections from '{loaded_pdf_name}' "
                f"into the CS Master Timetable ({st.session_state.get('t4_pdf_source_note', 'Cached')})."
            )

    # Refresh master catalog after possible PDF merge state
    loaded_pdf_rows = list(st.session_state.get("t4_pdf_rows") or [])
    loaded_pdf_name = str(st.session_state.get("t4_pdf_name") or "")
    uploaded_pdf_df = (
        timetable.slots_list_to_df(loaded_pdf_rows).drop(columns=["id"], errors="ignore")
        if loaded_pdf_rows
        else builtin_master_df.iloc[0:0].copy()
    )
    master_catalog_df = timetable.merge_pdf_catalog_with_master(uploaded_pdf_df, builtin_master_df)
    regular_section_options = timetable.get_regular_class_section_options(master_catalog_df)

    default_reg = "BCS-5A" if "BCS-5A" in regular_section_options else regular_section_options[0]
    if st.session_state.get("t4_regular_section") not in regular_section_options:
        st.session_state["t4_regular_section"] = default_reg

    # -----------------------------------------------------------------------
    # STEP 1: Select Your Regular Class / Section (auto-loads regular courses)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        cur_preview_sec = str(st.session_state.get("t4_regular_section") or default_reg)
        _render_html(
            f"""
            <div class="cf-step-head">
                <div class="cf-step-title-group">
                    <span class="cf-step-pill">Step 1</span>
                    <span class="cf-step-title">Select Your Regular Class / Section</span>
                </div>
                <span class="cf-pill-active">Regular Class: {html.escape(cur_preview_sec)}</span>
            </div>
            """
        )

        s1_l, s1_r = st.columns([1.6, 1], gap="medium")
        with s1_l:
            selected_regular_section = st.selectbox(
                "1. Select Your Regular Class / Section",
                options=regular_section_options,
                key="t4_regular_section",
                help="All regular courses for your selected class load automatically—no manual course picking needed.",
            )
        if st.session_state.get("t4_prev_regular_section") != selected_regular_section:
            st.session_state["t4_prev_regular_section"] = selected_regular_section
            st.session_state["t4_regular_swaps"] = {}
            st.session_state["t4_remaining_sections"] = {}

        regular_swaps_map: dict[str, str] = dict(st.session_state.get("t4_regular_swaps") or {})
        regular_courses = timetable.get_cohort_default_courses(master_catalog_df, selected_regular_section)
        regular_sections_map = {
            c: str(regular_swaps_map.get(c) or selected_regular_section)
            for c in regular_courses
        }

        norm_sec_col = master_catalog_df["section"].astype(str).apply(timetable.normalize_class_section)
        reg_frames: list[pd.DataFrame] = []
        for reg_c in regular_courses:
            assigned_sec = timetable.normalize_class_section(regular_sections_map[reg_c])
            rows_rc = master_catalog_df.loc[
                (master_catalog_df["course_code"].astype(str).str.upper() == reg_c)
                & (norm_sec_col == assigned_sec)
            ].copy()
            if not rows_rc.empty:
                reg_frames.append(rows_rc)
        base_regular_df = (
            pd.concat(reg_frames, ignore_index=True)
            if reg_frames
            else master_catalog_df.iloc[0:0].copy()
        )
        if not base_regular_df.empty:
            day_ord = {d: i for i, d in enumerate(timetable.COMSATS_ALL_DAYS)}
            base_regular_df["_d"] = base_regular_df["day"].map(day_ord).fillna(99)
            base_regular_df = (
                base_regular_df.sort_values(["_d", "start_time", "course_code"])
                .drop(columns=["_d"])
                .reset_index(drop=True)
            )

        with s1_r:
            _render_html("<div style='height:1.55rem;'></div>")
            if regular_swaps_map:
                if st.button("Reset Combined Swaps", key="t4_reset_reg_swaps", width="stretch"):
                    st.session_state["t4_regular_swaps"] = {}
                    st.rerun()
            else:
                st.metric("Auto-Loaded Regular Courses", len(regular_courses))

        if regular_courses:
            chips_html = " ".join(
                f'<span class="cf-source-pill">{html.escape(timetable.format_course_label(c, master_catalog_df))}'
                + (
                    f' [{html.escape(regular_sections_map[c])}]'
                    if regular_sections_map[c] != selected_regular_section
                    else ""
                )
                + "</span>"
                for c in regular_courses
            )
            _render_html(f'<div style="margin:0.25rem 0 0.6rem 0;display:flex;flex-wrap:wrap;gap:0.35rem;">{chips_html}</div>')
            st.markdown(f"##### Base Regular Weekly Timetable — {selected_regular_section}")
            _render_weekly_timetable(timetable.build_weekly_matrix(base_regular_df))
        else:
            st.warning(f"No regular courses found for {selected_regular_section} in the active catalog.")

    # -----------------------------------------------------------------------
    # STEP 2: Select Your Remaining / Repeater Courses
    # -----------------------------------------------------------------------
    all_master_codes = sorted(
        master_catalog_df["course_code"].dropna().astype(str).str.upper().unique().tolist()
    )
    reg_set = set(regular_courses)
    remaining_pool = [c for c in all_master_codes if c not in reg_set]
    existing_rem = [c for c in (st.session_state.get("t4_remaining_courses") or []) if c in remaining_pool]
    st.session_state["t4_remaining_courses"] = existing_rem

    with st.container(border=True):
        _render_html(
            """
            <div class="cf-step-head">
                <div class="cf-step-title-group">
                    <span class="cf-step-pill">Step 2</span>
                    <span class="cf-step-title">Select Your Remaining / Repeater Courses</span>
                </div>
                <span class="cf-muted" style="font-size:0.78rem;">Automatic prerequisite chain validation included</span>
            </div>
            """
        )
        selected_remaining = st.multiselect(
            "2. Select Your Remaining / Repeater Courses",
            options=remaining_pool,
            format_func=lambda c: timetable.format_course_label(c, master_catalog_df),
            key="t4_remaining_courses",
            help="All non-regular courses from the CS Master Timetable are listed here.",
        )

        # Clean out removed courses from chosen sections map
        rem_sec_map: dict[str, str] = {
            c: str(v)
            for c, v in (st.session_state.get("t4_remaining_sections") or {}).items()
            if c in selected_remaining
        }
        st.session_state["t4_remaining_sections"] = rem_sec_map

        prereq_warnings = timetable.check_failed_prereq_impacts_for_regular(
            regular_courses=regular_courses,
            failed_courses=selected_remaining,
            catalog_df=master_catalog_df,
        )
        if prereq_warnings:
            for pw in prereq_warnings:
                st.warning(f"**{pw['headline']}**  \n{pw['detail']}")
        elif selected_remaining:
            st.info(
                f"Prerequisite Check: None of your selected remaining courses ({', '.join(selected_remaining)}) "
                f"block any current regular course in {selected_regular_section}."
            )

    # -----------------------------------------------------------------------
    # STEP 3: Automatic Clash Check Across All Other Classes & Repeater Sections
    # -----------------------------------------------------------------------
    with st.container(border=True):
        _render_html(
            """
            <div class="cf-step-head">
                <div class="cf-step-title-group">
                    <span class="cf-step-pill">Step 3</span>
                    <span class="cf-step-title">Automatic Clash Check Across All Other Classes &amp; Repeater Sections</span>
                </div>
                <span class="cf-muted" style="font-size:0.78rem;">Compares Theory + Lab times (`start_a &lt; end_b and start_b &lt; end_a`)</span>
            </div>
            """
        )

        if not selected_remaining:
            st.info(
                "Select one or more remaining / failed courses in Step 2 above to check every CS class and RPT-CS repeater section automatically."
            )
        else:
            for rem_code in selected_remaining:
                # Baseline against which this course is checked = regular schedule + any OTHER already-chosen remaining courses
                other_rem_frames: list[pd.DataFrame] = []
                for other_c, other_s in rem_sec_map.items():
                    if other_c == rem_code:
                        continue
                    o_rows = master_catalog_df.loc[
                        (master_catalog_df["course_code"].astype(str).str.upper() == other_c)
                        & (norm_sec_col == timetable.normalize_class_section(other_s))
                    ].copy()
                    if not o_rows.empty:
                        other_rem_frames.append(o_rows)

                locked_others_df = (
                    pd.concat(other_rem_frames, ignore_index=True)
                    if other_rem_frames
                    else master_catalog_df.iloc[0:0].copy()
                )
                baseline_df = (
                    pd.concat([base_regular_df, locked_others_df], ignore_index=True)
                    if not locked_others_df.empty
                    else base_regular_df
                )

                section_checks = timetable.analyze_remaining_course_sections(
                    remaining_code=rem_code,
                    baseline_schedule_df=baseline_df,
                    master_catalog_df=master_catalog_df,
                )
                clash_free_secs = [s for s in section_checks if not s["has_clash"]]
                clashing_secs = [s for s in section_checks if s["has_clash"]]
                chosen_sec_for_rem = timetable.normalize_class_section(rem_sec_map.get(rem_code, ""))

                course_hdr_lbl = timetable.format_course_label(rem_code, master_catalog_df)
                status_tag = (
                    f'<span class="cf-badge-ok">✅ Assigned: {html.escape(chosen_sec_for_rem)}</span>'
                    if chosen_sec_for_rem
                    else f'<span class="cf-source-pill">{len(clash_free_secs)} clash-free / {len(clashing_secs)} clashing</span>'
                )
                with st.expander(
                    f"{course_hdr_lbl} — {len(clash_free_secs)} Clash-Free Option(s), {len(clashing_secs)} Clashing Option(s)",
                    expanded=True,
                ):
                    _render_html(f'<div style="margin-bottom:0.4rem;">{status_tag}</div>')

                    if clash_free_secs:
                        for idx_cf, opt in enumerate(clash_free_secs):
                            sec_lbl = str(opt["section"])
                            is_picked = sec_lbl == chosen_sec_for_rem
                            col_info, col_btn = st.columns([3.6, 1.35], gap="small")
                            _render_html(
                                f"""
                                <div class="cf-alt-row cf-alt-row-zero">
                                    <div>
                                        <span class="cf-badge-ok">✅ NO CLASH</span>
                                        <strong style="color:#0f172a;margin-left:0.45rem;">{html.escape(sec_lbl)}</strong>
                                        <span class="cf-muted" style="margin-left:0.4rem;font-size:0.78rem;">· {html.escape(str(opt['days_summary']))}</span>
                                        <div style="color:#1e293b;font-size:0.84rem;margin-top:0.22rem;">{html.escape(str(opt['schedule_summary']))}</div>
                                        <div class="cf-muted" style="font-size:0.77rem;">Rooms: {html.escape(str(opt['room_summary']))} · Instructor: {html.escape(str(opt['instructor']))}</div>
                                    </div>
                                </div>
                                """,
                                target=col_info,
                            )
                            _render_html("<div style='height:0.45rem;'></div>", target=col_btn)
                            if is_picked:
                                if col_btn.button(
                                    f"Selected: {sec_lbl} (Unselect)",
                                    key=f"t4_unsel_{rem_code}_{sec_lbl}_{idx_cf}",
                                    type="secondary",
                                    width="stretch",
                                ):
                                    next_map = dict(rem_sec_map)
                                    next_map.pop(rem_code, None)
                                    st.session_state["t4_pending_updates"] = {
                                        "t4_remaining_sections": next_map
                                    }
                                    st.rerun()
                            else:
                                if col_btn.button(
                                    f"Select {sec_lbl} (No Clash)",
                                    key=f"t4_sel_{rem_code}_{sec_lbl}_{idx_cf}",
                                    type="primary",
                                    width="stretch",
                                ):
                                    next_map = dict(rem_sec_map)
                                    next_map[rem_code] = sec_lbl
                                    st.session_state["t4_pending_updates"] = {
                                        "t4_remaining_sections": next_map
                                    }
                                    st.rerun()
                    else:
                        smart_swap = timetable.find_smart_regular_section_swap(
                            remaining_code=rem_code,
                            regular_courses=regular_courses,
                            regular_sections_map=regular_sections_map,
                            master_catalog_df=master_catalog_df,
                            locked_remaining_df=locked_others_df,
                        )
                        if smart_swap:
                            sw_items = list(smart_swap["regular_swaps"])
                            sw_desc = str(smart_swap["description"])
                            st.warning(
                                f"All sections of **{rem_code}** clash with your current regular classes. "
                                f"**Smart Regular-Section Swap Available:** {sw_desc} "
                                f"({smart_swap['remaining_summary']})."
                            )
                            if st.button(
                                f"Apply Combined Swap ({sw_items[0]['course_code']} → {sw_items[0]['to_section']} + {rem_code} {smart_swap['remaining_target_section']})",
                                key=f"t4_smart_swap_{rem_code}",
                                type="primary",
                                width="stretch",
                            ):
                                next_reg_swaps = dict(regular_swaps_map)
                                for sw in sw_items:
                                    next_reg_swaps[str(sw["course_code"])] = str(sw["to_section"])
                                next_rem_secs = dict(rem_sec_map)
                                next_rem_secs[rem_code] = str(smart_swap["remaining_target_section"])
                                st.session_state["t4_pending_updates"] = {
                                    "t4_regular_swaps": next_reg_swaps,
                                    "t4_remaining_sections": next_rem_secs,
                                }
                                st.rerun()
                        else:
                            st.error(
                                f"Every offering section for {rem_code} overlaps with your current regular timetable and no single regular-section swap clears it."
                            )

                    if clashing_secs:
                        st.caption("Clashing sections across CS classes:")
                        for opt_c in clashing_secs:
                            _render_html(
                                f"""
                                <div class="cf-alt-row cf-alt-row-clash">
                                    <div>
                                        <span class="cf-badge-bad">❌ CLASH</span>
                                        <strong style="color:#0f172a;margin-left:0.45rem;">{html.escape(str(opt_c['section']))}</strong>
                                        <span style="color:#991b1b;font-size:0.83rem;font-weight:650;margin-left:0.45rem;">{html.escape(str(opt_c['conflict_text']))}</span>
                                        <div class="cf-muted" style="font-size:0.78rem;margin-top:0.15rem;">{html.escape(str(opt_c['schedule_summary']))} · Instructor: {html.escape(str(opt_c['instructor']))}</div>
                                    </div>
                                </div>
                                """
                            )

    # Build Combined Merged Schedule (Regular Courses + Chosen Remaining Course Sections)
    merged_frames: list[pd.DataFrame] = []
    if not base_regular_df.empty:
        reg_tagged = base_regular_df.copy()
        reg_tagged["_course_kind"] = reg_tagged["course_code"].apply(
            lambda c: "Regular (Swapped)" if str(c).upper() in regular_swaps_map else "Regular Class"
        )
        merged_frames.append(reg_tagged)

    for rem_c in selected_remaining:
        chosen_s = timetable.normalize_class_section(rem_sec_map.get(rem_c, ""))
        if not chosen_s:
            continue
        rem_rows = master_catalog_df.loc[
            (master_catalog_df["course_code"].astype(str).str.upper() == rem_c)
            & (norm_sec_col == chosen_s)
        ].copy()
        if not rem_rows.empty:
            rem_rows["_course_kind"] = "Remaining / Repeater"
            merged_frames.append(rem_rows)

    if merged_frames:
        final_schedule_df = pd.concat(merged_frames, ignore_index=True)
        day_ord = {d: i for i, d in enumerate(timetable.COMSATS_ALL_DAYS)}
        final_schedule_df["_d"] = final_schedule_df["day"].map(day_ord).fillna(99)
        final_schedule_df = (
            final_schedule_df.sort_values(["_d", "start_time", "course_code"])
            .drop(columns=["_d"])
            .reset_index(drop=True)
        )
    else:
        final_schedule_df = master_catalog_df.iloc[0:0].copy()
        final_schedule_df["_course_kind"] = []

    # Persist planner state & synchronize Top Header KPI Strip
    final_records = final_schedule_df[
        ["course_code", "course_name", "section", "day", "start_time", "end_time", "room", "instructor"]
    ].to_dict(orient="records")
    prog_p, sec_p = timetable.split_class_section(selected_regular_section)
    database.save_planner_state(
        {
            "program": prog_p,
            "section_code": sec_p,
            "regular_section": selected_regular_section,
            "pdf_filename": loaded_pdf_name,
            "remaining_courses": list(selected_remaining),
            "remaining_sections": dict(rem_sec_map),
            "regular_swaps": dict(regular_swaps_map),
        }
    )

    working_sig = sorted(
        (str(r["course_code"]), str(r["section"]), str(r["day"]), str(r["start_time"]), str(r["end_time"]))
        for r in final_records
    )
    saved_sig = sorted(
        (str(r["course_code"]), str(r["section"]), str(r["day"]), str(r["start_time"]), str(r["end_time"]))
        for r in student_saved_slots
    )
    if working_sig != saved_sig:
        database.save_student_schedule(final_records)
        st.rerun()

    # -----------------------------------------------------------------------
    # STEP 4: Final Merged Timetable Grid & 1-Click PDF Download
    # -----------------------------------------------------------------------
    final_clashes_df = timetable.detect_clashes(final_schedule_df)
    unassigned_rems = [c for c in selected_remaining if not rem_sec_map.get(c)]
    with st.container(border=True):
        if final_clashes_df.empty and not unassigned_rems:
            final_pill = '<span class="cf-badge-ok">✅ CONFLICT-FREE (0 CLASHES)</span>'
        elif not final_clashes_df.empty:
            final_pill = f'<span class="cf-badge-bad">❌ {len(final_clashes_df)} CLASH(ES)</span>'
        else:
            final_pill = f'<span class="cf-source-pill">{len(unassigned_rems)} remaining course(s) pending section selection</span>'

        _render_html(
            f"""
            <div class="cf-step-head">
                <div class="cf-step-title-group">
                    <span class="cf-step-pill">Step 4</span>
                    <span class="cf-step-title">Final Merged Timetable Grid &amp; PDF Download</span>
                </div>
                {final_pill}
            </div>
            """
        )

        if unassigned_rems:
            st.info(
                f"Click **Select [Section] (No Clash)** (or **Apply Combined Swap**) in Step 3 for: "
                f"{', '.join(timetable.format_course_label(c, master_catalog_df) for c in unassigned_rems)}."
            )

        if final_schedule_df.empty:
            st.info("Select a regular class/section in Step 1 to generate your final merged timetable.")
        else:
            k1, k2, k3, k4 = st.columns(4, gap="small")
            k1.metric("Enrolled Courses", int(final_schedule_df["course_code"].nunique()))
            k2.metric("Weekly Classes & Labs", len(final_schedule_df))
            k3.metric("Sections Combined", int(final_schedule_df["section"].nunique()))
            k4.metric("Time Clashes", len(final_clashes_df))

            _render_weekly_timetable(timetable.build_weekly_matrix(final_schedule_df))

            summary_df = final_schedule_df.copy()
            summary_df["Course"] = summary_df["course_code"].apply(
                lambda c: timetable.format_course_label(str(c), master_catalog_df)
            )
            summary_df = summary_df[
                ["_course_kind", "Course", "section", "day", "start_time", "end_time", "room", "instructor"]
            ]
            summary_df.columns = ["Type", "Course", "Section", "Day", "Start", "End", "Room", "Instructor"]
            st.dataframe(summary_df, width="stretch", hide_index=True)

            swaps_notes_pdf = [
                f"Regular swap: {rc} moved {selected_regular_section} -> {new_sec}"
                for rc, new_sec in regular_swaps_map.items()
            ] + [
                f"Remaining / Repeater: {rc} scheduled in {sec_r}"
                for rc, sec_r in rem_sec_map.items()
            ]
            prereq_notes_pdf = [
                f"{pw['headline']} ({pw['detail']})"
                for pw in prereq_warnings
            ]
            final_pdf_bytes = timetable.generate_final_timetable_pdf(
                cohort_label=f"{selected_regular_section} (Fall 2026)",
                schedule_df=final_schedule_df,
                swaps_applied=swaps_notes_pdf,
                prereq_notes=prereq_notes_pdf,
            )
            dl_c1, dl_c2 = st.columns([2.2, 1], gap="small")
            dl_c1.download_button(
                "📄 Download Final Timetable (PDF)",
                data=final_pdf_bytes,
                file_name=f"campusflow_{selected_regular_section.lower()}_final_timetable.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
            dl_c2.caption(f"Printable A4 Landscape PDF · {max(1, len(final_pdf_bytes) // 1024)} KB")

ENHANCED_UI_JS = """
<div data-cf-js-hook="true" style="display:none;"></div>
<script>
(function () {
    try {
        var root = document;
        var win = window;
        var rafPending = false;

        function enhanceCampusFlowUI() {
            rafPending = false;
            var tiles = root.querySelectorAll(".cf-mini-stat");
            tiles.forEach(function (tile, idx) {
                var delayVal = (idx * 60) + "ms";
                if (tile.style.animationDelay !== delayVal) {
                    tile.style.animationDelay = delayVal;
                }
            });

            var studioAnchor = root.querySelector('[data-cf-studio="true"]');
            var rightAnchor = root.querySelector('[data-cf-right-stack="true"]');
            if (!studioAnchor || !rightAnchor) {
                return;
            }
            var leftBox = studioAnchor.closest('[data-testid="stVerticalBlock"]');
            var rightCol = rightAnchor.closest('[data-testid="stColumn"]');
            if (!leftBox || !rightCol) {
                return;
            }
            var rightCards = rightCol.querySelectorAll(".cf-card");
            if (!rightCards.length) {
                return;
            }
            if (win.innerWidth < 992) {
                if (leftBox.style.minHeight !== "") {
                    leftBox.style.minHeight = "";
                }
                return;
            }
            var firstRect = rightCards[0].getBoundingClientRect();
            var lastRect = rightCards[rightCards.length - 1].getBoundingClientRect();
            var targetHeight = Math.round(lastRect.bottom - firstRect.top);
            if (targetHeight > 220) {
                var nextVal = targetHeight + "px";
                if (leftBox.style.minHeight !== nextVal) {
                    leftBox.style.minHeight = nextVal;
                }
            }
        }

        function scheduleEnhance() {
            if (!rafPending) {
                rafPending = true;
                win.requestAnimationFrame(enhanceCampusFlowUI);
            }
        }

        scheduleEnhance();
        win.setTimeout(enhanceCampusFlowUI, 120);
        win.setTimeout(enhanceCampusFlowUI, 380);
        if (!win.__cfResizeBound) {
            win.addEventListener("resize", scheduleEnhance);
            if (win.MutationObserver && root.body) {
                var obs = new win.MutationObserver(scheduleEnhance);
                obs.observe(root.body, { childList: true, subtree: true });
            }
            win.__cfResizeBound = true;
        }
    } catch (err) {
        /* ignore sandbox frame restrictions */
    }
})();
</script>
"""
st.html(ENHANCED_UI_JS, unsafe_allow_javascript=True)
