# Hackathon Submission Details

## Project Overview

- **Project Title:** CampusFlow — Multimodal AI Lecture Companion & Smart Timetable Resolver
- **Event:** CUI Tech Fest FA'26 — Build with Google Hackathon
- **Challenge Domain:** Education & Learning
- **UN SDG Alignment:** SDG 4 (Quality Education) — Inclusive, bilingual (Urdu/English) study synthesis and academic schedule optimization.
- **Repository:** `https://github.com/adamkhan-dev/campusflow.git`

---

## Problem Statement

University students in Pakistan struggle to synthesize code-mixed Urdu-English lectures, complex whiteboard derivations, and slide decks into reliable study notes, while students repeating courses (such as those registered in `BCS-5A`) frequently encounter cross-section timetable clashes during course registration.

---

## Key Capabilities Demonstrated

1. **Multimodal Lecture Synthesis (`Capture`):**
   - Ingests `.wav`/`.mp3` spoken audio (handling Urdu-English technical code-mixing), `.png`/`.jpg` whiteboard photos, and `.pdf` lecture slides in a single structured Gemini call.
   - Pre-packaged with 12 reproducible multimodal assets in `demo_files/` spanning four Computer Science courses:
     - `CSC322 — Operating Systems` (FCFS CPU Scheduling — smudged P2 Burst Time flagged for human confirmation)
     - `CSC325 — Computer Organization & Assembly` (IEEE 754 Single-Precision Floating Point — smudged biased exponent flagged)
     - `AIC354 — Machine Learning Fundamentals` (Naive Bayes Classification — smudged Laplace smoothing denominator flagged)
     - `CSC301 — Design & Analysis of Algorithms` (Binary Search & Recurrence Complexity — smudged recurrence term flagged)
2. **Source-Cited Lecture Vault & Grounded Q&A (`Vault & Q&A`):**
   - Stores normalized knowledge units in SQLite and ranks relevant passages using Python TF-IDF + token similarity before answering student questions.
   - Prevents off-syllabus fabrication when no matching evidence exists.
3. **Interactive Revision Engine (`Quiz & Flashcards`):**
   - Generates a 10-question MCQ wizard with progress tracking, final score accuracy breakdown, and source-cited answer explanations, plus a 6-card interactive flashcard deck.
4. **COMSATS Fall 2026 Clash Resolver (`Timetable & Clash`):**
   - Models a realistic `BCS-5A` student schedule where `AIC354` overlaps with `CSC322` on `Mon 08:30–10:00`.
   - Uses deterministic interval arithmetic in Pandas across official sections (`BCS-5A`, `BCS-5B`, `BCS-1A`, `BCS-1B`, `BSE-1`, `BAI-1`, `BCT-1`), rooms (`106 AB-2`, `107 AB-2`, `CL-5`, `CL-6`), and faculty to offer 1-click conflict-free swaps (`Apply Swap`).

---

## Google Technologies & AI Models Disclosed

| Layer | Technology / Model | Purpose |
|---|---|---|
| Generative AI SDK | Google Gemini SDK (`google-genai`) | Multimodal ingestion, RAG Q&A, structured JSON quiz generation |
| AI Model | `gemini-2.5-flash` | Audio comprehension, whiteboard OCR, slide parsing, grounded Q&A |
| Output Validation | Pydantic v2 (`BaseModel`) | Strict JSON schema validation for notes, RAG citations, MCQs, flashcards |
| Persistence & Retrieval | SQLite (`sqlite3`) + Python TF-IDF | Local-first academic vault, audit trail, fast lexical retrieval |
| Schedule Solver | Pandas | Deterministic interval overlap detection and section swap ranking |
| Web Frontend | Streamlit | Multi-tab workspace with dark UI styling, session state sync, and `AppTest` suite |

---

## Responsible AI Implementation

- **Anti-Fabrication Instruction & Schema Field:** Every extracted knowledge unit carries `is_uncertain: bool` and `uncertainty_reason: str`. Whenever whiteboard writing or mathematical notation is smudged, occluded, or blurred, the model marks `[UNCLEAR: ...]` with `is_uncertain=True` rather than guessing numbers or formulas.
- **Human-in-the-Loop Gate:** Uncertain records appear inside highlighted **Needs Review** cards right after capture (`Confirm & Save`). Revision quiz generators explicitly avoid creating quiz questions from unverified entries until confirmed by the student.
- **Strict Grounding:** Q&A answers cite only verified evidence badges retrieved from the active lecture (`[AUDIO @ ...]`, `[WHITEBOARD_IMAGE @ ...]`, `[SLIDES_PDF @ ...]`).

---

## Automated Testing Summary

Run the complete test suite (`10` automated unit and end-to-end UI `AppTest` checks) from `E:\CampusFlow`:

```bash
python -m unittest campusflow/tests/test_campusflow.py -v
```

Coverage includes:
- SQLite schema migrations, CRUD, uncertainty verification, and TF-IDF ranking
- Pydantic validation for 10-MCQ + 6-flashcard contracts and response caching
- Missing API key error handling and ungrounded query safety
- COMSATS Fall 2026 timetable adjacency logic, `BCS-5A` clash detection, and swap resolution
- End-to-end Streamlit UI reactivity across all 4 subjects, 10-step MCQ wizard, flashcard flip, and schedule swap/restore
