# CampusFlow — Multimodal AI Lecture Companion & Smart Timetable Resolver

**Track:** Education & Learning  
**Event:** CUI Tech Fest FA'26 — Build with Google Hackathon  
**UN SDG Alignment:** SDG 4 (Quality Education) — Inclusive, bilingual (Urdu/English) study synthesis and academic schedule optimization.

---

## Problem Statement

University students in Pakistan regularly struggle to consolidate fast-paced lectures delivered in code-mixed Urdu and English across spoken audio, handwritten whiteboard derivations, and dense PDF slide decks. Key equations or smudged ink on classroom boards often lead to inaccurate guesswork in AI-generated notes. At the same time, repeating or cross-semester students—such as a COMSATS `BCS-5A` student balancing repeaters and core courses—frequently run into overlapping class slots during course registration and spend hours manually hunting for clash-free sections.

## Solution Overview

**CampusFlow** combines a multimodal Gemini lecture synthesis pipeline with a deterministic Pandas timetable conflict solver inside one full-width Streamlit workspace:

1. **Capture** — Accepts spoken bilingual lecture audio (`.wav`, `.mp3`, `.m4a`), classroom whiteboard snapshots (`.png`, `.jpg`), and slide decks (`.pdf`), synthesizing structured study notes, worked formulas/pseudocode, and source-attributed knowledge units in SQLite. Includes 1-click sample kits across 4 Computer Science courses (`CSC322`, `CSC325`, `AIC354`, `CSC301`) plus a **Needs Review** human verification panel for ambiguous board markings.
2. **Vault & Q&A** — Searchable lecture vault backed by SQLite TF-IDF retrieval and strictly grounded Q&A. Every response cites observed source badges (`[AUDIO @ 01:45]`, `[WHITEBOARD_IMAGE @ Whiteboard #1]`, `[SLIDES_PDF @ Slide 2]`) and warns explicitly if any referenced unit is still unverified.
3. **Quiz & Flashcards** — Generates 10-question interactive MCQ quizzes with a step-by-step evaluation wizard alongside a 6-card flip study deck grounded strictly in saved lecture notes.
4. **Timetable & Clash** — Models realistic COMSATS Fall 2026 enrollment (defaulting to a `BCS-5A` `Mon 08:30–10:00` collision between `AIC354` and `CSC322`), detects exact overlapping intervals via Pandas, and computes 1-click clash-free section swaps (`BCS-5B`, `BAI-1`, `BCT-1`).

---

## Google Technologies & AI Models

- **Google Gemini SDK (`google-genai`)** with structured Pydantic JSON schema enforcement (`LectureSynthesisResult`, `RAGAnswer`, `RevisionPack`) and resilient model fallback.
- **`gemini-2.5-flash`** for low-latency multimodal audio transcription (including Roman-Urdu/English code-mixing), whiteboard vision OCR, PDF slide parsing, and grounded Q&A synthesis.
- **Pydantic** for strict schema validation on all generated notes, citations, MCQs, and flashcards.
- **SQLite** for local lecture persistence, TF-IDF evidence retrieval, revision packs, and enrolled timetable slots.
- **Pandas** for deterministic slot overlap detection and clash-free section recommendation ranking.
- **Streamlit** for the responsive dark-mode multi-tab web application and automated `AppTest` verification.

---

## Responsible AI & Anti-Hallucination Design

- **Strict Uncertainty Flagging (`is_uncertain=True`):** Whenever handwriting on a whiteboard photo is smudged, partial math symbols are obscured, or audio is ambiguous, the synthesis pipeline writes `[UNCLEAR: ...]` and sets `is_uncertain=True` with a concrete `uncertainty_reason` instead of fabricating values.
- **Human-in-the-Loop Verification:** Flagged units are surfaced cleanly in the **Needs Review** review cards (`Confirm & Save`), requiring student verification before those details count as confirmed or feed quiz generation.
- **Zero-Hallucination Guardrail:** Grounded Q&A rejects unanswerable prompts when retrieval returns no matching lecture evidence (`grounded=False`) and filters output citations exclusively to verified source badges present in retrieved chunks.

---

## Project Structure

```text
CampusFlow/
├── campusflow/
│   ├── app.py                  # 4-tab Streamlit application & reactive state orchestration
│   ├── ai_engine.py            # Gemini SDK multimodal pipeline, RAG Q&A, quiz generation & cache
│   ├── database.py             # SQLite schema, CRUD, TF-IDF retriever & 4-subject demo seeding
│   ├── timetable.py            # COMSATS Fall 2026 catalog, clash detector & swap recommender
│   └── tests/
│       └── test_campusflow.py  # Automated unit & Streamlit AppTest verification suite
├── demo_files/                 # 12 multimodal sample artifacts (.wav, smudged .png, .pdf)
├── generate_demos.py           # Reproducible generator for the 4-subject multimodal demo kit
├── requirements.txt            # Python package dependencies
├── .env.example                # Environment variable template (GEMINI_API_KEY)
├── README.md
└── SUBMISSION_DETAILS.md
```

---

## Quick Start

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the application

```bash
streamlit run campusflow/app.py
```

The app initializes its SQLite database automatically with pre-seeded lectures (`CSC322`, `CSC325`, `AIC354`, `CSC301`) and the default `BCS-5A` schedule clash scenario ready to test immediately.

### 4. Run automated verification

```bash
python -m unittest campusflow/tests/test_campusflow.py -v
```
