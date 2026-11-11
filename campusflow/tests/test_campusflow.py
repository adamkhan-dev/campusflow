"""Automated verification suite for CAMPUSFLOW core modules and UI workflows."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

CAMPUSFLOW_DIR = Path(__file__).resolve().parents[1]
if str(CAMPUSFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPUSFLOW_DIR))

import ai_engine  # noqa: E402
import database  # noqa: E402
import timetable  # noqa: E402
import pandas as pd
from pydantic import ValidationError  # noqa: E402


class TestDatabaseAndRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_campusflow.db"
        database.init_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_schema_tables_and_columns(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertTrue(
                {"lectures", "knowledge_chunks", "revision_items", "lecture_feedback", "timetable_slots"}.issubset(
                    tables
                )
            )
            lec_cols = {r[1] for r in conn.execute("PRAGMA table_info(lectures)").fetchall()}
            chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_chunks)").fetchall()}
            rev_cols = {r[1] for r in conn.execute("PRAGMA table_info(revision_items)").fetchall()}
            tt_cols = {r[1] for r in conn.execute("PRAGMA table_info(timetable_slots)").fetchall()}
        finally:
            conn.close()

        self.assertTrue({"id", "course_code", "lecture_title", "created_at", "summary_json"}.issubset(lec_cols))
        self.assertTrue(
            {"chunk_id", "lecture_id", "topic", "content", "source_type", "source_reference", "is_uncertain"}.issubset(
                chunk_cols
            )
        )
        self.assertTrue({"id", "lecture_id", "item_type", "payload_json"}.issubset(rev_cols))
        self.assertTrue({"id", "course_code", "section", "day", "start_time", "end_time", "room", "instructor"}.issubset(tt_cols))

    def test_lecture_crud_verification_and_tfidf_retrieval(self) -> None:
        summary = {
            "executive_summary": "Lecture covers Binary Search on sorted arrays and O(log n) complexity.",
            "structured_notes": [
                {
                    "heading": "Binary Search Midpoint",
                    "bullets": ["Use mid = low + (high - low) // 2 to avoid integer overflow."],
                    "code_mix_terms": ["array ko half kar dete hain"],
                }
            ],
            "extracted_pseudocode": "while low <= high: mid = low + (high - low) // 2",
        }
        chunks = [
            {
                "topic": "Binary Search Precondition",
                "content": "Binary search requires the input array to be sorted before searching.",
                "source_type": "AUDIO",
                "source_reference": "00:42",
                "is_uncertain": False,
            },
            {
                "topic": "Midpoint Overflow Prevention",
                "content": "Professor derived mid = low + (high - low) // 2 on the whiteboard.",
                "source_type": "WHITEBOARD_IMAGE",
                "source_reference": "Whiteboard #1",
                "is_uncertain": False,
            },
            {
                "topic": "Time Complexity Derivation",
                "content": "[UNCLEAR: smudged exponent in recurrence relation on bottom corner]",
                "source_type": "WHITEBOARD_IMAGE",
                "source_reference": "Whiteboard #1",
                "is_uncertain": True,
                "uncertainty_reason": "Handwriting is smudged in lower-right corner.",
            },
        ]

        lecture_id = database.save_lecture(
            course_code="CSC211",
            lecture_title="Binary Search",
            summary_data=summary,
            chunks=chunks,
            db_path=self.db_path,
        )
        self.assertGreater(lecture_id, 0)

        lectures = database.list_lectures(db_path=self.db_path)
        self.assertEqual(len(lectures), 1)
        self.assertEqual(lectures[0]["chunk_count"], 3)
        self.assertEqual(lectures[0]["uncertain_count"], 1)

        uncertain = database.get_lecture_chunks(lecture_id, only_uncertain=True, db_path=self.db_path)
        self.assertEqual(len(uncertain), 1)
        self.assertTrue(
            database.verify_uncertain_chunk(
                uncertain[0]["chunk_id"],
                "Recurrence T(n) = T(n/2) + O(1) yields O(log n) time complexity.",
                db_path=self.db_path,
            )
        )
        self.assertEqual(
            len(database.get_lecture_chunks(lecture_id, only_uncertain=True, db_path=self.db_path)),
            0,
        )

        hits = database.retrieve_relevant_chunks(
            query="How do we avoid integer overflow in midpoint?",
            lecture_id=lecture_id,
            top_k=2,
            db_path=self.db_path,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["topic"], "Midpoint Overflow Prevention")
        self.assertEqual(hits[0]["source_badge"], "[WHITEBOARD_IMAGE @ Whiteboard #1]")

        no_hits = database.retrieve_relevant_chunks(
            query="photosynthesis chlorophyll mitochondria",
            lecture_id=lecture_id,
            db_path=self.db_path,
        )
        self.assertEqual(no_hits, [])

        mcqs = [
            {
                "question": f"Sample Q{i}",
                "options": ["A", "B", "C", "D"],
                "correct_index": 1,
                "explanation": "B is correct.",
                "source_badge": "[AUDIO @ 00:42]",
            }
            for i in range(10)
        ]
        database.save_revision_items(lecture_id, "mcq", mcqs, db_path=self.db_path)
        loaded = database.get_revision_items(lecture_id, "mcq", db_path=self.db_path)
        self.assertEqual(len(loaded), 10)

        # Saving another capture for the same course replaces rather than duplicating course rows
        replacement_id = database.save_lecture(
            course_code="CSC211",
            lecture_title="Binary Search — Updated Session",
            summary_data=summary,
            chunks=chunks[:2],
            db_path=self.db_path,
        )
        all_after_replace = database.list_lectures(db_path=self.db_path)
        self.assertEqual(len(all_after_replace), 1)
        self.assertEqual(all_after_replace[0]["id"], replacement_id)
        self.assertEqual(database.get_lecture_chunks(lecture_id, db_path=self.db_path), [])

        self.assertTrue(database.delete_lecture(replacement_id, db_path=self.db_path))
        self.assertEqual(database.get_lecture_chunks(replacement_id, db_path=self.db_path), [])
        self.assertEqual(database.get_revision_items(replacement_id, db_path=self.db_path), [])

    def test_feedback_and_update_lecture_notes(self) -> None:
        lec_id = database.save_lecture(
            course_code="CSC336",
            lecture_title="RESTful APIs",
            summary_data={
                "course_label": "CSC336 — Web Technologies",
                "generation_source": "gemini_live",
                "executive_summary": "Initial summary.",
                "structured_notes": [
                    {
                        "heading": "HTTP Verbs",
                        "source_tag": "SLIDES",
                        "bullets": ["GET reads, POST creates."],
                    }
                ],
                "extracted_pseudocode": "",
            },
            chunks=[
                {
                    "topic": "HTTP Methods",
                    "content": "GET retrieves state safely; POST creates resources.",
                    "source_type": "SLIDES_PDF",
                    "source_reference": "Slide 1",
                    "is_uncertain": False,
                }
            ],
            db_path=self.db_path,
        )
        self.assertIsNone(database.get_latest_lecture_feedback(lec_id, db_path=self.db_path))
        fb_good = database.save_lecture_feedback(lec_id, "good", "", db_path=self.db_path)
        self.assertGreater(fb_good, 0)
        latest = database.get_latest_lecture_feedback(lec_id, db_path=self.db_path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["rating"], "good")

        database.save_lecture_feedback(lec_id, "bad", "Add status codes & curl example", db_path=self.db_path)
        latest_bad = database.get_latest_lecture_feedback(lec_id, db_path=self.db_path)
        self.assertEqual(latest_bad["rating"], "bad")
        self.assertIn("status codes", latest_bad["feedback_text"])

        updated = database.update_lecture_notes(
            lec_id,
            {
                "course_label": "CSC336 — Web Technologies",
                "generation_source": "gemini_live",
                "executive_summary": "Deepened REST summary with status codes.",
                "structured_notes": [],
                "extracted_pseudocode": "",
            },
            db_path=self.db_path,
        )
        self.assertTrue(updated)
        reloaded = database.get_lecture(lec_id, db_path=self.db_path)
        self.assertEqual(reloaded["summary"]["executive_summary"], "Deepened REST summary with status codes.")

        # Deleting lecture cascades cleanly to feedback rows
        database.delete_lecture(lec_id, db_path=self.db_path)
        self.assertIsNone(database.get_latest_lecture_feedback(lec_id, db_path=self.db_path))


class TestAIEngineContracts(unittest.TestCase):
    def test_mime_inference_and_schema_validation(self) -> None:
        self.assertEqual(ai_engine.infer_mime_type("lecture.mp3"), "audio/mpeg")
        self.assertEqual(ai_engine.infer_mime_type("lecture.wav"), "audio/wav")
        self.assertEqual(ai_engine.infer_mime_type("board.jpg"), "image/jpeg")
        self.assertEqual(ai_engine.infer_mime_type("slides.pdf"), "application/pdf")

        res = ai_engine.answer_grounded_question("What is FCFS?", [])
        self.assertFalse(res.grounded)
        self.assertEqual(res.cited_badges, [])

        with self.assertRaises(ValidationError):
            ai_engine.MCQItem(
                question="Invalid options count?",
                options=["Only", "Two"],
                correct_index=0,
                explanation="Bad",
                source_badge="[AUDIO @ 00:10]",
            )

        sample_mcq = ai_engine.MCQItem(
            question="Q?",
            options=["A", "B", "C", "D"],
            correct_index=0,
            explanation="Because A.",
            source_badge="[SLIDES_PDF @ Slide 1]",
        )
        sample_fc = ai_engine.FlashcardItem(
            front="Front?",
            back="Back.",
            source_badge="[SLIDES_PDF @ Slide 1]",
        )
        with self.assertRaises(ValidationError):
            ai_engine.RevisionPack(mcqs=[sample_mcq] * 5, flashcards=[sample_fc] * 6)
        pack = ai_engine.RevisionPack(mcqs=[sample_mcq] * 10, flashcards=[sample_fc] * 6)
        self.assertEqual(len(pack.mcqs), 10)
        self.assertEqual(len(pack.flashcards), 6)

    def test_missing_api_key_raises_cleanly(self) -> None:
        orig_gem = os.environ.pop("GEMINI_API_KEY", None)
        orig_goog = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                ai_engine.get_client(api_key="")
        finally:
            if orig_gem is not None:
                os.environ["GEMINI_API_KEY"] = orig_gem
            if orig_goog is not None:
                os.environ["GOOGLE_API_KEY"] = orig_goog

    def test_gemini_response_cache_deduplicates_repeat_calls(self) -> None:
        ai_engine.clear_response_cache()
        fake_rag_json = (
            '{"answer":"FCFS is non-preemptive [AUDIO @ 00:42].",'
            '"cited_badges":["[AUDIO @ 00:42]"],"grounded":true}'
        )
        fake_resp = mock.Mock(text=fake_rag_json)
        chunks = [
            {
                "topic": "FCFS Policy",
                "content": "FCFS dispatches in arrival order.",
                "source_type": "AUDIO",
                "source_reference": "00:42",
                "is_uncertain": False,
                "source_badge": "[AUDIO @ 00:42]",
            }
        ]
        with (
            mock.patch.object(ai_engine, "get_client", return_value=mock.Mock()),
            mock.patch.object(
                ai_engine, "_generate_content_resilient", return_value=fake_resp
            ) as mock_gen,
        ):
            r1 = ai_engine.answer_grounded_question("What is FCFS?", chunks, model_name="gemini-3.6-flash")
            r2 = ai_engine.answer_grounded_question("What is FCFS?", chunks, model_name="gemini-3.6-flash")
            self.assertEqual(mock_gen.call_count, 1)
            self.assertEqual(r1.answer, r2.answer)
            self.assertTrue(r2.grounded)
        ai_engine.clear_response_cache()

    def test_single_and_multi_modality_timeout_fallback(self) -> None:
        pdf_only = [("slides.pdf", b"%PDF-1.4 (Binary Search Slide Deck)", "application/pdf")]
        audio_only = [("audio.wav", b"RIFF0000WAVE", "audio/wav")]
        img_only = [("board.png", b"\x89PNG\r\n\x1a\n", "image/png")]

        self.assertEqual(ai_engine.describe_provided_modalities(pdf_files=pdf_only)["inputs_active"], 1)
        self.assertEqual(
            ai_engine.describe_provided_modalities(audio_files=audio_only, pdf_files=pdf_only)["inputs_active"],
            2,
        )
        self.assertEqual(
            ai_engine.describe_provided_modalities(
                audio_files=audio_only, image_files=img_only, pdf_files=pdf_only
            )["inputs_active"],
            3,
        )
        self.assertEqual(ai_engine.describe_provided_modalities(extra_notes="Quick notes")["inputs_active"], 1)

        def _slow_gemini(*args, **kwargs):
            time.sleep(0.5)
            raise AssertionError("Should have timed out before completing")

        with (
            mock.patch.object(ai_engine, "is_api_configured", return_value=True),
            mock.patch.object(ai_engine, "process_lecture_materials", side_effect=_slow_gemini),
            mock.patch.object(ai_engine, "generate_revision_pack", side_effect=_slow_gemini),
        ):
            t0 = time.perf_counter()
            syn_pdf, used_fb = ai_engine.synthesize_lecture_with_timeout(
                course_code="CSC301",
                lecture_title="Slides Only Test",
                pdf_files=pdf_only,
                timeout_seconds=0.05,
            )
            elapsed = time.perf_counter() - t0
            self.assertTrue(used_fb)
            self.assertLess(elapsed, 0.45)
            self.assertTrue(all(c.source_type == "SLIDES_PDF" for c in syn_pdf.knowledge_chunks))
            self.assertGreaterEqual(len(syn_pdf.structured_notes), 1)
            self.assertTrue(
                all(
                    sec.source_type == "SLIDES_PDF"
                    and len(sec.subsections) >= 2
                    and all(sub.subheading and sub.body for sub in sec.subsections)
                    and sec.takeaway
                    for sec in syn_pdf.structured_notes
                )
            )

            syn_audio, _ = ai_engine.synthesize_lecture_with_timeout(
                course_code="CSC322",
                lecture_title="Audio Only Test",
                audio_files=audio_only,
                timeout_seconds=0.05,
            )
            self.assertTrue(all(c.source_type == "AUDIO" for c in syn_audio.knowledge_chunks))
            self.assertTrue(
                all(
                    sec.source_type == "AUDIO"
                    and len(sec.subsections) >= 2
                    and all(sub.subheading and sub.body for sub in sec.subsections)
                    and sec.takeaway
                    for sec in syn_audio.structured_notes
                )
            )

            syn_img, _ = ai_engine.synthesize_lecture_with_timeout(
                course_code="CSC325",
                lecture_title="Image Only Test",
                image_files=img_only,
                timeout_seconds=0.05,
            )
            self.assertTrue(all(c.source_type == "WHITEBOARD_IMAGE" for c in syn_img.knowledge_chunks))
            self.assertTrue(
                all(
                    sec.source_type == "WHITEBOARD_IMAGE"
                    and len(sec.subsections) >= 2
                    and all(sub.subheading and sub.body for sub in sec.subsections)
                    and sec.takeaway
                    for sec in syn_img.structured_notes
                )
            )

            bayes_only = [("bayes.webp", b"RIFF0000WEBP", "image/webp")]
            syn_bayes, _ = ai_engine.synthesize_lecture_with_timeout(
                course_code="CSC322",
                lecture_title="First-Come, First-Served (FCFS) CPU Scheduling",
                image_files=bayes_only,
                timeout_seconds=0.05,
            )
            self.assertIn("Bayes", syn_bayes.detected_topic)
            self.assertTrue(all(c.source_type == "WHITEBOARD_IMAGE" for c in syn_bayes.knowledge_chunks))
            self.assertTrue(all(sec.source_type == "WHITEBOARD_IMAGE" for sec in syn_bayes.structured_notes))
            self.assertIn("P(A | B)", syn_bayes.model_dump_json())

            syn_notes, _ = ai_engine.synthesize_lecture_with_timeout(
                course_code="AIC354",
                lecture_title="Notes Only Test",
                extra_notes="Laplace smoothing avoids zero probability.",
                timeout_seconds=0.05,
            )
            self.assertTrue(all(c.source_type == "NOTES" for c in syn_notes.knowledge_chunks))

            syn_custom, _ = ai_engine.synthesize_lecture_with_timeout(
                course_code="CSC411",
                lecture_title="Dijkstra Shortest Path",
                audio_files=audio_only,
                timeout_seconds=0.05,
            )
            self.assertTrue(all(c.source_type == "AUDIO" for c in syn_custom.knowledge_chunks))

            banned_logs = (
                "Processed audio file",
                "Captures spoken concept definitions",
                "Indexed board photo",
                "Preserves visual diagrams",
                "Parsed slides deck",
                "y_hat = argmax_y",
                "N_{y,i}",
                "Classroom note:",
                "code_mix_terms",
            )
            for syn_res in (syn_pdf, syn_audio, syn_img, syn_bayes, syn_notes, syn_custom):
                dumped_text = syn_res.model_dump_json()
                for phrase in banned_logs:
                    self.assertNotIn(phrase, dumped_text)

            chunk_dicts = [
                {
                    "topic": c.topic,
                    "content": c.content,
                    "source_type": c.source_type,
                    "source_reference": c.source_reference,
                }
                for c in syn_pdf.knowledge_chunks
            ]
            pack, rev_fb = ai_engine.generate_revision_pack_with_timeout(
                lecture_title="Slides Only Test",
                chunks=chunk_dicts,
                timeout_seconds=0.05,
            )
            self.assertTrue(rev_fb)
            self.assertEqual(len(pack.mcqs), 10)
            self.assertEqual(len(pack.flashcards), 6)

    def test_regenerate_lecture_notes_with_feedback_live_and_fallback(self) -> None:
        fake_regen = ai_engine.LectureSynthesisResult(
            detected_topic="Eigenvalues & Eigenvectors",
            executive_summary="Deep intuitive breakdown clarifying eigenvalues and spectral decomposition.",
            structured_notes=[
                ai_engine.StructuredNoteSection(
                    heading="Spectral Intuition & Eigendecomposition",
                    source_type="SLIDES_PDF",
                    subsections=[
                        ai_engine.NoteSubsection(
                            subheading="Geometric Meaning of A v = λ v",
                            body=(
                                "An eigenvector v ≠ 0 keeps its direction under matrix transformation A while stretching "
                                "by scalar eigenvalue λ.\n\n"
                                "Diagonalization changes coordinates into this eigenbasis so matrix powers become fast scalar exponents."
                            ),
                            formula="A v = λ v  |  det(A − λI) = 0",
                            bullets=[
                                "λ — Eigenvalue scaling factor.",
                                "v — Non-zero eigenvector invariant subspace.",
                            ],
                        ),
                        ai_engine.NoteSubsection(
                            subheading="Diagonal Matrix Example & Null-Space Extraction",
                            body="Solve characteristic determinant det(A − λI) = 0, then compute null spaces cleanly.",
                            formula="Aⁿ = P Dⁿ P⁻¹",
                            bullets=[
                                "For diag(3, 5), eigenvalues are 3 and 5 with standard basis eigenvectors.",
                                "Symmetric real matrices always yield orthogonal eigenvectors.",
                            ],
                        ),
                    ],
                    takeaway="Key Equation: A v = λ v  →  Aⁿ = P Dⁿ P⁻¹",
                )
            ],
            extracted_pseudocode="det(A − λI) = 0",
            knowledge_chunks=[
                ai_engine.KnowledgeChunkItem(
                    topic="Eigenvalue Definition",
                    content="A non-zero vector v satisfies A v = λ v.",
                    source_type="SLIDES_PDF",
                    source_reference="Slide 1",
                    is_uncertain=False,
                )
            ],
        )
        fake_resp = mock.Mock(text=fake_regen.model_dump_json())
        chunks = [
            {
                "topic": "Eigenvalue Definition",
                "content": "A v = λ v.",
                "source_type": "SLIDES_PDF",
                "source_reference": "Slide 1",
            }
        ]
        with (
            mock.patch.object(ai_engine, "get_client", return_value=mock.Mock()),
            mock.patch.object(ai_engine, "_generate_content_resilient", return_value=fake_resp) as mock_gen,
        ):
            res = ai_engine.regenerate_lecture_notes_with_feedback(
                course_code="MATH202",
                lecture_title="Eigenvalues & Eigenvectors",
                current_summary={"executive_summary": "Brief", "structured_notes": []},
                chunks=chunks,
                feedback_text="Make explanations deeper and add a worked example",
                model_name="gemini-3.5-flash-lite",
            )
            self.assertEqual(mock_gen.call_count, 1)
            self.assertEqual(res._model_used, "gemini-3.5-flash-lite")
            self.assertEqual(res.structured_notes[0].source_type, "SLIDES_PDF")
            self.assertGreaterEqual(len(res.structured_notes[0].subsections), 2)
            self.assertIn("Diagonalization changes coordinates", res.structured_notes[0].subsections[0].body)
            self.assertIn("diag(3, 5)", res.structured_notes[0].subsections[1].bullets[0])

        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            fb_res, used_fb = ai_engine.regenerate_lecture_notes_with_timeout(
                course_code="MATH202",
                lecture_title="Eigenvalues & Eigenvectors",
                current_summary={
                    "executive_summary": "Brief.",
                    "structured_notes": [
                        {"heading": "Intro", "source_tag": "SLIDES", "bullets": ["A v = λ v."]}
                    ],
                },
                chunks=chunks,
                feedback_text="Simplify formulas",
            )
            self.assertTrue(used_fb)
            self.assertEqual(fb_res.structured_notes[0].source_type, "SLIDES_PDF")
            self.assertGreaterEqual(len(fb_res.structured_notes[0].subsections), 2)
            self.assertTrue(all(sub.subheading and sub.body for sub in fb_res.structured_notes[0].subsections))
            self.assertTrue(fb_res.structured_notes[0].takeaway)


class TestTimetableSolver(unittest.TestCase):
    def test_official_fall_2026_catalog_and_clash_resolution(self) -> None:
        catalog5 = timetable.get_semester_5_catalog_df()
        sections = set(catalog5["section"].unique())
        rooms = set(catalog5["room"].unique())
        instructors = set(catalog5["instructor"].unique())

        self.assertEqual(sections, {"BCS-5A", "BCS-5B", "BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BCT-1"})
        self.assertEqual(rooms, {"106 AB-2", "107 AB-2", "CL-5", "CL-6"})
        self.assertEqual(instructors, {"Mr. Rizwan Rashid", "Mr. Khurram Iqbal", "Ms. Ayesha Hussain"})

        catalog = timetable.get_sample_catalog_df()
        default_sel = timetable.get_default_selected_sections()
        sched = timetable.build_selected_schedule_df(default_sel, catalog)
        clashes = timetable.detect_clashes(sched)

        self.assertEqual(len(clashes), 1)
        self.assertEqual(clashes.iloc[0]["day"], "Mon")
        self.assertEqual({clashes.iloc[0]["course_a"], clashes.iloc[0]["course_b"]}, {"CSC211", "CSC301"})

        recs = timetable.recommend_clash_free_alternatives(default_sel, catalog)
        self.assertTrue(any(r["resolves_all_clashes"] for r in recs))

        best = next(r for r in recs if r["resolves_all_clashes"])
        resolved_sel = dict(default_sel)
        resolved_sel[str(best["course_code"])] = str(best["to_section"])
        resolved_sched = timetable.build_selected_schedule_df(resolved_sel, catalog)
        self.assertTrue(timetable.detect_clashes(resolved_sched).empty)

        matrix = timetable.build_weekly_matrix(resolved_sched)
        self.assertEqual(list(matrix.columns), timetable.COMSATS_DAYS)

    def test_adjacent_slots_do_not_clash_and_formatting(self) -> None:
        self.assertFalse(timetable._slots_overlap("Mon", "08:30", "10:00", "Mon", "10:00", "11:30"))
        self.assertTrue(timetable._slots_overlap("Mon", "08:30", "10:00", "Mon", "09:30", "11:00"))
        self.assertFalse(timetable._slots_overlap("Mon", "08:30", "10:00", "Tue", "08:30", "10:00"))
        self.assertEqual(timetable.format_course_label("CSC322"), "CSC322 — Operating Systems")
        self.assertEqual(timetable.parse_course_code("CSC322 — Operating Systems"), "CSC322")

    def test_official_comsats_pdf_catalog_parser_and_multi_course_solver(self) -> None:
        pdf_df = timetable.get_official_comsats_pdf_catalog_df()
        self.assertEqual(
            set(pdf_df["section"].unique()),
            {"BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"},
        )
        self.assertIn("Sat", set(pdf_df["day"].unique()))
        matrix_sat = timetable.build_weekly_matrix(
            pdf_df.loc[pdf_df["section"] == "BCS-1A"].copy()
        )
        self.assertEqual(list(matrix_sat.columns), timetable.COMSATS_ALL_DAYS)

        combined_df = timetable.get_combined_comsats_catalog_df()
        self.assertTrue(
            {"BCS-5A", "BCS-5B", "BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"}.issubset(
                set(combined_df["section"].unique())
            )
        )

        sample_pdf_text = (
            "COMSATS University Islamabad - Department of Computer Science\n"
            "Fall 2026 - BCS - 1A\n"
            "Fall 2026 - BCT - 1\n"
            "BCS-3A CSC211 Mon 15:00-16:30 CL-9\n"
        )
        parsed_df = timetable.parse_comsats_timetable_pdf_text(sample_pdf_text)
        self.assertEqual(set(parsed_df["section"].unique()), {"BCS-1A", "BCT-1", "BCS-3A"})
        self.assertIn("BCS-1A", set(parsed_df["section"].unique()))
        self.assertIn("BCT-1", set(parsed_df["section"].unique()))
        self.assertFalse(parsed_df.loc[parsed_df["course_code"] == "HUM104"].empty)

        # Solve regular BCS-5A cohort + two conflicting remaining courses while detecting missing prerequisites
        report = timetable.solve_personalized_timetable(
            catalog_df=combined_df,
            cohort_section="BCS-5A",
            regular_courses=["CSC322", "CSC325", "CSC301"],
            extra_courses=["AIC354", "CSC101"],
            initial_sections={"AIC354": "BCS-5A", "CSC101": "BCS-1A"},
            completed_courses=["CSC101"],
        )
        self.assertFalse(report["initial_clashes_df"].empty)
        self.assertTrue(report["solved_zero_clash"])
        self.assertTrue(report["locked_cohort_preserved"])
        self.assertTrue(report["final_clashes_df"].empty)
        self.assertEqual(report["final_sections"]["CSC322"], "BCS-5A")
        self.assertEqual(report["final_sections"]["CSC325"], "BCS-5A")
        self.assertEqual(report["final_sections"]["CSC301"], "BCS-5A")
        prereq_by_code = {r["course_code"]: r for r in report["prerequisite_report"]}
        self.assertEqual(prereq_by_code["CSC322"]["status"], "ineligible")
        self.assertIn("CSC211", prereq_by_code["CSC322"]["missing"])
        self.assertEqual(prereq_by_code["CSC101"]["status"], "eligible")

        with (
            mock.patch.object(ai_engine, "_extract_pdf_text_for_timetable", return_value="Fall 2026 - BCS - 1A"),
            mock.patch.object(ai_engine, "is_api_configured", return_value=False),
        ):
            pdf_rows, used_ai = ai_engine.parse_timetable_pdf_with_fallback(b"%PDF-1.4")
            self.assertFalse(used_ai)
            self.assertGreaterEqual(len(pdf_rows), 9)
            self.assertTrue(all(r["section"] == "BCS-1A" for r in pdf_rows))

        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            adv_text, adv_fb = ai_engine.generate_timetable_advice_with_timeout(
                cohort_section="BCS-5A",
                clash_lines=["Mon 08:30–10:00 (CSC322 BCS-5A) vs (AIC354 BCS-5A)"],
                prereq_issues=[],
                swap_lines=["AIC354: BCS-5A → BCS-5B"],
                solved_zero_clash=True,
                locked_cohort_preserved=True,
            )
            self.assertTrue(adv_fb)
            self.assertIn("AIC354: BCS-5A → BCS-5B", adv_text)

    def test_pure_python_pdf_extraction_prereq_chains_all_classes_search_and_fpdf2_export(self) -> None:
        # 1. Sample COMSATS Fall 2026 PDF generation & pure-Python PDF parsing (zero .env)
        sample_pdf_bytes = timetable.generate_sample_comsats_pdf_bytes()
        self.assertTrue(sample_pdf_bytes.startswith(b"%PDF-"))
        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            parsed_rows, used_ai = ai_engine.parse_timetable_pdf_with_fallback(sample_pdf_bytes)
        self.assertFalse(used_ai)
        parsed_df = timetable.slots_list_to_df(parsed_rows)
        self.assertEqual(len(parsed_df), 54)
        self.assertEqual(
            set(parsed_df["section"].unique()),
            {"BCS-1A", "BCS-1B", "BSE-1", "BAI-1", "BDS-1", "BCT-1"},
        )

        # 2. Prerequisite chain impact analysis for failed & retake courses
        findings = timetable.analyze_prerequisite_chains(
            enrolled_courses=["CSC101", "CSC103", "CSC325"],
            failed_courses=["CSC101", "CSC103"],
            passed_courses=["HUM104"],
        )
        headlines = [f["headline"] for f in findings]
        self.assertTrue(any("CSC325: Prerequisite Blocker (CSC101)" in h for h in headlines))
        self.assertTrue(any("CSC103 Failed/Retake Chain Impact" in h for h in headlines))

        # 3. Search all other classes in uploaded PDF for a clashing repeat course
        bcs_1a = parsed_df.loc[parsed_df["section"] == "BCS-1A"].copy()
        other_than_103 = bcs_1a.loc[bcs_1a["course_code"] != "CSC103"].copy()
        alts = timetable.find_alternative_classes_in_pdf(
            course_code="CSC103",
            other_schedule_df=other_than_103,
            pdf_catalog_df=parsed_df,
            current_section="BCS-1B",
        )
        zero_clash_secs = {a["section"] for a in alts if not a["has_clash"]}
        clash_secs = {a["section"] for a in alts if a["has_clash"]}
        self.assertIn("BSE-1", zero_clash_secs)
        self.assertIn("BCS-1A", zero_clash_secs)
        self.assertIn("BCS-1B", clash_secs)

        # 4. Build final clash-free timetable (BCS-1A regular + CSC103 retake in BSE-1) & export via fpdf2
        bse1_103 = parsed_df.loc[(parsed_df["section"] == "BSE-1") & (parsed_df["course_code"] == "CSC103")]
        final_df = pd.concat([other_than_103, bse1_103], ignore_index=True)
        self.assertTrue(timetable.detect_clashes(final_df).empty)
        final_pdf = timetable.generate_final_timetable_pdf(
            cohort_label="BCS-1A (Fall 2026)",
            schedule_df=final_df,
            swaps_applied=["CSC103: BCS-1B -> BSE-1"],
            prereq_notes=["CSC103 Failed/Retake Chain Impact"],
        )
        self.assertTrue(final_pdf.startswith(b"%PDF-"))
        self.assertGreater(len(final_pdf), 1500)
        roundtrip_df = timetable.parse_timetable_pdf_bytes(final_pdf)
        self.assertGreaterEqual(len(roundtrip_df), len(final_df))

        # 5. Full CS Master Catalog (14 regular sections + RPT-CS), remaining clash check & smart regular swap
        master_df = timetable.get_cs_master_catalog_df()
        self.assertEqual(len(master_df["section"].unique()), 15)
        for sec in master_df["section"].unique():
            self.assertTrue(
                timetable.detect_clashes(master_df.loc[master_df["section"] == sec]).empty,
                f"Internal clash found in master section {sec}",
            )
        bcs5a_df = master_df.loc[master_df["section"] == "BCS-5A"].copy()
        bcs5a_courses = list(bcs5a_df["course_code"].unique())
        pw = timetable.check_failed_prereq_impacts_for_regular(
            regular_courses=bcs5a_courses,
            failed_courses=["CSC103", "CSC211"],
            catalog_df=master_df,
        )
        self.assertEqual([item["failed_code"] for item in pw], ["CSC103", "CSC211"])

        csc103_checks = timetable.analyze_remaining_course_sections("CSC103", bcs5a_df, master_df)
        self.assertIn("BDS-1", {x["section"] for x in csc103_checks if not x["has_clash"]})
        self.assertIn("RPT-CS", {x["section"] for x in csc103_checks if not x["has_clash"]})
        self.assertIn("BCS-1A", {x["section"] for x in csc103_checks if x["has_clash"]})

        eee_checks = timetable.analyze_remaining_course_sections("EEE241", bcs5a_df, master_df)
        self.assertTrue(all(x["has_clash"] for x in eee_checks))
        smart_swap = timetable.find_smart_regular_section_swap(
            remaining_code="EEE241",
            regular_courses=bcs5a_courses,
            regular_sections_map={c: "BCS-5A" for c in bcs5a_courses},
            master_catalog_df=master_df,
        )
        self.assertIsNotNone(smart_swap)
        self.assertEqual(smart_swap["remaining_target_section"], "BCS-2A")
        self.assertEqual(smart_swap["regular_swaps"][0]["course_code"], "CSC322")
        self.assertEqual(smart_swap["regular_swaps"][0]["to_section"], "BCS-5B")


class TestSeedingAndTimetablePersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "seed_test.db"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_four_subject_seed_legacy_cleanup_and_recommender(self) -> None:
        # Simulate legacy DB without touching filesystem deletion
        database.save_lecture(
            course_code="CSC301",
            lecture_title="Master Theorem & Recurrence Relations",
            summary_data={"executive_summary": "old", "structured_notes": [], "extracted_pseudocode": ""},
            chunks=[{"topic": "Old", "content": "old", "source_type": "AUDIO", "source_reference": "00:01"}],
            db_path=self.db_path,
        )
        database.seed_initial_data(db_path=self.db_path)
        database.seed_initial_data(db_path=self.db_path)  # Must be idempotent after upgrade

        lectures = database.list_lectures(db_path=self.db_path)
        self.assertEqual(len(lectures), 4)
        self.assertEqual([l["course_code"] for l in lectures], ["CSC322", "CSC325", "AIC354", "CSC301"])

        for lec in lectures:
            uncertain = database.get_lecture_chunks(lec["id"], only_uncertain=True, db_path=self.db_path)
            self.assertEqual(len(uncertain), 1, f"Expected 1 flagged item for {lec['course_code']}")
            mcqs = database.get_revision_items(lec["id"], "mcq", db_path=self.db_path)
            cards = database.get_revision_items(lec["id"], "flashcard", db_path=self.db_path)
            self.assertEqual(len(mcqs), 10, f"Expected 10 MCQs for {lec['course_code']}")
            self.assertEqual(len(cards), 6, f"Expected 6 Flashcards for {lec['course_code']}")

        slots = database.list_timetable_slots(db_path=self.db_path)
        self.assertEqual(len(slots), 8)
        self.assertTrue(all(s["section"] == "BCS-5A" for s in slots))
        self.assertTrue(all(s["instructor"] for s in slots))

        sched_df = timetable.slots_list_to_df(slots)
        clashes_df = timetable.detect_clashes(sched_df)
        self.assertEqual(len(clashes_df), 1)
        self.assertEqual({clashes_df.iloc[0]["course_a"], clashes_df.iloc[0]["course_b"]}, {"CSC322", "AIC354"})

        swaps = timetable.recommend_slot_swaps(sched_df)
        self.assertTrue(any(s["resolves_all_clashes"] for s in swaps))
        best = next(s for s in swaps if s["resolves_all_clashes"])
        database.replace_course_slots(
            str(best["course_code"]),
            list(best["replacement_slots"]),
            db_path=self.db_path,
        )
        updated_sched = timetable.slots_list_to_df(database.list_timetable_slots(db_path=self.db_path))
        self.assertTrue(timetable.detect_clashes(updated_sched).empty)


class TestStreamlitUI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "ui_test.db"
        self.prev_env = os.environ.get("CAMPUSFLOW_DB_PATH")
        os.environ["CAMPUSFLOW_DB_PATH"] = str(self.db_path)

    def tearDown(self) -> None:
        if self.prev_env is None:
            os.environ.pop("CAMPUSFLOW_DB_PATH", None)
        else:
            os.environ["CAMPUSFLOW_DB_PATH"] = self.prev_env
        self.tmp_dir.cleanup()

    def test_zero_emojis_and_full_width_css(self) -> None:
        allowed_ui_icons = {"✨", "⚡", "👍", "👎", "✅", "❌", "📄"}
        for rel in ("app.py", "timetable.py"):
            content = (CAMPUSFLOW_DIR / rel).read_text(encoding="utf-8")
            disallowed = [ch for ch in content if ord(ch) > 0x2600 and ch not in allowed_ui_icons]
            self.assertEqual(disallowed, [], f"Disallowed symbol chars found in {rel}: {disallowed}")
        app_src = (CAMPUSFLOW_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("max-width:1440px!important", app_src)

    def test_dynamic_four_subject_reactivity_no_dup_notes_wizard_and_schedule(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(CAMPUSFLOW_DIR / "app.py"), default_timeout=15).run()
        self.assertFalse(at.exception)
        self.assertEqual(
            [t.label for t in at.tabs],
            ["Capture", "Vault & Q&A", "Quiz & Flashcards", "Timetable & Clash"],
        )

        # Verify Capture Studio + 2 balanced right cards without Classroom Notes or Synthesized Lecture Preview
        tab1_md = "\n".join(m.value for m in at.tabs[0].markdown)
        tab2_md = "\n".join(m.value for m in at.tabs[1].markdown)
        self.assertEqual(len(at.tabs[0].text_area), 0)
        self.assertIn("Capture Studio", tab1_md)
        self.assertIn("Lecture Audio", tab1_md)
        self.assertIn("Whiteboard Photo", tab1_md)
        self.assertIn("Slides PDF", tab1_md)
        self.assertIn("Multimodal Capture Kit", tab1_md)
        self.assertIn("Capture Summary", tab1_md)
        self.assertIn("Inputs Active", tab1_md)
        self.assertIn("3 / 3", tab1_md)
        self.assertNotIn("Classroom Notes", tab1_md)
        self.assertNotIn("Synthesized Lecture Preview", tab1_md)
        self.assertNotIn("Needs Review", tab1_md)
        self.assertNotIn("cf-review-card", tab1_md)
        self.assertNotIn("#### Key Concepts", tab1_md)
        self.assertNotIn("#### Formulas, Code & Worked Example", tab1_md)
        self.assertNotIn("cf-notes-badge", tab1_md)
        self.assertNotIn("Notes Outline", tab1_md)
        self.assertNotIn("cf-notes-section", tab1_md)
        self.assertNotIn("cf-notes-h2", tab1_md)
        for m in at.markdown:
            for line in m.value.splitlines():
                self.assertFalse(
                    line.startswith("    ") and "<" in line,
                    f"Indented HTML line could leak as Markdown code block: {line!r}",
                )
        self.assertIn("cf-notes-badge", tab2_md)
        self.assertIn("Notes Outline", tab2_md)
        self.assertIn("cf-notes-section", tab2_md)
        self.assertIn("cf-notes-h2", tab2_md)
        self.assertIn("1.1", tab2_md)
        self.assertIn("#### Formulas, Code & Worked Example", tab2_md)
        self.assertIn("Key Takeaway", tab2_md)
        self.assertNotIn("Classroom note:", tab2_md)
        self.assertIn("Note Quality Feedback", tab2_md)
        self.assertIn("Loaded from Demo Kit", tab2_md)
        self.assertNotIn("Needs Review", tab2_md)
        self.assertNotIn("cf-review-card", tab2_md)
        self.assertNotIn("Confirm & Save", tab2_md)
        self.assertNotIn("Filter by Course", tab2_md)
        self.assertEqual(len(at.tabs[1].selectbox), 0)
        tab2_btn_keys = [getattr(b, "key", None) or "" for b in at.tabs[1].button]
        self.assertFalse(any(k.startswith("lib_pick_") for k in tab2_btn_keys))
        tab2_btn_labels = [b.label for b in at.tabs[1].button]
        self.assertIn("👍 Good Notes", tab2_btn_labels)
        self.assertIn("👎 Not Good — Improve & Regenerate", tab2_btn_labels)

        # Verify default CSC322 top status bar, buttons, and wizard
        all_md = "\n".join(m.value for m in at.markdown)
        self.assertIn("Fall 2026", all_md)
        self.assertIn("CSC322 — Operating Systems", all_md)
        self.assertIn("FCFS CPU Scheduling", all_md)
        self.assertIn("All Clear", all_md)
        self.assertIn("Question 1 of 10", all_md)

        btn_labels = [b.label for b in at.button]
        self.assertIn("Load CSC322 Sample Kit", btn_labels)
        self.assertIn("How was P2's missing burst time deduced from the Gantt chart?", btn_labels)

        # Switch Course in Tab 1 -> CSC325 updates Top Bar, Demo Kit, Sample Kit button, Tab 2 Auto-Sync, & Tab 3
        at.selectbox(key="cap_course_choice").select("CSC325 — Computer Organization & Assembly").run()
        self.assertFalse(at.exception)
        md_csc325 = "\n".join(m.value for m in at.markdown)
        tab2_csc325 = "\n".join(m.value for m in at.tabs[1].markdown)
        btns_csc325 = [b.label for b in at.button]
        self.assertIn("IEEE 754 Floating Point", md_csc325)
        self.assertIn("IEEE 754 Single-Precision Floating Point", tab2_csc325)
        self.assertIn("csc325_coal_audio.wav", md_csc325)
        self.assertIn("Load CSC325 Sample Kit", btns_csc325)
        self.assertIn("How is the 8-bit biased exponent for -12.625 calculated?", btns_csc325)
        self.assertIn("#### CSC325 — Computer Organization & Assembly · IEEE 754 Floating Point", md_csc325)

        # Switch Course in Tab 1 -> AIC354 updates Top Bar, Tab 1 Kit, Tab 2 Study Notes & Quick Questions, & Tab 3
        at.selectbox(key="cap_course_choice").select("AIC354 — Machine Learning Fundamentals").run()
        self.assertFalse(at.exception)
        md_aic354 = "\n".join(m.value for m in at.markdown)
        tab2_aic354 = "\n".join(m.value for m in at.tabs[1].markdown)
        btns_aic354 = [b.label for b in at.button]
        self.assertIn("Naive Bayes Classification", md_aic354)
        self.assertIn("Naive Bayes &amp; Bayesian Belief Classification", tab2_aic354)
        self.assertIn("aic354_ml_audio.wav", md_aic354)
        self.assertIn("Load AIC354 Sample Kit", btns_aic354)
        self.assertIn("Why do we add Vocabulary Size V in Laplace smoothing?", btns_aic354)

        # Switch Lecture in Tab 3 -> CSC301 updates Top Bar, Tab 1 Kit, & Tab 2
        at.selectbox(key="practice_lecture_select").select(
            "CSC301 — Design & Analysis of Algorithms · Binary Search & Recurrence"
        ).run()
        self.assertFalse(at.exception)
        md_csc301 = "\n".join(m.value for m in at.markdown)
        btns_csc301 = [b.label for b in at.button]
        self.assertIn("Binary Search & Recurrence", md_csc301)
        self.assertIn("csc301_daa_audio.wav", md_csc301)
        self.assertIn("Load CSC301 Sample Kit", btns_csc301)
        self.assertIn("Why is low + (high − low) ÷ 2 safer than averaging (low + high) ÷ 2?", btns_csc301)

        # Click a Quick Question in Tab 2 under offline local fallback -> grounded answer appears
        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            quick_btn = next(
                b
                for b in at.tabs[1].button
                if b.label == "Why is low + (high − low) ÷ 2 safer than averaging (low + high) ÷ 2?"
            )
            quick_btn.click().run()
            self.assertFalse(at.exception)
            tab2_after_q = "\n".join(m.value for m in at.tabs[1].markdown)
            self.assertIn("Clear Conversation", [b.label for b in at.tabs[1].button])
            self.assertIn("cf-source-pill", tab2_after_q)

        # Test Good Notes feedback persists verified badge in Vault & Q&A
        good_btn = next(b for b in at.tabs[1].button if b.label == "👍 Good Notes")
        good_btn.click().run()
        self.assertFalse(at.exception)
        tab2_after_good = "\n".join(m.value for m in at.tabs[1].markdown)
        self.assertIn("Feedback recorded — Notes marked as verified!", tab2_after_good)

        # Step through the 10-question MCQ wizard for active CSC301 lecture
        at.radio[0].set_value(0).run()
        self.assertFalse(at.exception)
        for step in range(9):
            next_btn = next(b for b in at.button if b.label == "Next Question")
            next_btn.click().run()
            self.assertFalse(at.exception)
            self.assertIn(f"Question {step + 2} of 10", "\n".join(m.value for m in at.markdown))

        finish_btn = next(b for b in at.button if b.label == "Finish & Evaluate Quiz")
        finish_btn.click().run()
        self.assertFalse(at.exception)
        eval_md = "\n".join(m.value for m in at.markdown)
        self.assertIn("Final Score", eval_md)
        self.assertIn("Accuracy", eval_md)

        retake_btn = next(b for b in at.button if b.label == "Retake Quiz")
        retake_btn.click().run()
        self.assertFalse(at.exception)
        self.assertIn("Question 1 of 10", "\n".join(m.value for m in at.markdown))

        # Switch to Flashcard Deck and flip card
        at.button_group[0].select("Flashcard Deck").run()
        self.assertFalse(at.exception)
        self.assertIn("Card 1 of 6", "\n".join(m.value for m in at.markdown))
        flip_btn = next(b for b in at.button if b.label == "Flip Card")
        flip_btn.click().run()
        self.assertFalse(at.exception)
        self.assertIn("Answer", "\n".join(m.value for m in at.markdown))

        # Schedule: 4-step CS Master Timetable, Remaining/Repeater Clash Solver, Smart Swap & PDF Export
        sample_pdf_bytes = timetable.generate_sample_comsats_pdf_bytes()
        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            # Step 1: Select BCS-5A -> regular courses auto-load without manual selection
            at.tabs[3].selectbox(key="t4_regular_section").select("BCS-5A").run()
            self.assertFalse(at.exception)
            tab4_reg_md = "\n".join(m.value for m in at.tabs[3].markdown)
            self.assertIn("Base Regular Weekly Timetable — BCS-5A", tab4_reg_md)
            self.assertIn("CSC322", tab4_reg_md)

            # Optional PDF Upload merges cleanly into CS Master Timetable
            at.tabs[3].file_uploader(key="t4_pdf_uploader").upload(
                "comsats_fall_2026.pdf", sample_pdf_bytes, "application/pdf"
            ).run()
            self.assertFalse(at.exception)
            self.assertTrue(
                any("Merged 54 PDF slots" in s.value for s in at.tabs[3].success)
            )

            # Step 2 & 3: Select failed/remaining course CSC103 -> Prereq Warning + Green BDS-1 Option
            at.tabs[3].multiselect(key="t4_remaining_courses").select("CSC103").run()
            self.assertFalse(at.exception)
            self.assertTrue(
                any("Prerequisite Warning: CSC103" in w.value for w in at.tabs[3].warning)
            )
            tab4_chk_md = "\n".join(m.value for m in at.tabs[3].markdown)
            self.assertIn("✅ NO CLASH", tab4_chk_md)
            self.assertIn("❌ CLASH", tab4_chk_md)

            bds_btn = next(
                b for b in at.tabs[3].button if b.label == "Select BDS-1 (No Clash)"
            )
            self.assertEqual(bds_btn.label, "Select BDS-1 (No Clash)")
            rpt_btn = next(
                b for b in at.tabs[3].button if b.label == "Select RPT-CS (No Clash)"
            )
            rpt_btn.click().run()
            self.assertFalse(at.exception)

            # Also pick EEE241 where all offerings clash -> Apply 1-click Smart Regular-Section Swap
            at.tabs[3].multiselect(key="t4_remaining_courses").select("EEE241").run()
            self.assertFalse(at.exception)
            swap_btn = next(
                b for b in at.tabs[3].button if b.label.startswith("Apply Combined Swap")
            )
            swap_btn.click().run()
            self.assertFalse(at.exception)

            tab4_final_md = "\n".join(m.value for m in at.tabs[3].markdown)
            self.assertIn("✅ CONFLICT-FREE (0 CLASHES)", tab4_final_md)
            self.assertIn("All Clear", "\n".join(m.value for m in at.markdown))
            dl_btns = [
                d
                for d in at.get("download_button")
                if d.label == "📄 Download Final Timetable (PDF)"
            ]
            self.assertEqual(len(dl_btns), 1)

            # Reset uploaded PDF catalog cleanly
            reset_pdf_btn = next(
                b for b in at.tabs[3].button if b.label == "Reset to Built-in Master"
            )
            reset_pdf_btn.click().run()
            self.assertFalse(at.exception)

    def test_instant_sample_kit_and_single_or_multi_file_process_lecture_in_ui(self) -> None:
        from streamlit.testing.v1 import AppTest

        with (
            mock.patch.object(ai_engine, "is_api_configured", return_value=False),
            mock.patch.object(
                ai_engine,
                "process_lecture_materials",
                side_effect=AssertionError("Sample Kit must not invoke live Gemini API"),
            ),
        ):
            at = AppTest.from_file(str(CAMPUSFLOW_DIR / "app.py"), default_timeout=15).run()
            self.assertFalse(at.exception)

            at.selectbox(key="cap_course_choice").select("CSC325 — Computer Organization & Assembly").run()
            self.assertFalse(at.exception)

            sample_btn = next(b for b in at.tabs[0].button if b.label == "Load CSC325 Sample Kit")
            sample_btn.click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Loaded CSC325 canonical sample kit" in s.value for s in at.success))

            tab1_after_kit = "\n".join(m.value for m in at.tabs[0].markdown)
            self.assertIn("Multimodal Capture Kit", tab1_after_kit)
            self.assertIn("Capture Summary", tab1_after_kit)
            self.assertIn("3 / 3", tab1_after_kit)
            self.assertNotIn("Synthesized Lecture Preview", tab1_after_kit)
            self.assertNotIn("Classroom Notes", tab1_after_kit)

            # Clicking Process Lecture with no files staged warns cleanly
            proc_btn = next(b for b in at.tabs[0].button if b.label == "Process Lecture")
            proc_btn.click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Upload at least one file" in w.value for w in at.tabs[0].warning))

            # Single-file upload (Slides PDF only) -> 1 / 3 Inputs Active
            at.text_input(key="cap_title").set_value("Memoization & DP").run()
            at.file_uploader(key="cap_pdf_files").upload(
                "dp_slides.pdf", b"%PDF-1.4 Dynamic Programming", "application/pdf"
            ).run()
            self.assertIn("Slides Parser (1 staged)", "\n".join(m.value for m in at.tabs[0].markdown))

            proc_btn = next(b for b in at.tabs[0].button if b.label == "Process Lecture")
            proc_btn.click().run()
            self.assertFalse(at.exception)
            self.assertTrue(
                any(
                    "CSC325" in s.value and "from Slides PDF" in s.value
                    for s in at.success
                )
            )

            tab1_after_pdf = "\n".join(m.value for m in at.tabs[0].markdown)
            tab2_after_pdf = "\n".join(m.value for m in at.tabs[1].markdown)
            self.assertIn("1 / 3", tab1_after_pdf)
            self.assertIn("✨ Processed from Uploaded Slides PDF", tab1_after_pdf)
            self.assertIn("Memoization &amp; DP", tab1_after_pdf)
            self.assertNotIn("cf-notes-badge", tab1_after_pdf)
            self.assertNotIn("cf-notes-section", tab1_after_pdf)
            self.assertIn("Memoization &amp; DP", tab2_after_pdf)
            self.assertIn("cf-notes-badge", tab2_after_pdf)
            self.assertIn("cf-notes-section", tab2_after_pdf)
            self.assertIn("cf-notes-h2", tab2_after_pdf)
            self.assertIn("SLIDES_PDF", tab2_after_pdf)
            self.assertNotIn("cf-src-board", tab2_after_pdf)
            self.assertNotIn("cf-src-audio", tab2_after_pdf)
            self.assertNotIn("Processed audio file", tab2_after_pdf)
            self.assertNotIn("Indexed board photo", tab2_after_pdf)

            # Clear PDF and stage 2 files (Audio + Whiteboard Photo) -> 2 / 3 Inputs Active
            at.file_uploader(key="cap_pdf_files").clear().run()
            at.text_input(key="cap_title").set_value("Pipelining Hazards").run()
            at.file_uploader(key="cap_audio_files").upload(
                "pipe_audio.wav", b"RIFF0000WAVE", "audio/wav"
            ).run()
            at.file_uploader(key="cap_image_files").upload(
                "pipe_board.png", b"\x89PNG\r\n\x1a\n", "image/png"
            ).run()
            proc_btn = next(b for b in at.tabs[0].button if b.label == "Process Lecture")
            proc_btn.click().run()
            self.assertFalse(at.exception)
            self.assertTrue(
                any(
                    "CSC325" in s.value and "from Audio, Board Photo" in s.value
                    for s in at.success
                )
            )
            tab1_after_two = "\n".join(m.value for m in at.tabs[0].markdown)
            tab2_after_two = "\n".join(m.value for m in at.tabs[1].markdown)
            self.assertIn("2 / 3", tab1_after_two)
            self.assertIn("✨ Processed from Uploaded Lecture Audio &amp; Whiteboard Photo", tab1_after_two)
            self.assertIn("Pipelining Hazards", tab1_after_two)
            self.assertIn("Pipelining Hazards", tab2_after_two)
            self.assertIn("AUDIO", tab2_after_two)
            self.assertIn("WHITEBOARD_IMAGE", tab2_after_two)
            self.assertNotIn("cf-src-slides", tab2_after_two)
            # Confirm course deduplication kept only 1 CSC325 row across repeat captures
            csc325_rows = [l for l in database.list_lectures(db_path=self.db_path) if l["course_code"] == "CSC325"]
            self.assertEqual(len(csc325_rows), 1)
            self.assertEqual(csc325_rows[0]["lecture_title"], "Pipelining Hazards")

    def test_custom_course_live_gemini_and_feedback_regeneration_ui(self) -> None:
        from streamlit.testing.v1 import AppTest

        live_syn = ai_engine.LectureSynthesisResult(
            detected_topic="REST APIs & JWT Auth",
            executive_summary=(
                "RESTful web architectures separate client and server state behind uniform resource identifiers, "
                "using stateless JWT bearer tokens for scalable verification."
            ),
            structured_notes=[
                ai_engine.StructuredNoteSection(
                    heading="Stateless REST Design & HTTP Semantics",
                    source_type="SLIDES_PDF",
                    subsections=[
                        ai_engine.NoteSubsection(
                            subheading="Statelessness & Horizontal Scaling",
                            body=(
                                "Every HTTP request in a RESTful API carries all authentication and context needed for the "
                                "server to execute independently.\n\n"
                                "Avoiding shared server-side session memory allows load balancers to scale replicas without sticky sessions."
                            ),
                            formula="Request = Method + URI + Bearer JWT + Payload",
                            bullets=[
                                "GET / PUT / DELETE — Idempotent operations repeating safely without duplicate mutations.",
                                "POST creates subordinate resources returning 201 Created.",
                            ],
                        ),
                        ai_engine.NoteSubsection(
                            subheading="JWT Bearer Flow & Order Creation Example",
                            body="Clients attach signed bearer credentials on each request cleanly.",
                            formula="Authorization: Bearer <header>.<payload>.<signature>",
                            bullets=[
                                "Client sends POST /api/orders -> server validates -> returns HTTP 201 + Location header.",
                            ],
                        ),
                    ],
                    takeaway="REST Core Rule: Statelessness + Uniform Interface -> Horizontal Scalability",
                )
            ],
            extracted_pseudocode="Authorization: Bearer <header>.<payload>.<signature>",
            knowledge_chunks=[
                ai_engine.KnowledgeChunkItem(
                    topic="Stateless REST Requests",
                    content="REST servers store zero client session state between HTTP requests.",
                    source_type="SLIDES_PDF",
                    source_reference="Slide 1",
                    is_uncertain=False,
                ),
                ai_engine.KnowledgeChunkItem(
                    topic="Idempotent HTTP Verbs",
                    content="GET, PUT, and DELETE are idempotent; POST creates new entities.",
                    source_type="SLIDES_PDF",
                    source_reference="Slide 2",
                    is_uncertain=False,
                ),
                ai_engine.KnowledgeChunkItem(
                    topic="JWT Structure",
                    content="JSON Web Tokens contain Header, Claims Payload, and HMAC/RSA Signature.",
                    source_type="SLIDES_PDF",
                    source_reference="Slide 3",
                    is_uncertain=False,
                ),
            ],
        )
        live_syn._model_used = "gemini-3.5-flash-lite"

        improved_syn = ai_engine.LectureSynthesisResult(
            detected_topic="REST APIs & JWT Auth",
            executive_summary=(
                "Regenerated REST API guide with clear HTTP status tables, deep statelessness rationale, and step-by-step JWT signature validation."
            ),
            structured_notes=[
                ai_engine.StructuredNoteSection(
                    heading="HTTP Status Codes & JWT Signing Deep Dive",
                    source_type="SLIDES_PDF",
                    subsections=[
                        ai_engine.NoteSubsection(
                            subheading="Authentication (401) vs Authorization (403)",
                            body=(
                                "Status codes communicate exact execution outcomes cleanly across distributed microservices.\n\n"
                                "Signing JWT payloads prevents tampering without extra database lookups per API call."
                            ),
                            formula="Signature = HMAC-SHA256(base64(Header) + '.' + base64(Payload), Secret)",
                            bullets=[
                                "401 Unauthorized — Missing or invalid token credentials.",
                                "403 Forbidden — Valid token lacking endpoint permissions.",
                            ],
                        ),
                        ai_engine.NoteSubsection(
                            subheading="Server-Side Token Verification Walkthrough",
                            body="Verify HMAC signatures before parsing user claims in request handlers.",
                            formula="",
                            bullets=[
                                "Decode JWT header.payload, compute HMAC with secret, and accept only exact matches.",
                                "200 OK / 201 Created confirm valid operations cleanly.",
                            ],
                        ),
                    ],
                    takeaway="Security Rule: Reject tokens with mismatched signatures immediately with HTTP 401.",
                )
            ],
            extracted_pseudocode="HTTP 200 / 201 / 401 / 403",
            knowledge_chunks=live_syn.knowledge_chunks,
        )
        improved_syn._model_used = "gemini-3.5-flash-lite"

        sample_mcq = ai_engine.MCQItem(
            question="Which HTTP status code indicates resource creation?",
            options=["200 OK", "201 Created", "304 Not Modified", "401 Unauthorized"],
            correct_index=1,
            explanation="POST creation responses return 201 Created.",
            source_badge="[SLIDES_PDF @ Slide 2]",
        )
        sample_fc = ai_engine.FlashcardItem(
            front="What are the 3 parts of a JWT?",
            back="Header, Payload, and Signature.",
            source_badge="[SLIDES_PDF @ Slide 3]",
        )
        live_pack = ai_engine.RevisionPack(mcqs=[sample_mcq] * 10, flashcards=[sample_fc] * 6)

        with (
            mock.patch.object(ai_engine, "is_api_configured", return_value=True),
            mock.patch.object(ai_engine, "process_lecture_materials", return_value=live_syn),
            mock.patch.object(ai_engine, "generate_revision_pack", return_value=live_pack),
            mock.patch.object(ai_engine, "regenerate_lecture_notes_with_feedback", return_value=improved_syn) as mock_regen,
        ):
            at = AppTest.from_file(str(CAMPUSFLOW_DIR / "app.py"), default_timeout=15).run()
            self.assertFalse(at.exception)

            at.selectbox(key="cap_course_choice").select("Custom Course / Any Subject...").run()
            self.assertFalse(at.exception)
            # Only Process Lecture shown for custom subjects (no Load Sample Kit button)
            self.assertFalse(any("Sample Kit" in b.label for b in at.tabs[0].button))

            at.text_input(key="cap_custom_course").set_value("CSC336 — Web Technologies").run()
            at.text_input(key="cap_title").set_value("REST APIs & JWT Auth").run()
            at.file_uploader(key="cap_pdf_files").upload(
                "rest_jwt_slides.pdf", b"%PDF-1.4 REST APIs and JWT Authentication", "application/pdf"
            ).run()

            proc_btn = next(b for b in at.tabs[0].button if b.label == "Process Lecture")
            proc_btn.click().run()
            self.assertFalse(at.exception)

            tab1_md = "\n".join(m.value for m in at.tabs[0].markdown)
            tab2_md = "\n".join(m.value for m in at.tabs[1].markdown)
            all_md = "\n".join(m.value for m in at.markdown)
            self.assertIn("CSC336 — Web Technologies", all_md)
            self.assertIn("REST APIs &amp; JWT Auth", tab1_md)
            self.assertIn("✨ Processed from Uploaded Slides PDF", tab1_md)
            self.assertIn("Generated Live via Gemini Flash API (gemini-3.5-flash-lite)", tab1_md)
            self.assertNotIn("cf-notes-badge", tab1_md)
            self.assertNotIn("cf-notes-section", tab1_md)
            self.assertIn("Generated Live via Gemini Flash API (gemini-3.5-flash-lite)", tab2_md)
            self.assertIn("cf-notes-badge", tab2_md)
            self.assertIn("cf-notes-section", tab2_md)
            self.assertIn("Statelessness &amp; Horizontal Scaling", tab2_md)
            self.assertIn("JWT Bearer Flow &amp; Order Creation Example", tab2_md)
            self.assertNotIn("Needs Review", tab2_md)

            # Trigger Negative Feedback -> Expand input -> Regenerate Notes with Gemini
            bad_btn = next(b for b in at.tabs[1].button if b.label == "👎 Not Good — Improve & Regenerate")
            bad_btn.click().run()
            self.assertFalse(at.exception)

            fb_inputs = [ti for ti in at.tabs[1].text_input if (getattr(ti, "key", "") or "").startswith("fb_text_")]
            self.assertEqual(len(fb_inputs), 1)
            fb_inputs[0].set_value("Add HTTP status codes and JWT signature walkthrough").run()

            regen_btn = next(b for b in at.tabs[1].button if b.label == "✨ Regenerate Notes with Gemini")
            regen_btn.click().run()
            self.assertFalse(at.exception)
            self.assertEqual(mock_regen.call_count, 1)

            tab2_after_regen = "\n".join(m.value for m in at.tabs[1].markdown)
            self.assertIn("HTTP Status Codes &amp; JWT Signing Deep Dive", tab2_after_regen)
            self.assertIn("Authentication (401) vs Authorization (403)", tab2_after_regen)
            self.assertIn("SLIDES_PDF", tab2_after_regen)
            self.assertNotIn("cf-src-board", tab2_after_regen)

    def test_strict_whiteboard_only_isolation_bayes_and_tab3_sync_in_ui(self) -> None:
        from PIL import Image
        import io
        from streamlit.testing.v1 import AppTest

        buf = io.BytesIO()
        Image.new("RGB", (64, 48), color=(30, 41, 59)).save(buf, format="PNG")
        webp_like_png_bytes = buf.getvalue()

        with mock.patch.object(ai_engine, "is_api_configured", return_value=False):
            at = AppTest.from_file(str(CAMPUSFLOW_DIR / "app.py"), default_timeout=15).run()
            self.assertFalse(at.exception)

            # Upload ONLY bayes.webp while default CSC322 course is selected
            at.file_uploader(key="cap_image_files").upload(
                "bayes.webp", webp_like_png_bytes, "image/webp"
            ).run()
            proc_btn = next(b for b in at.tabs[0].button if b.label == "Process Lecture")
            proc_btn.click().run()
            self.assertFalse(at.exception)

            tab1_md = "\n".join(m.value for m in at.tabs[0].markdown)
            tab2_md = "\n".join(m.value for m in at.tabs[1].markdown)
            tab3_md = "\n".join(m.value for m in at.tabs[2].markdown)
            self.assertIn("1 / 3", tab1_md)
            self.assertIn("✨ Processed from Uploaded Whiteboard Photo", tab1_md)
            self.assertIn("Bayes&#x27; Theorem", tab1_md)
            self.assertIn("Bayes&#x27; Theorem", tab2_md)
            self.assertIn("WHITEBOARD_IMAGE", tab2_md)
            self.assertNotIn("cf-src-slides", tab2_md)
            self.assertNotIn("cf-src-audio", tab2_md)
            self.assertNotIn("Classroom note:", tab2_md)
            self.assertIn("Bayes' Theorem", tab3_md)
            self.assertIn("Question 1 of 10", tab3_md)

            # Verify stored SQLite chunks are strictly WHITEBOARD_IMAGE only
            lectures = database.list_lectures(db_path=self.db_path)
            csc322_lec = next(l for l in lectures if l["course_code"] == "CSC322")
            chunks = database.get_lecture_chunks(csc322_lec["id"], db_path=self.db_path)
            self.assertGreaterEqual(len(chunks), 3)
            self.assertTrue(all(c["source_type"] == "WHITEBOARD_IMAGE" for c in chunks))
            mcqs = database.get_revision_items(csc322_lec["id"], "mcq", db_path=self.db_path)
            cards = database.get_revision_items(csc322_lec["id"], "flashcard", db_path=self.db_path)
            self.assertEqual(len(mcqs), 10)
            self.assertEqual(len(cards), 6)

            # Trigger Negative Feedback regeneration -> preserves WHITEBOARD_IMAGE & refreshes MCQs/Cards
            bad_btn = next(b for b in at.tabs[1].button if b.label == "👎 Not Good — Improve & Regenerate")
            bad_btn.click().run()
            self.assertFalse(at.exception)
            regen_btn = next(b for b in at.tabs[1].button if b.label == "✨ Regenerate Notes with Gemini")
            regen_btn.click().run()
            self.assertFalse(at.exception)

            refreshed_chunks = database.get_lecture_chunks(csc322_lec["id"], db_path=self.db_path)
            self.assertTrue(all(c["source_type"] == "WHITEBOARD_IMAGE" for c in refreshed_chunks))
            self.assertEqual(len(database.get_revision_items(csc322_lec["id"], "mcq", db_path=self.db_path)), 10)
            self.assertEqual(len(database.get_revision_items(csc322_lec["id"], "flashcard", db_path=self.db_path)), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
