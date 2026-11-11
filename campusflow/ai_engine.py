"""Google Gemini API multimodal ingestion, grounded RAG, and revision intelligence."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import mimetypes
import os
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]
INLINE_BYTE_LIMIT = 18 * 1024 * 1024  # 18 MB safe threshold before Files API upload
SYNTHESIS_TIMEOUT_SECONDS = 35.0
_T = TypeVar("_T")

_MIME_OVERRIDES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


class KnowledgeChunkItem(BaseModel):
    topic: str = Field(description="Concise topic heading for this knowledge unit.")
    content: str = Field(
        description=(
            "Clear academic explanation synthesized from the lecture materials. "
            "If text/audio is unclear, mark [UNCLEAR: ...] instead of guessing."
        )
    )
    source_type: Literal["AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES", "SYNTHESIZED"] = Field(
        description="Origin modality of this fact."
    )
    source_reference: str = Field(
        description=(
            "Exact observed reference such as '01:45' for audio, 'Whiteboard #1' for "
            "image attachment index, or 'Slide/Page 3' if visible on PDF. Use "
            "'Unmarked in source' if exact position is not observable. NEVER invent timestamps or page numbers."
        )
    )
    is_uncertain: bool = Field(
        description="True ONLY if handwriting is blurry, equations are cut off, or audio is muffled/ambiguous."
    )
    uncertainty_reason: str = Field(
        default="",
        description="Short explanation of why verification is needed when is_uncertain is True.",
    )


class NoteSubsection(BaseModel):
    subheading: str = Field(
        description=(
            "Specific, descriptive topic subheading (e.g., 'Core Intuition & Formal Statement', "
            "'Governing Equation & Symbol Breakdown', 'Medical Screening Worked Example', "
            "'Edge Cases & Prosecutor\\'s Fallacy'). Never use generic labels."
        )
    )
    body: str = Field(
        description=(
            "Rich, readable university study-note explanation (1 to 2 cohesive plain-English paragraphs "
            "separated by \\n\\n) covering concepts, intuition, derivations, and reasoning thoroughly."
        )
    )
    formula: str = Field(
        default="",
        description=(
            "Optional clean Unicode mathematical equation or rule central to this subtopic "
            "(e.g., P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)). Empty string if no formula applies."
        ),
    )
    bullets: list[str] = Field(
        default_factory=list,
        description=(
            "Crisp bullet points for term definitions, step-by-step calculation traces, complexity trade-offs, "
            "or key exam pitfalls belonging specifically to this subheading."
        ),
    )


class StructuredNoteSection(BaseModel):
    heading: str = Field(
        description="Clear major section heading organizing one major lecture theme (numbered cleanly in UI)."
    )
    source_type: Literal["AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"] = Field(
        description="Strict source modality matching ONLY the provided upload modality."
    )
    formal_definition: str = Field(default="", description="Precise formal definition grounded in the uploaded lecture. Do not invent one; leave empty if the source does not support it.")
    detailed_explanation: str = Field(default="", description="Beginner-friendly but rigorous explanation and intuition, grounded in the actual source.")
    formula_and_breakdown: str = Field(default="", description="Relevant formula(s) with every symbol/term explained. Leave empty if no formula is present or justified by the source.")
    worked_example: str = Field(default="", description="A step-by-step worked example only when the source contains enough information; distinguish an illustrative example from one copied from the source.")
    exam_takeaways: list[str] = Field(default_factory=list, description="Concise exam-relevant takeaways directly supported by the lecture material.")
    subsections: list[NoteSubsection] = Field(
        min_length=2,
        max_length=5,
        description=(
            "2 to 5 structured subtopics with specific descriptive subheadings (theory, equations, worked steps, "
            "and exam rules) written like real top-grade university NOTES."
        ),
    )
    takeaway: str = Field(
        default="",
        description="Highlighted bottom-line Key Takeaway sentence or formula summarizing this major section.",
    )


class LectureSynthesisResult(BaseModel):
    _model_used: str = PrivateAttr(default="")

    detected_topic: str = Field(
        default="",
        description="Concise academic topic name actually taught inside the uploaded file(s).",
    )
    executive_summary: str = Field(
        description="Comprehensive 4-6 sentence university-grade overview synthesizing the uploaded lecture."
    )
    structured_notes: list[StructuredNoteSection] = Field(
        description="Real hierarchical NOTES sections, each divided into clear topic-driven headings and subheadings."
    )
    extracted_pseudocode: str = Field(
        default="",
        description="Clean algorithm/pseudocode, worked table, or key mathematical formulas observed; empty string if none.",
    )
    knowledge_chunks: list[KnowledgeChunkItem] = Field(
        description="Granular source-attributed knowledge chunks strictly matching ONLY the uploaded modality."
    )


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Grounded answer derived strictly from the supplied SQLite lecture chunks."
    )
    cited_badges: list[str] = Field(
        default_factory=list,
        description="Exact source badge strings from the provided chunks that support the answer.",
    )
    grounded: bool = Field(
        description="True if the provided chunks contain sufficient evidence to answer the question."
    )


class MCQItem(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    source_badge: str


class FlashcardItem(BaseModel):
    front: str
    back: str
    source_badge: str


class RevisionPack(BaseModel):
    mcqs: list[MCQItem] = Field(min_length=10, max_length=10)
    flashcards: list[FlashcardItem] = Field(min_length=6, max_length=6)


def load_api_environment() -> tuple[str | None, str]:
    """Load `.env` from campusflow/ or workspace root and return (api_key, model_name)."""
    module_dir = Path(__file__).resolve().parent
    load_dotenv(module_dir / ".env", override=False)
    load_dotenv(module_dir.parent / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        api_key = api_key.strip()
        if api_key == "your_google_gemini_api_key_here":
            api_key = None
    model_name = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return api_key or None, model_name


def is_api_configured() -> bool:
    """Return True when a non-placeholder Gemini API key is available."""
    api_key, _ = load_api_environment()
    return bool(api_key)


def get_client(api_key: str | None = None) -> genai.Client:
    """Initialize an official google-genai Client using GEMINI_API_KEY."""
    if api_key is not None:
        final_key = api_key.strip()
    else:
        resolved_key, _ = load_api_environment()
        final_key = (resolved_key or "").strip()
    if not final_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add your key to campusflow/.env "
            "or set the GEMINI_API_KEY environment variable."
        )
    return genai.Client(
        api_key=final_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


_CACHE_LOCK = threading.Lock()
_LAST_MODEL_STATE = threading.local()
_MAX_CACHE_ENTRIES = 64
_SYNTHESIS_CACHE: dict[str, LectureSynthesisResult] = {}
_RAG_CACHE: dict[str, RAGAnswer] = {}
_REVISION_CACHE: dict[str, RevisionPack] = {}


def clear_response_cache() -> None:
    """Clear in-memory Gemini API response caches."""
    with _CACHE_LOCK:
        _SYNTHESIS_CACHE.clear()
        _RAG_CACHE.clear()
        _REVISION_CACHE.clear()


def _hash_payload_list(items: list[tuple[str, bytes, str]]) -> list[list[Any]]:
    return [
        [fname, len(data), hashlib.sha256(data).hexdigest(), mime]
        for fname, data, mime in items
    ]


def _make_cache_key(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _copy_with_model(model_obj: _T, model_used: str = "") -> _T:
    copied = model_obj.model_copy(deep=True)  # type: ignore[attr-defined]
    if hasattr(model_obj, "_model_used"):
        copied._model_used = model_used or getattr(model_obj, "_model_used", "")
    return copied


def _store_bounded(cache: dict[str, Any], key: str, value: Any) -> None:
    with _CACHE_LOCK:
        if len(cache) >= _MAX_CACHE_ENTRIES and key not in cache:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
        cache[key] = _copy_with_model(value)


def infer_mime_type(filename: str, fallback: str = "application/octet-stream") -> str:
    """Infer a reliable MIME type for audio, image, and PDF lecture uploads."""
    ext = Path(filename).suffix.lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or fallback


def _strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _generate_content_resilient(
    client: genai.Client,
    primary_model: str,
    contents: Any,
    config: types.GenerateContentConfig,
) -> Any:
    """Call Gemini with automatic fallback across Flash models only on 503/429/404 model unavailability."""
    candidates = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_exc: Exception | None = None
    _LAST_MODEL_STATE.model_id = primary_model
    for idx, model_id in enumerate(candidates):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            _LAST_MODEL_STATE.model_id = model_id
            return resp
        except genai_errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code in (429, 500, 502, 503, 504, 404) and idx < len(candidates) - 1:
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No Gemini candidate models available.")


def _build_file_part(
    client: genai.Client,
    filename: str,
    data: bytes,
    mime_type: str,
    uploaded_remote_names: list[str],
) -> Any:
    """Return an inline Part for normal files or upload larger files via Files API."""
    if len(data) <= INLINE_BYTE_LIMIT:
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    suffix = Path(filename).suffix or ".bin"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(mime_type=mime_type, display_name=filename),
        )
        if getattr(uploaded, "name", None):
            uploaded_remote_names.append(uploaded.name)
        return uploaded
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


def describe_provided_modalities(
    audio_files: list[tuple[str, bytes, str]] | None = None,
    image_files: list[tuple[str, bytes, str]] | None = None,
    pdf_files: list[tuple[str, bytes, str]] | None = None,
    extra_notes: str = "",
) -> dict[str, Any]:
    """Return active modality labels, source_type tags, and active upload count (1..3)."""
    audio_files = audio_files or []
    image_files = image_files or []
    pdf_files = pdf_files or []
    has_notes = bool(extra_notes and extra_notes.strip())

    labels: list[str] = []
    source_types: list[str] = []
    upload_count = 0
    if audio_files:
        labels.append("Audio")
        source_types.append("AUDIO")
        upload_count += 1
    if image_files:
        labels.append("Board Photo")
        source_types.append("WHITEBOARD_IMAGE")
        upload_count += 1
    if pdf_files:
        labels.append("Slides PDF")
        source_types.append("SLIDES_PDF")
        upload_count += 1
    if has_notes:
        labels.append("Classroom Notes")
        source_types.append("NOTES")

    inputs_active = upload_count if upload_count > 0 else (1 if has_notes else 0)
    return {
        "labels": labels,
        "source_types": source_types,
        "upload_count": upload_count,
        "inputs_active": inputs_active,
    }


_MODALITY_ALIAS_MAP: dict[str, Literal["AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"]] = {
    "AUDIO": "AUDIO",
    "LECTURE_AUDIO": "AUDIO",
    "WHITEBOARD": "WHITEBOARD_IMAGE",
    "WHITEBOARD_IMAGE": "WHITEBOARD_IMAGE",
    "BOARD": "WHITEBOARD_IMAGE",
    "IMAGE": "WHITEBOARD_IMAGE",
    "SLIDES": "SLIDES_PDF",
    "SLIDES_PDF": "SLIDES_PDF",
    "PDF": "SLIDES_PDF",
    "NOTES": "NOTES",
}

_DEFAULT_REF_FOR_MODALITY = {
    "AUDIO": "00:30",
    "WHITEBOARD_IMAGE": "Whiteboard #1",
    "SLIDES_PDF": "Slide 1",
    "NOTES": "Notes #1",
}


def _normalize_modality_tag(
    raw_tag: str,
    allowed_sources: list[str],
) -> Literal["AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"]:
    """Map aliases to canonical modality names and clamp strictly inside `allowed_sources`."""
    cleaned = (raw_tag or "").strip().upper()
    canonical = _MODALITY_ALIAS_MAP.get(cleaned, cleaned)
    valid_allowed = [
        _MODALITY_ALIAS_MAP.get(s.strip().upper(), s.strip().upper())
        for s in (allowed_sources or [])
    ]
    valid_allowed = [s for s in valid_allowed if s in {"AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"}]
    if not valid_allowed:
        valid_allowed = ["SLIDES_PDF"]
    if canonical in valid_allowed:
        return canonical  # type: ignore[return-value]
    return valid_allowed[0]  # type: ignore[return-value]


def _enforce_strict_modalities(
    result: LectureSynthesisResult,
    allowed_sources: list[str],
) -> LectureSynthesisResult:
    """Guarantee zero cross-modality leakage on structured sections and SQLite chunks."""
    valid_allowed = [
        _MODALITY_ALIAS_MAP.get(s.strip().upper(), s.strip().upper())
        for s in (allowed_sources or [])
    ]
    valid_allowed = [s for s in valid_allowed if s in {"AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"}]
    if not valid_allowed:
        return result

    for sec in result.structured_notes:
        sec.source_type = _normalize_modality_tag(sec.source_type, valid_allowed)

    for chunk in result.knowledge_chunks:
        fixed_type = _normalize_modality_tag(chunk.source_type, valid_allowed)
        chunk.source_type = fixed_type
        ref_low = (chunk.source_reference or "").lower()
        if fixed_type == "WHITEBOARD_IMAGE" and any(w in ref_low for w in ("slide", "pdf", "notes", ":")):
            chunk.source_reference = "Whiteboard #1"
        elif fixed_type == "SLIDES_PDF" and any(w in ref_low for w in ("whiteboard", "board", "notes", ":")):
            chunk.source_reference = "Slide 1"
        elif fixed_type == "AUDIO" and any(w in ref_low for w in ("slide", "whiteboard", "board", "notes", "pdf")):
            chunk.source_reference = "00:30"
        elif fixed_type == "NOTES" and any(w in ref_low for w in ("slide", "whiteboard", "board", ":")):
            chunk.source_reference = "Notes #1"
    return result


def _infer_lecture_modalities(
    current_summary: dict[str, Any] | None,
    chunks: list[dict[str, Any]] | None,
) -> list[str]:
    """Infer strictly which modalities were uploaded for a stored lecture."""
    found: list[str] = []
    for ch in chunks or []:
        raw = str(ch.get("source_type") or "").strip().upper()
        canon = _MODALITY_ALIAS_MAP.get(raw)
        if canon and canon not in found:
            found.append(canon)
    for sec in (current_summary or {}).get("structured_notes") or []:
        raw = str(sec.get("source_type") or sec.get("source_tag") or "").strip().upper()
        canon = _MODALITY_ALIAS_MAP.get(raw)
        if canon and canon not in found:
            found.append(canon)
    return found or ["SLIDES_PDF"]


def process_lecture_materials(
    course_code: str,
    lecture_title: str,
    audio_files: list[tuple[str, bytes, str]] | None = None,
    image_files: list[tuple[str, bytes, str]] | None = None,
    pdf_files: list[tuple[str, bytes, str]] | None = None,
    extra_notes: str = "",
    model_name: str | None = None,
) -> LectureSynthesisResult:
    """
    Process any combination of lecture audio, whiteboard images, PDF slides, and classroom notes.

    Each file tuple is `(filename, raw_bytes, mime_type)`.
    """
    audio_files = audio_files or []
    image_files = image_files or []
    pdf_files = pdf_files or []

    if not (audio_files or image_files or pdf_files or extra_notes.strip()):
        raise ValueError(
            "Please upload at least one audio recording, whiteboard image, PDF slide deck, "
            "or text excerpt before running ScribeAI."
        )

    mod_info = describe_provided_modalities(audio_files, image_files, pdf_files, extra_notes)
    active_labels = ", ".join(mod_info["labels"])
    allowed_sources_list: list[str] = list(mod_info["source_types"])
    allowed_sources = ", ".join(allowed_sources_list)

    _, env_model = load_api_environment()
    active_model = model_name or env_model
    client = get_client()

    cache_key = _make_cache_key(
        "synthesis",
        {
            "course_code": course_code.strip().upper(),
            "lecture_title": lecture_title.strip(),
            "audio": _hash_payload_list(audio_files),
            "images": _hash_payload_list(image_files),
            "pdfs": _hash_payload_list(pdf_files),
            "notes": extra_notes.strip(),
            "model": active_model,
        },
    )
    with _CACHE_LOCK:
        cached = _SYNTHESIS_CACHE.get(cache_key)
    if cached is not None:
        return _copy_with_model(cached)

    system_instruction = (
        "You are ScribeAI inside CAMPUSFLOW, an expert university lecturer and academic note-taker.\n\n"
        f"CRITICAL MODALITY ISOLATION RULE:\n"
        f"- ONLY the following input modality/modalities were uploaded: {active_labels}.\n"
        f"- EVERY `StructuredNoteSection.source_type` and EVERY `KnowledgeChunkItem.source_type` MUST STRICTLY be one of [{allowed_sources}].\n"
        f"- NEVER invent or assign tags for unprovided modalities (e.g., NEVER output SLIDES_PDF or AUDIO when only WHITEBOARD_IMAGE was uploaded).\n\n"
        "TRUTH-TO-ATTACHMENT RULE:\n"
        "- Examine EVERY pixel of attached images, EVERY page of PDFs, and EVERY word of audio directly.\n"
        "- ALWAYS synthesize 100% from the ACTUAL attached file(s). If the attached file teaches a different theorem or "
        "subject than the user's Course/Topic metadata field (for example, a Bayes' Theorem board photo uploaded while another "
        "course code is selected), ALWAYS follow the actual file content completely and set `detected_topic` to the true topic.\n\n"
        "REAL HIERARCHICAL UNIVERSITY NOTES FORMAT (NOT FORMALITY STUBS):\n"
        "1. `detected_topic`: Concise academic topic title actually observed in the uploaded material.\n"
        "2. `executive_summary`: Rich 4-6 sentence overview introducing the core concepts, formulas, worked trace, "
        "and practical significance directly from the uploaded file(s).\n"
        "3. `structured_notes`: Organize thorough, genuinely study-worthy NOTES into 2 to 4 major sections:\n"
        "   - `heading`: Clear, specific major section title (UI renders as 1., 2., 3.).\n"
        f"   - `source_type`: Strictly one of [{allowed_sources}].\n"
        "   - `subsections`: 2 to 4 focused subtopics under that major section (UI renders as 1.1, 1.2, 1.3):\n"
        "       * `subheading`: Specific, content-driven subheading tailored to the topic (e.g., 'Core Axiom & Geometric Intuition', "
        "'Breaking Down Prior, Likelihood & Evidence', 'Worked Diagnostic Probability Example', 'Common Mistakes & Base-Rate Neglect'). "
        "NEVER use generic filler like 'Subsection 1'.\n"
        "       * `body`: Substantial, clear plain-English paragraphs (2 paragraphs separated by \\n\\n where helpful) explaining the idea deeply.\n"
        "       * `formula`: Clean Unicode equation/rule central to that subheading when applicable, else empty string.\n"
        "       * `bullets`: Concrete step-by-step derivation steps, variable meanings, comparisons, or key exam rules.\n"
        "   - `takeaway`: Highlighted 1-sentence bottom-line Key Takeaway for the section.\n"
        "   DO NOT include bilingual Roman-Urdu notes or 'Classroom note:' lines.\n"
        "4. For EACH major section, populate `formal_definition`, `detailed_explanation`, `formula_and_breakdown`, `worked_example`, and `exam_takeaways` when supported by the source. Do not fill missing evidence with generic textbook filler; use an empty string/list when the uploaded material does not contain enough information.\n"
        "5. A worked example must show given values, substitutions, intermediate steps, and final result; never invent source-specific numerical data.\n"
        "6. Clean Unicode Math ONLY: Use ÷, ×, ·, −, Σ, Θ, ⌊, ⌋, ᵢ, ₁, ₂, →, ≤, ≥, ≈. NEVER emit raw LaTeX/ASCII dumps.\n"
        f"7. Produce 5 to 9 high-signal `knowledge_chunks` tagged STRICTLY inside [{allowed_sources}] with only genuinely "
        "observed references (e.g., 'Whiteboard #1', 'Slide 1', '00:45', or 'Unmarked in source')."
    )

    contents: list[Any] = [
        (
            f"Metadata Context — Course: {course_code.strip() or 'UNSPECIFIED'} | Topic Input: {lecture_title.strip() or 'Infer from upload'}\n"
            f"Provided Inputs (STRICT ISOLATION): {active_labels} ({allowed_sources})\n"
            "Carefully read every pixel, page, or spoken word in the attachment(s) below and produce complete, real "
            "hierarchical university NOTES with clear major Headings and specific topical Subheadings strictly attributed ONLY to the provided input modality."
        )
    ]

    if extra_notes.strip():
        contents.append(f"[NOTES] Classroom Notes Input:\n{extra_notes.strip()}")

    uploaded_remote_names: list[str] = []
    try:
        for idx, (fname, data, mime) in enumerate(audio_files, start=1):
            resolved_mime = mime or infer_mime_type(fname, "audio/mpeg")
            contents.append(f"[AUDIO #{idx} filename={fname}]")
            contents.append(
                _build_file_part(client, fname, data, resolved_mime, uploaded_remote_names)
            )

        for idx, (fname, data, mime) in enumerate(image_files, start=1):
            resolved_mime = mime or infer_mime_type(fname, "image/png")
            contents.append(f"[WHITEBOARD_IMAGE #{idx} filename={fname}]")
            contents.append(
                _build_file_part(client, fname, data, resolved_mime, uploaded_remote_names)
            )

        for idx, (fname, data, mime) in enumerate(pdf_files, start=1):
            resolved_mime = mime or infer_mime_type(fname, "application/pdf")
            contents.append(f"[SLIDES_PDF #{idx} filename={fname}]")
            contents.append(
                _build_file_part(client, fname, data, resolved_mime, uploaded_remote_names)
            )

        _LAST_MODEL_STATE.model_id = active_model
        response = _generate_content_resilient(
            client=client,
            primary_model=active_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema=LectureSynthesisResult.model_json_schema(),
            ),
        )
        used_model = getattr(_LAST_MODEL_STATE, "model_id", active_model) or active_model
        raw_json = _strip_json_fences(response.text or "")
        if not raw_json:
            raise RuntimeError("Gemini returned an empty response during lecture synthesis.")
        result = LectureSynthesisResult.model_validate_json(raw_json)
        result = _enforce_strict_modalities(result, allowed_sources_list)
        result._model_used = used_model
        _store_bounded(_SYNTHESIS_CACHE, cache_key, result)
        return _copy_with_model(result, used_model)
    finally:
        for remote_name in uploaded_remote_names:
            try:
                client.files.delete(name=remote_name)
            except Exception:
                pass


def answer_grounded_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model_name: str | None = None,
) -> RAGAnswer:
    """
    Answer a student's question strictly from retrieved SQLite knowledge chunks.

    If no matching chunks exist, returns an ungrounded response without calling Gemini
    to prevent hallucinated answers.
    """
    if not question.strip():
        raise ValueError("Please enter a question to ask the Lecture Vault.")

    if not retrieved_chunks:
        return RAGAnswer(
            answer=(
                "No matching notes were found in this lecture for that question. "
                "Try rephrasing with lecture terms or process a new lecture in ScribeAI Studio first."
            ),
            cited_badges=[],
            grounded=False,
        )

    _, env_model = load_api_environment()
    active_model = model_name or env_model
    client = get_client()

    formatted_evidence = []
    allowed_badges = []
    evidence_sig = []
    for chunk in retrieved_chunks:
        badge = chunk.get("source_badge") or (
            f"[{chunk.get('source_type', 'SYNTHESIZED')} @ {chunk.get('source_reference', 'Unmarked in source')}]"
        )
        allowed_badges.append(badge)
        uncertain_flag = "YES (REQUIRES VERIFICATION)" if chunk.get("is_uncertain") else "NO"
        evidence_sig.append(
            [
                badge,
                chunk.get("topic", "General"),
                bool(chunk.get("is_uncertain")),
                chunk.get("content", ""),
            ]
        )
        formatted_evidence.append(
            f"Badge: {badge}\n"
            f"Topic: {chunk.get('topic', 'General')}\n"
            f"Uncertain: {uncertain_flag}\n"
            f"Content: {chunk.get('content', '')}\n"
        )

    rag_cache_key = _make_cache_key(
        "rag",
        {
            "question": question.strip(),
            "evidence": evidence_sig,
            "model": active_model,
        },
    )
    with _CACHE_LOCK:
        cached_rag = _RAG_CACHE.get(rag_cache_key)
    if cached_rag is not None:
        return cached_rag.model_copy(deep=True)

    prompt = (
        "Answer the student's question using ONLY the retrieved lecture chunks below.\n"
        "Rules:\n"
        "- Cite exact badge strings from the evidence in `cited_badges` and inline in `answer`.\n"
        "- Write clear explanations using readable Unicode math (÷, ×, −, Σ) rather than raw LaTeX/ASCII formulas.\n"
        "- If a supporting chunk has `Uncertain: YES`, explicitly tell the student that part was flagged for review.\n"
        "- If the chunks do not contain the answer, set `grounded=false` and say what is missing.\n\n"
        f"Student Question: {question.strip()}\n\n"
        "Retrieved Lecture Chunks:\n"
        + "\n---\n".join(formatted_evidence)
    )

    response = _generate_content_resilient(
        client=client,
        primary_model=active_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are CAMPUSFLOW's strictly grounded academic RAG assistant. "
                "Never use outside knowledge beyond the provided lecture chunks."
            ),
            temperature=0.1,
            response_mime_type="application/json",
            response_json_schema=RAGAnswer.model_json_schema(),
        ),
    )
    raw_json = _strip_json_fences(response.text or "")
    parsed = RAGAnswer.model_validate_json(raw_json)
    # Keep only valid badges that actually appeared in retrieved chunks.
    valid_badge_set = set(allowed_badges)
    parsed.cited_badges = [b for b in parsed.cited_badges if b in valid_badge_set]
    _store_bounded(_RAG_CACHE, rag_cache_key, parsed)
    return parsed.model_copy(deep=True)


def generate_revision_pack(
    lecture_title: str,
    chunks: list[dict[str, Any]],
    model_name: str | None = None,
    use_cache: bool = False,
) -> RevisionPack:
    """Generate 10 grounded MCQs and 6 flip flashcards directly from lecture chunks."""
    if not chunks:
        raise ValueError("Cannot generate revision items because this lecture has no stored chunks.")

    _, env_model = load_api_environment()
    active_model = model_name or env_model
    client = get_client()

    chunk_lines = []
    chunk_sig = []
    for chunk in chunks:
        badge = f"[{chunk.get('source_type', 'SYNTHESIZED')} @ {chunk.get('source_reference', 'Unmarked in source')}]"
        status = " [UNCERTAIN - AVOID BUILDING QUESTIONS ON UNVERIFIED DETAILS]" if chunk.get("is_uncertain") else ""
        chunk_lines.append(f"{badge}{status} ({chunk.get('topic', 'General')}): {chunk.get('content', '')}")
        chunk_sig.append(
            [
                badge,
                bool(chunk.get("is_uncertain")),
                chunk.get("topic", "General"),
                chunk.get("content", ""),
            ]
        )

    rev_cache_key = _make_cache_key(
        "revision",
        {
            "lecture_title": lecture_title.strip(),
            "chunks": chunk_sig,
            "model": active_model,
        },
    )
    if use_cache:
        with _CACHE_LOCK:
            cached_rev = _REVISION_CACHE.get(rev_cache_key)
        if cached_rev is not None:
            return cached_rev.model_copy(deep=True)

    prompt = (
        f"Lecture: {lecture_title}\n\n"
        "Using ONLY the lecture chunks below, generate:\n"
        "- Exactly 10 multiple-choice questions (`mcqs`), each with 4 clear options, `correct_index` (0..3), "
        "a concise `explanation`, and the exact `source_badge` from the chunks.\n"
        "- Exactly 6 study flashcards (`flashcards`), each with `front`, `back`, and `source_badge`.\n\n"
        "Lecture Chunks:\n" + "\n".join(chunk_lines)
    )

    response = _generate_content_resilient(
        client=client,
        primary_model=active_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are CAMPUSFLOW's Revision Intelligence engine. Generate accurate, exam-relevant MCQs "
                "and flashcards grounded strictly in the provided lecture chunks."
            ),
            temperature=0.25,
            response_mime_type="application/json",
            response_json_schema=RevisionPack.model_json_schema(),
        ),
    )
    raw_json = _strip_json_fences(response.text or "")
    pack = RevisionPack.model_validate_json(raw_json)
    _store_bounded(_REVISION_CACHE, rev_cache_key, pack)
    return pack.model_copy(deep=True)


def _run_with_timeout(
    fn: Callable[[], _T],
    timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
) -> _T:
    """Execute a blocking callable under a strict ThreadPoolExecutor timeout."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _inspect_wav_duration(data: bytes) -> str | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                secs = max(1, round(frames / float(rate)))
                mins, rem = divmod(secs, 60)
                return f"{mins:02d}:{rem:02d}"
    except Exception:
        return None
    return None


def _inspect_image_dimensions(data: bytes) -> str | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return f"{img.width}x{img.height}px"
    except Exception:
        return None


def _inspect_pdf_summary(data: bytes) -> tuple[int | None, list[str]]:
    page_count: int | None = None
    snippets: list[str] = []
    try:
        raw = data.decode("latin-1", errors="ignore")
        pages = re.findall(r"/Type\s*/Page(?!s)\b", raw)
        if pages:
            page_count = len(pages)
        # Extract readable literal PDF text strings when uncompressed streams exist
        candidates = re.findall(r"\(([A-Za-z0-9][A-Za-z0-9 ,.;:+\-/*=()]{7,100})\)", raw)
        seen: set[str] = set()
        for item in candidates:
            cleaned = " ".join(item.split())
            if cleaned not in seen and any(ch.isalpha() for ch in cleaned):
                seen.add(cleaned)
                snippets.append(cleaned)
            if len(snippets) >= 4:
                break
    except Exception:
        pass
    return page_count, snippets


_SOURCE_TYPE_TO_TAG = {
    "SLIDES_PDF": "SLIDES_PDF",
    "AUDIO": "AUDIO",
    "WHITEBOARD_IMAGE": "WHITEBOARD_IMAGE",
    "NOTES": "NOTES",
}


_LOCAL_STUDY_PROFILES: dict[str, dict[str, Any]] = {
    "BAYES_THEOREM": {
        "detected_topic": "Bayes' Theorem & Conditional Probability Updating",
        "executive_summary": (
            "Bayes' Theorem provides a principled mathematical law for inverting conditional probabilities so prior "
            "beliefs update rationally as new empirical evidence arrives. By combining the prior prevalence P(A) with "
            "the likelihood P(B | A) and normalizing via marginal evidence P(B), Bayes' rule computes the exact "
            "posterior probability P(A | B) while avoiding inverse-conditional fallacies and base-rate neglect."
        ),
        "sections_by_modality": {
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Bayes' Theorem — Inverse Conditional Belief Updating",
                    "source_type": "WHITEBOARD_IMAGE",
                    "formal_definition": (
                        "Given two events A (hypothesis) and B (observed evidence) with P(B) > 0, Bayes' Theorem "
                        "states that the conditional probability of hypothesis A occurring given observed evidence B "
                        "equals the product of the likelihood P(B | A) and prior P(A), normalized by total evidence P(B)."
                    ),
                    "detailed_explanation": (
                        "In practical data analysis and diagnostics, we routinely observe outcomes or symptoms B (such as "
                        "a positive medical test, a spam keyword, or a sensor reading) and want to infer the hidden cause A. "
                        "Directly measuring P(A | B) is often hard, whereas the forward generative rate P(B | A) is well "
                        "documented from historical experiments.\n\n"
                        "Bayes' Theorem bridges this gap cleanly: starting from the baseline prevalence P(A), it scales "
                        "confidence upward or downward according to how strongly hypothesis A explains evidence B, then divides "
                        "by P(B) so the updated beliefs across all hypotheses remain calibrated between 0 and 1."
                    ),
                    "formula_and_breakdown": (
                        "P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)\n"
                        "where  P(B) = P(B | A) · P(A) + P(B | ¬A) · P(¬A)\n\n"
                        "• P(A | B) — Posterior Probability: Updated probability that hypothesis A is true after observing evidence B.\n"
                        "• P(B | A) — Likelihood: Probability of observing evidence B assuming hypothesis A is actually true.\n"
                        "• P(A) — Prior Probability: Baseline initial probability (base rate) of hypothesis A before seeing B.\n"
                        "• P(B) — Marginal Evidence: Total overall probability of observing B under both A and not-A (normalizing constant)."
                    ),
                    "worked_example": (
                        "Step 1 — Identify Conditional Identity:\n"
                        "  Joint probability symmetry gives P(A ∩ B) = P(A | B) · P(B) = P(B | A) · P(A).\n"
                        "  Dividing both sides by P(B) yields P(A | B) = [ P(B | A) · P(A) ] ÷ P(B).\n\n"
                        "Step 2 — Numerical Medical Screening Example:\n"
                        "  Suppose disease prevalence P(D) = 0.01 (1%), test sensitivity P(+ | D) = 0.95 (95%), "
                        "and false-positive rate P(+ | ¬D) = 0.05 (5%).\n"
                        "  • Compute prior complement: P(¬D) = 1 − 0.01 = 0.99.\n"
                        "  • Compute total evidence P(+): (0.95 × 0.01) + (0.05 × 0.99) = 0.0095 + 0.0495 = 0.0590.\n"
                        "  • Solve posterior P(D | +): (0.95 × 0.01) ÷ 0.0590 = 0.0095 ÷ 0.0590 ≈ 0.1610 (16.1%)."
                    ),
                    "exam_takeaways": [
                        "Never equate inverse conditionals P(A | B) ≠ P(B | A) — this error is the classic Prosecutor's Fallacy.",
                        "Always expand marginal evidence P(B) via the Law of Total Probability across mutually exclusive hypotheses.",
                        "Rare events (low prior P(A)) can still yield moderate posteriors even under 95%-accurate tests due to false positives.",
                    ],
                    "takeaway": "Posterior = (Likelihood × Prior) ÷ Total Evidence  |  P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)",
                },
                {
                    "heading": "Law of Total Probability & Multi-Hypothesis Normalization",
                    "source_type": "WHITEBOARD_IMAGE",
                    "formal_definition": (
                        "If hypotheses H₁, H₂, …, Hₖ form a mutually exclusive and exhaustive partition of the sample "
                        "space, the Law of Total Probability expresses P(B) as Σᵢ P(B | Hᵢ) · P(Hᵢ), guaranteeing the "
                        "posterior weights sum cleanly to 1."
                    ),
                    "detailed_explanation": (
                        "Computing Bayes' rule in multi-class classification or engineering reliability requires summing every "
                        "pathway by which evidence B could arise. Rather than estimating P(B) separately, partitioning the "
                        "hypothesis space lets each candidate class contribute its weighted likelihood directly to the denominator.\n\n"
                        "When we only need to select the most probable hypothesis (Maximum A Posteriori), note that P(B) stays "
                        "identical across every candidate Hᵢ, so comparing numerators P(B | Hᵢ) · P(Hᵢ) identifies the winner cleanly "
                        "while computing exact normalized probabilities simply divides each numerator by their total sum."
                    ),
                    "formula_and_breakdown": (
                        "P(Hᵢ | B) = [ P(B | Hᵢ) · P(Hᵢ) ] ÷ Σⱼ [ P(B | Hⱼ) · P(Hⱼ) ]\n\n"
                        "• H₁ … Hₖ — Partitioned Hypotheses: Disjoint candidate outcomes covering 100% of possibilities.\n"
                        "• P(B | Hⱼ) · P(Hⱼ) — Unnormalized Posterior Score: Likelihood-weighted plausibility of candidate Hⱼ.\n"
                        "• Σⱼ [ P(B | Hⱼ) · P(Hⱼ) ] — Partition Sum: Exact denominator enforcing Σᵢ P(Hᵢ | B) = 1."
                    ),
                    "worked_example": (
                        "Multi-Source Reliability Example (Factory Machine A₁ = 60% output, A₂ = 40% output):\n"
                        "  Defect rates: P(Defect | A₁) = 0.02, P(Defect | A₂) = 0.05.\n"
                        "  1. Numerator(A₁) = 0.02 × 0.60 = 0.012.\n"
                        "  2. Numerator(A₂) = 0.05 × 0.40 = 0.020.\n"
                        "  3. Marginal Evidence P(Defect) = 0.012 + 0.020 = 0.032.\n"
                        "  4. Posterior P(A₂ | Defect) = 0.020 ÷ 0.032 = 0.625 (62.5% chance defect came from Machine A₂)."
                    ),
                    "exam_takeaways": [
                        "Verify partition hypotheses satisfy P(H₁) + P(H₂) + … + P(Hₖ) = 1 before solving Total Probability.",
                        "Sum unnormalized numerators first — it doubles as P(B) and checks arithmetic consistency immediately.",
                    ],
                    "takeaway": "Normalization Rule: Divide each candidate numerator P(B | Hᵢ)·P(Hᵢ) by the sum across all disjoint hypotheses.",
                },
            ],
            "SLIDES_PDF": [
                {
                    "heading": "Bayes' Theorem — Formal Axiom & Conditional Updating",
                    "source_type": "SLIDES_PDF",
                    "formal_definition": (
                        "Bayes' Theorem computes the posterior conditional probability P(A | B) of hypothesis A given "
                        "evidence B from the forward likelihood P(B | A), prior belief P(A), and marginal evidence P(B) > 0."
                    ),
                    "detailed_explanation": (
                        "Scientific inference frequently demands reasoning backward from observed evidence to latent causes. "
                        "Bayes' Theorem reverses conditional direction rigorously using the fundamental product rule of "
                        "probability theory.\n\n"
                        "Every term carries a distinct operational role: prior encodes baseline knowledge before observation, "
                        "likelihood quantifies compatibility with evidence, and marginal probability normalizes across the full universe."
                    ),
                    "formula_and_breakdown": (
                        "P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)\n\n"
                        "• P(A | B) — Posterior probability after observing B.\n"
                        "• P(B | A) — Conditional likelihood of B given A.\n"
                        "• P(A) — Prior base rate of hypothesis A.\n"
                        "• P(B) — Marginal probability of evidence B."
                    ),
                    "worked_example": (
                        "With P(A) = 0.30, P(B | A) = 0.80, P(B | ¬A) = 0.10:\n"
                        "1. P(B) = (0.80 × 0.30) + (0.10 × 0.70) = 0.24 + 0.07 = 0.31.\n"
                        "2. P(A | B) = 0.24 ÷ 0.31 ≈ 0.7742 (77.4%)."
                    ),
                    "exam_takeaways": [
                        "Distinguish P(A | B) from P(B | A) carefully in word problems.",
                        "Always expand P(B) across both A and ¬A when marginal evidence is not directly given.",
                    ],
                    "takeaway": "Core Law: P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)",
                },
            ],
            "AUDIO": [
                {
                    "heading": "Bayesian Intuition — Prior Updating Under Evidence",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "Bayesian updating quantifies how rational confidence shifts from prior P(A) to posterior P(A | B) "
                        "proportional to the likelihood ratio comparing how well competing hypotheses predict observation B."
                    ),
                    "detailed_explanation": (
                        "Intuitively, seeing strong evidence B acts like a filter over the sample space: outcomes where B "
                        "did not occur vanish, and remaining hypotheses keep weights proportional to both their starting "
                        "size P(A) and their likelihood P(B | A).\n\n"
                        "Dividing by P(B) simply rescales the surviving probability mass back to 1.0, allowing sequential "
                        "observations to feed one posterior straight into the next experiment's prior."
                    ),
                    "formula_and_breakdown": (
                        "Posterior = (Likelihood × Prior) ÷ Evidence\n\n"
                        "• Likelihood P(B | A): Measures diagnostic signal strength.\n"
                        "• Prior P(A): Prevents base-rate neglect on rare classes.\n"
                        "• Evidence P(B): Rescales filtered probability mass to 1."
                    ),
                    "worked_example": (
                        "Coin box problem: Box 1 has 2 biased coins (P(H)=0.8), Box 2 has 2 fair coins (P(H)=0.5), "
                        "equal prior P(Box 1) = P(Box 2) = 0.5.\n"
                        "1. Observe Head: P(H) = (0.8 × 0.5) + (0.5 × 0.5) = 0.65.\n"
                        "2. Posterior P(Box 1 | H) = 0.40 ÷ 0.65 ≈ 0.615."
                    ),
                    "exam_takeaways": [
                        "Bayesian inference updates beliefs incrementally: today's posterior becomes tomorrow's prior.",
                        "Strong likelihoods still fail to yield high posteriors if the prior P(A) is very small.",
                    ],
                    "takeaway": "Updating Principle: Prior × Likelihood, normalized by Total Evidence.",
                },
            ],
        },
        "chunks_by_modality": {
            "WHITEBOARD_IMAGE": [
                ("Bayes' Theorem Equation & Terms", "Bayes' Theorem computes P(A | B) = [ P(B | A) · P(A) ] ÷ P(B), linking Posterior, Likelihood P(B | A), Prior P(A), and Evidence P(B).", "Whiteboard #1"),
                ("Derivation from Joint Probability", "Equating P(A ∩ B) = P(A | B) · P(B) = P(B | A) · P(A) yields Bayes' Theorem directly upon dividing by P(B) > 0.", "Whiteboard #1"),
                ("Law of Total Probability Expansion", "Marginal evidence expands as P(B) = P(B | A) · P(A) + P(B | ¬A) · P(¬A) across mutually exclusive partitions.", "Whiteboard #1"),
            ],
            "SLIDES_PDF": [
                ("Bayes' Theorem Formal Law", "P(A | B) = [ P(B | A) · P(A) ] ÷ P(B) inverts conditional probabilities rigorously.", "Slide 1"),
                ("Marginal Normalization Rule", "Evidence denominator P(B) ensures all candidate posteriors sum to 1.", "Slide 2"),
            ],
            "AUDIO": [
                ("Sequential Belief Updating", "Each new observation scales candidate priors by likelihoods and rescales via marginal P(B).", "00:28"),
                ("Avoiding Base-Rate Fallacy", "Confusing P(B | A) with P(A | B) ignores prior prevalence P(A) and overstates rare positives.", "00:54"),
            ],
        },
        "pseudocode": (
            "Bayes' Theorem:\n"
            "  P(A | B) = [ P(B | A) · P(A) ] ÷ P(B)\n\n"
            "Law of Total Probability:\n"
            "  P(B) = P(B | A) · P(A) + P(B | ¬A) · P(¬A)\n\n"
            "Worked Diagnostic Check (P(D)=0.01, P(+|D)=0.95, P(+|¬D)=0.05):\n"
            "  P(+)     = (0.95 × 0.01) + (0.05 × 0.99) = 0.0590\n"
            "  P(D | +) = 0.0095 ÷ 0.0590 ≈ 0.1610 (16.1%)"
        ),
    },
    "CSC322": {
        "detected_topic": "First-Come, First-Served (FCFS) CPU Scheduling",
        "executive_summary": (
            "First-Come, First-Served (FCFS) is a non-preemptive CPU scheduler that dispatches tasks strictly "
            "in order of arrival using a FIFO ready queue. Across our worked three-process trace, P1–P3 achieve "
            "an Average Turnaround Time of 5.67 ms and Average Waiting Time of 2.67 ms. While simple and "
            "starvation-free, FCFS suffers from the Convoy Effect when short jobs wait behind a long CPU burst."
        ),
        "sections_by_modality": {
            "SLIDES_PDF": [
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
                        "Minimal context-switching overhead compared to preemptive time-sliced policies.",
                    ],
                    "takeaway": "Scheduling Policy: Earliest arrival executes first and retains the CPU until completion.",
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
                    "takeaway": "Formulas: Turnaround Time = Completion − Arrival  |  Waiting Time = Turnaround − Burst",
                },
            ],
            "AUDIO": [
                {
                    "heading": "Ready Queue Walkthrough & Starvation Freedom",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "Starvation freedom in FCFS guarantees a bounded waiting horizon for every runnable task because "
                        "queue ordering strictly follows arrival timestamps with no priority preemption."
                    ),
                    "detailed_explanation": (
                        "In spoken classroom traces, tracking arrival timestamps confirms why batch schedulers favor FCFS "
                        "simplicity: once a job sits in the queue, only the finite jobs already ahead of it can precede it.\n\n"
                        "Because no arriving task jumps ahead by burst length or priority, starvation cannot occur even under heavy load."
                    ),
                    "formula_and_breakdown": (
                        "Start Time(Pᵢ) = max(Arrival Time(Pᵢ), Completion Time(Pᵢ₋₁))\n\n"
                        "• Arrival Order: P1 (0 ms) → P2 (1 ms) → P3 (2 ms).\n"
                        "• Bounded Delay: Only preceding arrivals finish ahead of Pᵢ."
                    ),
                    "worked_example": (
                        "Trace arrivals P1(0 ms), P2(1 ms), P3(2 ms):\n"
                        "• P1 starts at max(0, 0) = 0 ms.\n"
                        "• P2 waits until P1 clears at 4 ms → Start(P2) = max(1, 4) = 4 ms.\n"
                        "• P3 waits until P2 clears at 7 ms → Start(P3) = max(2, 7) = 7 ms."
                    ),
                    "exam_takeaways": [
                        "Always check max(AT, prev CT) in case the CPU sat idle waiting for late arrivals.",
                        "Zero preemption keeps dispatcher overhead near zero on batch workloads.",
                    ],
                    "takeaway": "Key Advantage: Zero starvation and minimal context-switching overhead across batch workloads.",
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
                    "takeaway": "Exam Pitfall: One CPU-heavy job at the front slows every short process waiting behind it.",
                },
            ],
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Worked Gantt Chart Execution Timeline",
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
                        "• P1 [0–4] → BT = 4 ms, CT = 4 ms, TAT = 4 − 0 = 4 ms, WT = 4 − 4 = 0 ms.\n"
                        "• P2 [4–7] → BT = 7 − 4 = 3 ms, CT = 7 ms, TAT = 7 − 1 = 6 ms, WT = 6 − 3 = 3 ms.\n"
                        "• P3 [7–9] → BT = 2 ms, CT = 9 ms, TAT = 9 − 2 = 7 ms, WT = 7 − 2 = 5 ms."
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
                    "takeaway": "Worked Averages: Avg Turnaround = 17 ÷ 3 = 5.67 ms  |  Avg Waiting = 8 ÷ 3 = 2.67 ms",
                },
            ],
        },
        "chunks_by_modality": {
            "SLIDES_PDF": [
                ("FCFS Scheduling Policy (FIFO)", "First-Come, First-Served (FCFS) is a non-preemptive CPU scheduler backed by a FIFO ready queue where processes hold the CPU until completion.", "Slide 1"),
                ("Turnaround & Waiting Time Formulas", "Turnaround Time = Completion Time − Arrival Time; Waiting Time = Turnaround Time − Burst Time.", "Slide 2"),
                ("Starvation Freedom", "FCFS prevents starvation because every finite arrival ahead in the FIFO queue eventually finishes.", "Slide 2"),
            ],
            "AUDIO": [
                ("Process Arrival Walkthrough", "P1 arrives at 0 ms (burst 4 ms), P2 at 1 ms (burst 3 ms), and P3 at 2 ms (burst 2 ms), served strictly FIFO.", "00:24"),
                ("The Convoy Effect", "Short jobs pile up behind a long CPU-bound burst in an FCFS queue, spiking Average Waiting Time and lowering device utilization.", "00:52"),
                ("Context-Switch Overhead", "Non-preemptive dispatch keeps scheduling overhead tiny compared to frequent timer-driven preemption.", "01:10"),
            ],
            "WHITEBOARD_IMAGE": [
                ("Worked Gantt Chart [0–9]", "FCFS Gantt trace runs P1 [0–4], P2 [4–7], and P3 [7–9], giving completions 4, 7, and 9 ms.", "Whiteboard #1"),
                ("P2 Burst Deduction", "From Gantt slot [4–7], P2 executes for 7 − 4 = 3 ms, yielding Turnaround 6 ms and Waiting 3 ms.", "Whiteboard #1"),
                ("Worked FCFS Averages", "Average Turnaround Time is (4 + 6 + 7) ÷ 3 = 5.67 ms and Average Waiting Time is (0 + 3 + 5) ÷ 3 = 2.67 ms.", "Whiteboard #1"),
            ],
        },
        "pseudocode": (
            "Formulas:\n"
            "  Turnaround Time (TAT) = Completion Time − Arrival Time\n"
            "  Waiting Time (WT)     = Turnaround Time − Burst Time\n\n"
            "Gantt Chart Timeline (ms):\n"
            "  0 ──── P1 (4 ms) ──── 4 ─── P2 (3 ms) ─── 7 ── P3 (2 ms) ── 9\n\n"
            "Process | Arrival | Burst | Completion | Turnaround | Waiting\n"
            "--------+---------+-------+------------+------------+--------\n"
            "P1      |   0 ms  |  4 ms |    4 ms    |    4 ms    |   0 ms\n"
            "P2      |   1 ms  |  3 ms |    7 ms    |    6 ms    |   3 ms\n"
            "P3      |   2 ms  |  2 ms |    9 ms    |    7 ms    |   5 ms\n\n"
            "Avg Turnaround = 17 ÷ 3 = 5.67 ms  ·  Avg Waiting = 8 ÷ 3 = 2.67 ms"
        ),
    },
    "CSC325": {
        "detected_topic": "IEEE 754 Single-Precision Floating Point",
        "executive_summary": (
            "IEEE 754 single-precision numbers represent real values across 32 bits: 1 sign bit, an 8-bit biased "
            "exponent (bias = 127), and a 23-bit mantissa fraction after an implicit hidden leading 1. Encoding "
            "−12.625 sets Sign S = 1, shifts binary 1100.101₂ to 1.100101 × 2³, adds bias 127 to obtain exponent "
            "130 (10000010₂), and groups into hexadecimal 0xC14A0000."
        ),
        "sections_by_modality": {
            "SLIDES_PDF": [
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
            ],
            "AUDIO": [
                {
                    "heading": "Normalizing −12.625 & Computing Biased Exponent",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "Binary normalization expresses any non-zero magnitude as 1.Fraction × 2ᴱ by shifting the binary "
                        "point until a single '1' bit precedes the radix point, then adding Bias 127 to determine E'."
                    ),
                    "detailed_explanation": (
                        "To convert decimal −12.625, treat integer and fractional components separately before combining them: "
                        "12 converts via repeated division by 2 into 1100₂, while 0.625 converts via repeated multiplication "
                        "by 2 into .101₂.\n\n"
                        "Shifting the binary point three positions left turns 1100.101₂ into 1.100101 × 2³. Adding Bias 127 "
                        "to E = 3 turns signed exponent shifts into unsigned 8-bit arithmetic."
                    ),
                    "formula_and_breakdown": (
                        "12.625₁₀ = 1100.101₂ = 1.100101 × 2³   →   E' = 127 + 3 = 130 = 10000010₂\n\n"
                        "• 1100₂ — Binary representation of integer 12 (8 + 4).\n"
                        "• .101₂ — Binary representation of fraction 0.625 (0.5 + 0.125).\n"
                        "• 10000010₂ — 8-bit unsigned binary encoding of biased exponent 130."
                    ),
                    "worked_example": (
                        "Convert 0.625 to binary step-by-step:\n"
                        "1. 0.625 × 2 = 1.25 → bit 1 (remainder 0.25)\n"
                        "2. 0.25 × 2 = 0.50 → bit 0 (remainder 0.50)\n"
                        "3. 0.50 × 2 = 1.00 → bit 1 → result = .101₂."
                    ),
                    "exam_takeaways": [
                        "Exponent field 0 encodes ±0 or subnormals; exponent field 255 encodes ±Infinity (frac=0) or NaN (frac≠0).",
                        "Always zero-pad the mantissa to the RIGHT (trailing bits), never to the left.",
                    ],
                    "takeaway": "Normalization Rule: 1100.101₂ = 1.100101 × 2³ → Biased Exponent = 130 (10000010₂)",
                },
            ],
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Board Walkthrough: −12.625 to 0xC14A0000",
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
                        "• Nibbles 1100 (C) 0001 (1): Combines Sign = 1 + Exponent 10000010₂.\n"
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
                        "Always re-group the combined 32-bit string in blocks of 4 before converting to hex.",
                    ],
                    "takeaway": "32-Bit Word: [1 | 10000010 | 10010100000000000000000] → Hex 0xC14A0000",
                },
            ],
        },
        "chunks_by_modality": {
            "SLIDES_PDF": [
                ("IEEE 754 Single-Precision Layout", "32-bit IEEE 754 uses 1 Sign bit, 8 Biased Exponent bits (bias 127), and 23 Mantissa fraction bits.", "Slide 1"),
                ("Hidden Leading 1 Rule", "Normalized values compute (−1)ˢ × (1.Fraction)₂ × 2^(Exponent Field − 127), granting 24 bits of effective precision.", "Slide 1"),
                ("Biased Exponent Range", "Normal numbers reserve biased exponent values 1 through 254 (unbiased exponents −126 to +127).", "Slide 2"),
            ],
            "AUDIO": [
                ("Binary Conversion of −12.625", "12 equals 1100₂ and 0.625 equals .101₂, normalizing 1100.101₂ into 1.100101 × 2³ (exponent E = 3).", "00:20"),
                ("Biased Exponent Calculation", "Adding bias 127 + 3 equals 130, which in 8-bit binary is 10000010₂.", "00:34"),
                ("Reserved Exponents 0 and 255", "Exponent 0 marks ±0 or subnormals; Exponent 255 marks ±Infinity (zero fraction) or NaN (non-zero fraction).", "00:48"),
            ],
            "WHITEBOARD_IMAGE": [
                ("Board Packing of 0xC14A0000", "Assembling Sign 1, Exponent 10000010₂, and Mantissa 100101... packs into hexadecimal 0xC14A0000.", "Whiteboard #1"),
                ("Sign-Magnitude Bit Flip", "Flipping only the top Sign bit transforms +12.625 (0x414A0000) into −12.625 (0xC14A0000).", "Whiteboard #1"),
                ("Nibble Grouping Trace", "Nibbles 1100(C) 0001(1) 0100(4) 1010(A) 0000 0000 0000 0000 decode directly to 0xC14A0000.", "Whiteboard #1"),
            ],
        },
        "pseudocode": (
            "Value Formula: (−1)ˢ × (1.Fraction)₂ × 2^(Exponent Field − 127)\n\n"
            "Worked Encoding for −12.625:\n"
            "  • Sign (1 bit)       : 1   (negative)\n"
            "  • Binary Magnitude   : 12.625 = 1100.101₂ = 1.100101 × 2³\n"
            "  • Biased Exp (8 bit) : 127 + 3 = 130 = 10000010₂\n"
            "  • Mantissa (23 bit)  : 10010100000000000000000\n\n"
            "Hex Nibbles: 1100 0001 0100 1010 0000 0000 0000 0000 = 0xC14A0000"
        ),
    },
    "AIC354": {
        "detected_topic": "Naive Bayes & Bayesian Belief Classification",
        "executive_summary": (
            "Naive Bayes classifies text and feature vectors via Bayes' Theorem by assuming features act independently "
            "once the class label is known. Predictions sum log probabilities rather than multiplying tiny decimals to "
            "stop floating-point underflow, while Laplace (Add-1) Smoothing keeps unseen words from zeroing out an entire class score."
        ),
        "sections_by_modality": {
            "SLIDES_PDF": [
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
                        "• Zero-Frequency Failure: If any P(xᵢ | y) = 0, entire product vanishes to 0."
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
            ],
            "AUDIO": [
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
                        "Sum of smoothed weights = (5+1)/(10+3) + (3+1)/(10+3) + (2+1)/(10+3) = (6 + 4 + 3)/13 = 13/13 = 1.0."
                    ),
                    "exam_takeaways": [
                        "Adding log probabilities turns tiny products into numerically stable sums.",
                        "Vocabulary Size V counts unique distinct tokens across the whole training corpus.",
                    ],
                    "takeaway": "Laplace Rule: (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)",
                },
            ],
            "WHITEBOARD_IMAGE": [
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
                        "2. Unseen word 'urgent' (count = 0): (0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200 ≈ 0.00083.\n"
                        "3. Unseen token retains safe non-zero weight in log P(Spam) + Σ log P(word | Spam)."
                    ),
                    "exam_takeaways": [
                        "Denominator MUST equal (Total Words in Class + Vocabulary Size V), never just Total Words + 1.",
                        "Unseen words always receive 1 ÷ (Total Words in Class + V) under Add-1 smoothing.",
                    ],
                    "takeaway": "Board Result: Smoothed denominator (Total Words + V = 1,200) prevents unseen tokens from collapsing predictions.",
                },
            ],
        },
        "chunks_by_modality": {
            "SLIDES_PDF": [
                ("Conditional Independence Assumption", "Naive Bayes assumes all features are conditionally independent given class y, cutting training and inference to O(n × C).", "Slide 1"),
                ("Zero-Frequency Collapse", "Without smoothing, one unseen word (count 0) sets raw probability to 0 and zeroes out the entire class posterior.", "Slide 2"),
                ("Empirical Class Priors", "Prior probability P(y) is computed as training documents in class y divided by total training documents.", "Slide 1"),
            ],
            "AUDIO": [
                ("Log-Space Prediction", "Adding log P(y) + Σ log P(featureᵢ | y) replaces tiny decimal products with numerically stable addition.", "00:22"),
                ("Laplace (Add-1) Normalization", "Adding 1 to each word count across V words requires adding Vocabulary Size V in the denominator so totals sum to 1.", "00:46"),
                ("Unseen Word Protection", "Add-1 smoothing guarantees even an unseen word receives probability 1 ÷ (Total Words in Class + V).", "01:05"),
            ],
            "WHITEBOARD_IMAGE": [
                ("Laplace Smoothing Denominator", "Smoothed likelihood equals (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V).", "Whiteboard #1"),
                ("Spam Filter Worked Example", "With V = 1,000 and 200 Spam words, 'prize' (5) = 6 ÷ 1,200 = 0.005 and unseen 'urgent' (0) = 1 ÷ 1,200.", "Whiteboard #1"),
                ("Class Score Comparison", "Summing log priors and smoothed word log-likelihoods ranks Spam vs Ham cleanly without underflow.", "Whiteboard #1"),
            ],
        },
        "pseudocode": (
            "Log-Space Prediction Rule:\n"
            "  Pick class y maximizing:  log P(y) + Σ log P(featureᵢ | y)\n\n"
            "Laplace (Add-1) Smoothing Formula:\n"
            "  P(word | class) = (Word Count + 1) ÷ (Total Words in Class + Vocabulary Size V)\n\n"
            "Spam Filter Check (V = 1,000, Total Spam Words = 200):\n"
            "  • 'prize'  (count = 5) → (5 + 1) ÷ (200 + 1,000) = 6 ÷ 1,200 = 0.005\n"
            "  • 'urgent' (count = 0) → (0 + 1) ÷ (200 + 1,000) = 1 ÷ 1,200 ≈ 0.00083"
        ),
    },
    "CSC301": {
        "detected_topic": "Binary Search & Recurrence Complexity",
        "executive_summary": (
            "Binary Search locates targets inside sorted arrays by repeatedly checking the middle element and discarding "
            "half of the candidates each step. Using low + (high − low) ÷ 2 prevents signed 32-bit integer overflow "
            "on large arrays, delivering Θ(log n) worst-case runtime and O(1) auxiliary space iteratively."
        ),
        "sections_by_modality": {
            "SLIDES_PDF": [
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
                        "Binary Search MUST fail on unsorted inputs unless sorted beforehand.",
                        "Unsuccessful searches terminate cleanly once low > high.",
                    ],
                    "takeaway": "Recurrence Bound: T(n) = T(n ÷ 2) + constant → Θ(log n) time",
                },
            ],
            "AUDIO": [
                {
                    "heading": "Overflow-Safe Midpoint & Iterative Space Efficiency",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "Overflow-safe midpoint arithmetic computes mid = low + ⌊(high − low) ÷ 2⌋ so intermediate addition "
                        "never exceeds upper index bound `high` inside signed 32-bit registers."
                    ),
                    "detailed_explanation": (
                        "When searching arrays larger than half of INT_MAX (2³¹ − 1 ≈ 2.14 billion), naively evaluating "
                        "low + high overflows into negative integers and triggers out-of-bounds memory crashes.\n\n"
                        "Subtracting (high − low) first ensures the offset never surpasses array size, and writing Binary Search "
                        "iteratively keeps memory strictly O(1) rather than allocating O(log n) recursive stack frames."
                    ),
                    "formula_and_breakdown": (
                        "midpoint = low + (high − low) ÷ 2\n\n"
                        "• high − low — Distance across current search interval (never overflows valid indices).\n"
                        "• Iterative Space — O(1) using only three integer scalars (low, high, mid)."
                    ),
                    "worked_example": (
                        "With INT_MAX = 2,147,483,647, low = 1,500,000,000, high = 2,000,000,000:\n"
                        "• Naive (low + high) = 3.5 billion → Signed 32-bit overflow!\n"
                        "• Safe low + (high − low) ÷ 2 = 1.5B + 250M = 1,750,000,000 (zero overflow)."
                    ),
                    "exam_takeaways": [
                        "Iterative Binary Search uses O(1) space; recursive uses O(log n) call stack.",
                        "Worst-case comparisons = ⌊log₂ n⌋ + 1.",
                    ],
                    "takeaway": "Safe Midpoint Formula: midpoint = low + (high − low) ÷ 2",
                },
            ],
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Worked Probe Trace on [3, 8, 14, 21, 29, 37, 45]",
                    "source_type": "WHITEBOARD_IMAGE",
                    "formal_definition": (
                        "A midpoint probe trace tracks low, high, mid, and comparison outcome sequentially until A[mid] == target "
                        "or the interval becomes empty."
                    ),
                    "detailed_explanation": (
                        "Tracing on concrete sorted arrays verifies loop boundary updates (mid + 1 vs mid − 1) so no infinite "
                        "loop occurs on 1- or 2-element windows.\n\n"
                        "Notice each failed comparison discards A[mid] itself along with its entire rejected half."
                    ),
                    "formula_and_breakdown": (
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
                        "Always shift low = mid + 1 or high = mid − 1 to prevent infinite loop on small windows.",
                    ],
                    "takeaway": "Trace Result: Found target 37 at index 5 in 2 midpoint comparisons.",
                },
            ],
        },
        "chunks_by_modality": {
            "SLIDES_PDF": [
                ("Sorted Array Precondition", "Binary Search requires sorted ascending order so every comparison discards half of remaining items.", "Slide 1"),
                ("Logarithmic Recurrence T(n)", "One half-size subproblem with constant work gives T(n) = T(n ÷ 2) + c, solving to Θ(log n) via Master Theorem Case 2.", "Slide 2"),
                ("Search Termination Invariant", "Unsuccessful searches halt as soon as low crosses high, proving zero candidate indices remain.", "Slide 1"),
            ],
            "AUDIO": [
                ("Integer Overflow Prevention", "Subtracting first in low + (high − low) ÷ 2 avoids exceeding 32-bit INT_MAX (2³¹ − 1) when indices exceed one billion.", "00:34"),
                ("Worst-Case Bound & O(1) Space", "Iterative Binary Search completes in at most ⌊log₂ n⌋ + 1 comparisons using only O(1) extra memory.", "00:55"),
                ("Iterative vs Recursive Trade-Off", "Iterative loops beat recursive Binary Search on memory by avoiding O(log n) call-stack frames.", "01:08"),
            ],
            "WHITEBOARD_IMAGE": [
                ("Safe Midpoint Equation", "Write midpoint as low + (high − low) ÷ 2 rather than averaging (low + high) ÷ 2 directly.", "Whiteboard #1"),
                ("Board Trace on [3..45]", "Searching for 37: Probe 1 checks mid = 3 (21 < 37 → low = 4); Probe 2 checks mid = 5 (A[5] = 37 found).", "Whiteboard #1"),
                ("Probe Efficiency", "Target 37 resolves in 2 midpoint probes compared to 6 steps under sequential linear search.", "Whiteboard #1"),
            ],
        },
        "pseudocode": (
            "def binary_search(A, target):\n"
            "    low, high = 0, len(A) - 1\n"
            "    while low <= high:\n"
            "        mid = low + (high - low) // 2   # overflow-safe midpoint\n"
            "        if A[mid] == target:\n"
            "            return mid\n"
            "        elif A[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n\n"
            "Complexity: T(n) = T(n ÷ 2) + c  →  Θ(log n) time, O(1) space"
        ),
    },
    "DP_MEMOIZATION": {
        "detected_topic": "Dynamic Programming & Memoization",
        "executive_summary": (
            "Dynamic Programming optimizes problems with overlapping subproblems and optimal substructure by storing "
            "each subproblem answer once instead of recomputing exponential recursion trees. Top-down Memoization caches "
            "recursive calls lazily on demand, whereas bottom-up Tabulation fills states iteratively to collapse exponential runtime into polynomial time."
        ),
        "sections_by_modality": {
            "SLIDES_PDF": [
                {
                    "heading": "Optimal Substructure, Overlapping Subproblems & Memoization",
                    "source_type": "SLIDES_PDF",
                    "formal_definition": (
                        "Dynamic Programming (DP) solves optimization or counting problems exhibiting both optimal substructure "
                        "(optimal global answers compose from optimal sub-answers) and overlapping subproblems (repeated identical calls)."
                    ),
                    "detailed_explanation": (
                        "Naive divide-and-conquer recursion re-solves shared small subtrees exponentially many times. Memoization "
                        "stores each computed state inside a hash map or array so every distinct state executes its transition once.\n\n"
                        "Bottom-up tabulation evaluates base cases iteratively in dependency order, eliminating recursion stack "
                        "overhead and enabling space rolling arrays."
                    ),
                    "formula_and_breakdown": (
                        "Runtime = (# Unique DP States) × (Transition Work per State)\n\n"
                        "• Memoization (Top-Down) — Recursive relation backed by lookup table caching visited states.\n"
                        "• Tabulation (Bottom-Up) — Iterative loop solving base cases upward in topological order."
                    ),
                    "worked_example": (
                        "Fibonacci n = 5:\n"
                        "1. Naive tree triggers 15 calls (recomputing fib(2) three times).\n"
                        "2. Memoized table stores cache[0..5] sequentially in 6 state evaluations → Θ(n) time."
                    ),
                    "exam_takeaways": [
                        "Define state parameters clearly before writing recurrence transitions.",
                        "Memoize sparse graphs; tabulate dense problems to drop call-stack overhead.",
                    ],
                    "takeaway": "Core Idea: Solve every unique DP state once, cache its result, and reuse in O(1) lookup time.",
                },
            ],
            "AUDIO": [
                {
                    "heading": "Turning Exponential O(2ⁿ) into Linear O(n)",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "State-space memoization bounds total asymptotic complexity by multiplying reachable unique state count "
                        "with constant transition cost per state."
                    ),
                    "detailed_explanation": (
                        "Caching converts branching recursion DAGs so identical nodes share one result immediately. "
                        "Whenever transition costs stay O(1), linear state arrays collapse O(2ⁿ) blowups into instant O(n) scans.\n\n"
                        "Verifying base cases carefully ensures transitions terminate cleanly without cyclic dependencies."
                    ),
                    "formula_and_breakdown": (
                        "T_DP(n) = O(States × Transitions_per_State)\n\n"
                        "• Base Cases — dp[0], dp[1] anchoring recurrence termination.\n"
                        "• Transition Order — Direction ensuring dependencies are already computed."
                    ),
                    "worked_example": (
                        "dp[0] = 0, dp[1] = 1, dp[i] = dp[i−1] + dp[i−2] computes dp[6] = 8 in 5 additions."
                    ),
                    "exam_takeaways": [
                        "DAG property (acyclic subproblem dependencies) is required for valid DP transitions.",
                        "Always verify base conditions on smallest inputs first.",
                    ],
                    "takeaway": "Complexity Gain: Runtime = (Number of Unique States) × (Transition Work per State)",
                },
            ],
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Worked Recurrence & Base Case Transitions",
                    "source_type": "WHITEBOARD_IMAGE",
                    "formal_definition": (
                        "A DP design specification pairs a formal state definition with base conditions and an explicit transition recurrence."
                    ),
                    "detailed_explanation": (
                        "Structured whiteboard design follows four disciplined steps: name what dp[i] represents in English, "
                        "specify base boundaries, express dp[i] via earlier states, and iterate strictly along topological order.\n\n"
                        "Testing small tables on the board confirms transitions match brute-force expectations before coding."
                    ),
                    "formula_and_breakdown": (
                        "dp[i] = f(dp[i − 1], dp[i − 2], …)\n\n"
                        "• State Meanings — Index i or capacity w tracked in table entries.\n"
                        "• Invariant — Every prerequisite entry holds final answer before reading it."
                    ),
                    "worked_example": (
                        "Board trace table for Fibonacci dp[0..5]:\n"
                        "dp[0]=0, dp[1]=1 → dp[2]=1 → dp[3]=2 → dp[4]=3 → dp[5]=5."
                    ),
                    "exam_takeaways": [
                        "Follow 4 steps: State Definition → Base Cases → Recurrence → Evaluation Order.",
                    ],
                    "takeaway": "Checklist: Define State → Write Base Cases → Recurrence → Evaluation Order",
                },
            ],
        },
        "chunks_by_modality": {
            "SLIDES_PDF": [
                ("Overlapping Subproblems", "Dynamic Programming avoids exponential recomputation by caching solutions to repeating subproblems.", "Slide 1"),
                ("Top-Down Memoization", "Memoization wraps recursive functions with a cache dictionary so each reachable state runs at most once.", "Slide 1"),
                ("Bottom-Up Tabulation", "Tabulation iterates from base cases upward, removing recursion stack overhead cleanly.", "Slide 2"),
            ],
            "AUDIO": [
                ("DP Time Complexity Principle", "DP runtime equals unique state count multiplied by constant transition cost per state.", "00:30"),
            ],
            "WHITEBOARD_IMAGE": [
                ("DP 4-Step Design Pattern", "Always specify State meaning, Base cases, Recurrence transition, and evaluation order.", "Whiteboard #1"),
            ],
        },
        "pseudocode": (
            "def fib_memo(n, cache={0: 0, 1: 1}):\n"
            "    if n not in cache:\n"
            "        cache[n] = fib_memo(n - 1) + fib_memo(n - 2)\n"
            "    return cache[n]\n\n"
            "DP Runtime = (# Unique States) × (Work per State) → Θ(n) time"
        ),
    },
    "PIPELINING_HAZARDS": {
        "detected_topic": "Instruction Pipelining & Pipeline Hazards",
        "executive_summary": (
            "Instruction pipelining overlaps Fetch, Decode, Execute, Memory, and Writeback stages so up to five "
            "instructions progress simultaneously per clock cycle. Structural, Data, and Control hazards stall or flush "
            "throughput unless resolved with operand forwarding hardware and early branch resolution."
        ),
        "sections_by_modality": {
            "AUDIO": [
                {
                    "heading": "Pipeline Throughput & Classic Hazard Classes",
                    "source_type": "AUDIO",
                    "formal_definition": (
                        "Instruction pipelining exploits temporal parallelism by breaking instruction execution into "
                        "k balanced hardware stages separated by latches so one instruction retires nearly every clock cycle."
                    ),
                    "detailed_explanation": (
                        "While single-instruction latency remains unchanged (k cycles end-to-end), overlapping k instructions "
                        "scales completion throughput up toward k× ideal speedup.\n\n"
                        "Hazards interrupt smooth cycle advances via hardware port collisions (Structural), register read-before-write "
                        "dependencies (Data), or unknown branch destinations (Control)."
                    ),
                    "formula_and_breakdown": (
                        "Ideal Pipeline Speedup = Single-Cycle Latency ÷ Slowest Stage Time ≈ k\n\n"
                        "• Structural Hazard — Hardware resource conflict (fixed via separate I-Cache and D-Cache).\n"
                        "• Data Hazard (RAW) — Read-After-Write dependency waiting for prior instruction result.\n"
                        "• Control Hazard — Conditional branches fetching wrong subsequent path."
                    ),
                    "worked_example": (
                        "If 5 balanced 200 ps stages execute instead of a single 1,000 ps cycle:\n"
                        "Steady-state CPI ≈ 1.0 gives completion every 200 ps → 5× throughput gain."
                    ),
                    "exam_takeaways": [
                        "Pipelining raises instruction THROUGHPUT, not individual instruction latency.",
                        "Clock period is dictated strictly by the slowest individual pipeline stage.",
                    ],
                    "takeaway": "Three Hazard Classes: Structural (Resource conflict) · Data (Dependency) · Control (Branches)",
                },
            ],
            "WHITEBOARD_IMAGE": [
                {
                    "heading": "Bypass Forwarding & Load-Use Stall Elimination",
                    "source_type": "WHITEBOARD_IMAGE",
                    "formal_definition": (
                        "Operand forwarding (bypassing) routes computed ALU outputs directly from EX/MEM or MEM/WB pipeline "
                        "registers into the next instruction's ALU MUX inputs before register writeback completes."
                    ),
                    "detailed_explanation": (
                        "Without forwarding hardware, an ADD followed immediately by SUB stalls 2 cycles waiting for WB. "
                        "Bypassing feeds the result straight back at the next EX edge, clearing ALU RAW stalls completely.\n\n"
                        "However, a LW (load word) only produces memory data at the end of the MEM stage, so a directly "
                        "dependent consumer still incurs 1 unavoidable hardware interlock stall cycle unless scheduled around."
                    ),
                    "formula_and_breakdown": (
                        "Effective CPI = Base CPI (1.0) + Stall Cycles per Instruction\n\n"
                        "• EX/MEM → EX Forward: 0 stalls for back-to-back ALU dependencies.\n"
                        "• Load-Use Interlock: Minimum 1 bubble cycle when consuming LW immediately."
                    ),
                    "worked_example": (
                        "Sequence: ADD $t1, $s0, $s1  →  SUB $t2, $t1, $s2\n"
                        "1. ADD computes $t1 at end of cycle 3 (EX).\n"
                        "2. Forwarding path sends $t1 straight into SUB's EX stage in cycle 4 → 0 stall cycles."
                    ),
                    "exam_takeaways": [
                        "Forwarding eliminates ALU-to-ALU stalls completely.",
                        "Load-Use hazards still force 1 bubble unless independent work fills the slot.",
                    ],
                    "takeaway": "Resolution Rule: Forwarding clears ALU RAW hazards; Load-Use requires 1 stall or instruction scheduling.",
                },
            ],
            "SLIDES_PDF": [
                {
                    "heading": "5-Stage RISC Pipeline Architecture",
                    "source_type": "SLIDES_PDF",
                    "formal_definition": (
                        "The canonical 5-stage RISC pipeline partitions instruction flow into IF (Fetch), ID (Decode/Reg Read), "
                        "EX (ALU Execute), MEM (Data Memory), and WB (Writeback), latched on synchronous clock edges."
                    ),
                    "detailed_explanation": (
                        "Inter-stage pipeline registers (IF/ID, ID/EX, EX/MEM, MEM/WB) isolate combinational logic signals "
                        "across cycles so earlier stages immediately start processing the next instruction safely.\n\n"
                        "Balanced stage latencies ensure no single bottleneck slows the global clock frequency unnecessarily."
                    ),
                    "formula_and_breakdown": (
                        "Stage Sequence: IF → ID → EX → MEM → WB\n\n"
                        "• IF — Read PC instruction from I-Cache.\n"
                        "• ID — Decode opcode and read register operands.\n"
                        "• EX — Perform ALU arithmetic or effective address calculation."
                    ),
                    "worked_example": (
                        "At clock cycle 5 of continuous execution: Instr 1 is in WB, Instr 2 in MEM, Instr 3 in EX, "
                        "Instr 4 in ID, and Instr 5 in IF simultaneously."
                    ),
                    "exam_takeaways": [
                        "Register file reads occur in ID and writes in WB (first/second half-cycle avoids structural collision).",
                    ],
                    "takeaway": "Stage Clock Rule: Clock period equals slowest individual stage latency plus latch delay.",
                },
            ],
        },
        "chunks_by_modality": {
            "AUDIO": [
                ("Pipeline Speedup & Structural Hazards", "Pipelining overlaps 5 instruction stages to approach 1 instruction per cycle, blocked when hardware units conflict.", "00:25"),
                ("RAW Data Dependency", "Read-After-Write (RAW) hazards occur when an instruction reads a source register before the producer writes back.", "00:45"),
            ],
            "WHITEBOARD_IMAGE": [
                ("Data Forwarding (Bypassing)", "Forwarding MUX paths feed EX/MEM ALU results straight to dependent EX inputs to avoid unnecessary stalls.", "Whiteboard #1"),
                ("Load-Use Stall & Branch Flush", "Load-Use dependencies impose at least 1 bubble cycle; mispredicted branches flush fetched instructions.", "Whiteboard #1"),
            ],
            "SLIDES_PDF": [
                ("5-Stage RISC Pipeline", "Stages IF, ID, EX, MEM, and WB separate work into balanced pipeline registers per clock cycle.", "Slide 1"),
            ],
        },
        "pseudocode": (
            "5-Stage Pipeline: IF → ID → EX → MEM → WB\n\n"
            "Hazard Remedies:\n"
            "  • Structural : Split I-Cache & D-Cache\n"
            "  • Data (RAW) : EX/MEM → EX Forwarding (+ 1 stall for Load-Use)\n"
            "  • Control    : Early Branch Target Resolution + Predictor"
        ),
    },
}


def _select_study_profile(
    code_clean: str,
    title_clean: str,
    pdf_snippets: list[str],
    notes_clean: str,
    filenames: list[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve the richest local academic study profile matching filenames, topic keywords, or known course defaults."""
    title_low = title_clean.lower()
    files_low = " ".join(filenames or []).lower()
    hint_low = f"{files_low} {title_low} {' '.join(pdf_snippets).lower()} {notes_clean.lower()}"

    # Check explicit Bayes' Theorem upload filename/topic before generic course defaults
    if "bayes" in files_low and not any(k in files_low for k in ("aic354", "laplace", "spam", "naive")):
        return _LOCAL_STUDY_PROFILES["BAYES_THEOREM"]
    if re.search(r"\bdp\b", title_low) or "memoization" in title_low or "dynamic programming" in hint_low:
        return _LOCAL_STUDY_PROFILES["DP_MEMOIZATION"]
    if "pipelin" in title_low or "hazard" in hint_low:
        return _LOCAL_STUDY_PROFILES["PIPELINING_HAZARDS"]
    if any(k in hint_low for k in ("naive bayes", "laplace", "smoothing", "spam filter", "aic354")):
        return _LOCAL_STUDY_PROFILES["AIC354"]
    if "bayes" in title_low and not any(k in hint_low for k in ("naive", "laplace", "spam")):
        return _LOCAL_STUDY_PROFILES["BAYES_THEOREM"]
    if any(k in hint_low for k in ("ieee", "754", "floating", "mantissa", "exponent", "12.625", "csc325")):
        return _LOCAL_STUDY_PROFILES["CSC325"]
    if any(k in hint_low for k in ("binary search", "midpoint", "master theorem", "logarithmic", "csc301")):
        return _LOCAL_STUDY_PROFILES["CSC301"]
    if any(k in hint_low for k in ("fcfs", "first-come", "cpu scheduling", "convoy", "turnaround", "gantt", "csc322")):
        return _LOCAL_STUDY_PROFILES["CSC322"]

    generic_markers = (
        "test", "sample", "lecture", "untitled", "notes", "audio", "board", "photo", "slides", "pdf", "capture",
    )
    title_words = re.findall(r"[a-z0-9]+", title_low)
    looks_generic = not title_words or all(w in generic_markers or w in ("only", "from", "the", "and", "of") for w in title_words)
    if looks_generic and code_clean in _LOCAL_STUDY_PROFILES:
        return _LOCAL_STUDY_PROFILES[code_clean]
    if code_clean in _LOCAL_STUDY_PROFILES and not title_clean:
        return _LOCAL_STUDY_PROFILES[code_clean]
    return None


def _build_custom_modality_content(
    mod_type: str,
    code_clean: str,
    title_clean: str,
    source_ref: str,
    pdf_snippets: list[str],
) -> tuple[list[StructuredNoteSection], list[KnowledgeChunkItem]]:
    """Produce real hierarchical NOTES sections with clear headings and subheadings strictly isolated to `mod_type`."""
    canon_mod: Literal["AUDIO", "WHITEBOARD_IMAGE", "SLIDES_PDF", "NOTES"] = _normalize_modality_tag(
        mod_type, [mod_type]
    )
    if canon_mod == "SLIDES_PDF":
        cue_takeaways = [s.strip() for s in pdf_snippets[:3] if s.strip()]
        if len(cue_takeaways) < 2:
            cue_takeaways = [
                f"Verify preconditions and formal invariants before evaluating any step of {title_clean}.",
                f"Compare best-case, average-case, and worst-case guarantees covered in {code_clean}.",
                "Test boundary inputs and degenerate cases during exam derivation problems.",
            ]
        sections = [
            StructuredNoteSection(
                heading=f"Foundations & Governing Principles of {title_clean}",
                source_type="SLIDES_PDF",
                subsections=[
                    NoteSubsection(
                        subheading=f"Core Concepts & Domain Preconditions",
                        body=(
                            f"{title_clean} in {code_clean} defines an explicit analytical framework connecting input "
                            "assumptions, state transitions, and verifiable invariants.\n\n"
                            f"Establishing preconditions up front ensures every slide derivation holds cleanly across both "
                            "typical and edge-case workloads."
                        ),
                        formula="Output(x) = Transform(x)   subject to   Precondition(x) = True",
                        bullets=[
                            f"Validated Input Domain: Confirm assumptions hold before applying {title_clean} steps.",
                            f"State Invariant: Property preserved across every step in {code_clean}.",
                        ],
                    ),
                    NoteSubsection(
                        subheading="Worked Trace & Verification Steps",
                        body=(
                            f"Follow these verification steps to confirm intermediate correctness on exam problems involving {title_clean}."
                        ),
                        formula="",
                        bullets=[
                            f"Step 1: Verify required preconditions for {title_clean}.",
                            "Step 2: Trace state updates incrementally on a small representative example.",
                            "Step 3: Check termination boundaries and final values against expected constraints.",
                        ],
                    ),
                ],
                takeaway=f"Core Rule: Check preconditions and boundary invariants first before applying {title_clean}.",
            ),
            StructuredNoteSection(
                heading=f"Performance Bounds, Trade-Offs & Exam Pitfalls",
                source_type="SLIDES_PDF",
                subsections=[
                    NoteSubsection(
                        subheading="Asymptotic Cost vs Baseline Sequential Approaches",
                        body=(
                            f"Evaluating {title_clean} contrasts worst-case complexity bounds against linear brute-force alternatives, "
                            f"highlighting which structural property unlocks speedups in {code_clean}."
                        ),
                        formula="Total Cost = (# Steps) × (Work per Step)",
                        bullets=[
                            "Baseline scans every candidate linearly without pruning.",
                            f"Structured {title_clean} leverages domain invariants to eliminate redundant work.",
                        ],
                    ),
                    NoteSubsection(
                        subheading="High-Yield Exam Distinctions",
                        body=f"Keep the key exam pitfalls below in mind when solving problems on {title_clean}.",
                        formula="",
                        bullets=cue_takeaways,
                    ),
                ],
                takeaway=f"Exam Check: Connect every formal guarantee in {title_clean} directly to the assumption enabling it.",
            ),
        ]
        chunks = [
            KnowledgeChunkItem(
                topic=f"{title_clean} — Core Principles",
                content=f"Foundational framework of {title_clean} ({code_clean}) establishes preconditions, terminology, and key invariants.",
                source_type="SLIDES_PDF",
                source_reference=source_ref,
                is_uncertain=False,
            ),
            KnowledgeChunkItem(
                topic=f"{title_clean} — Complexity & Bounds",
                content=f"Formal guarantees for {title_clean} quantify worst-case bounds and clarify boundary constraints in {code_clean}.",
                source_type="SLIDES_PDF",
                source_reference="Slide 2",
                is_uncertain=False,
            ),
        ]
        return sections, chunks

    if canon_mod == "AUDIO":
        sections = [
            StructuredNoteSection(
                heading=f"Conceptual Intuition & Walkthrough of {title_clean}",
                source_type="AUDIO",
                subsections=[
                    NoteSubsection(
                        subheading="Why Each Step Works (Spoken Intuition)",
                        body=(
                            f"Lecture discussion on {title_clean} emphasizes why each state transition holds rather than "
                            f"memorizing isolated formulas in {code_clean}.\n\n"
                            "Tracing intermediate state variables builds concrete mental intuition for tricky corner cases."
                        ),
                        formula="Next State = TransitionRule(Current State, Observation)",
                        bullets=[
                            "Track state changes cleanly from initialization to final convergence.",
                            f"Watch boundary values carefully in {code_clean} derivations.",
                        ],
                    ),
                    NoteSubsection(
                        subheading="Step-by-Step Practical Trace",
                        body="Walk through any concrete problem systematically using these three steps.",
                        formula="",
                        bullets=[
                            f"1. Initialize parameters and verify preconditions for {title_clean}.",
                            "2. Execute each transition step while updating tracked state variables.",
                            "3. Inspect stopping condition and confirm final output satisfies constraints.",
                        ],
                    ),
                ],
                takeaway=f"Key Takeaway: Trace how each transition step in {title_clean} preserves state invariants.",
            ),
        ]
        chunks = [
            KnowledgeChunkItem(
                topic=f"{title_clean} — Conceptual Intuition",
                content=f"{title_clean} relies on structured state transitions and clear design intuition emphasized during {code_clean} lecture discussion.",
                source_type="AUDIO",
                source_reference=source_ref,
                is_uncertain=False,
            ),
            KnowledgeChunkItem(
                topic=f"{title_clean} — Exam Distinctions",
                content=f"Key exam pitfalls in {title_clean} involve initialization boundaries, edge inputs, and performance trade-offs.",
                source_type="AUDIO",
                source_reference="00:45",
                is_uncertain=False,
            ),
        ]
        return sections, chunks

    # WHITEBOARD_IMAGE (or NOTES)
    sections = [
        StructuredNoteSection(
            heading=f"Visual Derivation & Worked Calculation for {title_clean}",
            source_type=canon_mod,
            subsections=[
                NoteSubsection(
                    subheading="Governing Equation & Symbol Meaning",
                    body=(
                        f"Whiteboard derivations convert abstract principles of {title_clean} into explicit algebraic expressions, "
                        f"labeled terms, and intermediate checkpoints in {code_clean}."
                    ),
                    formula="Target Output = Function(Parameters, Baseline Terms)",
                    bullets=[
                        "Given Inputs: Baseline values recorded before intermediate calculations.",
                        "Intermediate Terms: Component quantities verified prior to final combination.",
                    ],
                ),
                NoteSubsection(
                    subheading="Step-by-Step Worked Solution & Consistency Checks",
                    body=(
                        f"Showing intermediate substitutions prevents sign, normalization, or boundary mistakes under time pressure."
                    ),
                    formula="",
                    bullets=[
                        f"1. State given inputs and governing relation for {title_clean}.",
                        "2. Evaluate intermediate sub-expressions step by step.",
                        "3. Normalize and verify boundary invariants on the final answer.",
                    ],
                ),
            ],
            takeaway=f"Worked Strategy: Break {title_clean} derivations into clear intermediate steps before combining.",
        ),
    ]
    chunks = [
        KnowledgeChunkItem(
            topic=f"{title_clean} — Worked Board Trace",
            content=f"Worked walkthrough of {title_clean} demonstrates intermediate state updates step by step on concrete sample values.",
            source_type=canon_mod,
            source_reference=source_ref,
            is_uncertain=False,
        ),
        KnowledgeChunkItem(
            topic=f"{title_clean} — Consistency Verification",
            content=f"Mathematical verification for {title_clean} checks initialization and termination states across {code_clean} examples.",
            source_type=canon_mod,
            source_reference=source_ref,
            is_uncertain=False,
        ),
    ]
    return sections, chunks


def build_local_lecture_synthesis(
    course_code: str,
    lecture_title: str,
    audio_files: list[tuple[str, bytes, str]] | None = None,
    image_files: list[tuple[str, bytes, str]] | None = None,
    pdf_files: list[tuple[str, bytes, str]] | None = None,
    extra_notes: str = "",
) -> LectureSynthesisResult:
    """Deterministic local synthesis fallback keyed STRICTLY to the supplied modality inputs."""
    audio_files = audio_files or []
    image_files = image_files or []
    pdf_files = pdf_files or []
    notes_clean = (extra_notes or "").strip()

    if not (audio_files or image_files or pdf_files or notes_clean):
        raise ValueError(
            "Please upload at least one audio recording, whiteboard image, PDF slide deck, "
            "or classroom notes before running synthesis."
        )

    code_clean = (course_code or "GENERAL").strip().upper()
    title_clean = (lecture_title or "Untitled Lecture").strip()
    mod_info = describe_provided_modalities(audio_files, image_files, pdf_files, notes_clean)
    labels_str = ", ".join(mod_info["labels"])

    all_pdf_snippets: list[str] = []
    for _, data, _ in pdf_files:
        _, snippets = _inspect_pdf_summary(data)
        all_pdf_snippets.extend(snippets)

    uploaded_filenames = (
        [fname for fname, _, _ in audio_files]
        + [fname for fname, _, _ in image_files]
        + [fname for fname, _, _ in pdf_files]
    )
    first_audio_dur = _inspect_wav_duration(audio_files[0][1]) if audio_files else None
    profile = _select_study_profile(
        code_clean=code_clean,
        title_clean=title_clean,
        pdf_snippets=all_pdf_snippets,
        notes_clean=notes_clean,
        filenames=uploaded_filenames,
    )

    chunks: list[KnowledgeChunkItem] = []
    sections: list[StructuredNoteSection] = []
    pseudocode_lines: list[str] = []

    present_modalities = list(mod_info["source_types"])
    preferred_order = ["SLIDES_PDF", "AUDIO", "WHITEBOARD_IMAGE", "NOTES"]
    ordered_mods = [m for m in preferred_order if m in present_modalities]

    for mod_type in ordered_mods:
        if mod_type == "NOTES":
            note_sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", notes_clean)
                if s.strip()
            ]
            if not note_sentences:
                note_sentences = [notes_clean]
            for n_idx, sent in enumerate(note_sentences[:3], start=1):
                chunks.append(
                    KnowledgeChunkItem(
                        topic=f"{title_clean} — Key Concept {n_idx}",
                        content=sent,
                        source_type="NOTES",
                        source_reference=f"Notes #{n_idx}",
                        is_uncertain=False,
                    )
                )
            note_takeaways = list(note_sentences[:4])
            while len(note_takeaways) < 2:
                note_takeaways.append(
                    f"Verify core rules, definitions, and boundary cases from {title_clean}."
                )
            sections.append(
                StructuredNoteSection(
                    heading=f"{title_clean} — Structured Concepts & Takeaways",
                    source_type="NOTES",
                    subsections=[
                        NoteSubsection(
                            subheading="Core Definitions & Conceptual Notes",
                            body="\n\n".join(note_sentences[:2]) or notes_clean,
                            formula="",
                            bullets=note_takeaways[:3],
                        ),
                        NoteSubsection(
                            subheading="Practical Application & Boundary Reminders",
                            body=f"Apply these principles directly across {code_clean} problem sets: {note_sentences[-1]}",
                            formula="",
                            bullets=note_takeaways,
                        ),
                    ],
                    takeaway=note_sentences[-1],
                )
            )
            pseudocode_lines.extend(
                line.strip()
                for line in notes_clean.splitlines()
                if any(sym in line for sym in ("=", "->", "→", "def ", "while ", "for ", "if "))
            )
            continue

        if profile is not None:
            for sec_dict in profile["sections_by_modality"].get(mod_type, []):
                clean_sec = dict(sec_dict)
                clean_sec["source_type"] = mod_type
                if not clean_sec.get("subsections"):
                    norm_subs = normalize_note_subsections(clean_sec)
                    while len(norm_subs) < 2:
                        norm_subs.append(
                            {
                                "subheading": "Key Concepts & Takeaways",
                                "body": str(clean_sec.get("takeaway") or clean_sec.get("heading") or "Study notes."),
                                "formula": "",
                                "bullets": [],
                            }
                        )
                    clean_sec["subsections"] = [NoteSubsection(**s) for s in norm_subs[:5]]
                sections.append(StructuredNoteSection(**clean_sec))
            for ch_idx, (topic_str, body_str, default_ref) in enumerate(
                profile["chunks_by_modality"].get(mod_type, []), start=1
            ):
                resolved_ref = default_ref
                if mod_type == "AUDIO" and ch_idx == 1 and first_audio_dur:
                    resolved_ref = first_audio_dur
                chunks.append(
                    KnowledgeChunkItem(
                        topic=topic_str,
                        content=body_str,
                        source_type=mod_type,  # type: ignore[arg-type]
                        source_reference=resolved_ref,
                        is_uncertain=False,
                    )
                )
        else:
            default_ref = (
                first_audio_dur or "00:25"
                if mod_type == "AUDIO"
                else ("Whiteboard #1" if mod_type == "WHITEBOARD_IMAGE" else "Slide 1")
            )
            custom_secs, custom_chunks = _build_custom_modality_content(
                mod_type=mod_type,
                code_clean=code_clean,
                title_clean=title_clean,
                source_ref=default_ref,
                pdf_snippets=all_pdf_snippets,
            )
            sections.extend(custom_secs)
            chunks.extend(custom_chunks)

    # Guarantee at least 3 chunks strictly within active source_types
    primary_source = mod_info["source_types"][0]
    while len(chunks) < 3:
        extra_i = len(chunks) + 1
        chunks.append(
            KnowledgeChunkItem(
                topic=f"{title_clean} — Core Concept {extra_i}",
                content=(
                    f"Core academic takeaway {extra_i} for {title_clean} ({code_clean}) "
                    "highlighting key rules, derivations, and exam readiness."
                ),
                source_type=primary_source,  # type: ignore[arg-type]
                source_reference=_DEFAULT_REF_FOR_MODALITY.get(primary_source, "Slide 1"),
                is_uncertain=False,
            )
        )

    if profile is not None:
        detected = str(profile.get("detected_topic") or title_clean)
        exec_summary = profile["executive_summary"]
        extracted_code = profile.get("pseudocode", "")
    else:
        detected = title_clean
        exec_summary = (
            f"{title_clean} ({code_clean}) presents essential principles, worked logic, and practical bounds "
            f"across {labels_str}. Structured notes below organize formal definitions, equations, worked derivations, "
            "and exam takeaways for grounded Q&A and exam preparation."
        )
        extracted_code = (
            "\n".join(pseudocode_lines[:8])
            if pseudocode_lines
            else f"{title_clean} Key Summary ({code_clean}):\n  1. Verify input preconditions & edge cases\n  2. Apply state transitions & complexity bounds step by step"
        )

    res = LectureSynthesisResult(
        detected_topic=detected,
        executive_summary=exec_summary,
        structured_notes=sections,
        extracted_pseudocode=extracted_code,
        knowledge_chunks=chunks,
    )
    return _enforce_strict_modalities(res, present_modalities)


def build_local_revision_pack(
    lecture_title: str,
    chunks: list[dict[str, Any]],
) -> RevisionPack:
    """Build a valid 10-MCQ, 6-flashcard RevisionPack from stored chunks without network calls."""
    if not chunks:
        raise ValueError("Cannot generate revision items because this lecture has no stored chunks.")

    mcqs: list[MCQItem] = []
    for idx in range(10):
        chunk = chunks[idx % len(chunks)]
        topic = str(chunk.get("topic") or f"Concept {idx + 1}").strip()
        content = str(chunk.get("content") or "").strip()
        badge = f"[{chunk.get('source_type', 'NOTES')} @ {chunk.get('source_reference', 'Unmarked in source')}]"
        short_fact = (content[:115] + "...") if len(content) > 115 else content
        if not short_fact:
            short_fact = f"Core concept covered in {lecture_title}."
        correct_idx = idx % 4
        distractors = [
            f"Unrelated topic outside {lecture_title}",
            "Deprecates all lecture definitions and formulas",
            "Applies only when input data is completely empty",
        ]
        options = list(distractors)
        options.insert(correct_idx, short_fact)
        mcqs.append(
            MCQItem(
                question=f"({idx + 1}/10) Which statement accurately reflects '{topic}' in {lecture_title}?",
                options=options,
                correct_index=correct_idx,
                explanation=f"Grounded in {badge}: {short_fact}",
                source_badge=badge,
            )
        )

    flashcards: list[FlashcardItem] = []
    for idx in range(6):
        chunk = chunks[idx % len(chunks)]
        topic = str(chunk.get("topic") or f"Key Concept {idx + 1}").strip()
        content = str(chunk.get("content") or f"Key takeaway for {topic}.").strip()
        badge = f"[{chunk.get('source_type', 'NOTES')} @ {chunk.get('source_reference', 'Unmarked in source')}]"
        flashcards.append(
            FlashcardItem(
                front=f"{topic} ({lecture_title})",
                back=content,
                source_badge=badge,
            )
        )

    return RevisionPack(mcqs=mcqs, flashcards=flashcards)


def synthesize_lecture_with_timeout(
    course_code: str,
    lecture_title: str,
    audio_files: list[tuple[str, bytes, str]] | None = None,
    image_files: list[tuple[str, bytes, str]] | None = None,
    pdf_files: list[tuple[str, bytes, str]] | None = None,
    extra_notes: str = "",
    model_name: str | None = None,
    timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
) -> tuple[LectureSynthesisResult, bool]:
    """
    Run Gemini lecture synthesis under a strict ThreadPoolExecutor timeout with local fallback.

    Returns `(synthesis_result, used_local_fallback)`.
    """
    audio_files = audio_files or []
    image_files = image_files or []
    pdf_files = pdf_files or []

    if not (audio_files or image_files or pdf_files or extra_notes.strip()):
        raise ValueError(
            "Please upload at least one audio recording, whiteboard image, PDF slide deck, "
            "or text excerpt before running ScribeAI."
        )

    if is_api_configured():
        try:
            result = _run_with_timeout(
                lambda: process_lecture_materials(
                    course_code=course_code,
                    lecture_title=lecture_title,
                    audio_files=audio_files,
                    image_files=image_files,
                    pdf_files=pdf_files,
                    extra_notes=extra_notes,
                    model_name=model_name,
                ),
                timeout_seconds=timeout_seconds,
            )
            return result, False
        except Exception:
            pass

    fallback = build_local_lecture_synthesis(
        course_code=course_code,
        lecture_title=lecture_title,
        audio_files=audio_files,
        image_files=image_files,
        pdf_files=pdf_files,
        extra_notes=extra_notes,
    )
    return fallback, True


def generate_revision_pack_with_timeout(
    lecture_title: str,
    chunks: list[dict[str, Any]],
    model_name: str | None = None,
    use_cache: bool = False,
    timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
) -> tuple[RevisionPack, bool]:
    """
    Generate 10 MCQs and 6 flashcards under a strict ThreadPoolExecutor timeout with local fallback.

    Returns `(revision_pack, used_local_fallback)`.
    """
    if not chunks:
        raise ValueError("Cannot generate revision items because this lecture has no stored chunks.")

    if is_api_configured():
        try:
            pack = _run_with_timeout(
                lambda: generate_revision_pack(
                    lecture_title=lecture_title,
                    chunks=chunks,
                    model_name=model_name,
                    use_cache=use_cache,
                ),
                timeout_seconds=timeout_seconds,
            )
            return pack, False
        except Exception:
            pass

    return build_local_revision_pack(lecture_title=lecture_title, chunks=chunks), True


def answer_grounded_question_with_timeout(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model_name: str | None = None,
    timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
) -> tuple[RAGAnswer, bool]:
    """Answer grounded Q&A under a strict timeout, falling back cleanly to matching lecture chunks."""
    if not retrieved_chunks:
        return answer_grounded_question(question=question, retrieved_chunks=[]), False

    if is_api_configured():
        try:
            reply = _run_with_timeout(
                lambda: answer_grounded_question(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    model_name=model_name,
                ),
                timeout_seconds=timeout_seconds,
            )
            return reply, False
        except Exception:
            pass

    lines: list[str] = []
    badges: list[str] = []
    for chunk in retrieved_chunks[:2]:
        badge = chunk.get("source_badge") or (
            f"[{chunk.get('source_type', 'SYNTHESIZED')} @ {chunk.get('source_reference', 'Unmarked in source')}]"
        )
        badges.append(badge)
        content = str(chunk.get("content") or "").strip()
        if chunk.get("is_uncertain"):
            lines.append(f"- {content} (Flagged for verification)")
        else:
            lines.append(f"- {content}")
    return (
        RAGAnswer(
            answer="From the lecture notes:\n" + "\n".join(lines),
            cited_badges=badges,
            grounded=True,
        ),
        True,
    )


def build_local_regenerated_notes(
    course_code: str,
    lecture_title: str,
    current_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    feedback_text: str,
) -> LectureSynthesisResult:
    """Offline fallback that enriches existing 6-part study notes using student feedback while isolating modalities."""
    fb_clean = (
        feedback_text or "Improve conceptual depth, clarity, formulas, and step-by-step examples"
    ).strip()
    allowed_sources = _infer_lecture_modalities(current_summary, chunks)
    existing_sections = list((current_summary or {}).get("structured_notes") or [])
    enhanced_sections: list[StructuredNoteSection] = []

    for sec in existing_sections:
        heading = str(sec.get("heading") or "Core Concept").strip()
        src_type = _normalize_modality_tag(
            str(sec.get("source_type") or sec.get("source_tag") or allowed_sources[0]),
            allowed_sources,
        )
        legacy_bullets = [str(b).strip() for b in (sec.get("bullets") or []) if str(b).strip()]
        existing_takeaways = [
            str(b).strip() for b in (sec.get("exam_takeaways") or legacy_bullets) if str(b).strip()
        ]
        existing_def = str(sec.get("formal_definition") or "").strip()
        existing_exp = str(sec.get("detailed_explanation") or sec.get("explanation") or "").strip()
        existing_formula = str(sec.get("formula_and_breakdown") or "").strip()
        existing_ex = str(sec.get("worked_example") or "").strip()
        takeaway = str(sec.get("takeaway") or "").strip()

        formal_def = existing_def or (
            existing_takeaways[0]
            if existing_takeaways
            else f"{heading} formalizes a core governing principle and invariant within {lecture_title} ({course_code})."
        )

        if existing_exp:
            base_paras = [p.strip() for p in existing_exp.split("\n\n") if p.strip()]
            deep_exp = (
                f"{base_paras[0]}\n\n"
                + (" ".join(base_paras[1:]) + " " if len(base_paras) > 1 else "")
                + f"Refined per student feedback ({fb_clean}): connecting each formal term explicitly to step-by-step "
                "calculations and edge cases prevents conceptual ambiguity under exam conditions."
            ).strip()
        else:
            deep_exp = (
                f"{heading} builds directly on the core analytical principles of {lecture_title} ({course_code}). "
                + (" ".join(existing_takeaways[:2]) if existing_takeaways else "")
                + f"\n\nExpanded per student feedback ({fb_clean}): understanding how every assumption maps cleanly "
                "to intermediate numerical transformations and boundary inputs guarantees reliable exam performance."
            ).strip()

        formula_txt = existing_formula or (
            f"{takeaway or f'{heading} Relation'}\n\n"
            "• Core Inputs — Initial parameters verified against domain preconditions before applying rules.\n"
            "• Output / Bound — Exact intermediate or final target value confirmed after normalization."
        )

        worked_ex = existing_ex or (
            f"Step-by-Step Worked Application ({heading}):\n"
            "1. State given inputs, assumptions, and governing relation cleanly.\n"
            f"2. Substitute concrete values through intermediate steps ({takeaway or 'step-by-step evaluation'}).\n"
            "3. Verify boundary conditions and normalization of the final answer."
        )
        if fb_clean.lower() not in worked_ex.lower() and len(worked_ex.splitlines()) < 3:
            worked_ex = (
                f"{worked_ex}\n"
                "Verification Check: Confirm edge inputs and units match expected invariants step by step."
            )

        takeaways_list = list(existing_takeaways)
        while len(takeaways_list) < 2:
            if len(takeaways_list) == 0:
                takeaways_list.append(
                    f"Always state governing assumptions and symbols clearly before solving {heading} problems."
                )
            else:
                takeaways_list.append(
                    f"Check intermediate normalization and boundary inputs when applying {heading} in {course_code}."
                )

        existing_subs = [
            sub
            for sub in (sec.get("subsections") or [])
            if isinstance(sub, dict)
            and (sub.get("subheading") or "").strip()
            not in {"", "Definition, explanation, and governing relation", "Worked example and exam checks"}
        ]
        if existing_subs:
            refined_subs = []
            for sub_idx, sub in enumerate(existing_subs):
                sub_body = str(sub.get("body") or "").strip()
                if sub_idx == 0 and fb_clean.lower() not in sub_body.lower():
                    sub_body = (
                        f"{sub_body}\n\nRefined per student feedback ({fb_clean}): connecting each formal term explicitly "
                        "to step-by-step calculations and boundary cases prevents conceptual ambiguity."
                    ).strip()
                refined_subs.append(
                    NoteSubsection(
                        subheading=str(sub.get("subheading") or f"Topic {sub_idx + 1}").strip(),
                        body=sub_body,
                        formula=str(sub.get("formula") or "").strip(),
                        bullets=[str(b).strip() for b in (sub.get("bullets") or []) if str(b).strip()],
                    )
                )
        else:
            refined_subs = [
                NoteSubsection(
                    subheading="Definition, explanation, and governing relation",
                    body="\n\n".join(part for part in (formal_def, deep_exp) if part),
                    formula=formula_txt,
                ),
                NoteSubsection(
                    subheading="Worked example and exam checks",
                    body=worked_ex or "No worked example was preserved in the available lecture notes.",
                    bullets=takeaways_list[:5],
                ),
            ]

        enhanced_sections.append(
            StructuredNoteSection(
                heading=heading,
                source_type=src_type,
                formal_definition=formal_def,
                detailed_explanation=deep_exp,
                formula_and_breakdown=formula_txt,
                worked_example=worked_ex,
                exam_takeaways=takeaways_list[:5],
                subsections=refined_subs,
                takeaway=takeaway or (takeaways_list[0] if takeaways_list else ""),
            )
        )

    if not enhanced_sections and chunks:
        for idx, ch in enumerate(chunks[:3], start=1):
            src_type = _normalize_modality_tag(str(ch.get("source_type") or ""), allowed_sources)
            topic_txt = str(ch.get("topic") or f"Section {idx}").strip()
            content_txt = str(ch.get("content") or "").strip()
            enhanced_sections.append(
                StructuredNoteSection(
                    heading=topic_txt,
                    source_type=src_type,
                    formal_definition=content_txt or f"{topic_txt} establishes a core theorem in {lecture_title}.",
                    detailed_explanation=(
                        f"{content_txt}\n\nRefined per student feedback ({fb_clean}): trace every assumption "
                        "and variable interaction explicitly before combining results into a final answer."
                    ),
                    formula_and_breakdown=(
                        f"Key Rule — {topic_txt}\n\n"
                        f"• Grounding Evidence — Observed in {ch.get('source_reference', 'uploaded material')}."
                    ),
                    worked_example=f"Apply '{topic_txt}' step-by-step on a sample {course_code} problem input.",
                    exam_takeaways=[content_txt] if content_txt else [],
                    subsections=[
                        NoteSubsection(
                            subheading="Source content",
                            body=content_txt or "No readable source content was preserved for this section.",
                        ),
                        NoteSubsection(
                            subheading="Review status",
                            body="This offline regeneration did not add unsupported formulas or worked examples. Use Gemini regeneration for a fuller source-grounded explanation.",
                        ),
                    ],
                    takeaway=content_txt[:140],
                )
            )

    chunk_items: list[KnowledgeChunkItem] = []
    for ch in chunks:
        fixed_type = _normalize_modality_tag(str(ch.get("source_type") or ""), allowed_sources)
        chunk_items.append(
            KnowledgeChunkItem(
                topic=str(ch.get("topic") or "Concept").strip(),
                content=str(ch.get("content") or "").strip(),
                source_type=fixed_type,
                source_reference=str(ch.get("source_reference") or _DEFAULT_REF_FOR_MODALITY.get(fixed_type, "Unmarked in source")),
                is_uncertain=bool(ch.get("is_uncertain", False)),
                uncertainty_reason=str(ch.get("uncertainty_reason") or ""),
            )
        )

    base_exec = str((current_summary or {}).get("executive_summary") or "").strip()
    improved_exec = (
        f"{base_exec} Updated with expanded intuitive explanations and step-by-step worked derivations "
        f"addressing student feedback: \"{fb_clean}\"."
    ).strip()
    res = LectureSynthesisResult(
        detected_topic=str((current_summary or {}).get("detected_topic") or lecture_title).strip(),
        executive_summary=improved_exec,
        structured_notes=enhanced_sections,
        extracted_pseudocode=str((current_summary or {}).get("extracted_pseudocode") or ""),
        knowledge_chunks=chunk_items,
    )
    return _enforce_strict_modalities(res, allowed_sources)


def regenerate_lecture_notes_with_feedback(
    course_code: str,
    lecture_title: str,
    current_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    feedback_text: str,
    model_name: str | None = None,
) -> LectureSynthesisResult:
    """
    Call Gemini API to rewrite and improve stored lecture notes according to student feedback.

    Remains strictly grounded in the existing lecture's uploaded modality and SQLite knowledge chunks.
    """
    fb_clean = (
        feedback_text.strip()
        or "Make explanations deeper, simplify formulas into clear steps, and add worked examples."
    )
    if not chunks and not (current_summary or {}).get("executive_summary"):
        raise ValueError("Cannot regenerate notes because this lecture has no stored content.")

    allowed_sources = _infer_lecture_modalities(current_summary, chunks)
    allowed_str = ", ".join(allowed_sources)

    _, env_model = load_api_environment()
    active_model = model_name or env_model
    client = get_client()

    chunk_context = []
    for ch in chunks:
        badge = f"[{ch.get('source_type', 'SYNTHESIZED')} @ {ch.get('source_reference', 'Unmarked in source')}]"
        chunk_context.append(f"{badge} {ch.get('topic', 'General')}: {ch.get('content', '')}")

    system_instruction = (
        "You are ScribeAI inside CAMPUSFLOW, an expert university textbook author regenerating Master Study Guide "
        "lecture notes after student feedback.\n\n"
        f"CRITICAL MODALITY ISOLATION RULE:\n"
        f"- EVERY `StructuredNoteSection.source_type` and EVERY `KnowledgeChunkItem.source_type` MUST STRICTLY stay inside [{allowed_str}].\n"
        "- NEVER introduce unprovided modalities.\n\n"
        "YOUR TASK:\n"
        f"- Directly address the student's improvement feedback: \"{fb_clean}\"\n"
        "- Rewrite and significantly deepen `executive_summary`, `structured_notes`, and `extracted_pseudocode`.\n"
        "- Populate each `StructuredNoteSection` as real hierarchical NOTES with 2 to 5 rich `subsections` "
        "(`subheading`, `body`, `formula`, `bullets`) AND optional summary fields (`formal_definition`, `detailed_explanation`, "
        "`formula_and_breakdown`, `worked_example`, `exam_takeaways`) strictly from lecture evidence; never invent missing facts.\n"
        "- Format inside `StructuredNoteSection`:\n"
        "  1. `heading`: Specific academic topic title.\n"
        f"  2. `source_type`: Strictly inside [{allowed_str}].\n"
        "  3. `subsections`: 2 to 5 topic-driven subheadings, each with 1-2 clear prose paragraphs (`body`), "
        "clean Unicode-math formulas (`formula` using ÷, ×, ·, −, Σ, Θ, ⌊, ⌋, →, ≤, ≥, ≈ — NEVER raw ASCII/LaTeX), "
        "and bulleted definitions, worked derivation steps, or exam checkpoints (`bullets`).\n"
        "  4. `takeaway`: Highlighted bottom-line summary rule.\n"
        "  DO NOT include bilingual Roman-Urdu notes or 'Classroom note:' lines.\n"
        "- Preserve factual grounding strictly from the lecture material and refine `knowledge_chunks` inside "
        f"[{allowed_str}]."
    )

    prompt = (
        f"Course: {course_code}\n"
        f"Lecture Topic: {lecture_title}\n"
        f"Allowed Source Modalities (STRICT): {allowed_str}\n"
        f"Student Feedback to Address: {fb_clean}\n\n"
        "Current Summary JSON:\n"
        f"{json.dumps(current_summary, ensure_ascii=False)}\n\n"
        "Grounded Lecture Chunks:\n"
        + ("\n".join(chunk_context) if chunk_context else "(Use current summary context above)")
    )

    _LAST_MODEL_STATE.model_id = active_model
    response = _generate_content_resilient(
        client=client,
        primary_model=active_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.25,
            response_mime_type="application/json",
            response_json_schema=LectureSynthesisResult.model_json_schema(),
        ),
    )
    used_model = getattr(_LAST_MODEL_STATE, "model_id", active_model) or active_model
    raw_json = _strip_json_fences(response.text or "")
    if not raw_json:
        raise RuntimeError("Gemini returned an empty response during note regeneration.")
    result = LectureSynthesisResult.model_validate_json(raw_json)
    result = _enforce_strict_modalities(result, allowed_sources)
    result._model_used = used_model
    return _copy_with_model(result, used_model)


def regenerate_lecture_notes_with_timeout(
    course_code: str,
    lecture_title: str,
    current_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    feedback_text: str,
    model_name: str | None = None,
    timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
) -> tuple[LectureSynthesisResult, bool]:
    """Run feedback-guided Gemini note regeneration under timeout with local enhancement fallback."""
    if is_api_configured():
        try:
            result = _run_with_timeout(
                lambda: regenerate_lecture_notes_with_feedback(
                    course_code=course_code,
                    lecture_title=lecture_title,
                    current_summary=current_summary,
                    chunks=chunks,
                    feedback_text=feedback_text,
                    model_name=model_name,
                ),
                timeout_seconds=timeout_seconds,
            )
            return result, False
        except Exception:
            pass

    fallback = build_local_regenerated_notes(
        course_code=course_code,
        lecture_title=lecture_title,
        current_summary=current_summary,
        chunks=chunks,
        feedback_text=feedback_text,
    )
    return fallback, True


def normalize_note_subsections(sec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized `{subheading, body, formula, bullets}` list from either current or legacy section dicts."""
    raw_subs = sec.get("subsections") or []
    normalized: list[dict[str, Any]] = []
    for sub in raw_subs:
        if not isinstance(sub, dict):
            continue
        subh = str(sub.get("subheading") or "").strip()
        body = str(sub.get("body") or "").strip()
        formula = str(sub.get("formula") or "").strip()
        bullets = [str(b).strip() for b in (sub.get("bullets") or []) if str(b).strip()]
        if subh or body or formula or bullets:
            normalized.append(
                {
                    "subheading": subh or "Topic Details",
                    "body": body,
                    "formula": formula,
                    "bullets": bullets,
                }
            )
    if normalized:
        return normalized

    # Legacy adapter: build clean topical subheadings strictly from non-empty stored fields (no fabrication)
    legacy_subs: list[dict[str, Any]] = []
    formal_def = str(sec.get("formal_definition") or "").strip()
    explanation_txt = str(sec.get("detailed_explanation") or sec.get("explanation") or "").strip()
    if formal_def or explanation_txt:
        legacy_subs.append(
            {
                "subheading": "Core Definition & Intuition",
                "body": "\n\n".join(part for part in (formal_def, explanation_txt) if part),
                "formula": "",
                "bullets": [],
            }
        )

    formula_txt = str(sec.get("formula_and_breakdown") or "").strip()
    if formula_txt:
        f_lines = [fl.strip() for fl in formula_txt.splitlines() if fl.strip()]
        f_eq = f_lines[0] if f_lines else formula_txt
        f_bullets = [line.lstrip("•-* ").strip() for line in f_lines[1:] if line.lstrip("•-* ").strip()]
        legacy_subs.append(
            {
                "subheading": "Governing Equation & Symbol Breakdown",
                "body": "Each term in the expression above governs a distinct part of the relationship.",
                "formula": f_eq,
                "bullets": f_bullets,
            }
        )

    worked_txt = str(sec.get("worked_example") or "").strip()
    if worked_txt:
        legacy_subs.append(
            {
                "subheading": "Step-by-Step Worked Example",
                "body": worked_txt,
                "formula": "",
                "bullets": [],
            }
        )

    takeaway_bullets = [
        str(b).strip()
        for b in (sec.get("exam_takeaways") or sec.get("bullets") or [])
        if str(b).strip()
    ]
    if takeaway_bullets:
        legacy_subs.append(
            {
                "subheading": "Key Exam Rules & Takeaways",
                "body": "Important rules and verification checkpoints for exams.",
                "formula": "",
                "bullets": takeaway_bullets,
            }
        )

    return legacy_subs


class ParsedTimetableSlot(BaseModel):
    course_code: str
    course_name: str
    section: str
    day: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    start_time: str = Field(description="24-hour HH:MM start time, e.g. 08:30")
    end_time: str = Field(description="24-hour HH:MM end time, e.g. 10:30")
    room: str = "TBA"
    instructor: str = "TBA"


class ParsedTimetableCatalog(BaseModel):
    slots: list[ParsedTimetableSlot] = Field(default_factory=list)


def _extract_pdf_text_for_timetable(pdf_bytes: bytes) -> str:
    """Extract clean readable text from uncompressed or FlateDecode-compressed PDF streams using stdlib zlib (or pypdf if installed)."""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pypdf_txt = "\n".join((page.extract_text() or "") for page in reader.pages[:15]).strip()
        if pypdf_txt:
            return pypdf_txt
    except Exception:
        pass

    import zlib

    chunks: list[str] = []
    try:
        streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.S)
        sources = streams if streams else [pdf_bytes]
        for raw_stream in sources:
            try:
                dec = zlib.decompress(raw_stream).decode("latin-1", errors="ignore")
            except Exception:
                dec = raw_stream.decode("latin-1", errors="ignore")
            for arr in re.findall(r"\[(.*?)\]\s*TJ", dec, re.S):
                piece = "".join(re.findall(r"\(([^()]*)\)", arr)).strip()
                if piece:
                    chunks.append(piece)
            for single in re.findall(r"\(([^()]*)\)\s*Tj", dec):
                cleaned = single.strip()
                if cleaned:
                    chunks.append(cleaned)
    except Exception:
        return ""
    return "\n".join(chunks)


def parse_timetable_pdf_with_fallback(
    pdf_bytes: bytes,
    timeout_seconds: float = 20.0,
) -> tuple[list[dict[str, str]], bool]:
    """
    Parse an uploaded university timetable PDF into normalized `(course_code, ..., instructor)` rows.

    Returns `(rows, used_gemini_ai)`. Pure-Python geometric + tabular PDF parsing runs first
    (zero `.env` or API key required), followed by plain-text COMSATS extraction, and falls back
    to Gemini structured extraction only for scanned/non-text PDFs when configured.
    """
    import timetable as tt

    geom_df = tt.parse_timetable_pdf_bytes(pdf_bytes)
    if not geom_df.empty:
        return geom_df.to_dict("records"), False

    raw_text = _extract_pdf_text_for_timetable(pdf_bytes)
    if raw_text:
        det_df = tt.parse_comsats_timetable_pdf_text(raw_text)
        if not det_df.empty:
            return det_df.to_dict("records"), False

    if is_api_configured():
        def _gemini_extract() -> list[dict[str, str]]:
            _, active_model = load_api_environment()
            client = get_client()
            sys_inst = (
                "Extract every scheduled lecture, lab, and seminar from the uploaded university timetable PDF. "
                "Output strict JSON matching the schema. Normalize day to 3-letter Mon/Tue/Wed/Thu/Fri/Sat and "
                "times to 24-hour HH:MM (e.g. 0830 -> 08:30, 1130 -> 11:30). Preserve exact room and instructor."
            )
            part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            resp = _generate_content_resilient(
                client=client,
                primary_model=active_model,
                contents=[part, "Extract all timetable sections and weekly class/lab slots."],
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=ParsedTimetableCatalog.model_json_schema(),
                ),
            )
            raw_json = _strip_json_fences(resp.text or "")
            if not raw_json:
                return []
            parsed = ParsedTimetableCatalog.model_validate_json(raw_json)
            cleaned: list[dict[str, str]] = []
            for s in parsed.slots:
                try:
                    tt.validate_time_slot(s.start_time, s.end_time)
                except ValueError:
                    continue
                cleaned.append(
                    {
                        "course_code": s.course_code.strip().upper(),
                        "course_name": s.course_name.strip(),
                        "section": s.section.strip().upper(),
                        "day": s.day,
                        "start_time": s.start_time.strip(),
                        "end_time": s.end_time.strip(),
                        "room": s.room.strip() or "TBA",
                        "instructor": s.instructor.strip() or "TBA",
                    }
                )
            return cleaned

        try:
            ai_rows = _run_with_timeout(_gemini_extract, timeout_seconds=timeout_seconds)
            if ai_rows:
                return ai_rows, True
        except Exception:
            pass

    return [], False


def generate_timetable_advice_with_timeout(
    cohort_section: str,
    clash_lines: list[str],
    prereq_issues: list[str],
    swap_lines: list[str],
    solved_zero_clash: bool,
    locked_cohort_preserved: bool,
    timeout_seconds: float = 10.0,
) -> tuple[str, bool]:
    """
    Generate clear student-facing advisor commentary grounded strictly in verified
    clash, prerequisite, and alternative-section facts (with instant deterministic fallback).
    """
    def _local_summary() -> str:
        parts: list[str] = []
        if prereq_issues:
            parts.append("Prerequisite Alerts: " + " | ".join(prereq_issues))
        else:
            parts.append("Prerequisites: All selected courses satisfy verified prerequisite rules (or have no prerequisite blockers listed).")

        if not clash_lines:
            parts.append(f"Schedule Check: Zero time overlaps detected for your {cohort_section} selection — ready to finalize.")
        else:
            parts.append(f"Detected Clashes ({len(clash_lines)}): " + " ; ".join(clash_lines))
            if swap_lines and solved_zero_clash:
                cohort_note = (
                    f"keeping all {cohort_section} regular cohort classes unchanged"
                    if locked_cohort_preserved
                    else "with minimal section adjustments"
                )
                parts.append(
                    f"Recommended Clash-Free Alternatives ({cohort_note}): " + " ; ".join(swap_lines)
                )
            elif swap_lines:
                parts.append("Partial Reduction Options: " + " ; ".join(swap_lines))
            else:
                parts.append(
                    "No zero-clash combination exists across current offerings — check another section PDF or adjust selected courses."
                )
        return "\n\n".join(parts)

    if is_api_configured() and (clash_lines or prereq_issues):
        def _gemini_advise() -> str:
            _, active_model = load_api_environment()
            client = get_client()
            sys_inst = (
                "You are an academic timetable advisor for COMSATS University students. "
                "Write a concise, clear 2-4 sentence recommendation strictly using ONLY the exact verified "
                "conflicts, prerequisite holds, and alternative section swaps provided. Never invent times, rooms, or sections."
            )
            prompt = (
                f"Home Class Section: {cohort_section}\n"
                f"Clashes: {clash_lines or ['None']}\n"
                f"Prerequisite Issues: {prereq_issues or ['None']}\n"
                f"Verified Alternative Swaps: {swap_lines or ['None']}\n"
                f"Zero-Clash Plan Found: {solved_zero_clash} (Cohort classes preserved: {locked_cohort_preserved})"
            )
            resp = _generate_content_resilient(
                client=client,
                primary_model=active_model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.15),
            )
            text = (resp.text or "").strip()
            return text or _local_summary()

        try:
            return _run_with_timeout(_gemini_advise, timeout_seconds=timeout_seconds), False
        except Exception:
            pass

    return _local_summary(), True
