"""AttachIQ Streamlit demo — governance checkpoint UI."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from attachiq.config import DEMO_DIR
from attachiq.inference.pipeline import get_pipeline
from attachiq.schemas import InferenceRequest

DECISION_TONE = {
    "ALLOW": {
        "bg": "#eaf6ee",
        "border": "#bfe3cc",
        "fg": "#1f6b3a",
        "microcopy": "No drama detected.",
    },
    "REVIEW": {
        "bg": "#fff5e3",
        "border": "#ecd6a3",
        "fg": "#8a4f00",
        "microcopy": "Worth a human glance.",
    },
    "BLOCK": {
        "bg": "#fbe9e9",
        "border": "#eec1c1",
        "fg": "#9a1f1f",
        "microcopy": "Not today.",
    },
}

DEMO_PRESETS: list[dict[str, str | None]] = [
    {"label": "— None (use upload) —", "prompt": None, "image": None},
    {
        "label": "Summarize a slide",
        "prompt": "Summarize this slide",
        "image": "presentation_demo.png",
    },
    {
        "label": "Extract a total from an invoice",
        "prompt": "Extract the total amount",
        "image": "invoice_demo.png",
    },
    {
        "label": "Post a resume publicly",
        "prompt": "Post this publicly",
        "image": "resume_demo.png",
    },
    {
        "label": "Mismatch: extract a total from a slide",
        "prompt": "Extract the total amount",
        "image": "presentation_demo.png",
    },
    {
        "label": "Archive a letter",
        "prompt": "Archive this, do not delete it.",
        "image": "letter_demo.png",
    },
]

REQUEST_HUMAN: dict[str, str] = {
    "summarization": "The request appears to be asking for a summary.",
    "information_extraction": "The request appears to be asking for information extraction.",
    "financial_extraction": "The request appears to be asking to extract financial information.",
    "document_classification": "The request appears to be asking what kind of document this is.",
    "internal_sharing": "The request appears to be asking for internal sharing.",
    "public_sharing": "The request appears to be asking for public sharing.",
    "delete_permanent": "The request appears to be asking for permanent deletion.",
    "archive_retain": "The request appears to be asking to keep or archive the document.",
    "ambiguous_or_unclear": "The request is not very clear, so the system treats it cautiously.",
    "redaction_or_safe_transform": (
        "The request appears to be asking for a safer transformation, such as redaction."
    ),
}

DOCUMENT_HUMAN: dict[str, str] = {
    "invoice": "The attachment looks like an invoice.",
    "form": "The attachment looks like a form.",
    "letter": "The attachment looks like a letter.",
    "report": "The attachment looks like a report.",
    "email": "The attachment looks like an email.",
    "resume": "The attachment looks like a resume.",
    "presentation": "The attachment looks like a presentation.",
    "handwritten": "The attachment looks like a handwritten document.",
}

COMPAT_HUMAN: dict[str, str] = {
    "compatible_low_risk": "This combination looks compatible and low-risk.",
    "compatible_sensitive": "This combination looks compatible, but sensitive enough to deserve review.",
    "mismatch_unclear": (
        "The request and attachment do not line up cleanly, so the system treats it as unclear."
    ),
    "unsafe_external_action": "This combination suggests an unsafe or high-risk action.",
}

DECISION_TAIL: dict[str, str] = {
    "ALLOW": "So AttachIQ allows it.",
    "REVIEW": "So AttachIQ recommends human review.",
    "BLOCK": "So AttachIQ blocks it.",
}

DECISION_SUMMARY: dict[str, str] = {
    "ALLOW": "AttachIQ thinks this looks like a routine request that can proceed.",
    "REVIEW": (
        "AttachIQ thinks this request can proceed, but the situation is sensitive "
        "enough to deserve a second look."
    ),
    "BLOCK": "AttachIQ thinks this combination is risky and should not run as-is.",
}

MISMATCH_SUMMARY = "AttachIQ thinks the request and the attachment do not fit cleanly together."

MODE_HUMAN: dict[str, str] = {
    "text_only": "Based on the prompt only (no attachment was provided).",
    "image_only": "Based on the attachment only (no prompt was provided).",
    "text_plus_image": "Based on both the prompt and the attachment.",
}


CSS = """
<style>
:root {
  --aiq-bg: #f8f5ef;
  --aiq-bg-soft: #fdfbf6;
  --aiq-surface: #ffffff;
  --aiq-border: #ece6d9;
  --aiq-border-strong: #d8d0bd;
  --aiq-text: #172033;
  --aiq-muted: #64748b;
  --aiq-accent: #1f4e79;
  --aiq-accent-soft: #e7eef7;
  --aiq-accent-border: #c9d6e8;
  --aiq-gold: #c8a45d;
  --aiq-gold-soft: #f6efdd;
  --aiq-shadow: 0 1px 2px rgba(23, 32, 51, 0.05),
                0 6px 22px rgba(23, 32, 51, 0.06);
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
  background: var(--aiq-bg) !important;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif !important;
  color: var(--aiq-text);
}

section.main > div.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}

.aiq-hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  padding: 0 0 1.1rem 0;
  border-bottom: 1px solid var(--aiq-border);
  margin-bottom: 1.4rem;
}
.aiq-hero h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--aiq-text);
}
.aiq-hero .aiq-sub {
  color: var(--aiq-muted);
  font-size: 0.95rem;
  margin-top: 0.15rem;
}
.aiq-hero .aiq-tag {
  display: inline-block;
  padding: 3px 10px;
  background: var(--aiq-gold-soft);
  color: #8a6e2a;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  border: 1px solid #ead9b1;
  vertical-align: middle;
  margin-left: 0.5rem;
}

.aiq-card {
  background: var(--aiq-surface);
  border: 1px solid var(--aiq-border);
  border-radius: 14px;
  padding: 1.0rem 1.15rem;
  box-shadow: var(--aiq-shadow);
  margin-bottom: 0.95rem;
}
.aiq-card h3 {
  margin: 0 0 0.6rem 0;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--aiq-muted);
}
.aiq-card .aiq-card-title {
  margin: 0 0 0.3rem 0;
  font-size: 1.02rem;
  font-weight: 600;
  color: var(--aiq-text);
}
.aiq-card .aiq-card-hint {
  color: var(--aiq-muted);
  font-size: 0.85rem;
  margin-bottom: 0.2rem;
}

.aiq-decision {
  border-radius: 14px;
  border: 1px solid;
  padding: 1.15rem 1.3rem;
  margin-bottom: 0.95rem;
  box-shadow: var(--aiq-shadow);
}
.aiq-decision .aiq-decision-row {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.aiq-badge {
  display: inline-block;
  padding: 8px 18px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.95rem;
  letter-spacing: 0.08em;
  border: 1px solid;
  background: white;
}
.aiq-decision .aiq-microcopy {
  font-size: 1.05rem;
  font-weight: 600;
}

.aiq-why h4 {
  margin: 0 0 0.45rem 0;
  font-size: 1.02rem;
  font-weight: 600;
  color: var(--aiq-text);
}
.aiq-why ul {
  padding-left: 1.05rem;
  margin: 0.4rem 0 0.45rem 0;
}
.aiq-why ul li {
  margin: 0.2rem 0;
  font-size: 0.93rem;
  line-height: 1.45;
  color: var(--aiq-text);
}
.aiq-why .aiq-why-tail {
  margin-top: 0.3rem;
  font-size: 0.93rem;
  color: var(--aiq-text);
}
.aiq-why .aiq-why-foot {
  margin-top: 0.55rem;
  font-size: 0.78rem;
  color: var(--aiq-muted);
}

.aiq-tech {
  font-size: 0.9rem;
  color: var(--aiq-text);
  line-height: 1.5;
}

.aiq-metric {
  background: var(--aiq-bg-soft);
  border: 1px solid var(--aiq-border);
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  height: 100%;
}
.aiq-metric .aiq-metric-label {
  color: var(--aiq-muted);
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.18rem;
}
.aiq-metric .aiq-metric-value {
  font-size: 0.98rem;
  font-weight: 600;
  word-break: break-word;
  color: var(--aiq-text);
}
.aiq-metric.muted .aiq-metric-value { color: var(--aiq-muted); }

.aiq-pipeline {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.aiq-stage {
  background: var(--aiq-bg-soft);
  border: 1px solid var(--aiq-border);
  border-radius: 12px;
  padding: 0.65rem 0.7rem;
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  transition: all 200ms ease;
}
.aiq-stage .aiq-stage-icon {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
  background: #eef0f3;
  color: var(--aiq-muted);
  border: 1px solid var(--aiq-border-strong);
}
.aiq-stage .aiq-stage-name {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--aiq-text);
}
.aiq-stage .aiq-stage-status {
  font-size: 0.78rem;
  color: var(--aiq-muted);
  margin-top: 0.1rem;
  line-height: 1.35;
}
.aiq-stage.active {
  background: var(--aiq-accent-soft);
  border-color: var(--aiq-accent-border);
}
.aiq-stage.active .aiq-stage-icon {
  background: var(--aiq-accent);
  color: white;
  border-color: var(--aiq-accent);
}
.aiq-stage.active .aiq-stage-status {
  color: var(--aiq-accent);
}
.aiq-stage.done {
  background: #f1f7f2;
  border-color: #cfe5d6;
}
.aiq-stage.done .aiq-stage-icon {
  background: #2f7a48;
  color: white;
  border-color: #2f7a48;
}
.aiq-stage.skipped {
  background: var(--aiq-bg-soft);
  border-style: dashed;
}
.aiq-stage.skipped .aiq-stage-icon {
  background: white;
  color: var(--aiq-muted);
}

.aiq-empty {
  border: 1px dashed var(--aiq-border-strong);
  border-radius: 14px;
  padding: 1.2rem 1.1rem;
  text-align: center;
  color: var(--aiq-muted);
  background: var(--aiq-bg-soft);
}

.aiq-caveat {
  background: var(--aiq-bg-soft);
  border: 1px solid var(--aiq-border);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  color: var(--aiq-muted);
  font-size: 0.82rem;
  line-height: 1.5;
}

.aiq-img-wrap {
  border: 1px solid var(--aiq-border);
  background: var(--aiq-surface);
  border-radius: 12px;
  padding: 0.5rem;
  display: inline-block;
  box-shadow: var(--aiq-shadow);
  margin-bottom: 0.95rem;
  max-width: 100%;
}
[data-testid="stImage"] img {
  max-height: 300px !important;
  width: auto !important;
  border-radius: 8px;
  object-fit: contain;
}

div[data-testid="stFileUploaderDropzone"] {
  background: var(--aiq-bg-soft);
  border: 1.5px dashed var(--aiq-border-strong) !important;
  border-radius: 12px !important;
}

textarea {
  border-radius: 10px !important;
  border-color: var(--aiq-border-strong) !important;
  background: var(--aiq-surface) !important;
}

.stButton > button[kind="primary"] {
  background: var(--aiq-accent);
  border: 1px solid var(--aiq-accent);
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 0.6rem 1.1rem;
}
.stButton > button[kind="primary"]:hover {
  background: #194168;
  border-color: #194168;
}
</style>
"""


def _render_hero() -> None:
    st.markdown(
        """
        <div class="aiq-hero">
          <div>
            <h1>AttachIQ <span class="aiq-tag">LOCAL · MULTIMODAL · GOVERNED</span></h1>
            <div class="aiq-sub">Prompt + document triage before the assistant acts.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_legend() -> None:
    st.markdown(
        """
        <div class="aiq-card">
          <h3>How to read this</h3>
          <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px;">
            <div class="aiq-metric">
              <div class="aiq-metric-label" style="color:#1f6b3a;">ALLOW</div>
              <div class="aiq-metric-value" style="font-weight:500; font-size:0.85rem; color:#172033;">
                Compatible and low-risk.
              </div>
            </div>
            <div class="aiq-metric">
              <div class="aiq-metric-label" style="color:#8a4f00;">REVIEW</div>
              <div class="aiq-metric-value" style="font-weight:500; font-size:0.85rem; color:#172033;">
                Sensitive or unclear; worth a human glance.
              </div>
            </div>
            <div class="aiq-metric">
              <div class="aiq-metric-label" style="color:#9a1f1f;">BLOCK</div>
              <div class="aiq-metric-value" style="font-weight:500; font-size:0.85rem; color:#172033;">
                Unsafe external or destructive action.
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----- Pipeline visualization ------------------------------------------------

def _stage_html(name: str, status: str, idx: int, state: str) -> str:
    """state ∈ {'inactive','active','done','skipped'}."""
    icon: str
    if state == "done":
        icon = "✓"
    elif state == "skipped":
        icon = "—"
    elif state == "active":
        icon = "•"
    else:
        icon = str(idx)
    return (
        f'<div class="aiq-stage {state}">'
        f'<div class="aiq-stage-icon">{icon}</div>'
        f'<div>'
        f'<div class="aiq-stage-name">{name}</div>'
        f'<div class="aiq-stage-status">{status}</div>'
        f"</div>"
        f"</div>"
    )


def _pipeline_html(stages: list[tuple[str, str, str]]) -> str:
    """stages: list of (name, status_text, state)."""
    inner = "".join(
        _stage_html(name, status, i + 1, state)
        for i, (name, status, state) in enumerate(stages)
    )
    return (
        '<div class="aiq-card">'
        "<h3>Assessment pipeline</h3>"
        f'<div class="aiq-pipeline">{inner}</div>'
        "</div>"
    )


def _animate_pipeline(slot, has_text: bool, has_image: bool) -> None:
    """Animate stages sequentially. Total ~1.0s. Inference is separate."""
    base = [
        ["Text branch",
         "Reading request intent…" if has_text else "Skipped — no prompt.",
         "inactive" if has_text else "skipped"],
        ["Image branch",
         "Classifying document type…" if has_image else "Skipped — no image.",
         "inactive" if has_image else "skipped"],
        ["Fusion layer", "Combining signals + uncertainty…", "inactive"],
        ["Governance decision", "Mapping to ALLOW / REVIEW / BLOCK…", "inactive"],
    ]
    slot.markdown(_pipeline_html([tuple(s) for s in base]), unsafe_allow_html=True)
    time.sleep(0.15)

    for i in range(4):
        if base[i][2] == "skipped":
            continue
        base[i][2] = "active"
        slot.markdown(_pipeline_html([tuple(s) for s in base]), unsafe_allow_html=True)
        time.sleep(0.25)


def _finish_pipeline(slot, has_text: bool, has_image: bool) -> None:
    final = [
        ("Text branch",
         "Read the request intent." if has_text else "Skipped — no prompt.",
         "done" if has_text else "skipped"),
        ("Image branch",
         "Classified the document type." if has_image else "Skipped — no image.",
         "done" if has_image else "skipped"),
        ("Fusion layer", "Combined signals with uncertainty.", "done"),
        ("Governance decision", "Mapped to a final decision.", "done"),
    ]
    slot.markdown(_pipeline_html(final), unsafe_allow_html=True)


def _idle_pipeline(slot) -> None:
    idle = [
        ("Text branch", "Awaiting prompt.", "inactive"),
        ("Image branch", "Awaiting attachment.", "inactive"),
        ("Fusion layer", "Will combine signals + uncertainty.", "inactive"),
        ("Governance decision", "Will map to ALLOW / REVIEW / BLOCK.", "inactive"),
    ]
    slot.markdown(_pipeline_html(idle), unsafe_allow_html=True)


# ----- Decision rendering ----------------------------------------------------

def _render_decision(resp) -> None:
    tone = DECISION_TONE.get(resp.decision, DECISION_TONE["REVIEW"])
    st.markdown(
        f"""
        <div class="aiq-decision" style="background:{tone['bg']}; border-color:{tone['border']};">
          <div class="aiq-decision-row">
            <span class="aiq-badge"
                  style="color:{tone['fg']}; border-color:{tone['border']};">
              {resp.decision}
            </span>
            <span class="aiq-microcopy" style="color:{tone['fg']};">{tone['microcopy']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str | None, muted: bool = False) -> str:
    val = value if value not in (None, "") else "—"
    cls = "aiq-metric muted" if (muted or val == "—") else "aiq-metric"
    return (
        f'<div class="{cls}">'
        f'<div class="aiq-metric-label">{label}</div>'
        f'<div class="aiq-metric-value">{val}</div>'
        f"</div>"
    )


# ----- Human-language explanation -------------------------------------------

def _human_explanation(resp) -> tuple[str, list[str], str, str]:
    """Build a (summary, bullets, closing, footer) tuple from a response."""
    decision = resp.decision
    compat = resp.compatibility_label
    request_type = resp.request_type
    document_type = resp.document_type
    mode = resp.input_mode
    confidence = resp.confidence

    if compat == "mismatch_unclear":
        summary = MISMATCH_SUMMARY
    else:
        summary = DECISION_SUMMARY.get(decision, DECISION_SUMMARY["REVIEW"])

    bullets: list[str] = []
    if request_type:
        bullets.append(
            REQUEST_HUMAN.get(
                request_type,
                f"The request appears to involve {request_type.replace('_', ' ')}.",
            )
        )
    elif mode == "image_only":
        bullets.append("No prompt was provided, so the request side was left blank.")

    if document_type:
        bullets.append(
            DOCUMENT_HUMAN.get(
                document_type,
                f"The attachment looks like a {document_type.replace('_', ' ')}.",
            )
        )
    elif mode == "text_only":
        bullets.append("No attachment was provided, so the document side was left blank.")

    if compat:
        bullets.append(
            COMPAT_HUMAN.get(compat, "The system flagged this combination for triage.")
        )

    tail_decision = DECISION_TAIL.get(decision, "")
    closing = f"That is why AttachIQ returned {decision}. {tail_decision}".strip()

    mode_line = MODE_HUMAN.get(mode, "")
    footer_bits = [f"Fusion confidence {confidence:.2f}"]
    if mode_line:
        footer_bits.append(mode_line)
    footer = " · ".join(footer_bits)

    return summary, bullets, closing, footer


def _render_why(resp) -> None:
    summary, bullets, closing, footer = _human_explanation(resp)
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets) if bullets else ""
    st.markdown(
        f"""
        <div class="aiq-card aiq-why">
          <h3>Why this result</h3>
          <h4>{summary}</h4>
          {f"<ul>{bullets_html}</ul>" if bullets_html else ""}
          <div class="aiq-why-tail">{closing}</div>
          <div class="aiq-why-foot">{footer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_technical(resp) -> None:
    st.markdown(
        f"""
        <div class="aiq-card">
          <h3>Technical explanation</h3>
          <div class="aiq-tech">{resp.explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----- File handling ---------------------------------------------------------

def _resolve_demo_image(filename: str | None) -> str | None:
    if not filename:
        return None
    candidate = DEMO_DIR / filename
    return str(candidate) if candidate.exists() else None


def _materialize_uploaded_image(uploaded) -> str | None:
    """Persist the uploaded image to a safe temp file and return its path."""
    try:
        suffix = Path(uploaded.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        return None


# ----- Main ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AttachIQ — Governance checkpoint",
        page_icon="✦",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    _render_hero()

    pipe = get_pipeline()

    left, right = st.columns([42, 58], gap="large")

    with left:
        st.markdown(
            '<div class="aiq-card">'
            '<div class="aiq-card-title">1. What is the user asking?</div>'
            '<div class="aiq-card-hint">Plain language. The text branch handles paraphrases and short forms.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        prompt = st.text_area(
            "Prompt",
            placeholder="e.g. Summarize this slide / Extract the total amount / Post this publicly",
            height=140,
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="aiq-card">'
            '<div class="aiq-card-title">2. What is attached?</div>'
            '<div class="aiq-card-hint">Drag and drop a document image, or pick a packaged demo sample.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload a document image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        preset_labels = [p["label"] for p in DEMO_PRESETS]
        choice = st.selectbox("Or load a demo case", preset_labels, index=0)
        preset = next(p for p in DEMO_PRESETS if p["label"] == choice)

        effective_prompt = (prompt or "").strip()
        if not effective_prompt and preset["prompt"]:
            effective_prompt = preset["prompt"]

        demo_image_path = None
        if uploaded is None and preset["image"]:
            demo_image_path = _resolve_demo_image(preset["image"])

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        submitted = st.button("Assess request", type="primary", use_container_width=True)

    with right:
        pipeline_slot = st.empty()

        if not submitted:
            _idle_pipeline(pipeline_slot)
            st.markdown(
                """
                <div class="aiq-empty">
                  <div style="font-size:1.05rem; font-weight:600; color:#172033;">
                    Governance checkpoint
                  </div>
                  <div style="margin-top:0.35rem;">
                    Give me a prompt, an attachment, or both.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_legend()
            return

        has_text = bool(effective_prompt)
        has_uploaded = uploaded is not None
        has_demo_image = demo_image_path is not None
        has_image = has_uploaded or has_demo_image

        if not has_text and not has_image:
            _idle_pipeline(pipeline_slot)
            st.warning("Give me a prompt, an attachment, or both.")
            return

        image_path: str | None = None
        if has_uploaded:
            image_path = _materialize_uploaded_image(uploaded)
            if image_path is None:
                _idle_pipeline(pipeline_slot)
                st.error("That image could not be read. Try a different PNG or JPEG.")
                return
        elif has_demo_image:
            image_path = demo_image_path

        if image_path:
            try:
                st.markdown('<div class="aiq-img-wrap">', unsafe_allow_html=True)
                st.image(
                    image_path,
                    caption=(uploaded.name if has_uploaded else preset["image"]),
                )
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                st.warning("Image preview failed; the pipeline will still try to read the file.")

        try:
            req = InferenceRequest(
                prompt_text=effective_prompt if has_text else None,
                image_path=image_path,
            )
        except Exception as exc:
            _idle_pipeline(pipeline_slot)
            st.error(f"Invalid request: {exc}")
            return

        _animate_pipeline(pipeline_slot, has_text=has_text, has_image=has_image)

        try:
            resp = pipe.predict(req)
        except FileNotFoundError as exc:
            _idle_pipeline(pipeline_slot)
            st.error(f"Could not load attachment: {exc}")
            return
        except Exception as exc:
            _idle_pipeline(pipeline_slot)
            st.error(f"Pipeline error: {exc}")
            return

        _finish_pipeline(pipeline_slot, has_text=has_text, has_image=has_image)

        _render_decision(resp)
        _render_why(resp)
        _render_technical(resp)

        st.markdown(
            f"""
            <div class="aiq-card">
              <h3>Fusion checkpoint</h3>
              <div style="display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px;">
                {_metric_card("Decision", resp.decision)}
                {_metric_card("Compatibility", resp.compatibility_label)}
                {_metric_card("Confidence", f"{resp.confidence:.2f}")}
              </div>
            </div>

            <div class="aiq-card">
              <h3>Per-modality read</h3>
              <div style="display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px;">
                {_metric_card("Input mode", resp.input_mode)}
                {_metric_card("Request type (text branch)", resp.request_type, muted=resp.request_type is None)}
                {_metric_card("Document type (image branch)", resp.document_type, muted=resp.document_type is None)}
              </div>
              <div style="margin-top:0.6rem; color:#64748b; font-size:0.82rem;">
                Inference time: {resp.inference_time_ms:.1f} ms · local pipeline, no external calls.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Raw JSON response", expanded=False):
            st.code(json.dumps(resp.model_dump(), indent=2), language="json")

        st.markdown(
            """
            <div class="aiq-caveat">
              AttachIQ is a triage demo. It classifies request-document handling risk;
              it does not read document text and does not make legal or compliance decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


main()
