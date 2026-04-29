"""AttachIQ Streamlit demo."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from attachiq.inference.pipeline import get_pipeline
from attachiq.schemas import InferenceRequest

DECISION_COLORS = {
    "ALLOW": "#2e7d32",
    "REVIEW": "#ed6c02",
    "BLOCK": "#c62828",
}


def _decision_badge(decision: str) -> None:
    color = DECISION_COLORS.get(decision, "#555555")
    st.markdown(
        f"""
        <div style="
          display:inline-block;
          padding:8px 18px;
          border-radius:8px;
          background:{color};
          color:white;
          font-weight:700;
          font-size:20px;
          letter-spacing:1px;
        ">{decision}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AttachIQ", layout="centered")
    st.title("AttachIQ")
    st.caption("Multimodal Request-Attachment Triage for AI Assistants")

    pipe = get_pipeline()

    prompt = st.text_area(
        "What are you asking the AI assistant to do?",
        placeholder="e.g. Summarize this slide / Extract the total / Post this publicly",
        height=100,
    )
    uploaded = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

    submitted = st.button("Assess Request", type="primary")
    if not submitted:
        return

    has_text = bool(prompt and prompt.strip())
    has_image = uploaded is not None
    if not has_text and not has_image:
        st.error("Please provide a prompt, an image, or both.")
        return

    image_path = None
    if has_image:
        suffix = Path(uploaded.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        image_path = tmp.name
        st.image(image_path, caption=uploaded.name, width=400)

    try:
        req = InferenceRequest(
            prompt_text=prompt.strip() if has_text else None,
            image_path=image_path,
        )
    except Exception as exc:
        st.error(f"Invalid request: {exc}")
        return

    with st.spinner("Running multimodal triage..."):
        resp = pipe.predict(req)

    st.subheader("Decision")
    _decision_badge(resp.decision)

    col1, col2, col3 = st.columns(3)
    col1.metric("Input mode", resp.input_mode)
    col2.metric("Compatibility", resp.compatibility_label)
    col3.metric("Confidence", f"{resp.confidence:.2f}")

    col4, col5 = st.columns(2)
    col4.metric("Request type", resp.request_type or "—")
    col5.metric("Document type", resp.document_type or "—")

    st.markdown(f"**Explanation.** {resp.explanation}")
    st.caption(f"Inference time: {resp.inference_time_ms:.1f} ms")

    with st.expander("Full JSON output"):
        st.code(json.dumps(resp.model_dump(), indent=2), language="json")


main()
