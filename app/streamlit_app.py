"""
Streamlit app for the Pneumonia Detection educational project.

For the hosted deployment (Streamlit Community Cloud), this file runs the
full agentic pipeline IN-PROCESS - no separate FastAPI backend - since
Streamlit Cloud only hosts a single Streamlit app. The FastAPI backend
(api/main.py) and docker-compose.yml are still used for local/Docker
development; see README.

⚠️ EDUCATIONAL PROJECT ONLY — NOT a real diagnostic tool.
"""

import base64
import os
import sys
import traceback

# Make "from src...." imports resolve regardless of Streamlit Cloud's working
# directory: add the repo root (parent of this file's folder) to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# No display in a hosted environment - force matplotlib's headless backend
# before anything imports matplotlib (src.inference does, indirectly).
os.environ.setdefault("MPLBACKEND", "Agg")

import streamlit as st

# Streamlit Cloud secrets (Settings > Secrets) are read via st.secrets; mirror
# into os.environ because src/genai/report_chain.py reads GROQ_API_KEY via
# os.getenv(). Guarded so local runs (where .env + python-dotenv already set
# the real env var) don't break if no secrets.toml exists.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets.toml locally - fine, .env/python-dotenv already handled it

from src.agents.pipeline_graph import run_pipeline

st.set_page_config(
    page_title="Pneumonia Detection (Educational Demo)",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Chest X-Ray Pneumonia Detection")

st.warning(
    "⚠️ **EDUCATIONAL PROJECT ONLY.** This is NOT a real diagnostic tool and "
    "must not be used for actual medical decisions. Always consult a licensed "
    "physician or radiologist for any real health concern.",
    icon="⚠️",
)

st.write(
    "Upload a chest X-ray image to run the full AI pipeline: classification, "
    "a Grad-CAM visual explanation of which regions influenced the prediction, "
    "and an AI-generated, guardrail-checked educational report."
)

if "agentic_result" not in st.session_state:
    st.session_state.agentic_result = None

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyze X-ray", type="primary"):
        with st.spinner(
            "Running model inference, Grad-CAM, report generation and "
            "safety checks... this can take a few seconds (longer on first "
            "run while the model loads)."
        ):
            try:
                image_bytes = uploaded_file.getvalue()
                st.session_state.agentic_result = run_pipeline(image_bytes)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

result = st.session_state.agentic_result
if result is not None:
    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        label = result["predicted_label"]
        if label == "PNEUMONIA":
            st.error(f"**Predicted: {label}**")
        else:
            st.success(f"**Predicted: {label}**")
        st.metric("Model Confidence", f"{result['probability'] * 100:.2f}%")
        st.caption(f"Decision threshold: {result['threshold']}")

    with col2:
        overlay = result["overlay_image_base64"]
        # overlay is a data URI ("data:image/png;base64,....") - decode to
        # raw bytes, since st.image doesn't reliably handle a data-URI string.
        if overlay.startswith("data:image"):
            overlay_b64 = overlay.split(",", 1)[1]
        else:
            overlay_b64 = overlay
        overlay_bytes = base64.b64decode(overlay_b64)

        st.image(
            overlay_bytes,
            caption="Grad-CAM Heatmap Overlay",
            use_container_width=True,
        )

    st.caption(
        "🔥 Red/yellow regions indicate areas that most influenced the "
        "model's prediction. This is a visual explanation tool, not proof "
        "of diagnosis."
    )

    st.divider()
    st.subheader("AI-Generated Educational Report")

    st.markdown(result["report_text"])

    if result["guardrail_passed"]:
        st.success(
            f"✅ Guardrail checks passed (retries used: {result['retries_used']})"
        )
    else:
        st.warning(
            f"⚠️ Guardrail flagged potential issues: "
            f"{', '.join(result['guardrail_issues'])}"
        )

    with st.expander("Grounding sources used"):
        st.write(result["grounding_sources"])

st.divider()
st.caption(
    "This application is a student/educational demonstration of computer "
    "vision and generative AI techniques. It is NOT a certified medical "
    "device and has NOT been validated for clinical use."
)
