"""
Streamlit frontend for the Pneumonia Detection educational project.
Calls the FastAPI backend's /predict and /report endpoints.

⚠️ EDUCATIONAL PROJECT ONLY — NOT a real diagnostic tool.
"""
import os
import base64

import streamlit as st
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

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
    "Upload a chest X-ray image to see an AI-assisted classification, a "
    "Grad-CAM visual explanation of which regions influenced the prediction, "
    "and an optional AI-generated educational report."
)

# Session state holds the full agentic-report result across reruns, so any
# later rerun of the script (e.g. from an unrelated widget interaction)
# doesn't wipe out what's already on screen.
if "agentic_result" not in st.session_state:
    st.session_state.agentic_result = None

# Session state holds the prediction result across reruns, so clicking
# "Generate Report" later doesn't lose the prediction we already have.
#if "prediction_result" not in st.session_state:
#    st.session_state.prediction_result = None

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyze X-ray", type="primary"):
        with st.spinner(
            "Running model inference, Grad-CAM, report generation and "
            "safety checks... this can take a few seconds."
        ):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                   )
                }
                response = requests.post(
                    f"{API_BASE_URL}/agentic-report", files=files, timeout=120
                )
                response.raise_for_status()
                st.session_state.agentic_result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend API. Make sure the "
                    "FastAPI server is running (`uvicorn api.main:app --reload`)."
                )
            except requests.exceptions.HTTPError:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = response.text
                st.error(f"Analysis failed: {detail}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            

# Display prediction results if we have them
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
        # overlay comes back as a data URI ("data:image/png;base64,....").
        # st.image doesn't reliably decode a data-URI string, so strip the
        # prefix and decode to raw bytes ourselves.
        if overlay.startswith("data:image"):
            overlay_b64 = overlay.split(",", 1)[1]
        else:
            overlay_b64 = overlay
        overlay_bytes = base64.b64decode(overlay_b64)
        
        st.image(
            #result["overlay_image_base64"],
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
 
    st.caption(result.get("disclaimer", ""))
    

st.divider()
st.caption(
    "This application is a student/educational demonstration of computer "
    "vision and generative AI techniques. It is NOT a certified medical "
    "device and has NOT been validated for clinical use."
)
