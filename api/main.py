import logging
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference import predict_and_explain
from src.genai.report_chain import generate_report
from src.genai.guardrails import check_report
from src.agents.pipeline_graph import run_pipeline


app = FastAPI(
    title="Pneumonia Detection API (Educational Project)",
    description=(
        "⚠️ EDUCATIONAL PROJECT ONLY. This is NOT a real diagnostic tool and "
        "must not be used for actual medical decisions. Always consult a "
        "licensed physician or radiologist."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    predicted_label: str
    probability: float
    threshold: float


@app.get("/health")
def health_check():
    return {"status": "ok", "disclaimer": "Educational project only - not a diagnostic tool."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded chest X-ray image, returns the model's prediction,
    probability, decision threshold, and a Grad-CAM heatmap overlay
    (base64-encoded PNG).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        result = predict_and_explain(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    result["disclaimer"] = (
        "This is an educational AI demonstration, NOT a real medical diagnosis. "
        "Consult a licensed physician or radiologist for any real health concern."
    )
    return result


@app.post("/report")
def report(request: ReportRequest):
    """
    Generates a structured GenAI report based on a prior /predict result.
    The frontend passes back the predicted_label, probability, and threshold
    it received from /predict, so this endpoint doesn't need to re-run
    the model or re-upload the image.
    """
    try:
        result = generate_report(
            predicted_label=request.predicted_label,
            probability=request.probability,
            threshold=request.threshold,
        )
        guardrail_result = check_report(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    return {
        "report_text": result["report_text"],
        "grounding_sources": [doc["id"] for doc in result["grounding_context"]],
        "guardrail_passed": guardrail_result["passed"],
        "guardrail_issues": guardrail_result["issues"],
        "disclaimer": (
            "This AI-generated report is for educational demonstration only "
            "and must be reviewed by a licensed physician or radiologist."
        ),
    }


#logger = logging.getLogger("uvicorn.error")

@app.post("/agentic-report")
async def agentic_report(file: UploadFile = File(...)):
    """
    Runs the full multi-agent pipeline (Classifier -> Explainer ->
    Report Writer -> Safety Checker) in a single call. Unlike /predict +
    /report, this automatically retries report generation if the Safety
    Checker flags an issue, up to a small retry limit.
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        result = run_pipeline(image_bytes)
    except Exception as e:
        #logger.error("Agentic pipeline failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Agentic pipeline failed: {str(e)}")

    result["disclaimer"] = (
        "This is an educational AI demonstration, NOT a real medical diagnosis. "
        "Consult a licensed physician or radiologist for any real health concern."
    )
    return result
