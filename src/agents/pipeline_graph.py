from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.inference import (
    load_resources,
    preprocess_image_bytes,
    make_gradcam_heatmap,
    overlay_gradcam_to_base64,
    LAST_CONV_LAYER_NAME,
)
from src.genai.report_chain import generate_report
from src.genai.guardrails import check_report




MAX_REPORT_RETRIES = 2


class PipelineState(TypedDict):
    image_bytes: bytes
    predicted_label: Optional[str]
    probability: Optional[float]
    threshold: Optional[float]
    overlay_image_base64: Optional[str]
    report_text: Optional[str]
    grounding_sources: Optional[list]
    guardrail_passed: Optional[bool]
    guardrail_issues: Optional[list]
    retry_count: int


def classifier_node(state: PipelineState) -> PipelineState:
     """Agent 1: runs the CNN model and produces a prediction."""
     model, config = load_resources()
     threshold = config["decision_threshold"]

     img_array = preprocess_image_bytes(state["image_bytes"])
     probability = float(model.predict(img_array, verbose=0)[0][0])
     predicted_label = "PNEUMONIA" if probability >= threshold else "NORMAL"

     state["predicted_label"] = predicted_label
     state["probability"] = probability
     state["threshold"] = threshold
     #state["_img_array"] = img_array  # internal, passed to explainer node
     return state


def explainer_node(state: PipelineState) -> PipelineState:
     """Agent 2: generates the Grad-CAM visual explanation."""
     model, _ = load_resources()
     #img_array = state["_img_array"]
     img_array = preprocess_image_bytes(state["image_bytes"])

     heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
     overlay_base64 = overlay_gradcam_to_base64(img_array[0], heatmap)

     state["overlay_image_base64"] = overlay_base64
     return state


def report_writer_node(state: PipelineState) -> PipelineState:
     """Agent 3: writes the structured, RAG-grounded report."""
     result = generate_report(
         predicted_label=state["predicted_label"],
         probability=state["probability"],
         threshold=state["threshold"],
     )
     state["report_text"] = result["report_text"]
     state["grounding_sources"] = [doc["id"] for doc in result["grounding_context"]]
     return state



def safety_checker_node(state: PipelineState) -> PipelineState:
     """Agent 4: validates the report against guardrail rules."""
     check_input = {
         "report_text": state["report_text"],
         "predicted_label": state["predicted_label"],
         "probability": state["probability"],
    }
     guardrail_result = check_report(check_input)
     state["guardrail_passed"] = guardrail_result["passed"]
     state["guardrail_issues"] = guardrail_result["issues"]
     return state

def route_after_safety_check(state: PipelineState) -> str:
    """
    Decides where to go after the Safety Checker runs:
    - If passed, we're done.
    - If failed but retries remain, loop back to the Report Writer.
    - If failed and out of retries, stop anyway (better to surface a
      flagged report than to loop forever or silently hide the issue).
    """
    if state["guardrail_passed"]:
        return "done"
    if state["retry_count"] < MAX_REPORT_RETRIES:
        state["retry_count"] += 1
        return "retry"
    return "done"

def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("classifier", classifier_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("safety_checker", safety_checker_node)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "explainer")
    graph.add_edge("explainer", "report_writer")
    graph.add_edge("report_writer", "safety_checker")

    graph.add_conditional_edges(
        "safety_checker",
        route_after_safety_check,
        {"retry": "report_writer", "done": END},
    )

    return graph.compile()

def run_pipeline(image_bytes: bytes) -> dict:
    """
    Entry point: runs the full multi-agent pipeline on an uploaded image
    and returns the final state as a plain dict (minus internal fields).
    """
    pipeline = build_pipeline()
    initial_state: PipelineState = {
        "image_bytes": image_bytes,
        "predicted_label": None,
        "probability": None,
        "threshold": None,
        "overlay_image_base64": None,
        "report_text": None,
        "grounding_sources": None,
        "guardrail_passed": None,
        "guardrail_issues": None,
        "retry_count": 0,
    }

    final_state = pipeline.invoke(initial_state)

    return {
        "predicted_label": final_state["predicted_label"],
        "probability": final_state["probability"],
        "threshold": final_state["threshold"],
        "overlay_image_base64": final_state["overlay_image_base64"],
        "report_text": final_state["report_text"],
        "grounding_sources": final_state["grounding_sources"],
        "guardrail_passed": final_state["guardrail_passed"],
        "guardrail_issues": final_state["guardrail_issues"],
        "retries_used": final_state["retry_count"],
    } 

    
