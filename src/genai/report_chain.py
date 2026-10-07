import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.genai.rag_store import get_grounding_context

load_dotenv()

REPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant generating a structured, educational chest
X-ray screening report. You must ONLY use facts from the CONTEXT below.
Do not invent clinical details, patient history, or findings not grounded
in the CONTEXT or the MODEL OUTPUT. If something isn't in the CONTEXT,
do not claim it.

MODEL OUTPUT:
- Predicted classification: {predicted_label}
- Model confidence (probability of Pneumonia): {probability:.2%}
- Decision threshold used: {threshold}

CONTEXT (retrieved reference material - your only source of medical facts):
{context}

Write a structured report with exactly these three sections:

FINDINGS:
(2-3 sentences describing what the image pattern suggests, grounded in the
CONTEXT's findings description. Mention this is an AI-assisted screening
result, not a confirmed diagnosis.)

IMPRESSION:
(1-2 sentences summarizing the overall impression, stating the predicted
label and confidence level clearly.)

RECOMMENDATIONS:
(2-3 sentences based on the CONTEXT's recommendation guidance. Must include
a statement that this requires review by a licensed physician/radiologist.)

Keep the tone professional, cautious, and clearly educational - not a
real diagnosis. Do not use definitive language like "the patient has" -
use language like "the image pattern is consistent with" instead.
"""
)


def generate_report(predicted_label: str, probability: float, threshold: float) -> dict:
    context_docs = get_grounding_context(predicted_label, probability)
    context_text = "\n\n".join([f"[{doc['id']}]: {doc['text']}" for doc in context_docs])

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    chain = REPORT_PROMPT | llm | StrOutputParser()

    report_text = chain.invoke({
        "predicted_label": predicted_label,
        "probability": probability,
        "threshold": threshold,
        "context": context_text,
    })

    return {
        "report_text": report_text,
        "predicted_label": predicted_label,
        "probability": probability,
        "threshold": threshold,
        "grounding_context": context_docs,
    }
    
