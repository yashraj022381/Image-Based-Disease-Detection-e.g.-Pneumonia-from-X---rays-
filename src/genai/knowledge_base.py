"""
Simulated radiology knowledge base.
In a real production system, this would be sourced from actual clinical
guidelines (e.g., Fleischner Society, ACR Appropriateness Criteria).
These are simplified, illustrative snippets for an EDUCATIONAL project only.
"""

RADIOLOGY_KNOWLEDGE_BASE = [
    {
        "id": "doc_pneumonia_findings",
        "topic": "pneumonia_findings",
        "text": (
            "Pneumonia on chest radiographs typically presents as focal or "
            "multifocal areas of increased opacity (consolidation) within the "
            "lung fields, often with ill-defined margins. Air bronchograms may "
            "be visible within areas of consolidation. Findings can be lobar, "
            "segmental, or patchy/diffuse depending on the causative organism."
        ),
    },
    {
        "id": "doc_normal_findings",
        "topic": "normal_findings",
        "text": (
            "A normal chest radiograph shows clear, well-aerated lung fields "
            "without focal consolidation, effusion, or significant opacity. "
            "Cardiac silhouette size and mediastinal contours are within "
            "expected limits, and costophrenic angles are typically sharp."
        ),
    },
    {
        "id": "doc_recommendation_pneumonia",
        "topic": "recommendation_pneumonia",
        "text": (
            "When radiographic findings are suggestive of pneumonia, clinical "
            "correlation with patient symptoms (fever, cough, dyspnea), "
            "laboratory markers (white blood cell count, CRP), and physical "
            "examination is recommended. Follow-up imaging may be warranted "
            "to confirm resolution after treatment, particularly in high-risk "
            "patients."
        ),
    },
    {
        "id": "doc_recommendation_normal",
        "topic": "recommendation_normal",
        "text": (
            "When no acute radiographic abnormality is identified, but clinical "
            "suspicion for infection or other pathology remains, correlation "
            "with clinical findings and consideration of repeat imaging or "
            "additional workup (e.g., CT chest) may be appropriate if symptoms "
            "persist."
        ),
    },
    {
        "id": "doc_limitations",
        "topic": "limitations",
        "text": (
            "Chest radiographs have known limitations in sensitivity for early "
            "or mild pneumonia, and findings can be subtle or mimicked by other "
            "conditions such as atelectasis, pulmonary edema, or malignancy. "
            "A single imaging modality and AI-assisted interpretation should "
            "never replace comprehensive clinical evaluation by a qualified "
            "physician."
        ),
    },
    {
        "id": "doc_ai_disclaimer",
        "topic": "ai_disclaimer",
        "text": (
            "AI-based image classification tools are intended to assist, not "
            "replace, clinical decision-making. Performance metrics derived "
            "from research datasets may not generalize to all patient "
            "populations, imaging equipment, or acquisition protocols. Any "
            "AI-generated output should be reviewed and confirmed by a "
            "licensed radiologist or physician before clinical use."
        ),
    },
]
