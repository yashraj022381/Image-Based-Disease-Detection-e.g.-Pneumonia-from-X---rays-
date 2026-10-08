import re


MANDATORY_DISCLAIMER_PHRASES = [
    "review",
    "physician",
    "radiologist",
]


REQUIRED_SECTIONS = ["FINDINGS", "IMPRESSION", "RECOMMENDATIONS"]


# Phrases that would indicate the AI overstepped into a definitive diagnosis
# rather than a hedged, educational screening statement.
OVERCONFIDENT_PHRASES = [
    "the patient has",
    "confirmed diagnosis",
    "definitely has",
    "is diagnosed with",
]

NEGATION_WORDS = {"not", "no", "never", "without", "non", "cannot", "isn't", "doesn't", "don't", "can't"}
NEGATION_WINDOW = 6  # how many words back to look for a negation

def _is_negated(text_lower: str, phrase: str) -> bool:
    """
    Checks whether a matched phrase is preceded nearby by a negation word,
    e.g. 'does not constitute a confirmed diagnosis' should NOT be flagged,
    even though it contains 'confirmed diagnosis'.
    """
    idx = text_lower.find(phrase)
    if idx == -1:
        return False
    preceding_words = text_lower[:idx].split()[-NEGATION_WINDOW:]
    return any(
        neg_word in word
        for word in preceding_words
        for neg_word in NEGATION_WORDS
    )

    
def normalize_text(text: str) -> str:
    """
    Normalizes smart quotes, non-breaking hyphens/spaces, etc. to plain
    ASCII equivalents, so keyword checks below aren't fooled by styling
    differences in the LLM's output.
    """
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def check_report(report_result: dict) -> dict:
    """
    Runs a battery of rule-based checks against a generated report.
    Returns a dict with pass/fail status and a list of specific issues found.
    """
    issues = []
    text = normalize_text(report_result["report_text"])
    text_lower = text.lower()


    # Check 1: All three required sections are present
    for section in REQUIRED_SECTIONS:
        if section not in text.upper():
            issues.append(f"Missing required section: {section}")


    # Check 2: Mandatory disclaimer language is present somewhere
    disclaimer_found = any(phrase in text_lower for phrase in MANDATORY_DISCLAIMER_PHRASES)
    if not disclaimer_found:
        issues.append(
            "Missing mandatory disclaimer language (expected reference to "
            "physician/radiologist review)"
        )
        

    # Check 3: No overconfident/definitive diagnostic language
    for phrase in OVERCONFIDENT_PHRASES:
        if phrase in text_lower and not _is_negated(text_lower, phrase):
            issues.append(f"Overconfident language detected: '{phrase}'")
            

    # Check 4: The predicted label actually appears in the report
    predicted_label = report_result["predicted_label"]
    if predicted_label.lower() not in text_lower:
        issues.append(
            f"Predicted label '{predicted_label}' not mentioned anywhere in report text"
        )

    # Check 5: Model confidence appears in some reasonable numeric form.
    # LLMs may render the percentage/probability with varying decimal
    # precision (e.g. "99.90%" vs "100%" vs "0.9990" vs "1.00"), so we
    # check several plausible renderings rather than one exact format.
    probability = report_result["probability"]
    candidate_strings = [
        f"{probability * 100:.0f}",  # e.g. "100"
        f"{probability * 100:.1f}",   # e.g. "99.9"
        f"{probability * 100:.2f}",   # e.g. "99.90"
        f"{probability:.2f}",         # e.g. "1.00"
        f"{probability:.3f}",         # e.g. "0.999"
        f"{probability:.4f}",         # e.g. "0.9990"
    ]
    if not any(candidate in text for candidate in candidate_strings):
        issues.append("Model confidence percentage not found in report text")
    

    # Check 6: No grounding document was completely ignored in a way that
    # suggests the wrong context was used (e.g., opposite label's keywords)
    opposite_label = "NORMAL" if predicted_label == "PNEUMONIA" else "PNEUMONIA"
    if opposite_label.lower() in text_lower and predicted_label.lower() not in text_lower:
        issues.append(
            f"Report discusses '{opposite_label}' but not the actual "
            f"predicted label '{predicted_label}' - possible grounding mismatch"
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "checked_sections": REQUIRED_SECTIONS,
    }
