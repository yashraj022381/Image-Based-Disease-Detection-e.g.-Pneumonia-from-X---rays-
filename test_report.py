from src.genai.report_chain import generate_report
from src.genai.report_chain import generate_report
from src.genai.guardrails import check_report

result = generate_report(
    predicted_label="PNEUMONIA",
    probability=0.95,
    threshold=0.9928,
)

print("=" * 60)
print("GENERATED REPORT")
print("=" * 60)
print(result["report_text"])
print("\n" + "=" * 60)
print(f"Grounded on {len(result['grounding_context'])} documents:")
for doc in result["grounding_context"]:
    print(f"  -   {doc['id']}")
    
check = check_report(result)

print("GUARDRAIL CHECK RESULT")
print("=" * 60)
print(f"Passed: {check['passed']}")
if check["issues"]:
    print("Issues found:")
    for issue in check["issues"]:
        print(f"  - {issue}")
else:
    print("No issues found.")
