from src.agents.pipeline_graph import run_pipeline

with open("test_images/sample_pneumonia.jpeg", "rb") as f:
    image_bytes = f.read()

result = run_pipeline(image_bytes)

print("Predicted label:", result["predicted_label"])
print("Probability:", result["probability"])
print("Guardrail passed:", result["guardrail_passed"])
print("Guardrail issues:", result["guardrail_issues"])
print("Retries used:", result["retries_used"])
print("\n--- REPORT ---\n")
print(result["report_text"])
