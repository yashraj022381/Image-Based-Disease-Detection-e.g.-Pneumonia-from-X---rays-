from src.inference import predict_and_explain

with open("test_images/sample_pneumonia.jpeg", "rb") as f:
    image_bytes = f.read()
    
result = predict_and_explain(image_bytes)

print("Predicted label:", result["predicted_label"])
print("Probability:", result["probability"])
print("Threshold:", result["threshold"])
print("Overlay base64 length:", len(result["overlay_image_base64"]))
print("Overlay starts with:", result["overlay_image_base64"][:50])
