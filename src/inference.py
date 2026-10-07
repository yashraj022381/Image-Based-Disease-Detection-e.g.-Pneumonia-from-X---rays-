import json
import io
import base64

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib as mpl
from PIL import Image

MODEL_PATH = "models/stage2_final"
CONFIG_PATH = "models/model_config.json"
LAST_CONV_LAYER_NAME = "Conv_1"

_model = None
_config = None


def load_resources():
    """
    Loads the model and config once, caching them in module-level
    variables so repeated calls (e.g. multiple API requests) don't
    reload the ~9MB model from disk every time.
    """
    global _model, _config
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _config is None:
        with open(CONFIG_PATH, "r") as f:
            _config = json.load(f)
    return _model, _config



def preprocess_image_bytes(image_bytes: bytes):
    """
    Takes raw uploaded image bytes, returns a (1, 224, 224, 3) array
    ready for the model, matching the training-time preprocessing
    (resize to 224x224, rescale 0-1).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    array = keras_image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def overlay_gradcam_to_base64(original_rgb_array, heatmap, alpha=0.4):
    """
    Blends the heatmap onto the original image and returns the result
    as a base64-encoded PNG string, ready to embed directly in a JSON
    API response.
    """
    heatmap_resized = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap_img = jet_heatmap_img.resize((224, 224))
    jet_heatmap_array = keras_image.img_to_array(jet_heatmap_img)

    original_uint8 = np.uint8(original_rgb_array * 255)
    overlayed = jet_heatmap_array * alpha + original_uint8
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlayed)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"



def predict_and_explain(image_bytes: bytes) -> dict:
    """
    Full pipeline: raw image bytes in -> prediction + Grad-CAM overlay out.
    This is the single function the API endpoint will call.
    """
    model, config = load_resources()
    threshold = config["decision_threshold"]

    img_array = preprocess_image_bytes(image_bytes)

    probability = float(model.predict(img_array, verbose=0)[0][0])
    predicted_label = "PNEUMONIA" if probability >= threshold else "NORMAL"

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
    overlay_base64 = overlay_gradcam_to_base64(img_array[0], heatmap)

    return {
        "predicted_label": predicted_label,
        "probability": probability,
        "threshold": threshold,
        "overlay_image_base64": overlay_base64,
    }
