"""
model_loader.py — CNN Model Loader (Singleton Pattern)

Loads the pre-trained drowsiness detection model once at startup
and provides a prediction interface.

Model details:
- Binary classifier: Open Eyes vs Closed Eyes
- Input shape: (64, 64, 3) — RGB images
- Output: single sigmoid value
  - < 0.5 → Closed Eye
  - ≥ 0.5 → Open Eye
"""

import os
import numpy as np
import tensorflow as tf

# ─── Suppress TensorFlow info/warning logs for cleaner output ────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ─── Model Path ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "drowsiness_model_v3.h5"
)

# ─── Singleton Model Instance ────────────────────────────────────────────────
_model = None


def load_model():
    """
    Load the drowsiness detection CNN model from disk.
    Uses a singleton pattern — the model is loaded only once and reused.
    
    Returns:
        Loaded Keras model instance.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist.
        Exception: If the model fails to load.
    """
    global _model
    
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Please ensure 'drowsiness_model_v3.h5' is in the 'model/' directory."
        )
    
    print(f"[Model Loader] Loading model from: {MODEL_PATH}")
    _model = tf.keras.models.load_model(MODEL_PATH)
    
    # Log model details for debugging
    input_shape = _model.input_shape
    print(f"[Model Loader] [OK] Model loaded successfully")
    print(f"[Model Loader]   Input shape:  {input_shape}")
    print(f"[Model Loader]   Output shape: {_model.output_shape}")
    
    return _model


def predict(preprocessed_image: np.ndarray) -> dict:
    """
    Run prediction on a preprocessed eye image.
    
    Args:
        preprocessed_image: Numpy array of shape (1, 64, 64, 3),
                           normalized to [0, 1].
    
    Returns:
        Dictionary with:
        - 'prediction': str — "Open" or "Closed"
        - 'confidence': float — confidence score (0.0 to 1.0)
        - 'raw_score': float — raw model output (sigmoid value)
    """
    model = load_model()
    
    # Run inference
    raw_output = model.predict(preprocessed_image, verbose=0)
    score = float(raw_output[0][0])
    
    # Interpret result:
    #   score ≥ 0.5 → Open Eye
    #   score < 0.5 → Closed Eye
    if score >= 0.5:
        prediction = "Open"
        confidence = score                # How confident it's open
    else:
        prediction = "Closed"
        confidence = 1.0 - score          # How confident it's closed
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 4)
    }
