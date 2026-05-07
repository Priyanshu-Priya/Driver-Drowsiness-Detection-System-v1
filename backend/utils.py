"""
utils.py — Image Processing Utilities for Drowsiness Detection

Handles:
- Base64 image decoding
- Face and eye detection using Haar Cascades
- Eye region preprocessing for CNN input
"""

import cv2
import numpy as np
import base64
import os

# ─── Constants ───────────────────────────────────────────────────────────────
DROWSINESS_THRESHOLD = 20       # Consecutive closed-eye frames to trigger alert
EYE_INPUT_SIZE = (64, 64)       # CNN expected input dimensions
EYE_INPUT_CHANNELS = 3          # RGB — model trained with color_mode='rgb'

# ─── Load Haar Cascades (bundled with OpenCV) ────────────────────────────────
CASCADE_DIR = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
)

# Validate cascades loaded successfully
if face_cascade.empty():
    raise RuntimeError("Failed to load face cascade classifier")
if eye_cascade.empty():
    raise RuntimeError("Failed to load eye cascade classifier")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into an OpenCV BGR image.
    
    Args:
        base64_string: Base64-encoded JPEG/PNG image data.
                       May include 'data:image/...;base64,' prefix.
    
    Returns:
        OpenCV image as numpy array (BGR format).
    
    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Strip the data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    # Decode base64 → bytes → numpy array → OpenCV image
    image_bytes = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image from base64 data")
    
    return image


def detect_eyes(image: np.ndarray) -> list[np.ndarray]:
    """
    Detect eyes in an image using Haar Cascade classifiers.
    
    Strategy:
    1. Detect face first to narrow the search region
    2. Within each face, detect eyes
    3. If no eyes found via cascade, fall back to upper-face crop
    
    Args:
        image: OpenCV BGR image.
    
    Returns:
        List of cropped eye region images (BGR). 
        Empty list if no face detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cropped = []
    
    # Step 1: Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    if len(faces) == 0:
        return []
    
    # Use the largest face (closest to camera)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = largest_face
    
    # Extract face region
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]
    face_roi_color = image[fy:fy + fh, fx:fx + fw]
    
    # Step 2: Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(eyes) > 0:
        # Crop each detected eye region (from the color image)
        for (ex, ey, ew, eh) in eyes[:2]:  # Take at most 2 eyes
            eye_crop = face_roi_color[ey:ey + eh, ex:ex + ew]
            eyes_cropped.append(eye_crop)
    else:
        # Fallback: crop the upper half of the face (eye region approximation)
        # Eyes are typically in the upper 40-65% of the face
        upper_y = int(fh * 0.25)
        lower_y = int(fh * 0.55)
        left_x = int(fw * 0.1)
        right_x = int(fw * 0.9)
        
        eye_region = face_roi_color[upper_y:lower_y, left_x:right_x]
        if eye_region.size > 0:
            eyes_cropped.append(eye_region)
    
    return eyes_cropped


def preprocess_eye(eye_image: np.ndarray) -> np.ndarray:
    """
    Preprocess an eye region image for CNN prediction.
    
    Steps:
    1. Resize to 64x64
    2. Keep as RGB (3 channels) — model was trained on RGB
    3. Normalize pixel values to [0, 1]
    4. Reshape to batch format (1, 64, 64, 3)
    
    Args:
        eye_image: Cropped eye region (BGR format from OpenCV).
    
    Returns:
        Preprocessed numpy array of shape (1, 64, 64, 3).
    """
    # Resize to model input dimensions
    resized = cv2.resize(eye_image, EYE_INPUT_SIZE)
    
    # Convert BGR (OpenCV default) → RGB (what the model expects)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Reshape to batch format: (1, 64, 64, 3)
    batch = normalized.reshape(1, EYE_INPUT_SIZE[0], EYE_INPUT_SIZE[1], EYE_INPUT_CHANNELS)
    
    return batch
