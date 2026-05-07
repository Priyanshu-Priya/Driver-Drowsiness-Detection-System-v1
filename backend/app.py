"""
app.py — FastAPI Backend for AI Driver Drowsiness Detection System

Endpoints:
  POST /predict    — Accepts a base64 frame, returns drowsiness prediction
  GET  /reset      — Resets the consecutive closed-eye frame counter
  GET  /logs       — Returns the drowsiness event log
  GET  /status     — Health check

The server also serves the frontend static files.
"""

import os
import sys
import time
import threading
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── Add project root to path for imports ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import load_model, predict
from utils import (
    decode_base64_image,
    detect_eyes,
    preprocess_eye,
    DROWSINESS_THRESHOLD,
)


# ─── Drowsiness State (Thread-Safe) ──────────────────────────────────────────
class DrowsinessTracker:
    """
    Tracks consecutive closed-eye frames and maintains an event log.
    Thread-safe via a lock, since FastAPI can handle concurrent requests.
    """
    
    def __init__(self, threshold: int = DROWSINESS_THRESHOLD):
        self.threshold = threshold
        self.closed_frame_count = 0
        self.is_drowsy = False
        self.event_log = []          # List of {timestamp, event, duration}
        self._lock = threading.Lock()
        self._drowsy_start_time = None
    
    def update(self, is_closed: bool) -> dict:
        """Update tracker with a new frame result."""
        with self._lock:
            if is_closed:
                self.closed_frame_count += 1
                
                # Transition: not drowsy → drowsy
                if self.closed_frame_count >= self.threshold and not self.is_drowsy:
                    self.is_drowsy = True
                    self._drowsy_start_time = time.time()
                    self.event_log.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "event": "DROWSINESS_DETECTED",
                        "closed_frames": self.closed_frame_count
                    })
                    # Keep only last 50 events
                    if len(self.event_log) > 50:
                        self.event_log = self.event_log[-50:]
            else:
                # Eyes opened — check if we were drowsy and log recovery
                if self.is_drowsy and self._drowsy_start_time:
                    duration = round(time.time() - self._drowsy_start_time, 1)
                    self.event_log.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "event": "ALERT_CLEARED",
                        "duration_seconds": duration
                    })
                
                self.closed_frame_count = 0
                self.is_drowsy = False
                self._drowsy_start_time = None
            
            # Calculate drowsiness score (0-100)
            # Ramps up as closed frames approach and exceed threshold
            score = min(100, int((self.closed_frame_count / self.threshold) * 100))
            
            return {
                "closed_frames": self.closed_frame_count,
                "is_drowsy": self.is_drowsy,
                "drowsiness_score": score,
                "threshold": self.threshold
            }
    
    def reset(self):
        """Reset the tracker."""
        with self._lock:
            self.closed_frame_count = 0
            self.is_drowsy = False
            self._drowsy_start_time = None
    
    def get_logs(self) -> list:
        """Get the event log."""
        with self._lock:
            return list(self.event_log)


# ─── Global Tracker ──────────────────────────────────────────────────────────
tracker = DrowsinessTracker()


# ─── Application Lifespan (load model on startup) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CNN model when the server starts."""
    print("=" * 60)
    print("  AI Driver Drowsiness Detection System")
    print("  Starting server...")
    print("=" * 60)
    
    try:
        model = load_model()
        print(f"\n[Server] [OK] Model loaded. Threshold: {DROWSINESS_THRESHOLD} frames")
        print(f"[Server] [OK] Ready for predictions")
        print(f"[Server] --> Open http://localhost:8000 in your browser\n")
    except Exception as e:
        print(f"\n[Server] [FAIL] Failed to load model: {e}")
        raise
    
    yield  # Server is running
    
    print("\n[Server] Shutting down...")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Driver Drowsiness Detection System",
    description="Real-time drowsiness detection using CNN and webcam",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""
    image: str  # Base64-encoded image


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    prediction: str         # "Open" or "Closed"
    confidence: float       # 0.0 to 1.0
    raw_score: float        # Raw sigmoid output
    closed_frames: int      # Consecutive closed-eye frames
    is_drowsy: bool         # Whether drowsiness alert is active
    drowsiness_score: int   # 0-100 score
    threshold: int          # Frame threshold for alert
    face_detected: bool     # Whether a face was found
    timestamp: str          # Server timestamp


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict_drowsiness(request: PredictionRequest):
    """
    Predict drowsiness from a webcam frame.
    
    Flow:
    1. Decode base64 image
    2. Detect face and eyes using Haar Cascades
    3. Preprocess eye region (resize, normalize)
    4. Run CNN prediction
    5. Update drowsiness tracker
    6. Return comprehensive result
    """
    try:
        # Step 1: Decode image
        image = decode_base64_image(request.image)
        
        # Step 2: Detect eyes
        eyes = detect_eyes(image)
        
        if len(eyes) == 0:
            # No face/eyes detected — don't update counter
            # This prevents false alerts when looking away
            return PredictionResponse(
                prediction="No Face",
                confidence=0.0,
                raw_score=0.0,
                closed_frames=tracker.closed_frame_count,
                is_drowsy=tracker.is_drowsy,
                drowsiness_score=min(100, int(
                    (tracker.closed_frame_count / tracker.threshold) * 100
                )),
                threshold=tracker.threshold,
                face_detected=False,
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3]
            )
        
        # Step 3: Preprocess the first detected eye
        preprocessed = preprocess_eye(eyes[0])
        
        # Step 4: Run CNN prediction
        result = predict(preprocessed)
        
        # If we have 2 eyes, predict on both and use the "more closed" result
        if len(eyes) >= 2:
            preprocessed_2 = preprocess_eye(eyes[1])
            result_2 = predict(preprocessed_2)
            
            # If either eye is closed, consider eyes as closed
            if result_2["prediction"] == "Closed" or result["prediction"] == "Closed":
                # Use the result that's more "closed"
                if result_2["raw_score"] < result["raw_score"]:
                    result = result_2
        
        # Step 5: Update drowsiness tracker
        is_closed = result["prediction"] == "Closed"
        tracker_state = tracker.update(is_closed)
        
        # Step 6: Return result
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            raw_score=result["raw_score"],
            closed_frames=tracker_state["closed_frames"],
            is_drowsy=tracker_state["is_drowsy"],
            drowsiness_score=tracker_state["drowsiness_score"],
            threshold=tracker_state["threshold"],
            face_detected=True,
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/reset")
async def reset_tracker():
    """Reset the drowsiness frame counter and alert state."""
    tracker.reset()
    return {"message": "Tracker reset", "status": "ok"}


@app.get("/logs")
async def get_logs():
    """Retrieve the drowsiness event log."""
    return {"logs": tracker.get_logs()}


@app.get("/status")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": True,
        "threshold": DROWSINESS_THRESHOLD,
        "current_closed_frames": tracker.closed_frame_count,
        "is_drowsy": tracker.is_drowsy,
    }


# ─── Serve Frontend Static Files ─────────────────────────────────────────────
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

# Serve the frontend index.html at the root
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place files in frontend/ directory."}


# Mount static files (CSS, JS) 
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
