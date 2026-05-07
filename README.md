# 🛡️ AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system that uses a CNN deep learning model, OpenCV, and a modern web interface to monitor eye state and alert when drowsiness is detected.

> **Project History**: This is the advanced, production-ready iteration. The first version (V1) of this project was created as an experimental implementation for learning and research purposes in computer vision and deep learning. 
> 
> 🔗 **Current Version Repository**: [Priyanshu-Priya/Driver-Drowsiness-Detection-System](https://github.com/Priyanshu-Priya/Driver-Drowsiness-Detection-System)

> 📓 **Model Training Notebook**: [Kaggle Notebook](https://www.kaggle.com/code/PriyanshuPriyaLabs/ai-driver-drowsiness-detection-system)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)

---

## 🎯 Features

- **Real-time webcam monitoring** via WebRTC
- **CNN-based eye state classification** (Open vs Closed)
- **Drowsiness score** (0-100) with visual ring indicator
- **Visual alert overlay** with flashing warning
- **Audio alert** using Web Audio API (no external files needed)
- **Event logging** with timestamps
- **Responsive dark-themed UI** with glassmorphism design
- **100% local** — no external APIs or cloud services

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER                                 │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │  Webcam   │───▶│ Capture Frame│───▶│ POST /predict (Base64)│ │
│  │ (WebRTC)  │    │ every 200ms  │    │                       │ │
│  └──────────┘    └──────────────┘    └───────────┬───────────┘ │
│                                                   │             │
│  ┌──────────────────────────────────────────────┐ │             │
│  │ Update UI: Status, Score, Alerts, Event Log  │◀┘             │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                    HTTP POST │ (JSON + Base64)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│                                                                 │
│  1. Decode Base64 image                                         │
│  2. Detect face → Detect eyes (Haar Cascades)                   │
│  3. Crop & resize eye region to 64×64 RGB                       │
│  4. Normalize pixels (/255) → CNN prediction                    │
│  5. Track consecutive "closed" frames                           │
│  6. If closed_frames ≥ 20 → DROWSY alert                       │
│  7. Return JSON: prediction, confidence, score, is_drowsy       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI Driver Drowsiness Detection System/
├── backend/
│   ├── app.py              # FastAPI server + endpoints
│   ├── model_loader.py     # CNN model loading (singleton)
│   └── utils.py            # Image processing utilities
├── frontend/
│   ├── index.html          # Dashboard UI
│   ├── script.js           # Webcam + API + alerts logic
│   └── styles.css          # Premium dark theme
├── model/
│   └── drowsiness_model_v3.h5 # Pre-trained CNN model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- A webcam
- A modern web browser (Chrome, Firefox, Edge)

### Step 1: Clone and Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Priyanshu-Priya/Driver-Drowsiness-Detection-System.git
cd "Driver-Drowsiness-Detection-System"

# Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Server

```bash
cd backend
python app.py
```

*(Note: If you see a TensorFlow warning regarding `oneDNN custom operations`, it is completely harmless. You can safely ignore it or set the environment variable `TF_ENABLE_ONEDNN_OPTS=0` to hide it.)*

Or using uvicorn directly:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Open the App

Open your browser and navigate to:

```
http://localhost:8000
```

### Step 5: Start Detection

1. Click **"Start Detection"**
2. Allow webcam access when prompted
3. Position your face in front of the camera
4. The system will monitor your eye state in real-time

---

## 🧪 How to Test

### Quick Test
1. Start the server and open the web interface
2. Click "Start Detection" and look at the camera with open eyes → Status should show **"Eyes Open"** (green)
3. Close your eyes for ~4 seconds → Status should change to **"Eyes Closed"** (yellow), then **"DROWSY!"** (red) with audio alert
4. Open your eyes → Alert should clear automatically

### API Test (curl)
```bash
# Health check
curl http://localhost:8000/status

# Reset tracker
curl http://localhost:8000/reset

# View event logs
curl http://localhost:8000/logs
```

### API Documentation
FastAPI provides automatic interactive docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🧠 How It Works

### 1. Frame Capture
The browser captures webcam frames every 200ms using a `<canvas>` element and encodes them as Base64 JPEG.

### 2. Face & Eye Detection
OpenCV's Haar Cascade classifiers detect the face first, then locate the eyes within the face region. If the eye cascade fails, the system falls back to cropping the upper face region.

### 3. CNN Prediction
The cropped eye region is:
- Resized to 64×64 pixels
- Kept as RGB (3 channels) — **not converted to grayscale**
- Normalized (pixel values / 255.0)
- Fed to the CNN model

### 4. Output Interpretation
- Model output **≥ 0.5** → Open Eye
- Model output **< 0.5** → Closed Eye

### 5. Drowsiness Logic
A server-side counter tracks consecutive closed-eye frames:
- Counter increments when eyes are closed
- Counter resets to 0 when eyes open
- When counter reaches **20 frames** → drowsiness alert triggered

### 6. Alert System
- **Visual**: Red flashing overlay on the video feed + status change
- **Audio**: Pulsing beep via Web Audio API
- **Event Log**: All drowsiness events are logged with timestamps

---

## 📊 Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Capture interval | 200ms | Frame capture rate (~5 FPS) |
| Input size | 64×64×3 | CNN model input dimensions |
| Threshold | 20 frames | Closed frames before alert |
| JPEG quality | 0.6 | Compression for faster transfer |

---

## 🚀 Improvement Ideas

1. **MediaPipe Face Mesh** — More accurate eye detection with facial landmarks
2. **Yawning Detection** — Detect mouth opening as additional drowsiness signal
3. **Head Pose Estimation** — Detect head nodding/tilting
4. **Dashboard Analytics** — Charts showing drowsiness patterns over time
5. **Mobile App** — React Native or Flutter wrapper
6. **Edge Deployment** — TensorFlow Lite for on-device inference
7. **Multiple Driver Support** — Face recognition + per-driver profiles
8. **Cloud Logging** — Store events in a database for fleet management

---

## ⚠️ Important Notes

- The model expects **RGB (3-channel)** input, not grayscale
- Images are loaded as RGB by `flow_from_directory()` during training
- Do **NOT** convert frames to grayscale (`cv2.COLOR_BGR2GRAY`) before prediction
- The system runs **100% locally** — no internet required after setup
- Webcam permission must be granted in the browser

---

## 📄 License

This project is for educational purposes — university AI presentation and demonstration.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- OpenCV for computer vision utilities
- FastAPI for the high-performance backend
