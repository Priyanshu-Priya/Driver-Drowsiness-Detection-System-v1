/**
 * script.js — Frontend Logic for AI Driver Drowsiness Detection System
 *
 * Handles:
 * - Webcam access via WebRTC
 * - Frame capture and Base64 encoding
 * - API communication with FastAPI backend
 * - UI updates (status, score, alerts)
 * - Audio alert via Web Audio API
 */

// ─── Configuration ──────────────────────────────────────────────────────────
const API_URL = window.location.origin;
const CAPTURE_INTERVAL_MS = 200;  // Capture a frame every 200ms (~5 FPS)
const JPEG_QUALITY = 0.6;         // JPEG compression quality (0-1)

// ─── State ──────────────────────────────────────────────────────────────────
let isRunning = false;
let captureTimer = null;
let audioCtx = null;
let isAlertPlaying = false;
let alertOscillator = null;
let alertGain = null;
let frameCount = 0;
let fpsInterval = null;
let lastFpsTime = Date.now();
let fpsFrameCount = 0;

// ─── DOM Elements ───────────────────────────────────────────────────────────
const video = document.getElementById('webcam');
const canvas = document.getElementById('captureCanvas');
const ctx = canvas.getContext('2d');

const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnReset = document.getElementById('btnReset');

const statusIndicator = document.getElementById('statusIndicator');
const statusEmoji = document.getElementById('statusEmoji');
const statusLabel = document.getElementById('statusLabel');
const statusSublabel = document.getElementById('statusSublabel');

const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');

const metricClosedFrames = document.getElementById('metricClosedFrames');
const metricThreshold = document.getElementById('metricThreshold');

const scoreRingFill = document.getElementById('scoreRingFill');
const scoreNumber = document.getElementById('scoreNumber');

const alertOverlay = document.getElementById('alertOverlay');
const videoBadge = document.getElementById('videoBadge');
const fpsBadge = document.getElementById('fpsBadge');
const fpsText = document.getElementById('fpsText');
const eventLog = document.getElementById('eventLog');

// ─── Webcam Access ──────────────────────────────────────────────────────────

/**
 * Initialize the webcam stream using WebRTC.
 */
async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        video.srcObject = stream;
        await video.play();

        // Set canvas to match video dimensions
        canvas.width = 640;
        canvas.height = 480;

        return true;
    } catch (err) {
        console.error('Webcam access failed:', err);
        setStatus('error', '❌', 'Camera Error',
            'Unable to access webcam. Please grant permission.');
        return false;
    }
}

// ─── Detection Control ──────────────────────────────────────────────────────

/**
 * Start the drowsiness detection loop.
 */
async function startDetection() {
    if (isRunning) return;

    // Initialize webcam if not already done
    if (!video.srcObject) {
        const success = await initWebcam();
        if (!success) return;
    }

    isRunning = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    videoBadge.style.display = 'flex';
    fpsBadge.style.display = 'block';

    setStatus('open', '👁️', 'Monitoring...', 'Analyzing eye state in real-time');

    // Start frame capture loop
    captureTimer = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);

    // Start FPS counter
    lastFpsTime = Date.now();
    fpsFrameCount = 0;
    fpsInterval = setInterval(updateFPS, 1000);
}

/**
 * Stop the drowsiness detection loop.
 */
function stopDetection() {
    isRunning = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    videoBadge.style.display = 'none';
    fpsBadge.style.display = 'none';

    if (captureTimer) {
        clearInterval(captureTimer);
        captureTimer = null;
    }
    if (fpsInterval) {
        clearInterval(fpsInterval);
        fpsInterval = null;
    }

    stopAlertSound();
    alertOverlay.classList.remove('active');
    setStatus('idle', '😐', 'Stopped', 'Detection paused');
}

/**
 * Reset the drowsiness tracker on the server.
 */
async function resetTracker() {
    try {
        await fetch(`${API_URL}/reset`);
        stopAlertSound();
        alertOverlay.classList.remove('active');
        updateScore(0);
        metricClosedFrames.textContent = '0';
        setStatus('idle', '😐', 'Reset', 'Tracker has been reset');
        fetchLogs();
    } catch (err) {
        console.error('Reset failed:', err);
    }
}

// ─── Frame Capture & Prediction ─────────────────────────────────────────────

/**
 * Capture a frame from the webcam and send it to the backend for prediction.
 */
async function captureAndPredict() {
    if (!isRunning || video.readyState < 2) return;

    try {
        // Draw current video frame onto the hidden canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to Base64 JPEG
        const base64Image = canvas.toDataURL('image/jpeg', JPEG_QUALITY);

        // Send to backend
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        fpsFrameCount++;

        // Update UI with prediction results
        handlePrediction(data);

    } catch (err) {
        console.error('Prediction error:', err);
    }
}

// ─── UI Update Logic ────────────────────────────────────────────────────────

/**
 * Handle a prediction response from the backend.
 */
function handlePrediction(data) {
    const {
        prediction,
        confidence,
        closed_frames,
        is_drowsy,
        drowsiness_score,
        threshold,
        face_detected
    } = data;

    // Update threshold display
    metricThreshold.textContent = threshold;
    metricClosedFrames.textContent = closed_frames;

    // Update confidence bar
    const confPercent = Math.round(confidence * 100);
    confidenceValue.textContent = `${confPercent}%`;
    confidenceFill.style.width = `${confPercent}%`;

    // Update drowsiness score ring
    updateScore(drowsiness_score);

    if (!face_detected) {
        setStatus('idle', '🔍', 'No Face', 'Position your face in front of the camera');
        stopAlertSound();
        alertOverlay.classList.remove('active');
        return;
    }

    if (is_drowsy) {
        // DROWSY — trigger alert
        setStatus('drowsy', '🚨', 'DROWSY!', `Eyes closed for ${closed_frames} frames`);
        alertOverlay.classList.add('active');
        startAlertSound();
        fetchLogs();
    } else if (prediction === 'Closed') {
        // Eyes closed but not yet drowsy
        setStatus('closed', '😑', 'Eyes Closed',
            `${closed_frames}/${threshold} frames to alert`);
        alertOverlay.classList.remove('active');
        stopAlertSound();
    } else {
        // Eyes open
        setStatus('open', '👁️', 'Eyes Open', 'Driver is alert');
        alertOverlay.classList.remove('active');
        stopAlertSound();
    }
}

/**
 * Update the status indicator in the sidebar.
 */
function setStatus(state, emoji, label, sublabel) {
    statusIndicator.className = `status-indicator state-${state}`;
    statusEmoji.textContent = emoji;
    statusLabel.textContent = label;
    statusSublabel.textContent = sublabel;

    // Update label color based on state
    const colors = {
        idle: 'var(--text-secondary)',
        open: 'var(--accent-green)',
        closed: 'var(--accent-yellow)',
        drowsy: 'var(--accent-red)',
        error: 'var(--accent-red)'
    };
    statusLabel.style.color = colors[state] || 'var(--text-primary)';
}

/**
 * Update the drowsiness score ring.
 */
function updateScore(score) {
    scoreNumber.textContent = score;

    // Update ring fill (circumference of r=56 is ~351.86)
    const circumference = 2 * Math.PI * 56;
    const offset = circumference - (score / 100) * circumference;
    scoreRingFill.style.strokeDasharray = circumference;
    scoreRingFill.style.strokeDashoffset = offset;

    // Color based on score
    let color;
    if (score < 30) {
        color = 'var(--accent-green)';
        scoreNumber.style.color = 'var(--accent-green)';
    } else if (score < 70) {
        color = 'var(--accent-yellow)';
        scoreNumber.style.color = 'var(--accent-yellow)';
    } else {
        color = 'var(--accent-red)';
        scoreNumber.style.color = 'var(--accent-red)';
    }
    scoreRingFill.style.stroke = color;
}

/**
 * Update the FPS display.
 */
function updateFPS() {
    const now = Date.now();
    const elapsed = (now - lastFpsTime) / 1000;
    const fps = Math.round(fpsFrameCount / elapsed);
    
    fpsBadge.textContent = `${fps} FPS`;
    fpsText.textContent = `${fps} FPS`;
    
    fpsFrameCount = 0;
    lastFpsTime = now;
}

// ─── Audio Alert (Web Audio API) ────────────────────────────────────────────

/**
 * Start playing the alert sound — a pulsing beep using Web Audio API.
 * No external audio files needed.
 */
function startAlertSound() {
    if (isAlertPlaying) return;

    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }

        isAlertPlaying = true;
        playBeepPulse();
    } catch (err) {
        console.warn('Audio alert failed:', err);
    }
}

/**
 * Play a repeating beep pattern.
 */
function playBeepPulse() {
    if (!isAlertPlaying || !audioCtx) return;

    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.type = 'square';
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);

    gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

    oscillator.start(audioCtx.currentTime);
    oscillator.stop(audioCtx.currentTime + 0.3);

    // Repeat after a pause
    setTimeout(() => {
        if (isAlertPlaying) playBeepPulse();
    }, 600);
}

/**
 * Stop the alert sound.
 */
function stopAlertSound() {
    isAlertPlaying = false;
}

// ─── Event Log ──────────────────────────────────────────────────────────────

/**
 * Fetch and display drowsiness event logs from the server.
 */
async function fetchLogs() {
    try {
        const response = await fetch(`${API_URL}/logs`);
        const data = await response.json();
        renderLogs(data.logs);
    } catch (err) {
        console.error('Failed to fetch logs:', err);
    }
}

/**
 * Render the event log in the UI.
 */
function renderLogs(logs) {
    if (!logs || logs.length === 0) {
        eventLog.innerHTML = '<div class="event-log__empty">No events yet.</div>';
        return;
    }

    // Show newest first
    const reversed = [...logs].reverse();
    eventLog.innerHTML = reversed.map(log => {
        const isAlert = log.event === 'DROWSINESS_DETECTED';
        const dotClass = isAlert ? 'event-log__dot--alert' : 'event-log__dot--clear';
        const text = isAlert
            ? `Drowsiness detected (${log.closed_frames} frames)`
            : `Alert cleared (${log.duration_seconds}s)`;

        return `
            <div class="event-log__item">
                <span class="event-log__dot ${dotClass}"></span>
                <span class="event-log__text">${text}</span>
                <span class="event-log__time">${log.timestamp}</span>
            </div>
        `;
    }).join('');

    // Auto-scroll to top (newest)
    eventLog.scrollTop = 0;
}

// ─── Initialization ─────────────────────────────────────────────────────────

// Pre-initialize score ring
updateScore(0);

// Try to send a browser notification permission request
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}
