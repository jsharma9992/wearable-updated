# Wearable AI Reading Cap - Architectural & Pipeline Overview

This document provides a deep dive into the project structure, the execution pipeline, and the modular architecture of the Wearable AI Reading Cap.

## 1. Project Directory Tree

```text
wearable-updated/
├── camera/                  # Camera ingestion & hardware interface
│   ├── camera_manager.py      # Abstracted OpenCV camera handling
│   └── multi_shot_capture.py  # Logic for burst-mode capture
├── intelligence/            # High-level logic & OCR processing
│   ├── currency_detector.py   # Currency identification logic
│   ├── intent_resolver.py     # Main decision engine for system behavior
│   ├── ocr_engine.py          # EasyOCR wrapper with image preprocessing
│   ├── ocr_fusion.py          # Temporal fusion of OCR results across frames
│   └── text_cleaner.py        # Regex-based post-processing for TTS
├── interaction/             # User feedback & control systems
│   ├── guidance.py            # Spatial positioning audio cues
│   ├── state_machine.py       # Global system state management
│   ├── tts_manager.py         # Threaded offline text-to-speech
│   └── voice_controller.py    # Speech-to-command recognition
├── perception/              # Computer Vision & Scene Understanding
│   ├── document_detector.py   # Page boundary & perspective detection
│   ├── finger_tracker.py      # MediaPipe-based fingertip detection
│   ├── stability.py           # Optical flow motion analysis
│   └── text_detector.py       # Region-of-interest (ROI) detection
├── utils/                   # System-wide utilities
│   └── logger.py              # Centralized logging configuration
├── config.py                # Global parameters & tunable thresholds
├── main.py                  # System entry point & main event loop
└── requirements.txt         # Project dependencies
```

---

## 2. System Pipeline

The system operates as a continuous loop in `main.py`, branching into two distinct operational modes: **Trigger Mode** and **Continuous Mode**.

### A. The Core Processing Loop
1.  **Ingestion**: Capture a frame from the `CameraManager`.
2.  **Voice Control**: Poll the `VoiceController` for incoming commands ("capture", "stop", "repeat").
3.  **Perception**:
    -   Calculate frame stability using `StabilityDetector` (Optical Flow).
    -   (Continuous Mode) Detect text bounding boxes via `TextDetector`.
4.  **Intelligence**:
    -   `IntentResolver` evaluates current state, stability, and detected objects.
    -   If stability is high, `OCREngine` extracts text from the regions of interest.
5.  **Interaction**:
    -   `TextCleaner` sanitizes the output.
    -   `TTSManager` queues the text for spoken output.

### B. Trigger Mode (Default)
Optimized for high-accuracy reading on demand.
-   **Wait**: The system idles until a voice command ("Capture") or keyboard shortcut ('c') is received.
-   **Burst Capture**: The `MultiShotCapture` module takes a 3-frame burst.
-   **Fusion & Selection**: OCR is run on all frames; the system selects the most complete and confident result to minimize "misreads" caused by temporary blur.

### C. Continuous Mode
Optimized for hands-free scanning and guidance.
-   **Auto-Scan**: Continuously looks for text.
-   **Guidance**: If text is present but not centered or stable, the `GuidanceEngine` provides audio cues (e.g., "Move left").
-   **Auto-Read**: Once the frame is stable and text is centered, the system automatically reads the content without user intervention.
-   **Deduplication**: The `StateMachine` prevents the system from re-reading the same page repeatedly.

---

## 3. Detailed Architecture

The system follows a three-layer modular architecture to ensure separation of concerns and hardware portability.

### I. Perception Layer (Vision)
*Goal: Convert raw pixels into structured spatial data.*
-   **Stability Detection**: Uses Lucas-Kanade optical flow to determine if the user is holding the device steady enough for a clean OCR read.
-   **Text Detection**: Uses a lightweight detection model to find bounding boxes where text *might* be, allowing the system to ignore background noise.
-   **Document Detection**: Identifies the four corners of a page and applies a perspective warp (Warped Image) to straighten the text for the OCR engine.

### II. Intelligence Layer (Cognition)
*Goal: Process structured vision data into semantic meaning.*
-   **Intent Resolver**: The "brain" of the system. It decides if the system should be guiding the user, reading text, or staying idle based on confidence scores and motion data.
-   **OCR Engine**: Integrates `EasyOCR`. It applies grayscale conversion, contrast enhancement, and noise reduction before running inference.
-   **OCR Fusion**: Combines text results from multiple timestamps to fill in gaps caused by glare or shadows.
-   **Currency Detector**: A specialized module that identifies banknotes and totalizes them, providing an essential utility for visually impaired users.

### III. Interaction Layer (UI/UX)
*Goal: Communicate results back to the user via non-visual channels.*
-   **Voice Controller**: Uses `SpeechRecognition` to allow hands-free operation.
-   **Guidance Engine**: Analyzes the position of text boxes relative to the frame center and generates directional cues.
-   **TTS Manager**: A robust wrapper around `pyttsx3` that handles speech threading, ensuring that long texts don't block the vision pipeline.
-   **State Machine**: Manages transitions (IDLE → GUIDANCE → READING) and handles cooldown timers to ensure a smooth user experience.

---

## 4. Hardware Deployment Strategy
-   **Development**: Runs on Windows/macOS/Linux with webcam support.
-   **Production**: Designed for Raspberry Pi 4/5. 
-   **Performance**: Utilizes frame-skipping (Perception only runs every 5th frame) to maintain a responsive 30FPS preview while running heavy OCR in the background.
