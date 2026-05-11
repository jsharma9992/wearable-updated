# Mac Installation & Setup Guide: Wearable AI Reading Cap

Welcome! This guide provides step-by-step instructions for setting up the Wearable AI Reading Cap project on a macOS machine (MacBook, iMac, Mac Mini, etc.). 

Since this project relies on specific hardware (camera, microphone) and heavy computer vision/AI libraries, there are a few system-level dependencies required before installing the Python packages.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your Mac:

1. **Homebrew**: The macOS package manager. If you don't have it, open your Terminal and paste this command:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Python 3.9, 3.10, or 3.11**: We recommend **Python 3.10** for maximum compatibility with AI libraries like Torch and PaddleOCR.
   ```bash
   brew install python@3.10
   ```

---

## 🚀 Step-by-Step Installation

Open your **Terminal** application and follow these steps sequentially.

### Step 1: Install System Dependencies
The voice control feature uses `SpeechRecognition`, which relies on `PyAudio`. On a Mac, `PyAudio` requires the `portaudio` library to function properly.

```bash
brew install portaudio
```

### Step 2: Clone the Repository
Download the code from GitHub to your local machine. You can choose any folder to store it.

```bash
# Clone the repository
git clone https://github.com/jsharma9992/wearable-updated.git

# Navigate into the project directory
cd wearable-updated
```

### Step 3: Create a Virtual Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts with other Python projects on your Mac.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
*(Note: You will know it's activated when you see `(venv)` at the beginning of your terminal prompt. You must activate it every time you open a new terminal to run the project.)*

### Step 4: Install Python Dependencies
With your virtual environment activated, install all required packages listed in `requirements.txt`. The requirements file already includes macOS-specific packages for Text-to-Speech (`pyobjc-core` and `pyobjc-framework-Cocoa`).

```bash
# Upgrade pip to the latest version first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

*(Note: This step might take a few minutes as it downloads large machine learning libraries like PyTorch and PaddlePaddle).*

### Step 5: Configure Application Settings
Before running the application, you may need to adjust the configuration to match your Mac's setup. 

Open the `config.py` file in a text editor (like VS Code or TextEdit).
```bash
nano config.py
```

Pay attention to:
* `CAMERA_INDEX`: Usually `0` for the built-in MacBook webcam. If you are using an external USB camera, you might need to change this to `1` or `2`.
* `OCR_PRIMARY`: Set to `"paddle"` (default) or `"easyocr"`.

### Step 6: Run the Application!
Once everything is installed, you can start the application.

```bash
python main.py
```

---

## ⚠️ Troubleshooting on macOS

If you run into issues on your Mac, check these common fixes:

### 1. PyAudio Installation Failure
If `pip install -r requirements.txt` fails while trying to build `PyAudio`, you may need to explicitly tell pip where Homebrew installed `portaudio`:

```bash
# If using an Intel Mac:
ip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudiop

# If using an Apple Silicon (M1/M2/M3) Mac:
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```
After successfully installing PyAudio, re-run `pip install -r requirements.txt`.

### 2. Camera or Microphone Permissions
The first time you run `python main.py`, macOS will likely prompt you for permission to access the **Camera** and the **Microphone**.
* You **must** click **Allow**.
* If you accidentally clicked Deny, go to **System Settings > Privacy & Security > Camera** (and **Microphone**) and toggle the switch on for your Terminal application.
