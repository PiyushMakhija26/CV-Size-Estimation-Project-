# 👕 Vision-Fit: AI-Powered Body Measurement System

**Vision-Fit** is an intelligent body measurement application that uses **MediaPipe** pose detection to analyze shoulder width and provide accurate T-shirt size recommendations. Designed for e-commerce, fashion retailers, and personal sizing assistance.

---

## 🎯 Core Features

### 📐 Accurate Measurements
- **Shoulder Width Detection**: AI-powered pose estimation with +2cm deltoid offset
- **Dual Calibration Methods**:
  - **A4 Paper Calibration**: Place A4 paper at known distance (21cm width reference)
  - **Height Calibration**: Uses nose-to-heel body proportion (automatic if A4 unavailable)
- **Temporal Smoothing**: 20-frame averaging to reduce measurement jitter
- **Distance Validation**: Warns if user is too close to camera (>80% frame height)

### 📊 Intelligent Size Recommendation
- **Realistic Thresholds** (including +2cm offset):
  - **S**: < 40cm
  - **M**: 40–44cm
  - **L**: 44–48cm
  - **XL**: 48–52cm
  - **XXL**: > 52cm
- **BMI-Based Fit Notes**: Adjusts recommendations based on BMI (oversized vs. slim fit)

### 🖼️ Flexible Input
- **Webcam Capture**: Real-time camera capture
- **Image Upload**: Process existing photos (JPG, PNG, BMP, TIFF)
- **Timestamped Storage**: All captured images saved with auto-generated filenames

### 📱 Three User Interfaces
1. **Web App (Streamlit)**: Black/white theme, browser-based, mobile-friendly
2. **Desktop GUI (Tkinter)**: Standalone application with local result storage
3. **CLI**: Command-line utility for headless operation

---

## 📦 Installation

### Prerequisites
- Python 3.13+
- Webcam (optional, for capture mode)

### Setup Steps

1. **Clone/Download** this project to your machine

2. **(Optional) Create Virtual Environment**:
   ```powershell
   # Windows
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify MediaPipe Model**:
   - The app will auto-download `pose_landmarker_lite.task` on first run
   - If manual download required: Place model file in project root directory

---

## 🚀 Quick Start

### Option 1: Web App (Recommended for First-Time Users)
```bash
python -m streamlit run app.py
```
- Opens at: `http://localhost:8501`
- **Choose**: Take photo with camera OR upload existing image
- **Enter**: Your height (cm) and weight (kg)
- **Get**: Size recommendation with detailed metrics

### Option 2: Desktop GUI
```bash
python app_gui.py
```
Or on Windows:
```powershell
.\run_gui.bat
```
- **Features**: Capture/upload buttons, local file browser, status indicators
- **Saves**: All images to `captured_images/` folder, results to `results/` folder

### Option 3: Command-Line Interface
```bash
python vision_fit.py
```
- **Follow prompts**: Select calibration method (A4 or height)
- **Capture**: Press `c` when ready, then follow instructions
- **Export**: Results saved as timestamped text file

---

## 📂 Project Structure

```
CV project/
├── app.py                      # Streamlit web interface
├── app_gui.py                  # Tkinter desktop GUI
├── vision_fit.py               # CLI application
├── vision_fit_processor.py      # Core measurement engine
├── requirements.txt            # Python dependencies
├── pose_landmarker_lite.task    # MediaPipe model (auto-downloaded)
├── run_gui.bat                 # Windows GUI launcher
├── run_gui.sh                  # Linux/macOS GUI launcher
├── README.md                   # This file
├── captured_images/            # Timestamped captured photos (auto-created)
├── results/                    # Measurement results (auto-created)
└── __pycache__/               # Python cache (auto-created)
```

---

## 🔧 Key Components

### `vision_fit_processor.py` - Core Engine
**Self-contained measurement system** with:
- Pose landmark extraction (MediaPipe Tasks API)
- A4 paper detection (OpenCV contour analysis)
- Pixel-to-centimeter calibration
- Shoulder width calculation with offset
- BMI computation
- Size recommendation logic

### `app.py` - Streamlit Web Interface
**Browser-based UI** featuring:
- Radio button: Camera OR Upload selection
- Dual metric inputs (height, weight)
- Live image preview
- Results dashboard with:
  - Raw shoulder width
  - Smoothed shoulder width (temporal averaging)
  - BMI score
  - Recommended size + fit note
  - Annotated result image

### `app_gui.py` - Tkinter Desktop Application
**Standalone GUI** with:
- Two-button input (Capture / Upload)
- User metrics form
- Processing progress bar
- Scrollable results panel
- Auto-created `captured_images/` and `results/` folders

### `vision_fit.py` - CLI Utility
**Terminal-based interface** offering:
- Interactive calibration selection
- Webcam capture workflow
- Result export to timestamped text files
- Validation checks

---

## 📊 Measurement Workflow

```
1. Load Image (camera, upload, or webcam)
           ↓
2. Calibration (Detect A4 paper OR use height)
           ↓
3. Pose Detection (Extract 33 body landmarks)
           ↓
4. Shoulder Width Calculation (Left/right shoulder pixel distance)
           ↓
5. Apply Offset (+2cm for deltoid/muscle width)
           ↓
6. Temporal Smoothing (Average over 20 frames if available)
           ↓
7. Size Recommendation (Match to threshold ranges)
           ↓
8. BMI Adjustment (Suggest fit category: slim, regular, oversized)
           ↓
9. Display Results + Save (Images & text reports)
```

---

## 💡 Tips for Accurate Measurements

### Camera Setup
- **Distance**: 0.5–1.5 meters from camera (aim for 40–120cm)
- **Lighting**: Bright, evenly-lit room; avoid backlighting
- **Clothing**: Fitted or form-fitting upper body garment
- **Position**: Face forward, arms at sides or relaxed

### Calibration Options
- **A4 Paper**: Hold A4 sheet flat at chest/shoulder level (most stable)
- **Height**: Ensure full body visible from head to feet

### Input Quality
- **High Resolution**: Crisp, clear images produce better landmarks
- **Minimal Occlusion**: Avoid partially hidden shoulders or posture tilts

---

## 📈 Output Files & Storage

### Captured Images
**Location**: `captured_images/user_YYYYMMDD_HHMMSS.jpg`
- Automatically timestamped
- Includes detected pose landmarks in debug output

### Results Text
**Location**: `results/results_YYYYMMDD_HHMMSS.txt`
- Contains:
  - Shoulder width (raw & smoothed)
  - BMI
  - Recommended size & fit note
  - Calibration method used

### Debug Image
**Location**: `vision_fit_result.jpg`
- Annotated output with:
  - Detected size label
  - Shoulder measurements
  - BMI value

---

## 🔬 Technical Specifications

| Component | Technology | Details |
|-----------|-----------|---------|
| **Pose Detection** | MediaPipe Tasks API | 33 body landmarks, lightweight model |
| **Image Processing** | OpenCV (cv2) | Contour detection for A4, image conversion |
| **Calculations** | NumPy | Vectorized shoulder/BMI math |
| **Web UI** | Streamlit | Real-time interactivity, black/white theme |
| **Desktop UI** | Tkinter | Cross-platform native GUI |
| **Model Size** | ~29 MB | `pose_landmarker_lite.task` |
| **Python Version** | 3.13+ | Recommended for latest performance |

---

## ⚙️ Configuration

### Adjusting Size Thresholds
Edit `vision_fit_processor.py` (line ~35):
```python
self.SIZE_THRESHOLDS = {
    'S': (0, 40),      # Adjust upper limit as needed
    'M': (40, 44),
    'L': (44, 48),
    'XL': (48, 52),
    'XXL': (52, 300)
}
```

### Modifying Shoulder Offset
Edit `vision_fit_processor.py` (line ~31):
```python
self.SHOULDER_OFFSET = 2.0  # Change from 2.0 cm to desired value
```

### Distance Validation Threshold
Edit `vision_fit_processor.py` (line ~32):
```python
self.DISTANCE_THRESHOLD = 80  # % of frame height (change from 80%)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found** | Run app once; it auto-downloads. Or manually place `pose_landmarker_lite.task` in project root. |
| **Webcam not detected** | Check system permissions; restart app; test with `cv2.VideoCapture(0)`. |
| **Size seems too large** | Check distance (move closer); verify height input; ensure good lighting. |
| **Streamlit won't start** | Run `pip install streamlit` ; check Python 3.13+ installed. |
| **Images not saving** | Verify write permissions in project folder; check `captured_images/` exists. |

---

## 📝 Requirements

See `requirements.txt`:
```
opencv-python
mediapipe
numpy
pillow
streamlit
tkinter  # Usually bundled with Python
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎓 How It Works (Technical Overview)

1. **Calibration**: Converts pixel measurements to real-world centimeters using either A4 paper geometry or person height
2. **Pose Estimation**: MediaPipe detects 33 body landmarks (shoulders at indices 11 & 12)
3. **Shoulder Width**: Calculates pixel distance between left/right shoulder, converts to cm, adds +2cm offset
4. **Smoothing**: Stores measurements in 20-frame FIFO buffer, outputs moving average
5. **Sizing**: Maps smoothed width to size category; BMI further refines fit recommendation

---

## 📞 Support & Feedback

- **Issues**: Check troubleshooting section above
- **Features**: Open to enhancement suggestions
- **Performance**: Runs on CPU; GPU support available via MediaPipe config

---

## 📄 License

This project is provided as-is for personal and educational use.

---

**Version**: 1.0 | **Date**: March 2026 | **Python**: 3.13+
