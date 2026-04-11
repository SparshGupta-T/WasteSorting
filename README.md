# RecycleNet

RecycleNet is a real-time computer vision application that leverages YOLO model weights to identify and categorize different types of recyclable plastics based on live video feeds. It uses a modern web-based dashboard and supports dynamic inference models, running natively on Apple Silicon GPUs for fast processing.

## Current Supported Inference Models
- `yolo26_6k.pt` (Default)
- `yolo26m_6H.pt` (Alternative)
- *(Place any `.pt` YOLO weights you wish to use into the project directory and update the `index.html` dropdown)*

## Technology Stack
- **Backend**: Python (Flask)
- **Computer Vision**: Ultralytics YOLO & OpenCV
- **Frontend**: HTML5, Material You 3 Expressive CSS Patterns (Outfit Font)
- **Hardware Acceleration**: Apple Metal Performance Shaders (MPS) via PyTorch

---

## Project Structure
```
WasteSorting/
├── app.py                     # Main Flask Application & YOLO Engine
├── data.yaml                  # Class mappings configuration
├── yolo26_6k.pt               # Default Neural Network Weights
├── yolo26m_6H.pt              # Secondary Neural Network Weights
└── templates/
    └── index.html             # Material 3 Expressive Frontend UI
```

---

## Requirements

Before you can run the project, ensure you have a Python environment set up with all necessary libraries installed, particularly those required for PyTorch and Ultralytics to utilize macOS Apple Silicon:

```bash
pip install flask opencv-python pyyaml ultralytics
pip install torch torchvision torchaudio
```

---

## How to Run the Application

1. Open your terminal and navigate to the project directory:
   ```bash
   cd ~/Desktop/WasteSorting
   ```

2. Start the Flask server:
   ```bash
   python3 app.py
   ```

3. Open your web browser and navigate to the local server address:
   **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

4. **Web Interface Usage**:
   - Use the **INFERENCE MODEL** dropdown to seamlessly switch between `yolo26_6k.pt` and `yolo26m_6H.pt` on the fly.
   - Click the **Start Inference** button to activate the camera. The video feed will appear and objects detected in real-time will be bounded with boxes and class labels (e.g., HDPE, LDPE, PET).
   - Click **Stop Inference** to end the video stream and release the webcam.
   - Click **Shut Down Server** to stop the backend Flask process directly from the UI. *Note: You will need to rerun `python3 app.py` in your terminal to use the app again.*

---

## Technical Details & Issues Solved

During the development and upgrading phase of this project, a few critical issues were identified and successfully resolved to create a stable, performant application:

### 1. Hardware Inference Acceleration (Apple Silicon/MPS)
**Issue:** The PyTorch/YOLO inference engine was defaulting heavily to utilizing the `cpu`, severely bottle-necking performance on an M2 Apple Silicon chip.
**Cause:** The application was using the Ultralytics logic check `hasattr(model.model, 'pt')` which evaluated to `False` for the specific downloaded `.pt` weight files. When this failed, it fell back to executing against the CPU instead of polling for available GPUs.
**Fix:** The application was refactored to check whether explicit strings (`yolo26_6k.pt`) ended with `.pt`. By doing so, we bypass the failed YOLO introspection and explicitly bind the device correctly when `torch.backends.mps.is_available()` evaluates to True.
**Result:** Inference now correctly logs and uses the `mps` compute device.

### 2. Application Crashing on Shutdown (Segmentation Fault)
**Issue:** When clicking "Stop Inference", `app.py` would cause the entire Python runtime to yield a "Segmentation Fault" and crash the script entirely.
**Cause:** The OpenCV `cv2.VideoCapture()` engine utilizes complex C-level bindings for window/camera management. On app exit, simply setting the hardware frame capture (`cap`) variable to `None` wasn't explicitly telling the operating system to safely release those threading processes, causing an OS memory access error.
**Fix:** Explicit OS cleanup instructions were deployed dynamically into the `/stop` endpoint. 
```python
if state.cap is not None:
    state.cap.release()
    state.cap = None
cv2.destroyAllWindows()
cv2.waitKey(1)
```
**Result:** The application is completely thread-safe and can cleanly start and stop inference logic. 

### 3. User Interface Overhaul (Material 3 Expressive)
**Issue:** The previous interface felt basic and did not support dynamic model hot-swapping.
**Cause:** The system originally lacked any user-facing configuration options over the core application state loop.
**Fix:** A fully modern frontend integration using Google's Material 3 Expressive design paradigm was built without requiring any complex frontend UI frameworks (React/Vue). This implements fluid gradient backgrounds, responsive dropdown components using API AJAX calls (`/status`, `/set_model`), pill-shaped tactile buttons, and custom video placeholder graphics. This creates a state-of-the-art presentation utilizing the native browser DOM.
