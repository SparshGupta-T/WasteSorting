import cv2
import yaml
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# Initialize the Flask web server instance
app = Flask(__name__)

# ==========================================
# Application State Management
# ==========================================
# This class acts as a central storage for our application's state.
# It holds the camera capture object, the loaded YOLO model, and connection status.
class InferenceState:
    def __init__(self):
        self.is_running = False # Tracks if the camera inference is currently active
        self.cap = None         # Holds the OpenCV video capture object
        self.model = None       # Holds the loaded YOLO model instance
        self.classes = []       # List of recognizable class names loaded from data.yaml
        self.current_model_name = "yolo26_6k.pt" # Default starting model

# Instantiate the global state object to be used across route handlers
state = InferenceState()

# ==========================================
# Initialization & Setup Helper Functions
# ==========================================

def load_classes():
    """
    Reads the object classes from the data.yaml file.
    This tells the application what objects the YOLO model has been trained to detect.
    """
    try:
        with open("data.yaml", 'r') as f:
            data = yaml.safe_load(f)
            state.classes = data.get('names', [])
            print(f"Loaded classes: {state.classes}")
    except Exception as e:
        print(f"Error loading data.yaml: {e}")

def load_model(model_name):
    """
    Loads a YOLO model by its filename.
    Updates the global state to reflect the currently active model.
    """
    try:
        print(f"Loading YOLO model {model_name}...")
        state.model = YOLO(model_name)
        state.current_model_name = model_name
        print(f"Model {model_name} loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False

def initialize_system():
    load_classes()
    load_model("yolo26_6k.pt")

initialize_system()

# ==========================================
# Core Inference & Video Streaming Engine
# ==========================================

def generate_frames():
    """
    Generator function that continuously captures frames from the webcam,
    runs YOLO object detection inference, and yields the annotated video frames
    as a byte stream formatted for MJPEG (Motion JPEG).
    """
    while state.is_running and state.cap is not None:
        # Read the next frame from the webcam
        success, frame = state.cap.read()
        if not success:
            break
            
        # Determine best available hardware device (Apple Silicon, NVIDIA GPU, or CPU fallback)
        device = "cpu"
        if state.current_model_name.endswith('.pt'):
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps" # Apple Metal Performance Shaders
            elif torch.cuda.is_available():
                device = "cuda" # NVIDIA CUDA
            
        print(f"Using compute device: {device}")

        # Run YOLO inference using the automatically selected compute device
        if state.model is not None:
            results = state.model(frame, device=device, verbose=False)
            # Plot the bounding boxes and labels onto the original frame
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        # Encode the annotated frame into a compressed JPEG representation
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the encoded image bytes in the multipart HTTP response format required for MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# Flask Web Application Routes (API & UI)
# ==========================================

@app.route('/')
def index():
    """
    Serves the main HTML user interface page (index.html).
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    The video streaming route. The HTML page sets an <img> source to this URL.
    Returns a continuous multipart stream of JPEG frames if inference is running.
    """
    if not state.is_running:
        return "", 204
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_inference():
    """
    API endpoint to start the webcam source and begin YOLO inference processing.
    """
    if not state.is_running:
        state.cap = cv2.VideoCapture(0) # 0 connects to the default system camera
        if not state.cap.isOpened():
            return jsonify({"status": "error", "message": "Camera not accessible"}), 500
        state.is_running = True
        return jsonify({"status": "success", "message": "Inference started"})
    return jsonify({"status": "already_running", "message": "Inference is already running"})

@app.route('/stop', methods=['POST'])
def stop_inference():
    """
    API endpoint to stop the webcam feed and pause inference processing safely.
    Releases hardware resources.
    """
    if state.is_running:
        state.is_running = False
        if state.cap is not None:
            state.cap.release()
            state.cap = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return jsonify({"status": "success", "message": "Inference stopped"})
    return jsonify({"status": "already_stopped", "message": "Inference is not running"})

@app.route('/status', methods=['GET'])
def get_status():
    """
    API endpoint for the frontend UI to query the current server state
    (whether it's actively running inference and what model is selected).
    """
    return jsonify({
        "is_running": state.is_running,
        "current_model": state.current_model_name
    })

@app.route('/set_model', methods=['POST'])
def set_model():
    """
    API endpoint to switch the active YOLO model file based on frontend dropdown selection.
    """
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"status": "error", "message": "No model specified"}), 400
    
    model_name = data['model']
    if load_model(model_name):
        return jsonify({"status": "success", "message": f"Successfully switched to {model_name}"})
    else:
        return jsonify({"status": "error", "message": f"Failed to load {model_name}"}), 500


if __name__ == "__main__":
    print("Starting Flask YOLO Server on http://127.0.0.1:5000")
    # Run the Flask app on localhost
    app.run(host='127.0.0.1', port=5000, debug=False)
