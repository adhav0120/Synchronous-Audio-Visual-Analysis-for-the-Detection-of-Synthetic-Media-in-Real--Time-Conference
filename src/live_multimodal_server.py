import os
import uuid
import numpy as np
import soundfile as sf
import cv2
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from collections import deque

# Import both of your existing, functional prediction functions
from predict import predict_audio_deepfake
from predict_video import predict_video_deepfake

# --- Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(script_dir, 'static')

app = Flask(__name__, template_folder='templates', static_folder=static_folder_path)
app.config['SECRET_KEY'] = 'your-secret-key-for-multimodal!'
socketio = SocketIO(app)

# --- Buffers for Score Smoothing ---
# Use separate buffers to smooth the scores for a more stable output
AUDIO_BUFFER_SIZE = 5
VIDEO_BUFFER_SIZE = 5
audio_score_buffer = deque(maxlen=AUDIO_BUFFER_SIZE)
video_score_buffer = deque(maxlen=VIDEO_BUFFER_SIZE)

# --- Temporary File Storage ---
TEMP_FOLDER = os.path.join(script_dir, 'temp_live_files')
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
    
SAMPLE_RATE = 16000

# --- Routes ---
@app.route('/live_multimodal')
def live_multimodal():
    """Serves the HTML page for the live multimodal detector."""
    # Clear buffers for a new session
    audio_score_buffer.clear()
    video_score_buffer.clear()
    return render_template('live_multimodal.html')

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print('Client connected for multimodal analysis')
    audio_score_buffer.clear()
    video_score_buffer.clear()

@socketio.on('audio_chunk')
def handle_audio_chunk(chunk):
    """Receives and processes a live audio chunk."""
    try:
        audio_data = np.array(chunk, dtype=np.float32)
        temp_filepath = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.wav")
        sf.write(temp_filepath, audio_data, SAMPLE_RATE)

        raw_score = predict_audio_deepfake(temp_filepath)
        audio_score_buffer.append(raw_score)
        smoothed_score = np.mean(audio_score_buffer) if audio_score_buffer else 0

        # Emit only the audio score
        emit('prediction_result', {'audio_score': smoothed_score * 100})
        os.remove(temp_filepath)

    except Exception as e:
        print(f"Audio processing error: {e}")

@socketio.on('video_frame')
def handle_video_frame(data_url):
    """Receives and processes a live video frame."""
    try:
        # Decode the base64 image data from the browser
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        
        # Convert PIL Image to an OpenCV format (NumPy array)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        temp_filepath = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_filepath, frame)

        # We need a dummy video file for the predict_video function, 
        # so we'll just pass the path to the single image.
        # NOTE: Your predict_video.py should be robust enough to handle a single-frame "video".
        raw_score = predict_video_deepfake(temp_filepath)
        video_score_buffer.append(raw_score)
        smoothed_score = np.mean(video_score_buffer) if video_score_buffer else 0

        # Emit only the video score
        emit('prediction_result', {'video_score': smoothed_score * 100})
        os.remove(temp_filepath)

    except Exception as e:
        print(f"Video processing error: {e}")

if __name__ == '__main__':
    print("Starting LIVE MULTIMODAL server on http://127.0.0.1:5000/live_multimodal")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
