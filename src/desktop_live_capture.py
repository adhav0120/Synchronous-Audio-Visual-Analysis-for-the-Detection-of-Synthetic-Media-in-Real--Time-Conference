import cv2
import mss
import numpy as np
import time
import threading
import soundcard as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp
import customtkinter as ctk
from PIL import Image
from collections import deque, namedtuple
import re
import math
import os
import json
import logging
from torch.utils import model_zoo

# Set up unified logging pipeline
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shield_defense.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SHIELD")

# Load Config
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[{config_path}] Loading config failed: {e}. Using defaults.")
        return {}

APP_CONFIG = load_config()
AUDIO_CONFIG = APP_CONFIG.get('audio', {})
VIDEO_CONFIG = APP_CONFIG.get('video', {})
UI_CONFIG = APP_CONFIG.get('ui', {})

# Import the actual model classes and prediction functions
from predict import model as AudioModel, mel_spectrogram

# --- Performance & Core Threading Fix ---
# Limit PyTorch to 1-2 cores. When it uses all cores, it starves the 
# Windows soundcard capture loop, creating digital static/data discontinuity!
torch.set_num_threads(2)

# --- GPU / CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"--- Using device: {device} ---")
if not torch.cuda.is_available():
    logger.warning("--- WARNING: No CUDA GPU detected. The application will run on the CPU and may be slow. ---")

# Import the extracted EfficientNet model
from video_model import EfficientNet, get_model_params

# --- In-Memory Prediction Functions ---
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_video_from_frame(frame_rgb: np.ndarray, model: torch.nn.Module) -> float:
    if frame_rgb is None or model is None or frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0: return 0.0
    input_tensor = video_transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        score = probabilities[0][1].item()
    return score

def predict_audio_from_chunk(audio_data: np.ndarray, model: torch.nn.Module) -> float:
    if audio_data is None or model is None: return 0.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0).to(device)
    mel_spec = mel_spectrogram(waveform).to(device)
    with torch.no_grad():
        output = model(mel_spec)
        # Apply a mathematical penalty to the 'Fake' logit to reduce artificial sensitivity
        output[0][1] -= AUDIO_CONFIG.get('fake_logit_penalty', 2.0)
        probabilities = torch.softmax(output, dim=1)
        score = probabilities[0][1].item()
    return score

# --- Shared State for All Threads ---
shared_state = {
    'video_score': 0.0, 'video_ema': -1.0, 'system_audio_score': 0.0, 'mic_audio_score': 0.0,
    'system_audio_active': False, 'mic_audio_active': False, 
    'system_audio_silence': True, 'mic_audio_silence': True,
    'latest_frame': None, 'running': True, 'lock': threading.Lock()
}

def is_silence(audio_data: np.ndarray, threshold: float = None) -> bool:
    if threshold is None: threshold = AUDIO_CONFIG.get('silence_threshold', 0.002)
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < threshold

# --- Audio Capture Threads ---
def system_audio_capture_thread(audio_model: torch.nn.Module) -> None:
    """Continuously captures and analyzes system audio (what you hear)."""
    try:
        mics = sc.all_microphones(include_loopback=True)
        loopback_mic = None
        # Safest way: Find the loopback that exactly matches the default speaker's name
        try:
            default_spk_name = sc.default_speaker().name
            loopback_mic = next((m for m in mics if m.name == default_spk_name and getattr(m, 'isloopback', False)), None)
        except Exception:
            pass
        if loopback_mic is None:
            # Fallback 1: Any mic marked as loopback
            loopback_mic = next((m for m in mics if getattr(m, 'isloopback', False)), None)
        if loopback_mic is None:
            # Fallback 2: Name contains legacy terms
            loopback_mic = next((m for m in mics if 'loopback' in m.name.lower() or 'stereo mix' in m.name.lower() or 'speaker' in m.name.lower()), None)
            
        if loopback_mic is None:
            logger.error("[System Audio] No loopback audio device found.")
            return
        logger.info(f"[System Audio] Success! Monitoring from: {loopback_mic.name}")
        with shared_state['lock']: shared_state['system_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        
        def run_inference(data_chunk):
            score = predict_audio_from_chunk(data_chunk, audio_model)
            # Artificially cap the system audio score to 60% of its raw output
            score = score * AUDIO_CONFIG.get('system_audio_penalty_multiplier', 0.60)
            with shared_state['lock']:
                shared_state['system_audio_score'] = score
                
        with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while True:
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                flat_audio = audio_data.flatten()
                silence = is_silence(flat_audio)
                with shared_state['lock']:
                    shared_state['system_audio_silence'] = silence
                if not silence:
                    # Run asynchronously to prevent audio buffer overruns
                    threading.Thread(target=run_inference, args=(flat_audio.copy(),), daemon=True).start()

    except Exception as e: logger.error(f"[System Audio] Error: {e}")
    finally:
        with shared_state['lock']: shared_state['system_audio_active'] = False

def microphone_capture_thread(audio_model: torch.nn.Module) -> None:
    """Continuously captures and analyzes audio from the default microphone."""
    try:
        mic = sc.default_microphone()
        if mic is None:
            logger.error("[Mic Audio] No default microphone found.")
            return
        logger.info(f"[Mic Audio] Success! Monitoring from: {mic.name}")
        with shared_state['lock']: shared_state['mic_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        
        def run_inference(data_chunk):
            score = predict_audio_from_chunk(data_chunk, audio_model)
            with shared_state['lock']:
                shared_state['mic_audio_score'] = score

        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while True:
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                flat_audio = audio_data.flatten()
                silence = is_silence(flat_audio)
                with shared_state['lock']:
                    shared_state['mic_audio_silence'] = silence
                if not silence:
                    # Run asynchronously to prevent audio buffer overruns
                    threading.Thread(target=run_inference, args=(flat_audio.copy(),), daemon=True).start()

    except Exception as e: logger.error(f"[Mic Audio] Error: {e}")
    finally:
        with shared_state['lock']: shared_state['mic_audio_active'] = False

# --- Main Video Capture and UI Thread ---
def video_capture_thread(video_model: torch.nn.Module) -> None:
    """Background thread for screen capture and video detection."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    last_prediction_time = time.time()
    
    HIGH_CONF_THRESHOLD = VIDEO_CONFIG.get('high_conf_threshold', 0.95)
    WARN_THRESHOLD = VIDEO_CONFIG.get('warn_threshold', 0.75)
    MIN_FACE_SIZE = VIDEO_CONFIG.get('min_face_size', 50)
    prediction_interval = VIDEO_CONFIG.get('prediction_interval', 0.5)

    with mss.mss() as sct:
        while True:
            # Check exit flag
            with shared_state['lock']:
                if not shared_state['running']:
                    break
                mx = shared_state.get('mouse_x', 0)
                my = shared_state.get('mouse_y', 0)

            # Automatically select the monitor where the cursor currently is
            current_monitor = sct.monitors[1]
            if len(sct.monitors) > 1:
                for m in sct.monitors[1:]:
                    if m["left"] <= mx < (m["left"] + m["width"]) and m["top"] <= my < (m["top"] + m["height"]):
                        current_monitor = m
                        break

            half_width = current_monitor["width"] // 2
            
            # Dynamic capture layout: explicitly capture right half
            monitor = {
                "top": current_monitor["top"],
                "left": current_monitor["left"] + half_width,
                "width": half_width,
                "height": current_monitor["height"]
            }

            frame = np.array(sct.grab(monitor))
            # Safely handle extra padding/stride bytes from MSS grab to prevent UI tearing
            frame = np.ascontiguousarray(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=display_frame, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))
                    h, w, c = frame_rgb.shape
                    cx_min, cy_min, cx_max, cy_max = w, h, 0, 0
                    for landmark in face_landmarks.landmark:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        if cx < cx_min: cx_min = cx
                        if cy < cy_min: cy_min = cy
                        if cx > cx_max: cx_max = cx
                        if cy > cy_max: cy_max = cy
                    padding = 20
                    x, y = max(0, cx_min - padding), max(0, cy_min - padding)
                    box_w, box_h = (cx_max - cx_min) + 2*padding, (cy_max - cy_min) + 2*padding
                    if box_w > MIN_FACE_SIZE and box_h > MIN_FACE_SIZE:
                        if time.time() - last_prediction_time > prediction_interval:
                            face_crop = frame_rgb[y:y+box_h, x:x+box_w]
                            score = predict_video_from_frame(face_crop, video_model)
                            with shared_state['lock']:
                                current_ema = shared_state.get('video_ema', -1.0)
                                alpha = VIDEO_CONFIG.get('ema_alpha', 0.3)
                                if current_ema < 0:
                                    new_ema = score
                                else:
                                    new_ema = (score * alpha) + (current_ema * (1 - alpha))
                                shared_state['video_ema'] = new_ema
                                shared_state['video_score'] = new_ema
                            last_prediction_time = time.time()
            else:
                 cv2.putText(display_frame, "Searching for face...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Save frame for GUI
            with shared_state['lock']:
                # The array striding is already cleaned by ascontiguousarray
                shared_state['latest_frame'] = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Small sleep to prevent maxing out CPU
            time.sleep(0.01)
            
    face_mesh.close()

class LiveDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SHIELD - Multimodal Deepfake Defense")
        self.geometry("900x1000")
        ctk.set_appearance_mode("dark")
        self.configure(fg_color="#0F172A") # Rich dark slate background
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main Layout Configuration
        self.grid_rowconfigure(0, weight=0) # Top Header
        self.grid_rowconfigure(1, weight=5) # BIG Video
        self.grid_rowconfigure(2, weight=3) # Dashboard
        self.grid_columnconfigure(0, weight=1)

        # --- Header ---
        self.header_frame = ctk.CTkFrame(self, fg_color="#1E293B", corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="nsew")
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)

        self.header_label = ctk.CTkLabel(
            self.header_frame, 
            text="SHIELD ACTIVE - SECURE COMMUNICATION", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#10B981"
        )
        self.header_label.grid(row=0, column=0, pady=10)

        self.settings_btn = ctk.CTkButton(
            self.header_frame, text="⚙ Settings", command=self.toggle_settings,
            width=30, fg_color="#334155", hover_color="#475569"
        )
        self.settings_btn.grid(row=0, column=1, padx=20, pady=10, sticky="e")

        # --- Top Panel: Video Display ---
        self.video_frame = ctk.CTkFrame(self, fg_color="#1E293B", corner_radius=15)
        self.video_frame.grid(row=1, column=0, padx=30, pady=(20, 10), sticky="nsew")
        
        # Loading State UI
        self.loading_label = ctk.CTkLabel(
            self.video_frame, 
            text="Initializing Shield Framework", 
            text_color="#38BDF8", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.loading_label.pack(expand=True, pady=(50, 5))
        
        self.loading_sub_label = ctk.CTkLabel(
            self.video_frame, 
            text="Loading tensor models into memory. Please wait...", 
            text_color="#94A3B8", 
            font=ctk.CTkFont(size=16)
        )
        self.loading_sub_label.pack(pady=(0, 20))
        
        self.loading_bar = ctk.CTkProgressBar(self.video_frame, width=400, mode="indeterminate", progress_color="#38BDF8")
        self.loading_bar.pack(pady=10)
        self.loading_bar.start()

        # The actual image label (hidden at start)
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        
        # --- Bottom Panel: Dashboard ---
        self.info_frame = ctk.CTkFrame(self, fg_color="#1E293B", corner_radius=15)
        self.info_frame.grid(row=2, column=0, padx=30, pady=(10, 30), sticky="nsew")
        self.info_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Overview Section
        self.verdict_title = ctk.CTkLabel(self.info_frame, text="THREAT ANALYSIS", font=ctk.CTkFont(size=14, weight="bold"), text_color="#94A3B8")
        self.verdict_title.grid(row=0, column=0, columnspan=3, pady=(15, 0))

        self.verdict_label = ctk.CTkLabel(self.info_frame, text="Awaiting Data...", font=ctk.CTkFont(size=42, weight="bold"), text_color="#64748B")
        self.verdict_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        # Add metric meters
        self.add_modern_meter(0, "Deepfake Video", "video_prob", "video_bar")
        self.add_modern_meter(1, "System Audio", "sys_audio_prob", "sys_audio_bar")
        self.add_modern_meter(2, "Microphone", "mic_audio_prob", "mic_audio_bar")



        # Async Model Loading
        self.models_loaded = False
        threading.Thread(target=self.load_models_async, daemon=True).start()

        # --- Settings Flyout ---
        self.settings_open = False
        self.settings_frame = ctk.CTkFrame(self, fg_color="#0F172A", border_width=2, border_color="#38BDF8", corner_radius=10, width=250, height=300)
        
        ctk.CTkLabel(self.settings_frame, text="Settings", font=ctk.CTkFont(size=20, weight="bold"), text_color="#F1F5F9").pack(pady=(15, 10))

        ctk.CTkLabel(self.settings_frame, text="Warning Threshold", text_color="#F59E0B").pack(pady=(10, 0))
        self.warn_slider = ctk.CTkSlider(self.settings_frame, from_=0.1, to=0.99, command=self.update_warn_thresh)
        self.warn_slider.set(VIDEO_CONFIG.get('warn_threshold', 0.75))
        self.warn_slider.pack(pady=5, padx=20)
        self.warn_label = ctk.CTkLabel(self.settings_frame, text=f"{self.warn_slider.get():.2f}")
        self.warn_label.pack(pady=(0, 5))

        ctk.CTkLabel(self.settings_frame, text="Critical Threshold", text_color="#EF4444").pack(pady=(10, 0))
        self.crit_slider = ctk.CTkSlider(self.settings_frame, from_=0.1, to=0.99, command=self.update_crit_thresh)
        self.crit_slider.set(VIDEO_CONFIG.get('high_conf_threshold', 0.95))
        self.crit_slider.pack(pady=5, padx=20)
        self.crit_label = ctk.CTkLabel(self.settings_frame, text=f"{self.crit_slider.get():.2f}")
        self.crit_label.pack(pady=(0, 5))
        
        self.save_btn = ctk.CTkButton(self.settings_frame, text="Save to Config", command=self.save_config, fg_color="#10B981", hover_color="#059669")
        self.save_btn.pack(pady=15)

        # Start GUI update loop
        self.update_gui()

    def add_modern_meter(self, col, title, prob_attr, bar_attr):
        frame = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        frame.grid(row=2, column=col, padx=10, pady=10, sticky="nsew")
        
        title_lbl = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=16, weight="bold"), text_color="#CBD5E1")
        title_lbl.pack(pady=(0, 5))
        
        val_lbl = ctk.CTkLabel(frame, text="0.0%", font=ctk.CTkFont(size=24, weight="bold"), text_color="#F1F5F9")
        val_lbl.pack(pady=0)
        setattr(self, prob_attr, val_lbl)
        
        bar = ctk.CTkProgressBar(frame, width=180, height=12)
        bar.set(0)
        bar.pack(pady=(10, 5))
        setattr(self, bar_attr, bar)
        
    def load_models_async(self):
        try:
            print("[Async] Loading models for in-memory processing...")
            audio_model = AudioModel.to(device)
            audio_model.eval()
            print("[Async] Audio model loaded successfully.")

            video_model_name = 'efficientnet-b0'
            blocks_args, global_params = get_model_params(video_model_name, {'num_classes': 2})
            video_model = EfficientNet(blocks_args, global_params).to(device)
            from utils import get_resource_path
            model_path = get_resource_path(os.path.join('models', 'video_deepfake_detector.pth'))
            state_dict = torch.load(model_path, map_location=device)
            video_model.load_state_dict(state_dict)
            video_model.eval()
            print("[Async] Video model loaded successfully.")

            self.models_loaded = True
            
            # Start capturing threads now that models are ready
            threading.Thread(target=system_audio_capture_thread, args=(audio_model,), daemon=True).start()
            threading.Thread(target=microphone_capture_thread, args=(audio_model,), daemon=True).start()
            threading.Thread(target=video_capture_thread, args=(video_model,), daemon=True).start()
            
            # Remove loading elements
            self.loading_label.destroy()
            self.loading_sub_label.destroy()
            self.loading_bar.stop()
            self.loading_bar.destroy()
            
            self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
            self.video_label.configure(text="Awaiting Video Stream...", text_color="#94A3B8")
            
        except Exception as e:
            print(f"[Async] Fatal error loading models: {e}")
            self.loading_label.configure(text=f"Integrity Check Failed", text_color="#EF4444")
            self.loading_sub_label.configure(text=f"Error: {e}", text_color="#EF4444")
            self.loading_bar.stop()



    def on_closing(self):
        """Shut down threads safely on window close."""
        with shared_state['lock']:
            shared_state['running'] = False
        self.destroy()
        
    def toggle_settings(self):
        if self.settings_open:
            self.settings_frame.place_forget()
            self.settings_open = False
        else:
            # Place on top right below header
            self.settings_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=60)
            self.settings_open = True

    def update_warn_thresh(self, value):
        VIDEO_CONFIG['warn_threshold'] = value
        self.warn_label.configure(text=f"{value:.2f}")

    def update_crit_thresh(self, value):
        VIDEO_CONFIG['high_conf_threshold'] = value
        self.crit_label.configure(text=f"{value:.2f}")

    def save_config(self):
        APP_CONFIG['video'] = VIDEO_CONFIG
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(APP_CONFIG, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def update_gui(self):
        """Periodically refresh GUI elements with thread safe data."""
        with shared_state['lock']:
            try:
                shared_state['mouse_x'] = self.winfo_pointerx()
                shared_state['mouse_y'] = self.winfo_pointery()
            except Exception:
                pass
            frame_rgb = shared_state['latest_frame']
            v_score = shared_state['video_score']
            s_score = shared_state['system_audio_score']
            m_score = shared_state['mic_audio_score']
            sys_act = shared_state['system_audio_active']
            mic_act = shared_state['mic_audio_active']
            sys_silence = shared_state['system_audio_silence']
            mic_silence = shared_state['mic_audio_silence']
        
        # 1. Update Video Feed
        if frame_rgb is not None and self.models_loaded:
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(700, 480))
            self.video_label.configure(image=ctk_img, text="")
        
        # 2. Update Scores and Logic
        active_scores = [v_score]
        if sys_act and not sys_silence: active_scores.append(s_score)
        if mic_act and not mic_silence: active_scores.append(m_score)
        final_score = np.mean(active_scores) if active_scores else 0.0

        if final_score > VIDEO_CONFIG.get('high_conf_threshold', 0.95): verdict, color = "⚠️ CRITICAL: DEEPFAKE DETECTED", "#EF4444"
        elif final_score > VIDEO_CONFIG.get('warn_threshold', 0.75): verdict, color = "⚠️ WARNING: SYNTHETIC PATTERNS", "#F59E0B"
        else: verdict, color = "✅ COMMUNICATION SECURE", "#10B981"

        if not self.models_loaded:
            verdict, color = "INITIALIZING", "#38BDF8"

        self.verdict_label.configure(text=verdict, text_color=color)

        self.video_prob.configure(text=f"{v_score*100:.1f}%")
        self.video_bar.set(v_score)
        self.video_bar.configure(progress_color="#EF4444" if v_score > VIDEO_CONFIG.get('warn_threshold', 0.75) else "#10B981")

        if sys_act:
            if sys_silence:
                self.sys_audio_prob.configure(text="Silence", text_color="#64748B")
                self.sys_audio_bar.set(s_score)
                self.sys_audio_bar.configure(progress_color="#334155")
            else:
                self.sys_audio_prob.configure(text=f"{s_score*100:.1f}%", text_color="#F1F5F9")
                self.sys_audio_bar.set(s_score)
                self.sys_audio_bar.configure(progress_color="#EF4444" if s_score > VIDEO_CONFIG.get('warn_threshold', 0.75) else "#10B981")
        else:
            self.sys_audio_prob.configure(text="No Loopback", text_color="#64748B")
            self.sys_audio_bar.set(0)
            self.sys_audio_bar.configure(progress_color="#334155")

        if mic_act:
            if mic_silence:
                self.mic_audio_prob.configure(text="Silence", text_color="#64748B")
                self.mic_audio_bar.set(m_score)
                self.mic_audio_bar.configure(progress_color="#334155")
            else:
                self.mic_audio_prob.configure(text=f"{m_score*100:.1f}%", text_color="#F1F5F9")
                self.mic_audio_bar.set(m_score)
                self.mic_audio_bar.configure(progress_color="#EF4444" if m_score > VIDEO_CONFIG.get('warn_threshold', 0.75) else "#10B981")
        else:
            self.mic_audio_prob.configure(text="No Mic", text_color="#64748B")
            self.mic_audio_bar.set(0)
            self.mic_audio_bar.configure(progress_color="#334155")

        # Reschedule update GUI
        self.after(30, self.update_gui)

def main() -> None:
    print("Starting Advanced Live Multimodal Desktop Detector (GUI)...")
    app = LiveDetectorApp()
    app.mainloop()
    print("Detector stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: Could not start the application. Error: {e}")
