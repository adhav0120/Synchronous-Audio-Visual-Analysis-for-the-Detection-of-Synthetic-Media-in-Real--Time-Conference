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
import sys
import json
import os

# Limit PyTorch cores
torch.set_num_threads(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}

APP_CONFIG = load_config()
AUDIO_CONFIG = APP_CONFIG.get('audio', {})
VIDEO_CONFIG = APP_CONFIG.get('video', {})

from predict import model as AudioModel, mel_spectrogram
from video_model import EfficientNet, get_model_params

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
        probabilities = torch.softmax(output, dim=1)
        score = probabilities[0][1].item()
    return score

shared_state = {
    'video_score': 0.0, 'video_ema': -1.0, 'system_audio_score': 0.0, 'mic_audio_score': 0.0,
    'system_audio_active': False, 'mic_audio_active': False, 
    'system_audio_silence': True, 'mic_audio_silence': True,
    'running': True, 'active': False, 'lock': threading.Lock()
}

def is_silence(audio_data: np.ndarray, threshold: float = None) -> bool:
    if threshold is None: threshold = AUDIO_CONFIG.get('silence_threshold', 0.002)
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < threshold

def system_audio_capture_thread(audio_model: torch.nn.Module) -> None:
    try:
        mics = sc.all_microphones(include_loopback=True)
        loopback_mic = None
        try:
            default_spk_name = sc.default_speaker().name
            loopback_mic = next((m for m in mics if m.name == default_spk_name and getattr(m, 'isloopback', False)), None)
        except Exception:
            pass
        if loopback_mic is None:
            loopback_mic = next((m for m in mics if getattr(m, 'isloopback', False)), None)
        if loopback_mic is None:
            loopback_mic = next((m for m in mics if 'loopback' in m.name.lower() or 'stereo mix' in m.name.lower() or 'speaker' in m.name.lower()), None)
            
        if loopback_mic is None:
            return
        with shared_state['lock']: shared_state['system_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        
        def run_inference(data_chunk):
            score = predict_audio_from_chunk(data_chunk, audio_model)
            with shared_state['lock']:
                shared_state['system_audio_score'] = score
                
        with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while True:
                with shared_state['lock']:
                    if not shared_state['running']: break
                    is_active = shared_state['active']
                
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                if not is_active: continue

                flat_audio = audio_data.flatten()
                silence = is_silence(flat_audio)
                with shared_state['lock']:
                    shared_state['system_audio_silence'] = silence
                if not silence:
                    threading.Thread(target=run_inference, args=(flat_audio.copy(),), daemon=True).start()
    except Exception as e:
        pass
    finally:
        with shared_state['lock']: shared_state['system_audio_active'] = False

def microphone_capture_thread(audio_model: torch.nn.Module) -> None:
    try:
        mic = sc.default_microphone()
        if mic is None: return
        with shared_state['lock']: shared_state['mic_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        
        def run_inference(data_chunk):
            score = predict_audio_from_chunk(data_chunk, audio_model)
            with shared_state['lock']:
                shared_state['mic_audio_score'] = score

        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while True:
                with shared_state['lock']:
                    if not shared_state['running']: break
                    is_active = shared_state['active']
                
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                if not is_active: continue

                flat_audio = audio_data.flatten()
                silence = is_silence(flat_audio)
                with shared_state['lock']:
                    shared_state['mic_audio_silence'] = silence
                if not silence:
                    threading.Thread(target=run_inference, args=(flat_audio.copy(),), daemon=True).start()
    except Exception as e:
        pass
    finally:
        with shared_state['lock']: shared_state['mic_audio_active'] = False

def video_capture_thread(video_model: torch.nn.Module) -> None:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    last_prediction_time = time.time()
    
    MIN_FACE_SIZE = VIDEO_CONFIG.get('min_face_size', 50)
    prediction_interval = VIDEO_CONFIG.get('prediction_interval', 0.5)

    with mss.mss() as sct:
        while True:
            with shared_state['lock']:
                if not shared_state['running']:
                    break
                is_active = shared_state['active']
            
            if not is_active:
                time.sleep(0.5)
                continue

            # Default to primary monitor full screen 
            # (assuming WhatsApp is open or we want to capture the whole screen)
            current_monitor = sct.monitors[1]
            
            monitor = {
                "top": current_monitor["top"],
                "left": current_monitor["left"],
                "width": current_monitor["width"],
                "height": current_monitor["height"]
            }

            frame = np.array(sct.grab(monitor))
            frame = np.ascontiguousarray(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
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
            time.sleep(0.01)
    face_mesh.close()

def input_listener():
    try:
        # Listen for pause/resume commands
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == 'stop':
                with shared_state['lock']:
                    shared_state['running'] = False
                break
            elif cmd == 'pause':
                with shared_state['lock']:
                    shared_state['active'] = False
            elif cmd == 'resume':
                with shared_state['lock']:
                    shared_state['active'] = True
    except:
        pass
    with shared_state['lock']:
        shared_state['running'] = False

def main():
    try:
        audio_model = AudioModel.to(device)
        audio_model.eval()

        video_model_name = 'efficientnet-b0'
        blocks_args, global_params = get_model_params(video_model_name, {'num_classes': 2})
        video_model = EfficientNet(blocks_args, global_params).to(device)
        
        from utils import get_resource_path
        model_path = get_resource_path(os.path.join('models', 'video_deepfake_detector.pth'))
        state_dict = torch.load(model_path, map_location=device)
        video_model.load_state_dict(state_dict)
        video_model.eval()

        threading.Thread(target=system_audio_capture_thread, args=(audio_model,), daemon=True).start()
        threading.Thread(target=microphone_capture_thread, args=(audio_model,), daemon=True).start()
        threading.Thread(target=video_capture_thread, args=(video_model,), daemon=True).start()
        threading.Thread(target=input_listener, daemon=True).start()

        print(json.dumps({"event": "ready", "status": "started"}), flush=True)

        while True:
            # Output state every 1 second
            with shared_state['lock']:
                if not shared_state['running']:
                    break
                is_active = shared_state['active']
                v_score = shared_state['video_score']
                s_score = shared_state['system_audio_score']
                sys_act = shared_state['system_audio_active']
                sys_silence = shared_state['system_audio_silence']
                
            if is_active:
                payload = {
                    "event": "deepfake_score",
                    "payload": {
                        "video": v_score,
                        "system_audio": s_score if (sys_act and not sys_silence) else 0.0
                    }
                }
                print(json.dumps(payload), flush=True)
            time.sleep(1)
            time.sleep(1)

    except KeyboardInterrupt:
        with shared_state['lock']:
            shared_state['running'] = False

if __name__ == "__main__":
    main()
