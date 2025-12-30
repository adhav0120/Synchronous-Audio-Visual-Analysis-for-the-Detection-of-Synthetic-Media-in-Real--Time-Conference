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
# Reverted explicit imports that caused ModuleNotFoundError
# We will solve this via PyInstaller hidden imports instead
from collections import deque, namedtuple
import re
import math
import os
from torch.utils import model_zoo

# Import the actual model classes and prediction functions
from predict import model as AudioModel, mel_spectrogram

# --- GPU / CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")
if not torch.cuda.is_available():
    print("--- WARNING: No CUDA GPU detected. The application will run on the CPU and may be slow. ---")

# --- EfficientNet Model Definition (Self-contained for portability) ---
GlobalParams = namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])
BlockArgs = namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class MBConvBlock(nn.Module):
    """ Mobile Inverted Residual Bottleneck Block """
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k, s = self._block_args.kernel_size, self._block_args.stride
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, padding=(k - 1) // 2, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1: x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        if self.id_skip and self._block_args.stride == 1 and self._block_args.input_filters == self._block_args.output_filters:
            if drop_connect_rate: x = F.dropout(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

class EfficientNet(nn.Module):
    """ An EfficientNet model. """
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        self._global_params, self._blocks_args = global_params, blocks_args
        bn_mom, bn_eps = 1 - self._global_params.batch_norm_momentum, self._global_params.batch_norm_epsilon
        in_channels, out_channels = 3, round_filters(32, self._global_params)
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1): self._blocks.append(MBConvBlock(block_args, self._global_params))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def forward(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate: drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout: x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

def efficientnet_params(model_name): return (1.0, 1.0, 224, 0.2)

class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        ops, options = block_string.split('_'), {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2: key, value = splits[:2]; options[key] = value
        return BlockArgs(
            kernel_size=int(options['k']), num_repeat=int(options['r']),
            input_filters=int(options['i']), output_filters=int(options['o']),
            expand_ratio=int(options['e']), id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])
    @staticmethod
    def decode(string_list): return [BlockDecoder._decode_block_string(s) for s in string_list]

def get_model_params(model_name, override_params):
    w, d, s, p = efficientnet_params(model_name)
    blocks_args = BlockDecoder.decode(['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25'])
    global_params = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, dropout_rate=p, drop_connect_rate=0.2, num_classes=1000, width_coefficient=w, depth_coefficient=d, depth_divisor=8, min_depth=None, image_size=s)
    if override_params: global_params = global_params._replace(**override_params)
    return blocks_args, global_params
# --- End of EfficientNet Definition ---

# --- In-Memory Prediction Functions ---
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_video_from_frame(frame_rgb, model):
    if frame_rgb is None or model is None or frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0: return 0.0
    input_tensor = video_transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        score = probabilities[0][1].item()
    return score

def predict_audio_from_chunk(audio_data, model):
    if audio_data is None or model is None: return 0.0
    waveform = torch.from_numpy(audio_data).unsqueeze(0).to(device)
    mel_spec = mel_spectrogram(waveform).to(device)
    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.softmax(output, dim=1)
        score = probabilities[0][1].item()
    return score

# --- Shared State for All Threads ---
shared_state = {
    'video_score': 0.0, 'system_audio_score': 0.0, 'mic_audio_score': 0.0,
    'system_audio_active': False, 'mic_audio_active': False, 'lock': threading.Lock()
}

# --- Audio Capture Threads ---
def system_audio_capture_thread(audio_model):
    """Continuously captures and analyzes system audio (what you hear)."""
    try:
        mics = sc.all_microphones(include_loopback=True)
        loopback_mic = next((m for m in mics if 'loopback' in m.name.lower() or 'stereo mix' in m.name.lower()), None)
        if loopback_mic is None:
            print("[System Audio] ERROR: No loopback audio device found.")
            return
        print(f"\n[System Audio] Success! Monitoring from: {loopback_mic.name}")
        with shared_state['lock']: shared_state['system_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        while True:
            with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                score = predict_audio_from_chunk(audio_data.flatten(), audio_model)
                with shared_state['lock']: shared_state['system_audio_score'] = score
    except Exception as e: print(f"[System Audio] Error: {e}")
    finally:
        with shared_state['lock']: shared_state['system_audio_active'] = False

def microphone_capture_thread(audio_model):
    """Continuously captures and analyzes audio from the default microphone."""
    try:
        mic = sc.default_microphone()
        if mic is None:
            print("[Mic Audio] ERROR: No default microphone found.")
            return
        print(f"[Mic Audio] Success! Monitoring from: {mic.name}")
        with shared_state['lock']: shared_state['mic_audio_active'] = True
        SAMPLE_RATE, CHUNK_SAMPLES = 16000, 16000 * 2
        while True:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                audio_data = rec.record(numframes=CHUNK_SAMPLES)
                score = predict_audio_from_chunk(audio_data.flatten(), audio_model)
                with shared_state['lock']: shared_state['mic_audio_score'] = score
    except Exception as e: print(f"[Mic Audio] Error: {e}")
    finally:
        with shared_state['lock']: shared_state['mic_audio_active'] = False

# --- Main Video Capture and UI Thread ---
def main(video_model, audio_model):
    """Main function for screen capture, video detection, and UI rendering."""
    print("Starting Live Multimodal Desktop Detector...")
    print("Press 'q' on the detector window to quit.")
    
    system_audio_thread = threading.Thread(target=system_audio_capture_thread, args=(audio_model,), daemon=True)
    microphone_thread = threading.Thread(target=microphone_capture_thread, args=(audio_model,), daemon=True)
    system_audio_thread.start()
    microphone_thread.start()
    
    monitor = {"top": 100, "left": 100, "width": 1000, "height": 1000}
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    video_score_buffer = deque(maxlen=5)
    last_prediction_time = time.time()
    
    HIGH_CONF_THRESHOLD, WARN_THRESHOLD, MIN_FACE_SIZE = 0.95, 0.75, 50

    with mss.mss() as sct:
        while True:
            frame = np.array(sct.grab(monitor))
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
                        if time.time() - last_prediction_time > 0.5:
                            face_crop = frame_rgb[y:y+box_h, x:x+box_w]
                            score = predict_video_from_frame(face_crop, video_model)
                            video_score_buffer.append(score)
                            with shared_state['lock']:
                                shared_state['video_score'] = np.mean(video_score_buffer) if video_score_buffer else 0.0
                            last_prediction_time = time.time()
            else:
                 cv2.putText(display_frame, "Searching for face...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            with shared_state['lock']:
                video_score, system_audio_score, mic_audio_score, system_audio_active, mic_audio_active = (
                    shared_state['video_score'], shared_state['system_audio_score'], shared_state['mic_audio_score'],
                    shared_state['system_audio_active'], shared_state['mic_audio_active'])
            active_scores = [video_score]
            if system_audio_active: active_scores.append(system_audio_score)
            if mic_audio_active: active_scores.append(mic_audio_score)
            final_score = np.mean(active_scores) if active_scores else 0.0
            
            if final_score > HIGH_CONF_THRESHOLD: verdict, color = "Deepfake Detected", (0, 0, 255)
            elif final_score > WARN_THRESHOLD: verdict, color = "Warning: High Score", (0, 255, 255)
            else: verdict, color = "Likely Real", (0, 255, 0)
            
            y_pos = 40
            cv2.putText(display_frame, f"Verdict: {verdict} ({final_score * 100:.1f}%)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_pos += 40
            cv2.putText(display_frame, f"Video Score: {video_score * 100:.1f}% Fake", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 30
            if system_audio_active: cv2.putText(display_frame, f"System Audio: {system_audio_score * 100:.1f}% Fake", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else: cv2.putText(display_frame, "System Audio: Failed - Check Console", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_pos += 30
            if mic_audio_active: cv2.putText(display_frame, f"Mic Audio: {mic_audio_score * 100:.1f}% Fake", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else: cv2.putText(display_frame, "Mic Audio: Failed / Not Found", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("Live Multimodal Detector", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("c"):
                break

    cv2.destroyAllWindows()
    face_mesh.close()
    print("Detector stopped.")

if __name__ == "__main__":
    try:
        print("Loading models for in-memory processing...")
        # Audio Model
        audio_model = AudioModel.to(device)
        audio_model.eval()
        print("Audio model loaded successfully.")

        # Video Model
        video_model_name = 'efficientnet-b0'
        blocks_args, global_params = get_model_params(video_model_name, {'num_classes': 2})
        video_model = EfficientNet(blocks_args, global_params).to(device)
        from utils import get_resource_path
        model_path = get_resource_path(os.path.join('models', 'video_deepfake_detector.pth'))
        state_dict = torch.load(model_path, map_location=device)
        video_model.load_state_dict(state_dict)
        video_model.eval()
        print("Video model loaded successfully.")
        
        # Start the main application loop
        main(video_model, audio_model)

    except Exception as e:
        print(f"FATAL: Could not start the application. Error: {e}")

