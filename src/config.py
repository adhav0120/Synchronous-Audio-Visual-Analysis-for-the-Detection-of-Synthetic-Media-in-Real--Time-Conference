# src/config.py

import torch
from pathlib import Path

# --- Project Paths ---
# This automatically finds the root of your project (the 'deepfake-voice-detector' folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'

# ASVspoof Dataset paths (for voice)
# Actual extracted layout: data/ASVspoof 2019 Dataset/LA/LA/...
ASVSPOOF_LA_ROOT          = DATA_ROOT / 'ASVspoof 2019 Dataset' / 'LA' / 'LA'
VOICE_DATA_DIR            = ASVSPOOF_LA_ROOT / 'ASVspoof2019_LA_train' / 'flac'
VOICE_TRAIN_PROTOCOL_PATH = ASVSPOOF_LA_ROOT / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trn.txt'
VOICE_DEV_PROTOCOL_PATH   = ASVSPOOF_LA_ROOT / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.dev.trl.txt'
VOICE_EVAL_PROTOCOL_PATH  = ASVSPOOF_LA_ROOT / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.eval.trl.txt'
VOICE_EVAL_DATA_DIR       = ASVSPOOF_LA_ROOT / 'ASVspoof2019_LA_eval' / 'flac'

# LibriSpeech (genuine speech negatives for calibrated retraining)
LIBRISPEECH_DIR           = DATA_ROOT / 'LibriSpeech' / 'dev-clean'

# MUSAN Dataset Path
MUSAN_PATH = DATA_ROOT / 'musan'

# Model save paths
VOICE_MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'transformer_voice_detector.pth'
VIDEO_MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'video_deepfake_detector.pth'

# --- Model & Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE for video model training. Start with a smaller value.
BATCH_SIZE = 16 
NUM_WORKERS = 0 # Keep at 0 for Windows
LEARNING_RATE = 1e-4
EPOCHS = 10 # Start with 10 epochs for the first run

# --- Audio Feature Configuration (for voice model) ---
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
MAX_LEN_SECONDS = 5
