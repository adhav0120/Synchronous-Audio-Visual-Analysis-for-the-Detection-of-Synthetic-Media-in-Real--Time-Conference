# src/retrain_calibrated.py
"""
Calibrated Audio Model Retraining
===================================
Fine-tunes the existing transformer_voice_detector.pth on a balanced dataset:
  - Spoof (fake)    : ASVspoof 2019 LA train spoof samples
  - Bonafide (real) : ASVspoof 2019 LA train bonafide  +  LibriSpeech dev-clean

Uses WeightedRandomSampler to guarantee 50/50 class balance per batch,
and label-smoothed CrossEntropyLoss to prevent overconfident fake predictions.

Run from project root:
    .venv\\Scripts\\python.exe src/retrain_calibrated.py

After retraining, the old model is backed up as:
    models/transformer_voice_detector.pth.bak
and the new calibrated model is saved as:
    models/transformer_voice_detector.pth
"""

import os
import sys
import math
import shutil
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import soundfile as sf
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
SRC_DIR      = PROJECT_ROOT / "src"

# Add src/ to sys.path so we can import predict.py's AudioTransformer
sys.path.insert(0, str(SRC_DIR))

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("RETRAIN")

# ── hyperparameters ────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
MAX_LEN_SEC      = 5              # clip every sample to 5 s
N_FFT            = 1024
HOP_LENGTH       = 512
N_MELS           = 128
BATCH_SIZE       = 32             # larger batch to better utilise RTX 3050 VRAM
LEARNING_RATE    = 5e-5           # small LR for fine-tuning
EPOCHS           = 10
LABEL_SMOOTHING  = 0.1            # prevents overconfident logits
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SRC        = MODELS_DIR / "transformer_voice_detector.pth"
MODEL_BACKUP     = MODELS_DIR / "transformer_voice_detector.pth.bak"
MODEL_DEST       = MODELS_DIR / "transformer_voice_detector.pth"

# Dataset paths — actual extracted layout:
#   data/ASVspoof 2019 Dataset/LA/LA/ASVspoof2019_LA_train/flac/
ASVSPOOF_LA_ROOT       = DATA_DIR / "ASVspoof 2019 Dataset" / "LA" / "LA"
ASVSPOOOF_TRAIN_FLAC   = ASVSPOOF_LA_ROOT / "ASVspoof2019_LA_train" / "flac"
ASVSPOOF_DEV_FLAC      = ASVSPOOF_LA_ROOT / "ASVspoof2019_LA_dev" / "flac"
ASVSPOOF_TRAIN_PROTO   = ASVSPOOF_LA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
ASVSPOOF_DEV_PROTO     = ASVSPOOF_LA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
LIBRISPEECH_DIR        = DATA_DIR / "LibriSpeech" / "dev-clean"


# ═══════════════════════════════════════════════════════════════════════════
# Audio helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_and_pad(path: Path, max_len: int) -> torch.Tensor:
    """
    Load a flac/wav file via soundfile (BytesIO trick avoids libsndfile
    Windows bug with spaces in directory names), mono-mix, resample,
    pad/truncate to max_len samples.
    """
    # Read raw bytes first — soundfile's BytesIO path is immune to
    # Windows libsndfile space-in-path issues.
    raw = path.read_bytes()
    data, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)

    # soundfile returns [samples] for mono or [samples, channels] for stereo
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)   # [1, T]
    else:
        waveform = torch.from_numpy(data.T)              # [channels, T]
        waveform = waveform.mean(dim=0, keepdim=True)    # mono

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    # Pad or truncate
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    elif waveform.shape[1] < max_len:
        pad = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform  # [1, max_len]


mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
).to("cpu")

amp_to_db = T.AmplitudeToDB()


def waveform_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [1, T] waveform → [N_MELS, frames] mel spectrogram (on CPU)."""
    mel = mel_transform(waveform)          # [1, n_mels, frames]
    mel = amp_to_db(mel)
    return mel.squeeze(0)                  # [n_mels, frames]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class CalibratedAudioDataset(Dataset):
    """
    Combines three sources:
      label=1 (spoof/fake)   : ASVspoof 2019 LA train spoof files
      label=0 (bonafide/real): ASVspoof 2019 LA train bonafide  +  LibriSpeech dev-clean

    If ASVspoof data is not available, only LibriSpeech is used for bonafide
    (this won't work well — both datasets are needed for good discrimination).
    """

    def __init__(self, augment: bool = True):
        self.max_len  = MAX_LEN_SEC * SAMPLE_RATE
        self.augment  = augment
        self.samples: List[Tuple[Path, int]] = []   # (path, label)

        # ── ASVspoof 2019 LA train ────────────────────────────────────────
        if ASVSPOOF_TRAIN_PROTO.exists() and ASVSPOOOF_TRAIN_FLAC.exists():
            spoof_count, bonafide_count = 0, 0
            with open(ASVSPOOF_TRAIN_PROTO) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    fname, key = parts[1], parts[4]
                    audio_path = ASVSPOOOF_TRAIN_FLAC / f"{fname}.flac"
                    if not audio_path.exists():
                        continue
                    label = 1 if key == "spoof" else 0
                    self.samples.append((audio_path, label))
                    if label == 1:
                        spoof_count += 1
                    else:
                        bonafide_count += 1
            log.info(f"ASVspoof train: {spoof_count} spoof, {bonafide_count} bonafide")
        else:
            log.warning("ASVspoof 2019 LA train not found — skipping spoof samples!")

        # ── LibriSpeech dev-clean (all bonafide) ─────────────────────────
        if LIBRISPEECH_DIR.exists():
            libri_files = list(LIBRISPEECH_DIR.rglob("*.flac"))
            for p in libri_files:
                self.samples.append((p, 0))  # all bonafide
            log.info(f"LibriSpeech dev-clean: {len(libri_files)} bonafide clips")
        else:
            log.warning("LibriSpeech dev-clean not found — no real-world negatives!")

        if not self.samples:
            raise RuntimeError("No training samples found. Run download_datasets.py first.")

        # Shuffle for reproducibility
        random.seed(42)
        random.shuffle(self.samples)

        labels = [s[1] for s in self.samples]
        n_spoof = sum(labels)
        n_bona  = len(labels) - n_spoof
        log.info(f"Total dataset: {len(self.samples)} samples  "
                 f"({n_spoof} spoof / {n_bona} bonafide)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform = load_and_pad(path, self.max_len)
        except Exception as e:
            log.warning(f"Skipping corrupt file {path.name}: {e}")
            # Return a zero tensor so the batch doesn't crash
            waveform = torch.zeros(1, self.max_len)

        # Light augmentation for training
        if self.augment and random.random() < 0.5:
            # Add very subtle Gaussian noise (SNR ~30 dB) to improve robustness
            noise = torch.randn_like(waveform) * 0.001
            waveform = waveform + noise

        mel = waveform_to_mel(waveform)  # [n_mels, frames]
        return mel, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self) -> List[float]:
        """Compute per-sample weights for WeightedRandomSampler (50/50 balance)."""
        labels = [s[1] for s in self.samples]
        n_total  = len(labels)
        n_spoof  = sum(labels)
        n_bona   = n_total - n_spoof
        w_spoof  = 1.0 / max(n_spoof, 1)
        w_bona   = 1.0 / max(n_bona,  1)
        weights  = [w_spoof if l == 1 else w_bona for l in labels]
        return weights


# ASVspoof dev set for evaluation only — capped at 2000 samples for speed
class ASVSpoofDevDataset(Dataset):
    def __init__(self, max_samples: int = 2000):
        self.max_len = MAX_LEN_SEC * SAMPLE_RATE
        self.samples: List[Tuple[Path, int]] = []
        if ASVSPOOF_DEV_PROTO.exists() and ASVSPOOF_DEV_FLAC.exists():
            with open(ASVSPOOF_DEV_PROTO) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    fname, key = parts[1], parts[4]
                    p = ASVSPOOF_DEV_FLAC / f"{fname}.flac"
                    if p.exists():
                        self.samples.append((p, 1 if key == "spoof" else 0))
                    if len(self.samples) >= max_samples:
                        break
        log.info(f"ASVspoof dev set : {len(self.samples)} samples (capped at {max_samples})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform = load_and_pad(path, self.max_len)
        except Exception:
            waveform = torch.zeros(1, self.max_len)
        mel = waveform_to_mel(waveform)
        return mel, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# Model — reuse the exact AudioTransformer from predict.py
# ═══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].permute(1, 0, 2)
        return self.dropout(x)


class AudioTransformer(nn.Module):
    def __init__(self, n_mels=128, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src):
        # src: [B, n_mels, T]
        src = src.permute(0, 2, 1)             # → [B, T, n_mels]
        src = self.input_proj(src)             # → [B, T, d_model]
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)    # → [B, T, d_model]
        out = out.mean(dim=1)                  # → [B, d_model]
        return self.classifier(out)            # → [B, 2]


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_eer(scores: List[float], labels: List[int]) -> float:
    """Compute Equal Error Rate (EER) from raw fake-class probabilities."""
    import numpy as np
    scores = np.array(scores)
    labels = np.array(labels)
    thresholds = np.linspace(0, 1, 1000)
    best_eer = 1.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        fpr = fp / max(np.sum(labels == 0), 1)
        fnr = fn / max(np.sum(labels == 1), 1)
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer = eer
    return float(best_eer)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float, float]:
    """
    Returns (accuracy, EER, avg_bonafide_fake_logit, avg_spoof_fake_logit).
    The logit values tell us how much the model scores genuine speech as fake.
    """
    model.eval()
    all_scores, all_labels, all_logits = [], [], []
    correct = 0
    total   = 0
    for mels, labels in loader:
        mels = mels.to(DEVICE)
        logits = model(mels)                              # [B, 2]
        probs  = torch.softmax(logits, dim=1)[:, 1]      # fake probability
        preds  = torch.argmax(logits, dim=1)
        correct += (preds.cpu() == labels).sum().item()
        total   += labels.size(0)
        all_scores.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_logits.extend(logits[:, 1].cpu().tolist())   # raw fake logit

    acc  = correct / max(total, 1)
    eer  = compute_eer(all_scores, all_labels)

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    avg_bona_logit  = float(logits_arr[labels_arr == 0].mean()) if (labels_arr == 0).any() else 0.0
    avg_spoof_logit = float(logits_arr[labels_arr == 1].mean()) if (labels_arr == 1).any() else 0.0

    model.train()
    return acc, eer, avg_bona_logit, avg_spoof_logit


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 62)
    log.info("  SHIELD — Calibrated Audio Model Retraining")
    log.info("=" * 62)
    log.info(f"  Device  : {DEVICE}")
    log.info(f"  Epochs  : {EPOCHS}")
    log.info(f"  LR      : {LEARNING_RATE}")
    log.info(f"  Batch   : {BATCH_SIZE}")
    log.info(f"  Label Smoothing: {LABEL_SMOOTHING}")

    # ── Dataset ────────────────────────────────────────────────────────────
    log.info("\nBuilding training dataset…")
    train_dataset = CalibratedAudioDataset(augment=True)

    weights  = train_dataset.get_class_weights()
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,     # Windows compatibility
        pin_memory=False,
    )

    # Dev set — 2000-sample subset for fast per-epoch evaluation
    dev_dataset = ASVSpoofDevDataset(max_samples=2000)
    have_dev    = len(dev_dataset) > 0
    if have_dev:
        dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    log.info("\nLoading pretrained weights…")
    model = AudioTransformer().to(DEVICE)

    if MODEL_SRC.exists():
        state = torch.load(MODEL_SRC, map_location=DEVICE)
        model.load_state_dict(state)
        log.info(f"  ✓ Loaded: {MODEL_SRC}")
    else:
        log.warning(f"  No pretrained model at {MODEL_SRC} — training from scratch.")

    # ── Quick baseline calibration (500 samples) ───────────────────────────
    if have_dev:
        log.info("\n── Baseline check (500-sample subset) ──")
        quick_loader = DataLoader(
            torch.utils.data.Subset(dev_dataset, range(min(500, len(dev_dataset)))),
            batch_size=64, shuffle=False, num_workers=0
        )
        acc, eer, bona_l, spoof_l = evaluate(model, quick_loader)
        log.info(f"  Accuracy : {acc*100:.1f}%   EER: {eer*100:.1f}%")
        log.info(f"  Fake-logit on bonafide: {bona_l:+.2f}  (band-aid needed this much)")
        log.info(f"  Fake-logit on spoof   : {spoof_l:+.2f}")

    # ── Optimiser & Loss ───────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_eer   = 1.0
    best_state = None

    # ── Epoch loop ─────────────────────────────────────────────────────────
    log.info("\n── Training ──────────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for step, (mels, labels) in enumerate(train_loader, 1):
            # mels: [B, N_MELS, T]  (from CalibratedAudioDataset)
            mels   = mels.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(mels)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds   = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                acc_so_far = correct / max(total, 1)
                print(f"\r  Epoch {epoch}/{EPOCHS}  step {step}/{len(train_loader)}"
                      f"  loss={loss.item():.4f}  acc={acc_so_far*100:.1f}%",
                      end="", flush=True)

        print()
        epoch_loss = running_loss / max(total, 1)
        epoch_acc  = correct / max(total, 1)
        scheduler.step()

        # Evaluate on dev
        if have_dev:
            dev_acc, dev_eer, bona_l, spoof_l = evaluate(model, dev_loader)
            log.info(
                f"  Epoch {epoch:02d}/{EPOCHS}  "
                f"train_loss={epoch_loss:.4f}  train_acc={epoch_acc*100:.1f}%  "
                f"dev_acc={dev_acc*100:.1f}%  dev_EER={dev_eer*100:.2f}%  "
                f"bona_logit={bona_l:+.2f}  spoof_logit={spoof_l:+.2f}"
            )
            if dev_eer < best_eer:
                best_eer   = dev_eer
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info(f"    ★ New best EER: {best_eer*100:.2f}% — checkpoint saved")
        else:
            log.info(f"  Epoch {epoch:02d}/{EPOCHS}  "
                     f"train_loss={epoch_loss:.4f}  train_acc={epoch_acc*100:.1f}%")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save best checkpoint ───────────────────────────────────────────────
    log.info("\n── Saving model ─────────────────────────────────────────")
    if MODEL_SRC.exists():
        shutil.copy2(MODEL_SRC, MODEL_BACKUP)
        log.info(f"  ✓ Backup saved: {MODEL_BACKUP}")

    if best_state is not None:
        torch.save(best_state, MODEL_DEST)
    else:
        torch.save(model.state_dict(), MODEL_DEST)
    log.info(f"  ✓ Calibrated model saved: {MODEL_DEST}")

    # ── Final calibration report ───────────────────────────────────────────
    if have_dev:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        log.info("\n── Final Calibration Report ─────────────────────────────")
        acc, eer, bona_l, spoof_l = evaluate(model, dev_loader)
        log.info(f"  Dev accuracy         : {acc*100:.2f}%")
        log.info(f"  Dev EER              : {eer*100:.2f}%  (target < 5%)")
        log.info(f"  Avg fake-logit (bonafide) : {bona_l:+.3f}")
        log.info(f"  Avg fake-logit (spoof)    : {spoof_l:+.3f}")
        log.info("")
        if bona_l <= 0.5:
            log.info("  ✅ Model is naturally calibrated!")
            log.info("     The fake_logit_penalty in config.json can be removed.")
        else:
            log.info(f"  ⚠  Bonafide fake-logit still positive ({bona_l:+.2f}).")
            log.info(f"     Consider more epochs or more bonafide training data.")
        log.info("=" * 62)
        log.info("  Next step:  remove band-aids from desktop_live_capture.py")
        log.info("              by running:  python src/remove_bandaids.py")
        log.info("=" * 62)


if __name__ == "__main__":
    main()
