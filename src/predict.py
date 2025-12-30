import torch
import torch.nn as nn
import torchaudio
import os
import math

# =====================================================================================
# == ACTUAL TRANSFORMER MODEL ARCHITECTURE (CORRECTED & FINAL)                     ==
# =====================================================================================
# This architecture is based on the layer names from your error log.

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # CORRECTED: Reverted shape of `pe` to match the state_dict from the checkpoint.
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # CORRECTED: Adapt the non-batch-first `pe` tensor for use with batch-first `x`.
        # self.pe is [max_len, 1, dim], we permute it to [1, max_len, dim] to broadcast.
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return self.dropout(x)

class AudioTransformer(nn.Module):
    """
    A Transformer model for audio classification, with corrected hyperparameters.
    """
    def __init__(self, n_mels=128, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(AudioTransformer, self).__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src):
        # src shape: [batch_size, n_mels, seq_len]
        src = src.permute(0, 2, 1) # -> [batch_size, seq_len, n_mels]
        src = self.input_proj(src) # -> [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) # -> [batch_size, seq_len, d_model]
        output = output.mean(dim=1) # -> [batch_size, d_model]
        output = self.classifier(output) # -> [batch_size, 2]
        return output

# =====================================================================================

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the corrected model class.
model = AudioTransformer().to(device)

from utils import get_resource_path
model_path = get_resource_path(os.path.join('models', 'transformer_voice_detector.pth'))
print(f"Loading voice model from: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Preprocessing Setup ---
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
).to(device)


def predict_audio_deepfake(audio_path: str) -> float:
    """
    Analyzes an audio file to predict if it's a deepfake using the Transformer model.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.to(device)

        # --- PREPROCESSING ---
        spec = mel_spectrogram(waveform)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        # --- INFERENCE ---
        with torch.no_grad():
            output = model(spec)
            probabilities = torch.softmax(output, dim=1)
            score = probabilities[0, 1].item() # Probability of the "fake" class
            return score
            
    except Exception as e:
        print(f"Error during audio prediction: {e}")
        return 0.0
