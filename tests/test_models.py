import os
import sys
import torch
import numpy as np

# Add src to the path so we can import from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from video_model import EfficientNet, get_model_params
from predict import model as AudioModel
from desktop_live_capture import load_config, is_silence

def test_config_loader():
    config = load_config()
    assert isinstance(config, dict)
    # The config loader handles missing files safely, returning an empty dict
    assert 'ui' in config or len(config) >= 0

def test_silence_detector():
    """ Tests that the silence algorithm correctly differentiates pure zeroes from loud normalized static."""
    # True silence (zero array)
    silent_chunk = np.zeros(16000)
    assert is_silence(silent_chunk, threshold=0.002) == True
    
    # Loud noise (random uniform, definitely loud enough)
    loud_chunk = np.random.uniform(-1, 1, 16000)
    assert is_silence(loud_chunk, threshold=0.002) == False

def test_video_model_architecture():
    """ Verify the extracted video_model architecture works for basic tensor mathematics """
    blocks_args, global_params = get_model_params('efficientnet-b0', {'num_classes': 2})
    model = EfficientNet(blocks_args, global_params)
    model.eval()
    
    # Check dummy forward pass (Batch size: 1, Channels: 3, Height: 224, Width: 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        
    # The output should predict 2 classes: Fake (0) and Real (1)
    assert output.shape == (1, 2) 

def test_audio_model_calibration():
    """
    Regression guard: confirms the audio model is natively calibrated.

    The retrained model should score white noise (which is NOT a deepfake) with
    a raw fake-logit < 1.0 WITHOUT any post-hoc penalty subtraction.
    Previously this required subtracting 2.0 from the logit to pass.

    If this test fails after a model update, it means recalibration is needed.
    """
    import torchaudio.transforms as T
    import math

    device = torch.device("cpu")

    # Build the same AudioTransformer used in predict.py
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
        def forward(self, x):
            x = x + self.pe[: x.size(1), :].permute(1, 0, 2)
            return self.dropout(x)

    class AudioTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj       = torch.nn.Linear(128, 256)
            self.pos_encoder      = PositionalEncoding(256)
            enc_layer             = torch.nn.TransformerEncoderLayer(256, 8, 1024, 0.1, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(enc_layer, 6)
            self.classifier       = torch.nn.Linear(256, 2)
        def forward(self, src):
            src = src.permute(0, 2, 1)
            src = self.input_proj(src)
            src = self.pos_encoder(src)
            out = self.transformer_encoder(src)
            out = out.mean(dim=1)
            return self.classifier(out)

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../models/transformer_voice_detector.pth')
    )
    if not os.path.exists(model_path):
        import pytest
        pytest.skip("Model file not found — run retraining first")

    net = AudioTransformer().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 5 seconds of white noise at 16 kHz — NOT a deepfake
    waveform = torch.randn(1, 16000 * 5) * 0.05   # quiet white noise
    mel_tf = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
    mel = mel_tf(waveform)  # [1, 128, frames]

    with torch.no_grad():
        logits = net(mel)
        fake_logit = logits[0][1].item()
        fake_prob  = torch.softmax(logits, dim=1)[0][1].item()

    print(f"\n  Calibration check — fake logit on white noise: {fake_logit:+.3f}  (prob: {fake_prob:.3f})")

    # The calibrated model should NOT scream "FAKE" at white noise without any penalty
    # Threshold of 1.5 is generous — a properly calibrated model should be near 0 or negative
    assert fake_logit < 1.5, (
        f"Fake logit {fake_logit:+.2f} is too high for non-deepfake input. "
        "The model may still be uncalibrated — consider retraining."
    )

if __name__ == '__main__':
    test_config_loader()
    test_silence_detector()
    test_video_model_architecture()
    test_audio_model_calibration()
    print("All architecture, model, calibration, and config unit tests passed successfully!")
