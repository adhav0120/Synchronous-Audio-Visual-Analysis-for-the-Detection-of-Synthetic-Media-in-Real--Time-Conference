# src/live_predict.py

import torch
import torchaudio
import pyaudio
import wave
import time
import os

# Import project-specific modules
import config
from model import VoiceClfTransformer
from dataset import get_mel_spectrogram_transform

# --- Audio Recording Settings ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = config.SAMPLE_RATE
RECORD_SECONDS = config.MAX_LEN_SECONDS
TEMP_WAVE_FILENAME = "temp_live_audio.wav"

def record_audio():
    """Records audio from the microphone for a fixed duration."""
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"\n* Recording for {RECORD_SECONDS} seconds...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(TEMP_WAVE_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return TEMP_WAVE_FILENAME

def predict_single(model, audio_file_path, device):
    """Predicts if a single audio file is 'bonafide' or 'spoof'."""
    model.eval()
    transform = get_mel_spectrogram_transform(is_train=False)

    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    if sample_rate != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # The recording is already mono and at the correct length
    mel_spec = transform(waveform).to(device)
    
    # Add a batch dimension. Shape becomes [1, 1, n_mels, time]
    mel_spec = mel_spec.unsqueeze(0)

    # --- FIX ---
    # Squeeze the channel dimension to match the model's expected 3D input [batch, n_mels, time]
    if mel_spec.dim() == 4 and mel_spec.shape[1] == 1:
        mel_spec = mel_spec.squeeze(1)

    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

    class_mapping = {0: 'Genuine (Bonafide)', 1: 'AI-Generated (Spoof)'}
    prediction_label = class_mapping[predicted_class.item()]
    spoof_probability = probabilities[0, 1].item()

    print(f"Prediction: {prediction_label}")
    print(f"Confidence (Spoof Probability): {spoof_probability:.2%}")

def main():
    # --- Load Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VoiceClfTransformer().to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        return

    # --- Main Loop ---
    try:
        while True:
            audio_file = record_audio()
            predict_single(model, audio_file, device)
            # Clean up the temporary file
            os.remove(audio_file)
            time.sleep(1) # Pause before next recording
    except KeyboardInterrupt:
        print("\nExiting live prediction.")
    finally:
        # Final cleanup in case of exit
        if os.path.exists(TEMP_WAVE_FILENAME):
            os.remove(TEMP_WAVE_FILENAME)

if __name__ == "__main__":
    main()
