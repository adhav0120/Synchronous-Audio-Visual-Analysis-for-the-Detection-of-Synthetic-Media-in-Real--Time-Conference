# src/dataset.py

import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import random
import numpy as np
import glob

# Import project-specific modules
import config

class MusanAugment:
    """
    A class to handle augmentation by adding noise or music from the MUSAN dataset.
    """
    def __init__(self, musan_path):
        self.musan_path = musan_path
        
        # Find all noise and music files
        noise_path_pattern = os.path.join(musan_path, 'noise', '**', '*.wav')
        self.noise_files = glob.glob(noise_path_pattern, recursive=True)
        
        # --- FIX: Add a check to ensure noise files were found ---
        if not self.noise_files:
            raise FileNotFoundError(
                f"No noise files found matching the pattern: {noise_path_pattern}\n"
                "Please check the following:\n"
                "1. You have downloaded and extracted the MUSAN dataset.\n"
                "2. The 'musan' folder is located directly inside your 'data' directory.\n"
                "3. The path in 'src/config.py' for MUSAN_PATH is correct."
            )
        
        music_path_pattern = os.path.join(musan_path, 'music', '**', '*.wav')
        self.music_files = glob.glob(music_path_pattern, recursive=True)


    def add_noise(self, waveform, min_snr=5, max_snr=20):
        """Adds random noise to a waveform with a random SNR."""
        noise_file = random.choice(self.noise_files)
        noise_waveform, sr = torchaudio.load(noise_file)
        
        if sr != config.SAMPLE_RATE:
            noise_waveform = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)(noise_waveform)
        
        return self._mix_waveforms(waveform, noise_waveform, min_snr, max_snr)

    def add_music(self, waveform, min_snr=5, max_snr=15):
        """Adds random music to a waveform with a random SNR."""
        if not self.music_files:
            return waveform # Silently fail if no music files are found
        music_file = random.choice(self.music_files)
        music_waveform, sr = torchaudio.load(music_file)

        if sr != config.SAMPLE_RATE:
            music_waveform = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)(music_waveform)

        return self._mix_waveforms(waveform, music_waveform, min_snr, max_snr)

    def _mix_waveforms(self, waveform, noise_waveform, min_snr, max_snr):
        """Helper function to mix two waveforms at a given SNR."""
        if noise_waveform.shape[1] < waveform.shape[1]:
            # Pad noise if it's shorter
            pad_len = waveform.shape[1] - noise_waveform.shape[1]
            noise_waveform = torch.nn.functional.pad(noise_waveform, (0, pad_len))
        else:
            # Truncate noise if it's longer
            noise_waveform = noise_waveform[:, :waveform.shape[1]]

        # Calculate powers
        waveform_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)
        
        # Choose random SNR
        snr_db = random.uniform(min_snr, max_snr)
        snr = 10 ** (snr_db / 20)
        
        # Scale noise to desired SNR
        scale = waveform_power / (noise_power * snr)
        
        return waveform + (noise_waveform * scale)


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_file, data_dir, max_len_seconds, transform=None, is_train=True):
        self.protocol = pd.read_csv(protocol_file, sep=" ", header=None)
        self.protocol.columns = ['speaker_id', 'filename', 'system_id', 'null', 'label']
        self.data_dir = data_dir
        self.label_mapping = {'bonafide': 0, 'spoof': 1}
        self.transform = transform
        self.max_len = max_len_seconds * config.SAMPLE_RATE
        self.is_train = is_train
        
        # Initialize augmentation if it's a training set
        if self.is_train:
            self.augmenter = MusanAugment(config.MUSAN_PATH)


    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        filename = self.protocol.iloc[idx]['filename']
        label_str = self.protocol.iloc[idx]['label']
        label = self.label_mapping[label_str]
        
        audio_path = os.path.join(self.data_dir, f"{filename}.flac")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != config.SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, config.SAMPLE_RATE)(waveform)

        # Pad or truncate waveform to a fixed length
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # --- ADVANCED AUGMENTATION FOR TRAINING DATA ---
        # --- FIX: Removed reverb as it's not supported on Windows ---
        if self.is_train:
            # Randomly choose to add noise or do nothing
            if random.random() < 0.5: # Apply noise to 50% of samples
                waveform = self.augmenter.add_noise(waveform)

        if self.transform:
            mel_spec = self.transform(waveform)
        else:
            mel_spec = waveform

        return mel_spec, torch.tensor(label, dtype=torch.long)

def get_mel_spectrogram_transform(is_train=True):
    transforms_list = [
        torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        ),
        torchaudio.transforms.AmplitudeToDB()
    ]
    
    if is_train:
        transforms_list.extend([
            torchaudio.transforms.FrequencyMasking(freq_mask_param=25),
            torchaudio.transforms.TimeMasking(time_mask_param=50)
        ])
        
    return torch.nn.Sequential(*transforms_list)
