# src/final_test.py

import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import project-specific modules
import config
from dataset import ASVspoofDataset, get_mel_spectrogram_transform
from model import VoiceClfTransformer

def calculate_eer(y_true, y_scores):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def test_model(model, dataloader, device):
    """Tests the model on the final evaluation set."""
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Final Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if inputs.dim() == 4 and inputs.shape[1] == 1:
                inputs = inputs.squeeze(1)

            outputs = model(inputs)
            scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    eer = calculate_eer(np.array(all_labels), np.array(all_scores))
    return eer

def main():
    """Main function to run the final test."""
    print(f"Using device: {config.DEVICE}")
    
    # Load the best model
    model = VoiceClfTransformer().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        return

    # Load the final evaluation data
    transform = get_mel_spectrogram_transform(is_train=False)
    
    eval_dataset = ASVspoofDataset(
        protocol_file=config.EVAL_PROTOCOL_PATH,
        data_dir=config.EVAL_DATA_DIR,
        max_len_seconds=config.MAX_LEN_SECONDS,
        transform=transform,
        is_train=False
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Running final test on the evaluation set...")
    eer = test_model(model, eval_loader, config.DEVICE)
    print(f"\n--- FINAL RESULT ---")
    print(f"Equal Error Rate (EER) on the EVALUATION set: {eer * 100:.2f}%")

if __name__ == '__main__':
    main()
