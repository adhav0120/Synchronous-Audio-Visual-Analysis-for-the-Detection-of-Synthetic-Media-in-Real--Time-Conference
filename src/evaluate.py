# src/evaluate.py

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

def evaluate_model(model, dataloader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Squeeze channel dimension
            if inputs.dim() == 4 and inputs.shape[1] == 1:
                inputs = inputs.squeeze(1)

            outputs = model(inputs)
            # Use softmax to get scores for the 'spoof' class (class 1)
            scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    eer = calculate_eer(np.array(all_labels), np.array(all_scores))
    return eer

def main():
    """Main function to orchestrate the evaluation process."""
    print(f"Using device: {config.DEVICE}")
    
    # Load model
    model = VoiceClfTransformer().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first by running train.py")
        return

    # Define path to development data
    dev_data_dir = config.DATA_ROOT / 'ASVspoof2019_LA_dev' / 'flac'
    
    # Load dev data without augmentations
    transform = get_mel_spectrogram_transform(is_train=False)
    
    dev_dataset = ASVspoofDataset(
        protocol_file=config.DEV_PROTOCOL_PATH,
        data_dir=dev_data_dir,
        max_len_seconds=config.MAX_LEN_SECONDS,
        transform=transform,
        is_train=False  # <-- KEY CHANGE: Ensure no augmentations are applied
    )
    
    # Set num_workers=0 for Windows compatibility during evaluation
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Evaluating model...")
    eer = evaluate_model(model, dev_loader, config.DEVICE)
    print(f"Equal Error Rate (EER) on the development set: {eer * 100:.2f}%")

if __name__ == '__main__':
    main()
