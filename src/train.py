# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import project-specific modules
import config
from dataset import ASVspoofDataset, get_mel_spectrogram_transform
from model import VoiceClfTransformer

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Squeeze the channel dimension if it exists (from [Batch, 1, Mels, Time])
        if inputs.dim() == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        # Update progress bar description
        current_acc = (correct_predictions.double() / total_samples).item()
        progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}")
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def main():
    """
    Main function to orchestrate the training process.
    """
    print(f"Using device: {config.DEVICE}")
    
    # Ensure the model directory exists
    model_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Data loading with augmentation enabled
    transform = get_mel_spectrogram_transform(is_train=True)
    
    train_dataset = ASVspoofDataset(
        protocol_file=config.TRAIN_PROTOCOL_PATH,
        data_dir=config.DATA_DIR,
        max_len_seconds=config.MAX_LEN_SECONDS,
        transform=transform,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        # --- FIX: Set num_workers to 0 for Windows compatibility ---
        num_workers=0 
    )
    
    # 2. Model, loss, and optimizer setup
    model = VoiceClfTransformer().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 3. Training loop
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        
    # 4. Save the trained model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
