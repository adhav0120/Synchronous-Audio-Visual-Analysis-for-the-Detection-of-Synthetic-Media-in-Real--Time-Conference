# src/train_video.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from pathlib import Path

# Import our custom dataset modules
from face_dataset import FaceDataset, get_transforms
import config # We will use some variables from our config.py

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        current_acc = (correct_predictions.double() / total_samples).item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate_one_epoch(model, dataloader, criterion, device):
    """Validates the model for one epoch."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Validating")
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def main():
    """Main function to orchestrate the training process."""
    print(f"Using device: {config.DEVICE}")

    # --- 1. Dataset and DataLoaders ---
    FACE_DATA_ROOT = config.PROJECT_ROOT / "face_data"
    full_dataset = FaceDataset(root_dir=FACE_DATA_ROOT, transform=get_transforms())
    
    # Split dataset into training and validation (e.g., 90% train, 10% validation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # --- 2. Model Setup ---
    # Load a pre-trained EfficientNet-B0 model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Replace the final classification layer for our binary task (real vs. fake)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 2) # 2 classes: real, fake
    
    model = model.to(config.DEVICE)

    # --- 3. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 4. Training Loop ---
    best_val_acc = 0.0
    VIDEO_MODEL_SAVE_PATH = config.PROJECT_ROOT / "models" / "video_deepfake_detector.pth"

    print("Starting training for video deepfake detector...")
    try:
        for epoch in range(config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            print(f"Epoch {epoch+1} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
            
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
            print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
            # Save the model if it has the best validation accuracy so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), VIDEO_MODEL_SAVE_PATH)
                print(f"New best model saved to {VIDEO_MODEL_SAVE_PATH} with accuracy: {best_val_acc:.4f}")

            # --- FIX: Clear GPU cache after each epoch to manage memory ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print("\n--- CUDA Out of Memory ---")
        print("Your GPU ran out of memory. This is usually caused by a batch size that is too large.")
        print(f"Current batch size: {config.BATCH_SIZE}")
        print("Please open 'src/config.py' and lower the BATCH_SIZE to 16, 8, or 4 and try again.")
        print("--------------------------")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
