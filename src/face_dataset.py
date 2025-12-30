# src/face_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random

class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading face images from the preprocessed data folder.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str or Path): Directory with all the images, containing 'real' and 'fake' subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Create a list of all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        real_dir = self.root_dir / 'real'
        fake_dir = self.root_dir / 'fake'
        
        # Label 'real' as 0 and 'fake' as 1
        for img_path in real_dir.glob('*.png'):
            self.image_paths.append(img_path)
            self.labels.append(0)
            
        for img_path in fake_dir.glob('*.png'):
            self.image_paths.append(img_path)
            self.labels.append(1)
            
        # Shuffle the dataset to mix real and fake images
        temp = list(zip(self.image_paths, self.labels))
        random.shuffle(temp)
        self.image_paths, self.labels = zip(*temp)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image using Pillow
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    """
    Returns a set of standard transformations for image datasets.
    Includes resizing, conversion to tensor, and normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)), # Resize to the standard input size for many models
        transforms.ToTensor(),
        # Normalize with ImageNet's mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Example of how to use this dataset (optional, for testing)
if __name__ == '__main__':
    # Define the root of your project
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    FACE_DATA_ROOT = PROJECT_ROOT / "face_data"
    
    print(f"Loading dataset from: {FACE_DATA_ROOT}")
    
    # Create the dataset
    dataset = FaceDataset(root_dir=FACE_DATA_ROOT, transform=get_transforms())
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a sample batch
    images, labels = next(iter(dataloader))
    
    print(f"Sample batch shape: {images.shape}") # Should be [batch_size, 3, 224, 224]
    print(f"Sample labels shape: {labels.shape}")
    print(f"Sample labels: {labels}")
