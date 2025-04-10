"""
Dataset handling module for quantum-classical object detection.
This module provides utilities for loading and processing image datasets
for the hybrid quantum-classical object detection pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ObjectDetectionDataset(Dataset):
    """
    Dataset class for object detection tasks.
    Compatible with YOLO-format annotations.
    """
    def __init__(self, img_dir, annotation_dir, img_size=(640, 640), transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_dir (str): Path to the directory containing images
            annotation_dir (str): Path to the directory containing annotations
            img_size (tuple): Target size for resizing images (width, height)
            transform: Optional transform to be applied to the images
        """
        self.img_dir = Path(img_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files with supported extensions
        self.img_files = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Resize image
        img_resized = cv2.resize(img, self.img_size)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Load annotation if it exists
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:  # class x_center y_center width height
                        class_id = int(data[0])
                        # YOLO format: x_center, y_center, width, height (normalized)
                        x_center, y_center, box_width, box_height = map(float, data[1:5])
                        
                        # Convert to actual coordinates in the resized image
                        x_min = (x_center - box_width / 2) * self.img_size[0]
                        y_min = (y_center - box_height / 2) * self.img_size[1]
                        x_max = (x_center + box_width / 2) * self.img_size[0]
                        y_max = (y_center + box_height / 2) * self.img_size[1]
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0), dtype=np.int64)
        
        # Apply transforms if any
        if self.transform:
            img_normalized = self.transform(img_normalized)
        
        # Create target dictionary
        target = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([height, width]),
        }
        
        return torch.FloatTensor(img_normalized.transpose(2, 0, 1)), target  # Convert to CHW format for PyTorch

def create_dataloaders(img_dir, annotation_dir, img_size=(640, 640), batch_size=4, split=0.8):
    """
    Create training and validation dataloaders.
    
    Args:
        img_dir (str): Path to the directory containing images
        annotation_dir (str): Path to the directory containing annotations
        img_size (tuple): Target size for resizing images
        batch_size (int): Batch size for dataloaders
        split (float): Train/val split ratio
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Create a dataset with all images
    full_dataset = ObjectDetectionDataset(img_dir, annotation_dir, img_size)
    
    # Calculate the size of train and validation sets
    train_size = int(split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Custom collate function for handling variable size boxes
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    return train_dataloader, val_dataloader

def collate_fn(batch):
    """
    Custom collate function for the dataloader to handle variable number of objects.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    
    return images, targets

def convert_to_bitplanes(image):
    """
    Convert an image to bit-plane representation.
    
    Args:
        image (numpy.ndarray): Input image with pixel values in [0, 1]
        
    Returns:
        list: List of bit planes for each channel
    """
    # Ensure pixel values are in [0, 255] and convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Get number of channels
    if len(image_uint8.shape) == 2:  # Grayscale
        channels = [image_uint8]
    else:  # Color image
        channels = [image_uint8[:, :, i] for i in range(image_uint8.shape[2])]
    
    all_bitplanes = []
    
    for channel in channels:
        # Create bit planes for each bit position (0-7)
        bitplanes = [(channel >> i) & 1 for i in range(8)]
        all_bitplanes.append(bitplanes)
    
    return all_bitplanes

def load_single_image(image_path, img_size=(640, 640)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image
        img_size (tuple): Target size for resizing image (width, height)
        
    Returns:
        tuple: (original_image, preprocessed_image)
    """
    # Load image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(original_img, img_size)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return original_img, img_normalized

if __name__ == "__main__":
    # Example usage of the dataset
    print("Testing dataset handling...")
    # This is a placeholder for testing the dataset functionality
    # In a real implementation, you would need actual data
    print("Dataset handling module initialized.")