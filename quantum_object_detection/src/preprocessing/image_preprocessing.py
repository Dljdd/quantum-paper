"""
Image preprocessing module for quantum-classical object detection.
This module handles classical preprocessing steps before quantum encoding.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def preprocess_image(image, target_size=(640, 640), normalize=True):
    """
    Preprocess an image for the hybrid quantum-classical pipeline.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for resizing (width, height)
        normalize (bool): Whether to normalize pixel values to [0, 1]
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image[0, 0, 0], np.uint8):  # This is likely a BGR image from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    resized_image = cv2.resize(image, target_size)
    
    # Normalize pixel values if requested
    if normalize:
        if resized_image.dtype != np.float32 and resized_image.dtype != np.float64:
            # If image is not already in float format, convert it
            resized_image = resized_image.astype(np.float32)
            
        # Normalize to range [0, 1]
        if resized_image.max() > 1.0:
            resized_image = resized_image / 255.0
    
    return resized_image

def extract_bitplanes(image):
    """
    Extract bit-planes from an image.
    
    Args:
        image (numpy.ndarray): Input image (should be uint8 with values 0-255)
        
    Returns:
        dict: Dictionary containing bit planes for each channel
    """
    # Convert to uint8 if in float format
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Check image shape to determine if grayscale or color
    if len(image.shape) == 2:  # Grayscale
        channels = {'gray': image}
    else:  # Color (RGB)
        channels = {
            'red': image[:, :, 0],
            'green': image[:, :, 1],
            'blue': image[:, :, 2]
        }
    
    bitplanes = {}
    for channel_name, channel_data in channels.items():
        channel_bitplanes = {}
        for bit in range(8):  # 8 bits per channel for uint8
            # Extract the bit-plane by right shifting and masking with 1
            bitplane = (channel_data >> bit) & 1
            channel_bitplanes[f'bit_{bit}'] = bitplane
        
        bitplanes[channel_name] = channel_bitplanes
    
    return bitplanes

def visualize_bitplanes(bitplanes, save_path=None):
    """
    Visualize bit-planes from an image.
    
    Args:
        bitplanes (dict): Dictionary containing bit planes for each channel
        save_path (str, optional): Path to save the visualization
    """
    channels = list(bitplanes.keys())
    bits_per_channel = len(bitplanes[channels[0]])
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(channels), bits_per_channel, figsize=(15, 3 * len(channels)))
    
    # If there's only one channel, axes won't be 2D
    if len(channels) == 1:
        axes = np.array([axes])
    
    # Plot each bitplane
    for i, channel in enumerate(channels):
        for j, bit in enumerate(sorted(bitplanes[channel].keys())):
            if len(channels) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            ax.imshow(bitplanes[channel][bit], cmap='binary')
            ax.set_title(f"{channel} - {bit}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def create_test_image(size=(100, 100), mode='gradient'):
    """
    Create a test image for development and testing.
    
    Args:
        size (tuple): Size of the test image (width, height)
        mode (str): Type of test image ('gradient', 'checkerboard', etc.)
        
    Returns:
        numpy.ndarray: Test image
    """
    if mode == 'gradient':
        # Create a horizontal gradient (0-255)
        x = np.linspace(0, 255, size[0])
        image = np.tile(x, (size[1], 1)).astype(np.uint8)
    
    elif mode == 'checkerboard':
        # Create a checkerboard pattern
        x = np.zeros(size, dtype=np.uint8)
        x[::2, ::2] = 255
        x[1::2, 1::2] = 255
        image = x
    
    else:
        # Default to random noise
        image = np.random.randint(0, 256, size=size, dtype=np.uint8)
    
    return image

def apply_augmentations(image):
    """
    Apply data augmentations to the input image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Augmented image
    """
    # Random horizontal flip with 50% probability
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random brightness and contrast adjustments
    alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # contrast
    beta = np.random.uniform(-0.2, 0.2) * 255  # brightness
    
    # Apply brightness and contrast adjustments
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

def create_data_pipeline(image_path, target_size=(640, 640), augment=False):
    """
    Create a complete data preprocessing pipeline.
    
    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size for resizing
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (preprocessed_image, bitplanes)
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Apply augmentation if requested
    if augment:
        image = apply_augmentations(image)
    
    # Preprocess image
    preprocessed = preprocess_image(image, target_size=target_size)
    
    # Extract bitplanes
    bitplanes = extract_bitplanes(preprocessed)
    
    return preprocessed, bitplanes

if __name__ == "__main__":
    # Test the preprocessing functions
    print("Testing preprocessing module...")
    
    # Create a test image
    test_img = create_test_image(size=(100, 100), mode='gradient')
    
    # Preprocess the image
    preprocessed = preprocess_image(test_img)
    
    # Extract bitplanes
    bitplanes = extract_bitplanes(test_img)
    
    # Visualize bitplanes
    visualize_bitplanes(bitplanes, save_path="bitplane_visualization.png")
    
    print("Preprocessing module test completed.")