"""
Enhanced data augmentation for better generalization
Add this to your notebook as a new cell before training
"""

import torch
import torchvision.transforms as transforms

# Enhanced augmentation for training
enhanced_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # NEW: Insects can appear upside down
        transforms.RandomRotation(30),  # Increased from 15
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),  # NEW: Random translation
            scale=(0.9, 1.1)       # NEW: Random scaling
        ),
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3, 
            saturation=0.3,
            hue=0.1  # NEW: Color variation
        ),
        transforms.RandomGrayscale(p=0.1),  # NEW: Occasionally grayscale
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # NEW: Slight blur
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # NEW: Random erasing (occlusion)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Update your dataloaders with enhanced transforms
# In cell 5, replace data_transforms with enhanced_transforms

