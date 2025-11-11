"""
Advanced Training Techniques for Better Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================================================
# TECHNIQUE 1: MixUp Augmentation
# ==================================================
# Mix two images and their labels
# Expected gain: +1-2%

def mixup_data(x, y, alpha=0.4):
    """
    Apply MixUp augmentation
    
    Args:
        x: Input images
        y: Labels
        alpha: MixUp parameter (0.2-0.4 works well)
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Usage in training loop:
# inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
# outputs = model(inputs)
# loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)


# ==================================================
# TECHNIQUE 2: Focal Loss
# ==================================================
# Better for imbalanced classes
# Expected gain: +0.5-2% (depends on class imbalance)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Good if some insect classes have fewer samples
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Usage: Replace CrossEntropyLoss with FocalLoss
# criterion = FocalLoss(alpha=1, gamma=2)


# ==================================================
# TECHNIQUE 3: Stochastic Weight Averaging (SWA)
# ==================================================
# Average weights from multiple epochs
# Expected gain: +0.5-1.5%

from torch.optim.swa_utils import AveragedModel, SWALR

def train_with_swa(model, optimizer, scheduler, train_dataloader, device, 
                   num_epochs=25, swa_start=15):
    """
    Training with Stochastic Weight Averaging
    
    Args:
        model: Your model
        optimizer: Optimizer
        scheduler: LR scheduler
        train_dataloader: Training data loader
        device: Device to run on (cuda/cpu)
        num_epochs: Total epochs
        swa_start: Epoch to start SWA (usually 60-80% through training)
    """
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.0001)
    
    for epoch in range(num_epochs):
        # Regular training here...
        # train_one_epoch(model, optimizer, ...)
        
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
    
    # Update batch norm statistics for SWA model
    torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)
    
    return swa_model


# ==================================================
# TECHNIQUE 4: Gradual Unfreezing (for EfficientNet)
# ==================================================
# Unfreeze layers gradually during training
# Expected gain: +1-2%

def gradual_unfreeze_efficientnet(model, epoch, unfreeze_schedule):
    """
    Gradually unfreeze layers
    
    Args:
        model: EfficientNet model
        epoch: Current epoch
        unfreeze_schedule: Dict mapping epoch to layers to unfreeze
    """
    if epoch in unfreeze_schedule:
        layers_to_unfreeze = unfreeze_schedule[epoch]
        print(f"Epoch {epoch}: Unfreezing {layers_to_unfreeze}")
        
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True


# Example usage:
# unfreeze_schedule = {
#     0: ['classifier'],           # Only classifier
#     5: ['features.8'],           # Unfreeze last block
#     10: ['features.7', 'features.6'],  # Unfreeze more
#     15: ['features']             # Unfreeze all
# }


# ==================================================
# TECHNIQUE 5: CutMix Augmentation
# ==================================================
# Mix patches from different images
# Expected gain: +1-2%

def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: CutMix parameter
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    lam = np.random.beta(alpha, alpha)
    
    # Get random bounding box
    W = x.size()[3]
    H = x.size()[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# Usage (similar to MixUp):
# inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
# outputs = model(inputs)
# loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

