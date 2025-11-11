"""
Test-Time Augmentation (TTA) - Can boost accuracy by 1-3%
Add this function to your notebook
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

def predict_with_tta(model, image_tensor, n_augmentations=5):
    """
    Apply multiple augmentations and average predictions
    
    Args:
        model: Trained model
        image_tensor: Input image tensor (C, H, W)
        n_augmentations: Number of augmented versions to create
    
    Returns:
        averaged_prediction: Class probabilities averaged over augmentations
    """
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        original = model(image_tensor.unsqueeze(0).to(device))
        predictions = [F.softmax(original, dim=1)]
    
    # Augmentation transforms
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomRotation(-10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
    ]
    
    # Generate augmented predictions
    for i in range(min(n_augmentations, len(tta_transforms))):
        aug_image = tta_transforms[i](image_tensor)
        with torch.no_grad():
            pred = model(aug_image.unsqueeze(0).to(device))
            predictions.append(F.softmax(pred, dim=1))
    
    # Average all predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    
    return avg_prediction


def evaluate_model_with_tta(model, dataloader, dataset_name='Test'):
    """Enhanced evaluation with TTA"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Evaluating {dataset_name} with Test-Time Augmentation...")
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply TTA for each image
        batch_preds = []
        for img in inputs:
            tta_pred = predict_with_tta(model, img, n_augmentations=5)
            batch_preds.append(tta_pred)
        
        batch_preds = torch.cat(batch_preds, dim=0)
        _, preds = torch.max(batch_preds, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'{dataset_name} Accuracy with TTA: {accuracy:.4f}')
    print(f'Improvement: {(accuracy - test_acc_densenet)*100:.2f}%')
    
    return accuracy


# Usage in notebook:
# tta_accuracy = evaluate_model_with_tta(model_densenet, dataloaders['test'])

