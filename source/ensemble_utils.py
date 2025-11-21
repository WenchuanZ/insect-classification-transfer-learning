"""
Ensemble Utility Functions
===========================
Clean utility module for loading saved models and performing ensemble predictions.

Usage in Jupyter:
    from ensemble_utils import load_models, evaluate_single_model, evaluate_ensemble
    
    # Load models
    models = load_models(class_names)
    
    # Evaluate
    results = evaluate_single_model(models['resnet'], test_loader, "ResNet-18")
"""

import torch  
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import accuracy_score
import numpy as np


def load_resnet18(weights_path, num_classes, device='cuda:0'):
    """
    Load ResNet-18 finetuned model
    
    Args:
        weights_path: Path to .pth file
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def load_densenet121(weights_path, num_classes, dropout=0.4, device='cuda:0'):
    """
    Load DenseNet-121 model with dropout
    
    Args:
        weights_path: Path to .pth file
        num_classes: Number of output classes
        dropout: Dropout probability (must match training)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def load_efficientnet_v2_s(weights_path, num_classes, dropout=None, device='cuda:0'):
    """
    Load EfficientNet-V2-S model
    
    Args:
        weights_path: Path to .pth file
        num_classes: Number of output classes
        dropout: Dropout probability (None for no dropout, must match training)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    
    if dropout is not None:
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def load_models(num_classes, model_dir='.', device='cuda:0'):
    """
    Load all three saved models at once
    
    Args:
        num_classes: Number of output classes
        model_dir: Directory containing .pth files
        device: Device to load models on
    
    Returns:
        Dictionary with keys: 'resnet', 'densenet', 'efficientnet'
    """
    import os
    
    models_dict = {}
    
    # Load ResNet-18
    resnet_path = os.path.join(model_dir, 'insect_classifier_finetuned.pth')
    if os.path.exists(resnet_path):
        print(f"Loading ResNet-18 from {resnet_path}...")
        models_dict['resnet'] = load_resnet18(resnet_path, num_classes, device)
        print("✓ ResNet-18 loaded")
    
    # Load DenseNet-121
    densenet_path = os.path.join(model_dir, 'insect_classifier_densenet121.pth')
    if os.path.exists(densenet_path):
        print(f"Loading DenseNet-121 from {densenet_path}...")
        models_dict['densenet'] = load_densenet121(densenet_path, num_classes, dropout=0.4, device=device)
        print("✓ DenseNet-121 loaded")
    
    # Load EfficientNet-V2-S
    efficientnet_path = os.path.join(model_dir, 'insect_classifier_efficientnet_v2_s.pth')
    if os.path.exists(efficientnet_path):
        print(f"Loading EfficientNet-V2-S from {efficientnet_path}...")
        models_dict['efficientnet'] = load_efficientnet_v2_s(efficientnet_path, num_classes, dropout=None, device=device)
        print("✓ EfficientNet-V2-S loaded")
    
    print(f"\n✓ Loaded {len(models_dict)} models successfully!")
    return models_dict


def evaluate_single_model(model, dataloader, model_name=None, device='cuda:0'):
    """
    Evaluate a single model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        model_name: Name for display (optional)
        device: Device for computation
    
    Returns:
        accuracy: Test accuracy as float
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if model_name:
        print(f"{model_name:30s} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def ensemble_predict(models, inputs, weights=None, device='cuda:0'):
    """
    Get ensemble predictions for a batch of inputs
    
    Args:
        models: List of PyTorch models
        inputs: Input tensor
        weights: Optional weights for each model (default: equal weights)
        device: Device for computation
    
    Returns:
        ensemble_probs: Weighted probability predictions
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_probs = None
    
    for model, weight in zip(models, weights):
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            
            if ensemble_probs is None:
                ensemble_probs = weight * probs
            else:
                ensemble_probs += weight * probs
    
    return ensemble_probs


def evaluate_ensemble(models, dataloader, weights=None, ensemble_name=None, device='cuda:0'):
    """
    Evaluate an ensemble of models
    
    Args:
        models: List of PyTorch models
        dataloader: DataLoader for evaluation
        weights: Optional weights for each model (default: equal)
        ensemble_name: Name for display (optional)
        device: Device for computation
    
    Returns:
        accuracy: Ensemble accuracy as float
        predictions: List of predicted labels
        true_labels: List of true labels
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Get ensemble predictions
        ensemble_probs = ensemble_predict(models, inputs, weights, device)
        _, preds = torch.max(ensemble_probs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if ensemble_name:
        print(f"{ensemble_name:30s} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, all_preds, all_labels


def get_ensemble_weights(model_accuracies, strategy='proportional'):
    """
    Calculate ensemble weights based on model accuracies
    
    Args:
        model_accuracies: List or dict of model accuracies
        strategy: 'equal', 'proportional', or 'softmax'
    
    Returns:
        List of weights
    """
    if isinstance(model_accuracies, dict):
        model_accuracies = list(model_accuracies.values())
    
    if strategy == 'equal':
        return [1.0 / len(model_accuracies)] * len(model_accuracies)
    
    elif strategy == 'proportional':
        total = sum(model_accuracies)
        return [acc / total for acc in model_accuracies]
    
    elif strategy == 'softmax':
        # Softmax with temperature
        import math
        temp = 5.0
        exp_accs = [math.exp(acc * 100 / temp) for acc in model_accuracies]
        total = sum(exp_accs)
        return [exp_acc / total for exp_acc in exp_accs]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compare_results(results_dict, best_key='accuracy'):
    """
    Print a formatted comparison of results
    
    Args:
        results_dict: Dictionary of {name: accuracy}
        best_key: Key to determine best result
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Sort by accuracy
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, acc) in enumerate(sorted_results):
        marker = "🏆" if i == 0 else "  "
        print(f"{marker} {name:40s} {acc:.4f} ({acc*100:.2f}%)")
    
    print("="*70)
    
    best_name, best_acc = sorted_results[0]
    print(f"\n🏆 Best Model: {best_name}")
    print(f"✨ Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")


def get_tta_transforms():
    """
    Get a list of TTA (Test-Time Augmentation) transforms
    
    Returns:
        List of transforms that work on batched tensors [B, C, H, W]
    """
    # Use torch operations that work on batches (4D tensors)
    tta_transforms = [
        lambda x: torch.flip(x, dims=[3]),           # Horizontal flip
        lambda x: torch.flip(x, dims=[2]),           # Vertical flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotate 90° (H, W dims)
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Rotate 270° (H, W dims)
    ]
    
    return tta_transforms


def predict_with_tta(model, inputs, n_augmentations=4, device='cuda:0'):
    """
    Apply Test-Time Augmentation to a batch of inputs
    
    Args:
        model: PyTorch model
        inputs: Input tensor batch (B, C, H, W)
        n_augmentations: Number of augmentations to apply (max 4)
        device: Device for computation
    
    Returns:
        averaged_probs: Averaged probabilities over original + augmentations
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        original_logits = model(inputs)
        predictions = [F.softmax(original_logits, dim=1)]
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    # Apply augmentations
    for i in range(min(n_augmentations, len(tta_transforms))):
        aug_inputs = tta_transforms[i](inputs)
        with torch.no_grad():
            aug_logits = model(aug_inputs)
            predictions.append(F.softmax(aug_logits, dim=1))
    
    # Average all predictions
    avg_probs = torch.stack(predictions).mean(dim=0)
    
    return avg_probs


def evaluate_single_model_with_tta(model, dataloader, n_augmentations=4, model_name=None, device='cuda:0'):
    """
    Evaluate a single model with Test-Time Augmentation
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        n_augmentations: Number of augmentations per image
        model_name: Name for display (optional)
        device: Device for computation
    
    Returns:
        accuracy: Test accuracy as float
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Get TTA predictions
        tta_probs = predict_with_tta(model, inputs, n_augmentations, device)
        _, preds = torch.max(tta_probs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if model_name:
        print(f"{model_name:30s} Accuracy (TTA): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def evaluate_ensemble_with_tta(models, dataloader, n_augmentations=4, weights=None, 
                                ensemble_name=None, device='cuda:0'):
    """
    Evaluate ensemble with Test-Time Augmentation
    
    This combines TWO powerful techniques:
    1. TTA - Multiple augmented versions of each image
    2. Ensemble - Combining multiple models
    
    Args:
        models: List of PyTorch models
        dataloader: DataLoader for evaluation
        n_augmentations: Number of augmentations per image
        weights: Optional weights for each model (default: equal)
        ensemble_name: Name for display (optional)
        device: Device for computation
    
    Returns:
        accuracy: Ensemble accuracy with TTA as float
        predictions: List of predicted labels
        true_labels: List of true labels
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Get TTA predictions from each model
        ensemble_probs = None
        
        for model, weight in zip(models, weights):
            model.eval()
            # Apply TTA to this model's predictions
            tta_probs = predict_with_tta(model, inputs, n_augmentations, device)
            
            if ensemble_probs is None:
                ensemble_probs = weight * tta_probs
            else:
                ensemble_probs += weight * tta_probs
        
        _, preds = torch.max(ensemble_probs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if ensemble_name:
        print(f"{ensemble_name:30s} Accuracy (TTA): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, all_preds, all_labels


# For backwards compatibility with old scripts
def evaluate_model(model, dataloader, model_name, device='cuda:0'):
    """Alias for evaluate_single_model (backwards compatibility)"""
    return evaluate_single_model(model, dataloader, model_name, device)

